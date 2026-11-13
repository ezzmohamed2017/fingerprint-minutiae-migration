"""fpf_tf — TensorFlow helpers shared by the FPF training kernels (identical to fpm_tf used by FPM04C) (pairs, Siamese model, callbacks, embedding, timing)."""
import math, os, random, time
import numpy as np
import pandas as pd
import tensorflow as tf

def set_all_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed); random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)

def use_mixed_precision():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        tf.keras.mixed_precision.set_global_policy('mixed_float16'); return True
    return False

class PairSequence(tf.keras.utils.Sequence):
    """Rotating, balanced pair batches (the protocol's training schedule): each epoch takes `pairs_per_epoch` pairs,
    half genuine, half impostor, impostors 25% same-subject / 75% different-subject, cycling through the whole pool.
    `arrays` is a list of per-sample arrays (concatenated along the last axis); float16 is returned as float32."""
    def __init__(self, arrays, pairs, batch_size, shuffle, seed, pairs_per_epoch=None, same_subject_fraction=0.25, workers=4):
        super().__init__(workers=workers, use_multiprocessing=False, max_queue_size=8)
        self.arrays = arrays; self.pairs = pairs; self.batch_size = int(batch_size); self.shuffle = bool(shuffle)
        self.rng = np.random.default_rng(seed); self.ppe = None if pairs_per_epoch is None else int(pairs_per_epoch); self.ssf = float(same_subject_fraction)
        lab = pairs['label']; nt = pairs.get('negative_type', np.where(lab == 1, 0, 2))
        self.pools = {'positive': np.flatnonzero(lab == 1), 'same': np.flatnonzero((lab == 0) & (nt == 1)), 'diff': np.flatnonzero((lab == 0) & (nt != 1))}
        self.cursors = {k: 0 for k in self.pools}
        if self.shuffle:
            for p in self.pools.values(): self.rng.shuffle(p)
        self._build()
    def _get(self, idx):
        parts = [a[idx] for a in self.arrays]; x = np.concatenate(parts, axis=-1) if len(parts) > 1 else parts[0]
        return x.astype(np.float32) if x.dtype == np.float16 else x
    def __len__(self): return math.ceil(len(self.order) / self.batch_size)
    def __getitem__(self, b):
        pos = self.order[b * self.batch_size:(b + 1) * self.batch_size]
        return (self._get(self.pairs['index_a'][pos]), self._get(self.pairs['index_b'][pos])), self.pairs['label'][pos].astype(np.float32)
    def _take(self, k, n):
        pool = self.pools[k]
        if n <= 0 or len(pool) == 0: return np.empty(0, np.int64)
        if not self.shuffle: return pool[:min(n, len(pool))]
        out = []; r = n
        while r > 0:
            c = self.cursors[k]; t = min(r, len(pool) - c); out.append(pool[c:c + t]); self.cursors[k] += t; r -= t
            if self.cursors[k] >= len(pool): self.rng.shuffle(pool); self.cursors[k] = 0
        return np.concatenate(out)
    def _build(self):
        total = 2 * len(self.pools['positive']); target = total if self.ppe is None else min(self.ppe, total)
        n_pos = target // 2; n_same = int(round(n_pos * self.ssf)) if len(self.pools['same']) else 0
        pos = self._take('positive', n_pos); neg = np.concatenate([self._take('same', n_same), self._take('diff', n_pos - n_same)])
        if self.shuffle: self.rng.shuffle(neg)
        m = min(len(pos), len(neg)); self.order = np.column_stack([pos[:m], neg[:m]]).reshape(-1)
    def on_epoch_end(self):
        if self.shuffle: self._build()

class CosineCalibration(tf.keras.layers.Layer):
    """Turns a cosine into a same-finger probability: sigmoid(scale * cosine + bias). Used only for the BCE loss."""
    def build(self, input_shape):
        self.scale = self.add_weight(name='scale', shape=(), initializer=tf.keras.initializers.Constant(10.0), trainable=True, dtype='float32')
        self.bias = self.add_weight(name='bias', shape=(), initializer=tf.keras.initializers.Constant(-5.0), trainable=True, dtype='float32')
    def call(self, cosine): return tf.nn.sigmoid(tf.cast(cosine, tf.float32) * self.scale + self.bias)

def build_siamese(encoder, input_shape, input_dtype, learning_rate, weight_decay=1e-4, clipnorm=None):
    a = tf.keras.Input(input_shape, dtype=input_dtype, name='input_a'); b = tf.keras.Input(input_shape, dtype=input_dtype, name='input_b')
    p = CosineCalibration(name='same_finger_probability')(tf.keras.layers.Dot(axes=1, name='cosine_similarity')([encoder(a), encoder(b)]))
    model = tf.keras.Model([a, b], p, name='siamese')
    opt = tf.keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay, **({'clipnorm': clipnorm} if clipnorm else {}))
    model.compile(optimizer=opt, loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=[tf.keras.metrics.BinaryAccuracy(name='verification_accuracy'), tf.keras.metrics.AUC(name='verification_auc')])
    return model

def callbacks(weights_path, history_csv, patience=10, lr_patience=3, lr_factor=0.3, monitor='val_verification_auc', min_delta=1e-4):
    """Early stopping and learning-rate drops on INTERNAL-VALIDATION AUC (never on calibration, validation or test)."""
    return [tf.keras.callbacks.ModelCheckpoint(str(weights_path), monitor=monitor, mode='max', save_best_only=True, save_weights_only=True, verbose=1),
            tf.keras.callbacks.EarlyStopping(monitor=monitor, mode='max', min_delta=min_delta, patience=patience, restore_best_weights=True, verbose=1),
            tf.keras.callbacks.ReduceLROnPlateau(monitor=monitor, mode='max', min_delta=min_delta, factor=lr_factor, patience=lr_patience, min_lr=1e-7, verbose=1),
            tf.keras.callbacks.CSVLogger(str(history_csv))]

def embed(encoder, x, batch_size=128):
    return encoder.predict(x, batch_size=batch_size, verbose=0).astype(np.float32)

def time_encoder(encoder, batch, repeats=30):
    """Returns (ms per item at this batch size, ms for a single item) on the current device."""
    f = tf.function(lambda z: encoder(z, training=False)); f(batch); t0 = time.perf_counter(); f(batch).numpy(); thr = 1000 * (time.perf_counter() - t0) / len(batch)
    one = batch[:1]; f(one); t0 = time.perf_counter()
    for _ in range(repeats): f(one).numpy()
    return thr, 1000 * (time.perf_counter() - t0) / repeats

def count_params(model):
    tr = int(sum(np.prod(v.shape) for v in model.trainable_weights)); nt = int(sum(np.prod(v.shape) for v in model.non_trainable_weights))
    return {'parameters_total': tr + nt, 'parameters_trainable': tr}

def history_frame(csv_path, extra):
    h = pd.read_csv(csv_path).drop(columns='epoch', errors='ignore'); h.insert(0, 'epoch', np.arange(1, len(h) + 1))
    for k, v in extra.items(): h.insert(0, k, v)
    return h

def fit_finished(history_csv, epochs, patience, monitor='val_verification_auc', min_delta=1e-4):
    """True when a CSVLogger history shows the fit ended (all epochs, or early stopping would have triggered)."""
    try: h = pd.read_csv(history_csv)
    except Exception: return False
    if len(h) >= epochs: return True
    if monitor not in h or len(h) == 0: return False
    best = int(np.argmax(h[monitor].to_numpy())); return len(h) - 1 - best >= patience
