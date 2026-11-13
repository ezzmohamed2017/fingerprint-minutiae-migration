"""Model definitions, rebuilt exactly as trained so that the released ``.weights.h5`` files load by structure.

* image encoder  — CNN backbone + 1024-d embedding head;
* minutiae encoder — permutation-invariant encoder over the minutiae set;
* Siamese tower — maps both modalities into one shared space for the first-login comparison.

Layer names and creation order must not change: they define how weights are matched when loading.
"""
from __future__ import annotations

EMBEDDING_DIM = 1024
IMAGE_SIZE = 224
IMAGE_DROPOUT = 0.30

# minutiae encoder constants (fpp13_minutiae.py)
EDGE1, EDGE2, POINT = [64, 128], [256, 256], 512
MINUTIAE_DROPOUT = 0.3
GEO_SCALE = 4.0
DEFAULT_MINUTIAE_CONFIG = dict(quality=False, k=24, depth=3, N=128, aggregate="max", crop=True)

TOWER_HIDDEN = 2048
TOWER_DROPOUT = 0.1


def _tf():
    import tensorflow as tf  # noqa: WPS433
    return tf


def build_image_encoder(backbone: str = "EfficientNetV2S", image_size: int = IMAGE_SIZE, emb_dim: int = EMBEDDING_DIM):
    """uint8 (B, 224, 224, 1) letterboxed greyscale -> L2-normalised (B, 1024)."""
    tf = _tf()
    lyr = tf.keras.layers
    bb = getattr(tf.keras.applications, backbone)(include_top=False, weights=None,
                                                  input_shape=(image_size, image_size, 3), pooling="avg")
    image = tf.keras.Input((image_size, image_size, 1), dtype=tf.uint8, name="fingerprint_uint8")
    x = lyr.Lambda(lambda z: tf.cast(z, tf.float32), name="cast_float")(image)
    x = lyr.Concatenate(name="gray_to_rgb")([x, x, x])
    f = bb(x, training=False)
    f = lyr.Dropout(IMAGE_DROPOUT, name="head_dropout")(f)
    e = lyr.Dense(emb_dim, use_bias=False, name="embedding_dense")(f)
    e = lyr.BatchNormalization(name="embedding_bn")(e)
    e = lyr.Lambda(lambda z: tf.math.l2_normalize(tf.cast(z, tf.float32), axis=1), name="embedding")(e)
    return tf.keras.Model(image, e, name=f"{backbone}_full_encoder")


def build_minutiae_encoder(cfg: dict | None = None, emb_dim: int = EMBEDDING_DIM):
    """float32 (B, N, 8) minutiae set (see ``fpmig.minutiae.to_set``) -> L2-normalised (B, 1024)."""
    tf = _tf()
    lyr = tf.keras.layers
    cfg = dict(DEFAULT_MINUTIAE_CONFIG, **(cfg or {}))

    class MaskedPool(lyr.Layer):
        def call(self, inputs):
            h, m = inputs
            m3 = tf.cast(m, h.dtype)[..., None]
            cnt = tf.reduce_sum(m3, 1)
            mean = tf.reduce_sum(h * m3, 1) / tf.maximum(cnt, 1.0)
            mx = tf.reduce_max(h + (m3 - 1.0) * 1e9, 1)
            return tf.concat([tf.where(cnt > 0, mx, tf.zeros_like(mx)), mean], -1)

    class KNN(lyr.Layer):
        def __init__(self, k, **kw):
            super().__init__(**kw)
            self.k = int(k)

        def call(self, inputs):
            x, m = inputs
            px, py = x[..., 0], x[..., 1]
            n = tf.shape(x)[1]
            d2 = (px[:, None, :] - px[:, :, None]) ** 2 + (py[:, None, :] - py[:, :, None]) ** 2
            d2 = d2 + (1.0 - m[:, None, :]) * 1e9 + tf.eye(n, dtype=d2.dtype)[None] * 2e9
            return tf.math.top_k(-d2, k=self.k).indices

    class EdgeGeometry(lyr.Layer):
        def __init__(self, quality=False, **kw):
            super().__init__(**kw)
            self.quality = bool(quality)

        def call(self, inputs):
            x, m, idx = inputs
            xj = tf.gather(x, idx, batch_dims=1)
            xi = x[:, :, None, :]
            dx = xj[..., 0] - xi[..., 0]
            dy = xj[..., 1] - xi[..., 1]
            ci, si, cj, sj = xi[..., 2], xi[..., 3], xj[..., 2], xj[..., 3]
            u = dx * ci + dy * si
            v = -dx * si + dy * ci
            dist = tf.sqrt(u * u + v * v + 1e-12)
            ti = tf.broadcast_to(xi[..., 4:7], tf.shape(xj[..., 4:7]))
            valid = tf.gather(m, idx, batch_dims=1) * m[:, :, None]
            parts = [tf.stack([GEO_SCALE * u, GEO_SCALE * v, GEO_SCALE * dist, cj * ci + sj * si, sj * ci - cj * si], -1), ti, xj[..., 4:7]]
            if self.quality:
                parts += [tf.broadcast_to(xi[..., 7:8], tf.shape(xj[..., 7:8])), xj[..., 7:8]]
            return tf.concat(parts + [valid[..., None]], -1)

    class MaskedAttnK(lyr.Layer):
        def call(self, inputs):
            e, s, valid = inputs
            logits = s[..., 0] + (valid - 1.0) * 1e9
            w = tf.nn.softmax(logits, axis=2)
            out = tf.reduce_sum(w[..., None] * e, 2)
            return tf.where(tf.reduce_sum(valid, 2, keepdims=True) > 0, out, tf.zeros_like(out))

    class GatherNeighbours(lyr.Layer):
        def call(self, inputs):
            h, idx = inputs
            return tf.gather(h, idx, batch_dims=1)

    class MaskedMaxK(lyr.Layer):
        def call(self, inputs):
            e, valid = inputs
            v = valid[..., None]
            mx = tf.reduce_max(e + (v - 1.0) * 1e9, 2)
            return tf.where(tf.reduce_sum(v, 2) > 0, mx, tf.zeros_like(mx))

    def aggregate(e, valid, name):
        if cfg.get("aggregate", "max") == "max":
            return MaskedMaxK(name=name)([e, valid])
        return MaskedAttnK(name=name)([e, lyr.Dense(1, name=f"{name}_attn_score")(e), valid])

    f_ = 8 if cfg["quality"] else 7
    k = cfg["k"]
    inp = tf.keras.Input((cfg["N"], f_ + 1), dtype=tf.float32, name="minutiae_set")
    x = lyr.Lambda(lambda z: z[..., :f_], name="features")(inp)
    m = lyr.Lambda(lambda z: z[..., f_], name="mask")(inp)
    idx = KNN(k, name="knn")([x, m])
    eg = EdgeGeometry(cfg["quality"], name="edge_geometry")([x, m, idx])
    geo = lyr.Lambda(lambda z: z[..., :-1], name="edge_features")(eg)
    valid = lyr.Lambda(lambda z: z[..., -1], name="edge_valid")(eg)
    e = geo
    for i, u in enumerate(EDGE1):
        e = lyr.Dense(u, activation="relu", name=f"edge1_dense_{i}")(e)
    h = aggregate(e, valid, "local_structure_1")
    hs = [h]
    for layer in range(2, cfg["depth"] + 1):
        hj = GatherNeighbours(name=f"gather_{layer}")([h, idx])
        hi = lyr.Lambda(lambda z: tf.repeat(z[:, :, None, :], k, axis=2), name=f"tile_{layer}")(h)
        e2 = lyr.Concatenate(name=f"edge_concat_{layer}")([hi, lyr.Subtract(name=f"edge_diff_{layer}")([hj, hi]), geo])
        for i, u in enumerate(EDGE2):
            e2 = lyr.Dense(u, activation="relu", name=f"edge{layer}_dense_{i}")(e2)
        h = aggregate(e2, valid, f"local_structure_{layer}")
        hs.append(h)
    h = lyr.Dense(POINT, activation="relu", name="point_dense")(lyr.Concatenate(name="local_concat")(hs) if len(hs) > 1 else hs[0])
    pooled = MaskedPool(name="pool")([h, m])
    f = lyr.Dropout(MINUTIAE_DROPOUT, name="head_dropout")(pooled)
    e = lyr.Dense(emb_dim, use_bias=False, name="embedding_dense")(f)
    e = lyr.BatchNormalization(name="embedding_bn")(e)
    e = lyr.Lambda(lambda z: tf.math.l2_normalize(z, axis=1), name="embedding")(e)
    return tf.keras.Model(inp, e, name="minutiae_local_structure_encoder")


def build_siamese_towers(emb_dim: int = EMBEDDING_DIM):
    """Return (minutiae_tower, image_tower): one shared MLP, a learned offset per modality, L2-normalised outputs."""
    tf = _tf()
    lyr = tf.keras.layers

    class ModalityOffset(lyr.Layer):
        def build(self, shape):
            self.b = self.add_weight(name="offset", shape=(int(shape[-1]),), initializer="zeros")

        def call(self, x):
            return x + self.b

    xm = tf.keras.Input((emb_dim,), name="minutiae_embedding")
    xi = tf.keras.Input((emb_dim,), name="image_embedding")

    def l2(n):
        return lyr.Lambda(lambda z: tf.math.l2_normalize(z, axis=1), name=n)

    om = ModalityOffset(name="minutiae_offset")
    oi = ModalityOffset(name="image_offset")
    hd = lyr.Dense(TOWER_HIDDEN, activation="gelu", name="shared_hidden")
    dr = lyr.Dropout(TOWER_DROPOUT, name="shared_dropout")
    out_ = lyr.Dense(emb_dim, name="shared_out")
    f = tf.keras.Model(xm, l2("minutiae_shared_embedding")(out_(dr(hd(om(xm))))), name="siamese_minutiae")
    g = tf.keras.Model(xi, l2("image_shared_embedding")(out_(dr(hd(oi(xi))))), name="siamese_image")
    return f, g
