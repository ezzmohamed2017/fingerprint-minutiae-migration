"""fpp_lib — shared library of the FPP paper series (FingerPrint-Final workspace).

Protocol (fixed for every FPP kernel): MOLF DB1 + DB2, subject-disjoint 60 train / 15 validation / 5 calibration hold-out (bias check only; called
'adaptation' in the FPF00 split files) / 20 test subjects, 3 repetitions;
training on DB1, DB2 and DB1xDB2 cross-sensor pairs; early stopping, LR drops, best epoch and model selection on the SOURCE-BALANCED validation EER
(DB1, DB2 and DB1xDB2 validation weighted equally, one threshold; AUC breaks ties), validation only; FingerDOS (60 subjects) = external test, never used for
training, thresholds or selection. ONE frozen threshold per run = source-balanced EER of the VALIDATION pairs (DB1, DB2, DB1xDB2 weighted equally), raw
scores, no centering, applied unchanged to MOLF test and FingerDOS (validation rows are 'development' numbers). The 5 calibration subjects are used only to check
that the validation threshold is unbiased (calibration_check).
Paper metrics: AUC, EER, FMR, FNMR, F1-macro, accuracy, precision, recall (+ time, reported by each kernel).
Pair files: written once by FPP01 to output/FPP01_Evaluation_Protocol_Subject_Disjoint_Splits/protocol/<rep>/pairs/ (sha256 in the manifest) and read by every kernel.
Builds on fpf_lib (workspace, run folders, workbook, base metrics)."""
import itertools, json, os, time
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from scipy.stats import norm
import fpf_lib as L

VERSION = '1.13'   # 1.13: generator route dropped; FPP14 = timing (T4 accelerator as NPU proxy + CPU), FPP15 report; 1.12: 1.12: FPP13 = joint training of the minutiae encoder and the FPP12 Siamese matcher, FPP14 = generator route, FPP15 timing, FPP16 report; 1.11: FPP12 = migration matcher heads on both sides, FPP13 = generator route (image fine-tuning dropped: it would change the image templates); 1.10: FPP10 = cross-modal matching with frozen encoders (mapping only), FPP11 = minutiae encoder fine-tuned into the frozen image space; 1.9: FPP10 migration; 1.8: FPP08 = quality / crop / third layer; 1.7: minutiae branch FPP06 architecture, FPP07 neighbourhood/attention, FPP08 augmentation+quality, FPP09 shared-proxy loss; 1.5: validation_table / decision rule helpers
# 1.4: 1.4: FPP04L renamed (shared identity proxy losses)
# 1.3: 1.3: threshold = source-balanced VALIDATION EER (was training pairs); 5-subject calibration hold-out bias check
# 1.2: source-balanced validation EER for early stopping / best epoch / selection; balanced MOLF rows
FPP_KERNELS = {
    'FPP01': 'FPP01_Evaluation_Protocol_Subject_Disjoint_Splits',
    'FPP02': 'FPP02_FingerJetFX_Minutiae_and_Image_Caches',
    'FPP03': 'FPP03_BOZORTH3_Minutiae_Verification_Baseline',
    'FPP04': 'FPP04_Image_Siamese_Backbone_Ablation',
    'FPP04C': 'FPP04C_Selected_Backbone_Full_FineTuning_Augmentation',
    'FPP04L': 'FPP04L_Shared_Identity_Proxy_Losses',
    'FPP05': 'FPP05_Minutiae_Guided_Early_Layer_Attention',
    'FPP06': 'FPP06_Minutiae_Encoder_Architecture_Ablation',
    'FPP07': 'FPP07_Minutiae_Neighbourhood_and_Attention_Ablation',
    'FPP08': 'FPP08_Minutiae_Quality_Crop_and_Depth',
    'FPP09': 'FPP09_Minutiae_Shared_Proxy_Loss',
    'FPP10': 'FPP10_CrossModal_Matching_with_Frozen_Encoders',
    'FPP11': 'FPP11_Migration_Encoder_into_Frozen_Image_Space',
    'FPP12': 'FPP12_Migration_Matcher_Heads_on_Both_Sides',
    'FPP13': 'FPP13_Joint_Minutiae_Encoder_and_Siamese_Matcher',
    'FPP14': 'FPP14_Timing_Accelerator_and_CPU',
    'FPP15': 'FPP15_Final_Statistics_Figures_and_Report'}

# ------------------------------------------------------------------ protocol
PAIR_SETS = {   # name: (dataset of image a, dataset of image b, partition)
    'DB1_train': ('MOLF_DB1', 'MOLF_DB1', 'train'), 'DB2_train': ('MOLF_DB2', 'MOLF_DB2', 'train'), 'DB1xDB2_train': ('MOLF_DB1', 'MOLF_DB2', 'train'),
    'DB1_validation': ('MOLF_DB1', 'MOLF_DB1', 'validation'), 'DB2_validation': ('MOLF_DB2', 'MOLF_DB2', 'validation'), 'DB1xDB2_validation': ('MOLF_DB1', 'MOLF_DB2', 'validation'),
    'DB1_test': ('MOLF_DB1', 'MOLF_DB1', 'test'), 'DB2_test': ('MOLF_DB2', 'MOLF_DB2', 'test'), 'DB1xDB2_test': ('MOLF_DB1', 'MOLF_DB2', 'test'),
    'FingerDOS_test': ('FingerDOS', 'FingerDOS', 'external_test')}
JOINT_PAIR_FILES = ('DB2_train', 'DB1xDB2_train', 'DB2_validation', 'DB1xDB2_validation')       # order = RNG streams [seed, 1000 + k] (identical to FPF02)
TRAIN_SOURCES = ('DB1_train', 'DB2_train', 'DB1xDB2_train'); VAL_SOURCES = ('DB1_validation', 'DB2_validation', 'DB1xDB2_validation')
MOLF_TESTS = ('DB1_test', 'DB2_test', 'DB1xDB2_test'); TEST_SETS = MOLF_TESTS + ('FingerDOS_test',)
GROUPS = {'MOLF validation (pooled)': VAL_SOURCES, 'MOLF test (pooled)': MOLF_TESTS, 'MOLF test same-sensor (pooled)': ('DB1_test', 'DB2_test')}
DEV = 'development (threshold and selection tuned here)'
ROLE = {**{n: 'training' for n in TRAIN_SOURCES}, **{n: DEV for n in VAL_SOURCES}, **{n: 'test' for n in MOLF_TESTS}, 'FingerDOS_test': 'external test',
        'MOLF validation (pooled)': DEV, 'MOLF test (pooled)': 'test', 'MOLF test same-sensor (pooled)': 'test',
        'MOLF validation (balanced)': DEV, 'MOLF test (balanced)': 'test'}
# Source-balanced groups: DB1, DB2 and DB1xDB2 count 1/3 each whatever their pair counts (the cross-sensor set has the most pairs). One threshold for all members.
BALANCED_GROUPS = {'MOLF validation (balanced)': VAL_SOURCES, 'MOLF test (balanced)': MOLF_TESTS}
REPORT_ORDER = ['DB1_validation', 'DB2_validation', 'DB1xDB2_validation', 'MOLF validation (pooled)', 'MOLF validation (balanced)', 'DB1_test', 'DB2_test', 'DB1xDB2_test',
                'MOLF test (pooled)', 'MOLF test (balanced)', 'MOLF test same-sensor (pooled)', 'FingerDOS_test']
PAPER_SETS = ['MOLF validation (pooled)', 'MOLF test (pooled)', 'DB1xDB2_test', 'FingerDOS_test']
SELECTION_SET = 'MOLF validation (balanced)'
SELECTION_RULE = ('lowest source-balanced validation EER (DB1, DB2, DB1xDB2 validation weighted equally, one threshold; ties within 0.01 EER points -> higher balanced AUC); '
                  'the same rule drives early stopping, LR drops and the best epoch')
THRESHOLD_KEY = 'validation_EER'
THRESHOLD_RULE = ('validation_EER: source-balanced EER of the DB1, DB2 and DB1xDB2 validation pairs (one threshold, sources weighted equally), raw scores, '
                  'frozen for MOLF test and FingerDOS; validation rows are development numbers')
PROTOCOL_TABLE = '60 train / 15 validation (early stopping, selection, threshold) / 5 calibration hold-out (bias check only) / 20 test MOLF subjects; FingerDOS = external test'
METRICS = ['roc_auc', 'eer', 'fmr', 'fnmr', 'f1_macro', 'accuracy', 'precision', 'recall']
LABEL = {'roc_auc': 'AUC ↑', 'eer': 'EER ↓', 'fmr': 'FMR ↓', 'fnmr': 'FNMR ↓', 'f1_macro': 'F1-macro ↑', 'accuracy': 'Accuracy ↑', 'precision': 'Precision ↑', 'recall': 'Recall ↑'}
LOWER_IS_BETTER = {'eer', 'fmr', 'fnmr'}
PERCENT = METRICS + ['balanced_accuracy', 'specificity_tnr', 'f1_genuine']

def kernel_dir(ws, key):
    d = Path(ws['output']) / FPP_KERNELS[key]; d.mkdir(parents=True, exist_ok=True); return d
def protocol_dir(ws): return Path(ws['output']) / FPP_KERNELS['FPP01'] / 'protocol'

def load_pairs(ws, rep, verify=True):
    """FPF00 protocol (splits, samples) + the 10 FPP pair sets written by FPP01."""
    proto = L.load_protocol(ws, rep); d = protocol_dir(ws) / rep; mf = d / 'fpp_pairs_manifest.json'
    if not mf.is_file(): raise FileNotFoundError(f'{mf} missing: run FPP01 first')
    man = json.loads(mf.read_text()); P = {}
    for n in PAIR_SETS:
        f = d / 'pairs' / f'{n}_pairs.npz'
        if verify and L.sha256_of(f) != man['files'][n]['sha256']: raise RuntimeError(f'{rep} {n}: pair file changed since FPP01 (sha256 mismatch)')
        P[n] = L.load_npz(f)
    return proto, P

# ------------------------------------------------------------------ joint pair sampler (identical to FPF00 / FPF02)
def genuine_within(f, rows):
    a, b = [], []
    for _, g in f.loc[rows].groupby('identity_id'):
        for i, j in itertools.combinations(g.sample_index.to_numpy(), 2): a.append(i); b.append(j)
    return np.array(a, np.int64), np.array(b, np.int64)

def impostors(fa, fb, rows_a, rows_b, n, rng, same_fraction, cross=False):
    A = fa.loc[rows_a]; B = fb.loc[rows_b]
    a_idx = A.sample_index.to_numpy(); a_sub = A.subject_id.to_numpy(); a_id = A.identity_id.to_numpy()
    b_idx = B.sample_index.to_numpy(); b_sub = B.subject_id.to_numpy(); b_id = B.identity_id.to_numpy()
    n_same = int(round(n * same_fraction)); seen = set(); out = []
    def key(i, j): return (i, j) if cross else (min(i, j), max(i, j))
    for want, same in ((n_same, True), (n - n_same, False)):
        got, tries = 0, 0
        while got < want:
            tries += 1
            if tries > 200 * max(want, 1): raise RuntimeError('cannot sample enough impostor pairs')
            k = rng.integers(len(a_idx))
            cand = np.flatnonzero((b_sub == a_sub[k]) & (b_id != a_id[k])) if same else np.flatnonzero(b_sub != a_sub[k])
            if len(cand) == 0: continue
            j = cand[rng.integers(len(cand))]; kk = key(int(a_idx[k]), int(b_idx[j]))
            if kk in seen or (not cross and kk[0] == kk[1]): continue
            seen.add(kk); out.append((kk[0], kk[1], 1 if same else 2)); got += 1
    o = np.array(out, np.int64); return o[:, 0], o[:, 1], o[:, 2]

def make_pairs(fa, fb, rows_a, rows_b, ratio, rng, same_fraction, cross=False):
    if cross:
        A = fa.loc[rows_a]; B = fb.loc[rows_b].groupby('identity_id').sample_index.apply(np.asarray).to_dict(); ga, gb = [], []
        for i, ident in zip(A.sample_index.to_numpy(), A.identity_id.to_numpy()):
            for j in B[ident]: ga.append(i); gb.append(j)
        ga, gb = np.array(ga, np.int64), np.array(gb, np.int64)
    else: ga, gb = genuine_within(fa, rows_a)
    ia, ib, nt = impostors(fa, fb, rows_a, rows_b, int(round(ratio * len(ga))), rng, same_fraction, cross)
    order = rng.permutation(len(ga) + len(ia))
    return {'index_a': np.r_[ga, ia][order], 'index_b': np.r_[gb, ib][order], 'label': np.r_[np.ones(len(ga), np.int8), np.zeros(len(ia), np.int8)][order],
            'negative_type': np.r_[np.zeros(len(ga), np.int8), nt.astype(np.int8)][order]}

def build_joint_pairs(proto):
    """The 4 DB2 / DB1xDB2 pair sets with the FPF02 seeds (deterministic: byte-identical to FPF02's files)."""
    man = proto['manifest']; S = {ds: proto['samples'][ds] for ds in ('MOLF_DB1', 'MOLF_DB2')}; out = {}
    for k, name in enumerate(JOINT_PAIR_FILES):
        da, db, part = PAIR_SETS[name]; cross = da != db; ratio = man['train_impostor_ratio'] if part == 'train' else man['eval_impostor_ratio']
        ra = S[da].index[S[da].partition == part]; rb = S[db].index[S[db].partition == part]
        out[name] = make_pairs(S[da], S[db], ra, rb, ratio, np.random.default_rng([proto['seed'], 1000 + k]), man['same_subject_impostor_fraction'], cross)
    return out

class MixedPairs:
    """Source-balanced pair sampler (numpy core of the FPF02 MixedPairSequence): every epoch takes the same number of balanced pairs from each source
    (half genuine; impostors same_subject_fraction from the same subject), cycling through each pool; validation (shuffle=False) is deterministic."""
    def __init__(self, sources, batch_size, shuffle, seed, pairs_per_epoch=None, same_subject_fraction=0.25):
        self.batch_size = int(batch_size); self.shuffle = bool(shuffle); self.rng = np.random.default_rng(seed)
        self.ppe = None if pairs_per_epoch is None else int(pairs_per_epoch); self.ssf = float(same_subject_fraction); self.names = [s[0] for s in sources]
        ia, ib, lab, nt, src = [], [], [], [], []
        for k, (name, P, off_a, off_b) in enumerate(sources):
            ia.append(np.asarray(P['index_a'], np.int64) + off_a); ib.append(np.asarray(P['index_b'], np.int64) + off_b); lab.append(np.asarray(P['label'], np.int8))
            nt.append(np.asarray(P['negative_type'], np.int8)); src.append(np.full(len(P['label']), k, np.int8))
        self.pairs = {'index_a': np.concatenate(ia), 'index_b': np.concatenate(ib), 'label': np.concatenate(lab), 'negative_type': np.concatenate(nt), 'source': np.concatenate(src)}
        L_, N_, S_ = self.pairs['label'], self.pairs['negative_type'], self.pairs['source']; self.pools = {}
        for k in range(len(sources)):
            self.pools[(k, 'positive')] = np.flatnonzero((S_ == k) & (L_ == 1)); self.pools[(k, 'same')] = np.flatnonzero((S_ == k) & (L_ == 0) & (N_ == 1))
            self.pools[(k, 'diff')] = np.flatnonzero((S_ == k) & (L_ == 0) & (N_ != 1))
        self.cursors = {k: 0 for k in self.pools}
        if self.shuffle:
            for p in self.pools.values(): self.rng.shuffle(p)
        self.build()
    def __len__(self): return int(np.ceil(len(self.order) / self.batch_size))
    def batch(self, b):
        pos = self.order[b * self.batch_size:(b + 1) * self.batch_size]; return self.pairs['index_a'][pos], self.pairs['index_b'][pos], self.pairs['label'][pos].astype(np.float32)
    def _take(self, key, n):
        pool = self.pools[key]
        if n <= 0 or len(pool) == 0: return np.empty(0, np.int64)
        if not self.shuffle: return pool[:min(n, len(pool))]
        out = []; r = n
        while r > 0:
            c = self.cursors[key]; t = min(r, len(pool) - c); out.append(pool[c:c + t]); self.cursors[key] += t; r -= t
            if self.cursors[key] >= len(pool): self.rng.shuffle(pool); self.cursors[key] = 0
        return np.concatenate(out)
    def build(self):
        K = len(self.names); n_pos_min = min(len(self.pools[(k, 'positive')]) for k in range(K))
        n_pos = n_pos_min if self.ppe is None else self.ppe // (2 * K)
        n_pos = min(n_pos, max(len(self.pools[(k, 'positive')]) for k in range(K)))
        if not self.shuffle: n_pos = min(n_pos, n_pos_min)
        units, self.composition = [], {}
        for k in range(K):
            pos = self._take((k, 'positive'), n_pos); n_same = int(round(n_pos * self.ssf)) if len(self.pools[(k, 'same')]) else 0
            neg = np.concatenate([self._take((k, 'same'), n_same), self._take((k, 'diff'), n_pos - n_same)])
            if self.shuffle: self.rng.shuffle(neg)
            m = min(len(pos), len(neg)); units.append(np.column_stack([pos[:m], neg[:m]])); self.composition[self.names[k]] = 2 * m
        u = np.concatenate(units)
        if self.shuffle: u = u[self.rng.permutation(len(u))]
        self.order = u.reshape(-1)
    def on_epoch_end(self):
        if self.shuffle: self.build()

def sources(P, names, offset):
    return [(n, P[n], offset[PAIR_SETS[n][0]], offset[PAIR_SETS[n][1]]) for n in names]

# ------------------------------------------------------------------ materialized training augmentation (written once by FPP01, read by training kernels)
AUG_COPIES = 3
AUG_SEED = 20261107
AUG_DATASETS = ('MOLF_DB1', 'MOLF_DB2')                  # FingerDOS is never trained on -> never augmented
AUG_RECIPE = dict(rotation_degrees=5.0, translation_fraction=0.03, zoom_fraction=0.0, brightness_fraction=0.10, contrast_fraction=0.15, noise_std_grey_levels=6.0,
                  fill_value=255, interpolation='bilinear', order='rotation+translation(+zoom) -> brightness -> contrast -> noise', applied='always (each stored copy is augmented)')
AUG_PARAM_NAMES = ('angle_deg', 'shift_x_px', 'shift_y_px', 'zoom', 'brightness_delta', 'contrast_factor')

def augmentation_file(ws, dataset):
    return Path(ws['output']) / FPP_KERNELS['FPP01'] / 'augmentation' / f'{dataset}_augmented_x{AUG_COPIES}_224.h5'

def augment_image(img, rng, recipe=AUG_RECIPE):
    """One mild augmentation of a uint8 (H, W[, 1]) image (FPM04C 'mild' recipe, the same ranges as the Keras layers): returns (uint8 (H, W, 1), parameters)."""
    import cv2
    a = recipe; x = np.asarray(img, np.uint8).reshape(img.shape[0], img.shape[1]); h, w = x.shape
    ang = rng.uniform(-a['rotation_degrees'], a['rotation_degrees']); tx = rng.uniform(-a['translation_fraction'], a['translation_fraction']) * w
    ty = rng.uniform(-a['translation_fraction'], a['translation_fraction']) * h; zm = 1.0 + rng.uniform(-a['zoom_fraction'], a['zoom_fraction']) if a['zoom_fraction'] > 0 else 1.0
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), ang, zm); M[0, 2] += tx; M[1, 2] += ty
    y = cv2.warpAffine(x, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=int(a['fill_value'])).astype(np.float32)
    bd = rng.uniform(-a['brightness_fraction'], a['brightness_fraction']); y = y + 255.0 * bd
    cf = rng.uniform(1 - a['contrast_fraction'], 1 + a['contrast_fraction']); mu = y.mean(); y = (y - mu) * cf + mu
    y = y + rng.normal(0.0, a['noise_std_grey_levels'], y.shape)
    return np.clip(np.rint(y), 0, 255).astype(np.uint8)[..., None], np.array([ang, tx, ty, zm, bd, cf], np.float32)

def augment_copies(img, dataset, index, copies=AUG_COPIES):
    """The stored copies of one image: copy k uses the generator [AUG_SEED, dataset code, image index, k] (reproducible, independent of batch order)."""
    code = AUG_DATASETS.index(dataset); out, par = [], []
    for k in range(copies):
        a, p = augment_image(img, np.random.default_rng([AUG_SEED, code, int(index), k])); out.append(a); par.append(p)
    return np.stack(out), np.stack(par)

def mix_versions(images, aug, idx, p_original, rng):
    """Training view of a batch of image indices: each image independently stays original with probability p_original, otherwise one of its stored copies (uniform)."""
    idx = np.asarray(idx); out = np.array(images[idx], copy=True)
    if aug is None or p_original >= 1: return out
    use = rng.random(len(idx)) >= p_original; k = rng.integers(0, aug.shape[1], len(idx))
    if use.any(): out[use] = aug[idx[use], k[use]]
    return out

# ------------------------------------------------------------------ metrics, threshold, evaluation
def metrics(y, s, threshold):
    m = L.verification_metrics(y, s, threshold); m['recall'] = m['recall_tar']
    for k in ('tar_at_fmr_1pct', 'tar_at_fmr_0_1pct', 'recall_tar'): m.pop(k, None)
    return m

def class_source_weights(ys):
    """Weights making every source count equally and, inside a source, genuine and impostor pairs count equally: weighted FMR = mean of the per-source FMRs,
    weighted FNMR = mean of the per-source FNMRs. (The rule of the training threshold.)"""
    return np.concatenate([np.where(np.asarray(yy) == 1, 1.0 / max(np.sum(yy == 1), 1), 1.0 / max(np.sum(yy == 0), 1)) for yy in ys])

def source_weights(ys):
    """Each source weighted 1/3 overall, natural genuine : impostor mix inside it (for accuracy, precision, F1)."""
    return np.concatenate([np.full(len(yy), 1.0 / max(len(yy), 1)) for yy in ys])

def balanced_eer(ys, ss):
    """Source-balanced EER and AUC over several pair sets with ONE threshold. Returns (eer, auc, eer_threshold)."""
    y = np.concatenate([np.asarray(v, int) for v in ys]); s = np.concatenate([np.asarray(v, float) for v in ss]); w = class_source_weights(ys)
    fpr, tpr, thr = roc_curve(y, s, sample_weight=w, drop_intermediate=False); fnr = 1 - tpr; i = int(np.nanargmin(np.abs(fpr - fnr)))
    return float((fpr[i] + fnr[i]) / 2), float(roc_auc_score(y, s, sample_weight=w)), float(thr[i])

def eer_threshold_balanced(ys, ss): return balanced_eer(ys, ss)[2]

def balanced_metrics(ys, ss, threshold):
    """The paper metrics on a source-balanced group at a given (frozen) threshold; same keys as metrics().
    AUC, EER, FMR, FNMR (and recall = 1 - FNMR): every source and class weighted equally; accuracy, precision, F1: every source weighted equally."""
    ys = [np.asarray(v, int) for v in ys]; ss = [np.asarray(v, float) for v in ss]; eer, auc, eer_thr = balanced_eer(ys, ss)
    pred = [(v >= threshold).astype(int) for v in ss]
    fmr = float(np.mean([np.mean(p[y == 0]) if np.any(y == 0) else 0.0 for y, p in zip(ys, pred)])); fnmr = float(np.mean([np.mean(1 - p[y == 1]) if np.any(y == 1) else 0.0 for y, p in zip(ys, pred)]))
    y = np.concatenate(ys); p = np.concatenate(pred); w = source_weights(ys)
    tp = float(np.sum(w * (y == 1) * (p == 1))); tn = float(np.sum(w * (y == 0) * (p == 0))); fp = float(np.sum(w * (y == 0) * (p == 1))); fn = float(np.sum(w * (y == 1) * (p == 0)))
    f1g = 2 * tp / max(2 * tp + fp + fn, 1e-12); f1i = 2 * tn / max(2 * tn + fp + fn, 1e-12)
    return {'threshold': float(threshold), 'roc_auc': auc, 'eer': eer, 'eer_threshold_on_this_set': eer_thr, 'fmr': fmr, 'fnmr': fnmr, 'f1_macro': 0.5 * (f1g + f1i), 'f1_genuine': f1g,
            'balanced_accuracy': 1 - 0.5 * (fmr + fnmr), 'accuracy': (tp + tn) / max(tp + tn + fp + fn, 1e-12), 'precision': tp / max(tp + fp, 1e-12), 'recall': 1 - fnmr,
            'specificity_tnr': 1 - fmr, 'tp': int(np.sum((y == 1) & (p == 1))), 'tn': int(np.sum((y == 0) & (p == 0))), 'fp': int(np.sum((y == 0) & (p == 1))),
            'fn': int(np.sum((y == 1) & (p == 0))), 'pairs': int(len(y)), 'genuine_pairs': int(y.sum()), 'impostor_pairs': int((1 - y).sum()), 'weighting': 'source-balanced'}

class BalancedValidation:
    """Per-epoch validation of a training kernel (framework-neutral). Holds the DB1, DB2 and DB1xDB2 validation pair sets in one global image-index space;
    `indices` = the images to embed (originals, inference mode); `score(emb)` (rows aligned with `indices`, L2-normalised) -> source-balanced EER / AUC
    and the per-source EERs. `pair_score` can be replaced for non-cosine models."""
    def __init__(self, P, offset, names=VAL_SOURCES, max_pairs=None):
        self.names = list(names); self.sets = {}
        for n in self.names:
            a = np.asarray(P[n]['index_a'], np.int64) + offset[PAIR_SETS[n][0]]; b = np.asarray(P[n]['index_b'], np.int64) + offset[PAIR_SETS[n][1]]; y = np.asarray(P[n]['label'], int)
            if max_pairs is not None and len(y) > max_pairs: k = np.linspace(0, len(y) - 1, int(max_pairs)).astype(int); a, b, y = a[k], b[k], y[k]   # smoke tests only
            self.sets[n] = (a, b, y)
        self.indices = np.unique(np.concatenate([np.concatenate([a, b]) for a, b, _ in self.sets.values()])); pos = {int(g): i for i, g in enumerate(self.indices)}
        self.local = {n: (np.array([pos[int(g)] for g in a]), np.array([pos[int(g)] for g in b]), y) for n, (a, b, y) in self.sets.items()}
        self.composition = {n: int(len(y)) for n, (_, _, y) in self.sets.items()}
    @staticmethod
    def pair_score(E, ia, ib): return np.sum(E[ia] * E[ib], axis=1)
    def score(self, emb):
        E = np.asarray(emb, np.float32); ys, ss, out = [], [], {}
        for n, (ia, ib, y) in self.local.items():
            s = self.pair_score(E, ia, ib).astype(np.float64); ys.append(y); ss.append(s); out[f'val_eer_{n}'] = balanced_eer([y], [s])[0]
        eer, auc, thr = balanced_eer(ys, ss); out.update(val_balanced_eer=eer, val_balanced_auc=auc, val_balanced_eer_threshold=thr); return out

class BalancedEERStopping:
    """Early stopping, LR drops and best epoch on the source-balanced validation EER (lower is better); an epoch within `tie` EER of the best counts as
    better only if its balanced AUC is higher. Framework-neutral: call update() once per epoch and act on the returned flags."""
    def __init__(self, patience=10, lr_patience=3, lr_factor=0.3, min_lr=1e-7, min_delta=1e-4, tie=1e-4):
        self.patience, self.lr_patience, self.lr_factor, self.min_lr, self.min_delta, self.tie = patience, lr_patience, lr_factor, min_lr, min_delta, tie
        self.best_eer, self.best_auc, self.best_epoch, self.wait, self.lr_wait = np.inf, -np.inf, -1, 0, 0
    def better(self, eer, auc):
        return eer < self.best_eer - self.min_delta or (abs(eer - self.best_eer) <= self.tie and auc > self.best_auc)
    def update(self, epoch, eer, auc):
        if self.better(eer, auc):
            self.best_eer, self.best_auc, self.best_epoch, self.wait, self.lr_wait = float(eer), float(auc), int(epoch), 0, 0; return {'improved': True, 'reduce_lr': False, 'stop': False}
        self.wait += 1; self.lr_wait += 1; reduce = self.lr_wait >= self.lr_patience
        if reduce: self.lr_wait = 0
        return {'improved': False, 'reduce_lr': reduce, 'stop': self.wait >= self.patience}
    @classmethod
    def replay(cls, history, epochs, **kw):
        """(best row position, fit finished?) from a saved history with val_balanced_eer / val_balanced_auc columns."""
        h = pd.DataFrame(history); st = cls(**kw); stop = False
        for i, (e, a) in enumerate(zip(h['val_balanced_eer'], h['val_balanced_auc'])): stop = st.update(i, e, a)['stop'] or stop
        return st.best_epoch, bool(stop or len(h) >= epochs)

def validation_threshold(S):
    """THE threshold of the FPP series: source-balanced EER of the DB1, DB2 and DB1xDB2 validation pairs."""
    return balanced_eer([S[n][0] for n in VAL_SOURCES], [S[n][1] for n in VAL_SOURCES])[2]

def training_threshold(S):
    """Former rule (FPP04 v1.2 and the FPF pilots), kept for reference only."""
    return eer_threshold_balanced([S[n][0] for n in TRAIN_SOURCES], [S[n][1] for n in TRAIN_SOURCES])

# ------------------------------------------------------------------ 5-subject calibration hold-out (bias check of the validation threshold)
CALIBRATION_SOURCES = {'DB1_calibration': ('MOLF_DB1', 'MOLF_DB1'), 'DB2_calibration': ('MOLF_DB2', 'MOLF_DB2'), 'DB1xDB2_calibration': ('MOLF_DB1', 'MOLF_DB2')}

def calibration_pairs(proto):
    """Pairs among the 5 hold-out subjects (partition 'adaptation' in the split files), same generator and ratios as every FPP pair set, seed [rep_seed, 3000 + k]."""
    man = proto['manifest']; S = {d: proto['samples'][d] for d in ('MOLF_DB1', 'MOLF_DB2')}; out = {}
    for k, (name, (da, db)) in enumerate(CALIBRATION_SOURCES.items()):
        ra = S[da].index[S[da].partition == 'adaptation']; rb = S[db].index[S[db].partition == 'adaptation']
        if len(ra) and len(rb):
            out[name] = make_pairs(S[da], S[db], ra, rb, man['eval_impostor_ratio'], np.random.default_rng([proto['seed'], 3000 + k]), man['same_subject_impostor_fraction'], da != db)
    return out

def calibration_check(S, E, cal_pairs, pair_score=None):
    """Bias check: threshold from the 5 hold-out subjects vs the validation threshold, both applied to MOLF test (balanced). E = embeddings by dataset."""
    f = pair_score or (lambda Ea, Eb, p: L.pair_scores(Ea, Eb, p))
    cal = [(np.asarray(p['label'], int), f(E[CALIBRATION_SOURCES[n][0]], E[CALIBRATION_SOURCES[n][1]], p).astype(np.float64)) for n, p in cal_pairs.items()]
    t_val = validation_threshold(S); t_cal = balanced_eer([c[0] for c in cal], [c[1] for c in cal])[2]; out = {'threshold_validation': t_val, 'threshold_calibration': t_cal}
    for tag, t in (('validation', t_val), ('calibration', t_cal)):
        m = balanced_metrics([S[n][0] for n in MOLF_TESTS], [S[n][1] for n in MOLF_TESTS], t)
        out.update({f'test_fmr_{tag}': m['fmr'], f'test_fnmr_{tag}': m['fnmr'], f'test_f1_{tag}': m['f1_macro']})
    out['calibration_pairs'] = int(sum(len(c[0]) for c in cal)); return out

def group_scores(S, name):
    mem = GROUPS.get(name, (name,)); return np.concatenate([S[n][0] for n in mem]), np.concatenate([S[n][1] for n in mem])

def evaluate(S, kernel, arm, rep, extra=None, threshold=None):
    """S: {pair_set: (label, score)} for the 10 sets. One frozen threshold (source-balanced validation EER unless given); every set, pooled and balanced group
    uses it unchanged."""
    thr = validation_threshold(S) if threshold is None else float(threshold); rows = []
    for name in list(PAIR_SETS) + list(GROUPS):
        y, s = group_scores(S, name); m = metrics(y, s, thr)
        m.update({'kernel': kernel, 'arm': arm, 'repetition': rep, 'set': name, 'members': ' + '.join(GROUPS.get(name, (name,))), 'role': ROLE[name], 'threshold_rule': THRESHOLD_KEY if threshold is None else 'given', **(extra or {})})
        rows.append(m)
    for name, mem in BALANCED_GROUPS.items():
        m = balanced_metrics([S[n][0] for n in mem], [S[n][1] for n in mem], thr)
        m.update({'kernel': kernel, 'arm': arm, 'repetition': rep, 'set': name, 'members': ' + '.join(mem) + ' (weighted equally)', 'role': ROLE[name], 'threshold_rule': THRESHOLD_KEY if threshold is None else 'given', **(extra or {})})
        rows.append(m)
    return rows, thr

LOCAL_RUNS = Path('/content/fpp_local_runs')   # Colab disk: per-epoch checkpoints and all run files live here until the run is finished

def local_run_dir(kernel, rep, arm, root=None):
    d = Path(root or LOCAL_RUNS) / kernel / rep / arm; d.mkdir(parents=True, exist_ok=True); return d

def publish_run(local, drive, skip=()):
    """Copies a finished local run folder to Drive once; run_complete.json is copied LAST, so a Drive run counts as complete only when every file is there."""
    import shutil
    local, drive = Path(local), Path(drive); drive.mkdir(parents=True, exist_ok=True); files = [f for f in sorted(local.rglob('*')) if f.is_file() and f.name not in skip]
    for f in sorted(files, key=lambda f: f.name == 'run_complete.json'):
        dst = drive / f.relative_to(local); dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(f, dst)
    return len(files)

def save_run(rd, S, rows, thr, info):
    rd = Path(rd); save_scores(S, rd); pd.DataFrame(rows).to_csv(rd / 'metrics.csv', index=False)
    t_tr = training_threshold(S) if all(n in S for n in TRAIN_SOURCES) else float('nan')
    (rd / 'thresholds.json').write_text(json.dumps({THRESHOLD_KEY: thr, 'rule': THRESHOLD_RULE, 'train_EER_reference_only': t_tr}, indent=2))
    L.mark_complete(rd, {'threshold_' + THRESHOLD_KEY: thr, 'threshold_rule': THRESHOLD_KEY, **info})

def save_scores(S, rd):
    np.savez_compressed(Path(rd) / 'pair_scores.npz', **{f'{n}__raw__score': np.asarray(s, np.float32) for n, (y, s) in S.items()}, **{f'{n}__label': np.asarray(y, np.int8) for n, (y, s) in S.items()})
def load_scores(rd):
    z = np.load(Path(rd) / 'pair_scores.npz'); return {n: (z[f'{n}__label'].astype(int), z[f'{n}__raw__score'].astype(np.float64)) for n in PAIR_SETS if f'{n}__raw__score' in z.files}
def load_threshold(rd):
    """The run's frozen threshold under the CURRENT rule (recomputed from the saved scores for runs evaluated under an older rule)."""
    t = json.loads((Path(rd) / 'thresholds.json').read_text())
    return float(t[THRESHOLD_KEY]) if THRESHOLD_KEY in t else validation_threshold(load_scores(rd))

def threshold_current(rd):
    try: return THRESHOLD_KEY in json.loads((Path(rd) / 'thresholds.json').read_text())
    except Exception: return False

def to_percent(frame):
    f = frame.copy()
    for c in PERCENT:
        if c in f.columns: f[c] = 100 * f[c]
    return f

def summarize(results, keys):
    pct = to_percent(results); cols = [c for c in PERCENT if c in pct.columns]; g = pct.groupby(list(keys))[cols]
    out = g.mean().add_suffix('_mean').join(g.std(ddof=1).add_suffix('_std')).join(g.size().rename('repetitions')).reset_index()
    return out

def paper_table(results, models=None, sets=REPORT_ORDER, model_col='kernel'):
    """Formatted 'mean ± std' table of the paper metrics per set and model, plus the numeric summary."""
    s = summarize(results, [model_col, 'set']); s = s[s.set.isin(sets)].copy(); models = models or list(dict.fromkeys(results[model_col]))
    s = s[s[model_col].isin(models)]; s['set'] = pd.Categorical(s.set, list(sets)); s[model_col] = pd.Categorical(s[model_col], models); s = s.sort_values(['set', model_col])
    out = s[['set', model_col]].copy(); out.insert(1, 'role', [ROLE.get(x, '') for x in s.set.astype(str)])
    for m in METRICS: out[LABEL[m]] = [f'{a:.2f} ± {0 if np.isnan(b) else b:.2f}' for a, b in zip(s[f'{m}_mean'], s[f'{m}_std'])]
    out['repetitions'] = s['repetitions'].to_numpy()
    return out.rename(columns={model_col: 'model'}), s

def wide_table(snum, models, sets=PAPER_SETS, metrics_=('roc_auc', 'eer', 'f1_macro'), model_col='kernel'):
    """One row per model: '<set> <metric>' columns (mean ± std) — the compact paper table."""
    rows = []
    for mod in models:
        r = {'model': mod}
        for st in sets:
            q = snum[(snum[model_col].astype(str) == mod) & (snum.set.astype(str) == st)]
            for m in metrics_: r[f'{st} {LABEL[m]}'] = f"{q[f'{m}_mean'].iloc[0]:.2f} ± {0 if np.isnan(q[f'{m}_std'].iloc[0]) else q[f'{m}_std'].iloc[0]:.2f}" if len(q) else ''
        rows.append(r)
    return pd.DataFrame(rows)

def paired_differences(results, cur, refs, model_col='kernel'):
    key = ['repetition', 'set']; c = results[results[model_col] == cur].set_index(key); out = []
    for ref in refs:
        b = results[results[model_col] == ref].set_index(key)
        for m in METRICS:
            d = (100 * (c[m] - b[m])).dropna()
            for st, g in d.groupby(level='set'):
                if st not in REPORT_ORDER: continue
                better = (g < 0) if m in LOWER_IS_BETTER else (g > 0)
                out.append({'model': cur, 'vs': ref, 'set': st, 'metric': m, 'mean_difference_points': g.mean(), 'std': g.std(ddof=1), 'min': g.min(), 'max': g.max(),
                            'repetitions': len(g), 'repetitions_better': int(better.sum()), 'same_sign_all_reps': bool((g > 0).all() or (g < 0).all())})
    return pd.DataFrame(out)

def selection_table(results, candidates, model_col='arm', set_name=SELECTION_SET):
    v = results[results[model_col].isin(candidates) & (results.set == set_name)]
    t = v.groupby(model_col).agg(val_auc_mean=('roc_auc', 'mean'), val_auc_std=('roc_auc', 'std'), val_eer_mean=('eer', 'mean'), val_f1_macro_mean=('f1_macro', 'mean'),
                                 val_accuracy_mean=('accuracy', 'mean'), repetitions=('roc_auc', 'size'))
    for c in t.columns:
        if c != 'repetitions': t[c] = 100 * t[c]
    t['val_eer_std'] = 100 * v.groupby(model_col)['eer'].std()
    return t.sort_values(['val_eer_mean', 'val_auc_mean'], ascending=[True, False]).reset_index()   # SELECTION_RULE

VALIDATION_SETS = ['MOLF validation (balanced)'] + list(VAL_SOURCES)
DECISION_RULE = ('candidates = the previous winner (re-scored from its saved outputs, not re-trained) + the new arms; winner = lowest mean source-balanced '
                 'validation EER (ties within 0.01 points -> higher AUC); test and FingerDOS are reported for every candidate but never used for the decision')

def validation_table(results, models, model_col='kernel', sets=VALIDATION_SETS, metrics_=('eer', 'roc_auc', 'f1_macro', 'fmr', 'fnmr')):
    """Selection table: one row per model, 'mean ± std' over repetitions for every validation set (balanced + DB1 / DB2 / DB1xDB2) and metric (%)."""
    pct = to_percent(results[results[model_col].isin(models) & results.set.isin(sets)]); rows = []
    for mod in models:
        r = {model_col: mod}
        for st in sets:
            for m in metrics_:
                v = pct[(pct[model_col] == mod) & (pct.set == st)][m]
                r[f"{st.replace('_validation', '').replace('MOLF validation (balanced)', 'Balanced')} {LABEL.get(m, m)}"] = f'{v.mean():.2f} ± {v.std(ddof=1):.2f}' if len(v) else ''
        rows.append(r)
    return pd.DataFrame(rows)

def decide(results, candidates, model_col='kernel'):
    """DECISION_RULE: (winner, ranking) on the balanced validation EER only."""
    v = results[results[model_col].isin(candidates) & (results.set == SELECTION_SET)].groupby(model_col).agg(eer=('eer', 'mean'), eer_std=('eer', 'std'), auc=('roc_auc', 'mean'), reps=('eer', 'size'))
    v = v.sort_values(['eer', 'auc'], ascending=[True, False]); best = v.iloc[0]
    tie = v[(v.eer - best.eer).abs() <= 1e-4]; win = tie.sort_values('auc', ascending=False).index[0]
    v = (100 * v[['eer', 'eer_std', 'auc']]).join(v[['reps']]); v['gap_to_best_eer'] = v.eer - v.eer.min(); return win, v.reset_index()

def reference_rows(label, kdir, arm=None, reps=None, threshold_from_run=True):
    """Re-evaluates another kernel's saved pair scores on the identical pairs with the paper threshold rule; [] if missing."""
    kdir = Path(kdir)
    if arm is None:
        if not (kdir / 'winner_manifest.json').is_file(): print(f'{label}: not found ({kdir.name}) -> skipped'); return []
        arm = json.loads((kdir / 'winner_manifest.json').read_text())['winning_arm']
    rows = []
    for r in reps:
        rd = kdir / 'runs' / r / arm
        if not (rd / 'pair_scores.npz').is_file(): print(f'{label}: {r} missing -> skipped'); continue
        S = load_scores(rd)
        if set(S) != set(PAIR_SETS): print(f'{label}: {r} lacks some pair sets -> skipped'); continue
        rows += evaluate(S, label, arm, r)[0]
    return rows

def run_curves(run_dirs, name):
    """[(label, score, frozen threshold)] for one set or pooled group, one entry per repetition."""
    out = []
    for rd in run_dirs:
        S = load_scores(rd); y, s = group_scores(S, name); out.append((y, s, load_threshold(rd)))
    return out

# ------------------------------------------------------------------ figures (300 dpi)
FIG_DPI = 300; COLORS = plt.get_cmap('tab10').colors
_save = L._save

def _rates(y, s):
    fpr, tpr, thr = roc_curve(y, s, drop_intermediate=False); fnr = 1 - tpr; j = int(np.nanargmin(np.abs(fpr - fnr))); return fpr, tpr, fnr, thr, j

def fig_roc(curves_by_model, title, path, log_x=False):
    """Mean ROC over repetitions (±1 std band), AUC mean ± std and EER point per model. curves_by_model: {label: [(y, s[, thr]), ...]}."""
    grid = np.logspace(-4, 0, 400) if log_x else np.linspace(0, 1, 501); fig, ax = plt.subplots(figsize=(6, 5))
    for k, (lab, runs) in enumerate(curves_by_model.items()):
        T, A, E = [], [], []
        for r in runs:
            y, s = r[0], r[1]; fpr, tpr, fnr, _, j = _rates(y, s); T.append(np.interp(grid, fpr, tpr)); A.append(100 * roc_auc_score(y, s)); E.append((fpr[j], tpr[j]))
        T = np.array(T); m, sd = T.mean(0), T.std(0); c = COLORS[k % 10]
        ax.plot(grid, m, color=c, lw=1.6, label=f'{lab}  AUC {np.mean(A):.2f} ± {np.std(A, ddof=1) if len(A) > 1 else 0:.2f}'); ax.fill_between(grid, m - sd, m + sd, color=c, alpha=.15, lw=0)
        e = np.mean(E, 0); ax.plot(e[0], e[1], 'o', color=c, ms=5, mec='k', mew=.6)
    if not log_x: ax.plot([0, 1], [0, 1], ':', color='grey', lw=1); ax.plot([0, 1], [1, 0], '--', color='lightgrey', lw=.8)
    ax.set(xscale='log' if log_x else 'linear', xlabel='False match rate (FMR)', ylabel='True match rate (1 − FNMR)', title=title)
    ax.legend(fontsize=7, loc='lower right'); ax.grid(alpha=.3); return _save(fig, path)

def fig_det(curves_by_model, title, path):
    """DET curve (FNMR vs FMR on normal-deviate axes), mean over repetitions; EER points on the diagonal."""
    grid = np.logspace(-4, -0.01, 300); fig, ax = plt.subplots(figsize=(6, 5)); pp = lambda v: norm.ppf(np.clip(v, 1e-5, 1 - 1e-5))
    for k, (lab, runs) in enumerate(curves_by_model.items()):
        F = []
        for r in runs:
            fpr, tpr, fnr, _, j = _rates(r[0], r[1]); o = np.argsort(fpr); F.append(np.interp(grid, fpr[o], fnr[o]))
        F = np.mean(F, 0); ax.plot(pp(grid), pp(F), color=COLORS[k % 10], lw=1.6, label=lab)
    ticks = np.array([1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.2, 0.4, 0.6]); ax.set_xticks(pp(ticks)); ax.set_xticklabels([f'{100 * t:g}' for t in ticks], fontsize=7)
    ax.set_yticks(pp(ticks)); ax.set_yticklabels([f'{100 * t:g}' for t in ticks], fontsize=7); lim = (pp(1e-4), pp(0.6)); ax.set(xlim=lim, ylim=lim)
    ax.plot(lim, lim, '--', color='lightgrey', lw=.8); ax.set(xlabel='FMR (%)', ylabel='FNMR (%)', title=f'DET — {title}'); ax.legend(fontsize=7); ax.grid(alpha=.3); return _save(fig, path)

def fig_eer_crossing(runs, title, path):
    """FMR(t) and FNMR(t) against the decision threshold for each repetition; marks the EER crossing of this set and the frozen validation-EER threshold."""
    fig, ax = plt.subplots(figsize=(6.4, 4.4)); txt = []
    for k, (y, s, thr) in enumerate(runs):
        fpr, tpr, fnr, t, j = _rates(y, s); ok = np.isfinite(t); a = .9 if k == 0 else .45
        ax.plot(t[ok], fpr[ok], color=COLORS[3], lw=1.3, alpha=a, label='FMR' if k == 0 else None); ax.plot(t[ok], fnr[ok], color=COLORS[0], lw=1.3, alpha=a, label='FNMR' if k == 0 else None)
        ax.plot(t[j], fpr[j], 'o', color='k', ms=4, label='EER point (this set)' if k == 0 else None)
        ax.axvline(thr, color=COLORS[2], ls='--', lw=1, alpha=a, label='frozen validation-EER threshold' if k == 0 else None)
        m = metrics(y, s, thr); txt.append(f'rep {k + 1}: EER {100 * (fpr[j] + fnr[j]) / 2:.2f}% | at frozen thr FMR {100 * m["fmr"]:.2f}%, FNMR {100 * m["fnmr"]:.2f}%')
    lo = min(np.nanmin(r[1]) for r in runs); hi = max(np.nanmax(r[1]) for r in runs); ax.set_xlim(lo, hi)
    ax.set(xlabel='Decision threshold (score)', ylabel='Error rate', title=title); ax.legend(fontsize=7, loc='center right'); ax.grid(alpha=.3)
    ax.text(0.01, 0.99, '\n'.join(txt), transform=ax.transAxes, va='top', fontsize=6.5, family='monospace', bbox=dict(fc='white', alpha=.8, lw=0)); return _save(fig, path)

def fig_confusion(runs, title, path):
    """Confusion matrix at the frozen threshold: mean count over repetitions and row percentages (mean ± std)."""
    C = []
    for y, s, thr in runs:
        m = metrics(y, s, thr); C.append([[m['tn'], m['fp']], [m['fn'], m['tp']]])
    C = np.array(C, float); R = C / C.sum(2, keepdims=True); mc, mr, sr = C.mean(0), 100 * R.mean(0), 100 * R.std(0, ddof=1) if len(C) > 1 else np.zeros((2, 2))
    fig, ax = plt.subplots(figsize=(4.6, 4)); ax.imshow(mr, cmap='Blues', vmin=0, vmax=100)
    for i in range(2):
        for j in range(2): ax.text(j, i, f'{mc[i, j]:.0f}\n{mr[i, j]:.1f} ± {sr[i, j]:.1f}%', ha='center', va='center', fontsize=9, color='white' if mr[i, j] > 60 else 'black')
    ax.set_xticks([0, 1]); ax.set_xticklabels(['predicted\nimpostor', 'predicted\ngenuine']); ax.set_yticks([0, 1]); ax.set_yticklabels(['impostor', 'genuine'])
    ax.set_title(f'{title}\n(frozen validation-EER threshold; mean of {len(C)} reps)', fontsize=9); return _save(fig, path)

def fig_scores(runs, title, path, bins=60):
    """Genuine and impostor score distributions (all repetitions) with the frozen thresholds."""
    fig, ax = plt.subplots(figsize=(6.4, 4)); y = np.concatenate([r[0] for r in runs]); s = np.concatenate([r[1] for r in runs]); b = np.linspace(np.nanmin(s), np.nanmax(s), bins)
    ax.hist(s[y == 0], b, density=True, alpha=.55, color=COLORS[3], label='impostor'); ax.hist(s[y == 1], b, density=True, alpha=.55, color=COLORS[0], label='genuine')
    for k, r in enumerate(runs): ax.axvline(r[2], color=COLORS[2], ls='--', lw=1, label='frozen threshold' if k == 0 else None)
    ax.set(xlabel='Score', ylabel='Density', title=title); ax.legend(fontsize=7); ax.grid(alpha=.3); return _save(fig, path)

def fig_set_panel(model, runs_by_set, path):
    """One figure per model: rows = sets, columns = ROC | FMR/FNMR vs threshold | confusion matrix | score distributions."""
    n = len(runs_by_set); fig, axs = plt.subplots(n, 4, figsize=(20, 3.9 * n), squeeze=False)
    for r_, (name, runs) in enumerate(runs_by_set.items()):
        ax = axs[r_, 0]; A = []
        for k, (y, s, thr) in enumerate(runs):
            fpr, tpr, fnr, t, j = _rates(y, s); ax.plot(fpr, tpr, color=COLORS[k % 10], lw=1.2, label=f'rep {k + 1}'); ax.plot(fpr[j], tpr[j], 'ko', ms=3); A.append(100 * roc_auc_score(y, s))
        ax.plot([0, 1], [0, 1], ':', color='grey', lw=1); ax.set(title=f'{name}: ROC (AUC {np.mean(A):.2f})', xlabel='FMR', ylabel='1 − FNMR'); ax.legend(fontsize=6); ax.grid(alpha=.3)
        ax = axs[r_, 1]
        for k, (y, s, thr) in enumerate(runs):
            fpr, tpr, fnr, t, j = _rates(y, s); ok = np.isfinite(t)
            ax.plot(t[ok], fpr[ok], color=COLORS[3], lw=1, alpha=.8, label='FMR' if k == 0 else None); ax.plot(t[ok], fnr[ok], color=COLORS[0], lw=1, alpha=.8, label='FNMR' if k == 0 else None)
            ax.plot(t[j], fpr[j], 'ko', ms=3); ax.axvline(thr, color=COLORS[2], ls='--', lw=.9, label='frozen threshold' if k == 0 else None)
        ax.set_xlim(min(np.nanmin(r[1]) for r in runs), max(np.nanmax(r[1]) for r in runs)); ax.set(title=f'{name}: EER crossing', xlabel='threshold', ylabel='error rate'); ax.legend(fontsize=6); ax.grid(alpha=.3)
        ax = axs[r_, 2]; C = []
        for y, s, thr in runs:
            m = metrics(y, s, thr); C.append([[m['tn'], m['fp']], [m['fn'], m['tp']]])
        C = np.array(C, float); R = 100 * (C / C.sum(2, keepdims=True)).mean(0); ax.imshow(R, cmap='Blues', vmin=0, vmax=100)
        for i in range(2):
            for j in range(2): ax.text(j, i, f'{C.mean(0)[i, j]:.0f}\n{R[i, j]:.1f}%', ha='center', va='center', fontsize=8, color='white' if R[i, j] > 60 else 'black')
        ax.set_xticks([0, 1]); ax.set_xticklabels(['pred. impostor', 'pred. genuine'], fontsize=7); ax.set_yticks([0, 1]); ax.set_yticklabels(['impostor', 'genuine'], fontsize=7); ax.set_title(f'{name}: confusion matrix')
        ax = axs[r_, 3]; y = np.concatenate([r[0] for r in runs]); s = np.concatenate([r[1] for r in runs]); b = np.linspace(np.nanmin(s), np.nanmax(s), 50)
        ax.hist(s[y == 0], b, density=True, alpha=.55, color=COLORS[3], label='impostor'); ax.hist(s[y == 1], b, density=True, alpha=.55, color=COLORS[0], label='genuine')
        for k, rr in enumerate(runs): ax.axvline(rr[2], color=COLORS[2], ls='--', lw=.9)
        ax.set(title=f'{name}: scores', xlabel='score'); ax.legend(fontsize=6); ax.grid(alpha=.3)
    fig.suptitle(model, fontsize=12); fig.tight_layout(); return _save(fig, path)

def fig_metric_bars(snum, models, path, sets=PAPER_SETS, metrics_=METRICS, model_col='kernel', title='Paper metrics (frozen validation-EER threshold, mean ± std over repetitions)'):
    nc = 4; nr = int(np.ceil(len(metrics_) / nc)); fig, axs = plt.subplots(nr, nc, figsize=(5.2 * nc, 3.9 * nr), squeeze=False); x = np.arange(len(sets)); w = 0.8 / max(len(models), 1)
    for ax, m in zip(axs.ravel(), metrics_):
        for k, mod in enumerate(models):
            q = snum[snum[model_col].astype(str) == str(mod)].set_index(snum[snum[model_col].astype(str) == str(mod)].set.astype(str)).reindex(list(sets))
            ax.bar(x + (k - (len(models) - 1) / 2) * w, q[f'{m}_mean'], w, yerr=q[f'{m}_std'].fillna(0), capsize=2, color=COLORS[k % 10], label=str(mod))
        ax.set(xticks=x, ylabel=f'{LABEL[m]} (%)'); ax.set_xticklabels([s.replace(' (pooled)', '\n(pooled)') for s in sets], fontsize=7); ax.grid(alpha=.3, axis='y')
    for ax in axs.ravel()[len(metrics_):]: ax.axis('off')
    axs.ravel()[0].legend(fontsize=6); fig.suptitle(title, fontsize=11); fig.tight_layout(); return _save(fig, path)

def fig_speed_accuracy(df, x, y, label, path, title, xlabel, ylabel, logx=True):
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for k, r in df.reset_index(drop=True).iterrows():
        ax.scatter(r[x], r[y], s=60, color=COLORS[k % 10], edgecolor='k', zorder=3); ax.annotate(str(r[label]), (r[x], r[y]), fontsize=7, xytext=(4, 3), textcoords='offset points')
    ax.set(xscale='log' if logx else 'linear', xlabel=xlabel, ylabel=ylabel, title=title); ax.grid(alpha=.3, which='both'); return _save(fig, path)

def fig_training_balanced(history, title, path):
    """Loss | train AUC (augmented batches) vs source-balanced validation AUC | balanced and per-source validation EER, best epoch and LR drops marked."""
    h = pd.DataFrame(history); fig, axs = plt.subplots(1, 3, figsize=(15, 3.8)); x = h['epoch']
    if 'loss' in h: axs[0].plot(x, h['loss'], color=COLORS[0], label='train loss')
    axs[0].set(xlabel='Epoch', ylabel='Loss'); axs[0].legend(fontsize=8); axs[0].grid(alpha=.3)
    for col, lab, c in (('verification_auc', 'train AUC (augmented batches)', COLORS[0]), ('val_balanced_auc', 'validation AUC (balanced)', COLORS[3])):
        if col in h: axs[1].plot(x, 100 * h[col], color=c, label=lab)
    axs[1].set(xlabel='Epoch', ylabel='AUC (%)'); axs[1].legend(fontsize=8); axs[1].grid(alpha=.3)
    if 'val_balanced_eer' in h:
        axs[2].plot(x, 100 * h['val_balanced_eer'], color='k', lw=2, label='balanced (stopping)')
        for k, n in enumerate(VAL_SOURCES):
            if f'val_eer_{n}' in h: axs[2].plot(x, 100 * h[f'val_eer_{n}'], color=COLORS[k + 1], lw=1, alpha=.8, label=n.replace('_validation', ''))
        b, _ = BalancedEERStopping.replay(h, len(h) + 1); axs[2].axvline(x.iloc[b], color='g', ls='--', lw=1, label=f'best epoch {int(x.iloc[b])}')
        if 'lr' in h:
            for e in x[h['lr'].diff().fillna(0) < 0]: axs[2].axvline(e, color='grey', ls=':', lw=.8)
    axs[2].set(xlabel='Epoch', ylabel='validation EER (%)'); axs[2].legend(fontsize=7); axs[2].grid(alpha=.3)
    fig.suptitle(title, fontsize=10); return _save(fig, path)

def standard_figures(fig_dir, model_runs, sets=PAPER_SETS):
    """model_runs: {model label: [run_dir per repetition]} -> per set: mean ROC, DET, and per model: EER crossing, confusion matrix, scores, set panel."""
    fig_dir = Path(fig_dir); out = []
    for st in sets:
        tag = st.replace(' ', '_').replace('(', '').replace(')', '')
        curves = {lab: run_curves(dirs, st) for lab, dirs in model_runs.items()}
        out += [fig_roc(curves, f'ROC — {st}', fig_dir / f'roc_{tag}.png'), fig_roc(curves, f'ROC (log FMR) — {st}', fig_dir / f'roc_log_{tag}.png', log_x=True),
                fig_det(curves, st, fig_dir / f'det_{tag}.png')]
        for lab, runs in curves.items():
            mt = ''.join(ch if ch.isalnum() else '_' for ch in lab)[:60]
            out += [fig_eer_crossing(runs, f'{lab} — {st}', fig_dir / 'eer_crossing' / f'{mt}__{tag}.png'), fig_confusion(runs, f'{lab} — {st}', fig_dir / 'confusion' / f'{mt}__{tag}.png'),
                    fig_scores(runs, f'{lab} — {st}', fig_dir / 'scores' / f'{mt}__{tag}.png')]
    for lab, dirs in model_runs.items():
        mt = ''.join(ch if ch.isalnum() else '_' for ch in lab)[:60]
        out.append(fig_set_panel(lab, {st: run_curves(dirs, st) for st in sets}, fig_dir / 'panels' / f'{mt}.png'))
    return out
