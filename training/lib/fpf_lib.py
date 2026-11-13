"""fpf_lib — shared code for the FPF (FingerPrint-Final) kernels: MOLF DB1 training, MOLF DB2 adaptation / sensor transfer, FingerDOS external test.

Every FPF kernel copies this file from <FingerPrint-Final>/output/00_code/ and imports it, so all kernels use the same paths,
protocol loading, metrics, thresholds, figures, workbook format and run log. TensorFlow is NOT imported here (see fpf_tf.py)."""
import hashlib, json, math, os, platform, re, shutil, subprocess, time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

# ============================================================ workspace
DRIVE = Path('/content/drive/MyDrive')
FINAL_FOLDER_NAME = 'FingerPrint-Final'                       # Datasets/ (raw archives, read-only) and output/ (everything generated)
ARCHIVES = {'MOLF': 'MOLF_DB1_and_2.7z', 'FingerDOS': 'FingerDOS_v1.rar'}
DATASETS = ('MOLF_DB1', 'MOLF_DB2', 'FingerDOS')
SENSORS = {'MOLF_DB1': 'Lumidigm Venus IP65 (optical, multispectral)', 'MOLF_DB2': 'SecuGen Hamster IV (optical)', 'FingerDOS': 'SecuGen iD-USB SC (optical)'}
EXPECTED_SAMPLES = {'MOLF_DB1': 4000, 'MOLF_DB2': 4000, 'FingerDOS': 3600}
KERNELS = {'FPP02': 'FPP02_FingerJetFX_Minutiae_and_Image_Caches',                 # FPF00's output folder after the FPP reorganisation (same content)
           'FPF00': 'FPF00_MOLF_FingerDOS_Data_Prep_FingerJetFX_Minutiae_Subject_Splits_Pairs',
           'FPF01': 'FPF01_MobileNetV3Large_Siamese_FullFT_MildAug_DB1Train_FMR1pctThreshold_DB2MeanCentering_FingerDOSTest'}

def find_folder(name, root=DRIVE, max_depth=4):
    """Finds a Drive folder by exact name (breadth-first, depth-limited)."""
    frontier = [Path(root)]
    for _ in range(max_depth + 1):
        nxt = []
        for d in frontier:
            try: children = [c for c in d.iterdir() if c.is_dir()]
            except (PermissionError, OSError): continue
            for c in children:
                if c.name == name: return c
                nxt.append(c)
        frontier = nxt
    return None

def workspace():
    root = find_folder(FINAL_FOLDER_NAME)
    if root is None: raise FileNotFoundError(f"'{FINAL_FOLDER_NAME}' not found under {DRIVE} (add a shortcut to My Drive if it was shared with you)")
    out = root / 'output'
    prep = next((out / KERNELS[k] for k in ('FPP02', 'FPF00') if (out / KERNELS[k]).is_dir()), out / KERNELS['FPP02']); data = prep / 'data'   # data folder: new name first, old name still accepted
    ws = {'root': root, 'datasets': root / 'Datasets', 'output': out, 'code': out / '00_code', 'prep': prep, 'data': data,
          'index': data / 'index', 'originals': data / 'originals', 'images': data / 'images', 'templates': data / 'fingerjetfx_templates', 'protocol': prep / 'protocol'}
    for k in ('output', 'code'): ws[k].mkdir(parents=True, exist_ok=True)
    return ws

def kernel_dir(ws, key):
    d = ws['output'] / KERNELS[key]; (d / 'figures').mkdir(parents=True, exist_ok=True); (d / 'runs').mkdir(parents=True, exist_ok=True); return d

def copy_verified(src, dst):
    src, dst = Path(src), Path(dst); dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_file() and dst.stat().st_size == src.stat().st_size: return dst
    tmp = dst.with_name(dst.name + '.partial'); shutil.copy2(src, tmp); os.replace(tmp, dst)
    if dst.stat().st_size != src.stat().st_size: raise IOError(f'copy size mismatch: {dst}')
    return dst

def local_copy(path, local_dir='/content/fpf_local'):
    path = Path(path); local = Path(local_dir) / path.name; local.parent.mkdir(parents=True, exist_ok=True)
    if not local.is_file() or local.stat().st_size != path.stat().st_size: shutil.copy2(path, local)
    return local

def sha256_of(path, block=1 << 24):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(block), b''): h.update(b)
    return h.hexdigest()

def timer():
    t0 = time.perf_counter(); return lambda: time.perf_counter() - t0

def rel(ws, path): return str(Path(path).relative_to(ws['output']))

# ============================================================ protocol
PARTITIONS = ('train', 'validation', 'adaptation', 'test')
PAIR_SETS = {   # name: (dataset of image a, dataset of image b, role)
    'DB1_train':      ('MOLF_DB1', 'MOLF_DB1', 'training pairs; clean scores give the thresholds (FMR<=1% and EER)'),
    'DB1_validation': ('MOLF_DB1', 'MOLF_DB1', 'early stopping and model selection'),
    'DB1_test':       ('MOLF_DB1', 'MOLF_DB1', 'held-out source-sensor test'),
    'DB2_test':       ('MOLF_DB2', 'MOLF_DB2', 'held-out test on the second sensor (same 20 test subjects)'),
    'DB1xDB2_test':   ('MOLF_DB1', 'MOLF_DB2', 'cross-sensor test: DB1 image vs DB2 image of the test subjects'),
    'FingerDOS_test': ('FingerDOS', 'FingerDOS', 'independent external test (all 60 subjects)')}
TEST_SETS = ('DB1_test', 'DB2_test', 'DB1xDB2_test', 'FingerDOS_test')

def repetitions(ws):
    reps = sorted(p.name for p in ws['protocol'].glob('rep_*_seed_*') if p.is_dir())
    if not reps: raise FileNotFoundError(f"No repetitions in {ws['protocol']}; run FPF00 (now FPP02) first.")
    return reps

def rep_seed(rep): return int(rep.split('_seed_')[1])

def load_npz(path):
    with np.load(path, allow_pickle=False) as z: return {k: z[k] for k in z.files}

def load_protocol(ws, rep):
    d = ws['protocol'] / rep; man = json.loads((d / 'protocol_manifest.json').read_text())
    if int(man['repetition_seed']) != rep_seed(rep): raise RuntimeError(f'{rep}: seed mismatch in protocol manifest')
    pairs = {s: load_npz(d / 'pairs' / f'{s}_pairs.npz') for s in PAIR_SETS}
    splits = pd.read_csv(d / 'MOLF_subject_splits.csv')
    adapt = splits[splits.partition == 'adaptation'].sort_values('adaptation_rank').subject_id.astype(str).tolist()
    samples = {ds: pd.read_csv(d / f'{ds}_sample_splits.csv') for ds in DATASETS}
    return {'rep': rep, 'seed': rep_seed(rep), 'pairs': pairs, 'splits': splits, 'adaptation_subjects': adapt, 'samples': samples, 'manifest': man}

def sample_indices(proto, dataset, partition, subjects=None):
    s = proto['samples'][dataset]; m = s.partition == partition
    if subjects is not None: m &= s.subject_id.astype(str).isin([str(x) for x in subjects])
    return s.loc[m, 'sample_index'].to_numpy(np.int64)

# ============================================================ metrics and thresholds
METRIC_COLUMNS = ['roc_auc', 'eer', 'tar_at_fmr_1pct', 'tar_at_fmr_0_1pct', 'fmr', 'fnmr', 'f1_macro', 'balanced_accuracy', 'accuracy', 'f1_genuine', 'precision', 'recall_tar', 'specificity_tnr']
PERCENT_COLUMNS = list(METRIC_COLUMNS)

def eer_threshold(y, score):
    fpr, tpr, thr = roc_curve(y, score); i = int(np.nanargmin(np.abs(fpr - (1 - tpr)))); return float(thr[i])

def fmr_threshold(y, score, target=0.01):
    """Smallest threshold whose false-match rate on these pairs is <= target (accept if score >= threshold)."""
    imp = np.sort(np.asarray(score, np.float64)[np.asarray(y) == 0]); n = len(imp); k = int(math.floor(target * n))
    if n == 0: raise ValueError('no impostor pairs')
    t = np.nextafter(imp[n - k - 1], np.inf)                 # at most k impostors have score >= t (fewer with ties)
    return float(t)

def verification_metrics(y, score, threshold):
    y = np.asarray(y, int); score = np.asarray(score, float); pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    fpr, tpr, thr = roc_curve(y, score); fnr = 1 - tpr; j = int(np.nanargmin(np.abs(fpr - fnr)))
    def tar_at(target):
        ok = np.flatnonzero(fpr <= target); return float(tpr[ok[-1]]) if len(ok) else 0.0
    f1g = 2 * tp / max(2 * tp + fp + fn, 1); f1i = 2 * tn / max(2 * tn + fp + fn, 1)
    return {'threshold': float(threshold), 'roc_auc': float(roc_auc_score(y, score)), 'eer': float((fpr[j] + fnr[j]) / 2), 'eer_threshold_on_this_set': float(thr[j]),
            'tar_at_fmr_1pct': tar_at(1e-2), 'tar_at_fmr_0_1pct': tar_at(1e-3), 'fmr': fp / max(tn + fp, 1), 'fnmr': fn / max(tp + fn, 1),
            'f1_macro': 0.5 * (f1g + f1i), 'f1_genuine': f1g, 'balanced_accuracy': 0.5 * (tp / max(tp + fn, 1) + tn / max(tn + fp, 1)),
            'accuracy': (tp + tn) / max(len(y), 1), 'precision': tp / max(tp + fp, 1), 'recall_tar': tp / max(tp + fn, 1), 'specificity_tnr': tn / max(tn + fp, 1),
            'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'pairs': int(len(y)), 'genuine_pairs': int(y.sum()), 'impostor_pairs': int((1 - y).sum())}

def l2n(E): return E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
def center(E, mu): return l2n(np.asarray(E, np.float32) - np.asarray(mu, np.float32)[None, :])

def pair_scores(Ea, Eb, pairs):
    return np.sum(Ea[np.asarray(pairs['index_a'])] * Eb[np.asarray(pairs['index_b'])], axis=1)

def to_percent(frame):
    f = frame.copy()
    for c in PERCENT_COLUMNS:
        if c in f.columns: f[c] = 100 * f[c]
    return f

def summarize(results, keys):
    """Mean and std over repetitions (percent) for every group in `keys`, plus 'mean ± std' text columns."""
    pct = to_percent(results); cols = [c for c in METRIC_COLUMNS if c in pct.columns]
    g = pct.groupby(list(keys))[cols]; mean, std, n = g.mean(), g.std(ddof=1), g.size().rename('repetitions')
    out = mean.add_suffix('_mean').join(std.add_suffix('_std')).join(n).reset_index()
    for c in cols: out[f'{c} (mean ± std)'] = [f'{m:.2f} ± {0 if np.isnan(s) else s:.2f}' for m, s in zip(out[f'{c}_mean'], out[f'{c}_std'])]
    return out

# ============================================================ figures (300 dpi PNG)
FIG_DPI = 300
COLORS = plt.get_cmap('tab10').colors

def _save(fig, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(path, dpi=FIG_DPI, bbox_inches='tight'); plt.close(fig); return path

def fig_roc_det(curves, title, out_dir):
    """curves: {label: (y, score)} -> roc.png and det.png."""
    out_dir = Path(out_dir); paths = []
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    for k, (lab, (y, sc)) in enumerate(curves.items()):
        fpr, tpr, _ = roc_curve(y, sc); ax.plot(fpr, tpr, color=COLORS[k % 10], lw=1.5, label=f'{lab} (AUC {100 * roc_auc_score(y, sc):.2f})')
    ax.plot([0, 1], [0, 1], ':', color='grey', lw=1); ax.set(xlabel='False match rate (FMR)', ylabel='True accept rate (TAR)', title=f'ROC — {title}'); ax.legend(fontsize=7, loc='lower right'); ax.grid(alpha=.3)
    paths.append(_save(fig, out_dir / 'roc.png'))
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    for k, (lab, (y, sc)) in enumerate(curves.items()):
        fpr, tpr, _ = roc_curve(y, sc); m = (fpr > 0) & (tpr < 1); ax.plot(fpr[m], 1 - tpr[m], color=COLORS[k % 10], lw=1.5, label=lab)
    ax.axvline(0.01, color='grey', ls='--', lw=1); ax.set(xscale='log', yscale='log', xlabel='FMR', ylabel='FNMR', title=f'DET — {title}'); ax.legend(fontsize=7); ax.grid(alpha=.3, which='both')
    paths.append(_save(fig, out_dir / 'det.png')); return paths

def fig_score_distributions(panels, thresholds, title, path):
    """panels: {label: (y, score)}; thresholds: {name: value} drawn on every panel (frozen training thresholds)."""
    n = len(panels); fig, axs = plt.subplots(1, n, figsize=(4.3 * n, 3.6), squeeze=False)
    for a, (lab, (y, sc)) in zip(axs[0], panels.items()):
        bins = np.linspace(np.min(sc), np.max(sc), 60)
        a.hist(sc[y == 0], bins=bins, alpha=.6, color=COLORS[3], density=True, label='impostor'); a.hist(sc[y == 1], bins=bins, alpha=.6, color=COLORS[0], density=True, label='genuine')
        for k, (tn, tv) in enumerate(thresholds.items()): a.axvline(tv, color=['black', 'grey'][k % 2], ls=['--', ':'][k % 2], lw=1, label=tn)
        a.set(title=lab, xlabel='Cosine score'); a.legend(fontsize=6)
    fig.suptitle(title, fontsize=10); return _save(fig, path)

def fig_training(history, title, path, train_col='verification_auc', val_col='val_verification_auc'):
    h = pd.DataFrame(history); fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    if 'loss' in h: axs[0].plot(h.epoch, h['loss'], color=COLORS[0], label='train loss')
    if 'val_loss' in h: axs[0].plot(h.epoch, h['val_loss'], color=COLORS[3], label='validation loss')
    axs[0].set(xlabel='Epoch', ylabel='Loss'); axs[0].legend(fontsize=8); axs[0].grid(alpha=.3)
    for col, lab, c in ((train_col, 'train AUC', COLORS[0]), (val_col, 'validation AUC', COLORS[3])):
        if col in h: axs[1].plot(h.epoch, 100 * h[col], color=c, label=lab)
    axs[1].set(xlabel='Epoch', ylabel='AUC (%)'); axs[1].legend(fontsize=8); axs[1].grid(alpha=.3)
    fig.suptitle(title, fontsize=10); return _save(fig, path)

def fig_mean_roc(curves_by_arm, title, path, grid=np.linspace(0, 1, 501)):
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    for k, (arm, curves) in enumerate(curves_by_arm.items()):
        tprs, aucs = [], []
        for y, sc in curves:
            fpr, tpr, _ = roc_curve(y, sc); tprs.append(np.interp(grid, fpr, tpr)); aucs.append(roc_auc_score(y, sc))
        m, s = np.mean(tprs, 0), np.std(tprs, 0); c = COLORS[k % 10]
        ax.plot(grid, m, color=c, lw=1.5, label=f'{arm} ({100 * np.mean(aucs):.2f}±{100 * np.std(aucs):.2f})'); ax.fill_between(grid, np.clip(m - s, 0, 1), np.clip(m + s, 0, 1), color=c, alpha=.15)
    ax.plot([0, 1], [0, 1], ':', color='grey', lw=1); ax.set(xlabel='FMR', ylabel='TAR', title=title); ax.legend(fontsize=7, loc='lower right'); ax.grid(alpha=.3)
    return _save(fig, path)

# ============================================================ workbook, run log, environment
def env_info():
    info = {'python': platform.python_version(), 'platform': platform.platform(), 'cpu_count': os.cpu_count()}
    try: info['cpu_model'] = next((l.split(':', 1)[1].strip() for l in open('/proc/cpuinfo') if l.startswith('model name')), '')
    except OSError: pass
    try: info['ram_gb'] = round(os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 1e9, 1)
    except (ValueError, OSError): pass
    try: info['gpu'] = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], capture_output=True, text=True, timeout=10).stdout.strip() or 'none'
    except Exception: info['gpu'] = 'none'
    for mod in ('numpy', 'pandas', 'sklearn', 'tensorflow', 'keras', 'h5py', 'cv2'):
        try: info[mod] = __import__(mod).__version__
        except Exception: pass
    return info

class RunLog:
    def __init__(self, path): self.path = Path(path); self.rows = pd.read_csv(self.path).to_dict('records') if self.path.is_file() else []
    def add(self, **row):
        row = {'time_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'), **row}; self.rows.append(row); pd.DataFrame(self.rows).to_csv(self.path, index=False)
        print('[log]', {k: v for k, v in row.items() if k != 'time_utc'})
    def frame(self): return pd.DataFrame(self.rows)

def write_workbook(path, sheets):
    path = Path(path); tmp = path.with_name(path.stem + '.partial.xlsx')
    with pd.ExcelWriter(tmp, engine='openpyxl') as w:
        for name, df in sheets.items():
            if df is None: continue
            if isinstance(df, dict): df = pd.DataFrame([(k, json.dumps(v, default=str) if isinstance(v, (dict, list, tuple)) else v) for k, v in df.items()], columns=['parameter', 'value'])
            df.to_excel(w, sheet_name=str(name)[:31], index=False)
    os.replace(tmp, path); print('Workbook:', path); return path

def figure_index(kernel_root):
    return pd.DataFrame([{'figure': str(p.relative_to(kernel_root)), 'bytes': p.stat().st_size} for p in sorted(Path(kernel_root).rglob('*.png'))])

def run_dir(kdir, rep, arm):
    d = Path(kdir) / 'runs' / rep / arm; d.mkdir(parents=True, exist_ok=True); return d

def run_complete(d): return (Path(d) / 'run_complete.json').is_file()

def mark_complete(d, info):
    (Path(d) / 'run_complete.json').write_text(json.dumps({**info, 'completed_utc': datetime.now(timezone.utc).isoformat(timespec='seconds')}, indent=2, default=str))

def write_winner(kdir, kernel, arm, config, per_rep, selection_table, note=''):
    man = {'kernel': kernel, 'winning_arm': arm, 'config': config, 'selection': 'highest mean MOLF DB1 validation AUC over repetitions (secondary: FMR, FNMR, F1-macro at the training thresholds)',
           'selection_table': selection_table.to_dict('records') if isinstance(selection_table, pd.DataFrame) else selection_table, 'note': note, 'repetitions': per_rep,
           'written_utc': datetime.now(timezone.utc).isoformat(timespec='seconds')}
    (Path(kdir) / 'winner_manifest.json').write_text(json.dumps(man, indent=2, default=str)); print('Winner:', arm); return man
