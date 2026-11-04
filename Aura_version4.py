# ============================================================================
# Aura_version4.py
# Aura: Calibration-Free EMG Challenge (Kaggle / PiEEG) -- version 4
# ----------------------------------------------------------------------------
# Built from the Aura_benchmark.py leaderboard (out-of-fold macro-F1 on
# held-out people, 28 sessions, same folds as v3):
#     ens_top3 (LightGBM x2 + Riemannian TS-LR)   0.512
#     full_lgbm_sessz                               0.491
#     full_lgbm                                     0.468
#     riemann_ts_lr (+ prior matching)              0.448
#     riemann_fbts_lr_rc (+ prior matching)         0.428
#     cnn_ea / cnn_rotaug                           0.382 / 0.369
#     cnn_raw                                       0.305
#     Hudgins TD4 / TD-PSD + LDA/SVM                0.23 - 0.31
#
# So version 4 keeps only what won and combines it:
#   M1 lgbm_full_sz   our ~300 features + per-session z-scored copies (2 seeds)
#   M2 lgbm_plus      M1 + Riemannian tangent features (re-centered, filter-bank)
#   M3 lgbm_plus_et   M2 with extra-trees / stronger regularisation (diversity)
#   M4 riemann_ts_lr  covariance tangent space + logistic regression
#   M5 riemann_fbts_lr_rc  filter-bank, per-person re-centered
#   M6 cnn_ea_rot     compact CNN on Euclidean-aligned EMG + rotation augment.
#      (the two tricks that helped the CNNs in the benchmark, combined)
#   Riemannian members also enter the blend in a per-person prior-matched form
#   (that is what made riemann_ts_lr strong).
#
#   BLEND: non-negative weights searched for macro-F1 on the out-of-fold
#   predictions, then optional per-class score offsets (macro-F1 tuning) and
#   optional per-person post-processing (prior matching / v3 transductive
#   refinement). Every choice is scored with NESTED folds (fit on 4 folds'
#   held-out people, score on the 5th) so the reported number is honest.
#
# PhysioRobustNet (thesis model, v1-v3) scored ~0.43 per fold in v3, below
# LightGBM, so it is not part of v4.
#
# HOW TO RUN ON KAGGLE: competition data attached, GPU T4 (one is enough),
# paste into one cell. Writes /kaggle/working/submission.csv and reports in
# /kaggle/working/aura_v4/. AURA_QUICK=1 = tiny smoke run.
# Leaderboard-only note: prior matching / transductive steps use all windows
# of a test file together (unlabeled); for a live demo set POSTPROC='none'.
# ============================================================================


# ============================================================================
# CELL 0 - Setup
# ============================================================================

import os
import re
import sys
import gc
import copy
import json
import math
import time
import random
import warnings
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

for _imp, _pip in [('openpyxl', 'openpyxl'), ('pywt', 'PyWavelets'),
                   ('lightgbm', 'lightgbm')]:
    try:
        __import__(_imp)
    except ImportError:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', _pip], check=False)

import pywt
from scipy.signal import (butter, sosfiltfilt, iirnotch, filtfilt, welch,
                          hilbert, detrend)
from scipy.ndimage import uniform_filter1d
from scipy.stats import skew, kurtosis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (f1_score, accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.utils.class_weight import compute_class_weight
from joblib import Parallel, delayed
import lightgbm as lgb

import torch
import torch.nn as nn
import torch.nn.functional as F

QUICK = os.environ.get('AURA_QUICK', '0') == '1'

CFG = {
    'SEED': 42,
    # windows / signal -- identical to Aura_version3.py and the benchmark
    'FS': 200, 'WIN_SEC': 1.4, 'CROP_STRIDE_SEC': 0.10,
    'TRIM_GESTURE_START_SEC': 0.4, 'MIN_SEG_SEC': 1.20,
    'BANDPASS': (20.0, 80.0), 'NOTCH_HZ': (50.0, 60.0), 'NOTCH_Q': 30.0,
    'NORM_MODE': 'session', 'USE_IMU': True, 'CLIP_SIGNAL': 20.0,
    # validation -- identical folds to v3 / benchmark
    'N_FOLDS': 5, 'INNER_VAL_FRAC': 0.15,

    'MODELS': ['lgbm_full_sz', 'lgbm_plus', 'lgbm_plus_et',
               'riemann_ts_lr', 'riemann_fbts_lr_rc', 'cnn_ea_rot'],
    'LGBM_SEEDS': 2,
    'CNN_SEEDS': 3,
    'COV_SHRINK': 1e-3,
    'FB_BANDS': ((20.0, 40.0), (40.0, 80.0)),

    # CNN (benchmark settings)
    'CNN_EPOCHS': 40, 'CNN_PATIENCE': 8, 'CNN_BATCH': 256, 'CNN_LR': 1e-3,
    'CNN_WD': 1e-2, 'CNN_WIDTH': 64, 'CNN_DROPOUT': 0.3, 'LABEL_SMOOTHING': 0.1,
    'AUG_NOISE_STD': 0.05, 'AUG_CH_GAIN': 0.3, 'AUG_GLOBAL_GAIN': 0.3,

    # LightGBM (benchmark settings) + an extra-trees variant for diversity
    'LGBM_PARAMS': {
        'n_estimators': 3000, 'learning_rate': 0.03, 'num_leaves': 31,
        'min_child_samples': 20, 'subsample': 0.8, 'subsample_freq': 1,
        'colsample_bytree': 0.6, 'reg_lambda': 1.0,
        'class_weight': 'balanced', 'verbose': -1,
    },
    'LGBM_ET_PARAMS': {'extra_trees': True, 'colsample_bytree': 0.3,
                       'min_child_samples': 40, 'reg_lambda': 5.0},
    'LGBM_EARLY_STOP': 150,

    # blending / post-processing
    'BLEND_SEARCH_ITERS': 3000,       # Dirichlet weight candidates
    'CLASS_BIAS': 'auto',             # 'auto' | 'on' | 'off'
    'POSTPROC': 'auto',               # 'auto' | 'none' | 'prior' | 'transductive'
    'TD_ALPHAS': (0.8, 0.5, 0.2),
    'TD_TAUS': (0.05, 0.1, 0.2, 0.4),
    'TD_ITERS': 15,
}
if QUICK:
    CFG.update({'CNN_EPOCHS': 2, 'CNN_PATIENCE': 1, 'CROP_STRIDE_SEC': 0.3,
                'LGBM_SEEDS': 1, 'CNN_SEEDS': 1, 'BLEND_SEARCH_ITERS': 200})
    CFG['LGBM_PARAMS'] = dict(CFG['LGBM_PARAMS'], n_estimators=100)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything(CFG['SEED'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True
print('Device:', device)

N_JOBS = max(1, min(4, os.cpu_count() or 1))
EMG_CH = [f'ch{i}' for i in range(1, 9)]
IMU_CH = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
SIG_CH = EMG_CH + IMU_CH
N_EMG = len(EMG_CH)
FS = CFG['FS']
WIN = int(round(CFG['WIN_SEC'] * FS))
CLASS_ORDER = ['rest', 'fist', 'open', 'point', 'pinch', 'thumb-up']
MODEL_CH = list(range(len(SIG_CH))) if CFG['USE_IMU'] else list(range(N_EMG))


def find_data_root():
    cands = []
    if os.environ.get('AURA_DATA_DIR'):
        cands.append(Path(os.environ['AURA_DATA_DIR']))
    cands.append(Path('/kaggle/input/vr-calibration-free-emg-challenge-pieeg'))
    for c in cands:
        if c.exists() and any(c.rglob('train_*.xlsx')):
            return c
    for base in (Path('/kaggle/input'), Path('.')):
        if base.exists():
            for hit in sorted(base.rglob('train_*.xlsx')):
                return hit.parent.parent if hit.parent.name == 'train' else hit.parent
    raise FileNotFoundError('Competition data not found. Attach the dataset '
                            'or set AURA_DATA_DIR.')


DATA_ROOT = find_data_root()
if os.environ.get('AURA_OUT_DIR'):
    OUT_DIR = Path(os.environ['AURA_OUT_DIR'])
elif Path('/kaggle/working').exists():
    OUT_DIR = Path('/kaggle/working/aura_v4')
else:
    OUT_DIR = Path('aura_v4')
CACHE_DIR = OUT_DIR / 'cache'
for _d in (OUT_DIR, CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)
print('Data root :', DATA_ROOT)
print('Output dir:', OUT_DIR)


def natural_key(p):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', Path(p).stem)]


# ============================================================================
# CELL 1 - Load the Excel files (cached as pickle after the first read)
# ============================================================================

TRAIN_COLS = {'t', 'emg_new', 'imu_new', 'label', 'label_source', *SIG_CH}
TEST_COLS = {'action_id', 't', 'emg_new', 'imu_new', *SIG_CH}


def load_table(path, wanted):
    cache = CACHE_DIR / (Path(path).stem + '.pkl')
    if cache.exists():
        return pd.read_pickle(cache)
    df = pd.read_excel(path, sheet_name=0, engine='openpyxl',
                       usecols=lambda c: str(c).strip() in wanted)
    df.columns = [str(c).strip() for c in df.columns]
    df.to_pickle(cache)
    return df


# ============================================================================
# CELL 2 - Preprocessing: packets -> uniform grid -> windows -> filters
# ----------------------------------------------------------------------------
# Rows arrive in packets that share one timestamp, so the rows of a packet are
# spread evenly up to the next timestamp before linear interpolation onto a
# fixed 200 Hz grid. EMG uses rows with emg_new == 1, IMU rows with
# imu_new == 1 (IMU values are held between updates otherwise).
# ============================================================================

def dejitter_time(t):
    t = np.asarray(t, dtype=np.float64)
    n = len(t)
    if n < 2:
        return t.copy()
    change = np.flatnonzero(np.diff(t) != 0) + 1
    starts = np.r_[0, change]
    ends = np.r_[change, n]
    counts = ends - starts
    period = (t[-1] - t[0]) / max(n - 1, 1)
    if not np.isfinite(period) or period <= 0:
        period = 1.0 / FS
    nxt = np.r_[t[starts[1:]], t[starts[-1]] + counts[-1] * period]
    span = nxt - t[starts]
    grp = np.repeat(np.arange(len(starts)), counts)
    rank = np.arange(n) - starts[grp]
    return t[starts][grp] + span[grp] * rank / counts[grp]


def resample_to_grid(t, values, grid):
    # (n, C) samples at times t -> (C, len(grid)); NaN outside the recording
    out = np.full((values.shape[1], len(grid)), np.nan, dtype=np.float64)
    if len(t) < 2:
        return out
    inside = (grid >= t[0] - 0.5 / FS) & (grid <= t[-1] + 0.5 / FS)
    for c in range(values.shape[1]):
        out[c, inside] = np.interp(grid[inside], t, values[:, c])
    return out


def streams_from_rows(df):
    df = df.sort_values('t', kind='mergesort')
    t = df['t'].to_numpy(np.float64)
    sig = (df[SIG_CH].apply(pd.to_numeric, errors='coerce')
           .ffill().bfill().fillna(0.0).to_numpy(np.float64))
    n = len(df)
    em = (df['emg_new'].to_numpy() == 1) if 'emg_new' in df else np.ones(n, bool)
    im = (df['imu_new'].to_numpy() == 1) if 'imu_new' in df else np.ones(n, bool)
    if em.sum() < 2:
        em = np.ones(n, bool)
    if im.sum() < 2:
        im = np.ones(n, bool)
    return (dejitter_time(t[em]), sig[em, :N_EMG],
            dejitter_time(t[im]), sig[im, N_EMG:])


def to_grid(df, grid):
    te, emg, ti, imu = streams_from_rows(df)
    return np.concatenate([resample_to_grid(te, emg, grid),
                           resample_to_grid(ti, imu, grid)], 0)


def guided_segments(df):
    # contiguous runs of the same label with label_source == 'guided'
    df = df.sort_values('t', kind='mergesort').reset_index(drop=True)
    src = df['label_source'].astype(str).str.strip().str.lower()
    lab = df['label'].astype(str).str.strip()
    guided = (src == 'guided') & df['label'].notna()
    key = np.where(guided.to_numpy(), lab.to_numpy(), '__none__')
    run = np.r_[0, np.cumsum(key[1:] != key[:-1])]
    t = df['t'].to_numpy(np.float64)
    segs = []
    for r in np.unique(run[guided.to_numpy()]):
        idx = np.flatnonzero(run == r)
        segs.append((key[idx[0]], t[idx[0]], t[idx[-1]]))
    return segs


def process_train_file(path):
    name = Path(path).stem
    df = load_table(path, TRAIN_COLS)
    t = np.sort(df['t'].to_numpy(np.float64))
    grid = np.arange(t[0], t[-1] + 1e-9, 1.0 / FS)
    X = to_grid(df, grid)
    stride = max(1, int(round(CFG['CROP_STRIDE_SEC'] * FS)))
    min_len = int(round(CFG['MIN_SEG_SEC'] * FS))
    Xs, ys, refs, segs = [], [], [], []
    for k, (label, ts, te) in enumerate(guided_segments(df)):
        i0 = int(np.searchsorted(grid, ts, 'left'))
        i1 = int(np.searchsorted(grid, te, 'right'))
        seg = X[:, i0:i1]
        L = seg.shape[1]
        if L < min_len:
            continue
        if L >= WIN:
            # v3: muscle activity starts ~0.4 s after a gesture prompt, so
            # crops that begin before that are mostly rest -> skip them
            first = 0
            if label != 'rest':
                trim = int(round(CFG['TRIM_GESTURE_START_SEC'] * FS))
                if L - WIN >= trim:
                    first = trim
            starts = list(range(first, L - WIN + 1, stride))
            if starts[-1] != L - WIN:
                starts.append(L - WIN)
            centre = (L - WIN) // 2
            ref_start = min(starts, key=lambda s: abs(s - centre))
            for s in starts:
                Xs.append(seg[:, s:s + WIN])
                ys.append(label)
                refs.append(s == ref_start)
                segs.append(k)
        else:
            w = np.full((X.shape[0], WIN), np.nan)
            w[:, :L] = seg
            Xs.append(w)
            ys.append(label)
            refs.append(True)
            segs.append(k)
    if not Xs:
        return None
    return {'name': name, 'X': np.stack(Xs).astype(np.float32),
            'y': np.array(ys), 'ref': np.array(refs, bool),
            'seg': np.array(segs, np.int64)}


def process_test_file(path):
    name = Path(path).stem
    df = load_table(path, TEST_COLS)
    ids, Xs = [], []
    for aid, g in df.groupby('action_id', sort=False):
        t0 = float(np.nanmin(g['t'].to_numpy(np.float64)))
        grid = t0 + np.arange(WIN) / FS
        Xs.append(to_grid(g, grid))
        ids.append(str(aid))
    return {'name': name, 'X': np.stack(Xs).astype(np.float32),
            'ids': np.array(ids)}


SOS_BP = (butter(4, CFG['BANDPASS'], btype='band', fs=FS, output='sos')
          if CFG['BANDPASS'] else None)
NOTCHES = [iirnotch(f0, CFG['NOTCH_Q'], fs=FS)
           for f0 in (CFG['NOTCH_HZ'] or ()) if f0 < FS / 2]


def preprocess_windows(W, chunk=2048):
    # identical for train crops and test actions:
    # mean-centre per window (as the organisers did for test), then for EMG:
    # linear detrend -> bandpass -> notch(es); NaN padding becomes 0
    out = np.empty(W.shape, np.float32)
    for s in range(0, len(W), chunk):
        w = W[s:s + chunk].astype(np.float64)
        miss = ~np.isfinite(w)
        mu = np.nan_to_num(np.nanmean(np.where(miss, np.nan, w), axis=2,
                                      keepdims=True))
        w = np.where(miss, 0.0, w - mu)
        emg = detrend(w[:, :N_EMG], axis=-1, type='linear')
        if SOS_BP is not None:
            emg = sosfiltfilt(SOS_BP, emg, axis=-1)
        for b, a in NOTCHES:
            emg = filtfilt(b, a, emg, axis=-1)
        emg = np.where(miss[:, :N_EMG], 0.0, emg)
        emg = emg - emg.mean(-1, keepdims=True)
        imu = w[:, N_EMG:]
        imu = imu - imu.mean(-1, keepdims=True)
        out[s:s + chunk] = np.concatenate([emg, imu], 1)
    return out


def normalise_emg(X, groups, ref_mask=None):
    # session mode: divide each EMG channel by the median per-window std of
    # that session (train: one centre crop per segment, so the rest/gesture
    # mix matches the test files; test: all actions of the file).
    # No labels are used, so this is label-free test-time calibration.
    if CFG['NORM_MODE'] == 'session':
        for g in np.unique(groups):
            m = groups == g
            Z = X[m] if ref_mask is None else X[m & ref_mask]
            if len(Z) == 0:
                Z = X[m]
            sc = np.median(Z[:, :N_EMG].std(axis=2), axis=0)
            sc = np.maximum(sc, 1e-6).astype(np.float32)
            X[m, :N_EMG] /= sc[None, :, None]
    elif CFG['NORM_MODE'] == 'window':
        s = X[:, :N_EMG].std(axis=(1, 2), keepdims=True)
        X[:, :N_EMG] /= np.maximum(s, 1e-6)
    return X


# ============================================================================
# CELL 3 - Feature extraction (EMG + IMU), vectorised over windows
# ----------------------------------------------------------------------------
# The ECG families map to EMG as: amplitude stats -> RMS/MAV/WL/ZC/SSC/Hjorth,
# Welch bands -> 20-80 Hz relative band powers + MNF/MDF/entropy, wavelet
# energies -> db4 level-3 energies, Hilbert -> smoothed envelope dynamics,
# multi-lead -> spatial RMS pattern and envelope correlations between
# electrodes. Features are computed on the normalised windows.
# ============================================================================

BANDS = [(20, 35), (35, 50), (50, 65), (65, 80)]


def _features_chunk(X):
    feats = {}
    eps = 1e-8
    emg = X[:, :N_EMG].astype(np.float64)
    N, C, L = emg.shape

    rms = np.sqrt((emg ** 2).mean(-1))
    mav = np.abs(emg).mean(-1)
    d1 = np.diff(emg, axis=-1)
    d2 = np.diff(d1, axis=-1)
    var0 = emg.var(-1) + eps
    var1 = d1.var(-1) + eps
    var2 = d2.var(-1) + eps
    mob = np.sqrt(var1 / var0)
    comp = np.sqrt(var2 / var1) / (mob + eps)
    zc = (np.signbit(emg[..., 1:]) != np.signbit(emg[..., :-1])).mean(-1)
    ssc = (np.signbit(d1[..., 1:]) != np.signbit(d1[..., :-1])).mean(-1)
    per_ch = {
        'rms': rms, 'log_rms': np.log(rms + eps), 'mav': mav,
        'wl': np.abs(d1).mean(-1), 'var': var0, 'zc': zc, 'ssc': ssc,
        'hj_mob': mob, 'hj_comp': comp,
        'skew': np.nan_to_num(skew(emg, axis=-1)),
        'kurt': np.nan_to_num(kurtosis(emg, axis=-1)),
    }

    # Welch spectrum inside the EMG band
    f, P = welch(emg, fs=FS, nperseg=min(128, L), noverlap=min(64, L // 2),
                 axis=-1)
    lo, hi = CFG['BANDPASS'] if CFG['BANDPASS'] else (0.0, FS / 2)
    band = (f >= lo) & (f <= hi)
    fb, Pb = f[band], P[..., band]
    tot = Pb.sum(-1) + eps
    for b0, b1 in BANDS:
        sel = (f >= b0) & (f < b1)
        per_ch[f'bp_{b0}_{b1}'] = P[..., sel].sum(-1) / tot
    per_ch['mnf'] = (Pb * fb).sum(-1) / tot
    cs = np.cumsum(Pb, -1)
    per_ch['mdf'] = fb[np.clip((cs < cs[..., -1:] / 2).sum(-1), 0, len(fb) - 1)]
    pn = Pb / tot[..., None]
    per_ch['spec_ent'] = -(pn * np.log(pn + eps)).sum(-1) / np.log(len(fb))
    per_ch['peak_f'] = fb[np.argmax(Pb, -1)]

    # wavelet energies (db4, 3 levels)
    coeffs = pywt.wavedec(emg, 'db4', level=3, axis=-1)
    energies = np.stack([(c ** 2).sum(-1) for c in coeffs], -1)
    e_tot = energies.sum(-1) + eps
    for i in range(energies.shape[-1]):
        per_ch[f'wav_rel_L{i}'] = energies[..., i] / e_tot
    per_ch['wav_log_energy'] = np.log(e_tot)

    # Hilbert envelope (smoothed over 50 ms)
    env = uniform_filter1d(np.abs(hilbert(emg, axis=-1)),
                           size=max(1, int(0.05 * FS)), axis=-1)
    env_m = env.mean(-1) + eps
    tn = np.linspace(-1.0, 1.0, L)
    per_ch['env_mean'] = env_m
    per_ch['env_std'] = env.std(-1)
    per_ch['env_max'] = env.max(-1)
    per_ch['env_cv'] = env.std(-1) / env_m
    per_ch['env_slope'] = (env * tn).sum(-1) / (tn ** 2).sum() / env_m
    per_ch['env_half_ratio'] = np.log((env[..., L // 2:].mean(-1) + eps)
                                      / (env[..., :L // 2].mean(-1) + eps))

    # spatial pattern across electrodes (placement / person invariant-ish)
    per_ch['rel_rms'] = rms / (rms.sum(1, keepdims=True) + eps)
    per_ch['log_rel_rms'] = np.log(rms + eps) - np.log(rms + eps).mean(1, keepdims=True)

    for name, arr in per_ch.items():
        for c in range(C):
            feats[f'{EMG_CH[c]}_{name}'] = arr[:, c]

    envz = (env - env.mean(-1, keepdims=True)) / (env.std(-1, keepdims=True) + eps)
    corr = np.einsum('ncl,nkl->nck', envz, envz) / L
    for a in range(C):
        for b in range(a + 1, C):
            feats[f'envcorr_{EMG_CH[a]}_{EMG_CH[b]}'] = corr[:, a, b]

    feats['emg_log_total_rms'] = np.log(np.sqrt((rms ** 2).mean(1)) + eps)
    tot_env = env.mean(1)
    feats['emg_env_slope'] = (tot_env * tn).sum(-1) / (tn ** 2).sum() / (tot_env.mean(-1) + eps)

    if CFG['USE_IMU']:
        imu = X[:, N_EMG:].astype(np.float64)
        di = np.diff(imu, axis=-1)
        for c, ch in enumerate(IMU_CH):
            feats[f'{ch}_std'] = imu[:, c].std(-1)
            feats[f'{ch}_ptp'] = np.ptp(imu[:, c], axis=-1)
            feats[f'{ch}_rms_diff'] = np.sqrt((di[:, c] ** 2).mean(-1))
        acc = np.linalg.norm(imu[:, :3], axis=1)
        gyr = np.linalg.norm(imu[:, 3:], axis=1)
        feats['acc_mag_mean'] = acc.mean(-1)
        feats['acc_mag_std'] = acc.std(-1)
        feats['gyr_mag_mean'] = gyr.mean(-1)
        feats['gyr_mag_std'] = gyr.std(-1)
        feats['gyr_mag_max'] = gyr.max(-1)

    return pd.DataFrame(feats)


def extract_features(X, chunk=2048):
    parts = [_features_chunk(X[s:s + chunk]) for s in range(0, len(X), chunk)]
    df = pd.concat(parts, ignore_index=True)
    return df.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)


# ============================================================================
# CELL 5 - Covariances, Riemannian tangent space, re-centering, EA
# ============================================================================

def spd_fn(C, fn, clip=True):
    # f(C) = V f(w) V^T for symmetric C; clip eigenvalues only for SPD inputs
    w, V = np.linalg.eigh(C)
    if clip:
        w = np.maximum(w, 1e-12)
    return (V * fn(w)[..., None, :]) @ np.swapaxes(V, -1, -2)


def window_covs(E):
    E = E.astype(np.float64)
    C = np.einsum('ncl,nkl->nck', E, E) / E.shape[-1]
    tr = np.trace(C, axis1=1, axis2=2)[:, None, None] / C.shape[1]
    return C + CFG['COV_SHRINK'] * tr * np.eye(C.shape[1])[None]


def le_mean(C):
    # log-Euclidean mean of SPD matrices
    return spd_fn(spd_fn(C, np.log).mean(0)[None], np.exp, clip=False)[0]


def inv_sqrt(M):
    return spd_fn(M[None], lambda w: 1.0 / np.sqrt(w))[0]


def tangent(C, W):
    # vectorised log-map at the reference (W = ref^-1/2), sqrt(2) on off-diagonals
    S = W[None] @ C @ W[None]
    L = spd_fn(S, np.log)
    iu = np.triu_indices(C.shape[1])
    coef = np.where(iu[0] == iu[1], 1.0, np.sqrt(2.0))
    return (L[:, iu[0], iu[1]] * coef).astype(np.float32)


def recentered_tangent(C, groups, ref_mask=None):
    # Zanini et al. 2018: per person, whiten by that person's mean covariance
    n = C.shape[1]
    out = np.empty((len(C), n * (n + 1) // 2), np.float32)
    for g in np.unique(groups):
        m = groups == g
        r = m if ref_mask is None else (m & ref_mask)
        if r.sum() < 5:
            r = m
        out[m] = tangent(C[m], inv_sqrt(le_mean(C[r])))
    return out


def euclidean_align(X, groups, ref_mask=None):
    # He & Wu 2020: X_i <- R^-1/2 X_i, R = arithmetic mean covariance of the person
    Y = X.copy()
    for g in np.unique(groups):
        m = groups == g
        r = m if ref_mask is None else (m & ref_mask)
        if r.sum() < 5:
            r = m
        R = window_covs(X[r, :N_EMG]).mean(0)
        W = inv_sqrt(R).astype(np.float32)
        Y[m, :N_EMG] = np.einsum('ck,nkl->ncl', W, X[m, :N_EMG])
    return Y


def bandpass_windows(X, lo, hi):
    sos = butter(4, (lo, hi), btype='band', fs=FS, output='sos')
    return sosfiltfilt(sos, X[:, :N_EMG].astype(np.float64), axis=-1)


# ============================================================================
# CELL 6 - Compact CNN (Atzori 2016 / Cote-Allard 2019 style) + AdaBN
# ============================================================================

class SimpleEMGCNN(nn.Module):
    def __init__(self, in_ch, n_cls, width=64, dropout=0.3):
        super().__init__()

        def blk(i, o, k, s):
            return nn.Sequential(nn.Conv1d(i, o, k, stride=s, padding=k // 2),
                                 nn.BatchNorm1d(o), nn.GELU())
        self.body = nn.Sequential(blk(in_ch, width, 7, 1), blk(width, width, 5, 2),
                                  blk(width, 2 * width, 5, 2), blk(2 * width, 2 * width, 3, 2),
                                  blk(2 * width, 2 * width, 3, 1))
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(4 * width, n_cls)

    def forward(self, x):
        h = self.body(x)
        return self.fc(self.drop(torch.cat([h.mean(-1), h.amax(-1)], 1)))


class EEGNetLike(nn.Module):
    # AuraNet-style temporal-spatial CNN (the organisers' baseline design:
    # time conv -> depthwise channel mix -> separable conv -> linear), i.e.
    # EEGNet (Lawhern et al. 2018) adapted to 8 EMG channels at 200 Hz.
    def __init__(self, n_ch, n_cls, F1=16, D=2, F2=32, k1=25, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, F1, (1, k1), padding=(0, k1 // 2), bias=False), nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, (n_ch, 1), groups=F1, bias=False), nn.BatchNorm2d(F1 * D),
            nn.ELU(), nn.AvgPool2d((1, 4)), nn.Dropout(dropout),
            nn.Conv2d(F1 * D, F1 * D, (1, 15), padding=(0, 7), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, 1, bias=False), nn.BatchNorm2d(F2),
            nn.ELU(), nn.AvgPool2d((1, 8)), nn.Dropout(dropout))
        with torch.no_grad():
            n_flat = self.features(torch.zeros(1, 1, n_ch, WIN)).flatten(1).shape[1]
        self.fc = nn.Linear(n_flat, n_cls)

    def forward(self, x):
        return self.fc(self.features(x.unsqueeze(1)).flatten(1))


def make_model(arch, in_ch, n_cls):
    if arch == 'eegnet':
        return EEGNetLike(in_ch, n_cls, dropout=CFG['CNN_DROPOUT'])
    return SimpleEMGCNN(in_ch, n_cls, CFG['CNN_WIDTH'], CFG['CNN_DROPOUT'])


def cnn_augment(x, rot):
    B = x.shape[0]
    x = x.clone()
    e = x[:, :N_EMG]
    a, g = CFG['AUG_CH_GAIN'], CFG['AUG_GLOBAL_GAIN']
    e.mul_(torch.exp(torch.empty(B, N_EMG, 1, device=x.device).uniform_(-a, a)))
    e.mul_(torch.exp(torch.empty(B, 1, 1, device=x.device).uniform_(-g, g)))
    if rot:
        # barrel shift: every sample gets a random circular electrode shift
        shift = torch.randint(0, N_EMG, (B,), device=x.device)
        src = (torch.arange(N_EMG, device=x.device)[None, :] - shift[:, None]) % N_EMG
        e.copy_(torch.gather(e.clone(), 1, src[:, :, None].expand(-1, -1, e.shape[-1])))
    return x + torch.randn_like(x) * CFG['AUG_NOISE_STD']


@torch.no_grad()
def cnn_proba(m, X, bs=1024):
    m.eval()
    return np.concatenate([F.softmax(m(X[i:i + bs]), 1).cpu().numpy()
                           for i in range(0, len(X), bs)], 0)


@torch.no_grad()
def cnn_proba_adabn(model, X, groups):
    m = copy.deepcopy(model)
    base = {k: v.clone() for k, v in m.state_dict().items()}
    bns = [b for b in m.modules() if isinstance(b, nn.modules.batchnorm._BatchNorm)]
    P = np.zeros((len(X), m.fc.out_features), np.float32)
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        it = torch.as_tensor(idx, device=X.device)
        m.load_state_dict(base)
        if len(idx) >= 2:
            for b in bns:
                b.reset_running_stats()
                b.momentum = None
            m.eval()
            for b in bns:
                b.train()
            m(X[it])
        P[idx] = cnn_proba(m, X[it])
    return P


def train_cnn(Xtr, ytr, Xva, yva, n_cls, rot=False, tag='', arch='cnn'):
    model = make_model(arch, Xtr.shape[1], n_cls).to(device)
    cw = compute_class_weight('balanced', classes=np.arange(n_cls), y=ytr)
    ce = nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32, device=device),
                             label_smoothing=CFG['LABEL_SMOOTHING'])
    opt = torch.optim.AdamW(model.parameters(), lr=CFG['CNN_LR'], weight_decay=CFG['CNN_WD'])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CFG['CNN_EPOCHS'])
    Xt = torch.from_numpy(Xtr).to(device)
    yt = torch.from_numpy(ytr).to(device)
    Xv = torch.from_numpy(Xva).to(device)
    bs = min(CFG['CNN_BATCH'], len(ytr))
    best, best_state, patience = -1.0, None, 0
    for ep in range(CFG['CNN_EPOCHS']):
        model.train()
        perm = torch.randperm(len(yt), device=device)
        for i in range(len(yt) // bs):
            idx = perm[i * bs:(i + 1) * bs]
            loss = ce(model(cnn_augment(Xt[idx], rot)), yt[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sched.step()
        f1v = macro_f1(yva, cnn_proba(model, Xv))
        if f1v > best + 1e-4:
            best, patience = f1v, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= CFG['CNN_PATIENCE']:
                break
    model.load_state_dict(best_state)
    model.eval()
    print(f'      {tag} stopped ep {ep + 1}, inner-val mF1 {best:.4f}')
    return model


# ============================================================================
# CELL 7 - Shared helpers (metric, prior matching, classifiers)
# ============================================================================

def macro_f1(y, P):
    return float(f1_score(y, P.argmax(1), average='macro'))


def match_priors(P, target, iters=50):
    Q = P.copy() + 1e-9
    for _ in range(iters):
        Q *= (target / Q.mean(0))[None, :]
        Q /= Q.sum(1, keepdims=True)
    return Q


def match_priors_by_group(P, groups, target):
    Q = P.copy()
    for g in np.unique(groups):
        m = groups == g
        Q[m] = match_priors(P[m], target)
    return Q


def session_standardize(Fm, groups, ref_mask=None):
    out = np.empty_like(Fm, dtype=np.float32)
    for g in np.unique(groups):
        m = groups == g
        r = m if ref_mask is None else (m & ref_mask)
        if r.sum() < 5:
            r = m
        mu, sd = Fm[r].mean(0), Fm[r].std(0)
        sd[sd < 1e-6] = 1.0
        out[m] = (Fm[m] - mu) / sd
    return np.clip(np.nan_to_num(out), -10, 10).astype(np.float32)


def full_proba(clf, X, n_cls):
    # predict_proba with columns for every class (a fold may miss none, but be safe)
    P = np.zeros((len(X), n_cls), np.float32)
    P[:, clf.classes_] = clf.predict_proba(X)
    return P


def fit_lda(Xtr, ytr, n_cls):
    return make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(
        solver='lsqr', shrinkage='auto', priors=np.ones(n_cls) / n_cls)).fit(Xtr, ytr)


def fit_svm(Xtr, ytr, seed):
    n = min(CFG['SVM_MAX_TRAIN'], len(ytr))
    idx = np.random.RandomState(seed).choice(len(ytr), n, replace=False)
    return make_pipeline(StandardScaler(), SVC(C=1.0, gamma='scale', probability=True,
                                               class_weight='balanced', random_state=seed)
                         ).fit(Xtr[idx], ytr[idx])


def fit_lr(Xtr, ytr):
    return make_pipeline(StandardScaler(), LogisticRegression(
        C=1.0, max_iter=3000, class_weight='balanced')).fit(Xtr, ytr)


def mdm_proba(Ttr, ytr, Tq, n_cls):
    # log-Euclidean minimum distance to mean (distances in tangent coordinates)
    M = np.stack([Ttr[ytr == c].mean(0) for c in range(n_cls)])
    d2 = ((Tq[:, None, :] - M[None]) ** 2).sum(-1)
    z = -d2 / (np.median(d2) + 1e-9)
    z -= z.max(1, keepdims=True)
    P = np.exp(z)
    return (P / P.sum(1, keepdims=True)).astype(np.float32)


# ============================================================================
# CELL 8 - v4 model zoo. Each model returns (P_heldout, P_test) for one fold.
# D holds the precomputed representations, f the fold masks.
# ============================================================================

D = {}


def m_lgbm(rep, extra=None):
    def run(f):
        A, B = D[rep]
        y = D['y']
        P_ho = np.zeros((f['ho'].sum(), D['C']), np.float32)
        P_te = np.zeros((len(B), D['C']), np.float32)
        S = max(1, CFG['LGBM_SEEDS'])
        for s in range(S):
            params = dict(CFG['LGBM_PARAMS'], **(extra or {}),
                          random_state=CFG['SEED'] + 100 * f['k'] + s, n_jobs=-1,
                          objective='multiclass', num_class=D['C'])
            m = lgb.LGBMClassifier(**params)
            m.fit(A[f['tr']], y[f['tr']], eval_set=[(A[f['va']], y[f['va']])],
                  eval_metric='multi_logloss',
                  callbacks=[lgb.early_stopping(CFG['LGBM_EARLY_STOP'], verbose=False)])
            P_ho += full_proba(m, A[f['ho']], D['C']) / S
            P_te += full_proba(m, B, D['C']) / S
        return P_ho, P_te
    return run


def m_riemann_ts_lr(f):
    Ctr, Cte = D['cov']
    W = inv_sqrt(le_mean(Ctr[f['tr']]))
    clf = fit_lr(tangent(Ctr[f['tr']], W), D['y'][f['tr']])
    return (full_proba(clf, tangent(Ctr[f['ho']], W), D['C']),
            full_proba(clf, tangent(Cte, W), D['C']))


def m_fbts_lr_rc(f):
    A, B = D['fbts_rc']
    clf = fit_lr(A[f['tr']], D['y'][f['tr']])
    return full_proba(clf, A[f['ho']], D['C']), full_proba(clf, B, D['C'])


def m_cnn_ea_rot(f):
    A, B = D['cnn_ea']
    P_ho = np.zeros((f['ho'].sum(), D['C']), np.float32)
    P_te = np.zeros((len(B), D['C']), np.float32)
    S = max(1, CFG['CNN_SEEDS'])
    Xho = torch.from_numpy(A[f['ho']]).to(device)
    Xte = torch.from_numpy(B).to(device)
    for s in range(S):
        seed_everything(CFG['SEED'] + 1000 * (f['k'] + 1) + s)
        model = train_cnn(A[f['tr']], D['y'][f['tr']], A[f['va']], D['y'][f['va']],
                          D['C'], rot=True, tag=f'cnn_ea_rot fold {f["k"] + 1} seed {s + 1}')
        P_ho += cnn_proba(model, Xho) / S
        P_te += cnn_proba(model, Xte) / S
        del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return P_ho, P_te


MODELS = {
    # name: (representations needed, function, is_riemannian)
    'lgbm_full_sz':       ({'full_sz'}, m_lgbm('full_sz'), False),
    'lgbm_plus':          ({'plus'}, m_lgbm('plus'), False),
    'lgbm_plus_et':       ({'plus'}, m_lgbm('plus', CFG['LGBM_ET_PARAMS']), False),
    'riemann_ts_lr':      ({'cov'}, m_riemann_ts_lr, True),
    'riemann_fbts_lr_rc': ({'fbts_rc'}, m_fbts_lr_rc, True),
    'cnn_ea_rot':         ({'cnn_ea'}, m_cnn_ea_rot, False),
}


def build_representations(needed, X_tr, X_te, sess_tr, ref_tr, file_te):
    # label-free, per window / per person -> computed once for all folds
    t0 = time.time()
    Fdf = extract_features(X_tr)
    D['full_names'] = list(Fdf.columns)
    Ftr = Fdf.values.astype(np.float32)
    Fte = extract_features(X_te)[D['full_names']].values.astype(np.float32)
    D['full'] = (Ftr, Fte)
    if needed & {'full_sz', 'plus'}:
        D['full_sz'] = (np.concatenate([Ftr, session_standardize(Ftr, sess_tr, ref_tr)], 1),
                        np.concatenate([Fte, session_standardize(Fte, file_te, None)], 1))
    if needed & {'cov', 'plus'}:
        D['cov'] = (window_covs(X_tr[:, :N_EMG]), window_covs(X_te[:, :N_EMG]))
    if needed & {'fbts_rc', 'plus'}:
        parts_tr, parts_te = [], []
        for lo, hi in CFG['FB_BANDS']:
            Ctr = window_covs(bandpass_windows(X_tr, lo, hi))
            Cte = window_covs(bandpass_windows(X_te, lo, hi))
            parts_tr.append(recentered_tangent(Ctr, sess_tr, ref_tr))
            parts_te.append(recentered_tangent(Cte, file_te, None))
        D['fbts_rc'] = (np.concatenate(parts_tr, 1), np.concatenate(parts_te, 1))
    if 'plus' in needed:
        ts_tr = recentered_tangent(D['cov'][0], sess_tr, ref_tr)
        ts_te = recentered_tangent(D['cov'][1], file_te, None)
        D['plus'] = (np.concatenate([D['full_sz'][0], ts_tr, D['fbts_rc'][0]], 1),
                     np.concatenate([D['full_sz'][1], ts_te, D['fbts_rc'][1]], 1))
    if 'cnn_ea' in needed:
        Xe_tr = euclidean_align(X_tr, sess_tr, ref_tr)
        Xe_te = euclidean_align(X_te, file_te, None)
        D['cnn_ea'] = (np.clip(Xe_tr[:, MODEL_CH], -CFG['CLIP_SIGNAL'], CFG['CLIP_SIGNAL']),
                       np.clip(Xe_te[:, MODEL_CH], -CFG['CLIP_SIGNAL'], CFG['CLIP_SIGNAL']))
        del Xe_tr, Xe_te
    shapes = {k: v[0].shape[1:] for k, v in D.items()
              if isinstance(v, tuple) and len(v) == 2 and hasattr(v[0], 'shape')}
    print(f'  representations: {shapes} [{time.time() - t0:.0f}s]')


# ============================================================================
# CELL 9 - v3 transductive per-person refinement (unchanged)
# ============================================================================

TD_FEATURES = ([f'{c}_log_rms' for c in EMG_CH] + [f'{c}_log_rel_rms' for c in EMG_CH]
               + [f'{c}_mnf' for c in EMG_CH] + [f'{c}_wav_log_energy' for c in EMG_CH])


def transductive_refine(P0, Z, target, alpha, tau, iters):
    eps = 1e-9
    logP0 = np.log(P0 + eps)
    Q = match_priors(P0, target)
    for _ in range(iters):
        W = Q / (Q.sum(0, keepdims=True) + eps)
        mu = W.T @ Z
        d2 = ((Z[:, None, :] - mu[None, :, :]) ** 2).mean(-1)
        d2 = d2 / (np.median(d2) + eps)
        logits = alpha * logP0 - (1.0 - alpha) * d2 / tau
        logits -= logits.max(1, keepdims=True)
        Q = np.exp(logits)
        Q /= Q.sum(1, keepdims=True)
        Q = match_priors(Q, target)
    return Q


def refine_by_group(P, Zraw, groups, target, alpha, tau):
    out = P.copy()
    for g in np.unique(groups):
        m = groups == g
        if m.sum() < 2 * len(target):
            out[m] = match_priors(P[m], target)
            continue
        Z = Zraw[m]
        Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-6)
        out[m] = transductive_refine(P[m], Z, target, alpha, tau, CFG['TD_ITERS'])
    return out


# ============================================================================
# CELL 10 - Blending: weight search, class offsets, nested evaluation
# ============================================================================

def blend(Ps, w):
    return np.tensordot(np.asarray(w, np.float64), np.stack(Ps), 1).astype(np.float32)


def search_weights(Ps, y, seed):
    # non-negative weights summing to 1 that maximise macro-F1
    M = len(Ps)
    rng = np.random.RandomState(seed)
    cands = ([np.eye(M)[i] for i in range(M)] + [np.ones(M) / M]
             + list(rng.dirichlet(np.full(M, 0.5), CFG['BLEND_SEARCH_ITERS'])))
    stack = np.stack(Ps)
    best_w, best = None, -1.0
    for w in cands:
        s = macro_f1(y, np.tensordot(w, stack, 1))
        if s > best:
            best, best_w = s, w
    # local refinement around the best candidate
    for _ in range(300):
        w = np.clip(best_w + rng.normal(0, 0.05, M), 0, None)
        if w.sum() == 0:
            continue
        w /= w.sum()
        s = macro_f1(y, np.tensordot(w, stack, 1))
        if s > best:
            best, best_w = s, w
    return best_w, best


def apply_bias(P, b):
    L = np.log(P + 1e-9) + b[None, :]
    L -= L.max(1, keepdims=True)
    Q = np.exp(L)
    return (Q / Q.sum(1, keepdims=True)).astype(np.float32)


def tune_bias(P, y, rounds=3):
    # per-class additive log-score offsets that maximise macro-F1
    C = P.shape[1]
    b = np.zeros(C)
    grid = np.linspace(-1.5, 1.5, 31)
    best = macro_f1(y, P)
    for _ in range(rounds):
        for c in range(C):
            for v in grid:
                b2 = b.copy()
                b2[c] = v
                s = macro_f1(y, apply_bias(P, b2))
                if s > best + 1e-6:
                    best, b = s, b2
    return b, best


# ============================================================================
# CELL 11 - Main
# ============================================================================

def main():
    t_start = time.time()
    train_files = sorted(DATA_ROOT.rglob('train_*.xlsx'), key=natural_key)
    test_files = sorted(DATA_ROOT.rglob('test_*.xlsx'), key=natural_key)
    subs = sorted(DATA_ROOT.rglob('sample_submission.csv'))
    print(f'train files: {len(train_files)}  test files: {len(test_files)}')

    print('\n[1] Windows (same preprocessing as v3 / benchmark) ...')
    res = [r for r in Parallel(n_jobs=N_JOBS)(delayed(process_train_file)(p)
                                              for p in train_files) if r is not None]
    X_tr = np.concatenate([r['X'] for r in res])
    lab_tr = np.concatenate([r['y'] for r in res])
    ref_tr = np.concatenate([r['ref'] for r in res])
    sess_tr = np.concatenate([[r['name']] * len(r['y']) for r in res])
    del res
    found = sorted(set(lab_tr))
    classes = [c for c in CLASS_ORDER if c in found] + [c for c in found if c not in CLASS_ORDER]
    C = len(classes)
    y_tr = np.array([classes.index(c) for c in lab_tr], np.int64)
    sessions = sorted(np.unique(sess_tr), key=natural_key)

    X_tr = preprocess_windows(X_tr)
    X_tr = normalise_emg(X_tr, sess_tr, ref_tr)
    imu_scale = np.sqrt((X_tr[:, N_EMG:] ** 2).mean(axis=(0, 2))) + 1e-6
    X_tr[:, N_EMG:] /= imu_scale[None, :, None]
    np.clip(X_tr, -CFG['CLIP_SIGNAL'], CFG['CLIP_SIGNAL'], out=X_tr)

    tres = Parallel(n_jobs=N_JOBS)(delayed(process_test_file)(p) for p in test_files)
    X_te = np.concatenate([r['X'] for r in tres])
    ids_te = np.concatenate([r['ids'] for r in tres])
    file_te = np.concatenate([[r['name']] * len(r['ids']) for r in tres])
    del tres
    X_te = preprocess_windows(X_te)
    X_te = normalise_emg(X_te, file_te, None)
    X_te[:, N_EMG:] /= imu_scale[None, :, None]
    np.clip(X_te, -CFG['CLIP_SIGNAL'], CFG['CLIP_SIGNAL'], out=X_te)
    print(f'  train windows {len(y_tr)} (test-like {ref_tr.sum()}), sessions {len(sessions)}, '
          f'test actions {len(ids_te)}')

    print('\n[2] Representations ...')
    models = [m for m in CFG['MODELS'] if m in MODELS]
    D.update({'y': y_tr, 'C': C, 'sess': sess_tr, 'file_te': file_te})
    build_representations(set().union(*[MODELS[m][0] for m in models]),
                          X_tr, X_te, sess_tr, ref_tr, file_te)
    td_idx = [D['full_names'].index(n) for n in TD_FEATURES if n in D['full_names']]
    Zc_tr = D['full'][0][:, td_idx].astype(np.float64)
    Zc_te = D['full'][1][:, td_idx].astype(np.float64)
    del X_tr, X_te
    gc.collect()

    rng = np.random.RandomState(CFG['SEED'])
    n_folds = max(2, min(CFG['N_FOLDS'], len(sessions)))
    fold_of = {s: i % n_folds for i, s in enumerate(rng.permutation(sessions))}
    print(f'\n[3] {len(models)} models x {n_folds} folds (held-out = new people)')

    oof = {m: np.zeros((len(y_tr), C), np.float32) for m in models}
    test = {m: np.zeros((len(ids_te), C), np.float32) for m in models}
    oof_mask = np.zeros(len(y_tr), bool)
    for k in range(n_folds):
        seed_everything(CFG['SEED'] + k)
        ho_s = [s for s in sessions if fold_of[s] == k]
        rest_s = [s for s in sessions if fold_of[s] != k]
        if len(rest_s) >= 3:
            iv_s = list(np.random.RandomState(CFG['SEED'] + k).choice(
                rest_s, max(1, int(round(CFG['INNER_VAL_FRAC'] * len(rest_s)))), replace=False))
        else:
            iv_s = list(ho_s)
        tr_s = [s for s in rest_s if s not in iv_s]
        f = {'k': k, 'tr': np.isin(sess_tr, tr_s), 'va': np.isin(sess_tr, iv_s) & ref_tr,
             'ho': np.isin(sess_tr, ho_s) & ref_tr}
        oof_mask |= f['ho']
        print(f'\n=== Fold {k + 1}/{n_folds}  held-out: {[str(s) for s in ho_s]}')
        for m in models:
            t0 = time.time()
            p_ho, p_te = MODELS[m][1](f)
            oof[m][f['ho']] = p_ho
            test[m] += p_te / n_folds
            print(f'  {m:20s} held-out mF1 {macro_f1(y_tr[f["ho"]], p_ho):.4f}  '
                  f'[{time.time() - t0:.0f}s]')

    # ---------------- blend members ----------------
    yo, g_oof = y_tr[oof_mask], sess_tr[oof_mask]
    fold_oof = np.array([fold_of[s] for s in g_oof])
    target = np.bincount(y_tr[ref_tr], minlength=C) / ref_tr.sum()
    members, mem_oof, mem_te = [], [], []
    for m in models:
        members.append(m)
        mem_oof.append(oof[m][oof_mask])
        mem_te.append(test[m])
        if MODELS[m][2]:      # Riemannian: also a per-person prior-matched copy
            members.append(m + '_pm')
            mem_oof.append(match_priors_by_group(oof[m][oof_mask], g_oof, target))
            mem_te.append(match_priors_by_group(test[m], file_te, target))

    print('\n' + '=' * 78)
    print('[4] Out-of-fold macro-F1 per member (held-out people, test-like windows)')
    print('=' * 78)
    report = {'classes': classes, 'members': {}}
    for name, P in zip(members, mem_oof):
        report['members'][name] = macro_f1(yo, P)
        print(f'  {name:24s} {report["members"][name]:.4f}')

    # ---------------- nested evaluation of the blend ----------------
    P_nest = np.zeros_like(mem_oof[0])
    for k in range(n_folds):
        tr, te = fold_oof != k, fold_oof == k
        w_k, _ = search_weights([P[tr] for P in mem_oof], yo[tr], CFG['SEED'] + k)
        P_nest[te] = blend([P[te] for P in mem_oof], w_k)
    w_all, f1_fit = search_weights(mem_oof, yo, CFG['SEED'])
    report['blend_weights'] = {n: round(float(w), 4) for n, w in zip(members, w_all) if w > 1e-4}
    report['blend_f1_in_sample'] = f1_fit
    report['blend_f1_nested'] = macro_f1(yo, P_nest)
    print(f'\n  BLEND weights: {report["blend_weights"]}')
    print(f'  BLEND macro-F1: in-sample {f1_fit:.4f} | nested (honest) '
          f'{report["blend_f1_nested"]:.4f}')

    # ---------------- post-processing candidates (scored on nested blend) ----------------
    cands = {'none': P_nest}
    if CFG['CLASS_BIAS'] != 'off':
        P_b = np.zeros_like(P_nest)
        for k in range(n_folds):
            tr, te = fold_oof != k, fold_oof == k
            b_k, _ = tune_bias(P_nest[tr], yo[tr])
            P_b[te] = apply_bias(P_nest[te], b_k)
        cands['class_bias'] = P_b
    if CFG['POSTPROC'] in ('auto', 'prior'):
        cands['prior'] = match_priors_by_group(P_nest, g_oof, target)
    td_params = {}
    if CFG['POSTPROC'] in ('auto', 'transductive') and Zc_tr.shape[1] > 0:
        for a in CFG['TD_ALPHAS']:
            for tau in CFG['TD_TAUS']:
                key = f'td_a{a}_t{tau}'
                td_params[key] = (a, tau)
                cands[key] = refine_by_group(P_nest, Zc_tr[oof_mask], g_oof, target, a, tau)
    cand_f1 = {k: macro_f1(yo, v) for k, v in cands.items()}
    print('\n  post-processing on the nested blend (macro-F1):')
    for kk, v in sorted(cand_f1.items(), key=lambda kv: -kv[1]):
        print(f'    {kk:18s} {v:.4f}')
    allowed = list(cands)
    if CFG['CLASS_BIAS'] == 'on':
        allowed = ['class_bias']
    elif CFG['POSTPROC'] == 'prior':
        allowed = ['prior']
    elif CFG['POSTPROC'] == 'transductive':
        allowed = list(td_params)
    elif CFG['POSTPROC'] == 'none':
        allowed = [a for a in allowed if a in ('none', 'class_bias')]
    post = max(allowed, key=cand_f1.get)
    report['postproc_f1'] = cand_f1
    report['postproc_used'] = post
    report['oof_f1_final'] = cand_f1[post]
    P_final = cands[post]
    print(f'  -> used: {post}')
    print(f'\n  FINAL out-of-fold macro-F1 (nested, held-out people) = {report["oof_f1_final"]:.4f}')
    print(classification_report(yo, P_final.argmax(1), target_names=classes, digits=4))
    print('  confusion matrix (rows = true):')
    print(pd.DataFrame(confusion_matrix(yo, P_final.argmax(1)),
                       index=classes, columns=classes).to_string())
    per_sess = {str(s): macro_f1(yo[g_oof == s], P_final[g_oof == s])
                for s in sessions if (g_oof == s).any()}
    report['per_session_f1'] = per_sess
    print('  per-session macro-F1:', {s: round(v, 3) for s, v in per_sess.items()})

    # ---------------- test predictions ----------------
    print('\n[5] Test inference + submission')
    P_te = blend(mem_te, w_all)
    if post == 'class_bias':
        b_all, _ = tune_bias(blend(mem_oof, w_all), yo)
        report['class_bias'] = dict(zip(classes, np.round(b_all, 3).tolist()))
        P_te = apply_bias(P_te, b_all)
        print('  class offsets:', report['class_bias'])
    elif post == 'prior':
        P_te = match_priors_by_group(P_te, file_te, target)
    elif post in td_params:
        a, tau = td_params[post]
        P_te = refine_by_group(P_te, Zc_te, file_te, target, a, tau)
    pred = {i: classes[j] for i, j in zip(ids_te, P_te.argmax(1))}
    if subs:
        sample = pd.read_csv(subs[0])
        id_col = sample.columns[0]
        tgt_col = sample.columns[1] if sample.shape[1] > 1 else 'gesture'
        ids = sample[id_col].astype(str)
        missing = [i for i in ids if i not in pred]
        if missing:
            print(f'  [warn] {len(missing)} sample ids without a test window -> "{classes[0]}"')
        sub = pd.DataFrame({id_col: sample[id_col],
                            tgt_col: [pred.get(i, classes[0]) for i in ids]})
    else:
        sub = pd.DataFrame({'action_id': ids_te, 'gesture': [pred[i] for i in ids_te]})
    assert sub.iloc[:, 0].is_unique, 'duplicate ids in submission'
    assert set(sub.iloc[:, 1]) <= set(classes), 'unknown label in submission'
    sub.to_csv(OUT_DIR / 'submission.csv', index=False)
    if Path('/kaggle/working').exists():
        sub.to_csv('/kaggle/working/submission.csv', index=False)
    pd.DataFrame({'action_id': ids_te, 'file': file_te,
                  **{f'p_{c}': P_te[:, i] for i, c in enumerate(classes)}}
                 ).to_csv(OUT_DIR / 'test_probabilities.csv', index=False)
    np.savez_compressed(OUT_DIR / 'v4_member_probs.npz', classes=np.array(classes),
                        members=np.array(members), y_oof=yo, sess_oof=g_oof,
                        **{f'oof__{n}': P for n, P in zip(members, mem_oof)},
                        **{f'test__{n}': P for n, P in zip(members, mem_te)})
    print('  predicted class counts:', sub.iloc[:, 1].value_counts().to_dict())

    report['seconds_total'] = round(time.time() - t_start, 1)
    report['cfg'] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in CFG.items()}
    with open(OUT_DIR / 'cv_report.json', 'w') as fh:
        json.dump(report, fh, indent=2, default=lambda o: o.tolist()
                  if hasattr(o, 'tolist') else str(o))
    print(f'\nDone in {report["seconds_total"] / 60:.1f} min. Submission ->',
          '/kaggle/working/submission.csv' if Path('/kaggle/working').exists()
          else OUT_DIR / 'submission.csv')
    return sub, report


if __name__ == '__main__':
    submission, cv_report = main()
