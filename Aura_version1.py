# ============================================================================
# Aura_version1.py
# Aura: Calibration-Free EMG Challenge (Kaggle / PiEEG) -- version 1
# ----------------------------------------------------------------------------
# The PhysioRobustNet thesis pipeline (12-lead ECG, 4 rhythm classes) adapted
# to 8-channel EMG + 6-axis IMU hand-gesture recognition on people that are
# NOT in the training data. Scored with macro-F1 over 6 gestures:
#     rest, fist, open, point, pinch, thumb-up
#
# Thesis component                      ->  EMG version (this file)
# ---------------------------------------------------------------------------
# bandpass 0.5-40 Hz, wavelet denoise,  ->  irregular packets -> uniform 200 Hz
# z-score per lead                          grid, detrend, bandpass 20-80 Hz,
#                                           50/60 Hz notch, per-session
#                                           (per-person) channel scaling
# 348 raw + 250 engineered ECG feats    ->  ~300 EMG/IMU feats (time domain,
#                                           Welch bands, wavelet energies,
#                                           Hilbert envelope, spatial pattern,
#                                           envelope correlations, IMU motion)
# mRMR + Boruta + LightGBM vote         ->  same 3-way vote, fit inside each fold
# PhysioRobustNet (signal CNN + feature ->  same architecture, 14 input channels,
# MLP + uncertainty-aware fusion head)      1.4 s windows (280 samples)
# clean -> TRADES -> Gaussian smoothing ->  same stages + EMG augmentations
#                                           (electrode gain, channel dropout)
# stratified 70/15/15 random split      ->  GroupKFold over recording sessions,
#                                           inner validation sessions for
#                                           early stopping (new-person proxy)
# (new) LightGBM on the selected features, blended with the network using the
#       out-of-fold macro-F1.
#
# Data facts this is built around (checked on train_3, train_17, test_1..5):
#   * train_k.xlsx = one 180 s session; only label_source == 'guided' rows
#     match the test actions (45 rest + 6 x 5 gesture segments per session).
#   * test_k.xlsx  = 1.4 s actions (action_id), every channel mean-centred
#     per action, so gravity/orientation is removed from the IMU too.
#   * rows arrive in packets (many duplicate t values) and the row rate
#     differs between train (~155-180/s) and test (~215-245/s), so every
#     window is re-sampled on its time axis to a fixed 200 Hz grid.
#   * EMG amplitude differs ~13x between sessions -> per-session scaling.
#   * mains hum is visible at ~50 Hz in some files and ~60 Hz in others.
#
# HOW TO RUN ON KAGGLE
#   1. New notebook -> "Add Input" -> the competition
#      (vr-calibration-free-emg-challenge-pieeg).
#   2. Settings -> Accelerator: GPU T4 x2. Internet ON is only needed to pip
#      install mrmr-selection / Boruta (they are skipped if unavailable).
#   3. Paste this whole file into one cell and run (or upload it and run
#      `!python Aura_version1.py`).
#   4. Output: /kaggle/working/submission.csv (+ models / reports under
#      /kaggle/working/aura_v1/).
#   Optional env vars: AURA_DATA_DIR (data folder), AURA_OUT_DIR (outputs),
#   AURA_QUICK=1 (tiny smoke-test run).
# ============================================================================


# ============================================================================
# CELL 0 - Setup: packages, config, device, paths
# ============================================================================

import os
import re
import sys
import gc
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


def pip_install(pkgs):
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q'] + pkgs,
                   check=False)


for _imp, _pip in [('openpyxl', 'openpyxl'),
                   ('pywt', 'PyWavelets'),
                   ('lightgbm', 'lightgbm'),
                   ('mrmr', 'mrmr-selection'),
                   ('boruta', 'Boruta')]:
    try:
        __import__(_imp)
    except ImportError:
        print(f'installing {_pip} ...')
        pip_install([_pip])

# Older Boruta releases still use NumPy aliases removed in NumPy >= 1.24
for _alias, _type in (('int', int), ('float', float), ('bool', bool)):
    if not hasattr(np, _alias):
        setattr(np, _alias, _type)

import pywt
from scipy.signal import (butter, sosfiltfilt, iirnotch, filtfilt, welch,
                          hilbert, detrend)
from scipy.ndimage import uniform_filter1d
from scipy.stats import skew, kurtosis
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
from joblib import Parallel, delayed
import lightgbm as lgb

import torch
import torch.nn as nn
import torch.nn.functional as F

QUICK = os.environ.get('AURA_QUICK', '0') == '1'

CFG = {
    'SEED': 42,

    # ---- windows / signal ----
    'FS': 200,                    # uniform resampling rate (Hz)
    'WIN_SEC': 1.4,               # test actions are 1.4 s
    'CROP_STRIDE_SEC': 0.10,      # sliding crops inside each guided segment
    'MIN_SEG_SEC': 1.20,          # shorter guided segments are skipped
    'BANDPASS': (20.0, 80.0),     # EMG band (train row rate ~165/s -> keep < 82)
    'NOTCH_HZ': (50.0, 60.0),     # both mains frequencies were seen
    'NOTCH_Q': 30.0,
    'NORM_MODE': 'session',       # 'session' (per file = per person) | 'window'
    'USE_IMU': True,
    'CLIP_SIGNAL': 20.0,

    # ---- validation ----
    'N_FOLDS': 5,                 # GroupKFold over sessions
    'INNER_VAL_FRAC': 0.15,       # sessions held out of each fold for early stop

    # ---- feature selection (thesis: mRMR + Boruta + LightGBM vote) ----
    'FEATURE_SELECTION': True,
    'SEL_SUBSAMPLE': 3000,
    'SEL_K_FRAC': 0.30,
    'SEL_POLICY': 'auto',         # 'auto' | 'majority' | 'union'
    'BORUTA_MAX_ITER': 30,

    # ---- network stages ----
    'TRAIN_NN': True,
    'TRAIN_TRADES': True,
    'TRAIN_SMOOTHING': False,
    'FINAL_STAGE': 'best',        # 'best' | 'clean' | 'trades' | 'smoothing'

    # ---- optimisation ----
    'OPTIMIZER': 'adamw',         # 'adamw' | 'sgd' (thesis used SGD nesterov)
    'BATCH_SIZE': 256,
    'LEARNING_RATE': 2e-3,
    'MOMENTUM': 0.9,
    'WEIGHT_DECAY': 1e-2,
    'LABEL_SMOOTHING': 0.1,
    'GRAD_CLIP': 5.0,
    'LR_WARMUP_EPOCHS': 3,
    'CLEAN_EPOCHS': 60,
    'TRADES_EPOCHS': 25,
    'TRADES_LR_MULT': 0.3,
    'SMOOTH_EPOCHS': 20,

    # ---- architecture (PhysioRobustNet, sized for 280-sample windows) ----
    'WIDTH': 32,
    'STAGE_WIDTHS': (64, 96, 128),
    'DILATIONS': (1, 2, 4),
    'DROPOUT_TCN': 0.15,
    'DROPOUT_HEAD': 0.30,
    'FEAT_HIDDEN': 128,
    'FEAT_OUT': 64,
    'DROPOUT_FEAT': 0.30,
    'FEAT_NOISE_STD': 0.05,

    # ---- EMG augmentation (cross-person robustness) ----
    'AUGMENT': True,
    'AUG_NOISE_STD': 0.05,
    'AUG_CH_GAIN': 0.30,          # per-electrode log-gain jitter
    'AUG_GLOBAL_GAIN': 0.30,
    'AUG_CH_DROP_P': 0.10,        # lost electrode contact
    'AUG_ROTATE_P': 0.0,          # circular electrode shift (armband rotation)

    # ---- TRADES (Track A) / Gaussian smoothing (Track B) ----
    'TRADES_BETA': 6.0,
    'ADV_STEPS': 5,
    'ALPHA_RULE': 2.5,
    'ADV_WARMUP_EP': 5,
    'EPS_MAX': 0.05,
    'CLEAN_FRACTION': 0.30,
    'SMOOTH_SIGMA': 0.05,

    # ---- EMA / early stopping ----
    'USE_EMA': True,
    'EMA_DECAY': 0.99,
    'EARLY_STOP_PATIENCE': 12,
    'EARLY_STOP_MIN_EPOCHS': 10,
    'EARLY_STOP_MIN_DELTA': 1e-4,
    'USE_CLASS_WEIGHTED_CE': True,

    # ---- LightGBM feature model + blending ----
    'TRAIN_LGBM': True,
    'LGBM_PARAMS': {
        'n_estimators': 3000, 'learning_rate': 0.03, 'num_leaves': 31,
        'min_child_samples': 20, 'subsample': 0.8, 'subsample_freq': 1,
        'colsample_bytree': 0.6, 'reg_lambda': 1.0,
        'class_weight': 'balanced', 'verbose': -1,
    },
    'LGBM_EARLY_STOP': 150,

    # ---- test-time ----
    # Optional: rescale each test file's predicted class mix to the training
    # mix (about 50% rest, 10% per gesture). Off by default because it relies
    # on the assumption that each test file is one scripted session.
    'PRIOR_MATCHING': False,
}

if QUICK:
    CFG.update({'CLEAN_EPOCHS': 3, 'TRADES_EPOCHS': 2, 'SMOOTH_EPOCHS': 2,
                'ADV_STEPS': 2, 'ADV_WARMUP_EP': 1, 'LR_WARMUP_EPOCHS': 1,
                'EARLY_STOP_MIN_EPOCHS': 1, 'SEL_SUBSAMPLE': 800,
                'BORUTA_MAX_ITER': 8, 'CROP_STRIDE_SEC': 0.3})
    CFG['LGBM_PARAMS'] = dict(CFG['LGBM_PARAMS'], n_estimators=200)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything(CFG['SEED'])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
USE_DATA_PARALLEL = n_gpu >= 2
torch.backends.cudnn.benchmark = True
print('Device:', device, ' GPUs:', n_gpu, ' DataParallel:', USE_DATA_PARALLEL)
for _i in range(n_gpu):
    print('  GPU', _i, ':', torch.cuda.get_device_name(_i))

N_JOBS = max(1, min(4, os.cpu_count() or 1))

EMG_CH = [f'ch{i}' for i in range(1, 9)]
IMU_CH = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
SIG_CH = EMG_CH + IMU_CH
N_EMG = len(EMG_CH)
FS = CFG['FS']
WIN = int(round(CFG['WIN_SEC'] * FS))                     # 280 samples
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
                root = hit.parent.parent if hit.parent.name == 'train' else hit.parent
                return root
    raise FileNotFoundError('Competition data not found. Attach the dataset '
                            'or set AURA_DATA_DIR.')


DATA_ROOT = find_data_root()
if os.environ.get('AURA_OUT_DIR'):
    OUT_DIR = Path(os.environ['AURA_OUT_DIR'])
elif Path('/kaggle/working').exists():
    OUT_DIR = Path('/kaggle/working/aura_v1')
else:
    OUT_DIR = Path('aura_v1')
CACHE_DIR = OUT_DIR / 'cache'
MODEL_DIR = OUT_DIR / 'models'
for _d in (OUT_DIR, CACHE_DIR, MODEL_DIR):
    _d.mkdir(parents=True, exist_ok=True)

print('Data root :', DATA_ROOT)
print('Output dir:', OUT_DIR)
print('Quick mode:', QUICK)


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
            starts = list(range(0, L - WIN + 1, stride))
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
# CELL 4 - Feature selection: mRMR / Boruta / LightGBM vote (thesis cell 5)
# ----------------------------------------------------------------------------
# Fitted on the training sessions of each fold only (no leakage). The
# majority-vs-union decision uses a session-grouped split, because random
# splits leak overlapping crops of the same segment.
# ============================================================================

def select_features(F_tr, y_tr, g_tr, names, seed):
    if not CFG['FEATURE_SELECTION']:
        return list(names), {'policy': 'all'}
    Xz = StandardScaler().fit_transform(F_tr)
    n_sub = min(CFG['SEL_SUBSAMPLE'], len(Xz))
    if n_sub < len(Xz):
        sub, _ = train_test_split(np.arange(len(Xz)), train_size=n_sub,
                                  stratify=y_tr, random_state=seed)
    else:
        sub = np.arange(len(Xz))
    Xs, ys, gs = Xz[sub], y_tr[sub], g_tr[sub]
    K = max(15, int(CFG['SEL_K_FRAC'] * len(names)))
    voters = {}

    try:
        from mrmr import mrmr_classif
        voters['mrmr'] = list(mrmr_classif(X=pd.DataFrame(Xs, columns=names),
                                           y=pd.Series(ys), K=K,
                                           show_progress=False))
    except Exception as e:
        print('    [mRMR skipped]', type(e).__name__, str(e)[:80])

    try:
        from boruta import BorutaPy
        from sklearn.ensemble import RandomForestClassifier
        bor = BorutaPy(RandomForestClassifier(n_estimators=80, max_depth=8,
                                              class_weight='balanced',
                                              n_jobs=-1, random_state=seed),
                       n_estimators='auto', verbose=0, random_state=seed,
                       max_iter=CFG['BORUTA_MAX_ITER'])
        bor.fit(Xs, ys)
        voters['boruta'] = [f for f, k in zip(names, bor.support_) if k]
    except Exception as e:
        print('    [Boruta skipped]', type(e).__name__, str(e)[:80])

    gbm = lgb.LGBMClassifier(n_estimators=200, num_leaves=31, learning_rate=0.05,
                             class_weight='balanced', random_state=seed,
                             verbose=-1, n_jobs=-1)
    gbm.fit(Xs, ys)
    voters['lgbm'] = (pd.Series(gbm.booster_.feature_importance('gain'),
                                index=names).nlargest(K).index.tolist())

    votes = pd.Series(0, index=names)
    for sel in voters.values():
        votes[sel] += 1
    thr = 2 if len(voters) >= 3 else 1
    majority = votes[votes >= thr].index.tolist()
    union = votes[votes >= 1].index.tolist()
    if len(majority) < 10:
        majority = union

    policy = CFG['SEL_POLICY']
    acc = {}
    if policy == 'auto':
        if len(np.unique(gs)) >= 2:
            tr_i, te_i = next(GroupShuffleSplit(1, test_size=0.25,
                                                random_state=seed).split(Xs, ys, gs))
        else:
            tr_i, te_i = train_test_split(np.arange(len(ys)), test_size=0.25,
                                          stratify=ys, random_state=seed)
        for nm, fs in (('majority', majority), ('union', union)):
            idx = [names.index(f) for f in fs]
            m = lgb.LGBMClassifier(n_estimators=150, num_leaves=31,
                                   learning_rate=0.1, class_weight='balanced',
                                   random_state=seed, verbose=-1, n_jobs=-1)
            m.fit(Xs[tr_i][:, idx], ys[tr_i])
            acc[nm] = f1_score(ys[te_i], m.predict(Xs[te_i][:, idx]),
                               average='macro')
        policy = 'union' if acc['union'] > acc['majority'] + 0.005 else 'majority'
    chosen = union if policy == 'union' else majority
    meta = {'policy': policy, 'K': K, 'voters': {k: len(v) for k, v in voters.items()},
            'n_majority': len(majority), 'n_union': len(union),
            'eval_f1': acc, 'selected': chosen}
    return chosen, meta


# ============================================================================
# CELL 5 - Architecture: PhysioRobustNet (signal branch + feature branch)
# ============================================================================

class SEBlock(nn.Module):
    def __init__(self, ch, r=8):
        super().__init__()
        h = max(ch // r, 8)
        self.fc1 = nn.Conv1d(ch, h, 1)
        self.fc2 = nn.Conv1d(h, ch, 1)

    def forward(self, x):
        s = F.adaptive_avg_pool1d(x, 1)
        return x * torch.sigmoid(self.fc2(F.silu(self.fc1(s))))


class MorphologyExtractionModule(nn.Module):
    # three kernel sizes in parallel (75 / 45 / 125 ms at 200 Hz)
    def __init__(self, in_ch, width):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, width, 15, stride=2, padding=7)
        self.conv2 = nn.Conv1d(in_ch, width, 9, stride=2, padding=4)
        self.conv3 = nn.Conv1d(in_ch, width, 25, stride=2, padding=12)
        self.bn = nn.BatchNorm1d(width * 3)
        self.out_channels = width * 3

    def forward(self, x):
        a, b, c = self.conv1(x), self.conv2(x), self.conv3(x)
        n = min(a.shape[-1], b.shape[-1], c.shape[-1])
        return F.silu(self.bn(torch.cat([a[..., :n], b[..., :n], c[..., :n]], 1)))


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, dilation=1, dropout=0.0, use_se=True):
        super().__init__()
        self.c1 = nn.Conv1d(in_ch, out_ch, 3, stride=stride, padding=dilation,
                            dilation=dilation)
        self.b1 = nn.BatchNorm1d(out_ch)
        self.c2 = nn.Conv1d(out_ch, out_ch, 3, padding=dilation, dilation=dilation)
        self.b2 = nn.BatchNorm1d(out_ch)
        self.se = SEBlock(out_ch) if use_se else nn.Identity()
        self.drop = nn.Dropout(dropout)
        if stride != 1 or in_ch != out_ch:
            self.short = nn.Sequential(nn.Conv1d(in_ch, out_ch, 1, stride=stride),
                                       nn.BatchNorm1d(out_ch))
        else:
            self.short = nn.Identity()

    def forward(self, x):
        h = self.drop(F.silu(self.b1(self.c1(x))))
        h = self.se(self.b2(self.c2(h)))
        return F.silu(h + self.short(x))


class RhythmAnalysisModule(nn.Module):
    def __init__(self, in_ch, widths, dilations, dropout=0.1):
        super().__init__()
        stages, c = [], in_ch
        for w in widths:
            stages += [ResBlock(c, w, stride=2, dropout=dropout),
                       ResBlock(w, w, stride=1, dropout=dropout)]
            c = w
        self.stages = nn.Sequential(*stages)
        self.dilated = nn.Sequential(*[ResBlock(c, c, dilation=d, dropout=dropout)
                                       for d in dilations])
        self.out_channels = c

    def forward(self, x):
        return self.dilated(self.stages(x))


class FeatureBranch(nn.Module):
    def __init__(self, in_dim, hidden=128, out_dim=64, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, out_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)

    def forward(self, f):
        h = self.drop1(F.silu(self.bn1(self.fc1(f))))
        return F.silu(self.bn2(self.fc2(h)))


class UncertaintyAwareFusion(nn.Module):
    def __init__(self, in_dim, num_classes, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 256)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(dropout)
        self.fc_out = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.dropout1(F.silu(self.fc1(x)))
        x = self.dropout2(F.silu(self.fc2(x)))
        return self.fc_out(x)


class PhysioRobustNet(nn.Module):
    def __init__(self, num_classes=6, in_ch=14, width=32,
                 stage_widths=(64, 96, 128), dilations=(1, 2, 4),
                 dropout_tcn=0.15, dropout_head=0.3,
                 feat_dim=0, feat_hidden=128, feat_out=64, dropout_feat=0.3):
        super().__init__()
        self.morphology_module = MorphologyExtractionModule(in_ch, width)
        self.rhythm_module = RhythmAnalysisModule(self.morphology_module.out_channels,
                                                  tuple(stage_widths),
                                                  tuple(dilations), dropout_tcn)
        fuse_dim = self.rhythm_module.out_channels * 2
        if feat_dim > 0:
            self.feature_branch = FeatureBranch(feat_dim, feat_hidden, feat_out,
                                                dropout_feat)
            fuse_dim += feat_out
        else:
            self.feature_branch = None
        self.fusion_module = UncertaintyAwareFusion(fuse_dim, num_classes, dropout_head)

    def forward(self, x, f=None):
        h = self.rhythm_module(self.morphology_module(x))
        g = torch.cat([F.adaptive_avg_pool1d(h, 1).flatten(1),
                       F.adaptive_max_pool1d(h, 1).flatten(1)], 1)
        if self.feature_branch is not None and f is not None:
            g = torch.cat([g, self.feature_branch(f)], 1)
        return self.fusion_module(g)


def arch_args(num_classes, feat_dim):
    return {'num_classes': num_classes, 'in_ch': len(MODEL_CH),
            'width': CFG['WIDTH'], 'stage_widths': list(CFG['STAGE_WIDTHS']),
            'dilations': list(CFG['DILATIONS']), 'dropout_tcn': CFG['DROPOUT_TCN'],
            'dropout_head': CFG['DROPOUT_HEAD'], 'feat_dim': feat_dim,
            'feat_hidden': CFG['FEAT_HIDDEN'], 'feat_out': CFG['FEAT_OUT'],
            'dropout_feat': CFG['DROPOUT_FEAT']}


def build_model(num_classes, feat_dim):
    return PhysioRobustNet(**arch_args(num_classes, feat_dim)).to(device)


def maybe_wrap_dp(m):
    return nn.DataParallel(m) if USE_DATA_PARALLEL else m


def unwrap(m):
    return m.module if isinstance(m, nn.DataParallel) else m


# ============================================================================
# CELL 6 - EMA, augmentation, TRADES adversary, Gaussian noise, schedules
# ============================================================================

class EMA:
    def __init__(self, m, decay=0.99):
        self.decay = decay
        self.step = 0
        sd = unwrap(m).state_dict()
        self.shadow = {k: v.detach().clone().float() for k, v in sd.items()
                       if v.dtype.is_floating_point}
        self.buffers = {k: v.detach().clone() for k, v in sd.items()
                        if not v.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, m):
        d = min(self.decay, (1.0 + self.step) / (10.0 + self.step))
        self.step += 1
        for k, v in unwrap(m).state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(d).add_(v.detach().float(), alpha=1 - d)
            else:
                self.buffers[k].copy_(v.detach())

    def state_dict(self):
        return {**self.shadow, **self.buffers}


def augment(x):
    if not CFG['AUGMENT']:
        return x
    B = x.shape[0]
    x = x.clone()
    e = x[:, :N_EMG]                                    # view into x
    a = CFG['AUG_CH_GAIN']
    if a > 0:
        e.mul_(torch.exp(torch.empty(B, N_EMG, 1, device=x.device).uniform_(-a, a)))
    g = CFG['AUG_GLOBAL_GAIN']
    if g > 0:
        e.mul_(torch.exp(torch.empty(B, 1, 1, device=x.device).uniform_(-g, g)))
    p = CFG['AUG_ROTATE_P']
    if p > 0:
        r = torch.rand(B, device=x.device)
        for shift, m in ((1, r < p / 2), (-1, (r >= p / 2) & (r < p))):
            if m.any():
                e[m] = torch.roll(e[m], shifts=shift, dims=1)
    q = CFG['AUG_CH_DROP_P']
    if q > 0:
        e.mul_((torch.rand(B, N_EMG, 1, device=x.device) >= q).float())
    s = CFG['AUG_NOISE_STD']
    if s > 0:
        x = x + torch.randn_like(x) * s
    return x


def pgd_attack(m, x, f, y, eps, alpha, steps):
    if eps <= 0:
        return x.detach()
    was = m.training
    m.eval()
    xa = (x + torch.empty_like(x).uniform_(-eps, eps)).detach()
    with torch.enable_grad():
        for _ in range(steps):
            xa.requires_grad_(True)
            g = torch.autograd.grad(F.cross_entropy(m(xa, f), y), xa)[0]
            xa = torch.min(torch.max(xa.detach() + alpha * g.sign(), x - eps), x + eps)
    m.train(was)
    return xa.detach()


def trades_adv(m, x, f, eps, alpha, steps):
    # TRADES adversary (Zhang et al., ICML 2019): maximise KL(p_nat || p_adv)
    if eps <= 0:
        return x.detach()
    was = m.training
    m.eval()
    with torch.no_grad():
        p_nat = F.softmax(m(x, f), dim=1)
    xa = (x + 0.001 * torch.randn_like(x)).detach()
    kl = nn.KLDivLoss(reduction='sum')
    with torch.enable_grad():
        for _ in range(steps):
            xa.requires_grad_(True)
            g = torch.autograd.grad(kl(F.log_softmax(m(xa, f), dim=1), p_nat), xa)[0]
            xa = torch.min(torch.max(xa.detach() + alpha * g.sign(), x - eps), x + eps)
    m.train(was)
    return xa.detach()


def lr_at(epoch, total, warmup, base):
    if epoch < warmup:
        return base * (epoch + 1) / warmup
    p = (epoch - warmup) / max(total - warmup, 1)
    return 0.5 * base * (1 + math.cos(math.pi * p))


def eps_for_batch(epoch):
    ceiling = CFG['EPS_MAX'] * min(1.0, (epoch + 1) / max(CFG['ADV_WARMUP_EP'], 1))
    if np.random.rand() < CFG['CLEAN_FRACTION']:
        return 0.0
    return float(np.random.uniform(0.0, ceiling))


# ============================================================================
# CELL 7 - Evaluation helpers + one training stage (clean / trades / smoothing)
# ============================================================================

@torch.no_grad()
def predict_proba(m, X, Fe, bs=1024):
    m.eval()
    out = []
    for i in range(0, len(X), bs):
        out.append(F.softmax(m(X[i:i + bs], Fe[i:i + bs]).float(), 1).cpu().numpy())
    return np.concatenate(out, 0) if out else np.zeros((0, 1))


def macro_f1(y, P):
    return float(f1_score(y, P.argmax(1), average='macro'))


def train_stage(phase, epochs, tr, va, num_classes, feat_dim, class_w,
                init_state=None, tag=''):
    assert phase in ('clean', 'trades', 'smoothing')
    model = build_model(num_classes, feat_dim)
    if init_state is not None:
        model.load_state_dict(init_state)
    net = maybe_wrap_dp(model)
    ema = EMA(model, CFG['EMA_DECAY']) if CFG['USE_EMA'] else None
    eval_model = build_model(num_classes, feat_dim) if ema is not None else model

    ce = nn.CrossEntropyLoss(weight=class_w if CFG['USE_CLASS_WEIGHTED_CE'] else None,
                             label_smoothing=CFG['LABEL_SMOOTHING'])
    kl_loss = nn.KLDivLoss(reduction='batchmean')
    base_lr = CFG['LEARNING_RATE'] * (CFG['TRADES_LR_MULT'] if init_state is not None else 1.0)
    warmup = CFG['LR_WARMUP_EPOCHS'] if init_state is None else 1
    if CFG['OPTIMIZER'] == 'sgd':
        opt = torch.optim.SGD(net.parameters(), lr=base_lr, momentum=CFG['MOMENTUM'],
                              weight_decay=CFG['WEIGHT_DECAY'], nesterov=True)
    else:
        opt = torch.optim.AdamW(net.parameters(), lr=base_lr,
                                weight_decay=CFG['WEIGHT_DECAY'])

    n = len(tr['y'])
    bs = min(CFG['BATCH_SIZE'], n)
    steps = max(1, n // bs)
    hist = {'train_loss': [], 'train_acc': [], 'val_f1': [], 'lr': []}
    best_score, best_state, best_epoch, patience = -1.0, None, -1, 0
    t0 = time.time()

    for ep in range(epochs):
        lr = lr_at(ep, epochs, warmup, base_lr)
        for g in opt.param_groups:
            g['lr'] = lr
        net.train()
        perm = torch.randperm(n, device=device)
        loss_sum, correct, seen = 0.0, 0, 0
        for i in range(steps):
            idx = perm[i * bs:(i + 1) * bs]
            x = augment(tr['X'][idx])
            f = tr['F'][idx]
            if CFG['FEAT_NOISE_STD'] > 0:
                f = f + torch.randn_like(f) * CFG['FEAT_NOISE_STD']
            y = tr['y'][idx]

            if phase == 'clean':
                out = net(x, f)
                loss = ce(out, y)
            elif phase == 'trades':
                eps = eps_for_batch(ep)
                out = net(x, f)
                loss = ce(out, y)
                if eps > 0:
                    alpha = CFG['ALPHA_RULE'] * eps / CFG['ADV_STEPS']
                    x_adv = trades_adv(net, x, f, eps, alpha, CFG['ADV_STEPS'])
                    loss = loss + CFG['TRADES_BETA'] * kl_loss(
                        F.log_softmax(net(x_adv, f), dim=1), F.softmax(out, dim=1))
            else:
                out = net(x + torch.randn_like(x) * CFG['SMOOTH_SIGMA'], f)
                loss = ce(out, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), CFG['GRAD_CLIP'])
            opt.step()
            if ema is not None:
                ema.update(model)
            loss_sum += loss.item() * y.size(0)
            correct += out.argmax(1).eq(y).sum().item()
            seen += y.size(0)

        if ema is not None:
            eval_model.load_state_dict(ema.state_dict())
        val_f1 = macro_f1(va['y_np'], predict_proba(eval_model, va['X'], va['F']))
        hist['train_loss'].append(loss_sum / seen)
        hist['train_acc'].append(100.0 * correct / seen)
        hist['val_f1'].append(val_f1)
        hist['lr'].append(lr)

        improved = val_f1 > best_score + CFG['EARLY_STOP_MIN_DELTA']
        if improved:
            best_score, best_epoch, patience = val_f1, ep, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in eval_model.state_dict().items()}
        else:
            patience += 1
        if (ep + 1) % 5 == 0 or improved or ep == 0:
            print(f'    [{tag}{phase}] ep {ep + 1:3d}/{epochs} loss={loss_sum / seen:.4f} '
                  f'train_acc={100.0 * correct / seen:.1f} val_mF1={val_f1:.4f}'
                  f'{"  *" if improved else ""}  [{time.time() - t0:.0f}s]')
        if (ep + 1) >= CFG['EARLY_STOP_MIN_EPOCHS'] and patience >= CFG['EARLY_STOP_PATIENCE']:
            print(f'    [{tag}{phase}] early stop at ep {ep + 1}, best ep {best_epoch + 1} '
                  f'val_mF1={best_score:.4f}')
            break

    final = build_model(num_classes, feat_dim)
    final.load_state_dict(best_state)
    final.eval()
    del net, model, opt, ema, eval_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return final, hist, best_score


# ============================================================================
# CELL 8 - LightGBM on the selected features (fast, strong EMG baseline)
# ============================================================================

def train_lgbm(Ftr, ytr, Fva, yva, num_classes, seed):
    params = dict(CFG['LGBM_PARAMS'], random_state=seed, n_jobs=-1,
                  objective='multiclass', num_class=num_classes)
    m = lgb.LGBMClassifier(**params)
    m.fit(Ftr, ytr, eval_set=[(Fva, yva)], eval_metric='multi_logloss',
          callbacks=[lgb.early_stopping(CFG['LGBM_EARLY_STOP'], verbose=False)])
    return m


def match_priors(P, target, iters=50):
    # Sinkhorn-style rescaling so the mean prediction matches target proportions
    Q = P.copy() + 1e-9
    for _ in range(iters):
        Q *= (target / Q.mean(0))[None, :]
        Q /= Q.sum(1, keepdims=True)
    return Q


# ============================================================================
# CELL 9 - Main: data -> CV over sessions -> test inference -> submission
# ============================================================================

def main():
    t_start = time.time()
    train_files = sorted(DATA_ROOT.rglob('train_*.xlsx'), key=natural_key)
    test_files = sorted(DATA_ROOT.rglob('test_*.xlsx'), key=natural_key)
    subs = sorted(DATA_ROOT.rglob('sample_submission.csv'))
    print(f'train files: {len(train_files)}  test files: {len(test_files)}  '
          f'sample_submission: {bool(subs)}')
    assert train_files and test_files, 'train/test xlsx files not found'

    # ---------------- training windows ----------------
    print('\n[1] Reading + windowing training sessions ...')
    res = Parallel(n_jobs=N_JOBS)(delayed(process_train_file)(p) for p in train_files)
    res = [r for r in res if r is not None]
    X_tr = np.concatenate([r['X'] for r in res])
    lab_tr = np.concatenate([r['y'] for r in res])
    ref_tr = np.concatenate([r['ref'] for r in res])
    sess_tr = np.concatenate([[r['name']] * len(r['y']) for r in res])
    seg_tr = np.concatenate([r['seg'] for r in res])
    del res
    gc.collect()

    found = sorted(set(lab_tr))
    classes = [c for c in CLASS_ORDER if c in found] + [c for c in found if c not in CLASS_ORDER]
    C = len(classes)
    cls_idx = {c: i for i, c in enumerate(classes)}
    y_tr = np.array([cls_idx[c] for c in lab_tr], np.int64)
    sessions = sorted(np.unique(sess_tr), key=natural_key)
    print(f'  windows: {len(y_tr)} (reference/test-like: {ref_tr.sum()})  '
          f'sessions: {len(sessions)}  classes: {classes}')
    print('  reference windows per class:',
          {c: int(((y_tr == i) & ref_tr).sum()) for i, c in enumerate(classes)})

    X_tr = preprocess_windows(X_tr)
    X_tr = normalise_emg(X_tr, sess_tr, ref_tr)
    imu_scale = np.sqrt((X_tr[:, N_EMG:] ** 2).mean(axis=(0, 2))) + 1e-6
    X_tr[:, N_EMG:] /= imu_scale[None, :, None]
    np.clip(X_tr, -CFG['CLIP_SIGNAL'], CFG['CLIP_SIGNAL'], out=X_tr)

    # ---------------- test windows ----------------
    print('\n[2] Reading test actions ...')
    tres = Parallel(n_jobs=N_JOBS)(delayed(process_test_file)(p) for p in test_files)
    X_te = np.concatenate([r['X'] for r in tres])
    ids_te = np.concatenate([r['ids'] for r in tres])
    file_te = np.concatenate([[r['name']] * len(r['ids']) for r in tres])
    del tres
    X_te = preprocess_windows(X_te)
    X_te = normalise_emg(X_te, file_te, None)
    X_te[:, N_EMG:] /= imu_scale[None, :, None]
    np.clip(X_te, -CFG['CLIP_SIGNAL'], CFG['CLIP_SIGNAL'], out=X_te)
    print(f'  test actions: {len(ids_te)} from {len(np.unique(file_te))} files')

    # ---------------- features ----------------
    print('\n[3] Extracting features ...')
    F_tr_df = extract_features(X_tr)
    F_te_df = extract_features(X_te)
    feat_names = list(F_tr_df.columns)
    F_tr_all = F_tr_df.values
    F_te_all = F_te_df[feat_names].values
    print(f'  features: {len(feat_names)}')

    # ---------------- folds over sessions ----------------
    rng = np.random.RandomState(CFG['SEED'])
    n_folds = max(2, min(CFG['N_FOLDS'], len(sessions)))
    fold_of = {s: i % n_folds for i, s in enumerate(rng.permutation(sessions))}
    print(f'\n[4] Cross-validation: {n_folds} folds over {len(sessions)} sessions')

    stages = (['clean'] + (['trades'] if CFG['TRAIN_TRADES'] else [])
              + (['smoothing'] if CFG['TRAIN_SMOOTHING'] else [])) if CFG['TRAIN_NN'] else []
    oof = {k: np.zeros((len(y_tr), C), np.float32) for k in stages + ['nn', 'lgbm']}
    oof_mask = np.zeros(len(y_tr), bool)
    test_pred = {k: np.zeros((len(ids_te), C), np.float32) for k in ['nn', 'lgbm']}
    fold_log = []
    X_te_t = torch.from_numpy(X_te[:, MODEL_CH]).to(device)

    for k in range(n_folds):
        tk = time.time()
        seed_everything(CFG['SEED'] + k)
        ho_s = [s for s in sessions if fold_of[s] == k]
        rest_s = [s for s in sessions if fold_of[s] != k]
        n_iv = int(round(CFG['INNER_VAL_FRAC'] * len(rest_s)))
        if len(rest_s) >= 3:
            iv_s = list(np.random.RandomState(CFG['SEED'] + k).choice(
                rest_s, max(1, n_iv), replace=False))
        else:
            iv_s = list(ho_s)
            print('  [warn] too few sessions for an inner validation split; '
                  'early stopping uses the held-out fold (optimistic).')
        tr_s = [s for s in rest_s if s not in iv_s]
        m_tr = np.isin(sess_tr, tr_s)
        m_va = np.isin(sess_tr, iv_s) & ref_tr
        m_ho = np.isin(sess_tr, ho_s) & ref_tr
        print(f'\n=== Fold {k + 1}/{n_folds}  train={len(tr_s)} sess ({m_tr.sum()} win)  '
              f'inner-val={len(iv_s)} sess  held-out={ho_s}')

        # scale + select features (train sessions only)
        mu = F_tr_all[m_tr].mean(0)
        sd = F_tr_all[m_tr].std(0)
        sd[sd < 1e-8] = 1.0

        def scale(a):
            return np.clip(np.nan_to_num((a - mu) / sd), -10, 10).astype(np.float32)

        t_sel = time.time()
        sel, sel_meta = select_features(F_tr_all[m_tr], y_tr[m_tr], sess_tr[m_tr],
                                        feat_names, CFG['SEED'] + k)
        sel_idx = [feat_names.index(f) for f in sel]
        print(f'  feature selection: {sel_meta["policy"]} -> {len(sel)} features '
              f'{sel_meta.get("voters", "")} [{time.time() - t_sel:.0f}s]')
        Fs_tr = scale(F_tr_all[m_tr])[:, sel_idx]
        Fs_va = scale(F_tr_all[m_va])[:, sel_idx]
        Fs_ho = scale(F_tr_all[m_ho])[:, sel_idx]
        Fs_te = scale(F_te_all)[:, sel_idx]

        fold_rec = {'fold': k, 'heldout': ho_s, 'inner_val': iv_s,
                    'n_features': len(sel), 'selection': sel_meta}

        # ---- PhysioRobustNet stages ----
        if CFG['TRAIN_NN']:
            cw = compute_class_weight('balanced', classes=np.arange(C), y=y_tr[m_tr])
            cw_t = torch.tensor(cw, dtype=torch.float32, device=device)
            tr = {'X': torch.from_numpy(X_tr[m_tr][:, MODEL_CH]).to(device),
                  'F': torch.from_numpy(Fs_tr).to(device),
                  'y': torch.from_numpy(y_tr[m_tr]).to(device)}
            va = {'X': torch.from_numpy(X_tr[m_va][:, MODEL_CH]).to(device),
                  'F': torch.from_numpy(Fs_va).to(device), 'y_np': y_tr[m_va]}
            ho_X = torch.from_numpy(X_tr[m_ho][:, MODEL_CH]).to(device)
            ho_F = torch.from_numpy(Fs_ho).to(device)
            te_F = torch.from_numpy(Fs_te).to(device)

            models, scores = {}, {}
            m_clean, h_clean, s_clean = train_stage('clean', CFG['CLEAN_EPOCHS'], tr, va,
                                                    C, len(sel), cw_t, tag=f'f{k + 1} ')
            models['clean'], scores['clean'] = m_clean, s_clean
            init = {kk: v.to(device) for kk, v in m_clean.state_dict().items()}
            if CFG['TRAIN_TRADES']:
                models['trades'], _, scores['trades'] = train_stage(
                    'trades', CFG['TRADES_EPOCHS'], tr, va, C, len(sel), cw_t,
                    init_state=init, tag=f'f{k + 1} ')
            if CFG['TRAIN_SMOOTHING']:
                models['smoothing'], _, scores['smoothing'] = train_stage(
                    'smoothing', CFG['SMOOTH_EPOCHS'], tr, va, C, len(sel), cw_t,
                    init_state=init, tag=f'f{k + 1} ')
            for st, m in models.items():
                oof[st][m_ho] = predict_proba(m, ho_X, ho_F)
            final_st = (max(scores, key=scores.get) if CFG['FINAL_STAGE'] == 'best'
                        else CFG['FINAL_STAGE'])
            final_m = models[final_st]
            oof['nn'][m_ho] = oof[final_st][m_ho]
            test_pred['nn'] += predict_proba(final_m, X_te_t, te_F) / n_folds
            fold_rec.update({'stage_val_f1': scores, 'final_stage': final_st,
                             'heldout_f1_nn': macro_f1(y_tr[m_ho], oof['nn'][m_ho])})
            torch.save({'state_dict': final_m.state_dict(),
                        'arch': arch_args(C, len(sel)),
                        'class_names': classes, 'feat_names': sel,
                        'feat_mean': mu[sel_idx], 'feat_std': sd[sel_idx],
                        'imu_scale': imu_scale, 'model_channels': MODEL_CH,
                        'fs': FS, 'win': WIN, 'cfg': CFG, 'stage': final_st,
                        'fold': k, 'heldout_sessions': ho_s},
                       MODEL_DIR / f'physiorobustnet_emg_fold{k + 1}.pth')
            print(f'  NN stage val mF1: { {s: round(v, 4) for s, v in scores.items()} } '
                  f'-> {final_st}; held-out mF1 = {fold_rec["heldout_f1_nn"]:.4f}')
            del tr, va, ho_X, ho_F, te_F, models
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # ---- LightGBM ----
        if CFG['TRAIN_LGBM']:
            gbm = train_lgbm(Fs_tr, y_tr[m_tr], Fs_va, y_tr[m_va], C, CFG['SEED'] + k)
            oof['lgbm'][m_ho] = gbm.predict_proba(Fs_ho)
            test_pred['lgbm'] += gbm.predict_proba(Fs_te) / n_folds
            fold_rec['heldout_f1_lgbm'] = macro_f1(y_tr[m_ho], oof['lgbm'][m_ho])
            fold_rec['lgbm_best_iter'] = int(gbm.best_iteration_ or 0)
            gbm.booster_.save_model(str(MODEL_DIR / f'lgbm_fold{k + 1}.txt'))
            print(f'  LightGBM held-out mF1 = {fold_rec["heldout_f1_lgbm"]:.4f} '
                  f'(iters {fold_rec["lgbm_best_iter"]})')

        oof_mask |= m_ho
        fold_rec['seconds'] = round(time.time() - tk, 1)
        fold_log.append(fold_rec)

    # ---------------- out-of-fold report + blend ----------------
    print('\n[5] Out-of-fold results (one centre crop per guided segment, '
          'held-out sessions)')
    yo = y_tr[oof_mask]
    report = {'classes': classes, 'n_oof': int(oof_mask.sum()), 'folds': fold_log}
    for key, P in oof.items():
        if P[oof_mask].sum() > 0:
            report[f'oof_f1_{key}'] = macro_f1(yo, P[oof_mask])
            print(f'  {key:10s} macro-F1 = {report[f"oof_f1_{key}"]:.4f}')

    have_nn = CFG['TRAIN_NN']
    have_gbm = CFG['TRAIN_LGBM']
    if have_nn and have_gbm:
        grid_w = np.round(np.linspace(0, 1, 11), 2)
        scores_w = [macro_f1(yo, w * oof['nn'][oof_mask] + (1 - w) * oof['lgbm'][oof_mask])
                    for w in grid_w]
        w_best = float(grid_w[int(np.argmax(scores_w))])
        print('  blend weight (NN share) -> macro-F1:',
              {float(w): round(s, 4) for w, s in zip(grid_w, scores_w)})
    else:
        w_best = 1.0 if have_nn else 0.0
    P_oof = w_best * oof['nn'] + (1 - w_best) * oof['lgbm']
    report['blend_w_nn'] = w_best
    report['oof_f1_blend'] = macro_f1(yo, P_oof[oof_mask])
    print(f'  BLEND (w_nn={w_best}) macro-F1 = {report["oof_f1_blend"]:.4f}')
    print(classification_report(yo, P_oof[oof_mask].argmax(1), target_names=classes,
                                digits=4))
    print('  confusion matrix (rows = true):')
    print(pd.DataFrame(confusion_matrix(yo, P_oof[oof_mask].argmax(1)),
                       index=classes, columns=classes).to_string())
    per_sess = {s: macro_f1(y_tr[oof_mask & (sess_tr == s)],
                            P_oof[oof_mask & (sess_tr == s)])
                for s in sessions if (oof_mask & (sess_tr == s)).any()}
    report['per_session_f1'] = per_sess
    print('  per-session macro-F1:', {s: round(v, 3) for s, v in per_sess.items()})

    pd.DataFrame({'session': sess_tr[oof_mask], 'segment': seg_tr[oof_mask],
                  'true': [classes[i] for i in yo],
                  'pred': [classes[i] for i in P_oof[oof_mask].argmax(1)],
                  **{f'p_{c}': P_oof[oof_mask][:, i] for i, c in enumerate(classes)}}
                 ).to_csv(OUT_DIR / 'oof_predictions.csv', index=False)

    # ---------------- test predictions ----------------
    print('\n[6] Test inference + submission')
    P_te = w_best * test_pred['nn'] + (1 - w_best) * test_pred['lgbm']
    if CFG['PRIOR_MATCHING']:
        target = np.bincount(y_tr[ref_tr], minlength=C) / ref_tr.sum()
        for f in np.unique(file_te):
            m = file_te == f
            P_te[m] = match_priors(P_te[m], target)
        print('  prior matching applied per test file, target =',
              dict(zip(classes, np.round(target, 3))))
    pred = {i: classes[j] for i, j in zip(ids_te, P_te.argmax(1))}

    if subs:
        sample = pd.read_csv(subs[0])
        id_col = sample.columns[0]
        tgt_col = sample.columns[1] if sample.shape[1] > 1 else 'gesture'
        ids = sample[id_col].astype(str)
        missing = [i for i in ids if i not in pred]
        if missing:
            print(f'  [warn] {len(missing)} sample ids have no test window, '
                  f'filled with "{classes[0]}": {missing[:5]}')
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
