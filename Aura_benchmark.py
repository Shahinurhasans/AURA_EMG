# ============================================================================
# Aura_benchmark.py
# Aura: Calibration-Free EMG Challenge -- benchmark of well-known algorithms
# ----------------------------------------------------------------------------
# Runs the best-known EMG gesture-recognition / cross-subject transfer
# algorithms on EXACTLY the same windows and the same leave-sessions-out folds
# as Aura_version3.py, and prints one leaderboard of out-of-fold macro-F1
# (held-out people, one centre crop per guided segment = test-like).
#
# Algorithms (name -> idea -> reference)
#   hudgins_td4_lda     MAV/WL/ZC/SSC + LDA            Hudgins 1993; Englehart & Hudgins 2003
#   hudgins_td4_svm     same features + RBF-SVM
#   hudgins_td4_lda_ea  TD4 on Euclidean-aligned EMG    + He & Wu 2020 (EA)
#   tdpsd_lda           TD-PSD style spectral-moment    Khushaba et al. 2014
#                       descriptors + LDA                (limb-position invariant)
#   tdpsd_svm           same + RBF-SVM
#   full_lgbm           our ~300 v1-v3 features + LightGBM
#   full_lgbm_sessz     + per-session z-scored copies (v3 'both')
#   riemann_ts_lr       EMG covariance -> tangent space + logistic regression
#   riemann_ts_lr_rc    + per-person Riemannian re-centering  Zanini et al. 2018
#   riemann_mdm_rc      minimum-distance-to-mean on re-centered covariances
#   riemann_fbts_lr_rc  filter-bank (20-40 / 40-80 Hz) re-centered tangent space
#   cnn_raw             compact 1-D CNN on raw windows  Atzori 2016; Cote-Allard 2019
#   cnn_raw_adabn       same model + AdaBN per person   Du et al. 2017
#   cnn_ea              CNN on Euclidean-aligned EMG    He & Wu 2020
#   cnn_rotaug          CNN + barrel-shift (electrode rotation) augmentation
#                                                        Chamberland et al. 2023
#   ens_top3            average of the 3 best by out-of-fold macro-F1
#
# Every result is also scored after per-person class-prior matching (each
# scripted session is ~50% rest, ~10% per gesture).
#
# HOW TO RUN ON KAGGLE: add the competition data, GPU (T4 is plenty), Internet
# optional, paste this file into one cell. Prints the leaderboard and writes
#   /kaggle/working/aura_bench/benchmark_results.csv
#   /kaggle/working/aura_bench/per_session_f1.csv
#   /kaggle/working/aura_bench/submission_benchmark_best.csv
# Edit CFG['ALGORITHMS'] to run a subset. AURA_QUICK=1 = tiny smoke run.
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
from sklearn.metrics import f1_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from joblib import Parallel, delayed
import lightgbm as lgb

import torch
import torch.nn as nn
import torch.nn.functional as F

QUICK = os.environ.get('AURA_QUICK', '0') == '1'

CFG = {
    'SEED': 42,
    # windows / signal -- identical to Aura_version3.py
    'FS': 200, 'WIN_SEC': 1.4, 'CROP_STRIDE_SEC': 0.10,
    'TRIM_GESTURE_START_SEC': 0.4, 'MIN_SEG_SEC': 1.20,
    'BANDPASS': (20.0, 80.0), 'NOTCH_HZ': (50.0, 60.0), 'NOTCH_Q': 30.0,
    'NORM_MODE': 'session', 'USE_IMU': True, 'CLIP_SIGNAL': 20.0,
    # validation -- identical folds to Aura_version3.py
    'N_FOLDS': 5, 'INNER_VAL_FRAC': 0.15,

    'ALGORITHMS': [
        'hudgins_td4_lda', 'hudgins_td4_svm', 'hudgins_td4_lda_ea',
        'tdpsd_lda', 'tdpsd_svm',
        'full_lgbm', 'full_lgbm_sessz',
        'riemann_ts_lr', 'riemann_ts_lr_rc', 'riemann_mdm_rc', 'riemann_fbts_lr_rc',
        'cnn_raw', 'cnn_ea', 'cnn_rotaug',      # cnn_raw also yields cnn_raw_adabn
    ],
    'SVM_MAX_TRAIN': 5000,                       # RBF-SVM subsample per fold
    'COV_SHRINK': 1e-3,
    'FB_BANDS': ((20.0, 40.0), (40.0, 80.0)),

    # CNN
    'CNN_EPOCHS': 40, 'CNN_PATIENCE': 8, 'CNN_BATCH': 256, 'CNN_LR': 1e-3,
    'CNN_WD': 1e-2, 'CNN_WIDTH': 64, 'CNN_DROPOUT': 0.3, 'LABEL_SMOOTHING': 0.1,
    'AUG_NOISE_STD': 0.05, 'AUG_CH_GAIN': 0.3, 'AUG_GLOBAL_GAIN': 0.3,

    # LightGBM (same as v3)
    'LGBM_PARAMS': {
        'n_estimators': 3000, 'learning_rate': 0.03, 'num_leaves': 31,
        'min_child_samples': 20, 'subsample': 0.8, 'subsample_freq': 1,
        'colsample_bytree': 0.6, 'reg_lambda': 1.0,
        'class_weight': 'balanced', 'verbose': -1,
    },
    'LGBM_EARLY_STOP': 150,
}
if QUICK:
    CFG.update({'CNN_EPOCHS': 2, 'CNN_PATIENCE': 1, 'CROP_STRIDE_SEC': 0.3,
                'SVM_MAX_TRAIN': 500})
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
    OUT_DIR = Path('/kaggle/working/aura_bench')
else:
    OUT_DIR = Path('aura_bench')
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
# CELL 4 - Classic EMG feature sets
# ============================================================================

def hudgins_td4(X):
    # Hudgins et al. 1993 / Englehart & Hudgins 2003: MAV, WL, ZC, SSC
    e = X[:, :N_EMG].astype(np.float64)
    d = np.diff(e, axis=-1)
    thr = 0.01 * e.std(-1, keepdims=True)                 # small noise dead-zone
    mav = np.abs(e).mean(-1)
    wl = np.abs(d).mean(-1)
    zc = (((e[..., :-1] * e[..., 1:]) < 0) & (np.abs(d) >= thr)).mean(-1)
    ssc = (((d[..., :-1] * d[..., 1:]) < 0) &
           ((np.abs(d[..., :-1]) >= thr) | (np.abs(d[..., 1:]) >= thr))).mean(-1)
    return np.concatenate([np.log(mav + 1e-8), np.log(wl + 1e-8), zc, ssc], 1).astype(np.float32)


def tdpsd(X, lam=0.1):
    # TD-PSD style descriptors (Khushaba et al. 2014): spectral moments from the
    # time domain (m0, m2, m4 with power normalisation lambda = 0.1), sparseness,
    # irregularity factor, waveform-length ratio; computed on the signal and on
    # its log-power version, plus their orientation (-2ab / (a^2 + b^2)).
    eps = 1e-8
    e = X[:, :N_EMG].astype(np.float64)

    def six(x):
        d1 = np.diff(x, axis=-1)
        d2 = np.diff(d1, axis=-1)
        m0 = np.sqrt((x ** 2).sum(-1)) ** lam / lam
        m2 = np.sqrt((d1 ** 2).sum(-1)) ** lam / lam
        m4 = np.sqrt((d2 ** 2).sum(-1)) ** lam / lam
        f1 = np.log(m0 + eps)
        f2 = np.log(np.abs(m0 - m2) + eps)
        f3 = np.log(np.abs(m0 - m4) + eps)
        f4 = np.log(m0 / (np.sqrt(np.abs(m0 - m2) * np.abs(m0 - m4)) + eps) + eps)
        f5 = np.log(m2 / (np.sqrt(m0 * m4) + eps) + eps)
        f6 = np.log(np.abs(d1).sum(-1) / (np.abs(d2).sum(-1) + eps) + eps)
        return np.stack([f1, f2, f3, f4, f5, f6], -1)          # (N, 8, 6)

    a = six(e)
    b = six(np.log(e ** 2 + eps))
    orient = -2.0 * a * b / (a ** 2 + b ** 2 + eps)
    return np.concatenate([a.reshape(len(X), -1), orient.reshape(len(X), -1)], 1).astype(np.float32)


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


def train_cnn(Xtr, ytr, Xva, yva, n_cls, rot=False, tag=''):
    model = SimpleEMGCNN(Xtr.shape[1], n_cls, CFG['CNN_WIDTH'], CFG['CNN_DROPOUT']).to(device)
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
# CELL 8 - Algorithms. Each returns {name: (P_heldout, P_test)} for one fold.
# D holds every precomputed representation; f holds the fold masks.
# ============================================================================

D = {}


def _xy(rep, f):
    A, B = D[rep]
    return A[f['tr']], D['y'][f['tr']], A[f['ho']], B


def algo_feature_clf(rep, kind):
    def run(f):
        Xtr, ytr, Xho, Xte = _xy(rep, f)
        if kind == 'lda':
            clf = fit_lda(Xtr, ytr, D['C'])
        elif kind == 'svm':
            clf = fit_svm(Xtr, ytr, CFG['SEED'] + f['k'])
        else:
            clf = fit_lr(Xtr, ytr)
        return full_proba(clf, Xho, D['C']), full_proba(clf, Xte, D['C'])
    return run


def algo_lgbm(rep):
    def run(f):
        A, B = D[rep]
        params = dict(CFG['LGBM_PARAMS'], random_state=CFG['SEED'] + f['k'], n_jobs=-1,
                      objective='multiclass', num_class=D['C'])
        m = lgb.LGBMClassifier(**params)
        m.fit(A[f['tr']], D['y'][f['tr']], eval_set=[(A[f['va']], D['y'][f['va']])],
              eval_metric='multi_logloss',
              callbacks=[lgb.early_stopping(CFG['LGBM_EARLY_STOP'], verbose=False)])
        return full_proba(m, A[f['ho']], D['C']), full_proba(m, B, D['C'])
    return run


def algo_riemann_ts_lr(f):
    # tangent space at the TRAINING mean (no per-person alignment)
    Ctr, Cte = D['cov']
    W = inv_sqrt(le_mean(Ctr[f['tr']]))
    clf = fit_lr(tangent(Ctr[f['tr']], W), D['y'][f['tr']])
    return (full_proba(clf, tangent(Ctr[f['ho']], W), D['C']),
            full_proba(clf, tangent(Cte, W), D['C']))


def algo_mdm_rc(f):
    Ttr, Tte = D['ts_rc']
    return (mdm_proba(Ttr[f['tr']], D['y'][f['tr']], Ttr[f['ho']], D['C']),
            mdm_proba(Ttr[f['tr']], D['y'][f['tr']], Tte, D['C']))


def algo_cnn(rep, rot=False, adabn_name=None):
    def run(f):
        A, B = D[rep]
        name = 'cnn_rotaug' if rot else rep
        model = train_cnn(A[f['tr']], D['y'][f['tr']], A[f['va']], D['y'][f['va']],
                          D['C'], rot=rot, tag=f'{name} fold {f["k"] + 1}')
        Xho = torch.from_numpy(A[f['ho']]).to(device)
        Xte = torch.from_numpy(B).to(device)
        out = {name: (cnn_proba(model, Xho), cnn_proba(model, Xte))}
        if adabn_name:
            out[adabn_name] = (cnn_proba_adabn(model, Xho, D['sess'][f['ho']]),
                               cnn_proba_adabn(model, Xte, D['file_te']))
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return out
    return run


ALGOS = {
    'hudgins_td4_lda':    ('td4', algo_feature_clf('td4', 'lda')),
    'hudgins_td4_svm':    ('td4', algo_feature_clf('td4', 'svm')),
    'hudgins_td4_lda_ea': ('td4_ea', algo_feature_clf('td4_ea', 'lda')),
    'tdpsd_lda':          ('tdpsd', algo_feature_clf('tdpsd', 'lda')),
    'tdpsd_svm':          ('tdpsd', algo_feature_clf('tdpsd', 'svm')),
    'full_lgbm':          ('full', algo_lgbm('full')),
    'full_lgbm_sessz':    ('full_sz', algo_lgbm('full_sz')),
    'riemann_ts_lr':      ('cov', algo_riemann_ts_lr),
    'riemann_ts_lr_rc':   ('ts_rc', algo_feature_clf('ts_rc', 'lr')),
    'riemann_mdm_rc':     ('ts_rc', algo_mdm_rc),
    'riemann_fbts_lr_rc': ('fbts_rc', algo_feature_clf('fbts_rc', 'lr')),
    'cnn_raw':            ('cnn_raw', algo_cnn('cnn_raw', adabn_name='cnn_raw_adabn')),
    'cnn_ea':             ('cnn_ea', algo_cnn('cnn_ea')),
    'cnn_rotaug':         ('cnn_raw', algo_cnn('cnn_raw', rot=True)),
}


def build_representations(needed, X_tr, X_te, sess_tr, ref_tr, file_te):
    # everything here is label-free and per window / per person, so it is
    # computed once for all folds (no leakage)
    t0 = time.time()
    if 'td4' in needed:
        D['td4'] = (hudgins_td4(X_tr), hudgins_td4(X_te))
    if needed & {'td4_ea', 'cnn_ea'}:
        Xe_tr = euclidean_align(X_tr, sess_tr, ref_tr)
        Xe_te = euclidean_align(X_te, file_te, None)
        if 'td4_ea' in needed:
            D['td4_ea'] = (hudgins_td4(Xe_tr), hudgins_td4(Xe_te))
        if 'cnn_ea' in needed:
            D['cnn_ea'] = (np.clip(Xe_tr[:, MODEL_CH], -CFG['CLIP_SIGNAL'], CFG['CLIP_SIGNAL']),
                           np.clip(Xe_te[:, MODEL_CH], -CFG['CLIP_SIGNAL'], CFG['CLIP_SIGNAL']))
        del Xe_tr, Xe_te
    if 'tdpsd' in needed:
        D['tdpsd'] = (tdpsd(X_tr), tdpsd(X_te))
    if needed & {'full', 'full_sz'}:
        Ftr = extract_features(X_tr).values
        Fte = extract_features(X_te).values
        D['full'] = (Ftr, Fte)
        if 'full_sz' in needed:
            D['full_sz'] = (np.concatenate([Ftr, session_standardize(Ftr, sess_tr, ref_tr)], 1),
                            np.concatenate([Fte, session_standardize(Fte, file_te, None)], 1))
    if needed & {'cov', 'ts_rc'}:
        Ctr = window_covs(X_tr[:, :N_EMG])
        Cte = window_covs(X_te[:, :N_EMG])
        D['cov'] = (Ctr, Cte)
        if 'ts_rc' in needed:
            D['ts_rc'] = (recentered_tangent(Ctr, sess_tr, ref_tr),
                          recentered_tangent(Cte, file_te, None))
    if 'fbts_rc' in needed:
        parts_tr, parts_te = [], []
        for lo, hi in CFG['FB_BANDS']:
            Ctr = window_covs(bandpass_windows(X_tr, lo, hi))
            Cte = window_covs(bandpass_windows(X_te, lo, hi))
            parts_tr.append(recentered_tangent(Ctr, sess_tr, ref_tr))
            parts_te.append(recentered_tangent(Cte, file_te, None))
        D['fbts_rc'] = (np.concatenate(parts_tr, 1), np.concatenate(parts_te, 1))
    if 'cnn_raw' in needed:
        D['cnn_raw'] = (X_tr[:, MODEL_CH], X_te[:, MODEL_CH])
    print(f'  representations: {sorted(k for k in D if k not in ("y", "C", "sess", "file_te"))} '
          f'[{time.time() - t0:.0f}s]')


# ============================================================================
# CELL 9 - Main: windows -> representations -> CV over sessions -> leaderboard
# ============================================================================

def main():
    t_start = time.time()
    train_files = sorted(DATA_ROOT.rglob('train_*.xlsx'), key=natural_key)
    test_files = sorted(DATA_ROOT.rglob('test_*.xlsx'), key=natural_key)
    subs = sorted(DATA_ROOT.rglob('sample_submission.csv'))
    print(f'train files: {len(train_files)}  test files: {len(test_files)}')

    print('\n[1] Windows (same preprocessing as Aura_version3.py) ...')
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
    algos = [a for a in CFG['ALGORITHMS'] if a in ALGOS]
    D.update({'y': y_tr, 'C': C, 'sess': sess_tr, 'file_te': file_te})
    build_representations({ALGOS[a][0] for a in algos} | ({'cov'} if 'riemann_ts_lr' in algos else set()),
                          X_tr, X_te, sess_tr, ref_tr, file_te)
    del X_tr, X_te
    gc.collect()

    # same session folds as Aura_version3.py
    rng = np.random.RandomState(CFG['SEED'])
    n_folds = max(2, min(CFG['N_FOLDS'], len(sessions)))
    fold_of = {s: i % n_folds for i, s in enumerate(rng.permutation(sessions))}
    print(f'\n[3] {len(algos)} algorithms x {n_folds} folds (held-out = new people)')

    results, timing = {}, {}
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
        for a in algos:
            t0 = time.time()
            out = ALGOS[a][1](f)
            if not isinstance(out, dict):
                out = {a: out}
            for name, (p_ho, p_te) in out.items():
                r = results.setdefault(name, {'oof': np.zeros((len(y_tr), C), np.float32),
                                              'test': np.zeros((len(ids_te), C), np.float32)})
                r['oof'][f['ho']] = p_ho
                r['test'] += p_te / n_folds
                timing[name] = timing.get(name, 0.0) + time.time() - t0
                print(f'  {name:20s} held-out mF1 {macro_f1(y_tr[f["ho"]], p_ho):.4f}  '
                      f'[{time.time() - t0:.0f}s]')

    # ---------------- leaderboard ----------------
    yo, g_oof = y_tr[oof_mask], sess_tr[oof_mask]
    target = np.bincount(y_tr[ref_tr], minlength=C) / ref_tr.sum()

    def score(P):
        Pm = match_priors_by_group(P, g_oof, target)
        return (macro_f1(yo, P), macro_f1(yo, Pm),
                float(accuracy_score(yo, P.argmax(1))))

    rows = []
    for name, r in results.items():
        f1, f1pm, acc = score(r['oof'][oof_mask])
        rows.append({'algorithm': name, 'macro_f1': f1, 'macro_f1_prior_matched': f1pm,
                     'accuracy': acc, 'minutes': round(timing.get(name, 0) / 60, 2)})
    board = pd.DataFrame(rows)
    board['best_f1'] = board[['macro_f1', 'macro_f1_prior_matched']].max(1)
    board = board.sort_values('best_f1', ascending=False).reset_index(drop=True)
    top3 = board['algorithm'].head(3).tolist()
    ens = {'oof': np.mean([results[n]['oof'] for n in top3], 0),
           'test': np.mean([results[n]['test'] for n in top3], 0)}
    results['ens_top3'] = ens
    f1, f1pm, acc = score(ens['oof'][oof_mask])
    board = pd.concat([board, pd.DataFrame([{
        'algorithm': 'ens_top3 (' + '+'.join(top3) + ')', 'macro_f1': f1,
        'macro_f1_prior_matched': f1pm, 'accuracy': acc, 'minutes': 0.0,
        'best_f1': max(f1, f1pm)}])]).sort_values('best_f1', ascending=False).reset_index(drop=True)

    print('\n' + '=' * 78)
    print('LEADERBOARD  out-of-fold macro-F1 on held-out people '
          f'({oof_mask.sum()} test-like windows, {len(sessions)} sessions)')
    print('=' * 78)
    print(board.to_string(index=False, float_format=lambda v: f'{v:.4f}'))
    board.to_csv(OUT_DIR / 'benchmark_results.csv', index=False)

    per_sess = pd.DataFrame({name: {str(s): macro_f1(yo[g_oof == s], r['oof'][oof_mask][g_oof == s])
                                    for s in sessions if (g_oof == s).any()}
                             for name, r in results.items()})
    per_sess.to_csv(OUT_DIR / 'per_session_f1.csv')
    print('\nper-session macro-F1 of the top 5 (rows = held-out person):')
    top5 = [n.split(' (')[0] for n in board['algorithm'].head(5)]
    print(per_sess[[c for c in top5 if c in per_sess]].round(3).to_string())

    # ---------------- submission from the best entry ----------------
    best_name = board['algorithm'].iloc[0].split(' (')[0]
    use_pm = board['macro_f1_prior_matched'].iloc[0] > board['macro_f1'].iloc[0]
    P_te = results[best_name]['test']
    if use_pm:
        P_te = match_priors_by_group(P_te, file_te, target)
    pred = {i: classes[j] for i, j in zip(ids_te, P_te.argmax(1))}
    if subs:
        sample = pd.read_csv(subs[0])
        id_col = sample.columns[0]
        tgt_col = sample.columns[1] if sample.shape[1] > 1 else 'gesture'
        sub = pd.DataFrame({id_col: sample[id_col],
                            tgt_col: [pred.get(str(i), classes[0]) for i in sample[id_col]]})
    else:
        sub = pd.DataFrame({'action_id': ids_te, 'gesture': [pred[i] for i in ids_te]})
    sub.to_csv(OUT_DIR / 'submission_benchmark_best.csv', index=False)
    np.savez_compressed(OUT_DIR / 'benchmark_probs.npz', classes=np.array(classes),
                        ids_te=ids_te, file_te=file_te, oof_mask=oof_mask, y=y_tr,
                        sess=sess_tr, **{f'oof__{n}': r['oof'] for n, r in results.items()},
                        **{f'test__{n}': r['test'] for n, r in results.items()})
    print(f'\nBest: {best_name} (prior matching: {use_pm}) -> '
          f'{OUT_DIR / "submission_benchmark_best.csv"}')
    print('predicted class counts:', sub.iloc[:, 1].value_counts().to_dict())
    print(f'Done in {(time.time() - t_start) / 60:.1f} min.')
    return board


if __name__ == '__main__':
    leaderboard = main()
