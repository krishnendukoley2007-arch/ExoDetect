import streamlit as st
import lightkurve as lk
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.ensemble import RandomForestClassifier
import io
import os
import re
import time
import datetime
import joblib
import json
import plotly.graph_objects as go
import database as db

# ════════════════════════════════════════════════════════════
# PAGE CONFIG
# ════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="ExoDetect — BAH2026 PS7",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ════════════════════════════════════════════════════════════
# MATPLOTLIB GLOBAL SETTINGS — sharp, high-DPI plots
# ════════════════════════════════════════════════════════════
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'figure.facecolor': '#0a1220',
    'axes.facecolor': '#0a1220',
    'axes.edgecolor': '#1e3050',
    'axes.labelcolor': '#7090b8',
    'xtick.color': '#7090b8',
    'ytick.color': '#7090b8',
    'text.color': '#ddeeff',
    'grid.color': '#1e3050',
    'grid.alpha': 0.4,
    'lines.linewidth': 1.5,
    'font.family': 'DejaVu Sans',
    'axes.titlesize': 11,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
})

# ════════════════════════════════════════════════════════════
# CSS
# ════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Space+Grotesk:wght@400;600;700&display=swap');

.stApp {
    background: radial-gradient(ellipse at top, #0d1b2e 0%, #060b14 60%, #0a0e1a 100%);
    font-family: 'Inter', 'Space Grotesk', sans-serif;
}

@keyframes twinkle {
    0%   { opacity: 0.5; }
    50%  { opacity: 1.0; }
    100% { opacity: 0.6; }
}
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes pulse-ring {
    0%,100% { box-shadow: 0 0 0 0 rgba(100,160,255,0.0); }
    50%      { box-shadow: 0 0 0 6px rgba(100,160,255,0.15); }
}
@keyframes shimmer {
    0%   { background-position: -200% center; }
    100% { background-position: 200% center; }
}
@keyframes orbit {
    from { transform: rotate(0deg) translateX(26px) rotate(0deg); }
    to   { transform: rotate(360deg) translateX(26px) rotate(-360deg); }
}
@keyframes glow-border {
    0%,100% { border-color: #1e3050; box-shadow: 0 0 0 rgba(74,158,255,0); }
    50%      { border-color: #2a5a9a; box-shadow: 0 0 22px rgba(74,158,255,0.18); }
}
@keyframes float-y {
    0%,100% { transform: translateY(0); }
    50%      { transform: translateY(-6px); }
}

/* ── Database Explorer animated cards ── */
.db-stat-card {
    background: linear-gradient(135deg, #0f1828, #0c1422);
    border: 1px solid #1e3050;
    border-radius: 14px;
    padding: 18px 20px;
    text-align: center;
    animation: fadeInUp 0.5s ease-out both, glow-border 5s ease-in-out infinite;
    transition: transform 0.25s;
}
.db-stat-card:hover { transform: translateY(-4px) scale(1.02); }
.db-stat-card .db-num {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.1rem; font-weight: 700; color: #e0ecff;
    background: linear-gradient(90deg, #4a9eff, #aad4ff, #4a9eff);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shimmer 4s linear infinite;
}
.db-stat-card .db-label {
    color: #6080a8; font-size: 0.75rem; text-transform: uppercase;
    letter-spacing: 0.08em; margin-top: 4px;
}
.db-stat-card:nth-child(1) { animation-delay: 0.00s; }
.db-stat-card:nth-child(2) { animation-delay: 0.08s; }
.db-stat-card:nth-child(3) { animation-delay: 0.16s; }
.db-stat-card:nth-child(4) { animation-delay: 0.24s; }

/* ── Animated starfield background ── */
@keyframes drift {
    from { background-position: 0 0, 40px 60px, 130px 270px; }
    to   { background-position: -550px 0, -510px 60px, -420px 270px; }
}
@keyframes shooting {
    0%   { transform: translate(110vw, -10vh) rotate(-35deg); opacity: 0; }
    3%   { opacity: 1; }
    9%   { transform: translate(45vw, 45vh) rotate(-35deg); opacity: 0; }
    100% { transform: translate(45vw, 45vh) rotate(-35deg); opacity: 0; }
}
/* Starfield lives on the main content container only — never touches
   the sidebar's stacking context (which made the sidebar disappear). */
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: absolute; inset: 0; z-index: 0; pointer-events: none;
    background-image:
        radial-gradient(1px 1px at 25px 45px, #cfe4ff 50%, transparent 100%),
        radial-gradient(1.5px 1.5px at 120px 160px, #9dc4f5 50%, transparent 100%),
        radial-gradient(1px 1px at 230px 90px, #ffffff 50%, transparent 100%);
    background-size: 280px 280px, 340px 340px, 300px 300px;
    animation: drift 120s linear infinite, twinkle 5s ease-in-out infinite;
    opacity: 0.5;
}
[data-testid="stAppViewContainer"]::after {
    content: '';
    position: absolute; top: 0; left: 0; z-index: 0; pointer-events: none;
    width: 130px; height: 2px; border-radius: 2px;
    background: linear-gradient(90deg, rgba(255,255,255,0.9), transparent);
    box-shadow: 0 0 8px rgba(160,200,255,0.8);
    animation: shooting 14s linear infinite;
}

.orbit-wrap { display:inline-block; position:relative; width:64px; height:64px; animation: float-y 4s ease-in-out infinite; }
.orbit-star { position:absolute; inset:0; display:flex; align-items:center; justify-content:center; font-size:2rem; }
.orbit-planet { position:absolute; top:50%; left:50%; margin:-8px; font-size:1rem; animation: orbit 5s linear infinite; }

.hero-banner {
    background: linear-gradient(135deg, #0a1a35 0%, #112040 50%, #0a1a35 100%);
    border: 1px solid #1e3a60;
    border-radius: 20px;
    padding: 36px 28px 28px;
    text-align: center;
    margin-bottom: 28px;
    animation: fadeInUp 0.6s ease-out;
    position: relative;
    overflow: hidden;
}
.hero-banner::after {
    content: '';
    position: absolute;
    inset: 0;
    background:
        radial-gradient(circle at 20% 50%, rgba(80,140,255,0.07) 0%, transparent 55%),
        radial-gradient(circle at 80% 30%, rgba(255,120,50,0.05) 0%, transparent 45%);
    pointer-events: none;
}
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 3rem;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: -0.03em;
    margin: 0 0 10px;
    text-shadow: 0 0 60px rgba(80,150,255,0.4);
}
.hero-sub {
    color: #8ab0d8;
    font-size: 1rem;
    margin: 0;
    line-height: 1.7;
}

/* AI Insight card */
.ai-insight {
    background: linear-gradient(135deg, #0f1e38, #0a1628);
    border: 1px solid #2a5080;
    border-left: 4px solid #4a9eff;
    border-radius: 12px;
    padding: 18px 22px;
    margin: 16px 0;
    animation: fadeInUp 0.4s ease-out;
}
.ai-insight-header {
    color: #4a9eff;
    font-weight: 700;
    font-size: 0.85rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.ai-insight-text {
    color: #c4d8f0;
    font-size: 0.95rem;
    line-height: 1.65;
    margin: 0;
}

.verdict-box {
    border-radius: 14px;
    padding: 20px 22px;
    text-align: center;
    margin: 16px 0;
    animation: fadeInUp 0.4s ease-out;
}

.stat-card {
    background: linear-gradient(135deg, #0f1828, #0c1422);
    border: 1px solid #1e3050;
    border-radius: 12px;
    padding: 16px 18px;
    margin-bottom: 10px;
    transition: border-color 0.25s, transform 0.2s;
}
.stat-card:hover {
    border-color: #3060a0;
    transform: translateY(-2px);
}

.report-card {
    background: linear-gradient(135deg, #0f1828, #0c1422);
    border: 1px solid #1e3050;
    border-radius: 14px;
    padding: 22px 26px;
    margin-bottom: 16px;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #07101e 0%, #091220 100%);
    border-right: 1px solid #182538;
}

div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #0f1828, #0c1422);
    border: 1px solid #1e3050;
    border-radius: 12px;
    padding: 14px 16px;
    transition: border-color 0.25s, transform 0.2s;
    animation: pulse-ring 4s infinite;
}
div[data-testid="stMetric"]:hover {
    border-color: #3060a0;
    transform: translateY(-2px);
}
div[data-testid="stMetricLabel"] { color: #6080a8 !important; font-size: 0.78rem !important; text-transform: uppercase; letter-spacing: 0.05em; }
div[data-testid="stMetricValue"] { color: #e0ecff !important; font-weight: 700 !important; font-size: 1.4rem !important; }

.stButton > button {
    border-radius: 10px;
    font-weight: 600;
    border: 1px solid #243050;
    transition: all 0.2s;
    color: #b8ccec;
    background: #0f1828;
    font-size: 0.9rem;
}
.stButton > button:hover {
    border-color: #4a80c0;
    transform: translateY(-1px);
    box-shadow: 0 4px 16px rgba(60,120,200,0.2);
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #1a5eff 0%, #0040cc 100%);
    border: none;
    color: white;
    font-size: 1rem;
    letter-spacing: 0.02em;
}
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 6px 20px rgba(30,80,255,0.35);
    transform: translateY(-2px);
}

h1, h2, h3 { color: #d8eaff; font-family: 'Space Grotesk', sans-serif; }
p, .stMarkdown, label { color: #8aa8cc; }
hr { border-color: #182538 !important; }

[data-testid="stExpander"] {
    background: #0c1422;
    border: 1px solid #182538;
    border-radius: 10px;
}
div[data-baseweb="tab-list"] {
    background: #091220 !important;
    border-radius: 10px;
    padding: 4px;
}
div[data-baseweb="tab"] {
    border-radius: 8px !important;
    color: #6080a8 !important;
}
div[aria-selected="true"] {
    background: #1a2f50 !important;
    color: #90b8e8 !important;
}
.stSelectbox > div, .stMultiSelect > div {
    background: #0c1422 !important;
    border-color: #1e3050 !important;
    border-radius: 10px !important;
}
.stTextInput > div > div {
    background: #0c1422 !important;
    border-color: #1e3050 !important;
    border-radius: 10px !important;
    color: #c4d8f0 !important;
}
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# SESSION STATE
# ════════════════════════════════════════════════════════════
for key, default in [
    ("tic_id", "261136679"),
    ("history", []),
    ("last_result", None),
    ("results_cache", {}),
    ("compare_results", []),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# Feature list — feature_cols.pkl (saved by train_model_v10.py) is canonical;
# features_config.py is the shared fallback so the list can never go stale.
try:
    FEATURE_COLS = joblib.load("feature_cols.pkl")
except Exception:
    from features_config import FEATURE_COLS

# Stellar params — loaded once, used for live feature enrichment
@st.cache_data
def load_stellar_params():
    if os.path.exists("stellar_params.csv"):
        sp = pd.read_csv("stellar_params.csv")
        sp["tic_id"] = sp["tic_id"].astype(str)
        return sp.set_index("tic_id")
    return pd.DataFrame()

stellar_db = load_stellar_params()

def get_stellar_features(tic_id):
    """Return stellar param dict for a TIC ID, with safe fallbacks."""
    defaults = {"Teff": 5778.0, "rad": 1.0, "mass": 1.0,
                "logg": 4.44, "Tmag": 10.0, "contratio": 0.0}
    if stellar_db.empty:
        return defaults
    key = str(tic_id)
    if key in stellar_db.index:
        row = stellar_db.loc[key].to_dict()
        for k, v in defaults.items():
            if k in row and (pd.isna(row[k]) or row[k] is None):
                row[k] = v
        return row
    return defaults

QUICK_STARS = {
    "Pi Mensae c — Sub-Neptune": "261136679",
    "WASP-126 b — Hot Jupiter":  "25155310",
    "TIC 441075486 — Binary (FP)": "441075486",
}


# ════════════════════════════════════════════════════════════
# HELPER — safe numpy extraction from masked/astropy arrays
# ════════════════════════════════════════════════════════════
def to_arr(x):
    """Convert any astropy/masked array to clean plain float64 numpy array."""
    if hasattr(x, "filled"):
        x = x.filled(np.nan)
    arr = np.asarray(x, dtype=np.float64)
    return arr


def _thin(x_arr, y_arr, max_pts=40000):
    """Stride-subsample big arrays so interactive Plotly charts stay snappy."""
    if len(x_arr) > max_pts:
        step = int(np.ceil(len(x_arr) / max_pts))
        return x_arr[::step], y_arr[::step]
    return x_arr, y_arr


def clean_series(x_arr, y_arr):
    """Return only finite pairs — removes NaN/inf spikes before plotting."""
    x = to_arr(x_arr)
    y = to_arr(y_arr)
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


# ════════════════════════════════════════════════════════════
# DATASET POOL
# ════════════════════════════════════════════════════════════
@st.cache_data
def load_dataset_pool():
    if not os.path.exists("features_dataset.csv"):
        return pd.DataFrame()
    fname = "features_dataset_clean.csv" if os.path.exists("features_dataset_clean.csv") else "features_dataset.csv"
    df = pd.read_csv(fname)
    # Live analysis is TESS-only — Kepler (KIC) stars have no TESS pipeline,
    # so keep them out of the star pickers.
    df = df[~df["tic_id"].astype(str).str.startswith("KIC")]
    df["display_name"] = df.apply(
        lambda r: (
            f"{db.star_label(r['tic_id'])}  |  "
            f"{'🪐 Planet' if r['label']=='planet' else '⭐ False Positive'}  |  "
            f"SNR {r['snr']:.1f}  |  "
            f"depth {r['depth']*100:.3f}%"
        ), axis=1
    )
    return df

dataset_pool = load_dataset_pool()


# Cached wrappers for CSV-backed db loaders (files change rarely; 5-min TTL).
# SQLite-backed loaders (db_stats, fetch_analyses, load_frontier_results) stay
# uncached on purpose — they must reflect writes made during the session.
@st.cache_data(ttl=300)
def cached_catalog():
    return db.load_catalog()

@st.cache_data(ttl=300)
def cached_frontier(n=None):
    return db.load_frontier(n)

@st.cache_data(ttl=300)
def cached_sky_targets():
    return db.load_sky_targets()

@st.cache_data(ttl=300)
def cached_holdout():
    return db.load_holdout()


# ════════════════════════════════════════════════════════════
# CLASSIFIER
# ════════════════════════════════════════════════════════════
@st.cache_resource
def load_classifier():
    if os.path.exists("exoplanet_classifier.pkl"):
        clf = joblib.load("exoplanet_classifier.pkl")
        le  = joblib.load("label_encoder.pkl") if os.path.exists("label_encoder.pkl") else None
        n   = str(len(dataset_pool)) if not dataset_pool.empty else "?"
        if os.path.exists("model_metrics.json"):
            import json as _json
            try:
                with open("model_metrics.json") as _f:
                    m = _json.load(_f)
            except Exception as e:
                print(f"WARNING: model_metrics.json unreadable: {e}")
                m = {}
            desc = (f"XGBoost {m.get('version','v10')} (calibrated) — {n} stars | "
                    f"{len(FEATURE_COLS)} features | "
                    f"held-out accuracy {m.get('holdout_accuracy','?')}% | "
                    f"ROC-AUC {m.get('holdout_roc_auc','?')} | "
                    f"recall {m.get('holdout_recall','?')}%")
        else:
            desc = f"XGBoost (calibrated) — {n} stars | {len(FEATURE_COLS)} features"
        return clf, le, desc, True
    # No model = hard stop. A synthetic fallback would fabricate confident
    # predictions — worse than an honest error.
    st.error("❌ Model file `exoplanet_classifier.pkl` not found — deployment is "
             "incomplete. Run `python train_model_v10.py` to create it.")
    st.stop()

clf_model, label_encoder, model_source, is_real_model = load_classifier()

@st.cache_resource
def load_raw_booster():
    """Raw (uncalibrated) XGBoost model — used only for SHAP explanations."""
    if os.path.exists("xgb_raw.pkl"):
        try:
            return joblib.load("xgb_raw.pkl")
        except Exception:
            return None
    return None

xgb_raw = load_raw_booster()

def compute_shap(fv):
    """Per-feature SHAP contributions toward 'planet' for one feature vector."""
    if xgb_raw is None:
        return None
    try:
        import xgboost as _xgb
        contribs = xgb_raw.get_booster().predict(
            _xgb.DMatrix(np.array(fv), feature_names=None), pred_contribs=True)[0]
        # last element is the bias term; positive log-odds push toward class 1
        planet_is_1 = (label_encoder is not None and
                       list(label_encoder.classes_).index("planet") == 1)
        sign = 1.0 if planet_is_1 else -1.0
        return {col: sign * float(c) for col, c in zip(FEATURE_COLS, contribs[:-1])}
    except Exception:
        return None

def clf_predict(fv):
    """Unified predict — handles XGBoost (int labels) and legacy RF (string labels)."""
    proba_arr = clf_model.predict_proba(fv)[0]
    if label_encoder is not None:
        classes_str = label_encoder.classes_
    else:
        classes_str = clf_model.classes_
    proba_dict = dict(zip(classes_str, proba_arr))
    pred_label = classes_str[int(np.argmax(proba_arr))]
    return pred_label, proba_dict


# ════════════════════════════════════════════════════════════
# FIX 1 — ROBUST DOWNLOAD with retry + exponential backoff
# ════════════════════════════════════════════════════════════
def download_with_retry(tic_id, max_sectors, max_retries=3):
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            search = lk.search_lightcurve(
                f'TIC {tic_id}', mission='TESS', author='SPOC', exptime=120
            )
            if len(search) == 0:
                return None, None, None, "No TESS/SPOC data found. The star may exist but have no 2-min cadence data."
            n_sectors = min(max_sectors, len(search))
            lc_raw = search[:n_sectors].download_all().stitch()
            return lc_raw, n_sectors, len(search), None
        except Exception as e:
            last_err = e
            err_msg = str(e).lower()
            is_transient = any(s in err_msg for s in [
                "connection", "reset", "timed out", "timeout",
                "10054", "remote", "broken pipe", "temporarily", "aborted"
            ])
            if attempt < max_retries and is_transient:
                time.sleep(2 ** attempt)
                continue
            return None, None, None, f"Download failed after {attempt} attempt(s): {e}"
    return None, None, None, f"Download failed: {last_err}"


# ════════════════════════════════════════════════════════════
# FIX 2 — BOUNDED BLS PERIODOGRAM
# Root cause: n_periods = (period_max - period_min) / 0.001
# For 1–20 day range = 19,000 × 28 durations × lightkurve's
# internal frequency_factor=10 → 5.3M grid points → crash.
#
# Fix: use frequency_factor=1 (disables internal expansion),
# compute our own sensibly-sized grid (≤2000 periods), and
# fall back with coarser grids if still needed.
# ════════════════════════════════════════════════════════════
def compute_bls_periodogram(lc, period_min, period_max, max_retries=4):
    # Sensible starting grid — small enough to never blow up
    n_periods   = 2000
    n_durations = 14

    last_err = None
    for attempt in range(max_retries):
        periods   = np.linspace(period_min, period_max, n_periods)
        durations = np.linspace(0.02, 0.25, n_durations)

        try:
            # We pass an explicit `period` grid, so lightkurve's own grid is
            # never used — BUT its size pre-check still computes npoints from
            # frequency_factor and the baseline, and raises "too large" for
            # long-baseline stars. frequency_factor is otherwise unused when
            # `period` is given, so a huge value defuses the bogus check.
            pg = lc.to_periodogram(
                method='bls',
                period=periods,
                duration=durations,
                frequency_factor=1e6
            )
            return pg, None
        except Exception as e:
            last_err = e
            err_msg  = str(e).lower()
            # Only retry on "too large" errors
            if "too large" in err_msg or "periodogram" in err_msg:
                n_periods   = max(n_periods // 2, 300)
                n_durations = max(n_durations // 2, 4)
                continue
            # Other errors (value errors, etc.) — bail immediately
            return None, f"Periodogram error: {e}"

    return None, f"Periodogram still too large after {max_retries} retries: {last_err}"


# ════════════════════════════════════════════════════════════
# CORE PIPELINE
# ════════════════════════════════════════════════════════════
def run_pipeline(tic_id, max_sectors, period_min, period_max):
    result = {"tic_id": str(tic_id), "error": None}

    lc_raw, n_sectors, n_available, dl_err = download_with_retry(tic_id, max_sectors)
    if dl_err:
        result["error"] = dl_err
        return result
    result["n_sectors"]   = n_sectors
    result["n_available"] = n_available

    try:
        lc = lc_raw.normalize().flatten(window_length=401).remove_outliers(sigma=4)

        pg, pg_err = compute_bls_periodogram(lc, period_min, period_max)
        if pg_err:
            result["error"] = pg_err
            return result

        best_period   = pg.period_at_max_power
        best_power    = pg.max_power
        t0            = pg.transit_time_at_max_power
        duration_best = pg.duration_at_max_power

        # ── FIX 3: duration_best.value is in DAYS (not phase fraction).
        # half_width must be in PHASE units (fraction of period) for folding.
        # Compute properly:  half_width_phase = (duration_days / period_days) / 2
        duration_days = float(duration_best.to('d').value)
        period_days   = float(best_period.to('d').value)
        half_width    = (duration_days / period_days) / 2   # dimensionless phase fraction

        folded = lc.fold(period=best_period, epoch_time=t0)
        binned = folded.bin(time_bin_size=0.001)

        # ── Extract to clean plain arrays immediately ──
        phase_vals = to_arr(folded.time.value)
        flux_vals  = to_arr(folded.flux.value)
        bin_phase  = to_arr(binned.time.value)
        bin_flux   = to_arr(binned.flux.value)

        # ── FIX 4: Strip NaN/inf from binned arrays (prevents spike artifacts) ──
        bin_mask   = np.isfinite(bin_phase) & np.isfinite(bin_flux)
        bin_phase  = bin_phase[bin_mask]
        bin_flux   = bin_flux[bin_mask]

        # Transit statistics
        in_transit  = np.abs(phase_vals) < half_width * 1.3
        out_transit = (np.abs(phase_vals) > half_width * 3) & (np.abs(phase_vals) < 0.45)
        secondary   = np.abs(np.abs(phase_vals) - 0.5) < half_width * 1.3

        if np.sum(in_transit) > 5 and np.sum(out_transit) > 5:
            baseline    = float(np.nanmedian(flux_vals[out_transit]))
            transit_med = float(np.nanmedian(flux_vals[in_transit]))
            depth       = float(baseline - transit_med)
            noise       = float(np.nanstd(flux_vals[out_transit]))
            n_in        = int(np.sum(in_transit))
            snr         = (depth / noise) * np.sqrt(n_in) if noise > 0 else 0.0
        else:
            baseline = 1.0; depth = 0.0; snr = 0.0

        sec_depth = 0.0
        if np.sum(secondary) > 5:
            sec_depth = float(baseline - np.nanmedian(flux_vals[secondary]))

        sec_ratio      = (sec_depth / depth) if depth > 0 else 0.0
        duration_hours = duration_days * 24
        R_earth        = np.sqrt(abs(depth)) * 1.1 * 109.076

        # Odd-even check
        t_arr    = to_arr(lc.time.value)
        f_arr    = to_arr(lc.flux.value)
        cycle_num    = np.round((t_arr - t0.value) / period_days).astype(int)
        phase_global = ((t_arr - t0.value) % period_days) / period_days
        phase_global[phase_global > 0.5] -= 1
        in_g  = np.abs(phase_global) < half_width * 1.3
        o_d   = float(np.nanmedian(f_arr[in_g & (cycle_num % 2 == 1)])) if np.sum(in_g & (cycle_num%2==1)) > 2 else 1.0
        e_d   = float(np.nanmedian(f_arr[in_g & (cycle_num % 2 == 0)])) if np.sum(in_g & (cycle_num%2==0)) > 2 else 1.0
        odd_even_diff = abs(o_d - e_d)

        # ── 5 new physics-informed features (v9) ──
        edge_mask   = (np.abs(phase_vals) < half_width*1.3) & (np.abs(phase_vals) > half_width*0.7)
        center_mask = np.abs(phase_vals) < half_width*0.3
        edge_depth_v   = float(baseline - np.nanmedian(flux_vals[edge_mask]))   if np.sum(edge_mask)   > 2 else depth
        center_depth_v = float(baseline - np.nanmedian(flux_vals[center_mask])) if np.sum(center_mask) > 2 else depth
        transit_shape  = (center_depth_v / edge_depth_v) if abs(edge_depth_v) > 1e-8 else 1.0
        dur_period_ratio = duration_hours / (period_days * 24)

        per_transit_depths = []
        unique_cycles = np.unique(cycle_num[in_g])
        for c in unique_cycles:
            mc = in_g & (cycle_num == c)
            if np.sum(mc) > 1:
                per_transit_depths.append(baseline - float(np.nanmedian(f_arr[mc])))
        depth_consistency = (float(np.nanstd(per_transit_depths) /
                             (np.nanmean(per_transit_depths) + 1e-8))
                             if len(per_transit_depths) > 2 else 0.0)

        left_mask  = (phase_vals < -half_width*0.3) & (phase_vals > -half_width*1.3)
        right_mask = (phase_vals >  half_width*0.3) & (phase_vals <  half_width*1.3)
        left_d  = float(baseline - np.nanmedian(flux_vals[left_mask]))  if np.sum(left_mask)  > 2 else depth
        right_d = float(baseline - np.nanmedian(flux_vals[right_mask])) if np.sum(right_mask) > 2 else depth
        ingress_egress_asymmetry = abs(left_d - right_d) / (depth + 1e-8)
        odd_even_ratio = odd_even_diff / (depth + 1e-8)

        # ── Stellar params lookup (v9) ──
        sp = get_stellar_features(tic_id)
        result["stellar_params_found"] = (not stellar_db.empty and
                                          str(tic_id) in stellar_db.index)
        Teff_v     = sp.get("Teff",     5778.0)
        rad_v      = sp.get("rad",      1.0)
        mass_v     = sp.get("mass",     1.0)
        logg_v     = sp.get("logg",     4.44)
        Tmag_v     = sp.get("Tmag",     10.0)
        contratio_v= sp.get("contratio",0.0)

        # ── Engineered features via the SHARED implementation in
        # features_config (one-row DataFrame) — cannot drift from training ──
        from features_config import add_engineered_features, ENGINEERED_FEATURES
        feature_map = {
            "depth": abs(depth), "snr": max(snr, 0),
            "sec_ratio": abs(sec_ratio), "duration_hours": duration_hours,
            "bls_power": float(best_power), "odd_even_diff": odd_even_diff,
            "transit_shape": transit_shape, "dur_period_ratio": dur_period_ratio,
            "depth_consistency": depth_consistency,
            "ingress_egress_asymmetry": ingress_egress_asymmetry,
            "odd_even_ratio": odd_even_ratio,
            "period_days": period_days,
            "Teff": Teff_v, "rad": rad_v, "mass": mass_v,
            "logg": logg_v, "Tmag": Tmag_v, "contratio": contratio_v,
            "mission": 0.0,  # live analysis is always TESS (v10.2+ models)
        }
        _eng = add_engineered_features(pd.DataFrame([feature_map])).iloc[0]
        for _c in ENGINEERED_FEATURES:
            feature_map[_c] = float(_eng[_c])
        planet_radius_est = feature_map["planet_radius_est"]
        duration_expected_ratio = feature_map["duration_expected_ratio"]
        fv_values = [feature_map.get(col, 0.0) for col in FEATURE_COLS]
        fv = np.array([fv_values])

        # ML classification — calibrated probability vs decision threshold
        pred_label, proba_dict = clf_predict(fv)
        planet_proba = float(proba_dict.get("planet", 0.0))
        threshold = float(globals().get("planet_threshold", 0.5))
        shap_contribs = compute_shap(fv)

        if float(best_power) < 50 or snr < 3:
            ml_class   = "Weak / No Signal"
            confidence = 100.0
            proba_dict = {"Weak / No Signal": 1.0}
        elif planet_proba >= threshold:
            ml_class   = "Exoplanet Candidate"
            confidence = planet_proba * 100
        else:
            ml_class   = "Eclipsing Binary / False Positive"
            confidence = (1 - planet_proba) * 100

        is_eb = sec_depth > depth * 0.4 and sec_depth > 0.0008
        if float(best_power) > 300 and snr > 7 and not is_eb:
            rule_verdict = ("Hot Jupiter"      if depth > 0.005   else
                            "Sub-Neptune"       if depth > 0.0005  else
                            "Super-Earth"       if depth > 0.00005 else
                            "Small Rocky Planet")
        elif is_eb and float(best_power) > 300:
            rule_verdict = "Eclipsing Binary"
        elif snr > 5:
            rule_verdict = "Planet Candidate"
        else:
            rule_verdict = "Uncertain"

        result.update({
            "lc": lc, "pg": pg,
            "folded": folded, "binned": binned,
            "bin_phase": bin_phase, "bin_flux": bin_flux,
            "best_period": best_period, "best_power": best_power,
            "duration_best": duration_best, "half_width": half_width,
            "duration_days": duration_days, "period_days": period_days,
            "depth": depth, "snr": snr, "sec_depth": sec_depth,
            "sec_ratio": sec_ratio, "duration_hours": duration_hours,
            "odd_even_diff": odd_even_diff,
            "transit_shape": transit_shape, "dur_period_ratio": dur_period_ratio,
            "depth_consistency": depth_consistency,
            "ingress_egress_asymmetry": ingress_egress_asymmetry,
            "odd_even_ratio": odd_even_ratio,
            "stellar_Teff": Teff_v, "stellar_rad": rad_v, "stellar_mass": mass_v,
            "stellar_logg": logg_v, "stellar_Tmag": Tmag_v,
            "R_planet_earth": R_earth, "baseline": baseline,
            "ml_class": ml_class, "ml_confidence": confidence,
            "ml_proba": proba_dict, "rule_verdict": rule_verdict,
            "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
            "phase_vals": phase_vals, "flux_vals": flux_vals,
            "t_arr": t_arr, "f_arr": f_arr, "cycle_num": cycle_num,
            "phase_global": phase_global, "o_d": o_d, "e_d": e_d,
            "t0": t0.value,
            "planet_proba": planet_proba,
            "decision_threshold": threshold,
            "search_pmin": float(period_min), "search_pmax": float(period_max),
            "shap": shap_contribs,
        })
    except Exception as e:
        import traceback
        result["error"] = f"Pipeline error: {e}\n{traceback.format_exc()}"
    return result


def get_or_run(tic_id, max_sectors, period_min, period_max, force=False):
    key = str(tic_id)
    if not force and key in st.session_state.results_cache:
        return st.session_state.results_cache[key]
    result = run_pipeline(tic_id, max_sectors, period_min, period_max)
    if not result.get("error"):
        st.session_state.results_cache[key] = result
        st.session_state.history.append({
            "TIC ID":         key,
            "Period (days)":  round(float(result["period_days"]), 4),
            "Depth (%)":      round(result["depth"]*100, 5),
            "SNR":            round(result["snr"], 2),
            "Radius (R⊕)":   round(result["R_planet_earth"], 2),
            "ML Verdict":     result["ml_class"],
            "Confidence (%)": round(result["ml_confidence"], 1),
            "Rule-based":     result["rule_verdict"],
            "Sectors":        result["n_sectors"],
            "Time":           result["timestamp"],
        })
        # ── Persist to SQLite so results survive app restarts ──
        try:
            db.save_analysis(result, insight=generate_ai_insight(result))
        except Exception as e:
            # DB write failure must never break the analysis flow — but log it
            print(f"WARNING: DB save failed for TIC {key}: {e}")
    return result


# ════════════════════════════════════════════════════════════
# AI INTERPRETATION — natural language summary
# ════════════════════════════════════════════════════════════
def generate_ai_insight(r):
    """Generate a plain-English interpretation of the analysis results."""
    depth      = r["depth"]
    snr        = r["snr"]
    period     = r["period_days"]
    duration   = r["duration_hours"]
    radius     = r["R_planet_earth"]
    ml_class   = r["ml_class"]
    rule       = r["rule_verdict"]
    sec_ratio  = r["sec_ratio"]
    bls_power  = float(r["best_power"])
    oe_diff    = r["odd_even_diff"]
    confidence = r["ml_confidence"]

    lines = []

    # ── Signal quality ──
    if bls_power > 500 and snr > 10:
        lines.append(f"The BLS search found a **very strong periodic signal** (power {bls_power:.0f}, SNR {snr:.1f}) — this is a high-confidence transit detection, not noise.")
    elif bls_power > 100 and snr > 5:
        lines.append(f"A **clear periodic dip** was detected (BLS power {bls_power:.0f}, SNR {snr:.1f}). The signal is statistically significant.")
    else:
        lines.append(f"The signal is **weak** (BLS power {bls_power:.0f}, SNR {snr:.1f}). This may be noise, a grazing transit, or the period range doesn't match the real orbit.")

    # ── Classification reasoning ──
    if ml_class == "Exoplanet Candidate":
        lines.append(
            f"The ML classifier labels this a **planet candidate** with {confidence:.0f}% confidence. "
            f"The transit depth of {depth*100:.4f}% corresponds to a planet roughly **{radius:.1f}× Earth's radius** "
            f"— {'about the size of Neptune' if 3 < radius < 7 else 'a Hot Jupiter' if radius > 7 else 'a Super-Earth or sub-Neptune' if radius > 1.5 else 'potentially Earth-like'}. "
            f"It orbits every **{period:.3f} days** with a transit lasting **{duration:.1f} hours**."
        )
    elif "Binary" in ml_class or "False" in ml_class:
        reasons = []
        if sec_ratio > 0.4:
            reasons.append(f"a secondary eclipse at half-phase ({sec_ratio*100:.0f}% as deep as the primary — typical of eclipsing binaries)")
        if oe_diff > 0.002:
            reasons.append(f"odd-even depth alternation (every other transit is {oe_diff*100:.3f}% different — a hallmark of two stars eclipsing each other)")
        if depth > 0.01:
            reasons.append(f"an unusually deep transit ({depth*100:.2f}%) — real planets rarely block more than 1% of stellar light")
        reason_str = ", and ".join(reasons) if reasons else "anomalous photometric signature"
        lines.append(
            f"The classifier flags this as likely a **false positive** ({confidence:.0f}% confidence) due to: {reason_str}. "
            f"This is most likely an **eclipsing binary star system**, not a planet."
        )
    else:
        lines.append("The signal is too weak to classify confidently. Try stacking more sectors or widening the period search range.")

    # ── Sanity cross-check ──
    if ml_class == "Exoplanet Candidate" and rule != ml_class.replace("Exoplanet Candidate", "Planet Candidate") and rule not in ["Hot Jupiter","Sub-Neptune","Super-Earth","Small Rocky Planet","Planet Candidate"]:
        lines.append(f"⚠️ Note: the rule-based check returns **'{rule}'** — the two methods disagree, so treat this result with caution and check the transit shape manually.")
    elif ml_class == "Exoplanet Candidate":
        lines.append(f"The independent rule-based check agrees: **{rule}**. Both methods point to the same conclusion.")

    return " ".join(lines)


# ════════════════════════════════════════════════════════════
# PDF GENERATOR
# ════════════════════════════════════════════════════════════
def generate_pdf(result):
    """Vetting Report v2 — TFOP-memo-style per-star PDF.

    Page 1: verdict banner, calibrated probability, vetting checklist,
            cross-match, ephemeris, stellar host parameters, AI insight.
    Page 2: SHAP explanation (why the model decided this).
    Page 3: light curve / periodogram / phase fold.
    """
    import textwrap
    r = result
    buf = io.BytesIO()
    xm = None
    try:
        xm = db.lookup_toi(r["tic_id"])
    except Exception:
        pass

    with PdfPages(buf) as pdf:
        # ── PAGE 1 — vetting memo ──────────────────────────────
        fig = plt.figure(figsize=(8.5, 11), facecolor='white')
        ax0 = fig.add_axes([0, 0, 1, 1]); ax0.axis('off')
        fig.text(0.5, 0.965, "ExoDetect — Candidate Vetting Report",
                 ha='center', fontsize=17, fontweight='bold', color='#1a2a4a')
        fig.text(0.5, 0.942, f"TIC {r['tic_id']}  |  Team OrbitX2026  |  "
                 "BAH2026 PS7  |  Jadavpur University",
                 ha='center', fontsize=9, color='#4a6a9a')

        # Verdict banner
        is_planet = r['ml_class'] == "Exoplanet Candidate"
        vc = '#1a7a2a' if is_planet else ('#9a1010' if 'Binary' in r['ml_class']
                                          or 'False' in r['ml_class'] else '#555577')
        ax0.add_patch(plt.Rectangle((0.06, 0.875), 0.88, 0.045,
                                    facecolor=vc, alpha=0.12, edgecolor=vc,
                                    linewidth=1.5, transform=fig.transFigure))
        fig.text(0.5, 0.897, f"VERDICT: {r['ml_class'].upper()}   —   "
                 f"calibrated planet probability {r.get('planet_proba', 0)*100:.1f}%  "
                 f"(threshold {r.get('decision_threshold', 0.5):.2f})",
                 ha='center', fontsize=11, fontweight='bold', color=vc)

        # Signal parameters (left column)
        sig = [
            "SIGNAL PARAMETERS",
            "-" * 34,
            f"Orbital period    {r['period_days']:>12.4f} d",
            f"Transit duration  {r['duration_hours']:>12.2f} h",
            f"Transit depth     {r['depth']*100:>12.5f} %",
            f"Planet radius     {r['R_planet_earth']:>12.2f} R_Earth",
            f"Detection SNR     {r['snr']:>12.2f}",
            f"BLS power         {float(r['best_power']):>12.1f}",
            f"Sectors stacked   {r['n_sectors']:>7} / {r['n_available']}",
        ]
        fig.text(0.08, 0.855, "\n".join(sig), fontsize=8.5, va='top',
                 family='monospace', color='#1a2a4a')

        # Stellar host (right column)
        stel = [
            "HOST STAR (TIC catalog)",
            "-" * 34,
            f"Eff. temperature  {r.get('stellar_Teff', 0):>10.0f} K",
            f"Radius            {r.get('stellar_rad', 0):>10.2f} R_Sun",
            f"Mass              {r.get('stellar_mass', 0):>10.2f} M_Sun",
            f"log g             {r.get('stellar_logg', 0):>10.2f}",
            f"TESS magnitude    {r.get('stellar_Tmag', 0):>10.2f}",
        ]
        fig.text(0.54, 0.855, "\n".join(stel), fontsize=8.5, va='top',
                 family='monospace', color='#1a2a4a')

        # Vetting checklist — the classic false-positive tests
        sec_ok   = abs(r.get('sec_ratio', 0)) < 0.4
        oe_ok    = r.get('odd_even_ratio', 0) < 0.3
        shape_ok = r.get('transit_shape', 1.0) > 0.8
        asym_ok  = r.get('ingress_egress_asymmetry', 0) < 1.0
        cons_ok  = r.get('depth_consistency', 0) < 1.0
        def mark(ok): return "[PASS]" if ok else "[FLAG]"
        checks = [
            "VETTING CHECKLIST",
            "-" * 76,
            f"{mark(sec_ok)}  Secondary eclipse   ratio {abs(r.get('sec_ratio',0)):.3f}"
            f"  (eclipsing binaries show a 2nd dip at phase 0.5)",
            f"{mark(oe_ok)}  Odd vs even depth   ratio {r.get('odd_even_ratio',0):.3f}"
            f"  (unequal alternating depths reveal a binary at 2x period)",
            f"{mark(shape_ok)}  Transit shape       U/V  {r.get('transit_shape',1.0):.3f}"
            f"  (planets are flat-bottomed U; grazing binaries are V)",
            f"{mark(asym_ok)}  Ingress/egress sym  asym {r.get('ingress_egress_asymmetry',0):.3f}"
            f"  (real transits are left-right symmetric)",
            f"{mark(cons_ok)}  Depth consistency   scat {r.get('depth_consistency',0):.3f}"
            f"  (same depth every orbit; variable depth = artifact)",
        ]
        fig.text(0.08, 0.665, "\n".join(checks), fontsize=8, va='top',
                 family='monospace', color='#1a2a4a')

        # Cross-match + ephemeris
        info = ["CATALOG CROSS-MATCH & EPHEMERIS", "-" * 76]
        if xm:
            info.append(f"TOI {xm['toi']}  —  TFOPWG disposition: {xm['disposition']}"
                        + ("  (confirmed planet)" if xm['known_planet'] else
                           "  (known false positive)" if xm['known_fp'] else
                           "  (UNCONFIRMED candidate - this verdict is novel)"))
            if xm.get('catalog_period'):
                agree = abs(r['period_days'] - xm['catalog_period']) / xm['catalog_period'] < 0.03
                info.append(f"Catalog period {xm['catalog_period']:.4f} d vs detected "
                            f"{r['period_days']:.4f} d  -> "
                            + ("MATCH" if agree else "MISMATCH (check harmonics)"))
        else:
            info.append("Not in the TOI catalog - a real signal here would be a new detection.")
        try:
            from astropy.time import Time as _Time
            _jd0 = float(r["t0"]) + 2457000.0
            _P = float(r["period_days"])
            _n = int(np.ceil((_Time.now().jd - _jd0) / _P))
            nxt = [_Time(_jd0 + (_n + k) * _P, format="jd").iso[:16] for k in range(3)]
            info.append(f"Next transits (UTC): {nxt[0]}   |   {nxt[1]}   |   {nxt[2]}")
        except Exception:
            pass
        fig.text(0.08, 0.535, "\n".join(info), fontsize=8, va='top',
                 family='monospace', color='#1a2a4a')

        # AI insight
        insight = generate_ai_insight(r).replace("**", "")
        wrapped = textwrap.fill(insight, width=92)
        fig.text(0.08, 0.44, "AUTOMATED INTERPRETATION\n" + "-" * 76 + "\n" + wrapped,
                 fontsize=8, va='top', family='monospace', color='#1a2a4a')

        try:
            with open("model_metrics.json") as _pf:
                _pm = json.load(_pf)
            _model_line = (f"Model: calibrated XGBoost {_pm.get('version','')} "
                           f"(holdout acc {_pm.get('holdout_accuracy','?')}%, "
                           f"AUC {_pm.get('holdout_roc_auc','?')})")
        except Exception:
            _model_line = "Model: calibrated XGBoost"
        fig.text(0.5, 0.045,
                 f"Generated {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')} UTC-local  |  "
                 f"{_model_line}  |  "
                 "github.com/krishnendukoley2007-arch/ExoDetect",
                 ha='center', fontsize=7, color='#7a8aa8')
        pdf.savefig(fig, facecolor='white'); plt.close(fig)

        # ── PAGE 2 — SHAP explanation ─────────────────────────
        if r.get("shap"):
            shap_d = dict(sorted(r["shap"].items(), key=lambda kv: abs(kv[1])))
            names = list(shap_d.keys()); vals = list(shap_d.values())
            figs, axs = plt.subplots(figsize=(8.5, 11), facecolor='white')
            colors = ['#1a7a2a' if v > 0 else '#9a1010' for v in vals]
            axs.barh(names, vals, color=colors)
            axs.axvline(0, color='#888', linewidth=0.8)
            axs.set_title(f"Why the model decided — SHAP contributions for TIC {r['tic_id']}\n"
                          "green pushes toward PLANET, red pushes toward FALSE POSITIVE",
                          fontsize=11, color='#1a2a4a')
            axs.set_xlabel("Contribution to planet log-odds")
            axs.tick_params(labelsize=8)
            for sp in axs.spines.values():
                sp.set_color('#cccccc')
            figs.tight_layout()
            pdf.savefig(figs, facecolor='white'); plt.close(figs)

        fig2, axes = plt.subplots(3, 1, figsize=(8.5, 11),
                                   facecolor='#0a1220', constrained_layout=True)
        lc_t, lc_f = clean_series(result['lc'].time.value, result['lc'].flux.value)
        axes[0].scatter(lc_t, lc_f, s=0.8, alpha=0.5, color='#4488cc', rasterized=True)
        axes[0].set_title("Clean Light Curve", pad=6)
        axes[0].set_xlabel("Time (days)"); axes[0].set_ylabel("Normalized Flux")
        axes[0].grid(True)

        pg_p, pg_pw = clean_series(result['pg'].period.value, result['pg'].power.value)
        axes[1].plot(pg_p, pg_pw, color='#4488cc', linewidth=0.8)
        axes[1].axvline(x=result['period_days'], color='#ff6633',
                        linewidth=1.5, linestyle='--', label=f"Best = {result['period_days']:.4f} d")
        axes[1].set_title("BLS Periodogram", pad=6)
        axes[1].legend(fontsize=8)
        axes[1].grid(True)

        zoom = max(result['half_width'] * 8, 0.05)
        pv, fv = clean_series(result["phase_vals"], result["flux_vals"])
        bp, bf = result["bin_phase"], result["bin_flux"]
        axes[2].scatter(pv, fv, s=1.5, alpha=0.3, color='#4488cc', rasterized=True)
        if len(bp) > 2:
            axes[2].plot(bp, bf, color='#ff6633', linewidth=2)
        axes[2].set_xlim(-zoom, zoom)
        axes[2].set_title("Phase-Folded Transit", pad=6)
        axes[2].set_xlabel("Phase"); axes[2].set_ylabel("Normalized Flux")
        axes[2].grid(True)

        pdf.savefig(fig2); plt.close(fig2)
    buf.seek(0)
    return buf


def generate_extra_graphs_pdf(result, figs):
    """Bundles the 5 Deep Diagnostics figures (already rendered on screen)
    into a single downloadable PDF. figs may contain None for skipped plots."""
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        cover = plt.figure(figsize=(8.5, 11), facecolor='white')
        ax0 = cover.add_axes([0, 0, 1, 1]); ax0.axis('off')
        cover.text(0.5, 0.92, "ExoDetect — Deep Diagnostics Report",
                   ha='center', fontsize=18, fontweight='bold', color='#1a2a4a')
        cover.text(0.5, 0.89, f"TIC {result['tic_id']} | BAH2026 PS7 | Jadavpur University",
                   ha='center', fontsize=10, color='#4a6a9a')
        cover.text(0.08, 0.82,
            f"Generated : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
            f"Period    : {result['period_days']:.4f} days\n"
            f"Depth     : {result['depth']*100:.5f}%\n"
            f"SNR       : {result['snr']:.2f}\n"
            f"Verdict   : {result['ml_class']} ({result['ml_confidence']:.1f}%)\n\n"
            "Contents:\n"
            "  1. Odd vs Even Transit Comparison\n"
            "  2. Secondary Eclipse Zoom\n"
            "  3. River Plot (transit-by-transit)\n"
            "  4. Residuals Plot\n"
            "  5. Periodogram Zoom",
            fontsize=10, va='top', family='monospace', color='#1a2a4a')
        pdf.savefig(cover, facecolor='white'); plt.close(cover)

        for f in figs:
            if f is not None:
                pdf.savefig(f, facecolor=BG)
    buf.seek(0)
    return buf



# ════════════════════════════════════════════════════════════
# PLOT HELPERS — sharp, consistent styling
# ════════════════════════════════════════════════════════════
ACCENT   = '#4a9eff'
ORANGE   = '#ff7a45'
BG       = '#0a1220'
GRID_C   = '#1e3050'
TEXT_C   = '#ddeeff'
LABEL_C  = '#7090b8'

def style_ax(ax):
    ax.set_facecolor(BG)
    ax.tick_params(colors=LABEL_C, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_C)
    ax.grid(True, color=GRID_C, alpha=0.5, linewidth=0.5)
    ax.set_xlabel(ax.get_xlabel(), color=LABEL_C, fontsize=9)
    ax.set_ylabel(ax.get_ylabel(), color=LABEL_C, fontsize=9)
    ax.set_title(ax.get_title(), color=TEXT_C, fontweight='bold', fontsize=10, pad=8)

def make_fig(w=12, h=4):
    fig, ax = plt.subplots(figsize=(w, h), facecolor=BG)
    return fig, ax


# ════════════════════════════════════════════════════════════
# RENDER — single star result
# ════════════════════════════════════════════════════════════
def render_result(result):
    r          = result
    bp         = r["bin_phase"]
    bf         = r["bin_flux"]
    zoom       = max(r["half_width"] * 10, 0.04)
    ml_class   = r["ml_class"]
    depth      = r["depth"]
    snr        = r["snr"]
    period_d   = r["period_days"]
    best_power = float(r["best_power"])

    st.success(
        f"✅  TIC {r['tic_id']}  |  {r['n_sectors']}/{r['n_available']} sectors  |  "
        f"Period {period_d:.4f} d  |  BLS Power {best_power:.0f}"
    )
    if not r.get("stellar_params_found", True):
        st.warning("⚠️ No stellar parameters found for this TIC — using Sun-like "
                   "defaults (Teff 5778 K, 1 R☉, 1 M☉). The planet-radius estimate "
                   "and ML probability are less reliable for this star.")

    # ── Metrics row ──
    st.markdown("---")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("⏱️ Period",    f"{period_d:.3f} d")
    m2.metric("📉 Depth",     f"{depth*100:.4f}%")
    m3.metric("📡 SNR",       f"{snr:.1f}")
    m4.metric("🌍 Radius",    f"{r['R_planet_earth']:.2f} R⊕")
    m5.metric("🔍 BLS Power", f"{best_power:.0f}")
    m6.metric("⏳ Duration",   f"{r['duration_hours']:.2f} h")

    # ── Verdict box ──
    if ml_class == "Exoplanet Candidate":
        vc, vi, emoji = "#0a7a2a", "✓", "🪐"
    elif "Binary" in ml_class or "False" in ml_class:
        vc, vi, emoji = "#9a1010", "✗", "⭐"
    else:
        vc, vi, emoji = "#555577", "❓", "🔭"

    st.markdown(f"""
    <div class='verdict-box' style='background:{vc}22; border:2px solid {vc};'>
        <div style='font-size:2.2rem; margin-bottom:4px;'>{emoji}</div>
        <h2 style='color:{vc}; margin:4px 0; font-size:1.5rem;'>{vi} {ml_class.upper()}</h2>
        <p style='color:#b8d0ec; margin:0; font-size:0.93rem;'>
            ML Confidence: <b>{r['ml_confidence']:.1f}%</b> &nbsp;|&nbsp;
            Rule-based: <b>{r['rule_verdict']}</b> &nbsp;|&nbsp;
            Sectors stacked: <b>{r['n_sectors']}</b>
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Known-object cross-match (local TOI catalog) ──
    xm = db.lookup_toi(r["tic_id"])
    if xm:
        if xm["known_planet"]:
            st.success(f"📖 **Cross-match:** TOI {xm['toi']} — already a **confirmed/known "
                       f"planet** (disposition {xm['disposition']}). Catalog period: "
                       f"{xm['catalog_period']:.4f} d vs your detection {r['period_days']:.4f} d.")
        elif xm["known_fp"]:
            st.error(f"📖 **Cross-match:** TOI {xm['toi']} — cataloged as a **false positive** "
                     f"({xm['disposition']}). Good test of whether the model agrees.")
        elif xm["candidate"]:
            st.warning(f"🔭 **Cross-match:** TOI {xm['toi']} — an **unconfirmed candidate** "
                       f"({xm['disposition']}). Your classification is genuinely useful here: "
                       f"nobody has settled this one yet!")
    else:
        st.info("🆕 **Cross-match:** not in the TOI catalog — if this signal is real, "
                "it would be a new detection.")

    # ── Period guardrails — each problem comes with a one-click fix ──
    def _fix_button(new_lo, new_hi, key):
        """Render a 'Fix it' button that retunes the window and re-analyzes."""
        if st.button("🔧 Fix it for me — retune window & re-analyze",
                     key=key, type="primary", use_container_width=True):
            st.session_state["_pending_period_window"] = (
                float(np.clip(new_lo, 0.5, 5.0)),
                float(np.clip(new_hi, 5.0, 30.0)))
            st.session_state["_auto_reanalyze"] = str(r["tic_id"])
            st.rerun()

    _pmin = r.get("search_pmin"); _pmax = r.get("search_pmax")
    _Pc = xm.get("catalog_period") if xm else None
    _match = _Pc and abs(period_d - _Pc) / _Pc < 0.03
    _half   = _Pc and abs(period_d - _Pc / 2) / (_Pc / 2) < 0.03
    _double = _Pc and abs(period_d - _Pc * 2) / (_Pc * 2) < 0.03

    if _Pc and _match:
        st.success(f"✅ **Period check:** your detection ({period_d:.4f} d) matches "
                   f"the catalog period ({_Pc:.4f} d) — the model is judging the "
                   "right signal.")
    elif _Pc and (_half or _double):
        st.warning(
            f"⚠️ **Wrong period found (harmonic):** BLS found {period_d:.4f} d but the "
            f"true period is {_Pc:.4f} d — {'half' if _half else 'double'} of it. This "
            "mixes odd/even transits and can make a real planet look like a binary, "
            "so the verdict below may be wrong."
        )
        _fix_button(_Pc * 0.7, _Pc * 1.4, "fix_harmonic")
    elif _Pc and _Pc <= 28.5:
        st.error(
            f"🚨 **Wrong period found:** BLS found {period_d:.4f} d but the catalog "
            f"period is {_Pc:.4f} d. The verdict below is about the wrong signal — "
            "don't trust it."
        )
        _fix_button(_Pc * 0.7, _Pc * 1.4, "fix_mismatch")
    elif _pmin and _pmax and (period_d >= _pmax * 0.95 or period_d <= _pmin * 1.05):
        # Edge-of-window warning only when no catalog period could diagnose it
        st.warning(
            f"⚠️ **Period search hit the edge of your window** — the BLS peak "
            f"({period_d:.3f} d) is right at the boundary of your search range "
            f"({_pmin:.1f}–{_pmax:.1f} d). The true period may be *outside* this "
            "window."
        )
        if period_d >= _pmax * 0.95:
            _fix_button(_pmin, min(_pmax * 2, 30.0), "fix_edge")
        else:
            _fix_button(max(_pmin / 2, 0.5), _pmax, "fix_edge")

    # ── Transit ephemeris — when can a telescope catch the next transits? ──
    try:
        from astropy.time import Time as _Time
        _jd0 = float(r["t0"]) + 2457000.0          # BTJD → JD
        _P   = float(r["period_days"])
        _now = _Time.now().jd
        _n   = int(np.ceil((_now - _jd0) / _P))
        _next = [_Time(_jd0 + (_n + k) * _P, format="jd").iso[:16] for k in range(3)]
        st.markdown(f"""
        <div class='stat-card' style='text-align:center;'>
          <b style='color:#90b8e8;'>🔭 Next predicted transits (UTC):</b>
          &nbsp; {_next[0]} &nbsp;•&nbsp; {_next[1]} &nbsp;•&nbsp; {_next[2]}
          <span style='color:#6080a8; font-size:0.8rem;'>&nbsp; (from BLS ephemeris — for observation planning)</span>
        </div>
        """, unsafe_allow_html=True)
    except Exception:
        pass

    # ── AI Insight ──
    insight_text = generate_ai_insight(r)
    # Convert **markdown bold** to HTML <b> since this renders inside raw HTML
    import re as _re
    insight_text = _re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", insight_text)
    st.markdown(f"""
    <div class='ai-insight'>
        <div class='ai-insight-header'>🤖 AI Interpretation</div>
        <p class='ai-insight-text'>{insight_text}</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Tabs ──
    st.markdown("---")
    t1, t2, t3, t4, t5, t6 = st.tabs([
        "📉 Light Curve", "🔍 Period Search", "🪐 Transit Shape", "🤖 Classifier",
        "🔬 Deep Diagnostics", "🌍 3D Orbit"
    ])

    with t1:
        lc_t, lc_f = clean_series(r['lc'].time.value, r['lc'].flux.value)
        lc_t, lc_f = _thin(lc_t, lc_f)
        fig = go.Figure(go.Scattergl(
            x=lc_t, y=lc_f, mode="markers",
            marker=dict(size=2.5, color=ACCENT, opacity=0.45),
            name=f"{len(lc_t):,} data points",
            hovertemplate="t = %{x:.4f} BTJD<br>flux = %{y:.5f}<extra></extra>",
        ))
        fig.update_xaxes(title="Time (BTJD days)")
        fig.update_yaxes(title="Normalized Flux")
        st.plotly_chart(db.style_fig(
            fig, f"📉 Stacked Light Curve — TIC {r['tic_id']} ({r['n_sectors']} sectors) "
                 "— drag to zoom, double-click to reset", height=420),
            use_container_width=True)

    with t2:
        pg_p, pg_pw = clean_series(r['pg'].period.value, r['pg'].power.value)
        fig = go.Figure(go.Scattergl(
            x=pg_p, y=pg_pw, mode="lines",
            line=dict(color=ACCENT, width=1.2), fill="tozeroy",
            fillcolor="rgba(74,158,255,0.08)", name="BLS power",
            hovertemplate="period = %{x:.4f} d<br>power = %{y:.0f}<extra></extra>",
        ))
        fig.add_vline(x=period_d, line_dash="dash", line_color=ORANGE, line_width=2,
                      annotation_text=f"Peak {period_d:.4f} d (power {best_power:.0f})",
                      annotation_font=dict(color=ORANGE, size=11))
        fig.update_xaxes(title="Period (days)")
        fig.update_yaxes(title="BLS Power")
        st.plotly_chart(db.style_fig(
            fig, "🔍 BLS Periodogram — Box Least Squares Period Search", height=420),
            use_container_width=True)

    with t3:
        pv, fv = clean_series(r["phase_vals"], r["flux_vals"])
        pv_t, fv_t = _thin(pv, fv)

        def _fold_fig(xlim, title, show_floor=False):
            fig = go.Figure()
            fig.add_trace(go.Scattergl(
                x=pv_t, y=fv_t, mode="markers",
                marker=dict(size=2.5, color=ACCENT, opacity=0.25),
                name="Folded flux", hoverinfo="skip"))
            if len(bp) > 2:
                fig.add_trace(go.Scattergl(
                    x=bp, y=bf, mode="lines",
                    line=dict(color=ORANGE, width=2.6),
                    name=f"Binned model — depth {depth*100:.4f}%",
                    hovertemplate="phase = %{x:.4f}<br>flux = %{y:.5f}<extra></extra>"))
            fig.add_hline(y=r["baseline"], line_dash="dash", line_color="#556",
                          opacity=0.6)
            if show_floor:
                fig.add_hline(y=r["baseline"] - depth, line_dash="dot",
                              line_color="#ffcc44", opacity=0.8,
                              annotation_text="Transit floor",
                              annotation_font=dict(color="#ffcc44", size=10))
            fig.update_xaxes(title="Phase (fraction of orbit)", range=list(xlim))
            fig.update_yaxes(title="Normalized Flux")
            return db.style_fig(fig, title, height=440)

        cf1, cf2 = st.columns(2)
        cf1.plotly_chart(_fold_fig((-0.5, 0.5), "Full Phase-Folded View"),
                         use_container_width=True)
        cf2.plotly_chart(_fold_fig((-zoom, zoom),
                         f"🪐 Zoomed Transit — {r['R_planet_earth']:.1f} R⊕",
                         show_floor=True), use_container_width=True)

    with t4:
        classes_list = list(r["ml_proba"].keys())
        vals         = [v*100 for v in r["ml_proba"].values()]
        cmap         = {"planet": "#1a7a2a", "false_positive": "#9a1010",
                        "Weak / No Signal": "#444466"}
        bcolors      = [cmap.get(c, ACCENT) for c in classes_list]

        fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor=BG)

        bars = axes[0].bar(classes_list, vals, color=bcolors, width=0.45, zorder=2)
        axes[0].set_ylabel("Probability (%)"); axes[0].set_ylim(0, 115)
        axes[0].set_title("ML Class Probabilities")
        for bar, v in zip(bars, vals):
            axes[0].text(bar.get_x()+bar.get_width()/2, v+3,
                         f"{v:.1f}%", ha='center', color=TEXT_C, fontweight='bold', fontsize=10)
        style_ax(axes[0])

        try:
            imp  = clf_model.feature_importances_
            _fc  = FEATURE_COLS[:len(imp)]
            ypos = list(range(len(_fc)))
            bars2 = axes[1].barh(ypos, imp, color=ACCENT, height=0.6, zorder=2)
            axes[1].set_yticks(ypos)
            axes[1].set_yticklabels(_fc, color=LABEL_C, fontsize=8)
            axes[1].set_xlabel("Feature Importance")
            axes[1].set_title("RandomForest Feature Importance")
            for bar, v in zip(bars2, imp):
                axes[1].text(v+0.005, bar.get_y()+bar.get_height()/2,
                             f"{v:.3f}", va='center', color=TEXT_C, fontsize=8)
        except Exception:
            axes[1].axis('off')
        style_ax(axes[1])

        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)

        # ── SHAP explanation: why THIS star got THIS verdict ──
        if r.get("shap"):
            st.plotly_chart(db.fig_shap(r["shap"], ml_class), use_container_width=True)
            st.caption(
                f"Calibrated planet probability: **{r.get('planet_proba', 0)*100:.1f}%** "
                f"(decision threshold {r.get('decision_threshold', 0.5):.2f} — "
                "change it in the sidebar under Decision mode)."
            )

        if is_real_model:
            st.success(f"🧠 {model_source}")
        else:
            st.warning(f"⚠️ {model_source}")

    with t5:
        st.caption(
            "All plots below use the same real TESS data already downloaded for this star — "
            "no new data fetched, just additional views of the same measurements."
        )

        # ── 1. Odd-vs-Even transit comparison ──
        st.markdown("**1. Odd vs Even Transit Comparison**")
        st.caption(
            "Real eclipsing binaries often show alternating transit depths (odd vs even "
            "cycles differ). A consistent depth across both supports a genuine planet."
        )
        t_arr        = r["t_arr"]; f_arr = r["f_arr"]
        cycle_num    = r["cycle_num"]; phase_global = r["phase_global"]
        half_width   = r["half_width"]
        in_g         = np.abs(phase_global) < half_width * 3
        odd_mask     = in_g & (cycle_num % 2 == 1)
        even_mask    = in_g & (cycle_num % 2 == 0)

        fig, ax = make_fig(12, 4.5)
        if np.sum(odd_mask) > 2:
            ax.scatter(phase_global[odd_mask], f_arr[odd_mask], s=4, alpha=0.5,
                       color=ORANGE, label=f"Odd cycles (median {r['o_d']:.6f})")
        if np.sum(even_mask) > 2:
            ax.scatter(phase_global[even_mask], f_arr[even_mask], s=4, alpha=0.5,
                       color=ACCENT, label=f"Even cycles (median {r['e_d']:.6f})")
        ax.axhline(y=r["o_d"], color=ORANGE, linestyle='--', linewidth=1, alpha=0.7)
        ax.axhline(y=r["e_d"], color=ACCENT, linestyle='--', linewidth=1, alpha=0.7)
        ax.set_xlim(-half_width*3, half_width*3)
        ax.set_title(f"Odd-Even Depth Diff = {r['odd_even_diff']*100:.5f}%  "
                     f"({'⚠️ possible EB signature' if r['odd_even_diff'] > 0.0005 else '✓ consistent — planet-like'})")
        ax.set_xlabel("Phase (fraction of orbit)"); ax.set_ylabel("Normalized Flux")
        ax.legend(fontsize=8, facecolor='#0c1422', labelcolor=LABEL_C, framealpha=0.7)
        style_ax(ax); plt.tight_layout(); st.pyplot(fig, use_container_width=True)
        st.session_state.setdefault("extra_figs", {})["odd_even"] = fig

        # ── 2. Secondary eclipse zoom ──
        st.markdown("**2. Secondary Eclipse Zoom (Phase 0.5)**")
        st.caption(
            "Zoomed view around phase 0.5 (half an orbit after transit). A deep dip here "
            "indicates a second eclipsing body — a strong eclipsing-binary signature."
        )
        pv, fv = clean_series(r["phase_vals"], r["flux_vals"])
        sec_zoom_mask = np.abs(np.abs(pv) - 0.5) < half_width * 6
        fig2, ax2 = make_fig(12, 4)
        if np.sum(sec_zoom_mask) > 2:
            pv_shift = np.where(pv[sec_zoom_mask] < 0, pv[sec_zoom_mask] + 1, pv[sec_zoom_mask]) - 0.5
            ax2.scatter(pv_shift, fv[sec_zoom_mask], s=3, alpha=0.35, color=ACCENT, rasterized=True)
        ax2.axhline(y=r["baseline"], color='#888', linestyle='--', alpha=0.5, linewidth=0.8, label="Baseline")
        ax2.axhline(y=r["baseline"]-r["sec_depth"], color='#ff5555', linestyle=':', linewidth=1.4,
                    label=f"Secondary depth ({r['sec_depth']*100:.5f}%)")
        ax2.set_xlim(-half_width*6, half_width*6)
        ax2.set_title(f"Secondary/Primary Ratio = {r['sec_ratio']:.3f}  "
                      f"({'⚠️ likely EB' if r['sec_ratio'] > 0.4 else '✓ no significant secondary'})")
        ax2.set_xlabel("Phase offset from secondary position"); ax2.set_ylabel("Normalized Flux")
        ax2.legend(fontsize=8, facecolor='#0c1422', labelcolor=LABEL_C, framealpha=0.7)
        style_ax(ax2); plt.tight_layout(); st.pyplot(fig2, use_container_width=True)
        st.session_state.setdefault("extra_figs", {})["secondary"] = fig2

        # ── 3. River plot (transit-by-transit) ──
        st.markdown("**3. River Plot — Transit Consistency Across Cycles**")
        st.caption(
            "Each row is one orbital cycle, color = flux. A consistent vertical dark band "
            "in the middle across all rows means the signal repeats reliably every orbit."
        )
        period_days = r["period_days"]
        unique_cycles = np.unique(cycle_num[np.isfinite(cycle_num)])
        unique_cycles = unique_cycles[(unique_cycles >= cycle_num.min()) & (unique_cycles <= cycle_num.max())]
        n_bins = 60
        phase_bins = np.linspace(-0.5, 0.5, n_bins+1)
        river = np.full((len(unique_cycles), n_bins), np.nan)
        for i, c in enumerate(unique_cycles):
            mask_c = (cycle_num == c) & np.isfinite(phase_global) & np.isfinite(f_arr)
            if np.sum(mask_c) < 2:
                continue
            ph_c, fl_c = phase_global[mask_c], f_arr[mask_c]
            for b in range(n_bins):
                bmask = (ph_c >= phase_bins[b]) & (ph_c < phase_bins[b+1])
                if np.sum(bmask) > 0:
                    river[i, b] = np.nanmedian(fl_c[bmask])
        fig3, ax3 = plt.subplots(figsize=(12, max(3, min(8, len(unique_cycles)*0.3))), facecolor=BG)
        im = ax3.imshow(river, aspect='auto', cmap='RdBu_r',
                        extent=[-0.5, 0.5, len(unique_cycles), 0],
                        vmin=np.nanpercentile(river, 2) if np.isfinite(river).any() else 0.99,
                        vmax=np.nanpercentile(river, 98) if np.isfinite(river).any() else 1.01)
        ax3.set_xlabel("Phase"); ax3.set_ylabel("Orbital Cycle Number")
        ax3.set_title(f"River Plot — {len(unique_cycles)} cycles stacked")
        ax3.set_xlim(-half_width*8, half_width*8)
        cbar = plt.colorbar(im, ax=ax3); cbar.set_label("Normalized Flux", color=LABEL_C)
        cbar.ax.yaxis.set_tick_params(color=LABEL_C)
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=LABEL_C)
        style_ax(ax3); plt.tight_layout(); st.pyplot(fig3, use_container_width=True)
        st.session_state.setdefault("extra_figs", {})["river"] = fig3

        # ── 4. Residuals plot ──
        st.markdown("**4. Residuals — Data Minus Binned Model**")
        st.caption(
            "Shows leftover scatter after subtracting the binned transit model from the "
            "phase-folded data. Random scatter with no pattern indicates a clean fit."
        )
        bp, bf = r["bin_phase"], r["bin_flux"]
        if len(bp) > 2:
            interp_model = np.interp(pv, bp, bf)
            residuals = fv - interp_model
            fig4, ax4 = make_fig(12, 4)
            ax4.scatter(pv, residuals, s=1.5, alpha=0.3, color=ACCENT, rasterized=True)
            ax4.axhline(y=0, color=ORANGE, linewidth=1.2, linestyle='--')
            rms = np.nanstd(residuals)
            ax4.set_title(f"Residuals (RMS = {rms*100:.5f}%)")
            ax4.set_xlabel("Phase (fraction of orbit)"); ax4.set_ylabel("Flux − Model")
            ax4.set_xlim(-0.5, 0.5)
            style_ax(ax4); plt.tight_layout(); st.pyplot(fig4, use_container_width=True)
            st.session_state.setdefault("extra_figs", {})["residuals"] = fig4
        else:
            st.info("Not enough binned points to compute residuals for this star.")

        # ── 5. Periodogram zoom around best peak ──
        st.markdown("**5. Periodogram — Zoomed Around Best Peak**")
        st.caption(
            "Close-up of the BLS power curve right around the detected period. A sharp, "
            "narrow, isolated peak is stronger evidence than a broad or noisy one."
        )
        pg_p, pg_pw = clean_series(r['pg'].period.value, r['pg'].power.value)
        zoom_width = max(period_days * 0.08, 0.05)
        zmask = np.abs(pg_p - period_days) < zoom_width
        fig5, ax5 = make_fig(12, 4)
        if np.sum(zmask) > 2:
            ax5.plot(pg_p[zmask], pg_pw[zmask], color=ACCENT, linewidth=1.3)
            ax5.fill_between(pg_p[zmask], 0, pg_pw[zmask], alpha=0.12, color=ACCENT)
        ax5.axvline(x=period_days, color=ORANGE, linewidth=2, linestyle='--',
                   label=f'Peak = {period_days:.5f} d')
        ax5.set_title("Zoomed BLS Periodogram (peak sharpness check)")
        ax5.set_xlabel("Period (days)"); ax5.set_ylabel("BLS Power")
        ax5.legend(fontsize=9, facecolor='#0c1422', labelcolor=LABEL_C, framealpha=0.7)
        style_ax(ax5); plt.tight_layout(); st.pyplot(fig5, use_container_width=True)
        st.session_state.setdefault("extra_figs", {})["periodogram_zoom"] = fig5

        # ── Download all 5 as one PDF ──
        st.markdown("---")
        extra_pdf_buf = generate_extra_graphs_pdf(r, [fig, fig2, fig3, fig4 if len(bp) > 2 else None, fig5])
        st.download_button(
            "📥 Download Deep Diagnostics (5 graphs) as PDF",
            data=extra_pdf_buf,
            file_name=f"TIC_{r['tic_id']}_deep_diagnostics.pdf",
            mime="application/pdf",
        )
        for fig_to_close in [fig, fig2, fig3, fig5] + ([fig4] if len(bp) > 2 else []):
            plt.close(fig_to_close)

    with t6:
        st.caption(
            "Physics-derived visualization: orbit size from Kepler's 3rd law using this "
            "star's TIC mass; habitable zone from its luminosity; equilibrium temperature "
            "assumes Earth-like albedo 0.3. Drag to rotate, press ▶️ Orbit to animate."
        )
        sp = get_stellar_features(r["tic_id"])
        orbit_fig, phys = db.fig_orbit_3d(
            period_d, st_mass=sp.get("mass", 1.0), st_rad=sp.get("rad", 1.0),
            st_teff=sp.get("Teff", 5778.0),
            planet_radius_earth=r["R_planet_earth"],
            tic_id=r["tic_id"], ml_class=ml_class,
        )
        o1, o2, o3, o4 = st.columns(4)
        o1.metric("🛰️ Orbit Size",   f"{phys['a_au']:.4f} AU")
        o2.metric("🌡️ Equil. Temp",  f"{phys['teq']:.0f} K")
        o3.metric("☀️ Star Lum.",    f"{phys['lum']:.3f} L☉")
        o4.metric("🌿 Habitable Zone",
                  "INSIDE ✓" if phys["in_hz"] else
                  ("Too hot 🔥" if phys["a_au"] < phys["hz_in"] else "Too cold 🧊"))
        st.plotly_chart(orbit_fig, use_container_width=True)
        has_params = (not stellar_db.empty) and str(r["tic_id"]) in stellar_db.index
        if not has_params:
            st.info("ℹ️ No TIC stellar params for this star — orbit uses Sun-like "
                    "defaults. Run fetch_stellar_params.py to improve coverage.")

    # ── Full params expander ──
    st.markdown("---")
    with st.expander("💡 Full Parameter Details"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
| Parameter | Value |
|---|---|
| TIC ID | {r['tic_id']} |
| Orbital Period | {period_d:.4f} days |
| Transit Duration | {r['duration_hours']:.2f} hours |
| Transit Depth | {depth*100:.5f}% |
| Planet Radius | {r['R_planet_earth']:.2f} R⊕ |
| BLS Power | {best_power:.1f} |
            """)
        with col2:
            st.markdown(f"""
| Parameter | Value |
|---|---|
| SNR | {snr:.2f} |
| Secondary Eclipse Depth | {r['sec_depth']*100:.5f}% |
| Secondary / Primary Ratio | {r['sec_ratio']:.3f} |
| Odd-Even Depth Diff | {r['odd_even_diff']*100:.5f}% |
| Sectors Used | {r['n_sectors']} / {r['n_available']} |
| ML Confidence | {r['ml_confidence']:.1f}% |
            """)

    pdf_buf = generate_pdf(result)
    st.download_button(
        "📄 Download PDF Report",
        data=pdf_buf,
        file_name=f"ExoDetect_TIC{r['tic_id']}.pdf",
        mime="application/pdf",
        use_container_width=True,
        key=f"pdf_{r['tic_id']}_{r['timestamp']}"
    )


# ════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════
st.markdown("""
<div class='hero-banner'>
  <div class='hero-title'>🪐 ExoDetect</div>
  <p class='hero-sub'>
    AI-Enabled Exoplanet Detection from NASA TESS Light Curves<br>
    <span style='color:#4a6a9a; font-size:0.88rem;'>
      Bharatiya Antariksh Hackathon 2026 — PS7 &nbsp;|&nbsp; Jadavpur University
    </span>
  </p>
</div>
""", unsafe_allow_html=True)

if is_real_model:
    st.success(f"🧠 **Active model:** {model_source}")
else:
    st.warning(f"⚠️ **Active model:** {model_source}")


# ════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════
st.sidebar.markdown("""
<div style='text-align:center; padding:14px 0 10px;'>
  <span style='font-size:2.2rem;'>🪐</span><br>
  <span style='color:#90b8e8; font-weight:700; font-size:1.1rem; font-family:Space Grotesk,sans-serif;'>ExoDetect</span><br>
  <span style='color:#3a5070; font-size:0.72rem; letter-spacing:0.05em;'>BAH 2026 — PS7</span>
</div>
""", unsafe_allow_html=True)
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["🔭 Individual Analysis", "⚖️ Compare Stars", "🛰️ Batch Survey",
     "🏆 Frontier Leaderboard", "🗄️ Database Explorer", "🗺️ Sky Map",
     "🎯 Model Honesty", "📄 Project Report", "📜 History"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown("**⚙️ Pipeline Settings**")
_mode = st.sidebar.radio(
    "🎯 Decision mode",
    ["Balanced (0.50)", "Survey — high recall (0.35)", "Vetting — high precision (0.65)"],
    help="Threshold on the calibrated planet probability. Survey mode misses fewer "
         "planets; Vetting mode makes fewer false planet calls.",
)
planet_threshold = float(_mode.split("(")[1].rstrip(")"))
st.session_state.setdefault("period_min", 1.0)
st.session_state.setdefault("period_max", 20.0)
# Apply a pending auto-tune request (set by the Individual Analysis page)
# BEFORE the slider widgets are instantiated — Streamlit forbids writing a
# widget's session key after its widget exists in the same run.
_pw = st.session_state.pop("_pending_period_window", None)
if _pw:
    st.session_state["period_min"], st.session_state["period_max"] = _pw
max_sectors = st.sidebar.slider("Sectors to stack", 1, 10, 5)
period_min  = st.sidebar.slider("Min period (days)", 0.5, 5.0, key="period_min")
period_max  = st.sidebar.slider("Max period (days)", 5.0, 30.0, key="period_max")
if period_min >= period_max:  # both sliders can sit at 5.0 — BLS needs a window
    st.sidebar.warning("Min period must be below max — widening the window.")
    period_max = period_min + 0.5

if not dataset_pool.empty:
    st.sidebar.markdown("---")
    try:
        with open("model_metrics.json") as _df_:
            _n_train_total = json.load(_df_).get("n_total")
    except Exception:
        _n_train_total = None
    _tot_txt = f" · trained on {_n_train_total} (TESS+Kepler)" if _n_train_total else ""
    st.sidebar.markdown(f"**📊 Dataset:** {len(dataset_pool)} TESS stars for live "
                        f"analysis{_tot_txt}")
    n_planet = (dataset_pool['label']=='planet').sum()
    n_fp     = (dataset_pool['label']=='false_positive').sum()
    st.sidebar.markdown(f"🪐 Planets: **{n_planet}** &nbsp; ⭐ False Pos: **{n_fp}**")

try:
    _n_saved = db.db_stats()["total"]
    st.sidebar.markdown(f"🗄️ Saved analyses: **{_n_saved}**")
except Exception as e:
    print(f"WARNING: db_stats failed: {e}")

try:
    with open("model_metrics.json") as _mf:
        _mm = json.load(_mf)
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**🧠 Model {_mm.get('version','')} — honest holdout**")
    _ma, _mb = st.sidebar.columns(2)
    _ma.metric("Accuracy", f"{_mm['holdout_accuracy']:.1f}%")
    _mb.metric("Recall", f"{_mm['holdout_recall']:.1f}%")
    _mc, _md = st.sidebar.columns(2)
    _mc.metric("ROC-AUC", f"{_mm['holdout_roc_auc']:.3f}")
    _md.metric("Test stars", f"{_mm['n_test']}")
except Exception as e:
    print(f"WARNING: model_metrics.json sidebar card failed: {e}")

st.sidebar.markdown("---")
st.sidebar.markdown("**👥 Team Members**")
st.sidebar.markdown(
    "**Team Member 1 (Team Leader)**\n\n"
    "Krishnendu Koley\n\n"
    "Jadavpur University, Kolkata\n\n"
    "krishnendukoley2007@gmail.com"
)
st.sidebar.markdown(
    "**Team Member 2**\n\n"
    "Abhradeep Bera\n\n"
    "Jadavpur University, Kolkata\n\n"
    "abhradeepmsk23@gmail.com"
)
st.sidebar.markdown(
    "**Team Member 3**\n\n"
    "Asmit Dey\n\n"
    "Jadavpur University, Kolkata\n\n"
    "deyasmit07@gmail.com"
)
st.sidebar.markdown("**Team Name:** OrbitX2026")
st.sidebar.markdown("---")
try:
    with open("model_metrics.json") as _vf_:
        _ver_ = json.load(_vf_).get("version", "v10")
except Exception:
    _ver_ = "v10"
st.sidebar.caption(f"ExoDetect {_ver_} | BAH2026 PS7 | Team OrbitX2026")

st.markdown("---")


# ════════════════════════════════════════════════════════════
# PAGE 1 — INDIVIDUAL ANALYSIS
# ════════════════════════════════════════════════════════════
if page == "🔭 Individual Analysis":
    st.subheader("🔭 Individual Star Analysis")

    st.caption("Try a known example first — each demonstrates a different signal type:")
    q1, q2, q3 = st.columns(3)
    with q1:
        if st.button("🌍 Pi Mensae c", use_container_width=True,
                     help="Super-Earth, P=6.27 d — a clean, textbook transit"):
            st.session_state.tic_id = "261136679"
        st.caption("Super-Earth · clean U-shaped transit")
    with q2:
        if st.button("🪐 WASP-126 b", use_container_width=True,
                     help="Hot Jupiter, P=3.29 d — deep, obvious transit"):
            st.session_state.tic_id = "25155310"
        st.caption("Hot Jupiter · deep transit")
    with q3:
        if st.button("⭐ Binary Star", use_container_width=True,
                     help="Eclipsing binary — V-shaped dip + secondary eclipse"):
            st.session_state.tic_id = "441075486"
        st.caption("Eclipsing binary · the classic impostor")

    if not dataset_pool.empty:
        st.markdown("**Or pick from the real NASA dataset:**")
        pool_sorted = dataset_pool.sort_values("snr", ascending=False)

        pick_col1, pick_col2 = st.columns([1, 3])
        with pick_col1:
            label_f = st.selectbox("Filter by type",
                ["All", "🪐 Planets only", "⭐ False Positives only"])
        if label_f == "🪐 Planets only":
            pool_filtered = pool_sorted[pool_sorted['label'] == 'planet']
        elif label_f == "⭐ False Positives only":
            pool_filtered = pool_sorted[pool_sorted['label'] == 'false_positive']
        else:
            pool_filtered = pool_sorted

        with pick_col2:
            chosen = st.selectbox(
                f"Select star ({len(pool_filtered)} available — sorted by SNR)",
                options=["— type TIC ID below or select here —"] + pool_filtered["display_name"].tolist()
            )
        if chosen != "— type TIC ID below or select here —":
            id_map = dict(zip(pool_filtered["display_name"], pool_filtered["tic_id"].astype(str)))
            st.session_state.tic_id = id_map[chosen]

    tic_id = st.text_input("TIC Star ID (manual entry)", key="tic_id",
                           placeholder="e.g. 261136679 (digits only, or paste 'TIC 261136679')")
    # Accept sloppy input: "TIC 261136679", "tic261136679", spaces, commas
    tic_id = "".join(ch for ch in str(tic_id) if ch.isdigit())

    # ── Pre-flight check: what do we already know about this star? ──
    if tic_id:
        _pre = db.lookup_toi(tic_id)
        if _pre:
            _Pc = _pre.get("catalog_period")
            pf1, pf2 = st.columns([3, 1])
            with pf1:
                _disp_txt = {True: "🪐 confirmed planet"}.get(_pre["known_planet"]) or \
                            ("⭐ known false positive" if _pre["known_fp"] else
                             "🔭 unconfirmed candidate — your verdict matters!")
                st.info(
                    f"📖 **TOI {_pre['toi']}** ({_pre['disposition']} — {_disp_txt})"
                    + (f" &nbsp;|&nbsp; catalog period **{_Pc:.4f} d**" if _Pc else "")
                    + (f" &nbsp;|&nbsp; catalog radius **{_pre['catalog_radius']:.2f} R⊕**"
                       if _pre.get("catalog_radius") else "")
                )
            with pf2:
                if _Pc:
                    if _Pc > 28.5:
                        st.caption(f"⚠️ Catalog period {_Pc:.1f} d is beyond the 30 d "
                                   "search limit — BLS may only find a harmonic.")
                    elif st.button("📐 Auto-tune period window", use_container_width=True,
                                   help=f"Set the sidebar sliders to bracket the known "
                                        f"{_Pc:.2f} d period"):
                        # 0.7x–1.4x brackets the true period while EXCLUDING the
                        # P/2 and 2P harmonics that BLS otherwise loves to grab.
                        st.session_state["_pending_period_window"] = (
                            float(np.clip(_Pc * 0.7, 0.5, 5.0)),
                            float(np.clip(_Pc * 1.4, 5.0, 30.0)),
                        )
                        st.rerun()
        else:
            st.caption("🆕 Not in the TOI catalog — any signal found here would be a "
                       "new detection.")

    st.caption(f"⚙️ Current settings: search {period_min:.1f}–{period_max:.1f} d · "
               f"{max_sectors} sectors · threshold {planet_threshold:.2f} "
               "(change in sidebar)")
    run_btn = st.button(f"🔭 Analyze TIC {tic_id}" if tic_id else "🔭 Analyze",
                        type="primary", use_container_width=True,
                        disabled=not tic_id)

    # A guardrail 'Fix it' button queued a re-analysis with the corrected window
    _auto_tic = st.session_state.pop("_auto_reanalyze", None)
    if _auto_tic:
        tic_id = _auto_tic
        run_btn = True
        st.info(f"🔧 Window retuned to {period_min:.1f}–{period_max:.1f} d — "
                f"re-analyzing TIC {tic_id} automatically...")

    if run_btn:
        with st.spinner(f"🔭 Analyzing TIC {tic_id} — downloading TESS data → detrending → BLS search → ML classify..."):
            result = get_or_run(tic_id, max_sectors, period_min, period_max, force=True)
        if result.get("error"):
            err = result['error']
            # User-friendly error messages
            if "too large" in err.lower():
                st.error(f"❌ BLS periodogram overflow — try reducing 'Max period' in the sidebar or increasing 'Min period'.")
            elif "no tess" in err.lower() or "no data" in err.lower():
                st.error(f"❌ No TESS data found for TIC {tic_id}. This star may lack 2-minute cadence observations.")
            elif "download failed" in err.lower():
                st.error(f"❌ Download failed (MAST server issue). Please wait 30 seconds and retry.")
            else:
                st.error(f"❌ {err[:500]}")
            st.session_state.last_result = None
        else:
            st.session_state.last_result = result

    if st.session_state.last_result:
        render_result(st.session_state.last_result)


# ════════════════════════════════════════════════════════════
# PAGE 2 — COMPARE STARS
# ════════════════════════════════════════════════════════════
elif page == "⚖️ Compare Stars":
    st.subheader("⚖️ Multi-Star Comparison")

    tabA, tabB = st.tabs(["📚 Pick from dataset", "🎯 Quick stars + Custom IDs"])
    pool_ids = []; quick_ids = []; custom_ids = []

    with tabA:
        if dataset_pool.empty:
            st.info("features_dataset.csv not found — run extract_features.py first.")
        else:
            c1, c2 = st.columns(2)
            with c1:
                lf = st.multiselect("Filter by label",
                    sorted(dataset_pool['label'].unique()),
                    default=sorted(dataset_pool['label'].unique()))
            with c2:
                n_pick = st.slider("Max to pick", 2, 20, 6)
            filt   = dataset_pool[dataset_pool['label'].isin(lf)].sort_values("snr", ascending=False)
            chosen = st.multiselect(
                f"Select stars ({len(filt)} available — sorted by SNR)",
                options=filt["display_name"].tolist(),
                default=filt["display_name"].tolist()[:n_pick]
            )
            id_map   = dict(zip(filt["display_name"], filt["tic_id"].astype(str)))
            pool_ids = [id_map[d] for d in chosen]

    with tabB:
        qnames    = st.multiselect("Quick-select", list(QUICK_STARS.keys()), default=[])
        quick_ids = [QUICK_STARS[n] for n in qnames]
        ctext     = st.text_area("Custom TIC IDs (comma-separated)", value="")
        custom_ids = [x.strip() for x in ctext.split(",") if x.strip()]

    run_cmp = st.button("⚖️ Run Comparison", type="primary", use_container_width=True)

    if run_cmp:
        all_ids = list(dict.fromkeys(pool_ids + quick_ids + custom_ids))
        if len(all_ids) < 2:
            st.warning("Select at least 2 stars.")
        else:
            prog = st.progress(0, text="Starting...")
            results_cmp = []
            for i, sid in enumerate(all_ids):
                prog.progress(i/len(all_ids), text=f"TIC {sid} ({i+1}/{len(all_ids)})...")
                r = get_or_run(sid, max_sectors, period_min, period_max)
                if not r.get("error"):
                    results_cmp.append(r)
                else:
                    st.warning(f"TIC {sid} failed: {r['error'][:120]}")
            prog.progress(1.0, text="Done!")
            st.session_state["compare_results"] = results_cmp

    cmp_res = st.session_state.get("compare_results", [])

    if cmp_res:
        st.markdown("---")
        st.subheader(f"📋 Comparison Table — {len(cmp_res)} stars")

        rows = [{
            "TIC ID":       r["tic_id"],
            "Period (d)":   round(r["period_days"], 4),
            "Depth (%)":    round(r["depth"]*100, 5),
            "SNR":          round(r["snr"], 2),
            "Radius (R⊕)":  round(r["R_planet_earth"], 2),
            "Duration (h)": round(r["duration_hours"], 2),
            "BLS Power":    round(float(r["best_power"]), 0),
            "ML Verdict":   r["ml_class"],
            "Confidence":   f"{r['ml_confidence']:.1f}%",
        } for r in cmp_res]

        cdf = pd.DataFrame(rows)
        st.dataframe(cdf, use_container_width=True)
        csv_dl = cdf.to_csv(index=False).encode()
        st.download_button("⬇️ Download CSV", csv_dl,
                           "exodetect_comparison.csv", "text/csv", use_container_width=True)

        n_pl = sum(1 for r in cmp_res if r["ml_class"]=="Exoplanet Candidate")
        n_fp = sum(1 for r in cmp_res if "Binary" in r["ml_class"] or "False" in r["ml_class"])
        s1, s2, s3 = st.columns(3)
        s1.metric("✓ Planets detected", n_pl)
        s2.metric("✗ Binaries / FP",    n_fp)
        s3.metric("❓ Uncertain",         len(cmp_res)-n_pl-n_fp)

        # ── Overlaid transit shapes ──
        st.markdown("---")
        st.subheader("🪐 Overlaid Transit Shapes")
        zoom    = max((max(r["half_width"] for r in cmp_res) * 10), 0.04)
        palette = ['#4a9eff','#ff7a45','#22cc77','#ff4466',
                   '#aa55ff','#ffcc22','#00ccee','#ff88cc']

        fig, ax = plt.subplots(figsize=(13, 5.5), facecolor=BG)
        for i, r in enumerate(cmp_res):
            c  = palette[i % len(palette)]
            bp = r["bin_phase"]; bf = r["bin_flux"]
            if len(bp) > 2:
                ax.plot(bp, bf, color=c, linewidth=2.2, alpha=0.9,
                        label=f"TIC {r['tic_id']} — {r['ml_class'][:15]}")
        ax.axhline(y=1.0, color='#333', linestyle='--', alpha=0.5, linewidth=0.8)
        ax.set_xlim(-zoom, zoom)
        ax.set_title("Phase-Folded Transit Shapes Overlaid")
        ax.set_xlabel("Phase (fraction of orbit)"); ax.set_ylabel("Normalized Flux")
        ax.legend(fontsize=7, ncol=3, facecolor='#0c1422', labelcolor=LABEL_C, framealpha=0.8)
        style_ax(ax); plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)

        # ── Metric bars ──
        st.markdown("---")
        st.subheader("📊 Metric Comparison")
        bar_colors_list = [
            '#1a7a2a' if r["ml_class"]=="Exoplanet Candidate"
            else '#9a1010' if ("Binary" in r["ml_class"] or "False" in r["ml_class"])
            else '#555577'
            for r in cmp_res
        ]
        labels_x = [f"TIC {r['tic_id']}" for r in cmp_res]
        bc1, bc2, bc3 = st.columns(3)
        metrics_to_plot = [
            ("Transit Depth (%)",  [r["depth"]*100 for r in cmp_res]),
            ("SNR",                [r["snr"] for r in cmp_res]),
            ("Planet Radius (R⊕)", [r["R_planet_earth"] for r in cmp_res]),
        ]
        for col, (title, vals) in zip([bc1, bc2, bc3], metrics_to_plot):
            with col:
                fig, ax = plt.subplots(figsize=(5, 4), facecolor=BG)
                ax.bar(labels_x, vals, color=bar_colors_list, zorder=2)
                ax.set_title(title, fontsize=9)
                plt.setp(ax.get_xticklabels(), rotation=40, ha='right', fontsize=6)
                style_ax(ax); plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)
    else:
        st.info("Select at least 2 stars and click 'Run Comparison'.")


# ════════════════════════════════════════════════════════════
# PAGE — BATCH SURVEY MODE (ranked candidate list for scientists)
# ════════════════════════════════════════════════════════════
elif page == "🛰️ Batch Survey":
    st.subheader("🛰️ Batch Survey — rank many stars by planet probability")
    st.caption(
        "Run the full pipeline over a list of TIC IDs and get a ranked candidate "
        "table with calibrated probabilities — like a real vetting queue. "
        "Each star takes ~30–90 s (TESS download + BLS)."
    )

    bs1, bs2 = st.tabs(["✏️ Paste TIC IDs", "🔭 Frontier candidates (unconfirmed TOIs)"])
    batch_ids = []
    with bs1:
        btext = st.text_area("TIC IDs (comma or newline separated)",
                             placeholder="261136679, 25155310, 441075486")
        batch_ids = [x.strip() for x in btext.replace("\n", ",").split(",") if x.strip()]
        up = st.file_uploader("…or upload a CSV with a tic_id column", type="csv")
        if up is not None:
            try:
                if up.size > 10_000_000:
                    raise ValueError("file too large (max 10 MB)")
                _updf = pd.read_csv(up)
                col = "tic_id" if "tic_id" in _updf.columns else _updf.columns[0]
                batch_ids = _updf[col].astype(str).str.strip().tolist()
                st.success(f"Loaded {len(batch_ids)} IDs from CSV.")
            except Exception as e:
                st.error(f"Could not read CSV: {e}")
    with bs2:
        frontier = cached_frontier()
        if frontier.empty:
            st.info("frontier_targets.csv not found — run fetch_toi.py first.")
        else:
            st.markdown(
                f"**{len(frontier)} unconfirmed TOI candidates** (PC/APC) are waiting for "
                "a verdict. Pick a few — your model's call on these is genuinely novel."
            )
            n_front = st.slider("How many to survey (sorted by TESS magnitude, brightest first)",
                                2, 20, 5)
            if st.checkbox("Use these frontier candidates as the batch"):
                fsel = frontier.sort_values("st_tmag").head(n_front)
                batch_ids = fsel["tid"].astype(str).tolist()
                st.dataframe(fsel[["toi", "tid", "pl_orbper", "st_tmag"]],
                             use_container_width=True, height=220)

    run_batch = st.button(f"🛰️ Survey {len(batch_ids)} stars", type="primary",
                          use_container_width=True, disabled=len(batch_ids) < 1)

    if run_batch:
        prog = st.progress(0, text="Starting survey...")
        rows = []
        for i, sid in enumerate(dict.fromkeys(batch_ids)):
            prog.progress(i / len(batch_ids), text=f"TIC {sid} ({i+1}/{len(batch_ids)})...")
            res = get_or_run(sid, max_sectors, period_min, period_max)
            if res.get("error"):
                rows.append({"TIC ID": sid, "Status": "❌ " + res["error"][:60],
                             "Planet Prob (%)": None})
                continue
            xm = db.lookup_toi(sid)
            try:
                db.save_frontier_result(res)   # feeds the Frontier Leaderboard
            except Exception:
                pass
            rows.append({
                "TIC ID": sid,
                "Planet Prob (%)": round(res.get("planet_proba", 0) * 100, 1),
                "Verdict": res["ml_class"],
                "Period (d)": round(res["period_days"], 4),
                "Depth (%)": round(res["depth"] * 100, 4),
                "SNR": round(res["snr"], 1),
                "Radius (R⊕)": round(res["R_planet_earth"], 2),
                "TOI status": (xm["disposition"] if xm else "not in TOI"),
                "Status": "✓",
            })
        prog.progress(1.0, text="Survey complete!")
        st.session_state["batch_results"] = rows

    brows = st.session_state.get("batch_results", [])
    if brows:
        bdf = pd.DataFrame(brows)
        if "Planet Prob (%)" in bdf.columns:
            bdf = bdf.sort_values("Planet Prob (%)", ascending=False, na_position="last")
        st.markdown("### 🏆 Ranked Candidate List")
        st.dataframe(bdf, use_container_width=True)
        st.download_button("⬇️ Download ranked survey (CSV)",
                           bdf.to_csv(index=False).encode(),
                           "exodetect_survey_ranked.csv", "text/csv",
                           use_container_width=True)
        good = bdf[bdf["Planet Prob (%)"].notna() & (bdf["Planet Prob (%)"] >= 65)]
        if len(good):
            st.success(f"🌟 {len(good)} star(s) above 65% calibrated planet probability — "
                       "strongest follow-up targets at the top of the table.")


# ════════════════════════════════════════════════════════════
# PAGE — FRONTIER LEADERBOARD (accumulated verdicts on unconfirmed TOIs)
# ════════════════════════════════════════════════════════════
elif page == "🏆 Frontier Leaderboard":
    st.subheader("🏆 Frontier Leaderboard — the most-likely-real unconfirmed planets")
    st.caption(
        "Every Batch Survey run on an unconfirmed TOI (PC/APC) is stored here "
        "permanently. Over time this becomes a ranked, publishable list of "
        "candidates our calibrated model believes are real — before NASA has "
        "issued a verdict on them."
    )

    fres = db.load_frontier_results()
    frontier_all = cached_frontier()
    n_total = len(frontier_all) if not frontier_all.empty else 0

    if fres.empty:
        st.info(
            "No frontier verdicts stored yet. Go to **🛰️ Batch Survey → "
            "Frontier candidates**, survey a few unconfirmed TOIs, and they "
            "will appear here automatically."
        )
    else:
        HC_T = 0.70  # high-confidence tier threshold
        hc = fres[fres["planet_proba"] >= HC_T]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Candidates surveyed", f"{len(fres)}" + (f" / {n_total}" if n_total else ""))
        m2.metric(f"🥇 High-confidence (≥{HC_T*100:.0f}%)", len(hc))
        m3.metric("Best probability", f"{fres['planet_proba'].max()*100:.1f}%")
        m4.metric("Median probability", f"{fres['planet_proba'].median()*100:.1f}%")

        # measured precision at this threshold on the honest holdout
        _hc_prec = None
        try:
            _hp = cached_holdout()
            _sel = _hp[_hp["planet_proba"] >= HC_T]
            if len(_sel) >= 20:
                _hc_prec = _sel["y_true"].mean() * 100
        except Exception as e:
            print(f"WARNING: holdout precision calc failed: {e}")
        if not hc.empty:
            prec_txt = (f"On the honest holdout, {_hc_prec:.0f}% of calls at this "
                        f"confidence are real planets." if _hc_prec else "")
            st.success(
                f"🥇 **High-confidence tier** — {len(hc)} unconfirmed candidate(s) "
                f"pass the ≥{HC_T*100:.0f}% probability bar. {prec_txt}"
            )
            _hcs = hc.sort_values("planet_proba", ascending=False).head(10)
            st.markdown("  \n".join(
                f"• **{db.star_label(r['tic_id'])}** (TOI {r['toi']}) — "
                f"{r['planet_proba']*100:.1f}% | {r['radius_earth']:.1f} R⊕ | "
                f"P {r['period_days']:.2f} d"
                for _, r in _hcs.iterrows()))
        if n_total:
            st.progress(min(len(fres) / n_total, 1.0),
                        text=f"Frontier coverage: {len(fres)/n_total*100:.1f}% of "
                             f"{n_total} unconfirmed candidates")

        top_n = st.slider("Leaderboard size", 5, 50, 20)
        st.plotly_chart(db.fig_frontier_leaderboard(fres, top_n),
                        use_container_width=True)

        if len(fres) >= 3:
            st.plotly_chart(db.fig_frontier_scatter(fres), use_container_width=True)

        st.markdown("### 📋 Full frontier table")
        show = fres.copy()
        show["planet_proba"] = (show["planet_proba"] * 100).round(1)
        show = show.rename(columns={
            "tic_id": "TIC ID", "toi": "TOI", "disposition": "TFOPWG",
            "planet_proba": "Planet Prob (%)", "verdict": "Verdict",
            "period_days": "Period (d)", "depth_pct": "Depth (%)",
            "snr": "SNR", "radius_earth": "Radius (R⊕)",
            "tmag": "Tmag", "surveyed_at": "Surveyed",
        })
        st.dataframe(show, use_container_width=True, height=420)
        c1, c2 = st.columns(2)
        c1.download_button("⬇️ Download leaderboard (CSV)",
                           show.to_csv(index=False).encode(),
                           "exodetect_frontier_leaderboard.csv", "text/csv",
                           use_container_width=True)
        c2.download_button("⬇️ Download leaderboard (JSON)",
                           show.to_json(orient="records", indent=2).encode(),
                           "exodetect_frontier_leaderboard.json", "application/json",
                           use_container_width=True)


# ════════════════════════════════════════════════════════════
# PAGE 3 — DATABASE EXPLORER (powered by database.py)
# ════════════════════════════════════════════════════════════
elif page == "🗄️ Database Explorer":
    st.markdown("""
    <div style='display:flex; align-items:center; gap:16px; margin-bottom:6px;'>
      <div class='orbit-wrap'>
        <div class='orbit-star'>⭐</div>
        <div class='orbit-planet'>🪐</div>
      </div>
      <div>
        <h2 style='margin:0;'>Database Explorer</h2>
        <p style='margin:0; color:#6080a8; font-size:0.9rem;'>
          Persistent SQLite analysis vault + interactive analytics over the v9 NASA dataset
        </p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    stats   = db.db_stats()
    catalog = cached_catalog()

    # ── Animated stat cards ──
    st.markdown(f"""
    <div style='display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin:14px 0 6px;'>
      <div class='db-stat-card'><div class='db-num'>{len(catalog)}</div><div class='db-label'>Catalog Stars</div></div>
      <div class='db-stat-card'><div class='db-num'>{stats['total']}</div><div class='db-label'>Saved Analyses</div></div>
      <div class='db-stat-card'><div class='db-num'>{stats['planets']}</div><div class='db-label'>🪐 Planet Verdicts</div></div>
      <div class='db-stat-card'><div class='db-num'>{stats['best_snr']:.1f}</div><div class='db-label'>Best SNR</div></div>
    </div>
    """, unsafe_allow_html=True)
    st.caption(f"Last analysis saved: {stats['last']}  |  Storage: exodetect.db (survives restarts)")

    dbt1, dbt2, dbt3, dbt4 = st.tabs([
        "🌌 Catalog Analytics", "🧊 3D & Advanced", "💾 Saved Analyses", "📋 Data Table & Export"
    ])

    # ── Shared filters + filtered catalog ──
    if catalog.empty:
        for _t in (dbt1, dbt2):
            with _t:
                st.info("features_dataset.csv not found — run extract_features.py first.")
        cat = catalog
    else:
        with dbt1:
            fc1, fc2, fc3 = st.columns([1.2, 1.2, 1])
            with fc1:
                type_f = st.multiselect("Type", sorted(catalog["Type"].unique()),
                                        default=sorted(catalog["Type"].unique()))
            with fc2:
                snr_max = float(np.nanmax(catalog["snr"])) + 1
                snr_rng = st.slider("SNR range", 0.0, snr_max, (0.0, snr_max))
            with fc3:
                search = st.text_input("🔎 Search TIC ID", "")
            cat = catalog[catalog["Type"].isin(type_f)]
            cat = cat[(cat["snr"] >= snr_rng[0]) & (cat["snr"] <= snr_rng[1])]
            if search.strip():
                cat = cat[cat["tic_id"].astype(str).str.contains(re.escape(search.strip()))]

    # ── TAB 1: core catalog analytics ──
    if not catalog.empty:
        with dbt1:
            st.caption(f"Showing **{len(cat)}** of {len(catalog)} stars — every chart is "
                       "interactive: hover, zoom, pan, click legend entries to toggle.")
            if cat.empty:
                st.warning("No stars match the current filters.")
            else:
                g1, g2 = st.columns([1, 1.1])
                with g1:
                    st.plotly_chart(db.fig_class_donut(cat), use_container_width=True)
                with g2:
                    st.plotly_chart(db.fig_sunburst(cat), use_container_width=True)

                st.plotly_chart(db.fig_snr_depth(cat), use_container_width=True)
                st.plotly_chart(db.fig_period_radius(cat), use_container_width=True)
                st.plotly_chart(db.fig_animated_discovery(cat), use_container_width=True)

                # Feature distribution explorer
                st.markdown("**🔬 Feature Distribution Explorer**")
                feat_opts = db.available_features(cat)
                fsel = st.selectbox("Pick any of the 17 v9 features",
                                    options=list(feat_opts.keys()),
                                    format_func=lambda c: feat_opts[c], index=1)
                st.plotly_chart(db.fig_feature_histogram(cat, fsel), use_container_width=True)

        # ── TAB 2: 3D & advanced ──
        with dbt2:
            source = cat if not cat.empty else catalog
            st.caption("Advanced views over the currently filtered catalog "
                       f"({len(source)} stars).")
            st.plotly_chart(db.fig_3d_features(source), use_container_width=True)
            f3d2 = db.fig_3d_shape(source)
            if f3d2 is not None:
                st.plotly_chart(f3d2, use_container_width=True)
            st.plotly_chart(db.fig_radar_profile(source), use_container_width=True)
            st.plotly_chart(db.fig_parallel_coords(source), use_container_width=True)
            fhr = db.fig_hr_diagram(source)
            if fhr is not None:
                st.plotly_chart(fhr, use_container_width=True)
            fs3d = db.fig_stellar_3d(source)
            if fs3d is not None:
                st.plotly_chart(fs3d, use_container_width=True)
            else:
                st.info("HR diagram needs stellar params — run fetch_stellar_params.py "
                        "to extend coverage beyond the current subset.")
            st.plotly_chart(db.fig_corr_heatmap(source), use_container_width=True)

    # ── TAB 3: persistent saved analyses ──
    with dbt3:
        saved = db.fetch_analyses()
        if saved.empty:
            st.info("No analyses saved yet — run any star on the Individual Analysis page "
                    "and it will be stored here automatically, forever.")
        else:
            tl = db.fig_analyses_timeline(saved)
            if tl is not None:
                st.plotly_chart(tl, use_container_width=True)

            st.markdown("**🔬 Inspect a saved analysis**")
            options = [
                f"#{row.id} — TIC {row.tic_id} — {row.ml_class} ({row.ml_confidence:.0f}%) — {row.analyzed_at}"
                for row in saved.itertuples()
            ]
            pick = st.selectbox("Select record", options)
            rec = saved.iloc[options.index(pick)]

            ic1, ic2 = st.columns([1, 1.4])
            with ic1:
                st.plotly_chart(
                    db.fig_confidence_gauge(rec["ml_confidence"], rec["ml_class"]),
                    use_container_width=True)
            with ic2:
                st.markdown(f"""
| Parameter | Value |
|---|---|
| TIC ID | {rec['tic_id']} |
| Analyzed | {rec['analyzed_at']} |
| Period | {rec['period_days']:.4f} days |
| Depth | {rec['depth_pct']:.5f}% |
| SNR | {rec['snr']:.2f} |
| Radius | {rec['radius_earth']:.2f} R⊕ |
| BLS Power | {rec['bls_power']:.0f} |
| Rule-based | {rec['rule_verdict']} |
                """)
            if str(rec.get("ai_insight", "")).strip():
                st.markdown(f"""
                <div class='ai-insight'>
                    <div class='ai-insight-header'>🤖 Saved AI Interpretation</div>
                    <p class='ai-insight-text'>{rec['ai_insight']}</p>
                </div>
                """, unsafe_allow_html=True)

            d1, d2, d3 = st.columns(3)
            with d1:
                st.download_button("⬇️ Export all as CSV", db.export_csv(saved),
                                   "exodetect_database.csv", "text/csv",
                                   use_container_width=True)
            with d2:
                st.download_button("⬇️ Export all as JSON", db.export_json(saved),
                                   "exodetect_database.json", "application/json",
                                   use_container_width=True)
            with d3:
                if st.button("🗑️ Delete this record", use_container_width=True):
                    db.delete_analysis(rec["id"])
                    st.rerun()

            with st.expander("⚠️ Danger zone"):
                if st.button("🧨 Wipe entire analysis database"):
                    db.clear_analyses()
                    st.rerun()

    # ── TAB 4: raw table + exports ──
    with dbt4:
        src = st.radio("Data source", ["NASA Catalog (v9 training set)", "My Saved Analyses"],
                       horizontal=True)
        table = catalog.drop(columns=["display_name"], errors="ignore") \
                if src.startswith("NASA") else db.fetch_analyses()
        if table.empty:
            st.info("Nothing to show for this source yet.")
        else:
            st.dataframe(table, use_container_width=True, height=420)
            e1, e2 = st.columns(2)
            with e1:
                st.download_button("⬇️ Download CSV", db.export_csv(table),
                                   "exodetect_export.csv", "text/csv",
                                   use_container_width=True)
            with e2:
                st.download_button("⬇️ Download JSON", db.export_json(table),
                                   "exodetect_export.json", "application/json",
                                   use_container_width=True)


# ════════════════════════════════════════════════════════════
# PAGE — SKY MAP
# ════════════════════════════════════════════════════════════
elif page == "🗺️ Sky Map":
    st.subheader("🗺️ Sky Map — the TOI population on the celestial sphere")
    sky = cached_sky_targets()
    if sky.empty:
        st.info("toi_raw_full.csv not found — run fetch_toi.py first.")
    else:
        groups = st.multiselect("Show", sorted(sky["Group"].unique()),
                                default=sorted(sky["Group"].unique()))
        skyf = sky[sky["Group"].isin(groups)]
        c1, c2, c3 = st.columns(3)
        c1.metric("🪐 Confirmed planets", (skyf["Group"].str.contains("Confirmed")).sum())
        c2.metric("⭐ False positives",   (skyf["Group"].str.contains("False")).sum())
        c3.metric("🔭 Unconfirmed",       (skyf["Group"].str.contains("Unconfirmed")).sum())
        st.plotly_chart(db.fig_sky_3d(skyf), use_container_width=True)
        st.plotly_chart(db.fig_sky_map(skyf), use_container_width=True)
        st.caption(
            "The dense bands near ±84° declination are TESS's continuous viewing zones "
            "(observed in every sector — longest baselines, best small-planet sensitivity). "
            "Blue points are unconfirmed candidates: the Batch Survey page can rank them."
        )


# ════════════════════════════════════════════════════════════
# PAGE — MODEL HONESTY (ROC + calibration on held-out stars)
# ════════════════════════════════════════════════════════════
elif page == "🎯 Model Honesty":
    st.subheader("🎯 Model Honesty — evaluation on never-seen stars")
    st.caption(
        "Every number on this page comes from a 20% held-out test set that was locked "
        "away before training. No cross-validation optimism, no leakage — this is how "
        "the model performs on stars it has genuinely never seen."
    )
    hold = cached_holdout()
    if hold.empty:
        st.info("holdout_predictions.csv not found — run eval_holdout.py after training.")
    else:
        import json as _json
        if os.path.exists("model_metrics.json"):
            with open("model_metrics.json") as _f:
                mm = _json.load(_f)
            h1, h2, h3, h4, h5 = st.columns(5)
            h1.metric("Accuracy",  f"{mm['holdout_accuracy']}%")
            h2.metric("Precision", f"{mm['holdout_precision']}%")
            h3.metric("Recall",    f"{mm['holdout_recall']}%")
            h4.metric("ROC-AUC",   f"{mm['holdout_roc_auc']}")
            h5.metric("Brier",     f"{mm['holdout_brier']}")
            st.caption(f"Held-out set: {mm['n_test']} stars | trained on {mm['n_train']} | "
                       f"calibration: {mm['calibration']} | stellar coverage "
                       f"{mm['stellar_coverage_pct']}%")
        r1, r2 = st.columns(2)
        with r1:
            st.plotly_chart(db.fig_roc_curve(hold), use_container_width=True)
        with r2:
            st.plotly_chart(db.fig_reliability(hold), use_container_width=True)
        st.plotly_chart(db.fig_proba_split(hold), use_container_width=True)
        st.markdown("""
        <div class='report-card'>
        <h3>Why we report these numbers (and not 97%)</h3>
        <p>Earlier versions reported cross-validation accuracy, which can be inflated by
        subtle leakage between folds. v10+ locks away a 20% star-level holdout before
        any training or tuning, and applies <b>isotonic calibration</b> so the probability
        the app shows is trustworthy: when it says "80% planet", roughly 80% of such calls
        are real planets — verifiable in the reliability curve above. For telescope-time
        decisions, a calibrated 80% is worth more than an uncalibrated 97%.</p>
        </div>
        """, unsafe_allow_html=True)
        st.download_button("⬇️ Download held-out predictions (CSV)",
                           db.export_csv(hold), "holdout_predictions.csv", "text/csv")


# ════════════════════════════════════════════════════════════
# PAGE 4 — PROJECT REPORT
# ════════════════════════════════════════════════════════════
elif page == "📄 Project Report":
    st.info("💡 Press Ctrl+P → Save as PDF to export this page.")

    st.markdown("""
    <div class='report-card'>
    <h2>🪐 ExoDetect — Project Report</h2>
    <p><b>Bharatiya Antariksh Hackathon 2026 | Problem Statement 7</b><br>
    AI-Enabled Detection of Exoplanets from Noisy Astronomical Light Curves</p>
    <p><b>Team OrbitX2026:</b> Krishnendu Koley (Team Leader), Abhradeep Bera, Asmit Dey &nbsp;|&nbsp; <b>Institution:</b> Jadavpur University, Kolkata</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='report-card'>
    <h3>1. Problem Overview</h3>
    <p>When a planet transits its host star, the star dims slightly and periodically.
    NASA's TESS satellite records these brightness curves for millions of stars.
    This project builds a 6-stage pipeline to detect, characterize, and classify
    periodic transit signals — distinguishing real planets from eclipsing binaries
    and instrumental artifacts. An AI interpretation layer then translates the
    numerical outputs into plain-English conclusions.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='report-card'>
    <h3>2. Pipeline Stages</h3>
    <ol>
    <li><b>Data Acquisition</b> — Multi-sector TESS download via lightkurve / NASA MAST with retry logic</li>
    <li><b>De-trending</b> — Savitzky-Golay flattening (window 401) removes instrumental systematics</li>
    <li><b>BLS Period Search</b> — Bounded Box Least Squares (≤2000 periods, frequency_factor=1) prevents grid overflow</li>
    <li><b>Phase Folding + Feature Extraction</b> — 21 physics-informed features: 11 light-curve (depth, SNR, secondary-eclipse ratio, transit shape, odd-even, asymmetry…), 3 engineered (planet radius estimate, expected-duration ratio, period) + 6 stellar + mission flag</li>
    <li><b>ML Classification</b> — isotonic-calibrated XGBoost trained on 3,751 labeled NASA stars (TESS TOI + Kepler KOI), honest star-level holdout metrics</li>
    <li><b>AI Interpretation</b> — Natural language summary of signal quality, classification reasoning, and cross-check</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)

    dc1, dc2 = st.columns(2)
    with dc1:
        st.markdown("<div class='report-card'><h3>3. Dataset</h3>", unsafe_allow_html=True)
        if not dataset_pool.empty:
            _lc  = dataset_pool['label'].value_counts()
            _ldf = pd.DataFrame({"Label": _lc.index.astype(str), "Count": _lc.values})
            st.dataframe(_ldf, use_container_width=True)
            try:
                with open("model_metrics.json") as _rf:
                    _rm = json.load(_rf)
                st.caption(f"Live-analysis pool: {len(dataset_pool)} TESS stars | "
                           f"training set: {_rm.get('n_total','?')} stars (TESS+Kepler) | "
                           f"XGBoost {_rm.get('version','')} | {len(FEATURE_COLS)} features | "
                           f"{_rm.get('holdout_accuracy','?')}% holdout accuracy")
            except Exception:
                st.caption(f"Live-analysis pool: {len(dataset_pool)} TESS stars | "
                           f"{len(FEATURE_COLS)} features")
        else:
            st.info("Run extract_features.py to populate dataset.")
        st.markdown("</div>", unsafe_allow_html=True)

    with dc2:
        st.markdown("<div class='report-card'><h3>4. Model Comparison</h3>", unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({
            "Model":      ["RandomForest (features)", "1D-CNN (raw curves)"],
            "5-Fold CV":  ["87.35% ± 1.04%",  "88.65% ± 2.92%"],
            "Status":     ["✓ Deployed",      "✓ Trained"],
        }), use_container_width=True)
        st.caption("RF+GB Ensemble trained on 641 real NASA TESS stars. CNN trained on 493 phase-folded curves.")
        st.markdown("</div>", unsafe_allow_html=True)

    vc1, vc2 = st.columns(2)
    with vc1:
        if os.path.exists("model_validation.png"):
            st.image("model_validation.png", caption="RandomForest — Feature Importance + Confusion Matrix")
    with vc2:
        if os.path.exists("cnn_validation.png"):
            st.image("cnn_validation.png", caption="CNN — 5-Fold CV + Pooled Confusion Matrix")

    st.markdown("""
    <div class='report-card'>
    <h3>5. Key Improvements in v9.0</h3>
    <ul>
    <li><b>~2,000-star training dataset</b> — expanded from 641 to 1,971 NASA TOI stars (1,666 after quality cleaning), auto-balanced planet / false-positive classes</li>
    <li><b>XGBoost classifier with 17 features</b> — 11 physics-informed light-curve features (transit shape, depth consistency, ingress/egress asymmetry, odd-even ratio, …) + 6 TIC stellar parameters (Teff, radius, mass, log g, Tmag, contamination), reaching 97.60% CV accuracy and 0.9987 ROC-AUC</li>
    <li><b>Stellar-parameter enrichment</b> — live TIC catalog lookup feeds host-star physics into every classification</li>
    <li><b>Persistent SQLite database</b> — every analysis auto-saved to exodetect.db, browsable in the Database Explorer with interactive 3-D plots, radar fingerprints, parallel coordinates, HR diagram, and animated population charts</li>
    <li><b>Infinity/NaN guard</b> — feature matrix cleaned before training to prevent float32 overflow crashes</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# PAGE 4 — HISTORY
# ════════════════════════════════════════════════════════════
elif page == "📜 History":
    st.subheader("📜 Session Test History")

    if st.session_state.history:
        hist_df = pd.DataFrame(st.session_state.history)
        st.dataframe(hist_df, use_container_width=True)

        h1, h2 = st.columns(2)
        with h1:
            csv_dl = hist_df.to_csv(index=False).encode()
            st.download_button("⬇️ Download CSV", csv_dl,
                               "exodetect_history.csv", "text/csv", use_container_width=True)
        with h2:
            if st.button("🗑️ Clear All History", use_container_width=True):
                st.session_state.history       = []
                st.session_state.last_result   = None
                st.session_state.results_cache = {}
                st.session_state["compare_results"] = []
                st.rerun()

        st.markdown("---")
        st.subheader("📊 Session Summary")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Stars Analyzed",    len(hist_df))
        s2.metric("Planets Found",     (hist_df["ML Verdict"]=="Exoplanet Candidate").sum())
        s3.metric("Binaries Rejected", hist_df["ML Verdict"].str.contains("Binary|False", na=False).sum())
        s4.metric("Avg SNR",           f"{hist_df['SNR'].mean():.1f}" if 'SNR' in hist_df else "—")
    else:
        st.info("No analyses run yet. Go to Individual Analysis or Compare Stars.")

st.markdown("---")
st.caption(
    f"ExoDetect v9.0 | BAH2026 PS7 | Jadavpur University | "
    f"NASA TESS / MAST | {model_source}"
)
