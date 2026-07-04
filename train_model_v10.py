"""
train_model_v10.py
==================
ExoDetect v10 — honest, calibrated retraining.

What changes vs v9:
  1. STAR-LEVEL HOLDOUT — 20% of stars are locked away before anything
     else happens. All reported metrics come from that untouched set.
  2. CALIBRATED PROBABILITIES — isotonic calibration (5-fold) so
     "planet 90%" really means ~90% of such calls are planets.
  3. Saves TWO models:
       exoplanet_classifier.pkl — calibrated model the dashboard uses
       xgb_raw.pkl              — raw XGBoost booster for SHAP explanations
  4. model_metrics.json — honest numbers the dashboard displays.

Run AFTER fetch_stellar_params.py (uses the full stellar_params.csv).
"""

import json
import os
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (accuracy_score, brier_score_loss, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

BLS_FEATURES = ["depth", "snr", "sec_ratio", "duration_hours", "bls_power",
                "odd_even_diff", "transit_shape", "dur_period_ratio",
                "depth_consistency", "ingress_egress_asymmetry", "odd_even_ratio"]
STELLAR_FEATURES = ["Teff", "rad", "mass", "logg", "Tmag", "contratio"]
# v10.1 engineered features — combine light-curve + stellar physics:
#   period_days             raw orbital period (was unused!)
#   planet_radius_est       sqrt(depth) x stellar radius, in Earth radii —
#                           implausibly large => eclipsing binary
#   duration_expected_ratio observed / expected transit duration for a
#                           circular orbit around this star (classic vetting)
ENGINEERED_FEATURES = ["period_days", "planet_radius_est", "duration_expected_ratio"]
STELLAR_DEFAULTS = {"Teff": 5778.0, "rad": 1.0, "mass": 1.0,
                    "logg": 4.44, "Tmag": 10.0, "contratio": 0.0}

print("=" * 60)
print("ExoDetect v10 — Calibrated Retraining with Honest Holdout")
print("=" * 60)

# ── 1. Load + clean (same physical filters as v9) ──────────
df = pd.read_csv("features_dataset.csv")
if os.path.exists("features_dataset_kepler.csv"):
    kep = pd.read_csv("features_dataset_kepler.csv")
    print(f"  + Kepler expansion: {len(kep)} stars")
    df = pd.concat([df, kep], ignore_index=True)
before = len(df)
df = df[(df["depth"] > 0) & (df["depth"] < 0.5)]
df = df[(df["snr"] > 1.0) & (df["snr"] < 10000)]
df = df[(df["sec_ratio"] >= -0.5) & (df["sec_ratio"] < 5.0)]
df = df[df["duration_hours"] > 0]
df = df[df["bls_power"] > 0]
df = df[df["transit_shape"].between(-2, 10)]
df = df[df["depth_consistency"].between(-5, 20)]
df = df[df["ingress_egress_asymmetry"].between(-5, 100)]
df = df[df["odd_even_ratio"].between(-5, 100)]
df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=BLS_FEATURES)
df = df.drop_duplicates(subset="tic_id")
print(f"\n[1/5] Clean stars: {len(df)} (removed {before - len(df)})")
print(df["label"].value_counts().to_string())

# ── 2. Merge stellar params ─────────────────────────────────
sp = pd.read_csv("stellar_params.csv")
if os.path.exists("stellar_params_kepler.csv"):
    sp = pd.concat([sp, pd.read_csv("stellar_params_kepler.csv")], ignore_index=True)
sp["tic_id"] = sp["tic_id"].astype(str)
df["tic_id"] = df["tic_id"].astype(str)
keep = ["tic_id"] + [c for c in STELLAR_FEATURES if c in sp.columns]
df = df.merge(sp[keep], on="tic_id", how="left")
coverage = df["Teff"].notna().mean()
for c, v in STELLAR_DEFAULTS.items():
    if c in df.columns:
        df[c] = df[c].fillna(v)
    else:
        df[c] = v
print(f"\n[2/5] Stellar param coverage: {coverage*100:.1f}% of stars")

df["planet_radius_est"] = np.sqrt(df["depth"].clip(lower=0)) * df["rad"] * 109.076
t_exp_hours = 13.0 * (df["period_days"] / 365.25) ** (1 / 3) * df["rad"] / df["mass"] ** (1 / 3)
df["duration_expected_ratio"] = df["duration_hours"] / t_exp_hours.clip(lower=1e-6)

FEATURE_COLS = BLS_FEATURES + ENGINEERED_FEATURES + STELLAR_FEATURES
X = df[FEATURE_COLS].values
le = LabelEncoder()
y = le.fit_transform(df["label"])
planet_idx = int(np.where(le.classes_ == "planet")[0][0])

# ── 3. STAR-LEVEL HOLDOUT — the honest split ────────────────
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42)
print(f"\n[3/5] Train: {len(X_tr)} stars | Held-out test: {len(X_te)} stars "
      "(never seen during training or tuning)")

xgb_params = dict(
    n_estimators=400, max_depth=5, learning_rate=0.06,
    subsample=0.85, colsample_bytree=0.8, min_child_weight=3,
    reg_lambda=1.5, eval_metric="logloss", random_state=42, n_jobs=-1,
)

# 5-fold CV on the TRAIN portion only (for reference, not the headline)
cv_model = XGBClassifier(**xgb_params)
cv_scores = cross_val_score(cv_model, X_tr, y_tr, cv=5, scoring="accuracy")
print(f"  5-fold CV on train: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")

# ── 4. Fit raw model + calibrated wrapper ───────────────────
raw = XGBClassifier(**xgb_params)
raw.fit(X_tr, y_tr)
cal = CalibratedClassifierCV(XGBClassifier(**xgb_params), method="isotonic", cv=5)
cal.fit(X_tr, y_tr)

proba_te = cal.predict_proba(X_te)[:, planet_idx]
pred_te = (proba_te >= 0.5).astype(int)
y_te_planet = (y_te == planet_idx).astype(int)

metrics = {
    "version": "v10",
    "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
    "n_total": int(len(df)),
    "stellar_coverage_pct": round(float(coverage * 100), 1),
    "cv_accuracy_train": round(float(cv_scores.mean() * 100), 2),
    "cv_std_train": round(float(cv_scores.std() * 100), 2),
    "holdout_accuracy": round(float(accuracy_score(y_te_planet, pred_te) * 100), 2),
    "holdout_precision": round(float(precision_score(y_te_planet, pred_te) * 100), 2),
    "holdout_recall": round(float(recall_score(y_te_planet, pred_te) * 100), 2),
    "holdout_f1": round(float(f1_score(y_te_planet, pred_te) * 100), 2),
    "holdout_roc_auc": round(float(roc_auc_score(y_te_planet, proba_te)), 4),
    "holdout_brier": round(float(brier_score_loss(y_te_planet, proba_te)), 4),
    "calibration": "isotonic (5-fold)",
    "feature_cols": FEATURE_COLS,
}

print("\n[4/5] HONEST HELD-OUT TEST METRICS (never-seen stars):")
for k in ["holdout_accuracy", "holdout_precision", "holdout_recall",
          "holdout_f1", "holdout_roc_auc", "holdout_brier"]:
    print(f"  {k:22s}: {metrics[k]}")

# ── 5. Save everything ──────────────────────────────────────
if os.path.exists("exoplanet_classifier.pkl"):
    os.replace("exoplanet_classifier.pkl", "exoplanet_classifier_prev.pkl")
joblib.dump(cal, "exoplanet_classifier.pkl")
joblib.dump(raw, "xgb_raw.pkl")
joblib.dump(le, "label_encoder.pkl")
joblib.dump(FEATURE_COLS, "feature_cols.pkl")
with open("model_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("\n[5/5] Saved: exoplanet_classifier.pkl (calibrated), xgb_raw.pkl (SHAP),")
print("       label_encoder.pkl, feature_cols.pkl, model_metrics.json")
print("       (previous model backed up as exoplanet_classifier_prev.pkl)")
