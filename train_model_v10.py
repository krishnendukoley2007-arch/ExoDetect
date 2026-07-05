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

from features_config import FEATURE_COLS, load_clean

print("=" * 60)
print("ExoDetect v10 — Calibrated Retraining with Honest Holdout")
print("=" * 60)

# ── 1+2. Load + clean + stellar merge + engineered features ─
# (shared features_config.load_clean — same for train/eval/tune)
df = load_clean()
coverage = df.attrs.get("stellar_coverage", 0.0)
print(f"\n[1/5] Clean stars: {len(df)}")
print(df["label"].value_counts().to_string())
print(f"\n[2/5] Stellar param coverage: {coverage*100:.1f}% of stars")

X = df[FEATURE_COLS].values
le = LabelEncoder()
y = le.fit_transform(df["label"])
planet_idx = int(np.where(le.classes_ == "planet")[0][0])

# ── 3. STAR-LEVEL HOLDOUT — the honest split ────────────────
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42)
print(f"\n[3/5] Train: {len(X_tr)} stars | Held-out test: {len(X_te)} stars "
      "(never seen during training or tuning)")

# class balance: weight the planet class by n_negative/n_positive on train
n_pos = int((y_tr == planet_idx).sum())
spw = float((len(y_tr) - n_pos) / max(n_pos, 1)) if planet_idx == 1 else \
      float(n_pos / max(len(y_tr) - n_pos, 1))
print(f"  scale_pos_weight = {spw:.3f}")

xgb_params = dict(
    n_estimators=400, max_depth=5, learning_rate=0.06,
    subsample=0.85, colsample_bytree=0.8, min_child_weight=3,
    reg_lambda=1.5, scale_pos_weight=spw,
    eval_metric="logloss", random_state=42, n_jobs=-1,
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
    "version": "v10.2",
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

# per-mission breakdown (TESS vs Kepler) so a bad merge can't hide
mission_te = X_te[:, FEATURE_COLS.index("mission")].astype(int)
for m, name in [(0, "TESS"), (1, "Kepler")]:
    mask = mission_te == m
    if mask.sum() < 10:
        continue
    acc_m = accuracy_score(y_te_planet[mask], pred_te[mask]) * 100
    auc_m = roc_auc_score(y_te_planet[mask], proba_te[mask]) if \
        len(np.unique(y_te_planet[mask])) > 1 else float("nan")
    print(f"  {name:6s} holdout ({mask.sum()} stars): acc {acc_m:.2f}%  AUC {auc_m:.4f}")
    metrics[f"holdout_accuracy_{name.lower()}"] = round(float(acc_m), 2)
    metrics[f"holdout_roc_auc_{name.lower()}"] = round(float(auc_m), 4)

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
