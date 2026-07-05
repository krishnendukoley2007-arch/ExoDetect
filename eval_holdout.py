"""
eval_holdout.py
---------------
Regenerates the v10 held-out test predictions (same seed/split as
train_model_v10.py) and saves them to holdout_predictions.csv so the
dashboard can draw honest ROC + calibration (reliability) curves.

Also prints a threshold sweep (precision/recall/F1 vs decision threshold)
and per-mission (TESS vs Kepler) holdout metrics.

Run after train_model_v10.py. Takes seconds.
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split

from features_config import load_clean


def main():
    df = load_clean()
    cols = joblib.load("feature_cols.pkl")
    cal = joblib.load("exoplanet_classifier.pkl")
    le = joblib.load("label_encoder.pkl")
    planet_idx = int(np.where(le.classes_ == "planet")[0][0])

    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(f"feature_cols.pkl expects columns missing from the "
                         f"dataset: {missing} — retrain first (train_model_v10.py)")

    X = df[cols].values
    y = le.transform(df["label"])
    idx = np.arange(len(df))
    _, idx_te, _, y_te = train_test_split(
        idx, y, test_size=0.20, stratify=y, random_state=42)

    proba = cal.predict_proba(X[idx_te])[:, planet_idx]
    y_true = (y_te == planet_idx).astype(int)
    out = pd.DataFrame({
        "tic_id": df["tic_id"].values[idx_te],
        "y_true": y_true,
        "planet_proba": proba,
    })
    out.to_csv("holdout_predictions.csv", index=False)
    print(f"Saved holdout_predictions.csv — {len(out)} held-out stars "
          f"({out['y_true'].sum()} planets / {(1-out['y_true']).sum()} FPs)")

    # ── per-mission breakdown (TESS vs Kepler) ─────────────────
    is_kep = df["tic_id"].values[idx_te].astype(str)
    is_kep = np.char.startswith(is_kep.astype(str), "KIC")
    for mask, name in [(~is_kep, "TESS"), (is_kep, "Kepler")]:
        if mask.sum() < 10:
            continue
        pred = (proba[mask] >= 0.5).astype(int)
        auc = roc_auc_score(y_true[mask], proba[mask]) if \
            len(np.unique(y_true[mask])) > 1 else float("nan")
        print(f"  {name:6s} ({mask.sum():4d} stars): "
              f"acc {accuracy_score(y_true[mask], pred)*100:.2f}%  AUC {auc:.4f}")

    # ── threshold sweep — pick an operating point deliberately ──
    print("\nThreshold sweep (planet if proba >= t):")
    print(f"  {'t':>5s} {'acc%':>7s} {'prec%':>7s} {'rec%':>7s} {'F1%':>7s}")
    best_f1, best_t = -1.0, 0.5
    for t in np.arange(0.30, 0.81, 0.05):
        pred = (proba >= t).astype(int)
        p = precision_score(y_true, pred, zero_division=0) * 100
        r = recall_score(y_true, pred, zero_division=0) * 100
        f = f1_score(y_true, pred, zero_division=0) * 100
        a = accuracy_score(y_true, pred) * 100
        flag = ""
        if f > best_f1:
            best_f1, best_t = f, t
        print(f"  {t:5.2f} {a:7.2f} {p:7.2f} {r:7.2f} {f:7.2f}{flag}")
    print(f"  Best F1 at t={best_t:.2f} ({best_f1:.2f}%). Default 0.5 is used by "
          "the dashboard; for a precision-first frontier leaderboard consider a "
          "higher threshold.")


if __name__ == "__main__":
    main()
