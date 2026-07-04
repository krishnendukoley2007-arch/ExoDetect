"""
tune_model.py
-------------
Optuna hyperparameter sweep for the v10.1 XGBoost classifier.

Honesty rules:
  - Uses the SAME data cleaning + feature engineering as train_model_v10.py.
  - The sweep optimizes 5-fold CV ROC-AUC on the TRAIN split only
    (same random_state=42 split) — the holdout is never touched during tuning.
  - At the end, the best params are evaluated ONCE on the holdout and the
    model is saved only if holdout ROC-AUC beats the current model_metrics.json.

Run: python tune_model.py  (~15-40 min for 60 trials)
"""

import json
import os
import warnings

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (accuracy_score, brier_score_loss, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

N_TRIALS = 60

BLS_FEATURES = ["depth", "snr", "sec_ratio", "duration_hours", "bls_power",
                "odd_even_diff", "transit_shape", "dur_period_ratio",
                "depth_consistency", "ingress_egress_asymmetry", "odd_even_ratio"]
ENGINEERED_FEATURES = ["period_days", "planet_radius_est", "duration_expected_ratio"]
STELLAR_FEATURES = ["Teff", "rad", "mass", "logg", "Tmag", "contratio"]
STELLAR_DEFAULTS = {"Teff": 5778.0, "rad": 1.0, "mass": 1.0,
                    "logg": 4.44, "Tmag": 10.0, "contratio": 0.0}
FEATURE_COLS = BLS_FEATURES + ENGINEERED_FEATURES + STELLAR_FEATURES


def load_clean():
    """Identical to train_model_v10.py / eval_holdout.py — keep in sync."""
    df = pd.read_csv("features_dataset.csv")
    if os.path.exists("features_dataset_kepler.csv"):
        df = pd.concat([df, pd.read_csv("features_dataset_kepler.csv")],
                       ignore_index=True)
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

    sp = pd.read_csv("stellar_params.csv")
    if os.path.exists("stellar_params_kepler.csv"):
        sp = pd.concat([sp, pd.read_csv("stellar_params_kepler.csv")],
                       ignore_index=True)
    sp["tic_id"] = sp["tic_id"].astype(str)
    df["tic_id"] = df["tic_id"].astype(str)
    keep = ["tic_id"] + [c for c in STELLAR_FEATURES if c in sp.columns]
    df = df.merge(sp[keep], on="tic_id", how="left")
    for c, v in STELLAR_DEFAULTS.items():
        df[c] = df[c].fillna(v) if c in df.columns else v

    df["planet_radius_est"] = np.sqrt(df["depth"].clip(lower=0)) * df["rad"] * 109.076
    t_exp = 13.0 * (df["period_days"] / 365.25) ** (1 / 3) * df["rad"] / df["mass"] ** (1 / 3)
    df["duration_expected_ratio"] = df["duration_hours"] / t_exp.clip(lower=1e-6)
    return df


def main():
    df = load_clean()
    X = df[FEATURE_COLS].values
    le = LabelEncoder()
    y = le.fit_transform(df["label"])
    planet_idx = int(np.where(le.classes_ == "planet")[0][0])

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42)
    print(f"Train {len(X_tr)} | holdout {len(X_te)} (untouched during sweep)")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 200, 900, step=50),
            max_depth=trial.suggest_int("max_depth", 3, 8),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
            reg_lambda=trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
            gamma=trial.suggest_float("gamma", 1e-4, 2.0, log=True),
            eval_metric="logloss", random_state=42, n_jobs=-1,
        )
        model = XGBClassifier(**params)
        return cross_val_score(model, X_tr, y_tr, cv=cv, scoring="roc_auc").mean()

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    # seed the search with the current hand-picked params as a baseline
    study.enqueue_trial(dict(n_estimators=400, max_depth=5, learning_rate=0.06,
                             subsample=0.85, colsample_bytree=0.8,
                             min_child_weight=3, reg_lambda=1.5,
                             reg_alpha=1e-3, gamma=1e-4))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

    print(f"\nBest CV ROC-AUC on train: {study.best_value:.4f}")
    print("Best params:", study.best_params)

    best = dict(study.best_params)
    best.update(eval_metric="logloss", random_state=42, n_jobs=-1)

    raw = XGBClassifier(**best)
    raw.fit(X_tr, y_tr)
    cal = CalibratedClassifierCV(XGBClassifier(**best), method="isotonic", cv=5)
    cal.fit(X_tr, y_tr)

    proba = cal.predict_proba(X_te)[:, planet_idx]
    pred = (proba >= 0.5).astype(int)
    yt = (y_te == planet_idx).astype(int)
    new_auc = roc_auc_score(yt, proba)
    new_acc = accuracy_score(yt, pred) * 100

    try:
        with open("model_metrics.json") as f:
            cur = json.load(f)
        cur_auc = cur.get("holdout_roc_auc", 0)
        cur_acc = cur.get("holdout_accuracy", 0)
    except Exception:
        cur_auc = cur_acc = 0

    print(f"\nHoldout: AUC {new_auc:.4f} (current {cur_auc:.4f}) | "
          f"acc {new_acc:.2f}% (current {cur_acc:.2f}%)")

    if new_auc <= cur_auc:
        print("Tuned model does NOT beat the current one on holdout AUC — "
              "keeping the existing model. (Best params saved to tuned_params.json "
              "for reference.)")
        with open("tuned_params.json", "w") as f:
            json.dump(study.best_params, f, indent=2)
        return

    metrics = {
        "version": "v10.1-tuned",
        "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
        "n_total": int(len(df)),
        "stellar_coverage_pct": round(float(df["Teff"].notna().mean() * 100), 1),
        "cv_roc_auc_train": round(float(study.best_value), 4),
        "holdout_accuracy": round(float(new_acc), 2),
        "holdout_precision": round(float(precision_score(yt, pred) * 100), 2),
        "holdout_recall": round(float(recall_score(yt, pred) * 100), 2),
        "holdout_f1": round(float(f1_score(yt, pred) * 100), 2),
        "holdout_roc_auc": round(float(new_auc), 4),
        "holdout_brier": round(float(brier_score_loss(yt, proba)), 4),
        "calibration": "isotonic (5-fold)",
        "tuning": f"optuna TPE, {N_TRIALS} trials, CV ROC-AUC on train split",
        "best_params": study.best_params,
        "feature_cols": FEATURE_COLS,
    }

    if os.path.exists("exoplanet_classifier.pkl"):
        os.replace("exoplanet_classifier.pkl", "exoplanet_classifier_prev.pkl")
    joblib.dump(cal, "exoplanet_classifier.pkl")
    joblib.dump(raw, "xgb_raw.pkl")
    joblib.dump(le, "label_encoder.pkl")
    joblib.dump(FEATURE_COLS, "feature_cols.pkl")
    with open("model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open("tuned_params.json", "w") as f:
        json.dump(study.best_params, f, indent=2)
    print("SAVED tuned model -> exoplanet_classifier.pkl (+ xgb_raw, metrics). "
          "Now run: python eval_holdout.py")


if __name__ == "__main__":
    main()
