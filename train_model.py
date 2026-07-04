"""
train_model.py  (v9 — XGBoost + stellar params + proper validation)
=====================================================================
Trains ExoDetect v9 classifier on 11 BLS features + stellar params.

UPGRADES vs v8:
  - Proper data cleaning (outlier removal, physical range filters)
  - Stellar params merged (Teff, rad, mass, logg, contratio)
  - XGBoost replaces RF+GB VotingClassifier
  - Optuna hyperparameter tuning (50 trials)
  - Stratified train/val/test split (70/15/15)
  - Full metrics: accuracy, precision, recall, F1, ROC-AUC
  - Confusion matrix saved as PNG
  - Feature importance plot saved as PNG

Run AFTER extract_features.py completes.
"""

import pandas as pd
import numpy as np
import joblib
import warnings
import os
warnings.filterwarnings("ignore")

print("=" * 60)
print("ExoDetect v9 — Step 3: Train Model")
print("=" * 60)

# ── 1. Load data ───────────────────────────────────────────
print("\n[1/6] Loading features dataset...")
df = pd.read_csv("features_dataset.csv")
print(f"  Raw rows: {len(df)}")
print(f"  {df['label'].value_counts().to_string()}")

# ── 2. Data cleaning ───────────────────────────────────────
print("\n[2/6] Cleaning data...")

BLS_FEATURES = [
    "depth", "snr", "sec_ratio", "duration_hours", "bls_power", "odd_even_diff"
]
PHYSICS_FEATURES = [
    "transit_shape", "dur_period_ratio", "depth_consistency",
    "ingress_egress_asymmetry", "odd_even_ratio"
]
ALL_BLS = BLS_FEATURES + PHYSICS_FEATURES

before = len(df)

# Physical range filters — remove clearly broken rows
df = df[df["depth"] > 0]                        # depth must be positive
df = df[df["depth"] < 0.5]                      # depth < 50% (not a total eclipse)
df = df[df["snr"] > 1.0]                        # SNR must be meaningful
df = df[df["snr"] < 10000]                      # no absurd SNR
df = df[df["sec_ratio"] >= -0.5]               # sec_ratio can be slightly negative (noise)
df = df[df["sec_ratio"] < 5.0]                 # but not wildly positive
df = df[df["duration_hours"] > 0]              # duration must be positive
df = df[df["bls_power"] > 0]                   # BLS power must be positive
df = df[df["transit_shape"].between(-2, 10)]   # shape physically bounded
df = df[df["depth_consistency"].between(-5, 20)]
df = df[df["ingress_egress_asymmetry"].between(-5, 100)]
df = df[df["odd_even_ratio"].between(-5, 100)]

# Replace inf with nan then drop
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=ALL_BLS)

print(f"  Removed {before - len(df)} bad rows")
print(f"  Clean rows: {len(df)}")
print(f"  {df['label'].value_counts().to_string()}")

# ── 3. Merge stellar params ────────────────────────────────
print("\n[3/6] Merging stellar parameters...")

STELLAR_FEATURES = []

if os.path.exists("stellar_params.csv"):
    stellar = pd.read_csv("stellar_params.csv")
    stellar["tic_id"] = stellar["tic_id"].astype(str)
    df["tic_id"] = df["tic_id"].astype(str)

    # Select most useful stellar features (low missing %)
    USE_STELLAR = ["Teff", "rad", "mass", "logg", "Tmag", "contratio"]
    available = [c for c in USE_STELLAR if c in stellar.columns]
    stellar_sub = stellar[["tic_id"] + available].copy()

    df = df.merge(stellar_sub, on="tic_id", how="left")

    # Fill missing stellar params with class median (not global median)
    for col in available:
        for lbl in ["planet", "false_positive"]:
            mask = df["label"] == lbl
            median_val = df.loc[mask, col].median()
            df.loc[mask & df[col].isna(), col] = median_val

    STELLAR_FEATURES = available
    print(f"  Merged stellar features: {available}")
    print(f"  Missing after fill: {df[available].isna().sum().sum()}")
else:
    print("  stellar_params.csv not found — skipping stellar features")
    print("  (Run fetch_stellar_params.py first for better accuracy)")

FEATURE_COLS = ALL_BLS + STELLAR_FEATURES
print(f"\n  Total features used: {len(FEATURE_COLS)}")
print(f"  {FEATURE_COLS}")

# ── 4. Train/Val/Test split ────────────────────────────────
print("\n[4/6] Splitting data (70% train / 15% val / 15% test)...")
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

X = df[FEATURE_COLS].values
y = df["label"].values

le = LabelEncoder()
y_enc = le.fit_transform(y)
print(f"  Classes: {list(le.classes_)}")

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y_enc, test_size=0.15, random_state=42, stratify=y_enc
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp
)  # 0.176 of 0.85 ≈ 0.15 of total

print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

# ── 5. Optuna hyperparameter tuning ───────────────────────
print("\n[5/6] Tuning XGBoost with Optuna (50 trials)...")
import xgboost as xgb
import optuna
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial):
    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 200, 800),
        "max_depth":         trial.suggest_int("max_depth", 3, 8),
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight":  trial.suggest_int("min_child_weight", 1, 10),
        "gamma":             trial.suggest_float("gamma", 0.0, 1.0),
        "reg_alpha":         trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda":        trial.suggest_float("reg_lambda", 0.5, 2.0),
        "random_state": 42,
        "eval_metric": "logloss",
        "use_label_encoder": False,
    }
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              verbose=False)
    preds = model.predict(X_val)
    return accuracy_score(y_val, preds)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50, show_progress_bar=True)

best_params = study.best_params
best_params.update({"random_state": 42, "eval_metric": "logloss", "use_label_encoder": False})
print(f"\n  Best val accuracy: {study.best_value*100:.2f}%")
print(f"  Best params: {best_params}")

# ── 6. Train final model + evaluate ───────────────────────
print("\n[6/6] Training final model on train+val, evaluating on test...")

X_trainval = np.vstack([X_train, X_val])
y_trainval = np.concatenate([y_train, y_val])

final_model = xgb.XGBClassifier(**best_params)
final_model.fit(X_trainval, y_trainval, verbose=False)

# Save model
joblib.dump(final_model, "exoplanet_classifier.pkl")
joblib.dump(le, "label_encoder.pkl")
joblib.dump(FEATURE_COLS, "feature_cols.pkl")
print("  Saved: exoplanet_classifier.pkl, label_encoder.pkl, feature_cols.pkl")

# Test set evaluation
y_pred = final_model.predict(X_test)
y_prob = final_model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print(f"\n{'='*60}")
print(f"FINAL TEST SET RESULTS:")
print(f"  Accuracy:  {acc*100:.2f}%")
print(f"  ROC-AUC:   {auc:.4f}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# 5-fold CV on full clean data for comparison
from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(
    xgb.XGBClassifier(**best_params),
    X, y_enc, cv=5, scoring="accuracy"
)
print(f"5-Fold CV Accuracy: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")

# Confusion matrix plot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

cm = confusion_matrix(y_test, y_pred)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor("#0d1117")

# Confusion matrix
ax = axes[0]
ax.set_facecolor("#161b22")
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
ax.set_title(f"Confusion Matrix\nAcc={acc*100:.1f}% | AUC={auc:.3f}",
             color="white", fontsize=12)
ax.set_xlabel("Predicted", color="white")
ax.set_ylabel("True", color="white")
ax.tick_params(colors="white")

# Feature importance
ax2 = axes[1]
ax2.set_facecolor("#161b22")
importances = final_model.feature_importances_
sorted_idx = np.argsort(importances)[::-1]
top_n = min(15, len(FEATURE_COLS))
colors = ["#f97316" if i < 6 else "#3b82f6" if i < 11 else "#a855f7"
          for i in range(top_n)]
bars = ax2.barh(
    [FEATURE_COLS[i] for i in sorted_idx[:top_n]][::-1],
    [importances[i] for i in sorted_idx[:top_n]][::-1],
    color=colors[::-1]
)
ax2.set_title("Feature Importance (Top 15)", color="white", fontsize=12)
ax2.set_xlabel("Importance", color="white")
ax2.tick_params(colors="white")
ax2.spines[["top", "right"]].set_visible(False)

# Legend
from matplotlib.patches import Patch
legend = [
    Patch(color="#f97316", label="BLS features (6)"),
    Patch(color="#3b82f6", label="Physics features (5)"),
    Patch(color="#a855f7", label="Stellar params"),
]
ax2.legend(handles=legend, facecolor="#161b22", labelcolor="white", fontsize=8)

plt.suptitle(f"ExoDetect v9 | {len(df)} stars | {len(FEATURE_COLS)} features",
             color="white", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig("model_validation_v9.png", dpi=150, bbox_inches="tight",
            facecolor="#0d1117")
plt.close()
print("\n  Saved: model_validation_v9.png")

print(f"\n{'='*60}")
print("DONE! Model ready.")
print(f"  Stars trained on: {len(df)}")
print(f"  Features used:    {len(FEATURE_COLS)}")
print(f"  Test Accuracy:    {acc*100:.2f}%")
print(f"  ROC-AUC:          {auc:.4f}")
print(f"\nNext step: streamlit run dashboard.py")
print("=" * 60)
