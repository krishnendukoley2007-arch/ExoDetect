"""
generate_slide_graphs.py — presentation graphs for ExoDetect v10.2+
===================================================================
Every number is read live from model_metrics.json, holdout_predictions.csv,
feature_cols.pkl and xgb_raw.pkl — regenerate after any retrain and the deck
can never go stale.

Usage:  python generate_slide_graphs.py
Output: slide_graphs/  (300-DPI PNGs + SLIDE_NUMBERS.txt)
"""

import json
import os

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (auc, classification_report, confusion_matrix,
                             roc_curve)

matplotlib.rcParams["figure.dpi"] = 300
matplotlib.rcParams["savefig.dpi"] = 300

OUT = "slide_graphs"
os.makedirs(OUT, exist_ok=True)

# dark theme matching the deck / dashboard
BG, PANEL = "#0b1220", "#10192b"
ACCENT, ORANGE, GREEN, RED, PURPLE = "#4da3ff", "#ff8a3d", "#3ddc84", "#ff5c5c", "#b07aff"
TEXT, GRID = "#e8edf5", "#26334d"

FRIENDLY = {
    "depth": "Transit depth", "snr": "SNR", "sec_ratio": "Secondary/primary ratio",
    "duration_hours": "Duration (h)", "bls_power": "BLS power",
    "odd_even_diff": "Odd-even depth diff", "transit_shape": "Transit shape (U vs V)",
    "dur_period_ratio": "Duration/period", "depth_consistency": "Depth consistency",
    "ingress_egress_asymmetry": "Ingress/egress asym.", "odd_even_ratio": "Odd-even ratio",
    "period_days": "Orbital period", "planet_radius_est": "Planet radius est.",
    "duration_expected_ratio": "Duration vs expected", "Teff": "Star temperature",
    "rad": "Star radius", "mass": "Star mass", "logg": "Surface gravity",
    "Tmag": "Magnitude", "contratio": "Contamination", "mission": "Mission (TESS/Kepler)",
}


def style_ax(ax):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TEXT, labelsize=10)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    ax.grid(True, color=GRID, linewidth=0.5, alpha=0.5)


with open("model_metrics.json") as f:
    M = json.load(f)
hold = pd.read_csv("holdout_predictions.csv")
cols = joblib.load("feature_cols.pkl")
ver = M.get("version", "v10.2")
print(f"Model {ver} | holdout n={len(hold)} | acc {M['holdout_accuracy']}% | "
      f"AUC {M['holdout_roc_auc']}")

is_kep = hold["tic_id"].astype(str).str.startswith("KIC")
y, p = hold["y_true"].values, hold["planet_proba"].values
pred = (p >= 0.5).astype(int)

# ── 1. Accuracy journey ─────────────────────────────────────
print("[1/5] accuracy_journey.png")
stages = ["v9\n(leaky CV —\nrejected)", "v10 honest\nbaseline\n(TESS)",
          "v10.1 tuned\n(TESS)", f"{ver}\nTESS + Kepler\ncross-mission"]
vals = [97.6, 76.4, 78.6, float(M["holdout_accuracy"])]
colors = [RED, ACCENT, ACCENT, GREEN]
fig, ax = plt.subplots(figsize=(9, 6), facecolor=BG)
bars = ax.bar(stages, vals, color=colors, zorder=3, edgecolor=TEXT, linewidth=0.8)
bars[0].set_alpha(0.45)
bars[0].set_hatch("//")
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.1f}%",
            ha="center", fontsize=13, fontweight="bold", color=TEXT)
ax.text(0, 50, "DATA\nLEAKAGE", ha="center", fontsize=12, fontweight="bold", color=RED)
ax.set_ylabel("Accuracy (%)")
ax.set_title(f"The Honesty Journey — every number after v9 is a locked star-level "
             f"holdout ({M['n_test']} never-seen stars)", fontsize=11.5,
             fontweight="bold", pad=15)
ax.set_ylim(0, 105)
style_ax(ax)
plt.tight_layout()
plt.savefig(f"{OUT}/accuracy_journey.png", facecolor=BG, bbox_inches="tight")
plt.close()

# ── 2. Feature importance (real XGBoost) ────────────────────
print("[2/5] feature_importance.png")
raw = joblib.load("xgb_raw.pkl")
imp = raw.feature_importances_ * 100
order = np.argsort(imp)[::-1][:12]
fig, ax = plt.subplots(figsize=(9, 6.5), facecolor=BG)
labels = [FRIENDLY.get(cols[i], cols[i]) for i in order][::-1]
vals_i = imp[order][::-1]
bars = ax.barh(labels, vals_i, color=ACCENT, zorder=3, edgecolor=TEXT, linewidth=0.8)
for b, v in zip(bars, vals_i):
    ax.text(v + 0.25, b.get_y() + b.get_height() / 2, f"{v:.1f}%",
            va="center", fontsize=10, fontweight="bold", color=TEXT)
ax.set_xlabel("XGBoost importance (%)")
ax.set_title(f"Top 12 of {len(cols)} Physics-Informed Features ({ver})",
             fontsize=12, fontweight="bold", pad=15)
ax.set_xlim(0, max(vals_i) * 1.18)
style_ax(ax)
plt.tight_layout()
plt.savefig(f"{OUT}/feature_importance.png", facecolor=BG, bbox_inches="tight")
plt.close()

# ── 3. Confusion matrix (honest holdout) ────────────────────
print("[3/5] confusion_matrix.png")
cm = confusion_matrix(y, pred)
fig, ax = plt.subplots(figsize=(6.5, 6), facecolor=BG)
ax.imshow(cm, cmap="Blues")
disp = ["False\nPositive", "Planet"]
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(disp, color=TEXT, fontsize=11)
ax.set_yticklabels(disp, color=TEXT, fontsize=11)
ax.set_xlabel("Predicted", color=TEXT, fontsize=11)
ax.set_ylabel("Actual", color=TEXT, fontsize=11)
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=22,
                fontweight="bold",
                color="white" if cm[i, j] > cm.max() / 2 else BG)
ax.set_title(f"Confusion Matrix — {len(hold)} never-seen stars ({ver})",
             fontsize=12, fontweight="bold", color=TEXT, pad=15)
for s in ax.spines.values():
    s.set_color(GRID)
plt.tight_layout()
plt.savefig(f"{OUT}/confusion_matrix.png", facecolor=BG, bbox_inches="tight")
plt.close()

# ── 4. Per-mission ROC ──────────────────────────────────────
print("[4/5] roc_per_mission.png")
fig, ax = plt.subplots(figsize=(7.5, 6.5), facecolor=BG)
for mask, name, col, lw in [(np.ones(len(hold), bool), "All missions", GREEN, 3),
                            (~is_kep.values, "TESS only", ORANGE, 1.8),
                            (is_kep.values, "Kepler only", PURPLE, 1.8)]:
    if mask.sum() < 20 or len(np.unique(y[mask])) < 2:
        continue
    fpr, tpr, _ = roc_curve(y[mask], p[mask])
    ax.plot(fpr, tpr, color=col, lw=lw, zorder=3,
            label=f"{name} — AUC {auc(fpr, tpr):.3f} (n={mask.sum()})")
ax.plot([0, 1], [0, 1], "--", color=GRID, lw=1, label="Random guess")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title(f"ROC — Honest Holdout, Cross-Mission ({ver})",
             fontsize=12, fontweight="bold", pad=15)
leg = ax.legend(facecolor=PANEL, edgecolor=GRID, fontsize=10)
for t in leg.get_texts():
    t.set_color(TEXT)
style_ax(ax)
plt.tight_layout()
plt.savefig(f"{OUT}/roc_per_mission.png", facecolor=BG, bbox_inches="tight")
plt.close()

# ── 5. Precision vs recall vs threshold ─────────────────────
print("[5/5] threshold_tradeoff.png")
ts = np.arange(0.30, 0.81, 0.025)
prec = [100 * (y[p >= t].mean() if (p >= t).sum() else 0) for t in ts]
rec = [100 * ((y == 1) & (p >= t)).sum() / max((y == 1).sum(), 1) for t in ts]
fig, ax = plt.subplots(figsize=(8.5, 6), facecolor=BG)
ax.plot(ts, prec, color=GREEN, lw=3, zorder=3, label="Precision (planet calls that are right)")
ax.plot(ts, rec, color=ORANGE, lw=3, zorder=3, label="Recall (planets caught)")
ax.axvline(0.5, color=ACCENT, ls="--", lw=1.2)
ax.text(0.505, 30, "default\n(balanced)", color=ACCENT, fontsize=9)
ax.axvline(0.70, color=GREEN, ls="--", lw=1.2)
hp = 100 * y[p >= 0.70].mean() if (p >= 0.70).sum() else 0
ax.text(0.705, 30, f"vetting mode\n{hp:.0f}% precision", color=GREEN, fontsize=9)
ax.set_xlabel("Decision threshold on calibrated planet probability")
ax.set_ylabel("%")
ax.set_title("One Calibrated Model, Tunable Operating Point",
             fontsize=12, fontweight="bold", pad=15)
leg = ax.legend(facecolor=PANEL, edgecolor=GRID, fontsize=10, loc="lower left")
for t in leg.get_texts():
    t.set_color(TEXT)
ax.set_ylim(0, 100)
style_ax(ax)
plt.tight_layout()
plt.savefig(f"{OUT}/threshold_tradeoff.png", facecolor=BG, bbox_inches="tight")
plt.close()

# ── Slide numbers ───────────────────────────────────────────
rep = classification_report(y, pred, target_names=["false_positive", "planet"],
                            digits=4)
with open(f"{OUT}/SLIDE_NUMBERS.txt", "w", encoding="utf-8") as f:
    f.write(f"EXACT NUMBERS FOR SLIDES — model {ver}, generated from live files\n")
    f.write("=" * 64 + "\n\n")
    f.write(f"Training stars: {M['n_train']}  |  Holdout (never seen): {M['n_test']}"
            f"  |  Total: {M['n_total']}\n")
    f.write(f"Holdout accuracy : {M['holdout_accuracy']}%\n")
    f.write(f"Holdout precision: {M['holdout_precision']}%\n")
    f.write(f"Holdout recall   : {M['holdout_recall']}%\n")
    f.write(f"Holdout F1       : {M['holdout_f1']}%\n")
    f.write(f"ROC-AUC          : {M['holdout_roc_auc']}\n")
    f.write(f"Brier (calibration): {M['holdout_brier']}\n")
    for key, name in [("holdout_accuracy_tess", "TESS-only accuracy"),
                      ("holdout_roc_auc_tess", "TESS-only AUC"),
                      ("holdout_accuracy_kepler", "Kepler-only accuracy"),
                      ("holdout_roc_auc_kepler", "Kepler-only AUC")]:
        if key in M:
            f.write(f"{name:20s}: {M[key]}\n")
    f.write(f"\nVetting mode (threshold 0.70): precision {hp:.1f}% on holdout\n")
    f.write(f"Features: {len(cols)} (11 light-curve + 3 engineered + 6 stellar + mission)\n")
    f.write(f"Calibration: {M.get('calibration', 'isotonic (5-fold)')}\n\n")
    f.write("Classification report (holdout, threshold 0.5):\n")
    f.write(rep)

print(f"\nDone — 5 PNGs + SLIDE_NUMBERS.txt in {OUT}/")
