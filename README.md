# 🪐 ExoDetect v10 — AI Exoplanet Detection from NASA TESS Light Curves

**Bharatiya Antariksh Hackathon 2026 — Problem Statement 7 · Team OrbitX2026 · Jadavpur University**

ExoDetect downloads real NASA TESS photometry for any star, hunts for transits
with a Box Least Squares period search, extracts 20 physics-informed features,
and classifies the signal with an isotonic-calibrated XGBoost model — with
honest, held-out metrics and per-prediction SHAP explanations.

## Honest model performance (360 never-seen stars)
| Metric | Value |
|---|---|
| Accuracy | 78.6% |
| Recall (planets caught) | 79.2% |
| ROC-AUC | 0.877 |
| Brier score (calibration) | 0.143 |

Trained on 1,797 quality-filtered NASA TOI stars (Kepler expansion to ~6,000 in
progress). No leakage: a 20% star-level holdout is locked away before any
training or tuning.

## Run locally
```bash
pip install -r requirements.txt
python -m streamlit run dashboard.py
```

## Pages
- **🔭 Individual Analysis** — full pipeline on any TIC ID, with period-check
  guardrails, one-click auto-fix, SHAP explanation, 3-D orbit simulator
- **⚖️ Compare Stars** · **🛰️ Batch Survey** — ranked candidate vetting queue
- **🏆 Frontier Leaderboard** — accumulated verdicts on 5,162 unconfirmed TOIs
- **🗄️ Database Explorer** · **🗺️ Sky Map** · **🎯 Model Honesty** (ROC +
  reliability curves) · **📄 Project Report** · **📜 History**

## Team OrbitX2026
Krishnendu Koley · Abhradeep Bera · Asmit Dey — Jadavpur University, Kolkata
