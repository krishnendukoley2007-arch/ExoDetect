# 🪐 ExoDetect v10.2 — AI Exoplanet Detection from NASA TESS Light Curves

**Bharatiya Antariksh Hackathon 2026 — Problem Statement 7 · Team OrbitX2026 · Jadavpur University**
https://exodetect-wwahs5zzrdsayuhbzfxtfs.streamlit.app/ just click and open our project(If it does not work then follow the steps below. )

ExoDetect downloads real NASA TESS photometry for any star, hunts for transits
with a Box Least Squares period search, extracts 21 physics-informed features,
and classifies the signal with an isotonic-calibrated XGBoost model — with
honest, held-out metrics and per-prediction SHAP explanations.

## Honest model performance (751 never-seen stars)
| Metric | Value |
|---|---|
| Accuracy | 82.6% |
| Recall (planets caught) | 84.7% |
| ROC-AUC | 0.909 |
| Brier score (calibration) | 0.124 |

Per mission: TESS holdout 78.4% (AUC 0.872) · Kepler holdout 85.9% (AUC 0.933).

Trained on 3,751 quality-filtered NASA stars — 1,797 TESS TOIs + 1,954 Kepler
KOIs (cross-mission training with a mission flag). No leakage: a 20% star-level
holdout is locked away before any training or tuning.

## 🖥️ How to run ExoDetect on your own computer

No experience needed — follow these steps exactly.

### Step 1 — Install Python (skip if you already have it)

1. Go to https://www.python.org/downloads/ and download **Python 3.11 or newer**.
2. Run the installer. **IMPORTANT:** on the first screen, tick the checkbox
   **"Add Python to PATH"** before clicking Install.
3. To check it worked, open a terminal
   (**Windows:** press `Win + R`, type `cmd`, press Enter ·
   **Mac:** open the Terminal app) and type:
   ```bash
   python --version
   ```
   You should see something like `Python 3.11.9`. On Mac/Linux, if that fails,
   try `python3 --version` (and use `python3` in every command below).

### Step 2 — Download this project

**Option A — no tools needed (easiest):**
1. On this GitHub page, click the green **`<> Code`** button → **Download ZIP**.
2. Unzip it somewhere easy, e.g. your Desktop. You'll get a folder called
   `ExoDetect-main`.

**Option B — with git (if you have it/https://git-scm.com/install/windows download):**
```bash
git clone https://github.com/krishnendukoley2007-arch/ExoDetect.git
```

### Step 3 — Open a terminal inside the project folder

```bash
cd ExoDetect
```
(Adjust the path to wherever you unzipped/cloned it. Tip for Windows: you can
also open the folder in File Explorer, click the address bar, type `cmd`, and
press Enter — a terminal opens already in the right place.)

### Step 4 — Install the required libraries (one-time, ~5 minutes)

```bash
pip install -r requirements.txt
```
Wait for it to finish — it downloads Streamlit, NASA's lightkurve, XGBoost,
Plotly and friends. If you see `pip is not recognized`, use:
```bash
python -m pip install -r requirements.txt
```

### Step 5 — Launch the app 🚀

```bash
python -m streamlit run dashboard.py
```
Your browser opens automatically at **http://localhost:8501** with the
ExoDetect dashboard. Keep the terminal window open while you use the app —
closing it stops the app.

### Step 6 — Try it!

Click one of the demo buttons (**🌍 Pi Mensae c** is a great first star), then
**🔭 Analyze**. The app downloads real NASA TESS data live, so the first
analysis takes 30–90 seconds — that's normal.

### Troubleshooting

| Problem | Fix |
|---|---|
| `python` / `pip` not recognized | Reinstall Python and tick **"Add Python to PATH"**, then reopen the terminal |
| Install fails on one package | Run `python -m pip install --upgrade pip` then retry Step 4 |
| Browser doesn't open | Open http://localhost:8501 manually |
| "No TESS data found" for a star | That star has no 2-min-cadence data — try a demo button star instead |
| Analysis very slow / fails | NASA's MAST server is busy — wait 30 s and click Analyze again |
| Port already in use | Run `python -m streamlit run dashboard.py --server.port 8502` |

## Pages
- **🔭 Individual Analysis** — full pipeline on any TIC ID, with period-check
  guardrails, one-click auto-fix, SHAP explanation, 3-D orbit simulator
- **⚖️ Compare Stars** · **🛰️ Batch Survey** — ranked candidate vetting queue
- **🏆 Frontier Leaderboard** — accumulated verdicts on 5,162 unconfirmed TOIs
- **🗄️ Database Explorer** · **🗺️ Sky Map** · **🎯 Model Honesty** (ROC +
  reliability curves) · **📄 Project Report** · **📜 History**

## Team OrbitX2026
Krishnendu Koley · Abhradeep Bera · Asmit Dey — Jadavpur University, Kolkata
