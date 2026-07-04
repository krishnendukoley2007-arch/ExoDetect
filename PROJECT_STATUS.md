# ExoDetect — Project Status & Handoff

> **For a new Claude session:** read this file first. Complete state as of
> 2026-07-04 (evening). Working directory: `C:\Users\krish\planet`.
> Run locally: `python -m streamlit run dashboard.py` (system Python has all
> deps). Ignore the old v8 copies in `C:\Users\krish\Downloads\ExoDetect-*`.

## What this project is
**ExoDetect v10.1** — AI exoplanet detection from NASA TESS light curves.
Bharatiya Antariksh Hackathon 2026, PS7. Team OrbitX2026 (Krishnendu Koley
+ Abhradeep Bera + Asmit Dey, Jadavpur University).
Streamlit app: `dashboard.py` (~2300 lines) + `database.py` (SQLite +
~30 Plotly figure builders).

Pipeline: TESS download (lightkurve/MAST, retry + FFI fallback + 60s
timeouts) → detrend → BLS period search → phase fold → **20 features**
(11 light-curve + 3 engineered + 6 stellar) → isotonic-calibrated XGBoost →
SHAP explanation → AI text insight.

## DEPLOYED & PUBLIC
- **GitHub**: https://github.com/krishnendukoley2007-arch/ExoDetect (public,
  `main`; ~10 MB, includes model + catalogs + seeded exodetect.db)
- **Live app**: https://exodetect-wwahs5zzrdsayuhbzfxtfs.streamlit.app/
  (Streamlit Community Cloud, auto-redeploys on every `git push`)
- Cloud caveats: SQLite resets on app reboot (committed exodetect.db is the
  seed state); app sleeps after ~12 h idle (first visit wakes it, ~30 s).
- Push flow: `git add -A && git commit && git pull --rebase origin main &&
  git push` (user sometimes edits README on github.com — always rebase first).

## Current model — v10.1-tuned (all honest, star-level 20% holdout)
- `exoplanet_classifier.pkl` = isotonic-calibrated XGBoost (Optuna-tuned:
  n_est 550, depth 6, lr 0.0155, λ=3.0, mcw 6 — full params in
  tuned_params.json); `xgb_raw.pkl` for SHAP; `feature_cols.pkl` = 20 cols.
- **Holdout (360 never-seen stars): accuracy 78.61%, precision 76.97%,
  recall 79.19%, F1 78.06%, ROC-AUC 0.8769, Brier 0.143** (model_metrics.json).
  Journey today: 76.4% → 78.6%. The old "97.6%" was leakage — never resurrect.
- v10.1 engineered features (IDENTICAL in train_model_v10.py, eval_holdout.py,
  tune_model.py, and dashboard run_pipeline feature_map — keep in sync!):
  `period_days`, `planet_radius_est` = sqrt(depth)·R★·109.076,
  `duration_expected_ratio` = duration_hours / (13·(P/365.25)^⅓·R★/M★^⅓).
- Data: features_dataset.csv 2,137 TESS stars (1,797 clean = 863 planet /
  934 FP; features_dataset_clean.csv in sync); stellar_params.csv 2,137
  (100% coverage); toi_raw_full.csv 8,035 TOIs; frontier_targets.csv 5,162
  unconfirmed PC/APC.
- Retrain chain: `python train_model_v10.py && python eval_holdout.py &&
  python tune_model.py` (tune only replaces model if it beats holdout AUC;
  backup → exoplanet_classifier_prev.pkl). After retrain also regenerate
  features_dataset_clean.csv (same filters as train script) or UI counts stale.

## IN PROGRESS — Kepler expansion (running in USER'S terminal)
`extract_features_kepler.py` is downloading 4,427 labeled Kepler stars
(1,927 planet / 2,500 FP from KOI cumulative, brightest-first, conflicting
labels dropped). IDs stored as "KIC<kepid>" (never collides with TIC).
- Checkpoint: features_dataset_kepler_partial.csv (Ctrl+C safe, rerun resumes).
- Takes 1–3 days total; stopping early is fine (even ~1,500 stars is a big win).
- `kepler_targets.csv` + `stellar_params_kepler.csv` (from KOI table) already
  built by `build_kepler_targets.py`.
- When done (or stopped): the retrain chain above AUTO-DETECTS and merges
  features_dataset_kepler.csv + stellar_params_kepler.csv (all three training
  scripts concat them if present). Then update README metrics table + git push.
- Dashboard stays TESS-only for live analysis (correct — KIC stars have no
  TESS pipeline); Kepler only feeds training.

## Dashboard pages (9, all verified via streamlit.testing.v1 AppTest)
Individual Analysis (6 tabs; tabs 1–3 now interactive Plotly Scattergl with
`_thin()` 40k-pt downsampler; tabs 4–5 matplotlib — they feed PDFs; 3-D orbit
sim), Compare Stars, 🛰️ Batch Survey (frontier loader; every PC/APC result
upserted to SQLite `frontier_results`), **🏆 Frontier Leaderboard** (NEW —
top-N most-likely-real unconfirmed planets, coverage progress, charts,
CSV/JSON export), 🗄️ Database Explorer, 🗺️ Sky Map, 🎯 Model Honesty
(ROC + reliability from holdout_predictions.csv), 📄 Project Report, 📜 History.

## Key features added 2026-07-04
- **Period guardrails** in render_result: green ✅ when detected period
  matches TOI catalog; harmonic (P/2, 2P) warning; mismatch error; edge-of-
  window warning. Each problem has a **"🔧 Fix it for me" button** →
  sets `_pending_period_window` + `_auto_reanalyze` in session_state →
  applied at top of script BEFORE sliders instantiate (Streamlit forbids
  writing widget keys after creation — this pattern is load-bearing).
- Sidebar period sliders have keys `period_min`/`period_max`; auto-tune uses
  0.7×–1.4× of catalog period (0.5×/1.5× included harmonics — don't regress).
- Individual Analysis: input sanitized to digits ("TIC 123" ok), pre-flight
  TOI card with 📐 auto-tune button, demo buttons with captions, settings
  caption, Analyze disabled when empty.
- **Vetting Report PDF v2** (`generate_pdf`): 3 pages — verdict banner +
  signal/host tables + [PASS]/[FLAG] vetting checklist (secondary eclipse,
  odd/even, U-vs-V shape, asymmetry, depth consistency) + cross-match +
  next-3-transit ephemeris; SHAP bar page; evidence plots page.
- Sidebar shows live model report card from model_metrics.json.
- `tune_model.py` — Optuna sweep (60 TPE trials, CV AUC on train split only,
  saves only if beats incumbent holdout AUC).

## Known gotchas
- Windows console cp1252 → no emoji in `print()` in scripts.
- Plotly px.scatter log axes want LINEAR range values; size= rejects neg/inf.
- Starfield CSS scoped to `stAppViewContainer` — moving it hid the sidebar once.
- dashboard.py page structure: functions → CSS → sidebar → `elif page ==`
  chain. New page = radio list entry + elif block.
- features_dataset_clean.csv preferred by load_catalog — regen after retrains.
- exodetect.db committed to git = cloud seed; local runs modify it (dirty
  git status is normal).

## Remaining backlog (priority order)
1. **Finish Kepler** (user's terminal) → retrain → update README metrics → push.
2. **Seed cloud DB**: run 10–15 best stars locally, commit exodetect.db, push
   (so the live site opens with a rich history + frontier leaderboard).
3. CNN ensemble (retrain v8 1D-CNN on folded curves, average with XGBoost).
4. Batch-survey PDF export; tour/demo mode.
5. Optional: persistent cloud DB (Supabase/Turso free tier) instead of seeded
   SQLite.
