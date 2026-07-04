# ExoDetect — Project Status & Handoff

> **For a new Claude session:** read this file first. It is the complete state
> of the project as of 2026-07-04. Working directory: `C:\Users\krish\planet`.
> Run the app with `python -m streamlit run dashboard.py` (system Python has
> all deps; see requirements.txt). There is also an older v8 copy of the app in
> `C:\Users\krish\Downloads\ExoDetect-v8.0-CORRECTED (4)\ExoDetect-v8.0` — ignore
> it unless asked; `planet/` is the active project.

## What this project is
**ExoDetect v10** — AI exoplanet detection from NASA TESS light curves.
Bharatiya Antariksh Hackathon 2026, Problem Statement 7. Team OrbitX2026
(Krishnendu Koley + 2, Jadavpur University). Streamlit app (`dashboard.py`,
~2000 lines) + `database.py` (SQLite persistence + ~25 Plotly figure builders).

Pipeline: TESS download (lightkurve/MAST, retry + FFI fallback) → detrend →
bounded BLS period search → phase fold → 17 features (11 light-curve +
6 stellar from TIC) → calibrated XGBoost → AI text insight.

## Current state (all working, verified)
- **Model v10**: `exoplanet_classifier.pkl` = isotonic-calibrated XGBoost;
  `xgb_raw.pkl` = raw booster for SHAP. Honest held-out metrics
  (`model_metrics.json`): accuracy 76.4%, recall 80.1%, ROC-AUC 0.875,
  Brier 0.146 on 334 never-seen stars. **The old "97.6%" was leakage-inflated;
  do not resurrect it.** Old model backed up as `exoplanet_classifier_v9_backup.pkl`.
- **Data**: `features_dataset.csv` 1,971 stars (1,666 clean);
  `stellar_params.csv` now covers ALL 1,971 (was 641);
  `toi_raw_full.csv` = fresh 8,035-TOI pull (`fetch_toi.py`);
  `training_targets.csv` = 2,499 labeled v10 targets;
  `frontier_targets.csv` = 5,162 unconfirmed PC/APC candidates.
- **Dashboard pages**: Individual Analysis (6 tabs incl. animated 3-D orbit
  simulator with habitable-zone physics), Compare Stars, 🛰️ Batch Survey
  (ranked candidate lists, frontier loader), 🗄️ Database Explorer (4 tabs,
  SQLite vault `exodetect.db`, ~15 interactive charts incl. two 3-D scatters,
  radar, parallel coords, HR diagram, play-button discovery animation),
  🗺️ Sky Map (3-D celestial sphere + RA/Dec map of all TOIs), 🎯 Model Honesty
  (ROC + reliability curves from `holdout_predictions.csv` via `eval_holdout.py`),
  Project Report, History.
- **Features**: SHAP per-prediction explanations (native XGBoost pred_contribs),
  decision-threshold modes in sidebar (Balanced 0.50 / Survey 0.35 / Vetting 0.65
  → global `planet_threshold` used in `run_pipeline`), TOI cross-match badge
  (local `db.lookup_toi`), next-transit ephemeris (UTC), auto-save of every
  analysis to SQLite, CSV/JSON exports everywhere, animated starfield CSS
  (scoped to `stAppViewContainer` — do NOT move it to `.stApp` children;
  that hid the sidebar once already).

## ✅ DATA JOB FINISHED 2026-07-04
The v10 extraction + retrain chain is DONE: features_dataset.csv = 2,137 stars
(1,797 clean: 863 planets / 934 FPs), stellar_params.csv = 2,137 (100% of ids),
model retrained with the 20 v10.1 features, then Optuna-tuned (tune_model.py,
60 TPE trials on the train split only; best params in tuned_params.json):
**v10.1-tuned holdout accuracy 78.61%, recall 79.2%, ROC-AUC 0.877, Brier
0.143 on 360 never-seen stars**. clean csv,
holdout_predictions.csv, tic_id_list.csv all regenerated and in sync.
The section below is kept for reference of the original procedure.

## DONE (was: IN PROGRESS) — the one unfinished job
`extract_features.py` is downloading light curves for the ~528 new v10 targets.
- Checkpoint: `features_dataset_partial.csv` (currently 2,045 rows; ~454 targets
  remain, many will SKIP). Resume-safe: just re-run `python extract_features.py`
  in a normal terminal (it skips everything in the checkpoint). Background jobs
  started by Claude die when the session ends — prefer the user's own terminal.
- **When it finishes** (writes final `features_dataset.csv`):
  1. Regenerate `tic_id_list.csv` from the new features_dataset
     (`pd.read_csv('features_dataset.csv')['tic_id'].unique()` → csv), then
     `python fetch_stellar_params.py` (top up new stars).
  2. `python train_model_v10.py` (retrains calibrated model, backs up old,
     writes model_metrics.json).
  3. `python eval_holdout.py` (refreshes holdout_predictions.csv for the
     Model Honesty page).
  4. Delete `features_dataset_clean.csv` or regenerate it — `load_catalog()` and
     `load_dataset_pool()` prefer it over features_dataset.csv, so a stale copy
     would keep the UI showing old counts.

## Known gotchas
- Windows console can't print emoji (cp1252) — avoid emoji in `print()` in scripts.
- Plotly `px.scatter` log axes want LINEAR `range_x/range_y` values (it logs them).
- `size=` in px.scatter rejects negatives/inf — use the sanitized `size_pw` column.
- `dashboard.py` at import defines pages in order: functions → CSS → sidebar →
  page `elif` chain. New pages = add to `st.sidebar.radio` list + an `elif` block.
- Duplicate v8 folders in Downloads (`ExoDetect-BAH2026*`, `zip_check`) — the
  user occasionally launches the wrong copy; check the footer version string.

## Done on 2026-07-04 (this session)
- **v10.1 engineered features**: `period_days`, `planet_radius_est`
  (sqrt(depth)·R★ in R⊕), `duration_expected_ratio` (observed/expected transit
  duration). Added identically in `train_model_v10.py`, `eval_holdout.py`, and
  `run_pipeline` in dashboard.py (feature_map). Model retrained on the current
  1,666 clean stars: holdout accuracy 77.25% (was 76.4%), AUC 0.876. Backup now
  goes to `exoplanet_classifier_prev.pkl` (v9 backup preserved separately).
- **🏆 Frontier Leaderboard page**: new `frontier_results` SQLite table
  (upserted from every Batch Survey run on a PC/APC candidate via
  `db.save_frontier_result`), leaderboard bar chart + period/radius scatter +
  full table + CSV/JSON export. `database.py`: `save_frontier_result`,
  `load_frontier_results`, `fig_frontier_leaderboard`, `fig_frontier_scatter`.
- **Plotly light-curve tabs**: analysis tabs 1–3 (light curve, periodogram,
  phase fold) converted from matplotlib to interactive Plotly Scattergl with a
  `_thin()` downsampler (40k pt cap). Tab 4/5 remain matplotlib (feed the PDF).
- **Sidebar**: live model-metrics card from model_metrics.json; version caption
  fixed to v10; `features_dataset_clean.csv` regenerated (1,666 rows, in sync).
- All 9 pages verified clean via `streamlit.testing.v1.AppTest`.

## Improvement backlog (discussed with user, in priority order)
1. **Finish the data job above** → ~2,400-star retrain (extraction was running
   in background on 2026-07-04; after it finishes run steps in IN PROGRESS above).
3. **Vetting report PDF v2** — per-star PDF with SHAP chart, calibrated prob,
   TOI cross-match, ephemeris (TFOP-memo style). Extend `generate_pdf()`.
4. **Kepler/K2 expansion** — same pipeline, ~10k labeled stars, cross-mission
   generalization. Long download job.
5. **CNN ensemble** — retrain the v8 1D-CNN on phase-folded curves, average
   probabilities with XGBoost.
6. **Plotly light-curve tabs** — analysis tabs 1–3 still matplotlib.
7. **Deployment** to Streamlit Community Cloud.
8. Batch-survey PDF export; tour mode for demos.
