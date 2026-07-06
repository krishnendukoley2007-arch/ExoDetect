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

## Current model — v10.2 (cross-mission TESS+Kepler, honest 20% holdout)
- `exoplanet_classifier.pkl` = isotonic-calibrated XGBoost, 21 features
  (20 previous + `mission` flag 0=TESS/1=Kepler); `xgb_raw.pkl` for SHAP;
  `feature_cols.pkl` = 21 cols. scale_pos_weight set from train split.
- **Holdout (751 never-seen stars): accuracy 82.56%, precision 81.03%,
  recall 84.72%, F1 82.83%, ROC-AUC 0.9089, Brier 0.124.**
  Per mission: TESS 78.38% / AUC 0.8718 · Kepler 85.89% / AUC 0.9332.
  Journey: 76.4% → 78.6% (v10.1) → 82.6% (v10.2 Kepler merge).
  The old "97.6%" was leakage — never resurrect.
- Threshold sweep (eval_holdout.py prints it): best F1 at default 0.50;
  t=0.70 gives 91.9% precision — candidate for frontier vetting mode.
- Feature lists/filters/engineered formulas live ONLY in features_config.py
  (shared by train/eval/tune + dashboard fallback). Dashboard sends mission=0.
- Data: features_dataset.csv 2,137 TESS + features_dataset_kepler.csv 2,620
  Kepler (extraction finished 2026-07-05: 2,164 success of first pass + a
  final retry pass; 4,427 targeted). Clean training set 3,751 stars
  (1,861 planet / 1,890 FP); features_dataset_clean.csv in sync.
  stellar_params.csv 2,137 + stellar_params_kepler.csv (96.5% coverage).
  toi_raw_full.csv 8,035 TOIs; frontier_targets.csv 5,162 unconfirmed PC/APC.
- Retrain chain unchanged: train_model_v10.py && eval_holdout.py &&
  tune_model.py (tune only replaces model if it beats holdout AUC). After
  retrain regenerate features_dataset_clean.csv (see refactor note below).
- First post-merge tune (60 trials) did NOT beat the untuned v10.2 model —
  best params kept in tuned_params.json for reference only.
- Dashboard stays TESS-only for live analysis; Kepler only feeds training.

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

## Refactor 2026-07-04 (late) — pre-Kepler-merge hardening
- **features_config.py is now the single source of truth**: feature lists,
  physical filters, stellar merge, engineered features, and `load_clean()` —
  imported by train_model_v10.py, eval_holdout.py, tune_model.py, and
  dashboard.py (as FEATURE_COLS fallback). Never copy-paste lists again.
- **New `mission` feature** (0=TESS, 1=KIC prefix) → next retrain is v10.2 with
  21 features; absorbs Kepmag-in-Tmag + missing-contratio systematics.
  Dashboard feature_map sends mission=0 (live analysis is TESS-only).
- train + tune now set `scale_pos_weight`; train + eval print per-mission
  holdout metrics; load_clean logs filter drops per mission.
- eval_holdout.py prints a threshold sweep (verified: best F1 at t=0.30;
  t=0.75 gives 91.7% precision — candidate for frontier leaderboard mode).
- Dashboard: synthetic-RF fallback removed (missing model = st.error + stop),
  stale 17-feature fallback list removed, CSV-backed db loaders cached
  (5-min TTL; SQLite loaders deliberately uncached), silent excepts now
  print warnings, period min>=max guard, 10 MB upload cap, regex-escaped
  search, "Sun-like defaults" warning when stellar params missing.
- Verified: eval_holdout reproduces exactly 78.61%/0.8769 (shared cleaning is
  byte-identical); AppTest passes on all pages. NOT yet committed/pushed.

## Session 2026-07-05 (later) — UI upgrades + cloud DB seeded
- **Feature experiment (honest negative result)**: a_over_rstar / teq_est /
  stellar_density / transit_prob dropped holdout 82.56→81.49, AUC
  0.9089→0.9045 → reverted to the 21-feature v10.2 model (git restore of
  artifacts). Formulas remain in features_config.add_engineered_features
  (display use) but are EXCLUDED from ENGINEERED_FEATURES — see note there.
- Dashboard run_pipeline now computes engineered features via
  features_config.add_engineered_features on a one-row DataFrame — training
  and live formulas physically cannot drift.
- **Frontier Leaderboard: 🥇 high-confidence tier** (proba ≥ 0.70) with
  measured holdout precision quoted live from holdout_predictions.csv.
- **Model Honesty: per-mission ROC overlay** (TESS vs Kepler dotted curves)
  in db.fig_roc_curve when the holdout has both missions.
- **BLS "too large" bug FIXED**: lightkurve's size pre-check uses
  frequency_factor even when an explicit period grid is passed; we now pass
  frequency_factor=1e6 (unused for the real search) — long-baseline stars
  (e.g. TIC 428673146) analyze fine now.
- **Cloud DB seeded** via new seed_db.py (bare-mode import of dashboard,
  catalog-tuned 0.7–1.4× period windows): 78 analyses, 16 frontier verdicts
  in committed exodetect.db.

## Session 2026-07-06 — report graphs + wider frontier survey
- generate_slide_graphs.py (planet folder): 5 deck-ready PNGs + SLIDE_NUMBERS.txt
  in slide_graphs/, all computed live from model_metrics.json /
  holdout_predictions.csv / xgb_raw.pkl — rerun after any retrain.
- Project Report page: stale v8/v9 sections (RF+GB 87%, CNN table, 97.60%
  bullet) replaced with a Model Evolution table, the 5 evidence graphs
  (loaded from slide_graphs/), and v10.2 improvement bullets.
- Sidebar + footer read version/counts from model_metrics.json — no
  hardcoded metrics remain anywhere in the UI.
- Frontier survey widened: 27 verdicts / 89 analyses in committed
  exodetect.db. Best open candidate 63.4% — none clear the 0.70
  high-confidence bar yet (honest talking point: no cheap planet calls).

## Remaining backlog (priority order)
1. **Finish Kepler** (user's terminal) → retrain → update README metrics → push.
2. **Seed cloud DB**: run 10–15 best stars locally, commit exodetect.db, push
   (so the live site opens with a rich history + frontier leaderboard).
3. CNN ensemble (retrain v8 1D-CNN on folded curves, average with XGBoost).
4. Batch-survey PDF export; tour/demo mode.
5. Optional: persistent cloud DB (Supabase/Turso free tier) instead of seeded
   SQLite.
