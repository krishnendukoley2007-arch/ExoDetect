"""
seed_db.py — analyze a list of stars and persist results to exodetect.db,
so the committed database seeds the cloud deployment with a populated
History + Frontier Leaderboard.

Usage: python seed_db.py <tic_id> [<tic_id> ...]
Uses the dashboard's own run_pipeline (bare-mode Streamlit import), so seeded
rows are identical to what live analyses would produce.
"""

import os
import sys

import pandas as pd

import dashboard as dash  # bare-mode import: st calls become no-ops
import database as db


def catalog_window(tid):
    """0.7x-1.4x of the TOI catalog period — same rule as the dashboard's
    auto-tune button. Falls back to the default 1-20 d window."""
    try:
        if os.path.exists("frontier_targets.csv"):
            f = pd.read_csv("frontier_targets.csv")
            row = f[f["tid"].astype(str) == str(tid)]
            if not row.empty and pd.notna(row.iloc[0]["pl_orbper"]):
                p = float(row.iloc[0]["pl_orbper"])
                return max(0.5, 0.7 * p), 1.4 * p
    except Exception as e:
        print(f"    window lookup failed ({e}) — using default")
    return 1.0, 20.0


def seed(tic_ids):
    ok = fail = 0
    for tid in tic_ids:
        tid = str(tid).strip()
        pmin, pmax = catalog_window(tid)
        print(f"--- TIC {tid} (window {pmin:.2f}-{pmax:.2f} d) ...", flush=True)
        try:
            r = dash.run_pipeline(tid, max_sectors=5, period_min=pmin, period_max=pmax)
        except Exception as e:
            print(f"    pipeline crashed: {e}")
            fail += 1
            continue
        if r.get("error"):
            print(f"    FAILED: {str(r['error'])[:140]}")
            fail += 1
            continue
        db.save_analysis(r, insight=dash.generate_ai_insight(r))
        stored = db.save_frontier_result(r)
        print(f"    OK  P={r['period_days']:.3f} d  proba={r.get('planet_proba', 0)*100:.1f}%  "
              f"{r['ml_class']}  frontier={'yes' if stored else 'no'}")
        ok += 1
    print(f"\nDone: {ok} saved, {fail} failed. Now commit exodetect.db and push.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: python seed_db.py <tic_id> [...]")
    seed(sys.argv[1:])
