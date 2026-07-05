"""
features_config.py
==================
SINGLE SOURCE OF TRUTH for the model's feature lists, cleaning filters and
engineered-feature formulas. Imported by train_model_v10.py, eval_holdout.py,
tune_model.py and dashboard.py — edit HERE, never copy-paste lists again.
"""

import os

import numpy as np
import pandas as pd

BLS_FEATURES = ["depth", "snr", "sec_ratio", "duration_hours", "bls_power",
                "odd_even_diff", "transit_shape", "dur_period_ratio",
                "depth_consistency", "ingress_egress_asymmetry", "odd_even_ratio"]

# v10.1 engineered features — combine light-curve + stellar physics:
#   period_days             raw orbital period
#   planet_radius_est       sqrt(depth) x stellar radius, in Earth radii —
#                           implausibly large => eclipsing binary
#   duration_expected_ratio observed / expected transit duration for a
#                           circular orbit around this star (classic vetting)
# NOTE: a_over_rstar / teq_est / stellar_density / transit_prob are computed
# by add_engineered_features (used for display/physics), but EXCLUDED from the
# model: tested 2026-07-05, they dropped holdout acc 82.56→81.49 and AUC
# 0.9089→0.9045 — redundant with period+stellar params, added noise.
ENGINEERED_FEATURES = ["period_days", "planet_radius_est", "duration_expected_ratio"]

STELLAR_FEATURES = ["Teff", "rad", "mass", "logg", "Tmag", "contratio"]

# v10.2: mission flag (0 = TESS, 1 = Kepler). Kepler stars carry Kepmag in the
# Tmag column (different passband) and have no contamination ratio — this flag
# lets the model absorb those systematic offsets instead of being misled.
MISSION_FEATURES = ["mission"]

FEATURE_COLS = BLS_FEATURES + ENGINEERED_FEATURES + STELLAR_FEATURES + MISSION_FEATURES

STELLAR_DEFAULTS = {"Teff": 5778.0, "rad": 1.0, "mass": 1.0,
                    "logg": 4.44, "Tmag": 10.0, "contratio": 0.0}


def add_mission_flag(df):
    """mission = 1 for Kepler ('KIC...') ids, 0 for TESS."""
    df["mission"] = df["tic_id"].astype(str).str.startswith("KIC").astype(int)
    return df


def add_engineered_features(df):
    """The engineered physics features (v10.1 + v10.3). df needs depth, rad,
    mass, Teff, period_days, duration_hours columns. Keep this THE only
    implementation — the dashboard calls it too (on a one-row DataFrame)."""
    df["planet_radius_est"] = np.sqrt(df["depth"].clip(lower=0)) * df["rad"] * 109.076
    t_exp = 13.0 * (df["period_days"] / 365.25) ** (1 / 3) * df["rad"] / df["mass"] ** (1 / 3)
    df["duration_expected_ratio"] = df["duration_hours"] / t_exp.clip(lower=1e-6)

    # v10.3 — Kepler's 3rd law: a[AU] = (P[yr]^2 * M[Msun])^(1/3); 1 AU = 215.03 Rsun
    a_au = ((df["period_days"] / 365.25) ** 2 * df["mass"].clip(lower=1e-6)) ** (1 / 3)
    a_rsun = a_au * 215.032
    df["a_over_rstar"] = a_rsun / df["rad"].clip(lower=1e-6)
    df["teq_est"] = df["Teff"] * np.sqrt(1.0 / (2.0 * df["a_over_rstar"].clip(lower=1e-6)))
    df["stellar_density"] = df["mass"] / df["rad"].clip(lower=1e-6) ** 3
    df["transit_prob"] = (1.0 / df["a_over_rstar"].clip(lower=1e-6)).clip(upper=1.0)
    return df


def _mission_counts(df):
    kep = df["tic_id"].astype(str).str.startswith("KIC").sum()
    return len(df) - kep, kep


def load_clean(verbose=True):
    """Load features + Kepler expansion, apply physical filters, merge stellar
    params, add engineered + mission features. Shared by all three training
    scripts so cleaning can never drift out of sync."""
    df = pd.read_csv("features_dataset.csv")
    if os.path.exists("features_dataset_kepler.csv"):
        kep = pd.read_csv("features_dataset_kepler.csv")
        if verbose:
            print(f"  + Kepler expansion: {len(kep)} stars")
        df = pd.concat([df, kep], ignore_index=True)

    tess_before, kep_before = _mission_counts(df)
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
    tess_after, kep_after = _mission_counts(df)
    if verbose:
        print(f"  Physical filters dropped: TESS {tess_before - tess_after} "
              f"({tess_before}->{tess_after}), Kepler {kep_before - kep_after} "
              f"({kep_before}->{kep_after})")

    sp = pd.read_csv("stellar_params.csv")
    if os.path.exists("stellar_params_kepler.csv"):
        sp = pd.concat([sp, pd.read_csv("stellar_params_kepler.csv")],
                       ignore_index=True)
    sp["tic_id"] = sp["tic_id"].astype(str)
    df["tic_id"] = df["tic_id"].astype(str)
    keep = ["tic_id"] + [c for c in STELLAR_FEATURES if c in sp.columns]
    df = df.merge(sp[keep], on="tic_id", how="left")
    df.attrs["stellar_coverage"] = float(df["Teff"].notna().mean()) if "Teff" in df.columns else 0.0
    for c, v in STELLAR_DEFAULTS.items():
        df[c] = df[c].fillna(v) if c in df.columns else v

    df = add_engineered_features(df)
    df = add_mission_flag(df)
    return df
