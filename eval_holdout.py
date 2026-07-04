"""
eval_holdout.py
---------------
Regenerates the v10 held-out test predictions (same seed/split as
train_model_v10.py) and saves them to holdout_predictions.csv so the
dashboard can draw honest ROC + calibration (reliability) curves.

Run after train_model_v10.py. Takes seconds.
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

BLS_FEATURES = ["depth", "snr", "sec_ratio", "duration_hours", "bls_power",
                "odd_even_diff", "transit_shape", "dur_period_ratio",
                "depth_consistency", "ingress_egress_asymmetry", "odd_even_ratio"]
STELLAR_FEATURES = ["Teff", "rad", "mass", "logg", "Tmag", "contratio"]
STELLAR_DEFAULTS = {"Teff": 5778.0, "rad": 1.0, "mass": 1.0,
                    "logg": 4.44, "Tmag": 10.0, "contratio": 0.0}


def load_clean():
    """Identical cleaning to train_model_v10.py — keep in sync."""
    import os
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

    # v10.1 engineered features — keep identical to train_model_v10.py
    df["planet_radius_est"] = np.sqrt(df["depth"].clip(lower=0)) * df["rad"] * 109.076
    t_exp = 13.0 * (df["period_days"] / 365.25) ** (1 / 3) * df["rad"] / df["mass"] ** (1 / 3)
    df["duration_expected_ratio"] = df["duration_hours"] / t_exp.clip(lower=1e-6)
    return df


def main():
    df = load_clean()
    cols = joblib.load("feature_cols.pkl")
    cal = joblib.load("exoplanet_classifier.pkl")
    le = joblib.load("label_encoder.pkl")
    planet_idx = int(np.where(le.classes_ == "planet")[0][0])

    X = df[cols].values
    y = le.transform(df["label"])
    idx = np.arange(len(df))
    _, idx_te, _, y_te = train_test_split(
        idx, y, test_size=0.20, stratify=y, random_state=42)

    proba = cal.predict_proba(X[idx_te])[:, planet_idx]
    out = pd.DataFrame({
        "tic_id": df["tic_id"].values[idx_te],
        "y_true": (y_te == planet_idx).astype(int),
        "planet_proba": proba,
    })
    out.to_csv("holdout_predictions.csv", index=False)
    print(f"Saved holdout_predictions.csv — {len(out)} held-out stars "
          f"({out['y_true'].sum()} planets / {(1-out['y_true']).sum()} FPs)")


if __name__ == "__main__":
    main()
