"""
build_kepler_targets.py
-----------------------
Step 1 of the Kepler expansion: pull the KOI cumulative table from the
NASA Exoplanet Archive and build:

  kepler_targets.csv        — kepid, label (planet / false_positive), period
  stellar_params_kepler.csv — Teff/rad/mass/logg/Tmag per star, same column
                              names as stellar_params.csv so the training
                              scripts can simply concat the two files.

Label mapping (KOI dispositions):
  CONFIRMED       -> planet
  FALSE POSITIVE  -> false_positive
  CANDIDATE       -> skipped (unlabeled — not usable for training)

Stars whose KOIs disagree (one CONFIRMED + one FALSE POSITIVE entry) are
dropped entirely to keep labels clean.

IDs are stored as "KIC<kepid>" so they can never collide with TESS TIC ids.

Run: python build_kepler_targets.py
"""

import io

import numpy as np
import pandas as pd
import requests

MAX_PER_CLASS = 2500   # brightest-first cap per label; raise for more data

TAP = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
QUERY = (
    "select kepid,koi_disposition,koi_period,koi_depth,koi_duration,"
    "koi_steff,koi_srad,koi_smass,koi_slogg,koi_kepmag "
    "from cumulative"
)

print("Downloading KOI cumulative table from NASA Exoplanet Archive...")
resp = requests.get(TAP, params={"query": QUERY, "format": "csv"}, timeout=300)
resp.raise_for_status()
koi = pd.read_csv(io.StringIO(resp.text))
print(f"  {len(koi)} KOI rows")

label_map = {"CONFIRMED": "planet", "FALSE POSITIVE": "false_positive"}
koi["label"] = koi["koi_disposition"].map(label_map)
koi = koi.dropna(subset=["label", "koi_period"])

# Drop stars with conflicting labels across their KOIs
lab_per_star = koi.groupby("kepid")["label"].nunique()
conflicted = set(lab_per_star[lab_per_star > 1].index)
koi = koi[~koi["kepid"].isin(conflicted)]
print(f"  dropped {len(conflicted)} stars with conflicting KOI labels")

# One row per star: keep the deepest KOI (strongest signal to find again)
koi = koi.sort_values("koi_depth", ascending=False).drop_duplicates(subset="kepid")

# Brightest-first cap per class (bright = fast downloads, high SNR)
parts = []
for lab, grp in koi.groupby("label"):
    parts.append(grp.sort_values("koi_kepmag").head(MAX_PER_CLASS))
koi = pd.concat(parts, ignore_index=True)
print("  final label counts:")
print(koi["label"].value_counts().to_string())

targets = pd.DataFrame({
    "kepid": koi["kepid"].astype(int),
    "label": koi["label"],
    "pl_orbper": koi["koi_period"],
})
targets.to_csv("kepler_targets.csv", index=False)
print(f"Saved kepler_targets.csv ({len(targets)} stars)")

stellar = pd.DataFrame({
    "tic_id": "KIC" + koi["kepid"].astype(int).astype(str),
    "Teff": koi["koi_steff"],
    "rad": koi["koi_srad"],
    "mass": koi["koi_smass"],
    "logg": koi["koi_slogg"],
    "Tmag": koi["koi_kepmag"],      # Kepler mag ~ comparable brightness scale
    "contratio": np.nan,            # not available for KOIs -> default fills in
})
stellar.to_csv("stellar_params_kepler.csv", index=False)
print(f"Saved stellar_params_kepler.csv ({len(stellar)} stars)")
print("\nNext step: python extract_features_kepler.py")
