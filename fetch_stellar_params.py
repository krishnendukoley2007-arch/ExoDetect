"""
fetch_stellar_params.py
------------------------
Pulls TIC/CTL stellar host-star parameters (Teff, radius, mass, logg,
density, contamination ratio, magnitude, distance, extinction, etc.)
for ONLY the TIC IDs already in our labeled dataset (features_dataset.csv).

This avoids downloading the full-sky TIC/CTL catalogs (0.5-10.7 GB per
declination band, ~500 GB total) — we query MAST's TIC catalog API
directly for our ~641-700 known stars instead, which returns a file
of only a few hundred KB.

Run this on YOUR machine (needs internet access to archive.stsci.edu /
mast.stsci.edu — the analysis sandbox cannot reach it).

Requirements:
    pip install astroquery pandas

Output:
    stellar_params.csv  <-- send this file back, it's small (<1 MB)
"""

import pandas as pd
from astroquery.mast import Catalogs
import time

INPUT_CSV = "tic_id_list.csv"     # list of tic_id (from features_dataset.csv)
OUTPUT_CSV = "stellar_params.csv"

# Columns we actually want from the TIC catalog for modeling
KEEP_COLS = [
    "ID", "Tmag", "Teff", "e_Teff", "logg", "e_logg", "rad", "e_rad",
    "mass", "e_mass", "rho", "e_rho", "lum", "MH", "d", "e_d",
    "ebv", "contratio", "numcont", "priority", "disposition",
    "gaiabp", "gaiarp", "plx", "e_plx", "Vmag", "Jmag", "Hmag", "Kmag",
]

def main():
    ids = pd.read_csv(INPUT_CSV)["tic_id"].astype(str).tolist()
    print(f"Querying TIC catalog for {len(ids)} stars...")

    rows = []
    batch_size = 50
    for i in range(0, len(ids), batch_size):
        batch = ids[i:i + batch_size]
        try:
            result = Catalogs.query_criteria(catalog="Tic", ID=batch).to_pandas()
            rows.append(result)
            print(f"  {i + len(batch)}/{len(ids)} done")
        except Exception as e:
            print(f"  batch {i} failed: {e}")
        time.sleep(0.5)  # be polite to the API

    if not rows:
        print("No data retrieved.")
        return

    df = pd.concat(rows, ignore_index=True)
    available_cols = [c for c in KEEP_COLS if c in df.columns]
    df = df[available_cols].drop_duplicates(subset="ID")
    df = df.rename(columns={"ID": "tic_id"})
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {len(df)} rows -> {OUTPUT_CSV} (send this file back)")

if __name__ == "__main__":
    main()
