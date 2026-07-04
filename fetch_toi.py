"""
fetch_toi.py  (v10)
-------------------
Pulls the LIVE TESS Objects of Interest catalog (~7,400 targets) from the
NASA Exoplanet Archive TAP service and splits it into:

  toi_raw_full.csv          — everything, all columns we use
  training_targets_v10.csv  — labeled rows for training
                              (CP/KP -> planet, FP/FA -> false_positive)
  frontier_targets.csv      — unlabeled PC/APC candidates: the "frontier"
                              your model can rank for scientists

Run: python fetch_toi.py   (needs internet)
"""

import io
import urllib.request
import urllib.parse

import pandas as pd

TAP = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
QUERY = """
SELECT toi, tid, tfopwg_disp, pl_orbper, pl_trandep, pl_trandurh,
       pl_tranmid, pl_rade, pl_insol, pl_eqt,
       st_tmag, st_teff, st_rad, st_logg, st_dist, ra, dec
FROM toi
"""

LABEL_MAP = {
    "CP": "planet",          # confirmed planet
    "KP": "planet",          # known planet
    "FP": "false_positive",  # false positive
    "FA": "false_positive",  # false alarm
}


def main():
    url = TAP + "?" + urllib.parse.urlencode(
        {"query": " ".join(QUERY.split()), "format": "csv"})
    print("Querying NASA Exoplanet Archive TOI table...")
    with urllib.request.urlopen(url, timeout=180) as resp:
        raw = resp.read().decode()
    df = pd.read_csv(io.StringIO(raw))
    print(f"  Received {len(df)} TOIs")
    print(df["tfopwg_disp"].value_counts(dropna=False).to_string())

    df.to_csv("toi_raw_full.csv", index=False)

    labeled = df[df["tfopwg_disp"].isin(LABEL_MAP)].copy()
    labeled["label"] = labeled["tfopwg_disp"].map(LABEL_MAP)
    labeled = labeled.dropna(subset=["pl_orbper"])
    labeled = labeled.rename(columns={"pl_orbper": "pl_orbper"})
    out = labeled[["tid", "pl_orbper", "label", "pl_trandep", "pl_tranmid"]] \
        .drop_duplicates(subset="tid")
    out.to_csv("training_targets_v10.csv", index=False)
    print(f"\n  Labeled training targets : {len(out)} "
          f"({(out['label']=='planet').sum()} planet / "
          f"{(out['label']=='false_positive').sum()} FP)")

    frontier = df[df["tfopwg_disp"].isin(["PC", "APC"])].copy()
    frontier = frontier.dropna(subset=["pl_orbper"]).drop_duplicates(subset="tid")
    frontier.to_csv("frontier_targets.csv", index=False)
    print(f"  Frontier (unlabeled PC) : {len(frontier)} candidates")
    print("\nSaved: toi_raw_full.csv, training_targets_v10.csv, frontier_targets.csv")
    print("Next: run extract_features.py over the new targets, then train_model_v10.py")


if __name__ == "__main__":
    main()
