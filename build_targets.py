"""
build_targets.py  (v9 — quality-first rebuild)
================================================
Builds training_targets.csv from NASA TOI catalog.

KEY CHANGES vs v8:
  - CP + KP both mapped to "planet" (only CONFIRMED stars)
  - PC (Planet Candidate) SKIPPED — unconfirmed, noisy labels
  - APC SKIPPED — too ambiguous
  - FP + FA both mapped to "false_positive"
  - No artificial cap — use ALL quality labeled stars (~2,400)
  - Deduplication by TIC ID (multi-planet systems kept as one entry)
  - Period filter: 0.3d to 30d (wider than v8's 0.5-30d)

Run this FIRST, then extract_features.py.
"""

import pandas as pd
import numpy as np

print("=" * 60)
print("ExoDetect v9 — Step 1: Build Training Targets")
print("=" * 60)

# ── 1. Load TOI catalog ────────────────────────────────────
print("\n[1/4] Loading toi_raw.csv...")
df = pd.read_csv("toi_raw.csv", comment="#")
print(f"  Total TOI entries: {len(df)}")
print(f"\n  Disposition counts:")
print(df["tfopwg_disp"].value_counts().to_string())

# ── 2. Map dispositions — quality labels only ──────────────
print("\n[2/4] Mapping dispositions to labels...")
print("  CP + KP  → planet        (confirmed only)")
print("  FP + FA  → false_positive")
print("  PC + APC → SKIPPED       (unconfirmed, noisy)")

def map_label(disp):
    d = str(disp).strip().upper()
    if d in ["CP", "KP"]:
        return "planet"
    elif d in ["FP", "FA"]:
        return "false_positive"
    else:
        return None  # PC, APC, unknown — skip

df["label"] = df["tfopwg_disp"].apply(map_label)
df_filtered = df[df["label"].notna()].copy()

print(f"\n  After filtering:")
print(f"  {df_filtered['label'].value_counts().to_string()}")
print(f"  Total usable: {len(df_filtered)}")

# ── 3. Period + depth quality filter ──────────────────────
print("\n[3/4] Applying quality filters...")

before = len(df_filtered)
df_filtered = df_filtered[
    df_filtered["pl_orbper"].notna() &
    (df_filtered["pl_orbper"] >= 0.3) &
    (df_filtered["pl_orbper"] <= 30.0) &
    df_filtered["pl_trandep"].notna() &
    (df_filtered["pl_trandep"] > 0)
]
print(f"  Removed {before - len(df_filtered)} rows with missing/invalid period or depth")
print(f"  Remaining: {len(df_filtered)}")

# ── 4. Deduplicate by TIC ID ───────────────────────────────
# Multi-planet systems: keep the entry with the longest period
# (usually the outermost planet — strongest photometric signal)
print("\n[4/4] Deduplicating by TIC ID...")
before = len(df_filtered)
df_filtered = df_filtered.sort_values("pl_orbper", ascending=False)
df_filtered = df_filtered.drop_duplicates(subset="tid", keep="first")
print(f"  Removed {before - len(df_filtered)} duplicate TIC IDs")
print(f"  Final unique stars: {len(df_filtered)}")

# ── 5. Build final targets ─────────────────────────────────
final = df_filtered[["tid", "pl_orbper", "label", "pl_trandep"]].copy()
final.columns = ["tid", "pl_orbper", "label", "pl_trandep"]
final["tid"] = final["tid"].astype(int)
final = final.sort_values("label").reset_index(drop=True)

# Shuffle for training diversity
final = final.sample(frac=1, random_state=42).reset_index(drop=True)

final.to_csv("training_targets.csv", index=False)

print("\n" + "=" * 60)
print("DONE! Final training_targets.csv:")
print(f"  {final['label'].value_counts().to_string()}")
print(f"  Total: {len(final)} stars")
print(f"\n  Depth distribution (ppm):")
print(f"  Planet:         {final[final['label']=='planet']['pl_trandep'].describe()[['min','50%','max']].to_string()}")
print(f"  False Positive: {final[final['label']=='false_positive']['pl_trandep'].describe()[['min','50%','max']].to_string()}")
print("\n  Next step: python extract_features.py")
print("=" * 60)
