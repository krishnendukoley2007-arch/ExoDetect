"""
extract_features_kepler.py
--------------------------
Step 2 of the Kepler expansion: download Kepler long-cadence light curves
and extract the SAME 11 BLS features as extract_features.py (TESS), so the
two datasets can be concatenated for training.

Differences vs the TESS extractor:
  - IDs are "KIC<kepid>" (never collides with TIC ids)
  - mission="Kepler", long cadence (30 min); up to 3 quarters stitched
  - flatten window_length=27 (~13.5 h at 30-min cadence — same physical
    window as 401 points at TESS 2-min cadence)
  - output: features_dataset_kepler.csv
    checkpoint: features_dataset_kepler_partial.csv (Ctrl+C safe, resumable)

Run AFTER build_kepler_targets.py. Run in ONE terminal only.
"""

import concurrent.futures
import os
import time
import warnings

import astropy.units as u
import lightkurve as lk
import numpy as np
import pandas as pd
from astropy.timeseries import BoxLeastSquares

warnings.filterwarnings("ignore")

print("=" * 60)
print("ExoDetect — Kepler expansion: extract features (11 BLS features)")
print("=" * 60)

targets = pd.read_csv("kepler_targets.csv")
print(f"\nTotal targets: {len(targets)}")
print(f"Label distribution:\n{targets['label'].value_counts().to_string()}\n")

checkpoint_file = "features_dataset_kepler_partial.csv"
done_ids = set()
features_list = []

if os.path.exists(checkpoint_file):
    existing = pd.read_csv(checkpoint_file)
    features_list = existing.to_dict("records")
    done_ids = set(existing["tic_id"].astype(str).tolist())
    print(f"  Resuming from checkpoint — {len(done_ids)} already done.\n")

success_count = 0
fail_count = 0


def to_clean_array(x):
    arr = np.asarray(x)
    if hasattr(arr, "filled"):
        arr = arr.filled(np.nan)
    return np.asarray(arr, dtype=float)


def _try_download(kepid, max_quarters):
    search = lk.search_lightcurve(f"KIC {kepid}", mission="Kepler",
                                  author="Kepler", exptime=1800)
    if len(search) > 0:
        n = min(max_quarters, len(search))
        lc = search[:n].download_all().stitch()
        return lc, n
    return None, 0


def download_lc(kepid, retry=True):
    """Kepler long cadence, up to 3 quarters, hard 90 s timeout per attempt."""
    for attempt in range(2 if retry else 1):
        print("    downloading Kepler LC...", end=" ", flush=True)
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                future = ex.submit(_try_download, kepid, 3)
                lc, n = future.result(timeout=90)
            if lc is not None:
                print(f"OK ({n} quarters)")
                return lc, f"Kepler-LC({n}q)"
            print("none found")
            return None, "No Kepler long-cadence data found"
        except concurrent.futures.TimeoutError:
            print("TIMEOUT (90s)")
        except Exception as e:
            print(f"failed ({e})")
        if attempt == 0 and retry:
            time.sleep(3)
    return None, "Download failed / timed out"


def run_bls_wide(lc, baseline_days):
    """Identical grid philosophy to the TESS extractor."""
    p_lo = 0.3
    p_hi = min(15.0, baseline_days / 3.0)
    if p_hi <= p_lo:
        return None, "period range invalid"

    periods = np.exp(np.linspace(np.log(p_lo), np.log(p_hi), 1500))
    max_dur = min(0.3, p_lo * 0.4)
    durations = np.linspace(0.02, max_dur, 6)
    durations = durations[durations > 0]

    try:
        t = to_clean_array(lc.time.value)
        f = to_clean_array(lc.flux.value)
        good = np.isfinite(t) & np.isfinite(f)
        t, f = t[good], f[good]
        if len(t) < 50:
            return None, "too few finite points for BLS"
        bls = BoxLeastSquares(t * u.day, f)
        return bls.power(periods * u.day, durations * u.day), None
    except Exception as e:
        return None, str(e)


def check_harmonic(pg, best_period_val):
    """Same harmonic correction as the TESS extractor."""
    periods_arr = to_clean_array(pg.period.value)
    power_arr = to_clean_array(pg.power)
    best_power = float(np.nanmax(power_arr))
    candidates = [(0.5, 0.60), (1.0 / 3.0, 0.60), (2.0, 0.90), (3.0, 0.90)]
    best_final_period = best_period_val
    best_final_power = best_power
    for factor, threshold in candidates:
        cand = best_period_val * factor
        idx = np.nanargmin(np.abs(periods_arr - cand))
        if abs(periods_arr[idx] - cand) / cand > 0.05:
            continue
        cand_power = float(power_arr[idx])
        is_shorter = cand < best_final_period
        is_longer = cand > best_final_period
        if cand_power >= best_power * threshold:
            if is_shorter:
                best_final_period = periods_arr[idx]
                best_final_power = cand_power
            elif is_longer and cand_power >= best_final_power * threshold:
                best_final_period = periods_arr[idx]
                best_final_power = cand_power
    return best_final_period


for idx, row in targets.iterrows():
    kepid = int(row["kepid"])
    star_id = f"KIC{kepid}"
    label = row["label"]
    known_period = float(row["pl_orbper"])

    if star_id in done_ids:
        continue

    print(f"[{idx+1}/{len(targets)}] {star_id} ({label})")

    try:
        lc_raw, source = download_lc(kepid)
        if lc_raw is None:
            print(f"  SKIP: {source}")
            fail_count += 1
            continue

        # 27 points x 30 min = ~13.5 h — same physical window as TESS 401x2min
        lc = lc_raw.normalize().flatten(window_length=27).remove_outliers(sigma=4)
        if len(lc) < 100:
            print(f"  SKIP: too few points ({len(lc)})")
            fail_count += 1
            continue

        t_all = to_clean_array(lc.time.value)
        baseline_days = float(np.nanmax(t_all) - np.nanmin(t_all))
        if baseline_days < 1.0:
            print(f"  SKIP: baseline too short ({baseline_days:.2f}d)")
            fail_count += 1
            continue

        pg, bls_err = run_bls_wide(lc, baseline_days)
        if pg is None:
            print(f"  SKIP: BLS failed ({bls_err})")
            fail_count += 1
            continue

        best_idx = np.argmax(pg.power)
        raw_best_period = float(pg.period[best_idx].value)
        best_power = float(pg.power[best_idx])
        best_period = check_harmonic(pg, raw_best_period)
        t0_val = float(pg.transit_time[best_idx].value)
        duration_days = float(pg.duration[best_idx].value)

        half_width = (duration_days / best_period) / 2

        folded = lc.fold(period=best_period, epoch_time=t0_val)
        phase_vals = to_clean_array(folded.time.value)
        flux_vals = to_clean_array(folded.flux.value)

        in_transit = np.abs(phase_vals) < half_width * 1.3
        out_transit = ((np.abs(phase_vals) > half_width * 3) &
                       (np.abs(phase_vals) < 0.45))
        secondary = np.abs(np.abs(phase_vals) - 0.5) < half_width * 1.3

        if np.sum(in_transit) < 3 or np.sum(out_transit) < 10:
            print("  SKIP: not enough transit points")
            fail_count += 1
            continue

        baseline = float(np.nanmedian(flux_vals[out_transit]))
        transit_med = float(np.nanmedian(flux_vals[in_transit]))
        depth = float(baseline - transit_med)
        noise = float(np.nanstd(flux_vals[out_transit]))
        n_in = int(np.sum(in_transit))
        snr = (depth / noise) * np.sqrt(n_in) if noise > 0 else 0.0

        sec_depth = 0.0
        if np.sum(secondary) > 3:
            sec_depth = float(baseline - np.nanmedian(flux_vals[secondary]))
        sec_ratio = (sec_depth / depth) if depth > 0 else 0.0

        t_arr = to_clean_array(lc.time.value)
        f_arr = to_clean_array(lc.flux.value)
        cycle_num = np.round((t_arr - t0_val) / best_period).astype(int)
        phase_global = ((t_arr - t0_val) % best_period) / best_period
        phase_global[phase_global > 0.5] -= 1
        in_g = np.abs(phase_global) < half_width * 1.3

        o_d = (float(np.nanmedian(f_arr[in_g & (cycle_num % 2 == 1)]))
               if np.sum(in_g & (cycle_num % 2 == 1)) > 2 else 1.0)
        e_d = (float(np.nanmedian(f_arr[in_g & (cycle_num % 2 == 0)]))
               if np.sum(in_g & (cycle_num % 2 == 0)) > 2 else 1.0)
        odd_even_diff = abs(o_d - e_d)

        duration_hours = duration_days * 24

        edge_mask = (np.abs(phase_vals) < half_width * 1.3) & (np.abs(phase_vals) > half_width * 0.7)
        center_mask = np.abs(phase_vals) < half_width * 0.3
        edge_depth = float(baseline - np.nanmedian(flux_vals[edge_mask])) if np.sum(edge_mask) > 2 else depth
        center_depth = float(baseline - np.nanmedian(flux_vals[center_mask])) if np.sum(center_mask) > 2 else depth
        transit_shape = (center_depth / edge_depth) if edge_depth > 1e-8 else 1.0

        dur_period_ratio = duration_hours / (best_period * 24)

        per_transit_depths = []
        for c in np.unique(cycle_num[in_g]):
            mask_c = in_g & (cycle_num == c)
            if np.sum(mask_c) > 1:
                per_transit_depths.append(baseline - float(np.nanmedian(f_arr[mask_c])))
        depth_consistency = (float(np.nanstd(per_transit_depths) /
                             (np.nanmean(per_transit_depths) + 1e-8))
                             if len(per_transit_depths) > 2 else 0.0)

        left_mask = (phase_vals < -half_width * 0.3) & (phase_vals > -half_width * 1.3)
        right_mask = (phase_vals > half_width * 0.3) & (phase_vals < half_width * 1.3)
        left_depth = float(baseline - np.nanmedian(flux_vals[left_mask])) if np.sum(left_mask) > 2 else depth
        right_depth = float(baseline - np.nanmedian(flux_vals[right_mask])) if np.sum(right_mask) > 2 else depth
        ingress_egress_asymmetry = abs(left_depth - right_depth) / (depth + 1e-8)

        odd_even_ratio = odd_even_diff / (depth + 1e-8)

        features_list.append({
            "tic_id": star_id,
            "label": label,
            "catalog_period": known_period,
            "period_days": best_period,
            "depth": depth,
            "snr": snr,
            "sec_ratio": sec_ratio,
            "duration_hours": duration_hours,
            "bls_power": best_power,
            "odd_even_diff": odd_even_diff,
            "transit_shape": transit_shape,
            "dur_period_ratio": dur_period_ratio,
            "depth_consistency": depth_consistency,
            "ingress_egress_asymmetry": ingress_egress_asymmetry,
            "odd_even_ratio": odd_even_ratio,
            "source": source,
        })
        done_ids.add(star_id)
        success_count += 1
        print(f"  OK [{source}] period={best_period:.4f}d depth={depth*100:.4f}% snr={snr:.1f}")

    except Exception as e:
        print(f"  ERROR: {e}")
        fail_count += 1

    if (idx + 1) % 5 == 0:
        pd.DataFrame(features_list).to_csv(checkpoint_file, index=False)
        print(f"  [Checkpoint saved — {len(features_list)} done so far]")

features_df = pd.DataFrame(features_list)
features_df.to_csv("features_dataset_kepler.csv", index=False)

print("\n" + "=" * 60)
print(f"DONE!  Success: {success_count}  |  Failed: {fail_count}")
print(f"Total features saved: {len(features_df)}")
if len(features_df):
    print("\nLabel distribution:")
    print(features_df["label"].value_counts().to_string())
print("\nNext step: python train_model_v10.py && python eval_holdout.py")
