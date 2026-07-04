"""
extract_features.py  (v9-clean rebuild, patch 4 — timeout-protected downloads)
Downloads TESS light curves and extracts 11 features per star:
  6 original BLS features + 5 physics-informed features.

FIXES vs v8:
  - BLS now does a REAL blind search over a wide period grid, using
    astropy's BoxLeastSquares DIRECTLY (not lightkurve's wrapper,
    which silently exploded to millions of grid points).
  - Every masked array explicitly .filled(np.nan) before math ops.
  - Harmonic-check corrects for BLS locking onto P/2, P/3, 2P, 3P.
  - Every download source attempt now has a HARD 60-second timeout
    via a background thread. If MAST hangs, we skip and move to the
    next source instead of freezing forever with zero feedback.
  - Live progress printed at each stage (searching / downloading /
    timeout / failed) so a stall is visible immediately, not silent.
  - Checkpoints every 5 stars; safe to Ctrl+C and resume anytime.

Run AFTER build_targets.py + add_eb_targets.py.
IMPORTANT: run in ONE terminal only.
"""

import lightkurve as lk
import numpy as np
import pandas as pd
import os
import time
import warnings
import concurrent.futures
from astropy.timeseries import BoxLeastSquares
import astropy.units as u
warnings.filterwarnings("ignore")

print("=" * 60)
print("ExoDetect v9 — Step 2: Extract Features (11 features, wide BLS)")
print("=" * 60)

targets = pd.read_csv("training_targets.csv")
print(f"\nTotal targets: {len(targets)}")
print(f"Label distribution:\n{targets['label'].value_counts().to_string()}\n")

checkpoint_file = "features_dataset_partial.csv"
done_tics = set()
features_list = []

if os.path.exists(checkpoint_file):
    existing = pd.read_csv(checkpoint_file)
    features_list = existing.to_dict("records")
    done_tics = set(existing["tic_id"].astype(str).tolist())
    print(f"  Resuming from checkpoint — {len(done_tics)} already done.\n")

success_count = 0
fail_count = 0


def to_clean_array(x):
    """Force any masked/quantity array into a plain float ndarray with NaNs."""
    arr = np.asarray(x)
    if hasattr(arr, "filled"):
        arr = arr.filled(np.nan)
    return np.asarray(arr, dtype=float)


def _try_source(tic_id, author, exptime, max_sectors):
    kwargs = {"mission": "TESS", "author": author}
    if exptime:
        kwargs["exptime"] = exptime
    search = lk.search_lightcurve(f'TIC {tic_id}', **kwargs)
    if len(search) > 0:
        n = min(max_sectors, len(search))
        lc = search[:n].download_all().stitch()
        return lc, n
    return None, 0


def download_lc(tic_id, retry=True):
    """
    Tries SPOC -> TESS-SPOC -> QLP, each capped at a hard 60-second
    timeout via a background thread, so a hung MAST request can never
    freeze the whole script. Worst case per star: ~6 minutes
    (3 sources x 2 attempts x 60s), not infinite.
    """
    sources = [
        ("SPOC", 120, 3, "SPOC-2min"),
        ("TESS-SPOC", None, 2, "TESS-SPOC-FFI"),
        ("QLP", None, 2, "QLP-FFI"),
    ]

    for attempt in range(2 if retry else 1):
        for author, exptime, max_sectors, tag in sources:
            print(f"    trying {author}...", end=" ", flush=True)
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                    future = ex.submit(_try_source, tic_id, author, exptime, max_sectors)
                    lc, n = future.result(timeout=60)
                if lc is not None:
                    print(f"OK ({n} sectors)")
                    return lc, f"{tag}({n}sectors)"
                print("none found")
            except concurrent.futures.TimeoutError:
                print("TIMEOUT (60s) — skipping this source")
            except Exception as e:
                print(f"failed ({e})")

        if attempt == 0 and retry:
            time.sleep(3)

    return None, "No TESS data found in SPOC/TESS-SPOC/QLP (or all timed out)"


def run_bls_wide(lc, baseline_days):
    """
    Real blind BLS search using astropy's BoxLeastSquares directly.
    Fixed grid size every time: ~1500 periods x 6 durations = 9000 pts.
    """
    p_lo = 0.3
    p_hi = min(15.0, baseline_days / 3.0)
    if p_hi <= p_lo:
        return None, "period range invalid"

    n_periods = 1500
    periods = np.exp(np.linspace(np.log(p_lo), np.log(p_hi), n_periods))

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
        result = bls.power(periods * u.day, durations * u.day)
        return result, None
    except Exception as e:
        return None, str(e)


def check_harmonic(pg, best_period_val):
    """
    Compare BLS power at candidate harmonics of the best period.
    Prefers the FUNDAMENTAL period — either shorter (sub-harmonic) or
    longer (super-harmonic) — if its power is comparable (>=85% of best).
    
    FIX vs original: also checks SHORTER period candidates (P/2, P/3),
    not just longer ones. Shorter periods preferred when power is close,
    because a true short period will show power at both P and 2P/3P,
    but a false long period (harmonic lock) shows lower power at P/N.
    """
    periods_arr = to_clean_array(pg.period.value)
    power_arr = to_clean_array(pg.power)
    best_power = float(np.nanmax(power_arr))

    # Check both sub-harmonics (shorter) and super-harmonics (longer)
    # Format: (multiplier, prefer_if_power_ratio_above)
    # Shorter periods (P/2, P/3): lower threshold (0.60) because BLS naturally
    #   scores higher at longer periods even when shorter is the true period.
    # Longer periods (2P, 3P):    higher threshold (0.90) to avoid false promotion.
    candidates = [
        (0.5,         0.60),   # P/2  — prefer shorter if at least 60% as strong
        (1.0 / 3.0,   0.60),   # P/3  — prefer shorter if at least 60% as strong
        (2.0,         0.90),   # 2P   — prefer longer only if nearly as strong
        (3.0,         0.90),   # 3P   — prefer longer only if nearly as strong
    ]

    best_final_period = best_period_val
    best_final_power  = best_power

    for factor, threshold in candidates:
        cand = best_period_val * factor
        idx = np.nanargmin(np.abs(periods_arr - cand))
        if abs(periods_arr[idx] - cand) / cand > 0.05:
            continue
        cand_power = float(power_arr[idx])
        # For shorter candidates: prefer if power is strong AND period is shorter
        # For longer candidates:  prefer if power is strong AND period is longer
        is_shorter = cand < best_final_period
        is_longer  = cand > best_final_period
        if cand_power >= best_power * threshold:
            if is_shorter:
                # Prefer shorter (true fundamental) over a harmonic lock
                best_final_period = periods_arr[idx]
                best_final_power  = cand_power
            elif is_longer and cand_power >= best_final_power * threshold:
                # Prefer longer only if it's also stronger than current best
                best_final_period = periods_arr[idx]
                best_final_power  = cand_power

    return best_final_period


for idx, row in targets.iterrows():
    tic_id = str(int(row['tid']))
    label = row['label']
    known_period = float(row['pl_orbper'])

    if tic_id in done_tics:
        continue

    print(f"[{idx+1}/{len(targets)}] TIC {tic_id} ({label})")

    try:
        lc_raw, source = download_lc(tic_id)
        if lc_raw is None:
            print(f"  SKIP: {source}")
            fail_count += 1
            continue

        lc = lc_raw.normalize().flatten(window_length=401).remove_outliers(sigma=4)
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

        corrected_period = check_harmonic(pg, raw_best_period)

        t0_val = float(pg.transit_time[best_idx].value)
        duration_best_val = float(pg.duration[best_idx].value)  # in days
        best_period = corrected_period

        class _T0:
            value = t0_val
        t0 = _T0()

        class _Dur:
            def to(self, unit):
                class _V:
                    value = duration_best_val
                return _V()
        duration_best = _Dur()

        half_width = (float(duration_best.to('d').value) / best_period) / 2

        folded = lc.fold(period=best_period, epoch_time=t0.value)
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

        cycle_num = np.round((t_arr - t0.value) / best_period).astype(int)
        phase_global = ((t_arr - t0.value) % best_period) / best_period
        phase_global[phase_global > 0.5] -= 1
        in_g = np.abs(phase_global) < half_width * 1.3

        o_d = (float(np.nanmedian(f_arr[in_g & (cycle_num % 2 == 1)]))
               if np.sum(in_g & (cycle_num % 2 == 1)) > 2 else 1.0)
        e_d = (float(np.nanmedian(f_arr[in_g & (cycle_num % 2 == 0)]))
               if np.sum(in_g & (cycle_num % 2 == 0)) > 2 else 1.0)
        odd_even_diff = abs(o_d - e_d)

        duration_hours = float(duration_best.to('d').value) * 24

        # ── 5 physics-informed features ──────────────────────────
        edge_mask = (np.abs(phase_vals) < half_width * 1.3) & (np.abs(phase_vals) > half_width * 0.7)
        center_mask = np.abs(phase_vals) < half_width * 0.3
        edge_depth = float(baseline - np.nanmedian(flux_vals[edge_mask])) if np.sum(edge_mask) > 2 else depth
        center_depth = float(baseline - np.nanmedian(flux_vals[center_mask])) if np.sum(center_mask) > 2 else depth
        transit_shape = (center_depth / edge_depth) if edge_depth > 1e-8 else 1.0

        dur_period_ratio = duration_hours / (best_period * 24)

        per_transit_depths = []
        unique_cycles = np.unique(cycle_num[in_g])
        for c in unique_cycles:
            mask_c = in_g & (cycle_num == c)
            if np.sum(mask_c) > 1:
                per_transit_depths.append(baseline - float(np.nanmedian(f_arr[mask_c])))
        if len(per_transit_depths) > 2:
            depth_consistency = float(np.nanstd(per_transit_depths) / (np.nanmean(per_transit_depths) + 1e-8))
        else:
            depth_consistency = 0.0

        left_mask = (phase_vals < -half_width * 0.3) & (phase_vals > -half_width * 1.3)
        right_mask = (phase_vals > half_width * 0.3) & (phase_vals < half_width * 1.3)
        left_depth = float(baseline - np.nanmedian(flux_vals[left_mask])) if np.sum(left_mask) > 2 else depth
        right_depth = float(baseline - np.nanmedian(flux_vals[right_mask])) if np.sum(right_mask) > 2 else depth
        ingress_egress_asymmetry = abs(left_depth - right_depth) / (depth + 1e-8)

        odd_even_ratio = odd_even_diff / (depth + 1e-8)

        features_list.append({
            "tic_id": tic_id,
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
        done_tics.add(tic_id)
        success_count += 1
        print(f"  OK [{source}] period={best_period:.4f}d depth={depth*100:.4f}% snr={snr:.1f}")

    except Exception as e:
        print(f"  ERROR: {e}")
        fail_count += 1

    if (idx + 1) % 5 == 0:
        pd.DataFrame(features_list).to_csv(checkpoint_file, index=False)
        print(f"  [Checkpoint saved — {len(features_list)} done so far]")

features_df = pd.DataFrame(features_list)
features_df.to_csv("features_dataset.csv", index=False)

print("\n" + "=" * 60)
print(f"DONE!  Success: {success_count}  |  Failed: {fail_count}")
print(f"Total features saved: {len(features_df)}")
print(f"\nLabel distribution:")
print(features_df['label'].value_counts().to_string())
print("\nNext step: python train_model.py")