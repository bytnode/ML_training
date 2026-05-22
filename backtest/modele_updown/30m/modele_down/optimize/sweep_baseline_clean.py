"""
Sweep a few clean baseline configs to see where we are WITHOUT hour_boost
and bad_hours (only vol_frac calibration). The goal is to find the best
"vanilla" config we can build hour adjustments on top of.

Sweep:
  - use_full_val in {True, False}
  - calib_days in {20, 55, 90, 180} (only used if use_full_val=False)
  - vol_frac_wd in {0.015, 0.020, 0.030, 0.040, 0.060, 0.100}
  - vol_frac_we in {0.05, 0.10, 0.15, 0.20, 0.30}

For each config:
  taken / WR / EV  (no hour_boost, no bad_hours)
"""
from __future__ import annotations

import itertools
import numpy as np

from common import (
    load_dataset, feature_columns, train_once, evaluate_with_calibration,
)


USE_FULL_VAL_GRID = (True, False)
CALIB_DAYS_GRID   = (20, 55, 90, 180)  # only matters if use_full_val=False
WD_GRID = (0.015, 0.020, 0.025, 0.030, 0.040, 0.060, 0.100)
WE_GRID = (0.05, 0.08, 0.12, 0.15, 0.20, 0.30)
MIN_PARIS = 750  # we want ~750 paris/365d


def main():
    df = load_dataset()
    fc = feature_columns(df)
    print(f"Dataset: {len(df):,} rows, {len(fc)} features.")

    print("Training one model (default HPs from config)...")
    run = train_once(df, fc, seed=42)
    print(f"  best_iter={run['best_iter']} val_logloss={run['val_logloss']:.5f} "
          f"test_logloss={run['test_logloss']:.5f}")

    results = []
    grid = list(itertools.product(USE_FULL_VAL_GRID, CALIB_DAYS_GRID, WD_GRID, WE_GRID))
    print(f"\n{len(grid)} configs to evaluate (no hour_boost, no bad_hours)")

    for ufv, cd, wd, we in grid:
        # When use_full_val=True, calib_days doesn't matter; only run once
        if ufv and cd != CALIB_DAYS_GRID[0]:
            continue

        r = evaluate_with_calibration(
            run, vol_frac_wd=wd, vol_frac_we=we,
            good_hours=(), hour_boost_delta=0.0, skip_cells=(),
            bad_hours=(), use_full_val=ufv, calib_days=cd,
        )
        results.append({
            "use_full_val": ufv, "calib_days": cd, "wd": wd, "we": we,
            **r,
        })

    # Sort by WR desc among configs with taken >= MIN_PARIS
    qual = [r for r in results if r["taken"] >= MIN_PARIS]
    qual.sort(key=lambda r: (-r["wr"], -r["taken"]))

    near = [r for r in results if 500 <= r["taken"] < MIN_PARIS]
    near.sort(key=lambda r: (-r["wr"], -r["taken"]))

    print(f"\n  TOP 20 configs with taken >= {MIN_PARIS} (sorted by WR)")
    print("  " + "-" * 90)
    print(f"  {'ufv':>5} {'cd':>3} {'wd':>6} {'we':>6} {'taken':>6} {'wr':>7} "
          f"{'ev':>7} {'T_wd':>7} {'T_we':>7}")
    for r in qual[:20]:
        print(f"  {str(r['use_full_val']):>5} {r['calib_days']:>3} {r['wd']:>6.3f} {r['we']:>6.3f} "
              f"{r['taken']:>6} {r['wr']*100:>6.2f}% {r['ev']:>7.1f} "
              f"{(r['T_wd'] or 0):>7.4f} {(r['T_we'] or 0):>7.4f}")

    print(f"\n  TOP 20 configs with 500 <= taken < {MIN_PARIS} (sorted by WR)")
    print("  " + "-" * 90)
    print(f"  {'ufv':>5} {'cd':>3} {'wd':>6} {'we':>6} {'taken':>6} {'wr':>7} "
          f"{'ev':>7} {'T_wd':>7} {'T_we':>7}")
    for r in near[:20]:
        print(f"  {str(r['use_full_val']):>5} {r['calib_days']:>3} {r['wd']:>6.3f} {r['we']:>6.3f} "
              f"{r['taken']:>6} {r['wr']*100:>6.2f}% {r['ev']:>7.1f} "
              f"{(r['T_wd'] or 0):>7.4f} {(r['T_we'] or 0):>7.4f}")


if __name__ == "__main__":
    main()
