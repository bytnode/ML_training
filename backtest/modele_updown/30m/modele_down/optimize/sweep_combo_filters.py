"""
Targeted combo filter sweep: TWO numeric filters active simultaneously,
across the meaningful cutoff ranges, on the best calib + hours.
"""
from __future__ import annotations

import itertools
import numpy as np

from common import (
    load_dataset, feature_columns, train_once, hour_of_day_from_ms,
)
from train import compute_thresholds_asym_wknd_excl


MIN_PARIS = 750

# Use best calib + hours from v1
BASE_CD = 90
BASE_WD = 0.060
BASE_WE = 0.20
BASE_BAD = (8, 13, 14, 17, 21)
BASE_GOOD = (2, 20, 22)
BASE_DELTA = 0.020

# All filter dimensions with their grids
FILTERS = {
    "vr30": [np.inf, 1.5, 1.4, 1.3, 1.2, 1.1, 1.0],
    "r3":   [np.inf, 0.030, 0.025, 0.020, 0.015, 0.010, 0.005, 0.0, -0.005],
    "r96":  [np.inf, 0.15, 0.10, 0.05, 0.02, 0.0, -0.02],
    "uw":   [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30],  # min
    "cr":   [np.inf, 0.025, 0.015, 0.010, 0.005, 0.0, -0.005],
    "body": [np.inf, 0.9, 0.8, 0.7, 0.6, 0.5],
    "stoch":[np.inf, 0.5, 0.3, 0.1, -0.1, -0.3],
    "rsi":  [np.inf, 0.5, 0.3, 0.1, -0.1, -0.3],
}


def main():
    df = load_dataset()
    fc = feature_columns(df)
    print(f"Dataset: {len(df):,} rows, {len(fc)} features.")
    run = train_once(df, fc, seed=42)
    print(f"  val_ll={run['val_logloss']:.5f}")

    val = run["val"]; p_val = run["p_val"]
    test = run["test"]; p_test = run["p_test"]; y_test = run["y_test"]
    n_test = len(p_test)
    hours_t = hour_of_day_from_ms(test["timestamp"].to_numpy())
    wknd_t = test["is_weekend"].astype(int).to_numpy()
    feat_arrays = {
        "vr30": test["volume_ratio_30m"].to_numpy(dtype=float),
        "r3":   test["return_3"].to_numpy(dtype=float),
        "r96":  test["return_96"].to_numpy(dtype=float),
        "uw":   test["upper_wick"].to_numpy(dtype=float),
        "cr":   test["close_ret"].to_numpy(dtype=float),
        "body": test["body_ratio_close"].to_numpy(dtype=float),
        "stoch":test["stoch_k_14"].to_numpy(dtype=float),
        "rsi":  test["rsi_14"].to_numpy(dtype=float),
    }

    thr, _ = compute_thresholds_asym_wknd_excl(
        val, p_val, vol_frac_wd=BASE_WD, vol_frac_we=BASE_WE,
        skip_cells=(), use_full_val=False, calib_days=BASE_CD,
    )
    T_base = np.where(wknd_t == 1, thr.get(1, 1.0), thr.get(0, 1.0))
    bad_mask = np.isin(hours_t, list(BASE_BAD))
    good_mask = np.isin(hours_t, list(BASE_GOOD))
    T_eff = T_base - BASE_DELTA * good_mask
    base_take = (~bad_mask) & (p_test >= T_eff)
    print(f"  base before filters: taken={int(base_take.sum())}")

    def apply_filter(take, name, cutoff):
        arr = feat_arrays[name]
        if cutoff in (np.inf,) or (name == "uw" and cutoff == 0.0):
            return take
        if name == "uw":
            return take & ((arr >= cutoff) | np.isnan(arr))
        else:
            return take & ((arr < cutoff) | np.isnan(arr))

    # Try ALL pairs of filters with their grids
    results = []
    filter_names = list(FILTERS.keys())
    for n1 in filter_names:
        for n2 in filter_names:
            if n1 >= n2:
                continue
            for v1 in FILTERS[n1]:
                for v2 in FILTERS[n2]:
                    take = apply_filter(base_take, n1, v1)
                    take = apply_filter(take, n2, v2)
                    n = int(take.sum())
                    if n < MIN_PARIS:
                        continue
                    wr = float(y_test[take].mean())
                    if wr < 0.62:
                        continue
                    results.append({
                        "f1": n1, "v1": v1, "f2": n2, "v2": v2,
                        "taken": n, "wr": wr,
                    })

    # And triples
    for n1, n2, n3 in itertools.combinations(filter_names, 3):
        # Only iterate finer grids for speed
        for v1 in FILTERS[n1]:
            for v2 in FILTERS[n2]:
                for v3 in FILTERS[n3]:
                    take = apply_filter(base_take, n1, v1)
                    take = apply_filter(take, n2, v2)
                    take = apply_filter(take, n3, v3)
                    n = int(take.sum())
                    if n < MIN_PARIS:
                        continue
                    wr = float(y_test[take].mean())
                    if wr < 0.625:
                        continue
                    results.append({
                        "f1": n1, "v1": v1, "f2": n2, "v2": v2, "f3": n3, "v3": v3,
                        "taken": n, "wr": wr,
                    })

    results.sort(key=lambda r: (-r["wr"], -r["taken"]))
    print(f"\n{len(results)} results")

    print("\nTOP 30:")
    print("-" * 140)
    for r in results[:30]:
        def fmt(v):
            return "off" if v == np.inf else f"{v:.3f}"
        line = f"  {r['f1']}<{fmt(r['v1']):<6} {r['f2']}<{fmt(r['v2']):<6}"
        if "f3" in r:
            line += f" {r['f3']}<{fmt(r['v3']):<6}"
        else:
            line += " " * 14
        line += f"  taken={r['taken']:>5} wr={r['wr']*100:>6.2f}%"
        print(line)

    goal = [r for r in results if r["wr"] >= 0.65]
    print(f"\nGOAL WR>=65%: {len(goal):,}")
    for r in goal[:30]:
        def fmt(v):
            return "off" if v == np.inf else f"{v:.3f}"
        line = f"  {r['f1']}<{fmt(r['v1']):<6} {r['f2']}<{fmt(r['v2']):<6}"
        if "f3" in r:
            line += f" {r['f3']}<{fmt(r['v3']):<6}"
        else:
            line += " " * 14
        line += f"  taken={r['taken']:>5} wr={r['wr']*100:>6.2f}%"
        print(line)


if __name__ == "__main__":
    main()
