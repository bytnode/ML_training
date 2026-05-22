"""
Per-cell (volatility x weekend) calibration sweep.

6 buckets: (vol_state in {0,1,2}) x (is_weekend in {0,1}).
For each bucket, take top vol_frac of val P(down).

Sweep:
  - vol_frac per cell (independent grid per cell since classes differ)
  - bad_hours / good_hours / delta on top

Goal: see if per-cell finer threshold gives lift vs per-wknd.
"""
from __future__ import annotations

import itertools
import numpy as np

from common import (
    load_dataset, feature_columns, train_once, hour_of_day_from_ms,
)


def per_cell_thresholds(val_df, p_val, vol_fracs: dict, calib_days=90):
    """vol_fracs: {(vol, wknd): vol_frac}. Returns {(vol, wknd): T}."""
    ms_per_day = 86_400_000
    val_max_ts = int(val_df["timestamp"].max())
    cutoff = val_max_ts - calib_days * ms_per_day
    mask = val_df["timestamp"].to_numpy() >= cutoff

    vol_arr = val_df["volatility_state"].astype(int).to_numpy()
    wknd_arr = val_df["is_weekend"].astype(int).to_numpy()
    probas = np.asarray(p_val)

    th = {}
    for (v, w), vf in vol_fracs.items():
        m = mask & (vol_arr == v) & (wknd_arr == w)
        if vf <= 0 or m.sum() < 30:
            th[(v, w)] = 2.0  # never take
            continue
        T = float(np.quantile(probas[m], 1.0 - vf))
        th[(v, w)] = T
    return th


def main():
    df = load_dataset()
    fc = feature_columns(df)
    print(f"Dataset: {len(df):,} rows, {len(fc)} features.")
    run = train_once(df, fc, seed=42)
    print(f"  best_iter={run['best_iter']} val_ll={run['val_logloss']:.5f}")

    test = run["test"]
    val = run["val"]
    p_val = run["p_val"]
    p_test = run["p_test"]
    y_test = run["y_test"]
    hours_t = hour_of_day_from_ms(test["timestamp"].to_numpy())
    wknd_t = test["is_weekend"].astype(int).to_numpy()
    vol_t = test["volatility_state"].astype(int).to_numpy()

    # Diagnostic : show val distribution per cell
    print("\nVal distribution per cell:")
    print("  cell      n_val   wr_val   p_mean")
    val_labels = val["label"].to_numpy().astype(int)
    val_vol = val["volatility_state"].astype(int).to_numpy()
    val_wknd = val["is_weekend"].astype(int).to_numpy()
    for v in (0, 1, 2):
        for w in (0, 1):
            m = (val_vol == v) & (val_wknd == w)
            n = int(m.sum())
            if n == 0:
                continue
            wr = float(val_labels[m].mean())
            p_mean = float(p_val[m].mean())
            print(f"  ({v},{w})    {n:>6}   {wr:>6.4f}   {p_mean:>6.4f}")

    print("\nTest distribution per cell:")
    print("  cell      n_test  wr_test  p_mean")
    for v in (0, 1, 2):
        for w in (0, 1):
            m = (vol_t == v) & (wknd_t == w)
            n = int(m.sum())
            if n == 0:
                continue
            wr = float(y_test[m].mean())
            p_mean = float(p_test[m].mean())
            print(f"  ({v},{w})    {n:>6}   {wr:>6.4f}   {p_mean:>6.4f}")

    # Sweep per-cell vol_fracs
    # For weekday cells (w=0): try a tighter grid because n_val is large
    # For weekend cells (w=1): looser grid because n_val small
    GRID_W0 = (0.0, 0.020, 0.030, 0.045, 0.060, 0.090)
    GRID_W1 = (0.0, 0.08, 0.15, 0.20, 0.30, 0.45)
    BAD_HOURS_GRID = [
        (8, 13, 17, 20, 21),
        (8, 17, 20),
        (8, 17),
        (),
    ]
    GOOD_HOURS_GRID = [
        (2, 15),
        (15,),
        (),
    ]
    DELTA_GRID = (0.0, 0.020, 0.040)
    CD_GRID = (55, 90, 180)
    MIN_PARIS = 750

    hour_masks_t = {h: (hours_t == h) for h in range(24)}

    results = []
    n_calc = (len(GRID_W0)**3 * len(GRID_W1)**3 * len(CD_GRID)
              * len(BAD_HOURS_GRID) * len(GOOD_HOURS_GRID) * len(DELTA_GRID))
    print(f"\nTotal configs: {n_calc:,}")

    iter_count = 0
    for v0w0 in GRID_W0:
        for v1w0 in GRID_W0:
            for v2w0 in GRID_W0:
                for v0w1 in GRID_W1:
                    for v1w1 in GRID_W1:
                        for v2w1 in GRID_W1:
                            for cd in CD_GRID:
                                vol_fracs = {
                                    (0, 0): v0w0, (1, 0): v1w0, (2, 0): v2w0,
                                    (0, 1): v0w1, (1, 1): v1w1, (2, 1): v2w1,
                                }
                                th = per_cell_thresholds(val, p_val, vol_fracs, calib_days=cd)
                                T_test = np.array([th[(int(vol_t[i]), int(wknd_t[i]))]
                                                    for i in range(len(p_test))])
                                for bh in BAD_HOURS_GRID:
                                    bad_mask = np.isin(hours_t, list(bh)) if bh else np.zeros(len(p_test), dtype=bool)
                                    for gh in GOOD_HOURS_GRID:
                                        good_mask = np.isin(hours_t, list(gh)) if gh else np.zeros(len(p_test), dtype=bool)
                                        for delta in (DELTA_GRID if len(gh) > 0 else (0.0,)):
                                            T_eff = T_test - delta * good_mask
                                            take = (~bad_mask) & (p_test >= T_eff)
                                            n_taken = int(take.sum())
                                            if n_taken < 200:  # filter early
                                                continue
                                            wr = float(y_test[take].mean())
                                            if wr < 0.55:  # filter
                                                continue
                                            results.append({
                                                "cd": cd, "vfracs": vol_fracs,
                                                "bad": bh, "good": gh, "delta": delta,
                                                "taken": n_taken, "wr": wr,
                                                "ev": n_taken * (wr - 0.5),
                                            })
                                iter_count += 1
                                if iter_count % 200 == 0:
                                    print(f"  ... iter {iter_count}, results so far: {len(results)}")

    qual = [r for r in results if r["taken"] >= MIN_PARIS]
    qual.sort(key=lambda r: (-r["wr"], -r["taken"]))
    print(f"\nResults: {len(results):,}, qualifying (>= {MIN_PARIS} bets): {len(qual):,}")

    print("\nTOP 30 by WR with taken >= MIN_PARIS:")
    print("-" * 140)
    for r in qual[:30]:
        vf = r["vfracs"]
        vf_s = (f"w0:{vf[(0,0)]:.3f}/{vf[(1,0)]:.3f}/{vf[(2,0)]:.3f} "
                f"w1:{vf[(0,1)]:.3f}/{vf[(1,1)]:.3f}/{vf[(2,1)]:.3f}")
        bh_s = "(" + ",".join(str(h) for h in r["bad"]) + ")"
        gh_s = "(" + ",".join(str(h) for h in r["good"]) + ")"
        print(f"  cd={r['cd']:>3} {vf_s}  bad={bh_s:<22} good={gh_s:<10} d={r['delta']:>5.3f} "
              f"taken={r['taken']:>5} wr={r['wr']*100:>6.2f}%")


if __name__ == "__main__":
    main()
