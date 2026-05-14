"""
Sweep ASYM_VOL_FRAC_WEEKDAY x ASYM_VOL_FRAC_WEEKEND on the current model.

Train once, evaluate at many vol_frac combos -> table sorted by EV / WR / taken.
Helps find the (taken, WR) sweet spot for the mission target (>=750, >=60%).

Usage : cd modele_up && python optimize/sweep_calibration.py
"""
from __future__ import annotations
import time
import numpy as np

from common import (
    load_dataset, feature_columns, train_once, evaluate_with_calibration,
)


def sweep(n_seeds: int = 3,
          wd_grid=(0.015, 0.020, 0.025, 0.030, 0.035, 0.040, 0.045, 0.050,
                   0.060, 0.080, 0.100, 0.150),
          we_grid=(0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70)):
    print(f"=== Calibration sweep (n_seeds={n_seeds}) ===")
    df = load_dataset()
    fcols = feature_columns(df)
    print(f"  dataset : {len(df):,} rows x {len(fcols)} features")

    # Train n_seeds models, keep them for all evals
    t0 = time.time()
    runs = []
    for s in range(n_seeds):
        r = train_once(df, fcols, seed=42 + s)
        runs.append(r)
    print(f"  trained {n_seeds} model(s) in {time.time()-t0:.1f}s")
    print(f"  best_iters : {[r['best_iter'] for r in runs]}")
    print(f"  val_loglosses : {[round(r['val_logloss'], 5) for r in runs]}")

    rows = []
    for wd in wd_grid:
        for we in we_grid:
            takens, wrs = [], []
            for r in runs:
                res = evaluate_with_calibration(r, wd, we)
                takens.append(res["taken"])
                wrs.append(res["wr"])
            mean_taken = float(np.mean(takens))
            mean_wr    = float(np.mean(wrs))
            ev         = mean_taken * (mean_wr - 0.5)
            rows.append({
                "wd_frac": wd, "we_frac": we,
                "taken_avg": mean_taken,
                "wr_avg": mean_wr,
                "ev_avg": ev,
                "taken_std": float(np.std(takens, ddof=0)),
                "wr_std": float(np.std(wrs, ddof=0)),
            })

    # Print sorted by EV
    rows.sort(key=lambda r: -r["ev_avg"])
    print(f"\n  Top 20 by EV :")
    print(f"  {'wd_frac':>9}  {'we_frac':>9}  {'taken':>7}  {'WR':>8}  {'EV':>8}  {'taken_std':>10}  {'wr_std':>8}")
    for r in rows[:20]:
        print(f"  {r['wd_frac']:>9.3f}  {r['we_frac']:>9.3f}  {r['taken_avg']:>7.1f}  "
              f"{r['wr_avg']*100:>7.2f}%  {r['ev_avg']:>+8.2f}  "
              f"{r['taken_std']:>10.1f}  {r['wr_std']*100:>7.2f}%")

    # Filter on mission constraints
    print(f"\n  Configs hitting MISSION TARGET (taken>=750 AND WR>=0.60) :")
    hits = [r for r in rows if r["taken_avg"] >= 750 and r["wr_avg"] >= 0.60]
    if not hits:
        print(f"    NONE found in this grid.")
        # Show closest by relaxed constraints
        print(f"\n  Best by WR among >=750 paris :")
        candidates = sorted([r for r in rows if r["taken_avg"] >= 750],
                            key=lambda x: -x["wr_avg"])[:5]
        for r in candidates:
            print(f"    wd={r['wd_frac']:.3f} we={r['we_frac']:.3f}  "
                  f"taken={r['taken_avg']:.0f}  WR={r['wr_avg']*100:.2f}%  EV={r['ev_avg']:+.2f}")
        print(f"\n  Best by taken among >=60% WR :")
        candidates = sorted([r for r in rows if r["wr_avg"] >= 0.60],
                            key=lambda x: -x["taken_avg"])[:5]
        if not candidates:
            print(f"    no row reaches 60% WR in this grid.")
        else:
            for r in candidates:
                print(f"    wd={r['wd_frac']:.3f} we={r['we_frac']:.3f}  "
                      f"taken={r['taken_avg']:.0f}  WR={r['wr_avg']*100:.2f}%  EV={r['ev_avg']:+.2f}")
    else:
        for r in sorted(hits, key=lambda x: -x["ev_avg"]):
            print(f"    wd={r['wd_frac']:.3f} we={r['we_frac']:.3f}  "
                  f"taken={r['taken_avg']:.0f}  WR={r['wr_avg']*100:.2f}%  EV={r['ev_avg']:+.2f}")

    return rows


if __name__ == "__main__":
    sweep()
