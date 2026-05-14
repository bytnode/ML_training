"""
Sweep scale_pos_weight + skip_cells variations.

For each (spw, skip_cells) combo :
  - retrain (spw affects model)
  - sweep wd/we/hour_boost
  - max WR among taken>=750

Usage : cd modele_up && python optimize/sweep_spw_skip.py
"""
from __future__ import annotations
import itertools
import time
import numpy as np

from common import (
    load_dataset, feature_columns, train_once, evaluate_with_calibration,
)


SPW_GRID = (0.85, 0.90, 0.95, 1.00, 1.03, 1.05, 1.10, "auto")

# Cells: (vol_state ∈ {0,1,2}, is_weekend ∈ {0,1}) = 6 cells.
# We have (vol_state, wknd):
#   (0,0)=low-vol weekday
#   (1,0)=mid-vol weekday
#   (2,0)=high-vol weekday
#   (0,1)=low-vol weekend
#   (1,1)=mid-vol weekend
#   (2,1)=high-vol weekend
SKIP_CELL_VARIANTS = [
    ((2, 0), (2, 1)),                  # current: skip high-vol
    (),                                # no skip
    ((2, 0),),                         # skip high-vol weekday only
    ((2, 1),),                         # skip high-vol weekend only
    ((2, 0), (2, 1), (0, 0)),          # also skip low-vol weekday
    ((2, 0), (2, 1), (0, 1)),          # also skip low-vol weekend
    ((2, 0), (2, 1), (1, 1)),          # also skip mid-vol weekend
    ((0, 0), (2, 0), (2, 1)),          # skip low-vol-weekday + high-vol
    ((0, 1), (2, 0), (2, 1)),          # skip low-vol-weekend + high-vol
]

WD_GRID = (0.020, 0.025, 0.030, 0.035, 0.040, 0.050)
WE_GRID = (0.18, 0.20, 0.22, 0.25, 0.28, 0.30)
GOOD_HOURS_GRID = [(20,), (20, 21), (20, 21, 22)]
DELTA_GRID = (0.020, 0.025, 0.030, 0.035, 0.040)

MIN_PARIS = 750


def best_for_run(run, skip_cells_used):
    best = None
    for wd in WD_GRID:
        for we in WE_GRID:
            for gh in GOOD_HOURS_GRID:
                for d in DELTA_GRID:
                    res = evaluate_with_calibration(
                        run, wd, we, good_hours=gh, hour_boost_delta=d,
                        skip_cells=skip_cells_used,
                    )
                    if res["taken"] < MIN_PARIS:
                        continue
                    if best is None or res["wr"] > best["wr"]:
                        best = {**res, "wd": wd, "we": we, "good_hours": gh, "delta": d}
    return best


def main():
    df = load_dataset()
    fcols = feature_columns(df)
    print(f"=== SPW x skip_cells sweep ===")
    print(f"  features={len(fcols)}, SPW combos={len(SPW_GRID)}, skip variants={len(SKIP_CELL_VARIANTS)}\n")

    results = []
    t0 = time.time()
    n_total = len(SPW_GRID) * len(SKIP_CELL_VARIANTS)
    n_done = 0
    for spw in SPW_GRID:
        for skip in SKIP_CELL_VARIANTS:
            n_done += 1
            hp = {"scale_pos_weight": spw}
            try:
                run = train_once(df, fcols, seed=42, hp_overrides=hp)
            except Exception as e:
                print(f"  [{n_done}/{n_total}] FAIL spw={spw} : {e}")
                continue
            best = best_for_run(run, skip)
            if best is None:
                print(f"  [{n_done}/{n_total}] spw={spw} skip={skip} : no 750+ config")
                continue
            results.append({"spw": spw, "skip": skip, **best})
            marker = " <-- MISSION" if best["wr"] >= 0.60 else ""
            print(f"  [{n_done}/{n_total}] spw={spw} skip={skip}  "
                  f"-> taken={best['taken']} WR={best['wr']*100:.2f}% EV={best['ev']:+.2f}"
                  f" (wd={best['wd']:.3f} we={best['we']:.3f} gh={best['good_hours']} d={best['delta']:.3f}){marker}")

    results.sort(key=lambda r: -r["wr"])
    print(f"\n=== Top 15 by WR ===")
    for r in results[:15]:
        marker = " <-- MISSION" if r["wr"] >= 0.60 else ""
        print(f"  spw={str(r['spw']):<6} skip={str(r['skip']):<30}  "
              f"taken={r['taken']:>4} WR={r['wr']*100:>6.2f}% EV={r['ev']:>+6.2f} "
              f"wd={r['wd']:.3f} we={r['we']:.3f} gh={r['good_hours']} d={r['delta']:.3f}{marker}")

    print(f"\nElapsed : {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
