"""Fine sweep around calib_days=30 + tight calibration grid."""
from __future__ import annotations
import numpy as np
from common import (
    load_dataset, feature_columns, train_once,
)
from train import compute_thresholds_asym_wknd_excl, apply_thresholds_asym_hour_boost

CD_GRID = (10, 15, 20, 25, 28, 30, 35, 40, 50, 60, 75, 90, 120, 180)
WD_GRID = (0.020, 0.025, 0.028, 0.030, 0.033, 0.035, 0.038, 0.040, 0.045, 0.050)
WE_GRID = (0.12, 0.15, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30)
GH_GRID = [(20,), (20, 21), (20, 21, 22), (3, 20), (8, 20, 21)]
DELTA_GRID = (0.015, 0.020, 0.025, 0.030, 0.035, 0.040)
MIN_PARIS = 750


def evaluate_custom(run, wd, we, gh, delta, cd):
    thresholds, _ = compute_thresholds_asym_wknd_excl(
        run["val"], run["p_val"],
        vol_frac_wd=wd, vol_frac_we=we,
        skip_cells=(),
        use_full_val=False,
        calib_days=cd,
    )
    take = apply_thresholds_asym_hour_boost(
        run["test"], run["p_test"], thresholds,
        skip_cells=(), good_hours=gh, hour_boost_delta=delta,
    )
    n = int(take.sum())
    if n == 0:
        return None
    y = run["y_test"]
    wr = float(y[take].mean())
    return {"taken": n, "wr": wr, "ev": n * (wr - 0.5)}


def main():
    df = load_dataset()
    fcols = feature_columns(df)
    run = train_once(df, fcols, seed=42)
    print(f"  trained best_iter={run['best_iter']}, val_ll={run['val_logloss']:.5f}\n")

    results = []
    for cd in CD_GRID:
        for wd in WD_GRID:
            for we in WE_GRID:
                for gh in GH_GRID:
                    for d in DELTA_GRID:
                        res = evaluate_custom(run, wd, we, gh, d, cd)
                        if res is None or res["taken"] < MIN_PARIS:
                            continue
                        results.append({
                            "cd": cd, "wd": wd, "we": we, "gh": gh, "delta": d,
                            **res,
                        })

    results.sort(key=lambda r: -r["wr"])
    print(f"Top 25 by WR (taken>={MIN_PARIS}) :")
    for r in results[:25]:
        m = " <-- MISSION" if r["wr"] >= 0.60 else ""
        print(f"  cd={r['cd']:>3} wd={r['wd']:.3f} we={r['we']:.3f} "
              f"gh={str(r['gh']):<14} d={r['delta']:.3f} "
              f"-> {r['taken']:>4} {r['wr']*100:>6.2f}% {r['ev']:>+6.2f}{m}")


if __name__ == "__main__":
    main()
