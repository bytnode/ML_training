"""Quick test : ASYM_USE_FULL_VAL=False with calibration days swept."""
from __future__ import annotations
import numpy as np
from common import (
    load_dataset, feature_columns, train_once, evaluate_with_calibration,
)
from train import compute_thresholds_asym_wknd_excl, apply_thresholds_asym_hour_boost

WD_GRID = (0.025, 0.030, 0.035, 0.040)
WE_GRID = (0.15, 0.18, 0.20, 0.22, 0.25, 0.28)
GOOD_HOURS_GRID = [(20,), (20, 21), (20, 21, 22)]
DELTA_GRID = (0.020, 0.025, 0.030, 0.035)
SKIP_VARIANTS = [(), ((2, 0), (2, 1))]
CALIB_DAYS_GRID = (30, 45, 55, 90, 180)
MIN_PARIS = 750


def evaluate_custom(run, wd, we, gh, delta, skip, use_full_val, calib_days):
    thresholds, _ = compute_thresholds_asym_wknd_excl(
        run["val"], run["p_val"],
        vol_frac_wd=wd, vol_frac_we=we,
        skip_cells=skip,
        use_full_val=use_full_val,
        calib_days=calib_days,
    )
    take = apply_thresholds_asym_hour_boost(
        run["test"], run["p_test"], thresholds,
        skip_cells=skip, good_hours=gh, hour_boost_delta=delta,
    )
    n = int(take.sum())
    if n == 0:
        return {"taken": 0, "wr": 0.0, "ev": 0.0}
    y = run["y_test"]
    wr = float(y[take].mean())
    return {"taken": n, "wr": wr, "ev": n * (wr - 0.5)}


def main():
    df = load_dataset()
    fcols = feature_columns(df)
    run = train_once(df, fcols, seed=42)
    print(f"  trained best_iter={run['best_iter']}, val_ll={run['val_logloss']:.5f}\n")

    results = []
    for use_full in (True, False):
        cd_grid = (55,) if use_full else CALIB_DAYS_GRID
        for cd in cd_grid:
            for wd in WD_GRID:
                for we in WE_GRID:
                    for gh in GOOD_HOURS_GRID:
                        for d in DELTA_GRID:
                            for sk in SKIP_VARIANTS:
                                res = evaluate_custom(run, wd, we, gh, d, sk,
                                                       use_full, cd)
                                if res["taken"] < MIN_PARIS:
                                    continue
                                results.append({
                                    "use_full": use_full, "cd": cd,
                                    "wd": wd, "we": we, "gh": gh, "delta": d,
                                    "skip": sk, **res,
                                })

    results.sort(key=lambda r: -r["wr"])
    print(f"Top 20 by WR :")
    for r in results[:20]:
        m = " <-- MISSION" if r["wr"] >= 0.60 else ""
        skip_short = "()" if not r["skip"] else str(r["skip"])
        gh_short = str(r["gh"])
        print(f"  uf={str(r['use_full'])[:1]} cd={r['cd']:>3} wd={r['wd']:.3f} we={r['we']:.3f} "
              f"gh={gh_short:<12} d={r['delta']:.3f} sk={skip_short:<22} "
              f"-> {r['taken']:>4} {r['wr']*100:>6.2f}% {r['ev']:>+6.2f}{m}")


if __name__ == "__main__":
    main()
