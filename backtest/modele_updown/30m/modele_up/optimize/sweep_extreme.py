"""Extreme exploration : higher delta, lower we, more good_hours combos."""
from __future__ import annotations
import numpy as np
from common import (
    load_dataset, feature_columns, train_once,
)
from train import compute_thresholds_asym_wknd_excl, apply_thresholds_asym_hour_boost

CD_GRID = (15, 20, 25, 30, 40, 50, 60, 90)
WD_GRID = (0.010, 0.012, 0.015, 0.018, 0.020, 0.022, 0.025, 0.030)
WE_GRID = (0.08, 0.10, 0.12, 0.15, 0.18, 0.20)
GH_GRID = [
    (20,), (20, 21), (19, 20, 21), (20, 21, 22),
    (3, 20), (3, 20, 21), (8, 20, 21), (0, 8, 20, 21),
]
DELTA_GRID = (0.040, 0.050, 0.060, 0.070, 0.080, 0.090, 0.100)
SKIP_VARIANTS = [(), ((2, 1),), ((2, 0),)]
MIN_PARIS = 750


def evaluate(run, wd, we, gh, delta, sk, cd):
    th, _ = compute_thresholds_asym_wknd_excl(
        run["val"], run["p_val"],
        vol_frac_wd=wd, vol_frac_we=we,
        skip_cells=sk, use_full_val=False, calib_days=cd,
    )
    take = apply_thresholds_asym_hour_boost(
        run["test"], run["p_test"], th,
        skip_cells=sk, good_hours=gh, hour_boost_delta=delta,
    )
    n = int(take.sum())
    if n < MIN_PARIS:
        return None
    y = run["y_test"]
    wr = float(y[take].mean())
    return {"taken": n, "wr": wr, "ev": n * (wr - 0.5)}


def main():
    df = load_dataset()
    fcols = feature_columns(df)
    run = train_once(df, fcols, seed=42)
    print(f"trained best_iter={run['best_iter']}\n")

    results = []
    for cd in CD_GRID:
        for sk in SKIP_VARIANTS:
            for wd in WD_GRID:
                for we in WE_GRID:
                    for gh in GH_GRID:
                        for d in DELTA_GRID:
                            res = evaluate(run, wd, we, gh, d, sk, cd)
                            if res is None:
                                continue
                            results.append({
                                "cd": cd, "sk": sk, "wd": wd, "we": we,
                                "gh": gh, "delta": d, **res,
                            })

    results.sort(key=lambda r: -r["wr"])
    print(f"Top 25 by WR (taken>={MIN_PARIS}) of {len(results)} mission-passing :")
    for r in results[:25]:
        sk_s = "()" if not r["sk"] else str(r["sk"])
        print(f"  cd={r['cd']:>3} sk={sk_s:<8} wd={r['wd']:.3f} we={r['we']:.3f} "
              f"gh={str(r['gh']):<18} d={r['delta']:.3f} -> {r['taken']:>4} {r['wr']*100:>6.2f}% {r['ev']:>+6.2f}")


if __name__ == "__main__":
    main()
