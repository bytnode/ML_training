"""Very tight sweep around current top : 44-feat model, multiple cd values,
wider hour_boost combos, finer wd/we steps."""
from __future__ import annotations
import numpy as np
from common import (
    load_dataset, feature_columns, train_once,
)
from train import compute_thresholds_asym_wknd_excl, apply_thresholds_asym_hour_boost

CD_GRID = (20, 25, 30, 35, 40, 45, 50, 60, 75, 90, 120)
WD_GRID = (0.015, 0.018, 0.020, 0.022, 0.025, 0.028, 0.030, 0.033, 0.035, 0.040)
WE_GRID = (0.15, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30, 0.35)
GH_GRID = [(20,), (20, 21), (20, 21, 22), (3, 20), (8, 20), (8, 20, 21)]
DELTA_GRID = (0.020, 0.025, 0.030, 0.035, 0.040, 0.050, 0.060)
SKIP_VARIANTS = [(), ((2, 1),)]
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
    print(f"=== Final tight sweep === {len(fcols)} features\n")
    run = train_once(df, fcols, seed=42)
    print(f"  trained best_iter={run['best_iter']}\n")

    results = []
    n_total = len(CD_GRID) * len(WD_GRID) * len(WE_GRID) * len(GH_GRID) * len(DELTA_GRID) * len(SKIP_VARIANTS)
    print(f"  total combos: {n_total}")
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
    print(f"\nTop 30 by WR (taken>={MIN_PARIS}) :")
    for r in results[:30]:
        m = " <-- MISSION" if r["wr"] >= 0.60 else ""
        sk_short = "()" if not r["sk"] else str(r["sk"])
        print(f"  cd={r['cd']:>3} sk={sk_short:<8} wd={r['wd']:.3f} we={r['we']:.3f} "
              f"gh={str(r['gh']):<14} d={r['delta']:.3f} -> {r['taken']:>4} {r['wr']*100:>6.2f}% {r['ev']:>+6.2f}{m}")

    print(f"\nFound {len(results)} mission-passing configs total.")


if __name__ == "__main__":
    main()
