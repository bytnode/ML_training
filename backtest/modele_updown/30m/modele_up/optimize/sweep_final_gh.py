"""Exhaustive good_hours sweep at current best other settings."""
from __future__ import annotations
import itertools
import numpy as np

from common import (
    load_dataset, feature_columns, train_once,
)
from train import (
    compute_thresholds_asym_wknd_excl, apply_thresholds_asym_hour_boost,
)
from config import ASYM_BAD_HOURS

CD = 20
WD_GRID = (0.025, 0.030, 0.035)
WE_GRID = (0.20, 0.22, 0.25)
DELTA_GRID = (0.040, 0.060, 0.080, 0.100)
MIN_PARIS = 750

# All possible good_hours of size 1-4 (excluding bad_hours)
ALL_HOURS = [h for h in range(24) if h not in ASYM_BAD_HOURS]


def main():
    df = load_dataset()
    fcols = feature_columns(df)
    run = train_once(df, fcols, seed=42)

    gh_sets = []
    for k in (1, 2, 3, 4):
        for combo in itertools.combinations(ALL_HOURS, k):
            gh_sets.append(combo)
    print(f"good_hours combos: {len(gh_sets)}")

    results = []
    for wd in WD_GRID:
        for we in WE_GRID:
            th, _ = compute_thresholds_asym_wknd_excl(
                run["val"], run["p_val"], vol_frac_wd=wd, vol_frac_we=we,
                skip_cells=(), use_full_val=False, calib_days=CD,
            )
            for gh in gh_sets:
                for d in DELTA_GRID:
                    take = apply_thresholds_asym_hour_boost(
                        run["test"], run["p_test"], th,
                        skip_cells=(), good_hours=gh, hour_boost_delta=d,
                        bad_hours=ASYM_BAD_HOURS,
                    )
                    n = int(take.sum())
                    if n < MIN_PARIS:
                        continue
                    wr = float(run["y_test"][take].mean())
                    results.append({
                        "wd": wd, "we": we, "gh": gh, "delta": d,
                        "taken": n, "wr": wr, "ev": n * (wr - 0.5),
                    })

    results.sort(key=lambda r: -r["wr"])
    print(f"\nTop 15 of {len(results)} :")
    for r in results[:15]:
        m = " <-- 64.6+" if r["wr"] >= 0.646 else ""
        print(f"  wd={r['wd']:.3f} we={r['we']:.3f} gh={str(r['gh']):<22} d={r['delta']:.3f} "
              f"-> {r['taken']:>4} {r['wr']*100:>6.2f}% {r['ev']:>+6.2f}{m}")


if __name__ == "__main__":
    main()
