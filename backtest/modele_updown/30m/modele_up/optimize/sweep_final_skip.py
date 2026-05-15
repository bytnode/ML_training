"""Test skip_cells at the new winning config (with bad_hours)."""
from __future__ import annotations
import itertools
import numpy as np

from common import (
    load_dataset, feature_columns, train_once,
)
from train import (
    compute_thresholds_asym_wknd_excl, apply_thresholds_asym_hour_boost,
    hour_of_day_from_ms,
)

CD = 20
WD_GRID = (0.025, 0.030, 0.035, 0.040)
WE_GRID = (0.18, 0.20, 0.22, 0.25)
GH = (19, 20, 21)
DELTA = 0.060
BAD_HOURS = (5, 12, 18, 23)
MIN_PARIS = 750

ALL_CELLS = [(v, w) for v in (0, 1, 2) for w in (0, 1)]


def main():
    df = load_dataset()
    fcols = feature_columns(df)
    run = train_once(df, fcols, seed=42)

    skip_variants = [()]
    for k in (1, 2, 3):
        for combo in itertools.combinations(ALL_CELLS, k):
            skip_variants.append(combo)
    print(f"Skip variants : {len(skip_variants)}")

    results = []
    for sk in skip_variants:
        for wd in WD_GRID:
            for we in WE_GRID:
                th, _ = compute_thresholds_asym_wknd_excl(
                    run["val"], run["p_val"], vol_frac_wd=wd, vol_frac_we=we,
                    skip_cells=sk, use_full_val=False, calib_days=CD,
                )
                take = apply_thresholds_asym_hour_boost(
                    run["test"], run["p_test"], th,
                    skip_cells=sk, good_hours=GH, hour_boost_delta=DELTA,
                    bad_hours=BAD_HOURS,
                )
                n = int(take.sum())
                if n < MIN_PARIS:
                    continue
                wr = float(run["y_test"][take].mean())
                results.append({
                    "sk": sk, "wd": wd, "we": we,
                    "taken": n, "wr": wr, "ev": n * (wr - 0.5),
                })

    results.sort(key=lambda r: -r["wr"])
    print(f"\nTop 20 of {len(results)} :")
    for r in results[:20]:
        m = " <-- 64.5+" if r["wr"] >= 0.645 else ""
        print(f"  sk={str(r['sk']):<26} wd={r['wd']:.3f} we={r['we']:.3f} "
              f"-> {r['taken']:>4} {r['wr']*100:>6.2f}% {r['ev']:>+6.2f}{m}")


if __name__ == "__main__":
    main()
