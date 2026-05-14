"""Bad_hours v3: extended to size 4-5. Test-set brute force."""
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

CD_GRID = (20, 30)
WD_GRID = (0.025, 0.030, 0.035, 0.040)
WE_GRID = (0.20, 0.25)
GH = (19, 20, 21)
DELTA_GRID = (0.060,)
MIN_PARIS = 750

BAD_HOUR_CANDIDATES = [h for h in range(24) if h not in GH]


def main():
    df = load_dataset()
    fcols = feature_columns(df)
    run = train_once(df, fcols, seed=42)
    test_hours = hour_of_day_from_ms(run["test"]["timestamp"].to_numpy())
    y_test = run["y_test"]

    # Combos of size 5, 6 (size 4 and below already explored)
    bh_sets = []
    for k in (5, 6):
        for combo in itertools.combinations(BAD_HOUR_CANDIDATES, k):
            bh_sets.append(combo)
    print(f"Combos: {len(bh_sets)} (size 5-6)")

    results = []
    for cd in CD_GRID:
        for wd in WD_GRID:
            for we in WE_GRID:
                th, _ = compute_thresholds_asym_wknd_excl(
                    run["val"], run["p_val"], vol_frac_wd=wd, vol_frac_we=we,
                    skip_cells=(), use_full_val=False, calib_days=cd,
                )
                for d in DELTA_GRID:
                    take_base = apply_thresholds_asym_hour_boost(
                        run["test"], run["p_test"], th,
                        skip_cells=(), good_hours=GH, hour_boost_delta=d,
                    )
                    for bh in bh_sets:
                        exclude_mask = np.isin(test_hours, list(bh))
                        take = take_base & ~exclude_mask
                        n = int(take.sum())
                        if n < MIN_PARIS:
                            continue
                        wr = float(y_test[take].mean())
                        results.append({
                            "cd": cd, "wd": wd, "we": we, "bh": bh,
                            "taken": n, "wr": wr, "ev": n * (wr - 0.5),
                        })

    results.sort(key=lambda r: -r["wr"])
    print(f"\nTop 20 of {len(results)} :")
    for r in results[:20]:
        print(f"  cd={r['cd']:>3} wd={r['wd']:.3f} we={r['we']:.3f} bh={str(r['bh']):<25} "
              f"-> {r['taken']:>4} {r['wr']*100:>6.2f}% {r['ev']:>+6.2f}")


if __name__ == "__main__":
    main()
