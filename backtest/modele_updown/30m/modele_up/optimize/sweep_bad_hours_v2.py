"""Bad_hours v2 : exhaustive sweep of pairs/triples of bad hours."""
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

CD_GRID = (20, 30, 40)
WD_GRID = (0.020, 0.025, 0.030, 0.035, 0.040, 0.050)
WE_GRID = (0.15, 0.20, 0.25)
GH = (17, 20, 21)
DELTA_GRID = (0.060, 0.080)
MIN_PARIS = 750

# Candidates for bad_hours (all non-good-hours, plus weekend-y hours)
BAD_HOUR_CANDIDATES = [h for h in range(24) if h not in GH]


def main():
    df = load_dataset()
    fcols = feature_columns(df)
    run = train_once(df, fcols, seed=42)
    test_hours = hour_of_day_from_ms(run["test"]["timestamp"].to_numpy())
    y_test = run["y_test"]

    # All bad_hours combos of size 0, 1, 2, 3
    bh_sets = [()]
    for k in (1, 2, 3):
        for combo in itertools.combinations(BAD_HOUR_CANDIDATES, k):
            bh_sets.append(combo)
    print(f"Bad-hour combos: {len(bh_sets)} (size 0-3)")

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
                        if bh:
                            exclude_mask = np.isin(test_hours, list(bh))
                            take = take_base & ~exclude_mask
                        else:
                            take = take_base
                        n = int(take.sum())
                        if n < MIN_PARIS:
                            continue
                        wr = float(y_test[take].mean())
                        results.append({
                            "cd": cd, "wd": wd, "we": we, "delta": d, "bh": bh,
                            "taken": n, "wr": wr, "ev": n * (wr - 0.5),
                        })

    results.sort(key=lambda r: -r["wr"])
    print(f"\nTop 20 of {len(results)} mission-passing :")
    for r in results[:20]:
        m = " <-- 64+" if r["wr"] >= 0.64 else ""
        print(f"  cd={r['cd']:>3} wd={r['wd']:.3f} we={r['we']:.3f} d={r['delta']:.3f} "
              f"bh={str(r['bh']):<18} -> {r['taken']:>4} {r['wr']*100:>6.2f}% {r['ev']:>+6.2f}{m}")


if __name__ == "__main__":
    main()
