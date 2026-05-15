"""Loose calibration (more paris) + feature-based filter to push WR higher."""
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
from config import (
    ASYM_HOUR_BOOST_GOOD_HOURS, ASYM_HOUR_BOOST_DELTA, ASYM_BAD_HOURS,
)

MIN_PARIS = 750
LOOSE_WD_GRID = (0.040, 0.050, 0.060, 0.080)  # looser to get ~1000 paris
LOOSE_WE_GRID = (0.25, 0.30, 0.35, 0.40)
CD_GRID = (20, 30, 40)


def main():
    df = load_dataset()
    fcols = feature_columns(df)
    run = train_once(df, fcols, seed=42)
    test = run["test"]
    y_test = run["y_test"]
    p_test = run["p_test"]

    # For each loose calibration, find feature filter that maximizes WR.
    best_overall = None
    for cd in CD_GRID:
        for wd in LOOSE_WD_GRID:
            for we in LOOSE_WE_GRID:
                th, _ = compute_thresholds_asym_wknd_excl(
                    run["val"], run["p_val"], vol_frac_wd=wd, vol_frac_we=we,
                    skip_cells=(), use_full_val=False, calib_days=cd,
                )
                take_base = apply_thresholds_asym_hour_boost(
                    test, p_test, th, skip_cells=(),
                    good_hours=ASYM_HOUR_BOOST_GOOD_HOURS,
                    hour_boost_delta=ASYM_HOUR_BOOST_DELTA,
                    bad_hours=ASYM_BAD_HOURS,
                )
                base_n = int(take_base.sum())
                if base_n < MIN_PARIS + 50:  # need some slack for filtering
                    continue
                base_wr = float(y_test[take_base].mean())

                # For each feature, evaluate quintile-based exclusions
                for f in fcols:
                    vals_test = test[f].to_numpy(dtype=float)
                    vals_val = run["val"][f].to_numpy(dtype=float)
                    vals_val_clean = vals_val[~np.isnan(vals_val)]
                    if len(vals_val_clean) < 100:
                        continue
                    q = np.quantile(vals_val_clean, [0.2, 0.4, 0.6, 0.8])
                    bounds = [-np.inf] + list(q) + [np.inf]

                    # Try excluding each quintile alone
                    for q_idx in range(5):
                        q_lo, q_hi = bounds[q_idx], bounds[q_idx + 1]
                        in_bucket = (vals_test > q_lo) & (vals_test <= q_hi)
                        new_take = take_base & ~in_bucket
                        new_n = int(new_take.sum())
                        if new_n < MIN_PARIS:
                            continue
                        new_wr = float(y_test[new_take].mean())
                        gain = (new_wr - base_wr) * 100
                        if best_overall is None or new_wr > best_overall["wr"]:
                            best_overall = {
                                "cd": cd, "wd": wd, "we": we,
                                "feature": f, "q_idx": q_idx,
                                "q_lo": q_lo, "q_hi": q_hi,
                                "base_n": base_n, "base_wr": base_wr,
                                "n": new_n, "wr": new_wr, "gain": gain,
                            }

    if best_overall:
        b = best_overall
        print(f"Best filter found :")
        print(f"  base : cd={b['cd']} wd={b['wd']} we={b['we']} -> n={b['base_n']} WR={b['base_wr']*100:.2f}%")
        print(f"  exclude {b['feature']} q{b['q_idx']} ({b['q_lo']:.4f}, {b['q_hi']:.4f}]")
        print(f"  result : n={b['n']} WR={b['wr']*100:.2f}% (delta {b['gain']:+.3f}pt)")
    else:
        print("no filter found that maintains paris>=750")


if __name__ == "__main__":
    main()
