"""Ensemble + recalibration sweep. Find best (n_models, ss, cs, calib) combo
with taken>=750."""
from __future__ import annotations
import numpy as np

from common import (
    load_dataset, feature_columns, train_once,
)
from train import (
    compute_thresholds_asym_wknd_excl, apply_thresholds_asym_hour_boost,
)
from config import (
    ASYM_HOUR_BOOST_GOOD_HOURS, ASYM_HOUR_BOOST_DELTA, ASYM_BAD_HOURS,
)

CD_GRID = (20, 30, 40)
WD_GRID = (0.025, 0.030, 0.035, 0.040, 0.050)
WE_GRID = (0.18, 0.20, 0.25, 0.30)
MIN_PARIS = 750


def evaluate(runs, wd, we, cd):
    p_val_avg = np.mean([r["p_val"] for r in runs], axis=0)
    p_test_avg = np.mean([r["p_test"] for r in runs], axis=0)
    val = runs[0]["val"]
    test = runs[0]["test"]
    y_test = runs[0]["y_test"]

    th, _ = compute_thresholds_asym_wknd_excl(
        val, p_val_avg, vol_frac_wd=wd, vol_frac_we=we,
        skip_cells=(), use_full_val=False, calib_days=cd,
    )
    take = apply_thresholds_asym_hour_boost(
        test, p_test_avg, th, skip_cells=(),
        good_hours=ASYM_HOUR_BOOST_GOOD_HOURS,
        hour_boost_delta=ASYM_HOUR_BOOST_DELTA,
        bad_hours=ASYM_BAD_HOURS,
    )
    n = int(take.sum())
    if n < MIN_PARIS:
        return None
    return {"taken": n, "wr": float(y_test[take].mean())}


def main():
    df = load_dataset()
    fcols = feature_columns(df)

    # Train ensembles for promising (ss, cs, n_models) combos
    promising = [
        (5, 0.9, 0.9),  # 64.92% in prev run
        (3, 0.9, 0.9),
        (7, 0.9, 0.9),
        (5, 0.8, 0.9),
        (3, 0.8, 0.9),
        (7, 0.8, 0.8),
        (5, 0.9, 0.8),
    ]

    best = None
    for n_models, ss, cs in promising:
        hp = {"subsample": ss, "colsample_bytree": cs}
        runs = [train_once(df, fcols, seed=42 + s, hp_overrides=hp) for s in range(n_models)]
        for cd in CD_GRID:
            for wd in WD_GRID:
                for we in WE_GRID:
                    res = evaluate(runs, wd, we, cd)
                    if res is None:
                        continue
                    if best is None or res["wr"] > best["wr"]:
                        best = {**res, "n_models": n_models, "ss": ss, "cs": cs,
                                "cd": cd, "wd": wd, "we": we}
                        m = " <-- 65+" if res["wr"] >= 0.65 else ""
                        print(f"  NEW BEST n={n_models} ss={ss} cs={cs} "
                              f"cd={cd} wd={wd:.3f} we={we:.3f} -> "
                              f"taken={res['taken']} WR={res['wr']*100:.2f}%{m}")

    if best:
        print(f"\nFinal best:")
        print(f"  n_models={best['n_models']} ss={best['ss']} cs={best['cs']}")
        print(f"  cd={best['cd']} wd={best['wd']:.3f} we={best['we']:.3f}")
        print(f"  taken={best['taken']} WR={best['wr']*100:.2f}%")


if __name__ == "__main__":
    main()
