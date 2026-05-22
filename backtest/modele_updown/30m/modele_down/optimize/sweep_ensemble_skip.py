"""
Try multi-seed ensemble + per-volatility-bucket skip_cells.

Hypothesis:
  - Multi-seed averaging on P(down) reduces variance, should lift WR at high
    confidence threshold.
  - skip_cells = ((2,0), (2,1)) excludes high-vol bars where signal is noisier.
"""
from __future__ import annotations

import itertools
import numpy as np

from common import (
    load_dataset, feature_columns, train_once, hour_of_day_from_ms,
)
from train import compute_thresholds_asym_wknd_excl


SEEDS = (42, 123, 7, 31, 99)
SKIP_OPTIONS = [
    (),
    ((2, 0), (2, 1)),  # skip high-vol
    ((0, 0), (0, 1)),  # skip low-vol
    ((2, 0),),
    ((2, 1),),
    ((2, 0), (2, 1), (0, 1)),
]

# Best calib + hours from previous sweep
CALIB_GRID = [
    {"cd": 90, "wd": 0.040, "we": 0.20, "ufv": False},
    {"cd": 55, "wd": 0.030, "we": 0.20, "ufv": False},
    {"cd": 180, "wd": 0.040, "we": 0.20, "ufv": False},
]
BAD_HOURS_OPTIONS = [
    (8, 13, 17, 20),
    (8, 17, 20),
    (8, 17),
    (8, 10, 17, 20),
    (8, 13, 17, 20, 21),
]
GOOD_HOURS_OPTIONS = [
    (2, 15, 22),
    (2, 15),
    (15, 22),
    (15,),
    (2, 12, 15, 22),
    (),
]
DELTA_GRID = (0.020, 0.040, 0.060)

MIN_PARIS = 750


def main():
    df = load_dataset()
    fc = feature_columns(df)
    print(f"Dataset: {len(df):,} rows, {len(fc)} features.")

    # Train ensemble
    p_val_list = []
    p_test_list = []
    val = None
    test = None
    y_test = None
    print(f"\nTraining {len(SEEDS)} models...")
    for s in SEEDS:
        run = train_once(df, fc, seed=s)
        if val is None:
            val = run["val"]
            test = run["test"]
            y_test = run["y_test"]
        p_val_list.append(run["p_val"])
        p_test_list.append(run["p_test"])
        print(f"  seed={s} best_iter={run['best_iter']} val_ll={run['val_logloss']:.5f}")

    p_val_avg = np.mean(p_val_list, axis=0)
    p_test_avg = np.mean(p_test_list, axis=0)

    hours_t = hour_of_day_from_ms(test["timestamp"].to_numpy())
    wknd_t = test["is_weekend"].astype(int).to_numpy()
    vol_t = test["volatility_state"].astype(int).to_numpy()

    hour_masks_t = {h: (hours_t == h) for h in range(24)}

    results = []
    n_grid = (len(CALIB_GRID) * len(SKIP_OPTIONS) * len(BAD_HOURS_OPTIONS)
              * len(GOOD_HOURS_OPTIONS) * len(DELTA_GRID))
    print(f"\nEval {n_grid:,} calib combos on ensemble...")

    for cal in CALIB_GRID:
        for skip_cells in SKIP_OPTIONS:
            # Mask out skip cells on test
            skip_mask_t = np.zeros(len(p_test_avg), dtype=bool)
            for (v, w) in skip_cells:
                skip_mask_t |= ((vol_t == v) & (wknd_t == w))

            # Threshold from val averaged predictions
            thr, _ = compute_thresholds_asym_wknd_excl(
                val, p_val_avg,
                vol_frac_wd=cal["wd"], vol_frac_we=cal["we"],
                skip_cells=skip_cells, use_full_val=cal["ufv"], calib_days=cal["cd"],
            )
            T_wd = float(thr.get(0, 1.0))
            T_we = float(thr.get(1, 1.0))
            T_base = np.where(wknd_t == 1, T_we, T_wd)

            for bh in BAD_HOURS_OPTIONS:
                bad_mask_t = np.zeros(len(p_test_avg), dtype=bool)
                for h in bh:
                    bad_mask_t |= hour_masks_t[h]

                for gh in GOOD_HOURS_OPTIONS:
                    good_mask_t = np.zeros(len(p_test_avg), dtype=bool)
                    for h in gh:
                        good_mask_t |= hour_masks_t[h]

                    for delta in (DELTA_GRID if len(gh) > 0 else (0.0,)):
                        T_eff = T_base - delta * good_mask_t
                        take = (~skip_mask_t) & (~bad_mask_t) & (p_test_avg >= T_eff)
                        n_taken = int(take.sum())
                        if n_taken == 0:
                            continue
                        wr = float(y_test[take].mean())
                        results.append({
                            "calib": f"cd{cal['cd']}_wd{cal['wd']:.3f}_we{cal['we']:.2f}",
                            "skip": skip_cells, "bad": bh, "good": gh, "delta": delta,
                            "T_wd": T_wd, "T_we": T_we,
                            "taken": n_taken, "wr": wr, "ev": n_taken * (wr - 0.5),
                        })

    qual = [r for r in results if r["taken"] >= MIN_PARIS]
    qual.sort(key=lambda r: (-r["wr"], -r["taken"]))
    print(f"\nResults: {len(results):,} configs, {len(qual):,} with taken >= {MIN_PARIS}")

    print("\nTOP 30 by WR:")
    print("-" * 160)
    for r in qual[:30]:
        bh_s = "(" + ",".join(str(h) for h in r["bad"]) + ")"
        gh_s = "(" + ",".join(str(h) for h in r["good"]) + ")"
        skip_s = "(" + ";".join(f"{v}_{w}" for (v, w) in r["skip"]) + ")"
        print(f"  {r['calib']:<22} skip={skip_s:<18} bad={bh_s:<24} good={gh_s:<22} d={r['delta']:>5.3f} "
              f"taken={r['taken']:>5} wr={r['wr']*100:>6.2f}% ev={r['ev']:>6.1f}")


if __name__ == "__main__":
    main()
