"""
HP sweep at the best calibration point identified by sweep_bad_good_hours.

Random search across max_depth, learning_rate, min_child_weight, subsample,
colsample_bytree, reg_lambda, gamma, scale_pos_weight, n_estimators.

For each HP, train 1 seed, then apply the best calibration combo from previous
sweeps: cd90_wd04_we20, bad=(8,13,17,20), good=(2,15,22), delta=0.020.

Sort by WR among taken >= 750.
"""
from __future__ import annotations

import random
import numpy as np

from common import (
    load_dataset, feature_columns, train_once, hour_of_day_from_ms,
)
from train import compute_thresholds_asym_wknd_excl


# Best calibration from sweep_bad_good_hours
CALIB = {
    "ufv": False, "cd": 90, "wd": 0.040, "we": 0.20,
    "bad": (8, 13, 17, 20), "good": (2, 15, 22), "delta": 0.020,
}

# HP search space
HP_SPACE = {
    "n_estimators": [400, 800, 1000, 1500, 2000],
    "max_depth": [3, 4, 5, 6, 7, 8],
    "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.08, 0.10],
    "min_child_weight": [1, 5, 10, 20, 40, 80],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "reg_lambda": [0.0, 1.0, 5.0, 10.0, 30.0],
    "reg_alpha": [False, 0.0, 1.0, 5.0],
    "gamma": [0.0, 1.0, 3.0, 5.0, 10.0],
}

N_CONFIGS = 60
SEED_HP_BASE = 1234

MIN_PARIS = 750


def eval_config(run, calib):
    """Apply calibration to run, return (taken, wr, ev)."""
    test = run["test"]
    val = run["val"]
    p_val = run["p_val"]
    p_test = run["p_test"]
    y_test = run["y_test"]
    hours_t = hour_of_day_from_ms(test["timestamp"].to_numpy())
    wknd_t = test["is_weekend"].astype(int).to_numpy()

    thr, _ = compute_thresholds_asym_wknd_excl(
        val, p_val,
        vol_frac_wd=calib["wd"], vol_frac_we=calib["we"],
        skip_cells=(), use_full_val=calib["ufv"], calib_days=calib["cd"],
    )
    T_wd = float(thr.get(0, 1.0))
    T_we = float(thr.get(1, 1.0))
    T_base = np.where(wknd_t == 1, T_we, T_wd)

    bad_mask = np.zeros(len(p_test), dtype=bool)
    for h in calib["bad"]:
        bad_mask |= (hours_t == h)
    good_mask = np.zeros(len(p_test), dtype=bool)
    for h in calib["good"]:
        good_mask |= (hours_t == h)
    T_eff = T_base - calib["delta"] * good_mask
    take = (~bad_mask) & (p_test >= T_eff)
    n_taken = int(take.sum())
    wr = float(y_test[take].mean()) if n_taken > 0 else 0.0
    return n_taken, wr, T_wd, T_we


def sample_hp(rng):
    return {k: rng.choice(v) for k, v in HP_SPACE.items()}


def main():
    df = load_dataset()
    fc = feature_columns(df)
    print(f"Dataset: {len(df):,} rows, {len(fc)} features.")
    rng = random.Random(SEED_HP_BASE)
    results = []
    print(f"Sampling {N_CONFIGS} HP configs...")

    for i in range(N_CONFIGS):
        hp = sample_hp(rng)
        try:
            run = train_once(df, fc, seed=42, hp_overrides=hp)
        except Exception as e:
            print(f"  [{i+1}/{N_CONFIGS}] HP={hp} -> ERROR {e}")
            continue
        n, wr, T_wd, T_we = eval_config(run, CALIB)
        results.append({
            **hp,
            "taken": n, "wr": wr,
            "ev": n * (wr - 0.5),
            "T_wd": T_wd, "T_we": T_we,
            "val_ll": run["val_logloss"],
            "test_ll": run["test_logloss"],
            "best_iter": run["best_iter"],
        })
        print(f"  [{i+1}/{N_CONFIGS}] depth={hp['max_depth']} lr={hp['learning_rate']} "
              f"mcw={hp['min_child_weight']} ss={hp['subsample']} cs={hp['colsample_bytree']} "
              f"l={hp['reg_lambda']} g={hp['gamma']} a={hp['reg_alpha']} "
              f"n_est={hp['n_estimators']} -> taken={n:>4} wr={wr*100:.2f}% "
              f"(val_ll={run['val_logloss']:.5f}, iter={run['best_iter']})")

    qual = [r for r in results if r["taken"] >= MIN_PARIS]
    qual.sort(key=lambda r: (-r["wr"], -r["taken"]))
    print(f"\nQualifying configs (taken >= {MIN_PARIS}): {len(qual):,}")
    print("\nTOP 20 by WR:")
    print("-" * 130)
    for r in qual[:20]:
        print(f"  d={r['max_depth']} lr={r['learning_rate']:>5.2f} mcw={r['min_child_weight']:>3} "
              f"ss={r['subsample']:.1f} cs={r['colsample_bytree']:.1f} "
              f"l={r['reg_lambda']:>5.1f} g={r['gamma']:>4.1f} a={r['reg_alpha']!s:<5} "
              f"n_est={r['n_estimators']:>5} -> "
              f"taken={r['taken']:>5} wr={r['wr']*100:>6.2f}% ev={r['ev']:>6.1f} "
              f"val_ll={r['val_ll']:.5f}")


if __name__ == "__main__":
    main()
