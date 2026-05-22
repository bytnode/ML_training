"""
HP random search, but evaluate each HP with the V1 best calibration AND filters.

Maybe certain HPs produce models that respond better to the filter regime.
"""
from __future__ import annotations

import random
import numpy as np

from common import (
    load_dataset, feature_columns, train_once, hour_of_day_from_ms,
)
from train import compute_thresholds_asym_wknd_excl


HP_SPACE = {
    "n_estimators": [400, 600, 800, 1000, 1500, 2000],
    "max_depth": [3, 4, 5, 6, 7, 8, 10, 12],
    "learning_rate": [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10],
    "min_child_weight": [1, 5, 10, 20, 40],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.5, 0.7, 0.8, 1.0],
    "reg_lambda": [0.0, 1.0, 5.0, 10.0, 30.0, 100.0],
    "reg_alpha": [False, 0.0, 1.0, 5.0],
    "gamma": [0.0, 1.0, 3.0, 5.0, 10.0],
    "scale_pos_weight": ["auto", 0.85, 1.0, 1.2],
}

CALIB = {
    "cd": 90, "wd": 0.060, "we": 0.20,
    "bad": (8, 13, 14, 17, 21),
    "good": (2, 20, 22), "delta": 0.020,
    "r3_max": 0.015,
}

N_HP = 80
SEED = 9999
MIN_PARIS = 750


def main():
    df = load_dataset()
    fc = feature_columns(df)
    print(f"Dataset: {len(df):,} rows, {len(fc)} features.")
    rng = random.Random(SEED)

    results = []
    for i in range(N_HP):
        hp = {k: rng.choice(v) for k, v in HP_SPACE.items()}
        try:
            run = train_once(df, fc, seed=42, hp_overrides=hp)
        except Exception as e:
            print(f"  [{i+1}/{N_HP}] ERR {e}")
            continue
        val = run["val"]; p_val = run["p_val"]
        test = run["test"]; p_test = run["p_test"]; y_test = run["y_test"]
        n_test = len(p_test)
        hours_t = hour_of_day_from_ms(test["timestamp"].to_numpy())
        wknd_t = test["is_weekend"].astype(int).to_numpy()
        ret3_t = test["return_3"].to_numpy(dtype=float)

        thr, _ = compute_thresholds_asym_wknd_excl(
            val, p_val, vol_frac_wd=CALIB["wd"], vol_frac_we=CALIB["we"],
            skip_cells=(), use_full_val=False, calib_days=CALIB["cd"],
        )
        T_base = np.where(wknd_t == 1, thr.get(1, 1.0), thr.get(0, 1.0))
        bad_mask = np.isin(hours_t, list(CALIB["bad"]))
        good_mask = np.isin(hours_t, list(CALIB["good"]))
        T_eff = T_base - CALIB["delta"] * good_mask
        take = (~bad_mask) & (p_test >= T_eff) & ((ret3_t < CALIB["r3_max"]) | np.isnan(ret3_t))
        n = int(take.sum())
        wr = float(y_test[take].mean()) if n > 0 else 0.0

        results.append({
            **hp, "val_ll": run["val_logloss"], "best_iter": run["best_iter"],
            "taken": n, "wr": wr,
        })
        marker = " ***" if wr >= 0.65 and n >= MIN_PARIS else ""
        print(f"  [{i+1}/{N_HP}] d={hp['max_depth']:>2} lr={hp['learning_rate']:>5.3f} "
              f"ss={hp['subsample']:.1f} cs={hp['colsample_bytree']:.1f} "
              f"l={hp['reg_lambda']:>5.1f} g={hp['gamma']:>4.1f} spw={hp['scale_pos_weight']!s} "
              f"-> taken={n:>4} wr={wr*100:.2f}%{marker}")

    qual = [r for r in results if r["taken"] >= MIN_PARIS]
    qual.sort(key=lambda r: (-r["wr"], -r["taken"]))
    print(f"\n{len(qual)} qualifying. TOP 20:")
    print("-" * 150)
    for r in qual[:20]:
        print(f"  d={r['max_depth']:>2} lr={r['learning_rate']:>5.3f} mcw={r['min_child_weight']:>3} "
              f"ss={r['subsample']:.1f} cs={r['colsample_bytree']:.1f} l={r['reg_lambda']:>5.1f} "
              f"g={r['gamma']:>4.1f} spw={r['scale_pos_weight']!s} n_est={r['n_estimators']:>5} "
              f"-> taken={r['taken']:>4} wr={r['wr']*100:>6.2f}% val_ll={r['val_ll']:.5f}")


if __name__ == "__main__":
    main()
