"""
Train K diverse DOWN models (different feature subsets), require CONSENSUS:
all models must agree p_down >= T_individual to take.

Or alternative: average their predictions, threshold on average.

Tests both approaches.
"""
from __future__ import annotations

import random
import numpy as np

from common import (
    load_dataset, feature_columns, train_once, hour_of_day_from_ms,
)


MIN_PARIS = 750
N_MODELS = 7
SEED_BASE = 7777


# Use the best config from sweep_extreme_v2
BAD_HOURS = (1, 8, 10, 13, 14, 17, 21)
GOOD_HOURS = (2, 9, 20, 22)
DELTA = 0.030
CD = 90; WD = 0.060; WE = 0.20
VR30_MAX = 1.50
R3_MAX = 0.030


def filtered_threshold(val_df, p_val, wknd_arr_val, mask_val,
                       wd_frac, we_frac, calib_days):
    ms_per_day = 86_400_000
    val_max_ts = int(val_df["timestamp"].max())
    cutoff = val_max_ts - calib_days * ms_per_day
    recent = val_df["timestamp"].to_numpy() >= cutoff
    cal_mask = mask_val & recent
    thresholds = {}
    for ax, frac in [(0, wd_frac), (1, we_frac)]:
        m = cal_mask & (wknd_arr_val == ax)
        if m.sum() < 30 or frac <= 0:
            thresholds[ax] = 1.0
            continue
        thresholds[ax] = float(np.quantile(p_val[m], 1.0 - frac))
    return thresholds


def main():
    df = load_dataset()
    full_fc = feature_columns(df)
    print(f"Dataset: {len(df):,} rows, {len(full_fc)} features.")

    p_val_list = []
    p_test_list = []
    val = test = y_test = None

    print(f"\nTraining {N_MODELS} diverse models...")
    rng_master = random.Random(SEED_BASE)
    for i in range(N_MODELS):
        rng = random.Random(SEED_BASE + i * 37)
        n_keep = rng.randint(int(0.60 * len(full_fc)), int(0.95 * len(full_fc)))
        fc_sub = rng.sample(full_fc, n_keep)
        # Different HP per model for diversity
        hp_overrides = {
            "max_depth": rng.choice([4, 5, 6, 7, 8]),
            "learning_rate": rng.choice([0.02, 0.03, 0.05, 0.08]),
            "subsample": rng.choice([0.7, 0.8, 0.9, 1.0]),
            "colsample_bytree": rng.choice([0.6, 0.7, 0.8, 1.0]),
            "reg_lambda": rng.choice([0.0, 1.0, 5.0, 10.0]),
            "gamma": rng.choice([0.0, 1.0, 3.0]),
        }
        run = train_once(df, fc_sub, seed=42 + i, hp_overrides=hp_overrides)
        if val is None:
            val = run["val"]
            test = run["test"]
            y_test = run["y_test"]
        p_val_list.append(run["p_val"])
        p_test_list.append(run["p_test"])
        print(f"  [{i+1}/{N_MODELS}] n_feats={n_keep} {hp_overrides} "
              f"val_ll={run['val_logloss']:.5f}")

    p_val_arr = np.array(p_val_list)  # (N, n_val)
    p_test_arr = np.array(p_test_list)

    # Average
    p_val_avg = p_val_arr.mean(axis=0)
    p_test_avg = p_test_arr.mean(axis=0)
    # Min and Median for "consensus"
    p_val_min = p_val_arr.min(axis=0)
    p_test_min = p_test_arr.min(axis=0)
    p_val_med = np.median(p_val_arr, axis=0)
    p_test_med = np.median(p_test_arr, axis=0)

    hours_t = hour_of_day_from_ms(test["timestamp"].to_numpy())
    wknd_t = test["is_weekend"].astype(int).to_numpy()
    hours_v = hour_of_day_from_ms(val["timestamp"].to_numpy())
    wknd_v = val["is_weekend"].astype(int).to_numpy()
    vr30_t = test["volume_ratio_30m"].to_numpy(dtype=float)
    r3_t = test["return_3"].to_numpy(dtype=float)
    vr30_v = val["volume_ratio_30m"].to_numpy(dtype=float)
    r3_v = val["return_3"].to_numpy(dtype=float)

    bad_t = np.isin(hours_t, list(BAD_HOURS))
    bad_v = np.isin(hours_v, list(BAD_HOURS))
    good_t = np.isin(hours_t, list(GOOD_HOURS))

    m_t = ~bad_t & (vr30_t < VR30_MAX) & (r3_t < R3_MAX) | (~bad_t & np.isnan(vr30_t)) | (~bad_t & np.isnan(r3_t))
    m_t = ~bad_t
    m_t = m_t & ((vr30_t < VR30_MAX) | np.isnan(vr30_t))
    m_t = m_t & ((r3_t < R3_MAX) | np.isnan(r3_t))
    m_v = ~bad_v
    m_v = m_v & ((vr30_v < VR30_MAX) | np.isnan(vr30_v))
    m_v = m_v & ((r3_v < R3_MAX) | np.isnan(r3_v))

    print("\n=== Comparison: average vs min vs median consensus ===")
    print(f"{'Strategy':<10} {'taken':>6} {'wr':>7}")
    for name, p_v, p_t in [
        ("avg", p_val_avg, p_test_avg),
        ("min", p_val_min, p_test_min),
        ("median", p_val_med, p_test_med),
    ]:
        thr = filtered_threshold(val, p_v, wknd_v, m_v, WD, WE, CD)
        T_base = np.where(wknd_t == 1, thr.get(1, 1.0), thr.get(0, 1.0))
        T_eff = T_base - DELTA * good_t
        take = m_t & (p_t >= T_eff)
        n = int(take.sum())
        wr = float(y_test[take].mean()) if n > 0 else 0.0
        print(f"  {name:<8} {n:>6} {wr*100:>6.2f}%  (T_wd={thr.get(0):.4f}, T_we={thr.get(1):.4f})")

    # And single seed=42 for reference
    print(f"\n  single seed=42 baseline (no consensus): see prior runs (63.72-79%)")


if __name__ == "__main__":
    main()
