"""
Voting consensus: train N=20 diverse models. Each has its own threshold (top 8%).
Take a DOWN bet only when at least K models say "this is in top 8%".

K=N (unanimous) is strictest. K=N/2 is majority.
"""
from __future__ import annotations

import random
import numpy as np

from common import (
    load_dataset, feature_columns, train_once, hour_of_day_from_ms,
)


N_MODELS = 20
SEED_BASE = 5555
MIN_PARIS = 750

BAD_HOURS = (1, 8, 10, 13, 14, 17, 21)
GOOD_HOURS = (2, 9, 20, 22)
DELTA = 0.030
CD = 90; WD = 0.060; WE = 0.20
VR30_MAX = 1.50
R3_MAX = 0.030


def main():
    df = load_dataset()
    full_fc = feature_columns(df)
    print(f"Dataset: {len(df):,} rows, {len(full_fc)} features.")

    rng = random.Random(SEED_BASE)
    runs = []
    for i in range(N_MODELS):
        sub_rng = random.Random(SEED_BASE + i * 31)
        n_keep = sub_rng.randint(int(0.65 * len(full_fc)), len(full_fc))
        fc_sub = sub_rng.sample(full_fc, n_keep)
        hp_overrides = {
            "max_depth": sub_rng.choice([5, 6, 7, 8]),
            "learning_rate": sub_rng.choice([0.02, 0.03, 0.05]),
            "min_child_weight": sub_rng.choice([5, 10, 20]),
            "subsample": sub_rng.choice([0.7, 0.8, 0.9, 1.0]),
            "colsample_bytree": sub_rng.choice([0.7, 0.8, 1.0]),
            "reg_lambda": sub_rng.choice([1.0, 5.0, 10.0]),
            "gamma": sub_rng.choice([0.0, 1.0, 3.0]),
        }
        run = train_once(df, fc_sub, seed=42 + i, hp_overrides=hp_overrides)
        runs.append(run)
        print(f"  [{i+1}/{N_MODELS}] hp={hp_overrides} n_feats={n_keep} val_ll={run['val_logloss']:.5f}")

    test = runs[0]["test"]
    val = runs[0]["val"]
    y_test = runs[0]["y_test"]
    n_test = len(y_test)
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
    m_t = (~bad_t) & ((vr30_t < VR30_MAX) | np.isnan(vr30_t)) & ((r3_t < R3_MAX) | np.isnan(r3_t))
    m_v = (~bad_v) & ((vr30_v < VR30_MAX) | np.isnan(vr30_v)) & ((r3_v < R3_MAX) | np.isnan(r3_v))

    # Per-model: compute the take vector
    take_per_model = []
    for r in runs:
        # Recalibrate each model on its own val
        ms_per_day = 86_400_000
        val_max_ts = int(val["timestamp"].max())
        cutoff = val_max_ts - CD * ms_per_day
        cal_recent = val["timestamp"].to_numpy() >= cutoff
        cal_mask = m_v & cal_recent
        p_val_r = r["p_val"]
        thrs = {}
        for ax, frac in [(0, WD), (1, WE)]:
            mm = cal_mask & (wknd_v == ax)
            if mm.sum() < 30:
                thrs[ax] = 1.0
                continue
            thrs[ax] = float(np.quantile(p_val_r[mm], 1.0 - frac))
        T_base = np.where(wknd_t == 1, thrs.get(1, 1.0), thrs.get(0, 1.0))
        T_eff = T_base - DELTA * good_t
        take = m_t & (r["p_test"] >= T_eff)
        take_per_model.append(take)

    take_per_model = np.array(take_per_model)  # (N, n_test) bool
    # For each test row, count how many models say take
    take_count = take_per_model.sum(axis=0)

    print(f"\n=== Voting consensus results ===")
    print(f"{'K (min agreement)':>20} {'taken':>6} {'wr':>7}")
    for K in range(1, N_MODELS + 1):
        consensus = (take_count >= K)
        n = int(consensus.sum())
        if n == 0:
            continue
        wr = float(y_test[consensus].mean())
        marker = " *** GOAL" if (wr >= 0.65 and n >= MIN_PARIS) else ""
        print(f"  >= {K:>2}/{N_MODELS}      {n:>6}  {wr*100:>5.2f}%{marker}")


if __name__ == "__main__":
    main()
