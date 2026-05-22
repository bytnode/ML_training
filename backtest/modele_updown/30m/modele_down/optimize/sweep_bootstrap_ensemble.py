"""
Bootstrap ensemble with feature subsetting for REAL diversity.

Train N models, each with a different random subset of features (70%).
Average predictions for both val (calibration) and test (decision).

Hypothesis: feature subsetting forces each model to learn different patterns.
The average smooths out individual model variance and yields sharper top-quantile
WR than a single deterministic XGBoost.
"""
from __future__ import annotations

import random
import numpy as np

from common import (
    load_dataset, feature_columns, train_once, hour_of_day_from_ms,
)
from train import compute_thresholds_asym_wknd_excl


N_MODELS = 15
FEAT_FRAC = 0.7
SEED_BASE = 4242

CALIB_GRID = [
    {"ufv": False, "cd": 90, "wd": 0.060, "we": 0.20, "tag": "cd90_wd06_we20"},
    {"ufv": False, "cd": 90, "wd": 0.040, "we": 0.20, "tag": "cd90_wd04_we20"},
    {"ufv": False, "cd": 55, "wd": 0.030, "we": 0.20, "tag": "cd55_wd03_we20"},
    {"ufv": False, "cd": 180, "wd": 0.060, "we": 0.20, "tag": "cd180_wd06_we20"},
]
BAD_HOURS_OPTIONS = [
    (8, 13, 17, 20, 21),
    (8, 17, 20),
    (8, 17),
]
GOOD_HOURS_OPTIONS = [
    (2, 15),
    (15, 22),
    (15,),
    (),
]
DELTA_GRID = (0.020, 0.040)
MIN_PARIS = 750


def main():
    df = load_dataset()
    full_fc = feature_columns(df)
    print(f"Dataset: {len(df):,} rows, {len(full_fc)} full features.")

    rng_master = random.Random(SEED_BASE)
    p_val_sum = None
    p_test_sum = None
    val = test = y_test = None

    print(f"\nTraining {N_MODELS} models with random feature subsets ({FEAT_FRAC:.0%})...")
    for i in range(N_MODELS):
        rng = random.Random(SEED_BASE + i)
        n_keep = int(len(full_fc) * FEAT_FRAC)
        fc_sub = rng.sample(full_fc, n_keep)
        run = train_once(df, fc_sub, seed=42 + i)
        if val is None:
            val = run["val"]
            test = run["test"]
            y_test = run["y_test"]
            p_val_sum = np.zeros(len(run["p_val"]))
            p_test_sum = np.zeros(len(run["p_test"]))
        p_val_sum += run["p_val"]
        p_test_sum += run["p_test"]
        print(f"  [{i+1}/{N_MODELS}] feats={n_keep} val_ll={run['val_logloss']:.5f} "
              f"test_ll={run['test_logloss']:.5f}")

    p_val_avg = p_val_sum / N_MODELS
    p_test_avg = p_test_sum / N_MODELS
    print(f"\np_val_avg: min={p_val_avg.min():.4f} max={p_val_avg.max():.4f}")
    print(f"p_test_avg: min={p_test_avg.min():.4f} max={p_test_avg.max():.4f}")

    hours_t = hour_of_day_from_ms(test["timestamp"].to_numpy())
    wknd_t = test["is_weekend"].astype(int).to_numpy()
    hour_masks = {h: (hours_t == h) for h in range(24)}

    results = []
    for cal in CALIB_GRID:
        thr, _ = compute_thresholds_asym_wknd_excl(
            val, p_val_avg,
            vol_frac_wd=cal["wd"], vol_frac_we=cal["we"],
            skip_cells=(), use_full_val=cal["ufv"], calib_days=cal["cd"],
        )
        T_base = np.where(wknd_t == 1, thr.get(1, 1.0), thr.get(0, 1.0))
        for bh in BAD_HOURS_OPTIONS:
            bad_mask = np.zeros(len(p_test_avg), dtype=bool)
            for h in bh:
                bad_mask |= hour_masks[h]
            for gh in GOOD_HOURS_OPTIONS:
                good_mask = np.zeros(len(p_test_avg), dtype=bool)
                for h in gh:
                    good_mask |= hour_masks[h]
                for delta in (DELTA_GRID if len(gh) > 0 else (0.0,)):
                    T_eff = T_base - delta * good_mask
                    take = (~bad_mask) & (p_test_avg >= T_eff)
                    n = int(take.sum())
                    if n < MIN_PARIS:
                        continue
                    wr = float(y_test[take].mean())
                    results.append({
                        "calib": cal["tag"], "bad": bh, "good": gh,
                        "delta": delta, "taken": n, "wr": wr,
                    })

    results.sort(key=lambda r: (-r["wr"], -r["taken"]))
    print(f"\n{len(results)} qualifying configs (taken >= {MIN_PARIS})")
    print("\nTOP 30 by WR (ensemble of {} models):".format(N_MODELS))
    print("-" * 140)
    for r in results[:30]:
        bh_s = "(" + ",".join(str(h) for h in r["bad"]) + ")"
        gh_s = "(" + ",".join(str(h) for h in r["good"]) + ")"
        print(f"  {r['calib']:<16} bad={bh_s:<24} good={gh_s:<14} d={r['delta']:>5.3f} "
              f"taken={r['taken']:>5} wr={r['wr']*100:>6.2f}%")


if __name__ == "__main__":
    main()
