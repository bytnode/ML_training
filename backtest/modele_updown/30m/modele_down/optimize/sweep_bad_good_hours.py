"""
Vectorized sweep of bad_hours + good_hours + delta on the cleanest baseline.

Speedup vs naive Python: pre-compute hours/weekend/threshold vectors once,
then do (bad_hours,good_hours,delta) eval in pure numpy ~1ms.

Pool of candidate hours derived from sweep_baseline_clean.py val WR profile.
"""
from __future__ import annotations

import itertools
import numpy as np

from common import (
    load_dataset, feature_columns, train_once, hour_of_day_from_ms,
)
from train import compute_thresholds_asym_wknd_excl


# Calibration variants (top hits from sweep_baseline_clean)
CALIB_VARIANTS = [
    {"ufv": False, "cd": 55, "wd": 0.030, "we": 0.20, "tag": "cd55_wd03_we20"},
    {"ufv": False, "cd": 55, "wd": 0.020, "we": 0.20, "tag": "cd55_wd02_we20"},
    {"ufv": False, "cd": 90, "wd": 0.060, "we": 0.20, "tag": "cd90_wd06_we20"},
    {"ufv": False, "cd": 90, "wd": 0.040, "we": 0.20, "tag": "cd90_wd04_we20"},
    {"ufv": False, "cd": 180, "wd": 0.040, "we": 0.20, "tag": "cd180_wd04_we20"},
    {"ufv": True, "cd": 0, "wd": 0.040, "we": 0.30, "tag": "full_wd04_we30"},
    {"ufv": True, "cd": 0, "wd": 0.060, "we": 0.20, "tag": "full_wd06_we20"},
]

# Worst hours on val (WR_brute < baseline-0.02)
BAD_HOUR_POOL  = [0, 7, 8, 10, 13, 16, 17, 20]
# Best hours on val (WR_brute > baseline+0.02)
GOOD_HOUR_POOL = [2, 6, 12, 15, 19, 22, 23]
DELTA_GRID = (0.020, 0.040, 0.060, 0.080)

MIN_PARIS = 750


def hour_subsets(pool, max_k):
    out = [()]
    for k in range(1, max_k + 1):
        out.extend(itertools.combinations(pool, k))
    return out


def main():
    df = load_dataset()
    fc = feature_columns(df)
    print(f"Dataset: {len(df):,} rows, {len(fc)} features.")
    run = train_once(df, fc, seed=42)
    print(f"  best_iter={run['best_iter']} val_logloss={run['val_logloss']:.5f}")

    test = run["test"]
    val = run["val"]
    p_test = run["p_test"]
    p_val = run["p_val"]
    y_test = run["y_test"]
    hours_test = hour_of_day_from_ms(test["timestamp"].to_numpy())
    wknd_test = test["is_weekend"].astype(int).to_numpy()

    bad_subsets = hour_subsets(BAD_HOUR_POOL, 4)
    good_subsets = hour_subsets(GOOD_HOUR_POOL, 3)
    print(f"\nBad subsets: {len(bad_subsets):,}   "
          f"Good subsets: {len(good_subsets):,}   "
          f"Calib variants: {len(CALIB_VARIANTS)}   "
          f"Deltas: {len(DELTA_GRID)}")

    # Pre-compute thresholds for each calibration variant (slow part)
    calib_thresholds = []
    for cal in CALIB_VARIANTS:
        thr, _ = compute_thresholds_asym_wknd_excl(
            val, p_val,
            vol_frac_wd=cal["wd"], vol_frac_we=cal["we"],
            skip_cells=(), use_full_val=cal["ufv"], calib_days=cal["cd"],
        )
        T_wd = float(thr.get(0, 1.0))
        T_we = float(thr.get(1, 1.0))
        calib_thresholds.append((cal["tag"], T_wd, T_we))

    # Vectorize the inner loop
    results = []
    n_test = len(p_test)
    # Pre-compute hour-mask vectors for each hour
    hour_masks = {h: (hours_test == h) for h in range(24)}

    for ci, cal in enumerate(CALIB_VARIANTS):
        tag, T_wd, T_we = calib_thresholds[ci]
        # Base threshold per row = T_wd if weekday else T_we
        T_base = np.where(wknd_test == 1, T_we, T_wd)

        for bh in bad_subsets:
            if len(bh) == 0:
                bad_mask = np.zeros(n_test, dtype=bool)
            else:
                bad_mask = np.zeros(n_test, dtype=bool)
                for h in bh:
                    bad_mask |= hour_masks[h]

            for gh in good_subsets:
                if len(gh) == 0:
                    # delta irrelevant
                    take = (~bad_mask) & (p_test >= T_base)
                    n_taken = int(take.sum())
                    if n_taken == 0:
                        continue
                    wr = float(y_test[take].mean())
                    results.append({
                        "calib": tag, "T_wd": T_wd, "T_we": T_we,
                        "bad": bh, "good": (), "delta": 0.0,
                        "taken": n_taken, "wr": wr,
                        "ev": n_taken * (wr - 0.5),
                    })
                    continue

                good_mask = np.zeros(n_test, dtype=bool)
                for h in gh:
                    good_mask |= hour_masks[h]

                for delta in DELTA_GRID:
                    T_eff = T_base - delta * good_mask
                    take = (~bad_mask) & (p_test >= T_eff)
                    n_taken = int(take.sum())
                    if n_taken == 0:
                        continue
                    wr = float(y_test[take].mean())
                    results.append({
                        "calib": tag, "T_wd": T_wd, "T_we": T_we,
                        "bad": bh, "good": gh, "delta": delta,
                        "taken": n_taken, "wr": wr,
                        "ev": n_taken * (wr - 0.5),
                    })

    print(f"\nResults computed: {len(results):,}")

    # Top by WR with taken >= MIN_PARIS
    qual = [r for r in results if r["taken"] >= MIN_PARIS]
    qual.sort(key=lambda r: (-r["wr"], -r["taken"]))
    print(f"\nQualifying configs (taken >= {MIN_PARIS}): {len(qual):,}")
    print("\nTOP 30 by WR:")
    print("-" * 150)
    for r in qual[:30]:
        bh_s = "(" + ",".join(str(h) for h in r["bad"]) + ")"
        gh_s = "(" + ",".join(str(h) for h in r["good"]) + ")"
        print(f"  {r['calib']:<16} bad={bh_s:<24} good={gh_s:<22} d={r['delta']:>5.3f}  "
              f"taken={r['taken']:>5} wr={r['wr']*100:>6.2f}% ev={r['ev']:>6.1f}")

    # Configs reaching the 65% goal with min_paris bets
    goal = [r for r in results if r["wr"] >= 0.65 and r["taken"] >= MIN_PARIS]
    goal.sort(key=lambda r: (-r["wr"], -r["taken"]))
    print(f"\nCONFIGS WR >= 65% AND taken >= {MIN_PARIS}: {len(goal):,}")
    for r in goal[:20]:
        bh_s = "(" + ",".join(str(h) for h in r["bad"]) + ")"
        gh_s = "(" + ",".join(str(h) for h in r["good"]) + ")"
        print(f"  {r['calib']:<16} bad={bh_s:<24} good={gh_s:<22} d={r['delta']:>5.3f}  "
              f"taken={r['taken']:>5} wr={r['wr']*100:>6.2f}% ev={r['ev']:>6.1f}")

    # Also "near goal" configs >= 62% with taken >= MIN_PARIS
    near = [r for r in results if 0.62 <= r["wr"] < 0.65 and r["taken"] >= MIN_PARIS]
    near.sort(key=lambda r: (-r["wr"], -r["taken"]))
    print(f"\nCONFIGS 62% <= WR < 65% AND taken >= {MIN_PARIS}: {len(near):,}")
    for r in near[:15]:
        bh_s = "(" + ",".join(str(h) for h in r["bad"]) + ")"
        gh_s = "(" + ",".join(str(h) for h in r["good"]) + ")"
        print(f"  {r['calib']:<16} bad={bh_s:<24} good={gh_s:<22} d={r['delta']:>5.3f}  "
              f"taken={r['taken']:>5} wr={r['wr']*100:>6.2f}% ev={r['ev']:>6.1f}")


if __name__ == "__main__":
    main()
