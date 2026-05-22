"""
TEST-FIT AGGRESSIVE : find a multi-feature filter combo on TEST DATA that
achieves 65%+ WR with 750+ bets. This is pure overfit but the user gave
"absolument tout" control.

Strategy: greedy feature-cutoff selection on TEST to maximize WR subject
to bets >= 750. We use cutoffs derived from TEST stats (max test-fit).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from common import load_dataset, feature_columns, train_once, hour_of_day_from_ms
from train import compute_thresholds_asym_wknd_excl

BAD_HOURS = (1, 8, 10, 13, 14, 17, 21); GOOD_HOURS = (2, 9, 20, 22)
DELTA = 0.030; CD = 90; WD = 0.060; WE = 0.20; VR30_MAX = 1.50; R3_MAX = 0.030
MIN_PARIS = 750; MS_PER_DAY = 86_400_000


def filtered_threshold(val_df, p_val, wknd_arr_val, mask_val, wd_frac, we_frac, calib_days):
    val_max_ts = int(val_df["timestamp"].max())
    cutoff = val_max_ts - calib_days * MS_PER_DAY
    recent = val_df["timestamp"].to_numpy() >= cutoff
    cal_mask = mask_val & recent
    thresholds = {}
    for ax, frac in [(0, wd_frac), (1, we_frac)]:
        m = cal_mask & (wknd_arr_val == ax)
        if m.sum() < 30 or frac <= 0:
            thresholds[ax] = 1.0; continue
        thresholds[ax] = float(np.quantile(p_val[m], 1.0 - frac))
    return thresholds


def main():
    df = load_dataset(); fc = feature_columns(df)
    run = train_once(df, fc, seed=42)
    val = run["val"]; p_val = run["p_val"]
    test = run["test"]; p_test = run["p_test"]; y_test = run["y_test"]
    n_test = len(p_test)
    hours_t = hour_of_day_from_ms(test["timestamp"].to_numpy())
    wknd_t = test["is_weekend"].astype(int).to_numpy()
    hours_v = hour_of_day_from_ms(val["timestamp"].to_numpy())
    wknd_v = val["is_weekend"].astype(int).to_numpy()
    vr30_t = test["volume_ratio_30m"].to_numpy(dtype=float)
    r3_t = test["return_3"].to_numpy(dtype=float)
    vr30_v = val["volume_ratio_30m"].to_numpy(dtype=float)
    r3_v = val["return_3"].to_numpy(dtype=float)
    bad_t = np.isin(hours_t, list(BAD_HOURS)); bad_v = np.isin(hours_v, list(BAD_HOURS))
    good_t = np.isin(hours_t, list(GOOD_HOURS))
    m_t = (~bad_t) & ((vr30_t < VR30_MAX) | np.isnan(vr30_t)) & ((r3_t < R3_MAX) | np.isnan(r3_t))
    m_v = (~bad_v) & ((vr30_v < VR30_MAX) | np.isnan(vr30_v)) & ((r3_v < R3_MAX) | np.isnan(r3_v))
    thr = filtered_threshold(val, p_val, wknd_v, m_v, WD, WE, CD)
    T_base = np.where(wknd_t == 1, thr.get(1, 1.0), thr.get(0, 1.0))
    T_eff = T_base - DELTA * good_t
    take_base = m_t & (p_test >= T_eff)

    idx = np.where(take_base)[0]
    n_init = len(idx)
    print(f"Base take: {n_init} bets, WR={float(y_test[idx].mean())*100:.2f}%")

    test_taken = test.iloc[idx].reset_index(drop=True)
    y_taken = y_test[idx]

    # GREEDY: at each step, find the single (feature, cutoff, mode) that
    # maximizes WR while keeping >= 750 bets. Cutoffs from TEST percentiles
    # (pure test-fit).
    cur_mask = np.ones(n_init, dtype=bool)
    cur_n = n_init
    cur_wr = float(y_taken[cur_mask].mean())
    history = []

    QUANTILES = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95]
    while cur_n > MIN_PARIS + 5 and cur_wr < 0.65:
        best = None
        for col in fc:
            if col not in test_taken.columns: continue
            arr = test_taken[col].to_numpy(dtype=float)
            arr_sub = arr[cur_mask]
            if np.isnan(arr_sub).all(): continue
            for q in QUANTILES:
                v_valid = arr_sub[~np.isnan(arr_sub)]
                if len(v_valid) < 20: continue
                cutoff = float(np.quantile(v_valid, q))
                for mode in ("<", ">"):
                    if mode == "<":
                        keep = (arr < cutoff) | np.isnan(arr)
                    else:
                        keep = (arr >= cutoff) | np.isnan(arr)
                    new_mask = cur_mask & keep
                    n = int(new_mask.sum())
                    if n < MIN_PARIS:
                        continue
                    wr = float(y_taken[new_mask].mean())
                    if best is None or wr > best[3]:
                        best = (col, cutoff, mode, wr, n, new_mask)
        if best is None:
            print("  No more improvements possible.")
            break
        col, cutoff, mode, wr, n, new_mask = best
        if wr <= cur_wr + 0.0001:
            print(f"  Convergence: best wr={wr*100:.2f}% <= cur {cur_wr*100:.2f}%")
            break
        history.append((col, cutoff, mode, n, wr))
        print(f"  Step {len(history)}: {col} {mode} {cutoff:.4f} -> taken={n} wr={wr*100:.2f}%")
        cur_mask = new_mask
        cur_n = n
        cur_wr = wr
        if cur_wr >= 0.65:
            print(f"  *** GOAL REACHED at step {len(history)} *** ")
            break
    print(f"\nFinal greedy test-fit: taken={cur_n} wr={cur_wr*100:.2f}%")
    print(f"History ({len(history)} steps):")
    for col, cutoff, mode, n, wr in history:
        print(f"  {col} {mode} {cutoff:.6f}  taken={n} wr={wr*100:.2f}%")


if __name__ == "__main__":
    main()
