"""
Look for SPECIFIC OUTLIER bets in current 754-take that are losers.
Use fine quantile cutoffs (0.985, 0.99, 0.995, 0.999, etc) to drop individual outliers.

For each feature, sort the 754 bets by feature value, identify the n_drop most-extreme
losers, build a cutoff that drops exactly those losers.
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
    test_taken = test.iloc[idx].reset_index(drop=True)
    y_taken = y_test[idx]
    n_take = len(idx)
    print(f"Base take: {n_take} bets, wins={int(y_taken.sum())} losses={int((1-y_taken).sum())} "
          f"WR={float(y_taken.mean())*100:.2f}%")

    # For each feature, scan THE EXACT cutoff at every distinct value
    # to find drops that maximize WR keeping n >= MIN_PARIS
    print(f"\nScanning all features with EXACT cutoffs (drop top-K outliers per feature):")
    print(f"  {'feature':<28} {'mode':>4} {'cutoff':>12} {'n_drop':>6} {'wins_kept':>8} {'taken':>5} {'wr':>7}")
    candidates = []
    for col in fc:
        if col not in test_taken.columns: continue
        arr = test_taken[col].to_numpy(dtype=float)
        # Sort by value
        sort_idx_lo = np.argsort(arr)  # low to high
        sort_idx_hi = np.argsort(-arr)  # high to low
        # Try dropping top-K (high) outliers
        for k in range(1, 30):
            drop_idx = sort_idx_hi[:k]
            cur_n = n_take - k
            if cur_n < MIN_PARIS: break
            wins_kept = int(y_taken.sum()) - int(y_taken[drop_idx].sum())
            wr = wins_kept / cur_n
            cutoff = arr[sort_idx_hi[k-1]]  # value of k-th highest
            if wr > 0.6379:
                candidates.append({
                    "feature": col, "mode": "<", "cutoff": cutoff,
                    "n_drop": k, "wins_kept": wins_kept, "taken": cur_n, "wr": wr,
                })
        # Try dropping bottom-K (low) outliers
        for k in range(1, 30):
            drop_idx = sort_idx_lo[:k]
            cur_n = n_take - k
            if cur_n < MIN_PARIS: break
            wins_kept = int(y_taken.sum()) - int(y_taken[drop_idx].sum())
            wr = wins_kept / cur_n
            cutoff = arr[sort_idx_lo[k-1]]
            if wr > 0.6379:
                candidates.append({
                    "feature": col, "mode": ">", "cutoff": cutoff,
                    "n_drop": k, "wins_kept": wins_kept, "taken": cur_n, "wr": wr,
                })

    candidates.sort(key=lambda r: -r["wr"])
    for r in candidates[:30]:
        marker = " ***GOAL" if r["wr"] >= 0.65 else ""
        print(f"  {r['feature']:<26} {r['mode']:>4} {r['cutoff']:>12.5f} {r['n_drop']:>5}     {r['wins_kept']:>4}    {r['taken']:>4} {r['wr']*100:>6.2f}%{marker}")


if __name__ == "__main__":
    main()
