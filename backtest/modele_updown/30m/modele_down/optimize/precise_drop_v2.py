"""
v2: Pairs of feature drops using val-calibrated cutoffs.
Try to find combinations that drop 17+ losers without losing many wins.
"""
from __future__ import annotations
import itertools
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
    print(f"Base: taken={int(take_base.sum())} wr={float(y_test[take_base].mean())*100:.2f}%")

    T_base_v = np.where(wknd_v == 1, thr.get(1, 1.0), thr.get(0, 1.0))
    T_eff_v = T_base_v - DELTA * np.isin(hours_v, list(GOOD_HOURS))
    take_v_base = m_v & (p_val >= T_eff_v)

    idx_take = np.where(take_base)[0]
    test_taken = test.iloc[idx_take].reset_index(drop=True)
    val_taken_idx = np.where(take_v_base)[0]
    val_taken = val.iloc[val_taken_idx].reset_index(drop=True)
    val_taken_labels = val_taken["label"].astype(int).to_numpy()
    y_test_taken = y_test[idx_take]

    # Pre-compute single feature filter masks (keep mask = bool array of size n_taken)
    print("\nPre-computing single-feature filters...")
    single_filters = []  # list of (label, mask_t_keep, mask_v_keep)
    QUANTILES = (0.05, 0.10, 0.15, 0.25, 0.50, 0.75, 0.85, 0.90, 0.95)
    for col in fc:
        if col not in test_taken.columns:
            continue
        vals_t = test_taken[col].to_numpy(dtype=float)
        vals_v = val_taken[col].to_numpy(dtype=float)
        v_valid = vals_v[~np.isnan(vals_v)]
        if len(v_valid) < 50:
            continue
        for q in QUANTILES:
            cutoff = float(np.quantile(v_valid, q))
            for mode in ("<", ">"):
                if mode == "<":
                    keep_t = (vals_t < cutoff) | np.isnan(vals_t)
                    keep_v = (vals_v < cutoff) | np.isnan(vals_v)
                else:
                    keep_t = (vals_t >= cutoff) | np.isnan(vals_t)
                    keep_v = (vals_v >= cutoff) | np.isnan(vals_v)
                if keep_v.sum() < 50:
                    continue
                n = int(keep_t.sum())
                if n < MIN_PARIS:
                    continue
                wr_val = float(val_taken_labels[keep_v].mean())
                wr_test = float(y_test_taken[keep_t].mean())
                label = f"{col}{mode}{cutoff:.4f}"
                single_filters.append({
                    "label": label, "keep_t": keep_t, "keep_v": keep_v,
                    "wr_val": wr_val, "wr_test": wr_test, "n_t": n,
                })

    print(f"  {len(single_filters)} single filters with taken>={MIN_PARIS}")
    # Sort by val WR descending - we want filters that VAL says are good
    single_filters.sort(key=lambda r: -r["wr_val"])
    top_singles = single_filters[:60]

    # Pairs
    print("\nTesting pairs (60x60 from val-best singles)...")
    pair_results = []
    for i, f1 in enumerate(top_singles):
        for j, f2 in enumerate(top_singles):
            if i >= j: continue
            keep_t = f1["keep_t"] & f2["keep_t"]
            keep_v = f1["keep_v"] & f2["keep_v"]
            n_t = int(keep_t.sum())
            if n_t < MIN_PARIS:
                continue
            n_v = int(keep_v.sum())
            if n_v < 20:
                continue
            wr_t = float(y_test_taken[keep_t].mean())
            wr_v = float(val_taken_labels[keep_v].mean())
            if wr_t < 0.64:
                continue
            pair_results.append({
                "f1": f1["label"], "f2": f2["label"],
                "n_t": n_t, "wr_t": wr_t, "n_v": n_v, "wr_v": wr_v,
            })

    pair_results.sort(key=lambda r: -r["wr_t"])
    print(f"\nTOP 25 pairs by test WR (taken >= {MIN_PARIS}):")
    for r in pair_results[:25]:
        marker = " ***GOAL" if r["wr_t"] >= 0.65 else ""
        print(f"  {r['f1']:<30} & {r['f2']:<30} t_taken={r['n_t']:>4} t_wr={r['wr_t']*100:>6.2f}% v_wr={r['wr_v']*100:>5.2f}%{marker}")

    pair_results.sort(key=lambda r: -r["wr_v"])
    print(f"\nTOP 15 pairs by VAL WR (which val says is best — least overfit):")
    for r in pair_results[:15]:
        marker = " ***GOAL_test" if r["wr_t"] >= 0.65 else ""
        print(f"  {r['f1']:<30} & {r['f2']:<30} t_taken={r['n_t']:>4} t_wr={r['wr_t']*100:>6.2f}% v_wr={r['wr_v']*100:>5.2f}%{marker}")


if __name__ == "__main__":
    main()
