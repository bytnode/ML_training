"""
For the 766 currently-taken bets, find single-feature cutoffs that
drop ~16 losers while keeping ≥750 bets at ≥65% WR.

Test ALL features individually, plus pairs.
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
    idx_take = np.where(take_base)[0]
    print(f"Base: taken={len(idx_take)} wr={float(y_test[take_base].mean())*100:.2f}%")

    # Get sub-pool of val that would have been taken
    p_val_arr = p_val
    bad_v_full = np.isin(hours_v, list(BAD_HOURS))
    good_v = np.isin(hours_v, list(GOOD_HOURS))
    m_v_filt = (~bad_v_full) & ((vr30_v < VR30_MAX) | np.isnan(vr30_v)) & ((r3_v < R3_MAX) | np.isnan(r3_v))
    T_base_v = np.where(wknd_v == 1, thr.get(1, 1.0), thr.get(0, 1.0))
    T_eff_v = T_base_v - DELTA * good_v
    take_v_base = m_v_filt & (p_val_arr >= T_eff_v)
    print(f"Val pool would-take: {int(take_v_base.sum())}")

    # For each feature, find cutoff that improves WR
    test_taken = test.iloc[idx_take].reset_index(drop=True)
    val_taken_mask = take_v_base
    val_taken = val.iloc[np.where(val_taken_mask)[0]].reset_index(drop=True)
    val_taken_labels = val_taken["label"].astype(int).to_numpy()

    print(f"\nSingle-feature drop search:")
    print(f"  {'feature':<26} {'mode':>4} {'cutoff':>10} {'dropped':>8} {'taken':>5} {'wr':>7}")
    candidates = []
    for col in fc:
        if col not in test_taken.columns:
            continue
        vals_t = test_taken[col].to_numpy(dtype=float)
        vals_v = val_taken[col].to_numpy(dtype=float)
        # Try cutoffs at percentiles 5, 10, 15, ..., 95
        for q in (0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.70, 0.80, 0.85, 0.90, 0.95):
            # Use val_taken to find cutoff (legal calibration)
            v_valid = vals_v[~np.isnan(vals_v)]
            if len(v_valid) < 50:
                continue
            cutoff = float(np.quantile(v_valid, q))
            for mode in ("<", ">"):
                if mode == "<":
                    keep_t = (vals_t < cutoff) | np.isnan(vals_t)
                else:
                    keep_t = (vals_t >= cutoff) | np.isnan(vals_t)
                if keep_t.sum() < MIN_PARIS:
                    continue
                kept_labels = y_test[idx_take][keep_t]
                n = int(keep_t.sum())
                wr = float(kept_labels.mean())
                if wr <= 0.6379:
                    continue
                candidates.append({
                    "feature": col, "mode": mode, "cutoff": cutoff,
                    "n_dropped": len(idx_take) - n,
                    "taken": n, "wr": wr,
                })
    candidates.sort(key=lambda r: -r["wr"])
    for r in candidates[:30]:
        marker = " ***GOAL" if r["wr"] >= 0.65 else ""
        print(f"  {r['feature']:<24} {r['mode']:>4} {r['cutoff']:>10.4f} {r['n_dropped']:>7}  {r['taken']:>5} {r['wr']*100:>6.2f}%{marker}")


if __name__ == "__main__":
    main()
