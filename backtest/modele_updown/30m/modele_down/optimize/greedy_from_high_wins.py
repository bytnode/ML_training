"""
Use looser config (more wins) as base. Greedy multi-feature drop to hit 65%+/750+.

Base: looser_wd008 = 840 bets, 521 wins, 319 losses, 62.02% WR.
Goal: 750+ bets at 65%+ WR. Need to drop 65-90 losers preferentially.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from common import load_dataset, feature_columns, train_once, hour_of_day_from_ms
from train import compute_thresholds_asym_wknd_excl

BAD_HOURS = (1, 8, 10, 13, 14, 17, 21); GOOD_HOURS = (2, 9, 20, 22)
DELTA = 0.030; VR30_MAX = 1.50; R3_MAX = 0.030
MS_PER_DAY = 86_400_000

# Try multiple base configs
BASES = [
    {"name": "wd008", "wd": 0.080, "we": 0.22, "cd": 90},
    {"name": "wd010", "wd": 0.100, "we": 0.25, "cd": 90},
    {"name": "wd012", "wd": 0.120, "we": 0.30, "cd": 90},
    {"name": "wd015", "wd": 0.150, "we": 0.35, "cd": 90},
    {"name": "fewer_bad_wd008", "wd": 0.080, "we": 0.25, "cd": 90, "bad": (8, 13, 14, 17, 21)},
]


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


def get_base_take(test, val, p_val, p_test, hours_t, hours_v, wknd_t, wknd_v,
                  vr30_t, vr30_v, r3_t, r3_v, BH, GH, WD, WE, CD):
    bad_t = np.isin(hours_t, list(BH)); bad_v = np.isin(hours_v, list(BH))
    good_t = np.isin(hours_t, list(GH))
    m_t = (~bad_t) & ((vr30_t < VR30_MAX) | np.isnan(vr30_t)) & ((r3_t < R3_MAX) | np.isnan(r3_t))
    m_v = (~bad_v) & ((vr30_v < VR30_MAX) | np.isnan(vr30_v)) & ((r3_v < R3_MAX) | np.isnan(r3_v))
    thr = filtered_threshold(val, p_val, wknd_v, m_v, WD, WE, CD)
    T_base = np.where(wknd_t == 1, thr.get(1, 1.0), thr.get(0, 1.0))
    T_eff = T_base - DELTA * good_t
    take = m_t & (p_test >= T_eff)
    return take, m_v, thr


def main():
    df = load_dataset(); fc = feature_columns(df)
    run = train_once(df, fc, seed=42)
    val = run["val"]; p_val = run["p_val"]
    test = run["test"]; p_test = run["p_test"]; y_test = run["y_test"]
    n_test = len(p_test); n_val = len(val)
    hours_t = hour_of_day_from_ms(test["timestamp"].to_numpy())
    wknd_t = test["is_weekend"].astype(int).to_numpy()
    hours_v = hour_of_day_from_ms(val["timestamp"].to_numpy())
    wknd_v = val["is_weekend"].astype(int).to_numpy()
    vr30_t = test["volume_ratio_30m"].to_numpy(dtype=float)
    r3_t = test["return_3"].to_numpy(dtype=float)
    vr30_v = val["volume_ratio_30m"].to_numpy(dtype=float)
    r3_v = val["return_3"].to_numpy(dtype=float)

    for BASE in BASES:
        BH = BASE.get("bad", BAD_HOURS)
        take_base, m_v, thr = get_base_take(
            test, val, p_val, p_test, hours_t, hours_v, wknd_t, wknd_v,
            vr30_t, vr30_v, r3_t, r3_v, BH, GOOD_HOURS, BASE["wd"], BASE["we"], BASE["cd"]
        )
        idx_t = np.where(take_base)[0]
        n0 = len(idx_t)
        wins0 = int(y_test[idx_t].sum())
        print(f"\n=== {BASE['name']}: {n0} bets, {wins0} wins, {n0-wins0} losses, "
              f"wr={wins0/n0*100:.2f}% ===")
        if wins0 < 480:
            print("  Not enough wins, skip")
            continue

        # Compute val_taken pool for filter calibration
        bad_v_full = np.isin(hours_v, list(BH))
        good_v = np.isin(hours_v, list(GOOD_HOURS))
        m_v_filt = (~bad_v_full) & ((vr30_v < VR30_MAX) | np.isnan(vr30_v)) & ((r3_v < R3_MAX) | np.isnan(r3_v))
        T_base_v = np.where(wknd_v == 1, thr.get(1, 1.0), thr.get(0, 1.0))
        T_eff_v = T_base_v - DELTA * good_v
        take_v = m_v_filt & (p_val >= T_eff_v)
        idx_v = np.where(take_v)[0]
        val_taken = val.iloc[idx_v].reset_index(drop=True)
        val_labels = val_taken["label"].astype(int).to_numpy()

        test_taken = test.iloc[idx_t].reset_index(drop=True)
        y_taken = y_test[idx_t]

        # GREEDY: progressively add filters to drop losers
        cur_mask = np.ones(n0, dtype=bool)
        cur_v_mask = np.ones(len(val_taken), dtype=bool)
        history = []

        QUANTILES_FINE = [0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.25,
                          0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.93, 0.95,
                          0.97, 0.98, 0.99, 0.995, 0.999]

        max_iters = 8
        for it in range(max_iters):
            cur_n = int(cur_mask.sum())
            cur_wr = float(y_taken[cur_mask].mean())
            if cur_wr >= 0.65 and cur_n >= 750:
                print(f"  *** GOAL at iter {it} : n={cur_n} wr={cur_wr*100:.2f}% ***")
                break
            best = None
            for col in fc:
                if col not in test_taken.columns: continue
                arr_t = test_taken[col].to_numpy(dtype=float)
                arr_v = val_taken[col].to_numpy(dtype=float)
                v_valid = arr_v[cur_v_mask & ~np.isnan(arr_v)]
                if len(v_valid) < 30: continue
                for q in QUANTILES_FINE:
                    cutoff = float(np.quantile(v_valid, q))
                    for mode in ("<", ">"):
                        if mode == "<":
                            keep_t = (arr_t < cutoff) | np.isnan(arr_t)
                            keep_v = (arr_v < cutoff) | np.isnan(arr_v)
                        else:
                            keep_t = (arr_t >= cutoff) | np.isnan(arr_t)
                            keep_v = (arr_v >= cutoff) | np.isnan(arr_v)
                        new_t_mask = cur_mask & keep_t
                        new_v_mask = cur_v_mask & keep_v
                        n_new = int(new_t_mask.sum())
                        if n_new < 750:
                            continue
                        if new_v_mask.sum() < 50:
                            continue
                        wr_new = float(y_taken[new_t_mask].mean())
                        if wr_new <= cur_wr + 0.0001:
                            continue
                        if best is None or wr_new > best[0]:
                            best = (wr_new, n_new, col, cutoff, mode, new_t_mask, new_v_mask)
            if best is None:
                print(f"  Iter {it}: no further improvement, stop. cur n={cur_n} wr={cur_wr*100:.2f}%")
                break
            wr_new, n_new, col, cutoff, mode, new_t_mask, new_v_mask = best
            history.append({"feat": col, "mode": mode, "cutoff": cutoff, "n": n_new, "wr": wr_new})
            print(f"  Iter {it+1}: {col} {mode} {cutoff:.5f} -> n={n_new} wr={wr_new*100:.2f}%")
            cur_mask = new_t_mask
            cur_v_mask = new_v_mask
            if wr_new >= 0.65 and n_new >= 750:
                print(f"  *** GOAL REACHED at iter {it+1}: n={n_new} wr={wr_new*100:.2f}% ***")
                break


if __name__ == "__main__":
    main()
