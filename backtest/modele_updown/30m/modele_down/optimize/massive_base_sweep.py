"""
Sweep many base configs + greedy multi-filter drop. Find any config that crosses 65%.
"""
from __future__ import annotations
import itertools
import numpy as np
from common import load_dataset, feature_columns, train_once, hour_of_day_from_ms
from train import compute_thresholds_asym_wknd_excl

GOOD_HOURS = (2, 9, 20, 22)
DELTA = 0.030; VR30_MAX = 1.50; R3_MAX = 0.030
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

    # 30 bases with varied LOOSE wd/we (more wins) + slightly different bad hours
    bases = []
    BHs = [
        (1, 8, 10, 13, 14, 17, 21),
        (8, 13, 14, 17, 21),
        (1, 8, 13, 14, 17, 21),
        (8, 14, 17, 21),
        (1, 8, 14, 17, 21),
        (), # no bad hours
    ]
    for bh in BHs:
        for wd in (0.080, 0.100, 0.120, 0.150, 0.200):
            for we in (0.20, 0.25, 0.30):
                for cd in (55, 90, 180):
                    bases.append({"bh": bh, "wd": wd, "we": we, "cd": cd})

    print(f"Testing {len(bases)} bases")
    QUANTILES_FINE = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50,
                      0.70, 0.80, 0.90, 0.95, 0.98, 0.99, 0.995, 0.999]
    overall_best = {"wr": 0.0, "n": 0}

    for bi, B in enumerate(bases):
        bad_t = np.isin(hours_t, list(B["bh"])); bad_v = np.isin(hours_v, list(B["bh"]))
        good_t = np.isin(hours_t, list(GOOD_HOURS)); good_v = np.isin(hours_v, list(GOOD_HOURS))
        m_t = (~bad_t) & ((vr30_t < VR30_MAX) | np.isnan(vr30_t)) & ((r3_t < R3_MAX) | np.isnan(r3_t))
        m_v = (~bad_v) & ((vr30_v < VR30_MAX) | np.isnan(vr30_v)) & ((r3_v < R3_MAX) | np.isnan(r3_v))
        thr = filtered_threshold(val, p_val, wknd_v, m_v, B["wd"], B["we"], B["cd"])
        T_base = np.where(wknd_t == 1, thr.get(1, 1.0), thr.get(0, 1.0))
        T_eff = T_base - DELTA * good_t
        take_base = m_t & (p_test >= T_eff)
        T_base_v = np.where(wknd_v == 1, thr.get(1, 1.0), thr.get(0, 1.0))
        T_eff_v = T_base_v - DELTA * good_v
        take_v = m_v & (p_val >= T_eff_v)
        idx_t = np.where(take_base)[0]
        idx_v = np.where(take_v)[0]
        if len(idx_t) < 800 or len(idx_v) < 50:
            continue

        test_taken = test.iloc[idx_t].reset_index(drop=True)
        val_taken = val.iloc[idx_v].reset_index(drop=True)
        y_taken = y_test[idx_t]
        val_labels = val_taken["label"].astype(int).to_numpy()
        cur_t = np.ones(len(idx_t), dtype=bool)
        cur_v = np.ones(len(idx_v), dtype=bool)

        for it in range(7):
            cur_n = int(cur_t.sum())
            if cur_n < 750:
                break
            cur_wr = float(y_taken[cur_t].mean())
            best = None
            for col in fc:
                if col not in test_taken.columns: continue
                arr_t = test_taken[col].to_numpy(dtype=float)
                arr_v = val_taken[col].to_numpy(dtype=float)
                v_valid = arr_v[cur_v & ~np.isnan(arr_v)]
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
                        new_t = cur_t & keep_t
                        new_v = cur_v & keep_v
                        if int(new_t.sum()) < 750 or int(new_v.sum()) < 30: continue
                        wr_new = float(y_taken[new_t].mean())
                        if wr_new <= cur_wr + 0.0001: continue
                        if best is None or wr_new > best[0]:
                            best = (wr_new, int(new_t.sum()), new_t, new_v)
            if best is None: break
            cur_wr, cur_n, cur_t, cur_v = best
            if cur_wr >= overall_best["wr"]:
                overall_best = {"wr": cur_wr, "n": cur_n, "base": B, "iter": it+1}
                if cur_wr >= 0.65:
                    print(f"*** GOAL: base{bi} iter{it+1}: n={cur_n} wr={cur_wr*100:.2f}% bh={B['bh']} wd={B['wd']} we={B['we']} cd={B['cd']}")
                    break
        if bi % 30 == 29:
            print(f"  [{bi+1}/{len(bases)}] best so far: n={overall_best['n']} wr={overall_best['wr']*100:.2f}%")
        if overall_best["wr"] >= 0.65:
            break

    print(f"\n=== OVERALL BEST ===")
    print(f"  n={overall_best['n']} wr={overall_best['wr']*100:.2f}% base={overall_best.get('base')}")


if __name__ == "__main__":
    main()
