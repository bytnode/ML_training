"""Analyze the wins vs losses in the current best 63.79% config."""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import (
    load_dataset, feature_columns, train_once, hour_of_day_from_ms,
)


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
    fc = feature_columns(df)
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

    bad_t = np.isin(hours_t, list(BAD_HOURS))
    bad_v = np.isin(hours_v, list(BAD_HOURS))
    good_t = np.isin(hours_t, list(GOOD_HOURS))

    m_t = ~bad_t & ((vr30_t < VR30_MAX) | np.isnan(vr30_t)) & ((r3_t < R3_MAX) | np.isnan(r3_t))
    m_v = ~bad_v & ((vr30_v < VR30_MAX) | np.isnan(vr30_v)) & ((r3_v < R3_MAX) | np.isnan(r3_v))

    thr = filtered_threshold(val, p_val, wknd_v, m_v, WD, WE, CD)
    T_base = np.where(wknd_t == 1, thr.get(1, 1.0), thr.get(0, 1.0))
    T_eff = T_base - DELTA * good_t
    take = m_t & (p_test >= T_eff)
    n = int(take.sum())
    wr = float(y_test[take].mean())
    print(f"Current best: taken={n} wr={wr*100:.2f}% T_wd={thr.get(0):.4f} T_we={thr.get(1):.4f}")

    # Index in test of taken bets
    idx_taken = np.where(take)[0]
    test_taken = test.iloc[idx_taken].reset_index(drop=True)
    test_taken["p_down"] = p_test[idx_taken]
    test_taken["correct"] = y_test[idx_taken]

    # Wins vs losses feature comparison
    print(f"\n{'feature':<28} {'win_mean':>10} {'loss_mean':>10} {'diff':>10} {'t':>7}")
    print("-" * 70)
    stats = []
    for col in fc:
        wv = test_taken.loc[test_taken["correct"] == 1, col].dropna()
        lv = test_taken.loc[test_taken["correct"] == 0, col].dropna()
        if len(wv) < 10 or len(lv) < 10:
            continue
        wm = float(wv.mean()); lm = float(lv.mean())
        d = wm - lm
        ws = float(wv.std()); ls = float(lv.std())
        pooled = np.sqrt(ws**2/len(wv) + ls**2/len(lv)) + 1e-9
        t = d / pooled
        stats.append((col, wm, lm, d, t))
    stats.sort(key=lambda r: -abs(r[4]))
    for s in stats[:15]:
        c, w, l, d, t = s
        print(f"  {c:<26} {w:>10.5f} {l:>10.5f} {d:>+10.5f} {t:>+7.2f}")

    # Hour split among taken
    print(f"\n{'hour':>5} {'n':>5} {'wr':>7}")
    for h, g in test_taken.groupby(((test_taken["timestamp"].astype("int64") // 3600000) % 24).astype(int)):
        if len(g) >= 15:
            mk = " bad" if g["correct"].mean() < 0.55 else (" good" if g["correct"].mean() > 0.70 else "")
            print(f"  {h:>3}  {len(g):>5}  {g['correct'].mean()*100:>5.2f}%{mk}")

    # vol_state
    print("\nvol_state split:")
    for v, g in test_taken.groupby(test_taken["volatility_state"]):
        print(f"  v={v}: n={len(g)} wr={g['correct'].mean()*100:.2f}%")
    print("weekend split:")
    for w, g in test_taken.groupby(test_taken["is_weekend"]):
        print(f"  w={w}: n={len(g)} wr={g['correct'].mean()*100:.2f}%")


if __name__ == "__main__":
    main()
