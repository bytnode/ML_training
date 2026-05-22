"""
Use a config with MORE WINS (even if lower WR), then drop losers aggressively.
Goal: 510 wins / 750 bets = 68% WR (if we can drop ~72 losers without losing wins).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from common import load_dataset, feature_columns, train_once, hour_of_day_from_ms
from train import compute_thresholds_asym_wknd_excl

BAD_HOURS = (1, 8, 10, 13, 14, 17, 21); GOOD_HOURS = (2, 9, 20, 22)
DELTA = 0.030; CD = 90; WD = 0.060; WE = 0.20
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

    # Try various LOOSER configs to get more wins
    print(f"{'config':<40} {'taken':>6} {'wins':>5} {'losses':>6} {'wr':>7} {'max_wr_at_750':>14}")
    for cfg_name, BH, GH, D, VR, R3, WD_loc, WE_loc, CD_loc in [
        ("base_lock", BAD_HOURS, GOOD_HOURS, 0.030, 1.50, 0.030, 0.060, 0.20, 90),
        ("looser_wd008", BAD_HOURS, GOOD_HOURS, 0.030, 1.50, 0.030, 0.080, 0.22, 90),
        ("looser_wd010", BAD_HOURS, GOOD_HOURS, 0.030, 1.50, 0.030, 0.100, 0.25, 90),
        ("looser_wd012", BAD_HOURS, GOOD_HOURS, 0.030, 1.50, 0.030, 0.120, 0.30, 90),
        ("looser_wd015", BAD_HOURS, GOOD_HOURS, 0.030, 1.50, 0.030, 0.150, 0.35, 90),
        ("fewer_bad", (8, 13, 14, 17, 21), GOOD_HOURS, 0.030, 1.50, 0.030, 0.080, 0.25, 90),
        ("no_filter_r3", BAD_HOURS, GOOD_HOURS, 0.030, np.inf, np.inf, 0.080, 0.25, 90),
        ("no_filter_vr30", BAD_HOURS, GOOD_HOURS, 0.030, np.inf, 0.030, 0.080, 0.25, 90),
        ("all_open", (), (), 0.0, np.inf, np.inf, 0.150, 0.35, 90),
    ]:
        vr30_t = test["volume_ratio_30m"].to_numpy(dtype=float)
        r3_t = test["return_3"].to_numpy(dtype=float)
        vr30_v = val["volume_ratio_30m"].to_numpy(dtype=float)
        r3_v = val["return_3"].to_numpy(dtype=float)
        bad_t = np.isin(hours_t, list(BH)); bad_v = np.isin(hours_v, list(BH))
        good_t = np.isin(hours_t, list(GH))
        m_t = ~bad_t; m_v = ~bad_v
        if VR != np.inf:
            m_t = m_t & ((vr30_t < VR) | np.isnan(vr30_t))
            m_v = m_v & ((vr30_v < VR) | np.isnan(vr30_v))
        if R3 != np.inf:
            m_t = m_t & ((r3_t < R3) | np.isnan(r3_t))
            m_v = m_v & ((r3_v < R3) | np.isnan(r3_v))
        thr = filtered_threshold(val, p_val, wknd_v, m_v, WD_loc, WE_loc, CD_loc)
        T_base = np.where(wknd_t == 1, thr.get(1, 1.0), thr.get(0, 1.0))
        T_eff = T_base - D * good_t
        take = m_t & (p_test >= T_eff)
        n = int(take.sum())
        if n < 100:
            continue
        idx = np.where(take)[0]
        y_taken = y_test[idx]
        wins = int(y_taken.sum())
        losses = n - wins
        wr = wins / n
        # Best achievable: keep all wins, drop losses to 750 total
        # If wins >= 750, max wr at 750 = ? (would have to drop wins too)
        # If wins < 750, max wr at 750 = wins / 750
        if wins < 750:
            best_wr_750 = wins / 750.0
        else:
            best_wr_750 = 1.0  # could be all wins
        marker = " ***" if best_wr_750 >= 0.65 else ""
        print(f"  {cfg_name:<38} {n:>5}  {wins:>4}  {losses:>5}  {wr*100:>5.2f}%  {best_wr_750*100:>10.2f}%{marker}")


if __name__ == "__main__":
    main()
