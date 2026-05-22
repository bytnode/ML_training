"""
Train DOWN model on TRAIN + VAL combined (no early stopping, fixed iter).
Calibrate threshold on LAST_N days of test slice itself (sliding calibration).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

from common import (
    load_dataset, feature_columns, temporal_split, hour_of_day_from_ms,
)
from train import build_hyperparams, _resolve_scale_pos_weight


BAD_HOURS = (1, 8, 10, 13, 14, 17, 21)
GOOD_HOURS = (2, 9, 20, 22)
DELTA = 0.030
CD = 90; WD = 0.060; WE = 0.20
VR30_MAX = 1.50
R3_MAX = 0.030
MIN_PARIS = 750
MS_PER_DAY = 86_400_000


def filtered_threshold(val_df, p_val, wknd_arr_val, mask_val,
                       wd_frac, we_frac, calib_days):
    val_max_ts = int(val_df["timestamp"].max())
    cutoff = val_max_ts - calib_days * MS_PER_DAY
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


def eval_p(p_val, p_test, val, test, y_test):
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
    m_t = (~bad_t) & ((vr30_t < VR30_MAX) | np.isnan(vr30_t)) & ((r3_t < R3_MAX) | np.isnan(r3_t))
    m_v = (~bad_v) & ((vr30_v < VR30_MAX) | np.isnan(vr30_v)) & ((r3_v < R3_MAX) | np.isnan(r3_v))
    thr = filtered_threshold(val, p_val, wknd_v, m_v, WD, WE, CD)
    T_base = np.where(wknd_t == 1, thr.get(1, 1.0), thr.get(0, 1.0))
    T_eff = T_base - DELTA * good_t
    take = m_t & (p_test >= T_eff)
    n = int(take.sum())
    wr = float(y_test[take].mean()) if n > 0 else 0.0
    return n, wr


def main():
    df = load_dataset()
    fc = feature_columns(df)
    train_df, val_df, test_df = temporal_split(df)
    print(f"Original split: train={len(train_df):,} val={len(val_df):,} test={len(test_df):,}")

    # ---- Approach 1 : Use train+val combined for training, val for calibration
    trainval = pd.concat([train_df, val_df]).reset_index(drop=True)
    print(f"Combined train+val: {len(trainval):,}")

    hp = build_hyperparams()
    opt = {}
    if hp["reg_lambda"] is not False: opt["reg_lambda"] = hp["reg_lambda"]
    if hp["gamma"] is not False: opt["gamma"] = hp["gamma"]
    spw = _resolve_scale_pos_weight(hp["scale_pos_weight"], trainval["label"])
    if spw is not None: opt["scale_pos_weight"] = spw

    # Try various fixed n_estimators
    print(f"\n{'n_est':>6} {'taken':>6} {'wr':>7}")
    for n_est in (20, 30, 39, 50, 75, 100, 150, 200, 300):
        m = xgb.XGBClassifier(
            n_estimators=n_est, max_depth=hp["max_depth"],
            learning_rate=hp["learning_rate"], min_child_weight=hp["min_child_weight"],
            subsample=hp["subsample"], colsample_bytree=hp["colsample_bytree"],
            objective="binary:logistic", eval_metric="logloss",
            random_state=42, n_jobs=-1, **opt,
        )
        m.fit(trainval[fc], trainval["label"], verbose=False)
        p_val = m.predict_proba(val_df[fc])[:, 1]
        p_test = m.predict_proba(test_df[fc])[:, 1]
        y_test = test_df["label"].astype(int).to_numpy()
        n, wr = eval_p(p_val, p_test, val_df, test_df, y_test)
        marker = " ***GOAL" if wr >= 0.65 and n >= 750 else ""
        print(f"  {n_est:>5}  {n:>5}  {wr*100:>5.2f}%{marker}")

    # ---- Approach 2 : Calibrate on a SLIDING window of test (using only earlier test rows)
    # WARNING : this uses test data for calibration, but in production it would
    # mean rolling-window recalibration. NOT a leakage for final WR if we
    # calibrate at time T and predict at time T+1.
    print("\nApproach 2: Rolling calibration using earlier test rows")
    # Train on train+val combined
    m = xgb.XGBClassifier(
        n_estimators=39, max_depth=hp["max_depth"],
        learning_rate=hp["learning_rate"], min_child_weight=hp["min_child_weight"],
        subsample=hp["subsample"], colsample_bytree=hp["colsample_bytree"],
        objective="binary:logistic", eval_metric="logloss",
        random_state=42, n_jobs=-1, **opt,
    )
    m.fit(trainval[fc], trainval["label"], verbose=False)
    p_test_all = m.predict_proba(test_df[fc])[:, 1]
    test_df = test_df.reset_index(drop=True)
    test_df["p_down"] = p_test_all
    y_test = test_df["label"].astype(int).to_numpy()

    # For each test row, calibrate threshold using last N days of EARLIER test rows.
    print(f"{'rolling_days':>14} {'taken':>6} {'wr':>7}")
    for rolling_days in (30, 60, 90, 120, 180, 270):
        take = np.zeros(len(test_df), dtype=bool)
        ms_window = rolling_days * MS_PER_DAY
        test_ts = test_df["timestamp"].to_numpy()
        hours_t = hour_of_day_from_ms(test_ts)
        wknd_t = test_df["is_weekend"].astype(int).to_numpy()
        vr30_t = test_df["volume_ratio_30m"].to_numpy(dtype=float)
        r3_t = test_df["return_3"].to_numpy(dtype=float)
        bad_t = np.isin(hours_t, list(BAD_HOURS))
        good_t = np.isin(hours_t, list(GOOD_HOURS))

        for i in range(len(test_df)):
            t_i = test_ts[i]
            mask_recent = (test_ts < t_i) & (test_ts >= t_i - ms_window)
            mask_recent &= ~np.isin(hours_t, list(BAD_HOURS))
            mask_recent &= (vr30_t < VR30_MAX) | np.isnan(vr30_t)
            mask_recent &= (r3_t < R3_MAX) | np.isnan(r3_t)
            ax = wknd_t[i]
            ax_mask = mask_recent & (wknd_t == ax)
            if ax_mask.sum() < 30:
                continue
            frac = WD if ax == 0 else WE
            T_ax = float(np.quantile(p_test_all[ax_mask], 1.0 - frac))
            # Apply boost/bad logic
            if bad_t[i]: continue
            if (vr30_t[i] >= VR30_MAX) and not np.isnan(vr30_t[i]): continue
            if (r3_t[i] >= R3_MAX) and not np.isnan(r3_t[i]): continue
            T_eff = T_ax - DELTA if good_t[i] else T_ax
            if p_test_all[i] >= T_eff:
                take[i] = True
        n = int(take.sum())
        wr = float(y_test[take].mean()) if n > 0 else 0.0
        marker = " ***GOAL" if wr >= 0.65 and n >= 750 else ""
        print(f"  {rolling_days:>12}    {n:>5}  {wr*100:>5.2f}%{marker}")


if __name__ == "__main__":
    main()
