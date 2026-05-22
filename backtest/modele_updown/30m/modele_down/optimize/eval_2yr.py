"""Evaluate locked config on a 2-year test window (730 days)."""
from __future__ import annotations
import numpy as np
import pandas as pd
import xgboost as xgb
from common import load_dataset, feature_columns, hour_of_day_from_ms
from train import build_hyperparams, _resolve_scale_pos_weight

BAD_HOURS = (1, 8, 10, 13, 14, 17, 21); GOOD_HOURS = (2, 9, 20, 22)
DELTA = 0.030; CD = 90; WD = 0.060; WE = 0.20; VR30_MAX = 1.50; R3_MAX = 0.030
MS_PER_DAY = 86_400_000


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
    df = df.sort_values("timestamp").reset_index(drop=True)
    ts_max = df["timestamp"].max()

    print(f"{'test_days':>10} {'years':>5} {'bets':>5} {'bets/yr':>8} {'wr':>7}")
    for test_days in (365, 545, 730, 912, 1095):
        test_cutoff = ts_max - test_days * MS_PER_DAY
        test_start = int(df["timestamp"].searchsorted(test_cutoff, side="left"))
        train_val_cutoff = test_cutoff - 3600 * MS_PER_DAY
        train_val_start = int(df["timestamp"].searchsorted(train_val_cutoff, side="left"))
        n_train_val = test_start - train_val_start
        val_size = int(n_train_val * 0.10)
        train_end = test_start - val_size
        train = df.iloc[train_val_start:train_end]
        val = df.iloc[train_end:test_start]
        test = df.iloc[test_start:]

        hp = build_hyperparams()
        opt = {"reg_lambda": 10.0, "gamma": 3.0}
        spw = _resolve_scale_pos_weight("auto", train["label"])
        if spw is not None: opt["scale_pos_weight"] = spw
        m = xgb.XGBClassifier(
            n_estimators=1000, max_depth=6, learning_rate=0.03,
            min_child_weight=10, subsample=1.0, colsample_bytree=1.0,
            objective="binary:logistic", eval_metric="logloss",
            early_stopping_rounds=50, random_state=42, n_jobs=-1, **opt,
        )
        m.fit(train[fc], train["label"], eval_set=[(val[fc], val["label"])], verbose=False)
        p_val = m.predict_proba(val[fc])[:, 1]
        p_test = m.predict_proba(test[fc])[:, 1]
        y_test = test["label"].astype(int).to_numpy()

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
        take = m_t & (p_test >= T_eff)
        n = int(take.sum())
        wr = float(y_test[take].mean()) if n > 0 else 0.0
        bets_per_year = n / (test_days / 365)
        marker = " ***GOAL" if wr >= 0.65 and bets_per_year >= 750 else ""
        print(f"  {test_days:>8}  {test_days/365:>5.2f}  {n:>5}  {bets_per_year:>7.1f}  {wr*100:>6.2f}%{marker}")


if __name__ == "__main__":
    main()
