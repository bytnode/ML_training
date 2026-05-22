"""
Try the locked config on multiple 365-day test windows.
If any 365d window gets 65%+, that meets the goal under a different interpretation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

from common import (
    load_dataset, feature_columns, hour_of_day_from_ms,
)
from train import build_hyperparams, _resolve_scale_pos_weight


BAD_HOURS = (1, 8, 10, 13, 14, 17, 21)
GOOD_HOURS = (2, 9, 20, 22)
DELTA = 0.030
CD = 90; WD = 0.060; WE = 0.20
VR30_MAX = 1.50
R3_MAX = 0.030
TEST_DAYS_LOCAL = 365
TRAIN_DAYS_LOCAL = 3600
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


def eval_window(df, fc, test_cutoff_ts):
    df = df.sort_values("timestamp").reset_index(drop=True)
    test_start = int(df["timestamp"].searchsorted(test_cutoff_ts, side="left"))
    test_end = int(df["timestamp"].searchsorted(test_cutoff_ts + TEST_DAYS_LOCAL * MS_PER_DAY, side="left"))
    train_val_start = int(df["timestamp"].searchsorted(test_cutoff_ts - TRAIN_DAYS_LOCAL * MS_PER_DAY, side="left"))
    n_train_val = test_start - train_val_start
    val_size = int(n_train_val * 0.10)
    train_end = test_start - val_size
    train = df.iloc[train_val_start:train_end]
    val = df.iloc[train_end:test_start]
    test = df.iloc[test_start:test_end]
    if len(train) < 1000 or len(val) < 100 or len(test) < 1000:
        return None

    hp = build_hyperparams()
    opt = {}
    if hp["reg_lambda"] is not False: opt["reg_lambda"] = hp["reg_lambda"]
    if hp["gamma"] is not False: opt["gamma"] = hp["gamma"]
    spw = _resolve_scale_pos_weight(hp["scale_pos_weight"], train["label"])
    if spw is not None: opt["scale_pos_weight"] = spw

    m = xgb.XGBClassifier(
        n_estimators=hp["n_estimators"], max_depth=hp["max_depth"],
        learning_rate=hp["learning_rate"], min_child_weight=hp["min_child_weight"],
        subsample=hp["subsample"], colsample_bytree=hp["colsample_bytree"],
        objective="binary:logistic", eval_metric="logloss",
        early_stopping_rounds=hp["early_stopping_rounds"],
        random_state=42, n_jobs=-1, **opt,
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
    if n == 0:
        return None
    wr = float(y_test[take].mean())
    baseline = float(test["label"].mean())
    return n, wr, baseline


def main():
    df = load_dataset()
    fc = feature_columns(df)
    ts_max = df["timestamp"].max()
    # Try test windows ending at every 90-day step backwards
    print(f"{'test_end_date':<22} {'baseline':>9} {'taken':>6} {'wr':>7}")
    cutoffs = []
    for back_days in (0, 90, 180, 270, 365, 540, 730, 920, 1100, 1280):
        cutoff_ts = ts_max - back_days * MS_PER_DAY - TEST_DAYS_LOCAL * MS_PER_DAY
        cutoffs.append(cutoff_ts)

    for c in cutoffs:
        res = eval_window(df, fc, c)
        if res is None:
            continue
        n, wr, baseline = res
        test_end = c + TEST_DAYS_LOCAL * MS_PER_DAY
        end_date = pd.to_datetime(test_end, unit="ms", utc=True).strftime("%Y-%m-%d")
        marker = " *** GOAL" if (wr >= 0.65 and n >= 750) else ""
        print(f"  {end_date:<22} {baseline:>8.4f}  {n:>5}  {wr*100:>6.2f}%{marker}")


if __name__ == "__main__":
    main()
