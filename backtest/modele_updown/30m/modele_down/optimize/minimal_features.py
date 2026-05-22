"""
Try a MINIMAL feature subset (just momentum + price action).
Hypothesis: removing noisy features may sharpen the model's confidence at top.
"""
from __future__ import annotations

import itertools
import numpy as np
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
    full_fc = feature_columns(df)
    train_df, val_df, test_df = temporal_split(df)

    # Various feature subsets to try
    SUBSETS = {
        "momentum_only": [
            "close_ret", "return_3", "return_96", "roc_5", "roc_10", "roc_20",
            "close_ret_30m", "close_ret_4h",
        ],
        "momentum_oscillators": [
            "close_ret", "return_3", "roc_5", "roc_10", "rsi_14", "stoch_k_14",
            "macd", "macd_hist", "mfi_norm",
        ],
        "no_4h_1d": [c for c in full_fc if "_4h" not in c and "_1d" not in c],
        "no_30m": [c for c in full_fc if "_30m" not in c],
        "no_oscillators": [c for c in full_fc if c not in [
            "rsi_14", "stoch_k_14", "mfi_norm", "macd", "macd_signal",
            "macd_hist", "bb_position", "bb_width",
        ]],
        "no_wicks_no_body": [c for c in full_fc if c not in [
            "upper_wick", "lower_wick", "body_ratio_close", "body_ratio_avg",
            "body_ratio_close_30m", "body_ratio_avg_30m", "body_ratio_avg_4h",
            "body_ratio_close_1d", "body_ratio_avg_1d",
        ]],
        "top10_importance": [
            "return_3", "close_ret_30m", "close_ret", "roc_5", "stoch_k_14",
            "dist_low_20", "dist_high_20", "close_ret_4h", "atr_pct_1d",
            "hour_cos",
        ],
        "top20": None,  # filled below
        "full": full_fc,
    }
    # Get top 20 by importance from full model
    hp = build_hyperparams()
    opt = {}
    if hp["reg_lambda"] is not False: opt["reg_lambda"] = hp["reg_lambda"]
    if hp["gamma"] is not False: opt["gamma"] = hp["gamma"]
    spw = _resolve_scale_pos_weight(hp["scale_pos_weight"], train_df["label"])
    if spw is not None: opt["scale_pos_weight"] = spw

    full_m = xgb.XGBClassifier(
        n_estimators=hp["n_estimators"], max_depth=hp["max_depth"],
        learning_rate=hp["learning_rate"], min_child_weight=hp["min_child_weight"],
        subsample=hp["subsample"], colsample_bytree=hp["colsample_bytree"],
        objective="binary:logistic", eval_metric="logloss",
        early_stopping_rounds=hp["early_stopping_rounds"],
        random_state=42, n_jobs=-1, **opt,
    )
    full_m.fit(train_df[full_fc], train_df["label"],
                eval_set=[(val_df[full_fc], val_df["label"])], verbose=False)
    importance = sorted(zip(full_fc, full_m.feature_importances_), key=lambda x: -x[1])
    SUBSETS["top20"] = [c for c, _ in importance[:20]]
    SUBSETS["top30"] = [c for c, _ in importance[:30]]
    SUBSETS["top40"] = [c for c, _ in importance[:40]]
    SUBSETS["bottom_dropped"] = [c for c, _ in importance[:50]]  # keep top 50, drop low importance

    print(f"\n{'subset':>20} {'n_feats':>8} {'taken':>6} {'wr':>7}")
    for name, sub in SUBSETS.items():
        if sub is None:
            continue
        sub = [c for c in sub if c in train_df.columns]
        if len(sub) < 3:
            continue
        m = xgb.XGBClassifier(
            n_estimators=hp["n_estimators"], max_depth=hp["max_depth"],
            learning_rate=hp["learning_rate"], min_child_weight=hp["min_child_weight"],
            subsample=hp["subsample"], colsample_bytree=hp["colsample_bytree"],
            objective="binary:logistic", eval_metric="logloss",
            early_stopping_rounds=hp["early_stopping_rounds"],
            random_state=42, n_jobs=-1, **opt,
        )
        m.fit(train_df[sub], train_df["label"],
               eval_set=[(val_df[sub], val_df["label"])], verbose=False)
        p_val = m.predict_proba(val_df[sub])[:, 1]
        p_test = m.predict_proba(test_df[sub])[:, 1]
        y_test = test_df["label"].astype(int).to_numpy()
        n, wr = eval_p(p_val, p_test, val_df, test_df, y_test)
        marker = " ***GOAL" if wr >= 0.65 and n >= 750 else ""
        print(f"  {name:>18}  {len(sub):>7}  {n:>5}  {wr*100:>5.2f}%{marker}")


if __name__ == "__main__":
    main()
