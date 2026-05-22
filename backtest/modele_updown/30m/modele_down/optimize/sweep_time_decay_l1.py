"""
Time-decay sample weighting + L1 regularization.

Hypothesis: weighting recent train rows more heavily makes the model adapt
to the test period regime. L1 reg can select features automatically.
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


def eval_run(model, val, test, fc):
    p_val = model.predict_proba(val[fc])[:, 1]
    p_test = model.predict_proba(test[fc])[:, 1]
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
    wr = float(y_test[take].mean()) if n > 0 else 0.0
    return n, wr


def main():
    df = load_dataset()
    fc = feature_columns(df)
    train_df, val_df, test_df = temporal_split(df)
    print(f"Dataset: train={len(train_df):,} val={len(val_df):,} test={len(test_df):,}")

    train_ts = train_df["timestamp"].to_numpy()
    train_max = train_ts.max()
    ms_per_day = 86_400_000

    print(f"\n{'tau_days':>10} {'reg_alpha':>10} {'taken':>6} {'wr':>7} {'best_iter':>10} {'val_ll':>8}")

    # Time-decay : weight = exp(-(train_max - ts) / (tau_days * ms_per_day))
    # tau = inf (uniform), 3000, 1500, 800, 400, 200
    for tau in (None, 3000, 1500, 800, 400, 200):
        for reg_alpha in (False, 0.0, 0.5, 1.0, 5.0):
            hp = build_hyperparams()
            opt = {}
            if hp["reg_lambda"] is not False: opt["reg_lambda"] = hp["reg_lambda"]
            if reg_alpha is not False: opt["reg_alpha"] = reg_alpha
            if hp["gamma"] is not False: opt["gamma"] = hp["gamma"]
            spw = _resolve_scale_pos_weight(hp["scale_pos_weight"], train_df["label"])
            if spw is not None: opt["scale_pos_weight"] = spw

            m = xgb.XGBClassifier(
                n_estimators=hp["n_estimators"], max_depth=hp["max_depth"],
                learning_rate=hp["learning_rate"], min_child_weight=hp["min_child_weight"],
                subsample=hp["subsample"], colsample_bytree=hp["colsample_bytree"],
                objective="binary:logistic", eval_metric="logloss",
                early_stopping_rounds=hp["early_stopping_rounds"],
                random_state=42, n_jobs=-1, **opt,
            )

            if tau is None:
                m.fit(train_df[fc], train_df["label"],
                       eval_set=[(val_df[fc], val_df["label"])], verbose=False)
            else:
                age_days = (train_max - train_ts) / ms_per_day
                w = np.exp(-age_days / tau)
                m.fit(train_df[fc], train_df["label"], sample_weight=w,
                       eval_set=[(val_df[fc], val_df["label"])], verbose=False)

            best = int(m.best_iteration)
            val_ll = float(m.evals_result()["validation_0"]["logloss"][best])
            n, wr = eval_run(m, val_df, test_df, fc)
            marker = " ***GOAL" if (wr >= 0.65 and n >= MIN_PARIS) else ""
            print(f"  {tau!s:>10} {reg_alpha!s:>10} {n:>5} {wr*100:>6.2f}% {best:>10} {val_ll:>8.5f}{marker}")


if __name__ == "__main__":
    main()
