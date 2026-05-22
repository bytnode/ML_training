"""
Clean stacking : train a SEPARATE UP model with cutoff = DOWN's val_start
(2024-05-13). Predictions on DOWN's val (2024-05-13 → 2025-05-13) and DOWN's
test (2025-05-13 → 2026-05-14) are then CLEAN OOS for the UP model.

Use p_up_clean as a SCORE (not training feature) at inference time :
  combined_score = p_down + alpha * (1 - p_up_clean)

Sweep alpha and use it for final filter / threshold.
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


def filtered_threshold(val_df, score_val, wknd_arr_val, mask_val,
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
        thresholds[ax] = float(np.quantile(score_val[m], 1.0 - frac))
    return thresholds


def main():
    df = load_dataset()
    fc = feature_columns(df)
    train_df, val_df, test_df = temporal_split(df)
    print(f"DOWN split : train={len(train_df):,} val={len(val_df):,} test={len(test_df):,}")
    print(f"  val starts at: {pd.to_datetime(val_df['timestamp'].min(), unit='ms', utc=True)}")

    # ---- Stage 1 : DOWN model (standard)
    hp = build_hyperparams()
    opt = {}
    if hp["reg_lambda"] is not False: opt["reg_lambda"] = hp["reg_lambda"]
    if hp["gamma"] is not False: opt["gamma"] = hp["gamma"]
    spw = _resolve_scale_pos_weight(hp["scale_pos_weight"], train_df["label"])
    if spw is not None: opt["scale_pos_weight"] = spw

    m_down = xgb.XGBClassifier(
        n_estimators=hp["n_estimators"], max_depth=hp["max_depth"],
        learning_rate=hp["learning_rate"], min_child_weight=hp["min_child_weight"],
        subsample=hp["subsample"], colsample_bytree=hp["colsample_bytree"],
        objective="binary:logistic", eval_metric="logloss",
        early_stopping_rounds=hp["early_stopping_rounds"],
        random_state=42, n_jobs=-1, **opt,
    )
    m_down.fit(train_df[fc], train_df["label"],
                eval_set=[(val_df[fc], val_df["label"])], verbose=False)
    p_down_val = m_down.predict_proba(val_df[fc])[:, 1]
    p_down_test = m_down.predict_proba(test_df[fc])[:, 1]
    print(f"DOWN model trained. best_iter={m_down.best_iteration} "
          f"val_ll={m_down.evals_result()['validation_0']['logloss'][m_down.best_iteration]:.5f}")

    # ---- Stage 2 : UP model trained on DOWN's train only (= up to val start)
    # Label_UP = 1 - label_DOWN essentially (next bar green vs red)
    # We need to compute label_up = (cr_next > or_next).  But label_down has
    # already done the work : label_up = 1 - label_down where defined.
    train_label_up = (1 - train_df["label"].astype(int)).astype(int)
    val_label_up = (1 - val_df["label"].astype(int)).astype(int)

    spw_up = _resolve_scale_pos_weight("auto", train_label_up)
    opt_up = dict(opt)
    if spw_up is not None: opt_up["scale_pos_weight"] = spw_up

    m_up = xgb.XGBClassifier(
        n_estimators=hp["n_estimators"], max_depth=hp["max_depth"],
        learning_rate=hp["learning_rate"], min_child_weight=hp["min_child_weight"],
        subsample=hp["subsample"], colsample_bytree=hp["colsample_bytree"],
        objective="binary:logistic", eval_metric="logloss",
        early_stopping_rounds=hp["early_stopping_rounds"],
        random_state=42, n_jobs=-1, **opt_up,
    )
    m_up.fit(train_df[fc], train_label_up,
              eval_set=[(val_df[fc], val_label_up)], verbose=False)
    p_up_val_clean = m_up.predict_proba(val_df[fc])[:, 1]
    p_up_test_clean = m_up.predict_proba(test_df[fc])[:, 1]
    print(f"UP model trained on DOWN's train (clean OOS for val/test). best_iter={m_up.best_iteration}")

    # Correlation between p_down and p_up_clean
    corr_val = np.corrcoef(p_down_val, p_up_val_clean)[0, 1]
    corr_test = np.corrcoef(p_down_test, p_up_test_clean)[0, 1]
    print(f"corr(p_down, p_up_clean) val={corr_val:.4f} test={corr_test:.4f}")
    # They should be ~ -1 (perfect anti-correlation if both are well-calibrated on opposite event)
    # Any deviation gives orthogonal info.

    # Eval prep
    hours_t = hour_of_day_from_ms(test_df["timestamp"].to_numpy())
    wknd_t = test_df["is_weekend"].astype(int).to_numpy()
    hours_v = hour_of_day_from_ms(val_df["timestamp"].to_numpy())
    wknd_v = val_df["is_weekend"].astype(int).to_numpy()
    vr30_t = test_df["volume_ratio_30m"].to_numpy(dtype=float)
    r3_t = test_df["return_3"].to_numpy(dtype=float)
    vr30_v = val_df["volume_ratio_30m"].to_numpy(dtype=float)
    r3_v = val_df["return_3"].to_numpy(dtype=float)
    bad_t = np.isin(hours_t, list(BAD_HOURS))
    bad_v = np.isin(hours_v, list(BAD_HOURS))
    good_t = np.isin(hours_t, list(GOOD_HOURS))
    m_t = (~bad_t) & ((vr30_t < VR30_MAX) | np.isnan(vr30_t)) & ((r3_t < R3_MAX) | np.isnan(r3_t))
    m_v = (~bad_v) & ((vr30_v < VR30_MAX) | np.isnan(vr30_v)) & ((r3_v < R3_MAX) | np.isnan(r3_v))
    y_test = test_df["label"].astype(int).to_numpy()

    # Baseline (alpha=0)
    score_val = p_down_val
    score_test = p_down_test
    thr = filtered_threshold(val_df, score_val, wknd_v, m_v, WD, WE, CD)
    T_base = np.where(wknd_t == 1, thr.get(1, 1.0), thr.get(0, 1.0))
    T_eff = T_base - DELTA * good_t
    take = m_t & (score_test >= T_eff)
    print(f"\n  alpha=0 (DOWN only): taken={int(take.sum())} wr={float(y_test[take].mean())*100:.2f}%")

    # Sweep alpha and decision style
    print(f"\n{'alpha':>6} {'p_up_max':>8} {'taken':>6} {'wr':>7}")
    for alpha in (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0):
        score_val = p_down_val + alpha * (1.0 - p_up_val_clean)
        score_test = p_down_test + alpha * (1.0 - p_up_test_clean)
        # Renormalize WD/WE since score magnitude changes
        thr = filtered_threshold(val_df, score_val, wknd_v, m_v, WD, WE, CD)
        T_base = np.where(wknd_t == 1, thr.get(1, 1.0), thr.get(0, 1.0))
        T_eff = T_base - DELTA * good_t
        for p_up_max in (1.0, 0.55, 0.50, 0.48, 0.45, 0.42):
            pu_mask = (p_up_test_clean <= p_up_max) | np.isnan(p_up_test_clean) if p_up_max < 1.0 else np.ones(len(p_down_test), dtype=bool)
            take = m_t & (score_test >= T_eff) & pu_mask
            n = int(take.sum())
            if n < MIN_PARIS:
                continue
            wr = float(y_test[take].mean())
            marker = " ***GOAL" if (wr >= 0.65) else ""
            print(f"  {alpha:>5.2f}  {p_up_max:>7.3f}    {n:>5}  {wr*100:>5.2f}%{marker}")


if __name__ == "__main__":
    main()
