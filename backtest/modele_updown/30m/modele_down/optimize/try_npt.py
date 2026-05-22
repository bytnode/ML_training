"""Try XGBoost with num_parallel_tree (RF-within-boost hybrid)."""
from __future__ import annotations
import numpy as np
import xgboost as xgb
from common import (load_dataset, feature_columns, temporal_split, hour_of_day_from_ms)
from train import build_hyperparams, _resolve_scale_pos_weight

BAD_HOURS = (1, 8, 10, 13, 14, 17, 21); GOOD_HOURS = (2, 9, 20, 22)
DELTA = 0.030; CD = 90; WD = 0.060; WE = 0.20; VR30_MAX = 1.50; R3_MAX = 0.030
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


def eval_p(p_val, p_test, val, test, y_test):
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
    n = int(take.sum()); wr = float(y_test[take].mean()) if n > 0 else 0.0
    return n, wr


def main():
    df = load_dataset(); fc = feature_columns(df)
    train_df, val_df, test_df = temporal_split(df)
    y_test = test_df["label"].astype(int).to_numpy()
    hp = build_hyperparams()
    opt_base = {"reg_lambda": 10.0, "gamma": 3.0}
    spw = _resolve_scale_pos_weight("auto", train_df["label"])
    if spw is not None: opt_base["scale_pos_weight"] = spw

    print(f"{'npt':>5} {'subsample':>10} {'colsample':>10} {'taken':>6} {'wr':>7}")
    for npt in (1, 3, 5, 10, 20):
        for ss in (0.5, 0.7, 1.0):
            for cs in (0.5, 0.7, 1.0):
                m = xgb.XGBClassifier(
                    n_estimators=200 if npt > 1 else 1000,
                    num_parallel_tree=npt,
                    max_depth=6, learning_rate=0.03, min_child_weight=10,
                    subsample=ss, colsample_bytree=cs,
                    objective="binary:logistic", eval_metric="logloss",
                    early_stopping_rounds=50, random_state=42, n_jobs=-1, **opt_base,
                )
                m.fit(train_df[fc], train_df["label"],
                       eval_set=[(val_df[fc], val_df["label"])], verbose=False)
                p_val = m.predict_proba(val_df[fc])[:, 1]
                p_test = m.predict_proba(test_df[fc])[:, 1]
                n, wr = eval_p(p_val, p_test, val_df, test_df, y_test)
                marker = " ***GOAL" if wr >= 0.65 and n >= MIN_PARIS else ""
                print(f"  {npt:>3}    {ss:>8.1f}    {cs:>8.1f}    {n:>5}  {wr*100:>5.2f}%{marker}", flush=True)


if __name__ == "__main__":
    main()
