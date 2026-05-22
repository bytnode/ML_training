"""
Try LightGBM (different gradient boosting implementation) vs XGBoost.

Different splitting algo (leaf-wise vs level-wise), different handling
of categorical features, may extract different signal.

Compare LGB alone and LGB ensembled with XGB.
"""
from __future__ import annotations

import numpy as np
import lightgbm as lgb
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

    # --- XGBoost baseline (current locked)
    hp = build_hyperparams()
    opt = {}
    if hp["reg_lambda"] is not False: opt["reg_lambda"] = hp["reg_lambda"]
    if hp["gamma"] is not False: opt["gamma"] = hp["gamma"]
    spw = _resolve_scale_pos_weight(hp["scale_pos_weight"], train_df["label"])
    if spw is not None: opt["scale_pos_weight"] = spw
    xgb_m = xgb.XGBClassifier(
        n_estimators=hp["n_estimators"], max_depth=hp["max_depth"],
        learning_rate=hp["learning_rate"], min_child_weight=hp["min_child_weight"],
        subsample=hp["subsample"], colsample_bytree=hp["colsample_bytree"],
        objective="binary:logistic", eval_metric="logloss",
        early_stopping_rounds=hp["early_stopping_rounds"],
        random_state=42, n_jobs=-1, **opt,
    )
    xgb_m.fit(train_df[fc], train_df["label"],
               eval_set=[(val_df[fc], val_df["label"])], verbose=False)
    p_val_xgb = xgb_m.predict_proba(val_df[fc])[:, 1]
    p_test_xgb = xgb_m.predict_proba(test_df[fc])[:, 1]
    y_test = test_df["label"].astype(int).to_numpy()
    n, wr = eval_p(p_val_xgb, p_test_xgb, val_df, test_df, y_test)
    print(f"XGB baseline: n={n} wr={wr*100:.2f}%")

    # --- LightGBM ---
    print("\nTrying LightGBM variants:")
    print(f"{'config':>30} {'taken':>6} {'wr':>7}")
    LGB_CONFIGS = [
        {"name": "lgb_default", "num_leaves": 31, "max_depth": -1, "learning_rate": 0.03,
         "min_child_samples": 20, "reg_lambda": 10.0, "reg_alpha": 0.0, "n_estimators": 1000},
        {"name": "lgb_d6", "num_leaves": 63, "max_depth": 6, "learning_rate": 0.03,
         "min_child_samples": 20, "reg_lambda": 10.0, "n_estimators": 1000},
        {"name": "lgb_shallow", "num_leaves": 15, "max_depth": 4, "learning_rate": 0.05,
         "min_child_samples": 30, "reg_lambda": 5.0, "n_estimators": 1000},
        {"name": "lgb_deep", "num_leaves": 127, "max_depth": 8, "learning_rate": 0.02,
         "min_child_samples": 10, "reg_lambda": 1.0, "n_estimators": 2000},
        {"name": "lgb_dart", "num_leaves": 31, "max_depth": -1, "learning_rate": 0.05,
         "boosting_type": "dart", "drop_rate": 0.1, "n_estimators": 500},
        {"name": "lgb_goss", "num_leaves": 31, "max_depth": -1, "learning_rate": 0.03,
         "boosting_type": "goss", "n_estimators": 1000},
    ]
    lgb_runs = []
    for cfg in LGB_CONFIGS:
        cfg_copy = dict(cfg)
        name = cfg_copy.pop("name")
        try:
            m = lgb.LGBMClassifier(
                objective="binary", metric="binary_logloss", verbose=-1,
                n_jobs=-1, random_state=42, **cfg_copy,
            )
            m.fit(train_df[fc], train_df["label"],
                   eval_set=[(val_df[fc], val_df["label"])],
                   callbacks=[lgb.early_stopping(50)])
            p_val_lgb = m.predict_proba(val_df[fc])[:, 1]
            p_test_lgb = m.predict_proba(test_df[fc])[:, 1]
            n, wr = eval_p(p_val_lgb, p_test_lgb, val_df, test_df, y_test)
            lgb_runs.append({"p_val": p_val_lgb, "p_test": p_test_lgb, "name": name, "wr": wr, "n": n})
            print(f"  {name:>28}  {n:>5}  {wr*100:>5.2f}%")
        except Exception as e:
            print(f"  {name:>28} ERROR: {e}")

    # --- Ensemble : average XGB + best LGB ---
    print("\nEnsembles (XGB + LGB blends):")
    for lgb_r in lgb_runs:
        for alpha in (0.3, 0.5, 0.7):
            p_val_avg = alpha * p_val_xgb + (1 - alpha) * lgb_r["p_val"]
            p_test_avg = alpha * p_test_xgb + (1 - alpha) * lgb_r["p_test"]
            n, wr = eval_p(p_val_avg, p_test_avg, val_df, test_df, y_test)
            print(f"  XGB*{alpha:.1f} + {lgb_r['name']}*{1-alpha:.1f}: n={n} wr={wr*100:.2f}%")


if __name__ == "__main__":
    main()
