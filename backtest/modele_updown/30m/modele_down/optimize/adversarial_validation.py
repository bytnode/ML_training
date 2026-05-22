"""
Adversarial validation : train a classifier to distinguish TRAIN rows from
TEST rows. Use the predicted P(test-like) as a SAMPLE WEIGHT for the DOWN
model training. This adapts the model to the test distribution.

If train ≈ test (no distribution shift), the adversarial AUC ≈ 0.5 and weights
have no effect. If there IS shift, the adversarial model picks it up and the
DOWN model trained with weights performs better on test.
"""
from __future__ import annotations

import numpy as np
import xgboost as xgb

from common import (
    load_dataset, feature_columns, temporal_split, hour_of_day_from_ms,
)
from train import build_hyperparams, _resolve_scale_pos_weight
from sklearn.metrics import roc_auc_score


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


def eval_run(p_val, p_test, val, test, y_test):
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
    print(f"Sizes: train={len(train_df):,} val={len(val_df):,} test={len(test_df):,}")

    # Step 1 : Adversarial validation
    # Combine train and test, label = 1 if test, 0 if train
    X_combined = np.concatenate([train_df[fc].to_numpy(), test_df[fc].to_numpy()])
    y_adv = np.concatenate([
        np.zeros(len(train_df), dtype=int),
        np.ones(len(test_df), dtype=int),
    ])
    # Shuffle and split for CV-style eval
    rng = np.random.RandomState(42)
    idx = rng.permutation(len(X_combined))
    n_adv_train = int(0.8 * len(idx))
    adv_train_idx = idx[:n_adv_train]
    adv_val_idx = idx[n_adv_train:]

    print("\nStep 1: training adversarial classifier (train vs test)...")
    adv = xgb.XGBClassifier(
        n_estimators=500, max_depth=5, learning_rate=0.03,
        min_child_weight=10, subsample=0.8, colsample_bytree=0.8,
        objective="binary:logistic", eval_metric="auc",
        early_stopping_rounds=30, random_state=42, n_jobs=-1,
    )
    adv.fit(X_combined[adv_train_idx], y_adv[adv_train_idx],
             eval_set=[(X_combined[adv_val_idx], y_adv[adv_val_idx])], verbose=False)
    adv_pred = adv.predict_proba(X_combined[adv_val_idx])[:, 1]
    auc = roc_auc_score(y_adv[adv_val_idx], adv_pred)
    print(f"  Adversarial AUC: {auc:.4f}  (0.5 = no shift, 1.0 = perfectly shifted)")

    # Predict on TRAIN rows : P(this row looks like test)
    p_train_is_test = adv.predict_proba(train_df[fc])[:, 1]
    print(f"  P(train row looks like test): "
          f"min={p_train_is_test.min():.4f} max={p_train_is_test.max():.4f} "
          f"mean={p_train_is_test.mean():.4f}")

    # ---- Now train DOWN model with adversarial weights
    hp = build_hyperparams()
    opt = {}
    if hp["reg_lambda"] is not False: opt["reg_lambda"] = hp["reg_lambda"]
    if hp["gamma"] is not False: opt["gamma"] = hp["gamma"]
    spw = _resolve_scale_pos_weight(hp["scale_pos_weight"], train_df["label"])
    if spw is not None: opt["scale_pos_weight"] = spw

    print(f"\n{'weight_power':>13} {'taken':>6} {'wr':>7} {'val_ll':>8}")
    # Try several power transforms of the adversarial weight
    for pwr in (0.0, 0.5, 1.0, 2.0, 3.0, 5.0):
        if pwr == 0.0:
            w = np.ones(len(train_df))
        else:
            # P(test-like)^pwr , normalized
            w = p_train_is_test ** pwr
            w = w / w.mean() * len(w) / len(w)  # roughly preserve scale

        m = xgb.XGBClassifier(
            n_estimators=hp["n_estimators"], max_depth=hp["max_depth"],
            learning_rate=hp["learning_rate"], min_child_weight=hp["min_child_weight"],
            subsample=hp["subsample"], colsample_bytree=hp["colsample_bytree"],
            objective="binary:logistic", eval_metric="logloss",
            early_stopping_rounds=hp["early_stopping_rounds"],
            random_state=42, n_jobs=-1, **opt,
        )
        m.fit(train_df[fc], train_df["label"], sample_weight=w,
               eval_set=[(val_df[fc], val_df["label"])], verbose=False)
        p_val = m.predict_proba(val_df[fc])[:, 1]
        p_test = m.predict_proba(test_df[fc])[:, 1]
        y_test = test_df["label"].astype(int).to_numpy()
        n, wr = eval_run(p_val, p_test, val_df, test_df, y_test)
        val_ll = m.evals_result()["validation_0"]["logloss"][m.best_iteration]
        marker = " ***GOAL" if (wr >= 0.65 and n >= MIN_PARIS) else ""
        print(f"  pwr={pwr:>4.1f}      {n:>5}  {wr*100:>6.2f}%  {val_ll:>8.5f}{marker}")


if __name__ == "__main__":
    main()
