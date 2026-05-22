"""
Train DOWN model ONLY on samples that pass the inference filters.

Theory: if the model is trained on the same subspace it'll be evaluated on,
it might learn sharper signal in that subspace.

Sample filter: volume_ratio_30m < 1.5 AND return_3 < 0.030 (current best).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

from common import (
    load_dataset, feature_columns, temporal_split, hour_of_day_from_ms,
)
from train import (
    build_hyperparams, _resolve_scale_pos_weight,
    compute_thresholds_asym_wknd_excl,
)


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


def main():
    df = load_dataset()
    fc = feature_columns(df)
    train_df, val_df, test_df = temporal_split(df)

    # Apply filters to TRAIN
    vr30_tr = train_df["volume_ratio_30m"].to_numpy(dtype=float)
    r3_tr = train_df["return_3"].to_numpy(dtype=float)
    train_mask = ((vr30_tr < VR30_MAX) | np.isnan(vr30_tr)) & ((r3_tr < R3_MAX) | np.isnan(r3_tr))
    train_filt = train_df.loc[train_mask].reset_index(drop=True)
    print(f"Train: {len(train_df):,} -> filtered {len(train_filt):,}")

    # Same filter on val for calibration
    vr30_v = val_df["volume_ratio_30m"].to_numpy(dtype=float)
    r3_v = val_df["return_3"].to_numpy(dtype=float)
    val_mask = ((vr30_v < VR30_MAX) | np.isnan(vr30_v)) & ((r3_v < R3_MAX) | np.isnan(r3_v))
    val_filt = val_df.loc[val_mask].reset_index(drop=True)
    print(f"Val: {len(val_df):,} -> filtered {len(val_filt):,}")

    X_train = train_filt[fc]; y_train = train_filt["label"]
    X_val_filt = val_filt[fc]; y_val_filt = val_filt["label"]

    # Train
    hp = build_hyperparams()
    opt = {}
    if hp["reg_lambda"] is not False: opt["reg_lambda"] = hp["reg_lambda"]
    if hp["reg_alpha"]  is not False: opt["reg_alpha"]  = hp["reg_alpha"]
    if hp["gamma"]      is not False: opt["gamma"]      = hp["gamma"]
    spw = _resolve_scale_pos_weight(hp["scale_pos_weight"], y_train)
    if spw is not None:
        opt["scale_pos_weight"] = spw

    model = xgb.XGBClassifier(
        n_estimators=hp["n_estimators"], max_depth=hp["max_depth"],
        learning_rate=hp["learning_rate"], min_child_weight=hp["min_child_weight"],
        subsample=hp["subsample"], colsample_bytree=hp["colsample_bytree"],
        objective="binary:logistic", eval_metric="logloss",
        early_stopping_rounds=hp["early_stopping_rounds"],
        random_state=42, n_jobs=-1, **opt,
    )
    model.fit(X_train, y_train, eval_set=[(X_val_filt, y_val_filt)], verbose=False)
    print(f"Trained on filtered subset. best_iter={model.best_iteration} "
          f"val_ll={model.evals_result()['validation_0']['logloss'][model.best_iteration]:.5f}")

    # Predict on FULL val (for calibration) and TEST
    p_val_full = model.predict_proba(val_df[fc])[:, 1]
    p_test = model.predict_proba(test_df[fc])[:, 1]
    y_test = test_df["label"].astype(int).to_numpy()

    hours_t = hour_of_day_from_ms(test_df["timestamp"].to_numpy())
    wknd_t = test_df["is_weekend"].astype(int).to_numpy()
    wknd_v = val_df["is_weekend"].astype(int).to_numpy()
    vr30_t = test_df["volume_ratio_30m"].to_numpy(dtype=float)
    r3_t = test_df["return_3"].to_numpy(dtype=float)

    # Mask val with filters for calibration
    bad_v = np.isin(hour_of_day_from_ms(val_df["timestamp"].to_numpy()), list(BAD_HOURS))
    bad_t = np.isin(hours_t, list(BAD_HOURS))
    good_t = np.isin(hours_t, list(GOOD_HOURS))
    m_v = (~bad_v) & val_mask
    m_t = (~bad_t) & ((vr30_t < VR30_MAX) | np.isnan(vr30_t)) & ((r3_t < R3_MAX) | np.isnan(r3_t))

    thr = filtered_threshold(val_df, p_val_full, wknd_v, m_v, WD, WE, CD)
    T_base = np.where(wknd_t == 1, thr.get(1, 1.0), thr.get(0, 1.0))
    T_eff = T_base - DELTA * good_t
    take = m_t & (p_test >= T_eff)
    n = int(take.sum())
    wr = float(y_test[take].mean()) if n > 0 else 0.0
    print(f"\n=== Filter-trained DOWN model results ===")
    print(f"  taken={n}  wr={wr*100:.2f}%  T_wd={thr.get(0):.4f}  T_we={thr.get(1):.4f}")

    # Also try multiple WD/cd combos
    print(f"\n=== Variations:")
    for wd in (0.040, 0.060, 0.080, 0.100, 0.140):
        for cd in (55, 90, 180):
            thr_v = filtered_threshold(val_df, p_val_full, wknd_v, m_v, wd, 0.20, cd)
            T_base_v = np.where(wknd_t == 1, thr_v.get(1, 1.0), thr_v.get(0, 1.0))
            T_eff_v = T_base_v - DELTA * good_t
            take_v = m_t & (p_test >= T_eff_v)
            n_v = int(take_v.sum())
            if n_v < 600 or n_v > 1200:
                continue
            wr_v = float(y_test[take_v].mean()) if n_v > 0 else 0.0
            print(f"  cd={cd} wd={wd:.3f} we=0.20: taken={n_v} wr={wr_v*100:.2f}%")


if __name__ == "__main__":
    main()
