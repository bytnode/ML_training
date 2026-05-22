"""
Train 24 expert models, one per UTC hour. Each is specialized on bars at that
hour. Then at inference, use the matching expert.
"""
from __future__ import annotations

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


def main():
    df = load_dataset()
    fc = feature_columns(df)
    train_df, val_df, test_df = temporal_split(df)
    print(f"Splits: train={len(train_df):,} val={len(val_df):,} test={len(test_df):,}")

    hp = build_hyperparams()
    opt = {}
    if hp["reg_lambda"] is not False: opt["reg_lambda"] = hp["reg_lambda"]
    if hp["gamma"] is not False: opt["gamma"] = hp["gamma"]

    hours_train = hour_of_day_from_ms(train_df["timestamp"].to_numpy())
    hours_val = hour_of_day_from_ms(val_df["timestamp"].to_numpy())
    hours_test = hour_of_day_from_ms(test_df["timestamp"].to_numpy())

    p_val_exp = np.zeros(len(val_df))
    p_test_exp = np.zeros(len(test_df))

    print(f"\nTraining per-hour experts...")
    for h in range(24):
        m_tr = hours_train == h
        m_va = hours_val == h
        m_te = hours_test == h
        if m_tr.sum() < 100:
            continue
        spw = _resolve_scale_pos_weight(hp["scale_pos_weight"], train_df.loc[m_tr, "label"])
        opt_h = dict(opt)
        if spw is not None: opt_h["scale_pos_weight"] = spw

        # Smaller HPs for smaller dataset
        m = xgb.XGBClassifier(
            n_estimators=500, max_depth=4,
            learning_rate=0.03, min_child_weight=5,
            subsample=0.9, colsample_bytree=0.8,
            objective="binary:logistic", eval_metric="logloss",
            early_stopping_rounds=30, random_state=42, n_jobs=-1, **opt_h,
        )
        if m_va.sum() < 10:
            m.fit(train_df.loc[m_tr, fc], train_df.loc[m_tr, "label"], verbose=False)
        else:
            m.fit(train_df.loc[m_tr, fc], train_df.loc[m_tr, "label"],
                   eval_set=[(val_df.loc[m_va, fc], val_df.loc[m_va, "label"])], verbose=False)
        if m_va.any():
            p_val_exp[m_va] = m.predict_proba(val_df.loc[m_va, fc])[:, 1]
        if m_te.any():
            p_test_exp[m_te] = m.predict_proba(test_df.loc[m_te, fc])[:, 1]
        print(f"  h{h:02d}: n_tr={int(m_tr.sum())} n_va={int(m_va.sum())} n_te={int(m_te.sum())}", flush=True)

    # Filter + calibrate per-hour-expert predictions
    wknd_t = test_df["is_weekend"].astype(int).to_numpy()
    wknd_v = val_df["is_weekend"].astype(int).to_numpy()
    vr30_t = test_df["volume_ratio_30m"].to_numpy(dtype=float)
    r3_t = test_df["return_3"].to_numpy(dtype=float)
    vr30_v = val_df["volume_ratio_30m"].to_numpy(dtype=float)
    r3_v = val_df["return_3"].to_numpy(dtype=float)
    bad_t = np.isin(hours_test, list(BAD_HOURS))
    bad_v = np.isin(hours_val, list(BAD_HOURS))
    good_t = np.isin(hours_test, list(GOOD_HOURS))
    m_t = (~bad_t) & ((vr30_t < VR30_MAX) | np.isnan(vr30_t)) & ((r3_t < R3_MAX) | np.isnan(r3_t))
    m_v = (~bad_v) & ((vr30_v < VR30_MAX) | np.isnan(vr30_v)) & ((r3_v < R3_MAX) | np.isnan(r3_v))

    val_max_ts = int(val_df["timestamp"].max())
    cutoff = val_max_ts - CD * MS_PER_DAY
    recent = val_df["timestamp"].to_numpy() >= cutoff
    cal_mask = m_v & recent

    thr = {}
    for ax, frac in [(0, WD), (1, WE)]:
        mm = cal_mask & (wknd_v == ax)
        if mm.sum() < 30:
            thr[ax] = 1.0; continue
        thr[ax] = float(np.quantile(p_val_exp[mm], 1.0 - frac))
    T_base = np.where(wknd_t == 1, thr.get(1, 1.0), thr.get(0, 1.0))
    T_eff = T_base - DELTA * good_t
    take = m_t & (p_test_exp >= T_eff)
    n = int(take.sum())
    y_test = test_df["label"].astype(int).to_numpy()
    wr = float(y_test[take].mean()) if n > 0 else 0.0
    print(f"\nPer-hour experts: n={n} wr={wr*100:.2f}%")


if __name__ == "__main__":
    main()
