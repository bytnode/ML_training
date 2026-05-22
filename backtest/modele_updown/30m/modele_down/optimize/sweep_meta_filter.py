"""
Stage 2 meta-classifier: predict whether stage 1 (DOWN model + filters)
will be CORRECT on a given row.

Train on TRAIN where label = stage1_correct = (model_says_take AND label=1).
Then at inference, only take when stage1_says_take AND stage2 predicts correct.

If stage 2 can identify FALSE POSITIVES in stage 1, this lifts WR.
"""
from __future__ import annotations

import numpy as np
import xgboost as xgb

from common import (
    load_dataset, feature_columns, temporal_split, hour_of_day_from_ms,
)
from train import (
    build_hyperparams, _resolve_scale_pos_weight,
)


BAD_HOURS = (1, 8, 10, 13, 14, 17, 21)
GOOD_HOURS = (2, 9, 20, 22)
DELTA = 0.030
CD = 90; WD = 0.060; WE = 0.20
VR30_MAX = 1.50
R3_MAX = 0.030
MIN_PARIS = 750


def compute_stage1_take(df, p_arr, val_df, p_val_arr, wd_frac, we_frac, cd_days,
                        bad_hours, good_hours, delta, vr30_max, r3_max):
    hours = hour_of_day_from_ms(df["timestamp"].to_numpy())
    wknd = df["is_weekend"].astype(int).to_numpy()
    wknd_v = val_df["is_weekend"].astype(int).to_numpy()
    vr30 = df["volume_ratio_30m"].to_numpy(dtype=float)
    r3 = df["return_3"].to_numpy(dtype=float)
    vr30_v = val_df["volume_ratio_30m"].to_numpy(dtype=float)
    r3_v = val_df["return_3"].to_numpy(dtype=float)
    hours_v = hour_of_day_from_ms(val_df["timestamp"].to_numpy())
    bad = np.isin(hours, list(bad_hours))
    bad_v = np.isin(hours_v, list(bad_hours))
    good = np.isin(hours, list(good_hours))

    m = (~bad) & ((vr30 < vr30_max) | np.isnan(vr30)) & ((r3 < r3_max) | np.isnan(r3))
    m_v = (~bad_v) & ((vr30_v < vr30_max) | np.isnan(vr30_v)) & ((r3_v < r3_max) | np.isnan(r3_v))
    ms_per_day = 86_400_000
    val_max_ts = int(val_df["timestamp"].max())
    cutoff = val_max_ts - cd_days * ms_per_day
    recent = val_df["timestamp"].to_numpy() >= cutoff
    cal_mask = m_v & recent
    thrs = {}
    for ax, frac in [(0, wd_frac), (1, we_frac)]:
        mm = cal_mask & (wknd_v == ax)
        if mm.sum() < 30:
            thrs[ax] = 1.0
            continue
        thrs[ax] = float(np.quantile(p_val_arr[mm], 1.0 - frac))
    T_base = np.where(wknd == 1, thrs.get(1, 1.0), thrs.get(0, 1.0))
    T_eff = T_base - delta * good
    take = m & (p_arr >= T_eff)
    return take


def main():
    df = load_dataset()
    fc = feature_columns(df)
    train_df, val_df, test_df = temporal_split(df)

    # Stage 1: predict with default config
    hp = build_hyperparams()
    opt = {}
    if hp["reg_lambda"] is not False: opt["reg_lambda"] = hp["reg_lambda"]
    if hp["reg_alpha"]  is not False: opt["reg_alpha"]  = hp["reg_alpha"]
    if hp["gamma"]      is not False: opt["gamma"]      = hp["gamma"]
    spw = _resolve_scale_pos_weight(hp["scale_pos_weight"], train_df["label"])
    if spw is not None: opt["scale_pos_weight"] = spw
    s1 = xgb.XGBClassifier(
        n_estimators=hp["n_estimators"], max_depth=hp["max_depth"],
        learning_rate=hp["learning_rate"], min_child_weight=hp["min_child_weight"],
        subsample=hp["subsample"], colsample_bytree=hp["colsample_bytree"],
        objective="binary:logistic", eval_metric="logloss",
        early_stopping_rounds=hp["early_stopping_rounds"],
        random_state=42, n_jobs=-1, **opt,
    )
    s1.fit(train_df[fc], train_df["label"], eval_set=[(val_df[fc], val_df["label"])], verbose=False)
    print(f"Stage 1 trained. best_iter={s1.best_iteration}")

    # Predict on all
    p_train = s1.predict_proba(train_df[fc])[:, 1]
    p_val = s1.predict_proba(val_df[fc])[:, 1]
    p_test = s1.predict_proba(test_df[fc])[:, 1]

    # In TRAIN, simulate stage 1 take decisions. We need a "fake val" for calib.
    # Use the last VAL_FRACTION fraction of train as fake val for stage1.
    train_split_idx = int(len(train_df) * 0.9)
    sub_train = train_df.iloc[:train_split_idx].reset_index(drop=True)
    sub_val = train_df.iloc[train_split_idx:].reset_index(drop=True)
    p_sub_train = p_train[:train_split_idx]
    p_sub_val = p_train[train_split_idx:]

    take_train = compute_stage1_take(sub_train, p_sub_train, sub_val, p_sub_val,
                                      WD, WE, CD, BAD_HOURS, GOOD_HOURS, DELTA,
                                      VR30_MAX, R3_MAX)
    # Label for stage 2: 1 if stage-1 would have been CORRECT on this row
    label_train = sub_train["label"].astype(int).to_numpy()
    stage2_label_train = (take_train & (label_train == 1)).astype(int)
    # Only train stage2 on rows where stage1 took (= positive context for the meta)
    # Or train on all rows: stage2 = "is this a good DOWN bet?"
    print(f"Train rows where stage1 would take: {int(take_train.sum())}")
    print(f"  Of those, correct (next bar red): {int((take_train & (label_train == 1)).sum())}")
    print(f"  WR on stage1 train takes: {(take_train & (label_train == 1)).sum() / take_train.sum():.4f}")

    # Stage 2 features: same fc, label = stage1_correct (only on take_train rows)
    train_mask_idx = np.where(take_train)[0]
    if len(train_mask_idx) < 100:
        print("Not enough stage1 takes in train, abort")
        return
    X_s2 = sub_train.iloc[train_mask_idx][fc]
    y_s2 = stage2_label_train[train_mask_idx]
    # Build val for stage 2 from the last 10% of train_mask
    n_val_s2 = max(50, int(len(train_mask_idx) * 0.1))
    X_train_s2 = X_s2.iloc[:-n_val_s2]
    y_train_s2 = y_s2[:-n_val_s2]
    X_val_s2 = X_s2.iloc[-n_val_s2:]
    y_val_s2 = y_s2[-n_val_s2:]
    print(f"\nStage 2 train: {len(X_train_s2)} rows, base WR = {y_train_s2.mean():.4f}")
    print(f"Stage 2 val:   {len(X_val_s2)} rows, base WR = {y_val_s2.mean():.4f}")

    s2 = xgb.XGBClassifier(
        n_estimators=500, max_depth=4, learning_rate=0.03,
        min_child_weight=10, subsample=0.8, colsample_bytree=0.8,
        objective="binary:logistic", eval_metric="logloss",
        early_stopping_rounds=30, random_state=42, n_jobs=-1,
    )
    s2.fit(X_train_s2, y_train_s2, eval_set=[(X_val_s2, y_val_s2)], verbose=False)
    print(f"Stage 2 trained. best_iter={s2.best_iteration}")

    # Apply to TEST: get stage1 take + stage2 predictions
    take_test = compute_stage1_take(test_df, p_test, val_df, p_val,
                                     WD, WE, CD, BAD_HOURS, GOOD_HOURS, DELTA,
                                     VR30_MAX, R3_MAX)
    print(f"\nStage 1 alone on test: take={int(take_test.sum())} "
          f"wr={test_df['label'].iloc[np.where(take_test)[0]].mean()*100:.2f}%")

    # Stage 2 on test (predict on rows where stage1 took)
    test_idx = np.where(take_test)[0]
    p_s2_test = s2.predict_proba(test_df.iloc[test_idx][fc])[:, 1]
    print(f"\nStage 2 P(correct) on stage1 takes: "
          f"min={p_s2_test.min():.4f} max={p_s2_test.max():.4f} mean={p_s2_test.mean():.4f}")

    # Sweep stage 2 thresholds
    print(f"\n{'s2_threshold':>14} {'taken':>6} {'wr':>7}")
    for t2 in (0.0, 0.40, 0.45, 0.48, 0.50, 0.52, 0.55, 0.58, 0.60):
        keep = p_s2_test >= t2
        kept_idx = test_idx[keep]
        n = len(kept_idx)
        if n == 0:
            continue
        wr = float(test_df["label"].iloc[kept_idx].mean())
        marker = " *** GOAL" if (wr >= 0.65 and n >= MIN_PARIS) else ""
        print(f"  >={t2:.3f}     {n:>5} {wr*100:>6.2f}%{marker}")


if __name__ == "__main__":
    main()
