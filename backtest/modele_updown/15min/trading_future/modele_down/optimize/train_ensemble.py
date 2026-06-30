"""Train K models with different random seeds, average predictions.

Saves the AVERAGED test predictions to logs/test_predictions.csv (overrides
the standard train.py output). Also trains a FIRST model (seed=42) for the
artifact (btc_updown_model.json) so live still has a deployable single model.

Reports the WR of the ensemble vs the seed=42 single model.

Idea: model variance is one source of WR noise ; averaging K seeds reduces
it. For tail decisions (where p is near the threshold), the consensus is
more reliable than any single tree's vote.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

PKG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG_ROOT))

from config import (
    DB_PATH, TRAIN_DAYS, TEST_DAYS, VAL_FRACTION,
    ASYM_VOL_FRAC_WEEKDAY, ASYM_VOL_FRAC_WEEKEND, ASYM_SKIP_CELLS,
    ASYM_BAD_HOURS, ASYM_MAX_P_UP, ASYM_MODELE_UP_PREDS,
    ASYM_HOUR_BOOST_GOOD_HOURS, ASYM_HOUR_BOOST_DELTA,
    XGB_N_ESTIMATORS, XGB_MAX_DEPTH, XGB_LEARNING_RATE,
    XGB_MIN_CHILD_WEIGHT, XGB_SUBSAMPLE, XGB_COLSAMPLE_BYTREE,
    XGB_GAMMA, XGB_EARLY_STOPPING_ROUNDS, XGB_SCALE_POS_WEIGHT,
)
from features import build_training_dataframe
from train import (
    compute_thresholds_asym_wknd_excl,
    apply_thresholds_asym_hour_boost,
    _resolve_scale_pos_weight,
    _load_p_up_keep_mask,
)

SEEDS = [42, 7, 123, 2024, 9999]


def main():
    df = build_training_dataframe(DB_PATH, verbose=False)
    df = df.sort_values("timestamp").reset_index(drop=True)
    ms_per_day = 86_400 * 1000
    ts_max = df["timestamp"].max()
    test_cutoff = ts_max - TEST_DAYS * ms_per_day
    test_start = int(df["timestamp"].searchsorted(test_cutoff, side="left"))
    train_val_cutoff = test_cutoff - TRAIN_DAYS * ms_per_day
    train_val_start = int(df["timestamp"].searchsorted(train_val_cutoff, side="left"))
    n_tv = test_start - train_val_start
    val_size = int(n_tv * VAL_FRACTION)
    train_end = test_start - val_size

    train = df.iloc[train_val_start:train_end]
    val   = df.iloc[train_end:test_start]
    test  = df.iloc[test_start:]

    feature_cols = [c for c in df.columns if c not in ("timestamp", "label", "volatility_state")]
    X_train, y_train = train[feature_cols], train["label"]
    X_val,   y_val   = val[feature_cols],   val["label"]
    X_test,  y_test  = test[feature_cols],  test["label"]

    print(f"train={len(train):,}  val={len(val):,}  test={len(test):,}  features={len(feature_cols)}")

    spw = _resolve_scale_pos_weight(XGB_SCALE_POS_WEIGHT, y_train)

    test_preds_per_seed = []
    val_preds_per_seed  = []

    for seed in SEEDS:
        model = xgb.XGBClassifier(
            n_estimators=XGB_N_ESTIMATORS, max_depth=XGB_MAX_DEPTH,
            learning_rate=XGB_LEARNING_RATE, min_child_weight=XGB_MIN_CHILD_WEIGHT,
            subsample=XGB_SUBSAMPLE, colsample_bytree=XGB_COLSAMPLE_BYTREE,
            gamma=XGB_GAMMA, early_stopping_rounds=XGB_EARLY_STOPPING_ROUNDS,
            random_state=seed, scale_pos_weight=spw,
            tree_method="hist", n_jobs=-1, eval_metric="logloss",
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        p_val  = model.predict_proba(X_val)[:, 1]
        p_test = model.predict_proba(X_test)[:, 1]
        val_preds_per_seed.append(p_val)
        test_preds_per_seed.append(p_test)
        print(f"  seed={seed:5d}  best_iter={model.best_iteration:4d}  "
              f"val_p_down_mean={p_val.mean():.4f}  test_p_down_mean={p_test.mean():.4f}")
        if seed == 42:
            # Save the seed=42 model so live still has a deployable single model.
            model.save_model("btc_updown_model.json")

    # Average predictions.
    p_val_avg  = np.mean(val_preds_per_seed,  axis=0)
    p_test_avg = np.mean(test_preds_per_seed, axis=0)

    # Calibration on AVERAGED val.
    thresholds, _ = compute_thresholds_asym_wknd_excl(
        val, p_val_avg, bad_hours=ASYM_BAD_HOURS,
    )
    take = apply_thresholds_asym_hour_boost(
        test, p_test_avg, thresholds, bad_hours=ASYM_BAD_HOURS,
    )

    # Cross-model runtime filter (if active).
    n_pre = int(take.sum())
    wr_pre = float(y_test.to_numpy().astype(int)[take].mean()) if n_pre > 0 else None
    if ASYM_MAX_P_UP < 1.0:
        keep = _load_p_up_keep_mask(
            test["timestamp"].to_numpy(),
            preds_path=ASYM_MODELE_UP_PREDS,
            max_p_up=ASYM_MAX_P_UP,
        )
        take = take & keep
        print(f"  cross-model filter: {n_pre} -> {int(take.sum())} bets  (pre_wr={wr_pre:.4f})")

    label = y_test.to_numpy().astype(int)
    n = int(take.sum())
    wr = float(label[take].mean()) if n > 0 else 0.0
    print(f"\n=== ENSEMBLE ({len(SEEDS)} seeds) ===")
    print(f"  bets = {n}  WR = {wr:.4f}")
    print(f"  thresholds: weekday={thresholds.get(0, 1.0):.4f}  weekend={thresholds.get(1, 1.0):.4f}")

    # Save averaged test predictions in standard format.
    out = pd.DataFrame({
        "timestamp":         test["timestamp"].values,
        "is_weekend":        test["is_weekend"].values,
        "volatility_state":  test["volatility_state"].values,
        "label":             test["label"].values,
        "p_down":            p_test_avg,
        "confidence":        p_test_avg,
    })
    out.to_csv("logs/test_predictions.csv", index=False)
    print(f"  saved averaged test_predictions.csv ({len(out)} rows)")


if __name__ == "__main__":
    main()
