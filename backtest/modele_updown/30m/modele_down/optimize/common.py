"""
Shared helpers for autonomous optimization sweeps in modele_down/optimize/.

All scripts assume cwd = modele_down/. Run them as `python optimize/<name>.py`.

Mirrors modele_up/optimize/common.py but supports bad_hours and DOWN-only:
the threshold quantile is on P(down), label=1 means next candle is red.
"""
from __future__ import annotations

import sys
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from xgboost import XGBClassifier

from config import (
    DB_PATH, TRAIN_DAYS, TEST_DAYS, VAL_FRACTION, XGB_RANDOM_STATE,
    ASYM_SKIP_CELLS, ASYM_HOUR_BOOST_GOOD_HOURS, ASYM_HOUR_BOOST_DELTA,
    ASYM_BAD_HOURS, ASYM_USE_FULL_VAL, ASYM_CALIB_DAYS,
)
from features import build_training_dataframe
from train import (
    build_hyperparams, _resolve_scale_pos_weight,
    apply_thresholds_asym_hour_boost,
    compute_thresholds_asym_wknd_excl,
)


def temporal_split(df: pd.DataFrame):
    """train / val / test split identique a train.py."""
    df = df.sort_values("timestamp").reset_index(drop=True)
    ms_per_day = 86_400_000
    ts_max = df["timestamp"].max()
    test_cutoff = ts_max - TEST_DAYS * ms_per_day
    test_start = int(df["timestamp"].searchsorted(test_cutoff, side="left"))
    if TRAIN_DAYS is None:
        tv_start = 0
    else:
        tv_cutoff = test_cutoff - TRAIN_DAYS * ms_per_day
        tv_start = int(df["timestamp"].searchsorted(tv_cutoff, side="left"))
    n_tv = test_start - tv_start
    val_size = int(n_tv * VAL_FRACTION)
    train_end = test_start - val_size
    return df.iloc[tv_start:train_end], df.iloc[train_end:test_start], df.iloc[test_start:]


def build_model(y_train, seed, hp_overrides: dict | None = None):
    hp = build_hyperparams(hp_overrides)
    optional_kwargs = {}
    if hp["reg_lambda"] is not False: optional_kwargs["reg_lambda"] = hp["reg_lambda"]
    if hp["reg_alpha"]  is not False: optional_kwargs["reg_alpha"]  = hp["reg_alpha"]
    if hp["gamma"]      is not False: optional_kwargs["gamma"]      = hp["gamma"]
    spw = _resolve_scale_pos_weight(hp["scale_pos_weight"], y_train)
    if spw is not None:
        optional_kwargs["scale_pos_weight"] = spw
    return XGBClassifier(
        n_estimators=hp["n_estimators"],
        max_depth=hp["max_depth"],
        learning_rate=hp["learning_rate"],
        min_child_weight=hp["min_child_weight"],
        subsample=hp["subsample"],
        colsample_bytree=hp["colsample_bytree"],
        objective="binary:logistic",
        eval_metric="logloss",
        early_stopping_rounds=hp["early_stopping_rounds"],
        random_state=seed,
        n_jobs=-1,
        **optional_kwargs,
    )


def train_once(df: pd.DataFrame, feature_cols: list[str], seed: int = XGB_RANDOM_STATE,
               hp_overrides: dict | None = None):
    """Train + return everything needed for downstream eval."""
    train, val, test = temporal_split(df)
    X_train, y_train = train[feature_cols], train["label"]
    X_val,   y_val   = val[feature_cols],   val["label"]
    X_test,  y_test  = test[feature_cols],  test["label"]

    model = build_model(y_train, seed, hp_overrides)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    best = int(model.best_iteration)
    val_ll = float(model.evals_result()["validation_0"]["logloss"][best])
    p_val  = model.predict_proba(X_val)[:, 1]
    p_test = model.predict_proba(X_test)[:, 1]
    test_ll = float(log_loss(y_test, p_test))

    return {
        "model": model, "best_iter": best,
        "val_logloss": val_ll, "test_logloss": test_ll,
        "train": train, "val": val, "test": test,
        "p_val": p_val, "p_test": p_test,
        "y_test": y_test.to_numpy().astype(int),
    }


def evaluate_with_calibration(run,
                              vol_frac_wd: float,
                              vol_frac_we: float,
                              good_hours: tuple = ASYM_HOUR_BOOST_GOOD_HOURS,
                              hour_boost_delta: float = ASYM_HOUR_BOOST_DELTA,
                              skip_cells=ASYM_SKIP_CELLS,
                              bad_hours: tuple = ASYM_BAD_HOURS,
                              use_full_val: bool = ASYM_USE_FULL_VAL,
                              calib_days: int = ASYM_CALIB_DAYS) -> dict:
    """Apply (vol_frac, hour_boost, bad_hours) calibration to a pre-trained run."""
    thresholds_per_wknd, _ = compute_thresholds_asym_wknd_excl(
        run["val"], run["p_val"],
        vol_frac_wd=vol_frac_wd, vol_frac_we=vol_frac_we,
        skip_cells=skip_cells, use_full_val=use_full_val, calib_days=calib_days,
    )
    take = apply_thresholds_asym_hour_boost(
        run["test"], run["p_test"], thresholds_per_wknd,
        skip_cells=skip_cells, good_hours=good_hours,
        hour_boost_delta=hour_boost_delta, bad_hours=bad_hours,
    )
    n_taken = int(take.sum())
    y_test = run["y_test"]
    wr = float(y_test[take].mean()) if n_taken > 0 else 0.0
    ev = n_taken * (wr - 0.5)
    return {
        "taken": n_taken, "wr": wr, "ev": ev,
        "T_wd": thresholds_per_wknd.get(0, None),
        "T_we": thresholds_per_wknd.get(1, None),
    }


def load_dataset():
    """Load + filter the parquet (= what build_training_dataframe does)."""
    return build_training_dataframe(DB_PATH, verbose=False)


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns
            if c not in ("timestamp", "label", "volatility_state", "is_weekend")]


def hour_of_day_from_ms(ts_ms: np.ndarray) -> np.ndarray:
    ts_ms = np.asarray(ts_ms, dtype=np.int64)
    return ((ts_ms // (60 * 60 * 1000)) % 24).astype(int)
