"""Helpers for training-side experiments: next-candle body (for alternative
labels / sample weighting) and a meta-filter trainer."""
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

import lib

DB = lib.PKG / "DB" / "btc_multi_tf.db"


def next_candle_maps():
    """Return (abs_body_map, signed_body_map) keyed by timestamp_ms.
    body = (close-open)/close_prev = close_ret - open_ret of the NEXT 15m
    candle. signed: negative = red. Computed on the full contiguous series."""
    conn = sqlite3.connect(DB)
    try:
        d = pd.read_sql_query(
            "SELECT timestamp_ms, open_ret, close_ret FROM candles "
            "WHERE timeframe='15m' ORDER BY timestamp_ms", conn)
    finally:
        conn.close()
    body = (d["close_ret"] - d["open_ret"])
    next_signed = body.shift(-1).to_numpy()
    ts = d["timestamp_ms"].to_numpy(np.int64)
    return (dict(zip(ts, np.abs(next_signed))), dict(zip(ts, next_signed)))


def attach_next_body(df):
    """Add 'next_body_abs' and 'next_body_signed' columns to a harness df."""
    am, sm = next_candle_maps()
    df = df.copy()
    df["next_body_abs"] = df["timestamp"].map(am).astype(float)
    df["next_body_signed"] = df["timestamp"].map(sm).astype(float)
    return df


def train_raw(X_train, y_train, X_val, y_val, hp=None, sample_weight=None):
    """Thin XGBClassifier trainer with the package's hyperparam plumbing."""
    import config
    hp = hp or {}
    p = dict(n_estimators=config.XGB_N_ESTIMATORS, max_depth=config.XGB_MAX_DEPTH,
             learning_rate=config.XGB_LEARNING_RATE,
             min_child_weight=config.XGB_MIN_CHILD_WEIGHT,
             subsample=config.XGB_SUBSAMPLE, colsample_bytree=config.XGB_COLSAMPLE_BYTREE,
             reg_lambda=config.XGB_REG_LAMBDA, reg_alpha=config.XGB_REG_ALPHA,
             gamma=config.XGB_GAMMA, early_stopping_rounds=config.XGB_EARLY_STOPPING_ROUNDS,
             random_state=config.XGB_RANDOM_STATE)
    p.update(hp)
    opt = {}
    for k in ("reg_lambda", "reg_alpha", "gamma"):
        if p[k] is not False:
            opt[k] = p[k]
    m = xgb.XGBClassifier(
        n_estimators=p["n_estimators"], max_depth=p["max_depth"],
        learning_rate=p["learning_rate"], min_child_weight=p["min_child_weight"],
        subsample=p["subsample"], colsample_bytree=p["colsample_bytree"],
        objective="binary:logistic", eval_metric="logloss",
        early_stopping_rounds=p["early_stopping_rounds"],
        random_state=p["random_state"], n_jobs=-1, **opt)
    m.fit(X_train, y_train, eval_set=[(X_val, y_val)],
          sample_weight=sample_weight, verbose=False)
    return m
