"""Shared harness for modele_down 15m volume-optimization experiments.

Goal of this work: lift the DOWN model from 1800 bets @ 60.06% WR to
>=2500 bets while keeping WR >= 60% on the 365d test window.

Everything here mirrors train.py exactly (split, model, scale_pos_weight)
so numbers are comparable to logs/runs.csv. Calibration is generalized so
we can experiment with finer bucket structures than weekday/weekend.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

PKG = Path(__file__).resolve().parent.parent          # .../15min/modele_down
os.chdir(PKG)
sys.path.insert(0, str(PKG))

import config                                          # noqa: E402
from features import build_training_dataframe          # noqa: E402

MS_DAY = 86_400_000


# ============================================================
#  Data loading + temporal split (identical to train.py)
# ============================================================
def load_master(parquet_path=None) -> pd.DataFrame:
    """df exactly as train.py sees it: warmup applied, nan labels dropped,
    timestamp_ms renamed -> timestamp. Sorted by timestamp."""
    path = parquet_path if parquet_path is not None else config.DB_PATH
    df = build_training_dataframe(path, verbose=False)
    return df.sort_values("timestamp").reset_index(drop=True)


def temporal_split(df, train_days=None, test_days=None, val_fraction=None,
                   end_ts=None):
    """Same split as train.py. end_ts lets us slide the test window for
    walk-forward robustness checks (default = last bar in df)."""
    train_days = config.TRAIN_DAYS if train_days is None else train_days
    test_days = config.TEST_DAYS if test_days is None else test_days
    val_fraction = config.VAL_FRACTION if val_fraction is None else val_fraction

    ts = df["timestamp"]
    ts_anchor = ts.max() if end_ts is None else end_ts
    test_cutoff = ts_anchor - test_days * MS_DAY
    test_start = int(ts.searchsorted(test_cutoff, side="left"))
    test_end = int(ts.searchsorted(ts_anchor, side="right"))

    if train_days is None:
        tv_start = 0
    else:
        tv_cutoff = test_cutoff - train_days * MS_DAY
        tv_start = int(ts.searchsorted(tv_cutoff, side="left"))

    n_tv = test_start - tv_start
    val_size = int(n_tv * val_fraction)
    train_end = test_start - val_size

    train = df.iloc[tv_start:train_end].reset_index(drop=True)
    val = df.iloc[train_end:test_start].reset_index(drop=True)
    test = df.iloc[test_start:test_end].reset_index(drop=True)
    return train, val, test


def feature_cols(df):
    """Feature columns train.py feeds the model (excludes timestamp/label and
    volatility_state, which is kept only as a bucket key)."""
    return [c for c in df.columns if c not in ("timestamp", "label", "volatility_state")]


# ============================================================
#  Model (identical hyperparam plumbing to train.py)
# ============================================================
def train_model(train, val, feat_cols, hp=None):
    """Train an XGBClassifier. hp overrides config defaults."""
    hp = hp or {}
    p = dict(
        n_estimators=config.XGB_N_ESTIMATORS,
        max_depth=config.XGB_MAX_DEPTH,
        learning_rate=config.XGB_LEARNING_RATE,
        min_child_weight=config.XGB_MIN_CHILD_WEIGHT,
        subsample=config.XGB_SUBSAMPLE,
        colsample_bytree=config.XGB_COLSAMPLE_BYTREE,
        reg_lambda=config.XGB_REG_LAMBDA,
        reg_alpha=config.XGB_REG_ALPHA,
        gamma=config.XGB_GAMMA,
        early_stopping_rounds=config.XGB_EARLY_STOPPING_ROUNDS,
        random_state=config.XGB_RANDOM_STATE,
        scale_pos_weight=config.XGB_SCALE_POS_WEIGHT,
    )
    p.update(hp)

    opt = {}
    for k in ("reg_lambda", "reg_alpha", "gamma"):
        if p[k] is not False:
            opt[k] = p[k]
    spw = p["scale_pos_weight"]
    if spw is not False:
        if spw == "auto":
            n_pos = int((train["label"] == 1).sum())
            n_neg = int((train["label"] == 0).sum())
            opt["scale_pos_weight"] = (n_neg / n_pos) if n_pos else 1.0
        else:
            opt["scale_pos_weight"] = float(spw)

    m = xgb.XGBClassifier(
        n_estimators=p["n_estimators"],
        max_depth=p["max_depth"],
        learning_rate=p["learning_rate"],
        min_child_weight=p["min_child_weight"],
        subsample=p["subsample"],
        colsample_bytree=p["colsample_bytree"],
        objective="binary:logistic",
        eval_metric="logloss",
        early_stopping_rounds=p["early_stopping_rounds"],
        random_state=p["random_state"],
        n_jobs=-1,
        **opt,
    )
    m.fit(train[feat_cols], train["label"],
          eval_set=[(val[feat_cols], val["label"])], verbose=False)
    return m


def predict(model, df, feat_cols):
    return model.predict_proba(df[feat_cols])[:, 1]


# ============================================================
#  Bucket helpers
# ============================================================
def hour_utc(df):
    """UTC hour 0..23 from the ms timestamp column."""
    ts = df["timestamp"].to_numpy(dtype=np.int64)
    return ((ts // (15 * 60 * 1000)) % 96 // 4).astype(int)


# ============================================================
#  Generic calibration  (quantile per axis, with skip + boost)
# ============================================================
def calibrate(val_df, p_val, *, skip_fn, axis_fn, vol_frac, min_samples=30):
    """Calibrate a quantile threshold per axis key on the non-skipped val rows.

    skip_fn(df)  -> bool array : rows never bet
    axis_fn(df)  -> array      : threshold-axis key per row
    vol_frac     -> float OR dict{key: float} : fraction of non-skip val of
                    that axis kept (top quantile).
    Returns {axis_key: threshold}.
    """
    skip = np.asarray(skip_fn(val_df), dtype=bool)
    axes = np.asarray(axis_fn(val_df))
    thr = {}
    for k in np.unique(axes):
        m = (axes == k) & (~skip)
        n = int(m.sum())
        frac = vol_frac[k] if isinstance(vol_frac, dict) else vol_frac
        if n < min_samples or frac <= 0:
            thr[k] = 1.0
            continue
        thr[k] = float(np.quantile(p_val[m], 1.0 - frac))
    return thr


def evaluate(df, p, thr, *, skip_fn, axis_fn, boost_fn=None):
    """Apply thresholds. boost_fn(df) -> per-row threshold reduction (>=0).
    Returns dict(n, wr, take)."""
    p = np.asarray(p, dtype=float)
    skip = np.asarray(skip_fn(df), dtype=bool)
    axes = np.asarray(axis_fn(df))
    T = np.array([thr.get(k, 1.0) for k in axes], dtype=float)
    if boost_fn is not None:
        T = T - np.asarray(boost_fn(df), dtype=float)
    take = (~skip) & (p >= T)
    y = df["label"].to_numpy().astype(int)
    n = int(take.sum())
    wr = float(y[take].mean()) if n > 0 else float("nan")
    return dict(n=n, wr=wr, take=take)


# ============================================================
#  Current production strategy  (V2: asym_wknd_excl_hour_boost)
# ============================================================
def skip_vol2(df):
    return df["volatility_state"].to_numpy().astype(int) == 2


def axis_weekend(df):
    return df["is_weekend"].to_numpy().astype(int)


def make_hour_boost(good_hours=(20, 21, 22), delta=0.02):
    good = set(int(h) for h in good_hours)

    def _boost(df):
        h = hour_utc(df)
        return np.where(np.isin(h, list(good)), delta, 0.0)
    return _boost


def current_strategy_curve(val_df, p_val, test_df, p_test,
                           we_frac=0.1, wd_fracs=None,
                           good_hours=(20, 21, 22), delta=0.02):
    """Sweep weekday vol_frac, hold weekend frac. Returns list of dicts with
    test n / wr for the production V2 strategy."""
    if wd_fracs is None:
        wd_fracs = np.round(np.arange(0.005, 0.121, 0.005), 4)
    boost = make_hour_boost(good_hours, delta)
    rows = []
    for wd in wd_fracs:
        thr = calibrate(val_df, p_val, skip_fn=skip_vol2, axis_fn=axis_weekend,
                        vol_frac={0: float(wd), 1: float(we_frac)})
        res = evaluate(test_df, p_test, thr, skip_fn=skip_vol2,
                       axis_fn=axis_weekend, boost_fn=boost)
        rows.append(dict(wd_frac=float(wd), we_frac=float(we_frac),
                         n=res["n"], wr=res["wr"]))
    return rows


# ============================================================
#  Test-fitted frontier (optimistic ceiling)
# ============================================================
def test_frontier(p_test, y_test, allowed_mask, volumes):
    """Sort allowed test bets by p desc; report WR at each cumulative volume.
    This is the BEST POSSIBLE wr at a given volume if we could rank+threshold
    perfectly on the test set itself. Pure upper bound, not achievable honestly.
    """
    p = np.asarray(p_test, dtype=float)
    y = np.asarray(y_test, dtype=int)
    allowed = np.asarray(allowed_mask, dtype=bool)
    order = np.argsort(-p[allowed])
    y_sorted = y[allowed][order]
    cum = np.cumsum(y_sorted)
    out = []
    for v in volumes:
        v = int(v)
        if v <= 0 or v > len(y_sorted):
            continue
        out.append(dict(n=v, wr=float(cum[v - 1] / v)))
    return out


def summary(tag, n, wr, target_n=2500, target_wr=0.60):
    ok = "OK " if (n >= target_n and wr >= target_wr) else "   "
    print(f"  {ok}{tag:<34} n={n:>5}  wr={wr*100:6.2f}%")
