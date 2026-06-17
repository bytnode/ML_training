"""Candidate feature engineering from the SQLite `candles` table (6 timeframes
of OHLCV + normalized features). Produces a wide DataFrame keyed by the 15m
timestamp_ms, no-lookahead joined. All candidate columns are prefixed `cf_`
to avoid collisions with the master parquet's 25 features.

Higher-TF joins reuse data.py's convention: for 15m bar i with decision time
T = open_i + 15m, take the latest TF bar with open_time <= T - tf_step.
"""
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

PKG = Path(__file__).resolve().parent.parent
DB = PKG / "DB" / "btc_multi_tf.db"

MS15 = 15 * 60 * 1000
STEP_MS = {"5m": 5 * 60 * 1000, "15m": MS15, "30m": 30 * 60 * 1000,
           "1h": 60 * 60 * 1000, "4h": 4 * 60 * 60 * 1000, "1d": 24 * 60 * 60 * 1000}


def _load_tf(conn, tf):
    df = pd.read_sql_query(
        "SELECT timestamp_ms, close, volume, open_ret, high_ret, low_ret, "
        "close_ret, volume_ratio, body_ratio_close, body_ratio_avg, "
        "obv_pressure, atr_pct FROM candles WHERE timeframe=? ORDER BY timestamp_ms",
        conn, params=(tf,))
    return df.reset_index(drop=True)


def _ema(s, span):
    return s.ewm(span=span, adjust=False).mean()


def _rsi(close, n):
    d = close.diff()
    ag = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    al = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = ag / al.replace(0, np.nan)
    return ((100 - 100 / (1 + rs) - 50) / 50).to_numpy()


def _streak(close_ret):
    sign = np.sign(close_ret.to_numpy())
    s = pd.Series(sign)
    grp = (s != s.shift()).cumsum()
    run = s.groupby(grp).cumcount().to_numpy() + 1
    return run * sign


def _ohlc(df):
    close = df["close"].astype(float)
    cp = close.shift(1)
    return (cp * (1 + df["open_ret"]), cp * (1 + df["high_ret"]),
            cp * (1 + df["low_ret"]), close)


def _tf_features(df, tf):
    """Per-TF feature dict (bars <= i only). Keys suffixed by tf (except 15m)."""
    o, h, l, c = _ohlc(df)
    cr = df["close_ret"].astype(float)
    sfx = "" if tf == "15m" else f"_{tf}"
    f = {}

    for n in (3, 6, 12, 24, 48, 96, 192):
        f[f"ret{n}{sfx}"] = (c / c.shift(n) - 1.0).to_numpy()
    rv_s = cr.rolling(12, min_periods=12).std()
    rv_l = cr.rolling(96, min_periods=96).std()
    f[f"rv12{sfx}"] = rv_s.to_numpy()
    f[f"rv96{sfx}"] = rv_l.to_numpy()
    f[f"rv_ratio{sfx}"] = (rv_s / rv_l.replace(0, np.nan)).to_numpy()
    for n in (24, 96, 384):
        lo = l.rolling(n, min_periods=n).min()
        hi = h.rolling(n, min_periods=n).max()
        f[f"rangepos{n}{sfx}"] = ((c - lo) / (hi - lo).replace(0, np.nan)).to_numpy()
    for n in (50, 100, 200):
        f[f"dist_ema{n}{sfx}"] = ((c - _ema(c, n)) / c).to_numpy()
    f[f"rsi14{sfx}"] = _rsi(c, 14)
    f[f"rsi48{sfx}"] = _rsi(c, 48)
    rng = (h - l).replace(0, np.nan)
    f[f"upwick{sfx}"] = (((h - np.maximum(o, c)) / rng).clip(0, 1)).to_numpy()
    f[f"lowick{sfx}"] = (((np.minimum(o, c) - l) / rng).clip(0, 1)).to_numpy()
    f[f"streak{sfx}"] = _streak(cr)
    for n in (12, 48):
        f[f"redfrac{n}{sfx}"] = (cr < 0).rolling(n, min_periods=n).mean().to_numpy()
    vol = df["volume"].astype(float)
    vm = vol.rolling(96, min_periods=96).mean()
    vs = vol.rolling(96, min_periods=96).std()
    f[f"vol_z{sfx}"] = ((vol - vm) / vs.replace(0, np.nan)).to_numpy()
    r6 = c / c.shift(6) - 1.0
    f[f"accel6{sfx}"] = (r6 - r6.shift(6)).to_numpy()
    f[f"body_signed{sfx}"] = (df["close_ret"] - df["open_ret"]).to_numpy()
    for col in ("close_ret", "low_ret", "high_ret", "volume_ratio",
                "body_ratio_close", "body_ratio_avg", "obv_pressure", "atr_pct"):
        f[f"raw_{col}{sfx}"] = df[col].to_numpy()
    return f


def _join_nolookahead(base_ts, ts_src, feat_dict, tf):
    decision = base_ts + MS15
    target = decision - STEP_MS[tf]
    idx = np.searchsorted(ts_src, target, side="right") - 1
    valid = idx >= 0
    out = {}
    for k, src in feat_dict.items():
        vals = np.full(len(base_ts), np.nan)
        vals[valid] = np.asarray(src, float)[idx[valid]]
        out[k] = vals
    return out


def build_candidates(verbose=True):
    """DataFrame keyed by 15m timestamp_ms; all feature cols prefixed `cf_`."""
    conn = sqlite3.connect(DB)
    try:
        tf_raw = {tf: _load_tf(conn, tf) for tf in ("5m", "15m", "30m", "1h", "4h", "1d")}
    finally:
        conn.close()

    base_ts = tf_raw["15m"]["timestamp_ms"].to_numpy(np.int64)
    cols = {"timestamp_ms": base_ts}

    f15 = _tf_features(tf_raw["15m"], "15m")          # closed at decision time
    for k, v in f15.items():
        cols[f"cf_{k}"] = v
    for tf in ("5m", "30m", "1h", "4h", "1d"):
        ftf = _tf_features(tf_raw[tf], tf)
        src_ts = tf_raw[tf]["timestamp_ms"].to_numpy(np.int64)
        for k, v in _join_nolookahead(base_ts, src_ts, ftf, tf).items():
            cols[f"cf_{k}"] = v

    result = pd.DataFrame(cols)
    if verbose:
        print(f"build_candidates: {len(result):,} rows x {len(result.columns)-1} features")
    return result


# Feature groups for staged testing (cf_-prefixed names)
GROUPS = {
    "g_long15": ["cf_ret24", "cf_ret48", "cf_ret96", "cf_ret192", "cf_rv12",
                 "cf_rv96", "cf_rv_ratio", "cf_accel6", "cf_streak",
                 "cf_redfrac12", "cf_redfrac48"],
    "g_range15": ["cf_rangepos24", "cf_rangepos96", "cf_rangepos384",
                  "cf_dist_ema50", "cf_dist_ema100", "cf_dist_ema200",
                  "cf_rsi14", "cf_rsi48", "cf_upwick", "cf_lowick"],
    "g_4h": ["cf_raw_close_ret_4h", "cf_ret12_4h", "cf_ret48_4h", "cf_rsi14_4h",
             "cf_rv_ratio_4h", "cf_rangepos96_4h", "cf_dist_ema50_4h",
             "cf_raw_obv_pressure_4h", "cf_raw_atr_pct_4h",
             "cf_raw_volume_ratio_4h", "cf_streak_4h"],
    "g_1d": ["cf_raw_close_ret_1d", "cf_ret6_1d", "cf_ret24_1d", "cf_rsi14_1d",
             "cf_rangepos96_1d", "cf_dist_ema50_1d", "cf_raw_obv_pressure_1d",
             "cf_raw_atr_pct_1d", "cf_streak_1d"],
    "g_1h": ["cf_ret12_1h", "cf_ret48_1h", "cf_rsi14_1h", "cf_rv_ratio_1h",
             "cf_rangepos96_1h", "cf_dist_ema50_1h", "cf_raw_obv_pressure_1h",
             "cf_streak_1h", "cf_vol_z_1h"],
    "g_micro": ["cf_ret12_5m", "cf_rv_ratio_5m", "cf_rsi14_5m", "cf_streak_5m",
                "cf_vol_z_5m", "cf_ret24_30m", "cf_rsi14_30m", "cf_rv_ratio_30m",
                "cf_streak_30m", "cf_rangepos96_30m"],
    "g_vol15": ["cf_vol_z", "cf_body_signed", "cf_raw_atr_pct", "cf_rv12", "cf_rv96"],
}
