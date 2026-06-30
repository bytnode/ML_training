"""External-data features: ETH cross-pair, Coinbase cross-exchange, funding
rate. Built from optimize/data/ caches, aligned no-lookahead to the 15m grid.

ETH/CB concurrent bar [open_i, open_i+15m] closes at decision time T -> usable.
Funding: latest event with fundingTime <= T.
All columns prefixed `xf_`.
"""
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
DB = HERE.parent / "DB" / "btc_multi_tf.db"
MS15 = 15 * 60 * 1000


def _ema(s, span):
    return s.ewm(span=span, adjust=False).mean()


def _rsi(close, n=14):
    d = close.diff()
    ag = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    al = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = ag / al.replace(0, np.nan)
    return ((100 - 100 / (1 + rs) - 50) / 50).to_numpy()


def _streak(cr):
    sign = np.sign(cr.to_numpy())
    s = pd.Series(sign)
    grp = (s != s.shift()).cumsum()
    return (s.groupby(grp).cumcount().to_numpy() + 1) * sign


def _align_exact(base_ts, src_ts, src_vals):
    """Map src values to base grid by exact timestamp match."""
    idx = np.searchsorted(src_ts, base_ts)
    out = np.full(len(base_ts), np.nan)
    clip = np.clip(idx, 0, len(src_ts) - 1)
    valid = (idx < len(src_ts)) & (src_ts[clip] == base_ts)
    out[valid] = src_vals[idx[valid]]
    return out


def _pair_features(df, prefix):
    """Cross-pair OHLCV feature frame for an exchange's 15m series."""
    close = df["close"].astype(float)
    cp = close.shift(1)
    cr = (close - cp) / cp
    vol = df["volume"].astype(float)
    vm = vol.rolling(96, min_periods=96).mean()
    vs = vol.rolling(96, min_periods=96).std()
    hi = df["high"].astype(float).rolling(96, min_periods=96).max()
    lo = df["low"].astype(float).rolling(96, min_periods=96).min()
    feat = {
        f"{prefix}_close_ret": cr.to_numpy(),
        f"{prefix}_ret4": (close / close.shift(4) - 1).to_numpy(),
        f"{prefix}_ret16": (close / close.shift(16) - 1).to_numpy(),
        f"{prefix}_rsi14": _rsi(close, 14),
        f"{prefix}_vol_z": ((vol - vm) / vs.replace(0, np.nan)).to_numpy(),
        f"{prefix}_rangepos96": ((close - lo) / (hi - lo).replace(0, np.nan)).to_numpy(),
        f"{prefix}_streak": _streak(cr),
        f"{prefix}_dist_ema50": ((close - _ema(close, 50)) / close).to_numpy(),
    }
    return feat


def build_external(base_ts, verbose=True):
    """Return DataFrame keyed by timestamp_ms (= base_ts) with xf_ columns."""
    base_ts = np.asarray(base_ts, dtype=np.int64)
    cols = {"timestamp_ms": base_ts}

    # ---- ETH ----
    eth_p = DATA / "eth_15m.parquet"
    if eth_p.exists():
        eth = pd.read_parquet(eth_p).sort_values("timestamp_ms").reset_index(drop=True)
        ets = eth["timestamp_ms"].to_numpy(np.int64)
        ef = _pair_features(eth, "eth")
        for k, v in ef.items():
            cols[f"xf_{k}"] = _align_exact(base_ts, ets, np.asarray(v, float))
        if verbose:
            print(f"  ETH: {len(eth):,} bars")

    # ---- Coinbase ----
    cb_p = DATA / "cb_15m.parquet"
    if cb_p.exists():
        cb = pd.read_parquet(cb_p).sort_values("timestamp_ms").reset_index(drop=True)
        cts = cb["timestamp_ms"].to_numpy(np.int64)
        cf = _pair_features(cb, "cb")
        for k, v in cf.items():
            cols[f"xf_{k}"] = _align_exact(base_ts, cts, np.asarray(v, float))
        # cross-exchange spread vs Bitstamp close
        conn = sqlite3.connect(DB)
        try:
            bs = pd.read_sql_query(
                "SELECT timestamp_ms, close FROM candles WHERE timeframe='15m' "
                "ORDER BY timestamp_ms", conn)
        finally:
            conn.close()
        bs_close = _align_exact(base_ts, bs["timestamp_ms"].to_numpy(np.int64),
                                bs["close"].to_numpy(float))
        cb_close = _align_exact(base_ts, cts, cb["close"].to_numpy(float))
        cols["xf_cb_bs_spread"] = (cb_close - bs_close) / bs_close
        if verbose:
            print(f"  CB: {len(cb):,} bars")
    elif verbose:
        print("  CB: cache not ready, skipped")

    # ---- Funding ----
    fund_p = DATA / "funding.parquet"
    if fund_p.exists():
        fund = pd.read_parquet(fund_p).sort_values("ts_ms").reset_index(drop=True)
        fts = fund["ts_ms"].to_numpy(np.int64)
        frate = fund["rate"].to_numpy(float)
        # rolling stats on the funding event series
        fr = pd.Series(frate)
        favg = fr.rolling(3, min_periods=1).mean().to_numpy()      # ~24h
        fslope = (fr - fr.shift(3)).to_numpy()                      # ~24h change
        fcum = fr.rolling(21, min_periods=1).sum().to_numpy()       # ~7d cumulative
        fz_m = fr.rolling(90, min_periods=20).mean()
        fz_s = fr.rolling(90, min_periods=20).std()
        fz = ((fr - fz_m) / fz_s.replace(0, np.nan)).to_numpy()
        # latest funding event known at decision time T = open + 15m
        decision = base_ts + MS15
        idx = np.searchsorted(fts, decision, side="right") - 1
        valid = idx >= 0
        for name, arr in (("funding_rate", frate), ("funding_avg24h", favg),
                          ("funding_slope", fslope), ("funding_cum7d", fcum),
                          ("funding_z", fz)):
            out = np.full(len(base_ts), np.nan)
            out[valid] = arr[idx[valid]]
            cols[f"xf_{name}"] = out
        if verbose:
            print(f"  funding: {len(fund):,} events")

    result = pd.DataFrame(cols)
    if verbose:
        print(f"build_external: {len(result):,} rows x {len(result.columns)-1} cols")
    return result


XF_GROUPS = {
    "x_eth": ["xf_eth_close_ret", "xf_eth_ret4", "xf_eth_ret16", "xf_eth_rsi14",
              "xf_eth_vol_z", "xf_eth_rangepos96", "xf_eth_streak",
              "xf_eth_dist_ema50"],
    "x_cb": ["xf_cb_close_ret", "xf_cb_ret4", "xf_cb_rsi14", "xf_cb_vol_z",
             "xf_cb_streak", "xf_cb_bs_spread"],
    "x_fund": ["xf_funding_rate", "xf_funding_avg24h", "xf_funding_slope",
               "xf_funding_cum7d", "xf_funding_z"],
}
