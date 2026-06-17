"""Rich ETH/USD feature engineering — ETH is the one external source that
lifts the model's honest frontier. Builds ETH features on 15m + resampled
1h/4h, all aligned no-lookahead to the BTC 15m grid. Columns prefixed `ef_`.
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent / "data"
MS15 = 15 * 60 * 1000
STEP = {"15m": MS15, "1h": 4 * MS15, "4h": 16 * MS15}


def _ema(s, span):
    return s.ewm(span=span, adjust=False).mean()


def _rsi(close, n):
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


def _resample(df, factor):
    span = factor * MS15
    key = (df["timestamp_ms"] // span) * span
    agg = df.groupby(key).agg(open=("open", "first"), high=("high", "max"),
                              low=("low", "min"), close=("close", "last"),
                              volume=("volume", "sum"))
    return agg.reset_index().rename(columns={"index": "timestamp_ms"})


def _tf_feats(df, tf):
    o = df["open"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    c = df["close"].astype(float)
    cp = c.shift(1)
    cr = (c - cp) / cp
    vol = df["volume"].astype(float)
    sfx = f"_{tf}" if tf != "15m" else ""
    f = {}
    for n in (1, 2, 4, 8, 16, 48, 96):
        f[f"ef_ret{n}{sfx}"] = (c / c.shift(n) - 1).to_numpy()
    f[f"ef_rsi14{sfx}"] = _rsi(c, 14)
    f[f"ef_rsi48{sfx}"] = _rsi(c, 48)
    vm = vol.rolling(96, min_periods=96).mean()
    vs = vol.rolling(96, min_periods=96).std()
    f[f"ef_vol_z{sfx}"] = ((vol - vm) / vs.replace(0, np.nan)).to_numpy()
    for n in (24, 96, 384):
        lo = l.rolling(n, min_periods=n).min()
        hi = h.rolling(n, min_periods=n).max()
        f[f"ef_rangepos{n}{sfx}"] = ((c - lo) / (hi - lo).replace(0, np.nan)).to_numpy()
    f[f"ef_streak{sfx}"] = _streak(cr)
    for n in (50, 200):
        f[f"ef_dist_ema{n}{sfx}"] = ((c - _ema(c, n)) / c).to_numpy()
    rv_s = cr.rolling(12, min_periods=12).std()
    rv_l = cr.rolling(96, min_periods=96).std()
    f[f"ef_rv_ratio{sfx}"] = (rv_s / rv_l.replace(0, np.nan)).to_numpy()
    # MACD
    macd = (_ema(c, 12) - _ema(c, 26)) / c
    f[f"ef_macd{sfx}"] = macd.to_numpy()
    f[f"ef_macd_hist{sfx}"] = (macd - _ema(macd, 9)).to_numpy()
    # BB position
    mid = c.rolling(20, min_periods=20).mean()
    sd = c.rolling(20, min_periods=20).std(ddof=0)
    f[f"ef_bb_pos{sfx}"] = ((c - (mid - 2 * sd)) / (4 * sd).replace(0, np.nan)).to_numpy()
    # wicks + body
    rng = (h - l).replace(0, np.nan)
    f[f"ef_upwick{sfx}"] = (((h - np.maximum(o, c)) / rng).clip(0, 1)).to_numpy()
    f[f"ef_lowick{sfx}"] = (((np.minimum(o, c) - l) / rng).clip(0, 1)).to_numpy()
    f[f"ef_body{sfx}"] = ((c - o) / cp).to_numpy()
    return f


def _align_exact(base_ts, src_ts, vals):
    idx = np.searchsorted(src_ts, base_ts)
    out = np.full(len(base_ts), np.nan)
    clip = np.clip(idx, 0, len(src_ts) - 1)
    valid = (idx < len(src_ts)) & (src_ts[clip] == base_ts)
    out[valid] = vals[idx[valid]]
    return out


def _join_nolookahead(base_ts, src_ts, vals, tf):
    target = base_ts + MS15 - STEP[tf]
    idx = np.searchsorted(src_ts, target, side="right") - 1
    valid = idx >= 0
    out = np.full(len(base_ts), np.nan)
    out[valid] = vals[idx[valid]]
    return out


def build_eth(base_ts, verbose=True):
    """DataFrame keyed by timestamp_ms (= base_ts) with ef_ columns."""
    base_ts = np.asarray(base_ts, dtype=np.int64)
    eth = pd.read_parquet(DATA / "eth_15m.parquet").sort_values(
        "timestamp_ms").reset_index(drop=True)
    cols = {"timestamp_ms": base_ts}

    # 15m: concurrent bar closes at decision time -> exact align
    f15 = _tf_feats(eth, "15m")
    ets = eth["timestamp_ms"].to_numpy(np.int64)
    for k, v in f15.items():
        cols[k] = _align_exact(base_ts, ets, np.asarray(v, float))

    # 1h / 4h resampled, no-lookahead joined
    for tf, factor in (("1h", 4), ("4h", 16)):
        rs = _resample(eth, factor)
        ff = _tf_feats(rs, tf)
        rts = rs["timestamp_ms"].to_numpy(np.int64)
        for k, v in ff.items():
            cols[k] = _join_nolookahead(base_ts, rts, np.asarray(v, float), tf)

    res = pd.DataFrame(cols)
    if verbose:
        print(f"build_eth: {len(res):,} rows x {len(res.columns)-1} ETH features")
    return res


ETH_GROUPS = {
    "e15_mom": ["ef_ret1", "ef_ret2", "ef_ret4", "ef_ret8", "ef_ret16",
                "ef_ret48", "ef_ret96", "ef_streak", "ef_macd", "ef_macd_hist"],
    "e15_ind": ["ef_rsi14", "ef_rsi48", "ef_bb_pos", "ef_rangepos24",
                "ef_rangepos96", "ef_rangepos384", "ef_dist_ema50",
                "ef_dist_ema200", "ef_rv_ratio"],
    "e15_micro": ["ef_vol_z", "ef_upwick", "ef_lowick", "ef_body"],
    "e1h": ["ef_ret1_1h", "ef_ret4_1h", "ef_ret16_1h", "ef_rsi14_1h",
            "ef_streak_1h", "ef_macd_1h", "ef_rangepos96_1h", "ef_dist_ema50_1h",
            "ef_rv_ratio_1h", "ef_vol_z_1h"],
    "e4h": ["ef_ret1_4h", "ef_ret4_4h", "ef_rsi14_4h", "ef_streak_4h",
            "ef_macd_4h", "ef_rangepos96_4h", "ef_dist_ema50_4h"],
}
