"""Live one-bar feature builder for modele_down.

Reuses helpers from `modele_down/data.py` (compute_normalized_features,
_add_tier1_derived, _add_tier2) by sys.path injection. The pipeline mirrors
build_master_15m but runs on klines buffers instead of the SQLite DB.

Public API
----------
    build_features(klines_15m, klines_5m, klines_30m, klines_1h,
                   klines_eth_15m) -> dict
        Returns the feature dict for the LAST CLOSED 15m bar, including the
        14 exogenous ETH/USD `ef_*` features (ported from the backtest patch
        modele_down/add_eth_features.py).

Convention : klines = list of [open_time_ms, open, high, low, close, volume].
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

# sys.path injection so we can import modele_down/data.py helpers.
# __file__ = live/modele_15m/live_helpers/features_down.py
# parents: [0] live_helpers/, [1] modele_15m/. modele_down/ is a sibling of
# live_helpers/ inside this self-contained folder.
_MODELE_DOWN_DIR = str(Path(__file__).resolve().parents[1] / "modele_down")
if _MODELE_DOWN_DIR not in sys.path:
    sys.path.insert(0, _MODELE_DOWN_DIR)

from data import (  # noqa: E402  (after sys.path injection)
    compute_normalized_features,
    _add_tier1_derived,
    _add_tier2,
)


INTERVAL_MS = {
    "5m":  5  * 60 * 1000,
    "15m": 15 * 60 * 1000,
    "30m": 30 * 60 * 1000,
    "1h":  60 * 60 * 1000,
}


def _klines_to_df(klines: Sequence) -> pd.DataFrame:
    df = pd.DataFrame(klines, columns=["timestamp_ms", "open", "high", "low", "close", "volume"])
    df["timestamp_ms"] = df["timestamp_ms"].astype(np.int64)
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = df[c].astype(float)
    return df


def _last_closed_index(times: np.ndarray, T: int, interval_ms: int) -> int:
    cutoff = T - interval_ms
    idx = int(np.searchsorted(times, cutoff, side="right") - 1)
    return idx if idx >= 0 else -1


def _filter_5m_close_aligned(feat_5m: pd.DataFrame) -> pd.DataFrame:
    """The modele_down pipeline keeps only 5m bars whose open_time = T + 10min
    (i.e. they close exactly at the same instant as a 15m bar)."""
    offset_ms = feat_5m["timestamp_ms"].to_numpy(dtype=np.int64) % INTERVAL_MS["15m"]
    keep_mask = offset_ms == (10 * 60 * 1000)
    return feat_5m.loc[keep_mask].reset_index(drop=True)


def _add_lookup_features_live(df_15m: pd.DataFrame, df_other: pd.DataFrame,
                              tf_suffix: str, cols: list[str]) -> pd.DataFrame:
    """No-lookahead join of higher-TF features onto 15m.
    For each 15m row at open_time T, picks the latest other-TF bar with
    close_time <= T + 15min. Adds columns suffixed with `tf_suffix`.
    """
    other_times = df_other["timestamp_ms"].to_numpy(dtype=np.int64)
    iv = INTERVAL_MS[tf_suffix]
    T15 = df_15m["timestamp_ms"].to_numpy(dtype=np.int64) + INTERVAL_MS["15m"]
    # For each 15m T, find the latest other-TF bar with open_time + iv <= T
    cutoff = T15 - iv
    idx = np.searchsorted(other_times, cutoff, side="right") - 1
    valid = idx >= 0

    for col in cols:
        new_col = f"{col}_{tf_suffix}"
        vals = np.full(len(df_15m), np.nan, dtype=float)
        if valid.any():
            vals[valid] = df_other[col].to_numpy(dtype=float)[idx[valid]]
        df_15m[new_col] = vals
    return df_15m


# Columns from modele_down parquet (= the feature schema the model was trained on).
# 2026-05-25 cleanup: dropped volume_ratio_30m, body_ratio_close_30m, hour_sin;
# added volume_rel_30d (ported from modele_up). See backtest clean_parquet.py.
_PARQUET_COLS = [
    "high_ret", "low_ret", "close_ret",
    "high_ret_5m", "low_ret_5m", "close_ret_5m", "obv_pressure_5m",
    "low_ret_30m", "close_ret_30m",
    "low_ret_1h", "close_ret_1h",
    "dist_ema_20", "return_3",
    "macd", "macd_hist", "bb_position", "mfi_norm",
    "is_weekend", "volatility_state",
    "dist_to_high_50", "dist_to_low_50",
    "autocorr_24", "acc_4",
    "volume_rel_30d",
]

# 5 exogenous ETH/USD 15m features kept after the 2026-05-25 cleanup. Pruned
# from 14 -> 5 to remove redundancy (ef_ret1<->ef_body r=0.98, ef_macd_hist<->
# ef_ret8 r=0.84, etc.). Joined onto each BTC bar by EXACT timestamp -- the
# concurrent ETH 15m bar shares the BTC open_time and closes at decision time T.
_ETH_FEATURES = ["ef_ret2", "ef_ret8", "ef_streak", "ef_macd", "ef_body"]


def _eth_ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def _eth_streak(close_ret: pd.Series) -> np.ndarray:
    """Signed run length: +k after k green ETH bars, -k after k red bars."""
    sign = np.sign(close_ret.to_numpy())
    s = pd.Series(sign)
    grp = (s != s.shift()).cumsum()
    run = s.groupby(grp).cumcount().to_numpy() + 1
    return run * sign


def _compute_eth_features(eth: pd.DataFrame) -> pd.DataFrame:
    """ETH 15m OHLCV -> DataFrame[timestamp_ms + 14 ef_ feature columns].

    Exact port of modele_down/add_eth_features.compute_eth_features so the
    live ef_* values reproduce the patched parquet the model trained on.
    """
    eth = eth.sort_values("timestamp_ms").reset_index(drop=True)
    o = eth["open"].astype(float)
    h = eth["high"].astype(float)
    l = eth["low"].astype(float)
    c = eth["close"].astype(float)
    cp = c.shift(1)
    cr = (c - cp) / cp
    vol = eth["volume"].astype(float)

    f = {"timestamp_ms": eth["timestamp_ms"].to_numpy(dtype=np.int64)}

    # momentum -- only ef_ret2 and ef_ret8 are kept after the 2026-05-25 prune.
    for n in (2, 8):
        f[f"ef_ret{n}"] = (c / c.shift(n) - 1.0).to_numpy()

    # trend
    f["ef_streak"] = _eth_streak(cr)
    macd = (_eth_ema(c, 12) - _eth_ema(c, 26)) / c
    f["ef_macd"] = macd.to_numpy()

    # microstructure -- only ef_body retained.
    f["ef_body"] = ((c - o) / cp).to_numpy()

    return pd.DataFrame(f)


def _eth_features_for_ts(klines_eth_15m, target_ts: int) -> dict:
    """The 14 ef_* values for the ETH 15m bar at exactly `target_ts`.

    Missing ETH data (empty buffer or no concurrent bar) -> all-NaN. XGBoost
    routes NaN natively, exactly as the backtest parquet does for BTC bars
    that predate ETH history.
    """
    nan_row = {col: float("nan") for col in _ETH_FEATURES}
    if not klines_eth_15m:
        return nan_row
    eth_feat = _compute_eth_features(_klines_to_df(klines_eth_15m))
    hit = eth_feat.loc[eth_feat["timestamp_ms"] == int(target_ts)]
    if len(hit) != 1:
        return nan_row
    r = hit.iloc[0]
    return {col: float(r[col]) for col in _ETH_FEATURES}


def build_features(klines_15m, klines_5m, klines_30m, klines_1h,
                   klines_eth_15m=None, target_open_ms: int | None = None) -> dict:
    """Build the feature row for the LAST CLOSED 15m bar.

    Target-bar selection: if `target_open_ms` is given, pick the row whose
    open_time == target_open_ms (so we don't depend on the currently-open bar
    being present in the buffer); otherwise legacy positional pick last_idx=-2.
    Feature values are identical either way (all backward-looking).

    `klines_eth_15m` is the ETH/USD 15m buffer used for the 5 kept `ef_*`
    features; pass None (or an empty list) and they fall back to NaN.

    The 2026-05-25 cleanup removed `p_up_xmodel` (was in-sample leakage):
    the cross-model gating is now only applied at runtime via the
    `max_p_up_runtime` sidecar field inside live.py.
    """
    # Phase 1: compute per-TF normalized features
    feat_15 = compute_normalized_features(_klines_to_df(klines_15m))
    feat_5  = compute_normalized_features(_klines_to_df(klines_5m))
    feat_30 = compute_normalized_features(_klines_to_df(klines_30m))
    feat_1h = compute_normalized_features(_klines_to_df(klines_1h))

    feat_5 = _filter_5m_close_aligned(feat_5)

    # Phase 2: MTF joins onto 15m (same column subsets as modele_down parquet)
    feat_15 = _add_lookup_features_live(
        feat_15, feat_5, "5m",
        cols=["high_ret", "low_ret", "close_ret", "obv_pressure"],
    )
    feat_15 = _add_lookup_features_live(
        feat_15, feat_30, "30m",
        cols=["low_ret", "close_ret", "volume_ratio", "body_ratio_close"],
    )
    feat_15 = _add_lookup_features_live(
        feat_15, feat_1h, "1h",
        cols=["low_ret", "close_ret"],
    )

    # Phase 3: Tier 1 + Tier 2 (15m-derived features)
    _add_tier1_derived(feat_15)
    _add_tier2(feat_15)

    # Phase 4: dist_to_high_50 / dist_to_low_50 / autocorr_24 / acc_4 / volume_rel_30d
    close = feat_15["close"].to_numpy(dtype=float)
    high_50 = pd.Series((feat_15["high_ret"] + 1) * pd.Series(close).shift(1)).rolling(50).max()
    low_50  = pd.Series((feat_15["low_ret"]  + 1) * pd.Series(close).shift(1)).rolling(50).min()
    feat_15["dist_to_high_50"] = (close - high_50) / close
    feat_15["dist_to_low_50"]  = (close - low_50) / close
    cr = feat_15["close_ret"]
    feat_15["autocorr_24"] = cr.rolling(24).corr(cr.shift(1))
    feat_15["acc_4"] = (pd.Series(close) - 2 * pd.Series(close).shift(4) + pd.Series(close).shift(8)) / pd.Series(close).shift(8)

    # volume_rel_30d : volume relative to its 30-day rolling mean (2880 15m bars).
    # Ported from modele_up parquet on 2026-05-25; min_periods set high enough to
    # avoid noisy values on cold start (warmup config in live.py = 3200 bars).
    vol = feat_15["volume"].astype(float)
    feat_15["volume_rel_30d"] = vol / vol.rolling(2880, min_periods=2880).mean()

    # Phase 5: is_weekend + volatility_state
    ts = pd.to_datetime(feat_15["timestamp_ms"], unit="ms", utc=True)
    feat_15["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)
    atr_pct = feat_15["atr_pct"]
    atr_long = atr_pct.rolling(100, min_periods=20).mean()
    ratio = atr_pct / atr_long
    feat_15["volatility_state"] = np.where(ratio > 1.5, 2,
                                           np.where(ratio < 0.7, 0, 1)).astype(int)

    # Pick the target CLOSED bar : by timestamp if named, else legacy last_idx=-2.
    if target_open_ms is not None:
        pos = np.where(feat_15["timestamp_ms"].to_numpy() == int(target_open_ms))[0]
        if len(pos) == 0:
            raise ValueError(f"target_open_ms {target_open_ms} not found in 15m buffer.")
        row = feat_15.iloc[int(pos[-1])]
    else:
        if len(feat_15) < 2:
            raise ValueError("klines_15m too short.")
        row = feat_15.iloc[-2]
    out = {"timestamp": int(row["timestamp_ms"])}
    for c in _PARQUET_COLS:
        if c not in row.index:
            raise KeyError(f"features_down: missing column '{c}' after pipeline.")
        v = row[c]
        if c in ("is_weekend", "volatility_state"):
            out[c] = int(v)
        else:
            out[c] = float(v)
    # Exogenous ETH/USD features -- exact-timestamp join onto the target bar.
    out.update(_eth_features_for_ts(klines_eth_15m, int(row["timestamp_ms"])))
    return out


def validate_against_parquet(parquet_path, n_test_bars: int = 5, tol: float = 1e-5,
                              sqlite_path=None) -> None:
    """Recompute features for n_test_bars known parquet rows using SQLite OHLCV.

    Skips MTF columns because the SQLite stores those TFs downsampled by the
    pipeline itself, so reconstruction can't perfectly mimic the original full-
    density 5m series at validation time. We trust Bitstamp to give the full
    series in live mode.
    """
    import sqlite3
    pq_path = Path(parquet_path)
    if sqlite_path is None:
        sqlite_path = pq_path.parent / "btc_multi_tf.db"
    pq = pd.read_parquet(pq_path).sort_values("timestamp_ms").reset_index(drop=True)

    con = sqlite3.connect(sqlite_path)

    def _load(tf):
        df = pd.read_sql(
            f"SELECT timestamp_ms, close, volume, open_ret, high_ret, low_ret FROM candles WHERE timeframe='{tf}' ORDER BY timestamp_ms",
            con,
        )
        prev_close = df["close"].shift(1)
        df["open"]  = prev_close * (1 + df["open_ret"])
        df["high"]  = prev_close * (1 + df["high_ret"])
        df["low"]   = prev_close * (1 + df["low_ret"])
        return df[["timestamp_ms", "open", "high", "low", "close", "volume"]].dropna()

    sq_15m = _load("15m").values.tolist()
    sq_5m  = _load("5m").values.tolist()
    sq_30m = _load("30m").values.tolist()
    sq_1h  = _load("1h").values.tolist()
    con.close()

    test_idx = np.linspace(len(pq) - 10000, len(pq) - 100, n_test_bars).astype(int)
    failures = []
    MTF_COLS = {"high_ret_5m", "low_ret_5m", "close_ret_5m", "obv_pressure_5m",
                "low_ret_30m", "close_ret_30m", "volume_ratio_30m", "body_ratio_close_30m",
                "low_ret_1h", "close_ret_1h"}
    print(f"features_down validation: {len(test_idx)} bars, tol={tol}...")
    for i in test_idx:
        ts = int(pq.iloc[i]["timestamp_ms"])
        cut15 = [k for k in sq_15m if int(k[0]) <= ts]
        if not cut15 or int(cut15[-1][0]) != ts:
            failures.append((ts, "target not in SQLite 15m"))
            continue
        phantom = list(cut15[-1])
        phantom[0] = ts + INTERVAL_MS["15m"]
        cut15 = cut15 + [phantom]
        T = ts + INTERVAL_MS["15m"]
        cut5  = [k for k in sq_5m  if int(k[0]) + INTERVAL_MS["5m"]  <= T]
        cut30 = [k for k in sq_30m if int(k[0]) + INTERVAL_MS["30m"] <= T]
        cut1h = [k for k in sq_1h  if int(k[0]) + INTERVAL_MS["1h"]  <= T]

        try:
            feats = build_features(cut15, cut5, cut30, cut1h)
        except Exception as e:
            failures.append((ts, f"build_features raised: {e}"))
            continue
        row = pq.iloc[i]
        for c in _PARQUET_COLS:
            if c in MTF_COLS:
                continue
            if c not in row.index:
                continue
            v_live = feats[c]
            v_pq = row[c]
            if pd.isna(v_pq):
                continue
            diff = abs(v_live - v_pq)
            rel = diff / max(abs(v_pq), 1e-9)
            if c in ("is_weekend", "volatility_state"):
                if int(v_live) != int(v_pq):
                    failures.append((ts, f"{c}: live={v_live} pq={v_pq}"))
            elif diff > tol and rel > tol:
                failures.append((ts, f"{c}: live={v_live:.6e} pq={v_pq:.6e} diff={diff:.2e}"))

    if failures:
        msg = f"features_down.validate: {len(failures)} mismatches\n"
        for ts, err in failures[:20]:
            msg += f"  ts={ts}: {err}\n"
        raise AssertionError(msg)
    print(f"  OK: {len(test_idx)} bars match within tol={tol}.")


if __name__ == "__main__":
    # Dev validator only - the parquet is not shipped in this self-contained
    # folder; point it at the backtest package if you want to run it.
    PARQUET = Path(__file__).resolve().parents[1] / "modele_down" / "DB" / "btc_multi_tf_15m.parquet"
    validate_against_parquet(PARQUET, n_test_bars=5, tol=1e-5)
