"""add_eth_features.py -- patch: add ETH/USD exogenous features to modele_down.

Adds 14 ETH 15m features to modele_down's master parquet, exact-aligned
(no-lookahead) onto each BTC 15m bar. ETH's concurrent 15m bar shares the
BTC bar's open_time and closes at the same decision time T = open_15m + 15m,
so the exact-timestamp join is legitimate -- the ETH bar is provably closed
at T, exactly like BTC's own just-closed base-15m features.

Why ETH: walk-forward over 6 non-overlapping yearly windows (~15k pooled
bets, SE ~0.4pp) lifts modele_down's honest pooled WR from 57.24% to 58.66%
-- and every single window improves. ETH is the one exogenous source that
genuinely raises the model's ranking frontier.

Features (all on the ETH 15m series, prefixed `ef_`):
    ef_ret1 ef_ret2 ef_ret4 ef_ret8 ef_ret16 ef_ret48 ef_ret96   (momentum)
    ef_streak ef_macd ef_macd_hist                               (trend)
    ef_vol_z ef_upwick ef_lowick ef_body                         (micro)

Idempotent: every existing `ef_` column is dropped and rebuilt. A one-off
<parquet>.bak is kept (originals stay recoverable).

Data: eth_data/eth_15m.parquet (ETH/USD 15m OHLCV, 2017-08 -> present).
BTC bars before ETH history get NaN -- XGBoost routes NaN natively, and the
test/val folds (2020+) are fully ETH-covered so api.py drops no rows.

Re-running modele_down/data.py rebuilds the parquet WITHOUT these columns;
re-run this script afterwards.

Usage:
    cd backtest/modele_updown/15min/modele_down
    python add_eth_features.py
    python train.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PKG = Path(__file__).resolve().parent
ETH_PARQUET = _PKG / "eth_data" / "eth_15m.parquet"
BTC_PARQUET = _PKG / "DB" / "btc_multi_tf_15m.parquet"

# The 14 ETH features, in a fixed order (= eth_best_cols.txt).
ETH_FEATURES = [
    "ef_ret1", "ef_ret2", "ef_ret4", "ef_ret8", "ef_ret16", "ef_ret48",
    "ef_ret96", "ef_streak", "ef_macd", "ef_macd_hist", "ef_vol_z",
    "ef_upwick", "ef_lowick", "ef_body",
]


def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def _streak(close_ret: pd.Series) -> np.ndarray:
    """Signed run length: +k after k green ETH bars, -k after k red bars."""
    sign = np.sign(close_ret.to_numpy())
    s = pd.Series(sign)
    grp = (s != s.shift()).cumsum()
    run = s.groupby(grp).cumcount().to_numpy() + 1
    return run * sign


def compute_eth_features(eth: pd.DataFrame) -> pd.DataFrame:
    """ETH 15m OHLCV -> DataFrame[timestamp_ms + 14 ef_ feature columns].

    Identical math to the walk-forward-validated optimize/ethfeats.py 15m
    branch, so the patched parquet reproduces the +ETH walk-forward result.
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

    # momentum -- close-to-close returns over several horizons
    for n in (1, 2, 4, 8, 16, 48, 96):
        f[f"ef_ret{n}"] = (c / c.shift(n) - 1.0).to_numpy()

    # trend
    f["ef_streak"] = _streak(cr)
    macd = (_ema(c, 12) - _ema(c, 26)) / c
    f["ef_macd"] = macd.to_numpy()
    f["ef_macd_hist"] = (macd - _ema(macd, 9)).to_numpy()

    # microstructure
    vm = vol.rolling(96, min_periods=96).mean()
    vs = vol.rolling(96, min_periods=96).std()
    f["ef_vol_z"] = ((vol - vm) / vs.replace(0, np.nan)).to_numpy()
    rng = (h - l).replace(0, np.nan)
    f["ef_upwick"] = ((h - np.maximum(o, c)) / rng).clip(0, 1).to_numpy()
    f["ef_lowick"] = ((np.minimum(o, c) - l) / rng).clip(0, 1).to_numpy()
    f["ef_body"] = ((c - o) / cp).to_numpy()

    return pd.DataFrame(f)


def align_exact(btc_ts: np.ndarray, eth_ts: np.ndarray,
                eth_vals: np.ndarray) -> np.ndarray:
    """For each BTC 15m bar, the ETH 15m bar with the exact same open_time
    (concurrent bar, closed at the same decision time T). No-lookahead."""
    idx = np.searchsorted(eth_ts, btc_ts)
    out = np.full(len(btc_ts), np.nan, dtype=float)
    clip = np.clip(idx, 0, len(eth_ts) - 1)
    hit = (idx < len(eth_ts)) & (eth_ts[clip] == btc_ts)
    out[hit] = eth_vals[idx[hit]]
    return out


def main() -> int:
    print("=== add_eth_features.py -- 14 ETH features -> modele_down ===\n")

    if not ETH_PARQUET.exists():
        sys.exit(f"{ETH_PARQUET} introuvable.")
    if not BTC_PARQUET.exists():
        sys.exit(f"{BTC_PARQUET} introuvable -- lance d'abord data.py.")

    eth = pd.read_parquet(ETH_PARQUET)
    eth_feat = compute_eth_features(eth)
    eth_ts = eth_feat["timestamp_ms"].to_numpy(dtype=np.int64)
    print(f"  ETH 15m : {len(eth_feat):,} bougies   "
          f"{pd.Timestamp(int(eth_ts[0]), unit='ms'):%Y-%m-%d} -> "
          f"{pd.Timestamp(int(eth_ts[-1]), unit='ms'):%Y-%m-%d}")

    bak = Path(str(BTC_PARQUET) + ".bak")
    if not bak.exists():
        shutil.copy2(BTC_PARQUET, bak)
        print(f"  [bak ] {bak.name} cree (original preserve).")

    df = pd.read_parquet(BTC_PARQUET)
    df = df.drop(columns=[c for c in df.columns if c.startswith("ef_")])

    btc_ts = df["timestamp_ms"].to_numpy(dtype=np.int64)
    for col in ETH_FEATURES:
        df[col] = align_exact(btc_ts, eth_ts, eth_feat[col].to_numpy(dtype=float))

    n_cov = int((~df["ef_ret1"].isna()).sum())
    df.to_parquet(BTC_PARQUET, index=False)
    print(f"  [ok  ] +{len(ETH_FEATURES)} colonnes ETH   ({len(df):,} rows, "
          f"{len(df.columns)} cols)")
    print(f"         couverture ETH : {n_cov:,}/{len(df):,} bars "
          f"({n_cov / len(df) * 100:.1f}%) -- le reste (pre-2017) = NaN")
    print("\nTermine. Reentraine : python train.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
