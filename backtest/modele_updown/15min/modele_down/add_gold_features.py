"""add_gold_features.py -- patch: add gold (XAU) close-return features.

Adds, to each master parquet, gold's close-to-close return at the
timeframes selected PER MODEL in PARQUET_GOLD below, joined no-lookahead
onto each BTC 15m bar (the last gold bar of that timeframe provably
closed at the decision time T = open_15m + 15m).

Current selection (from the gold ablation analysis) :
    modele_down : no gold feature   -- gold always degraded its WR
    modele_up   : gold_close_ret_15m only  -- mild +0.2 pt; the 5m gold
                  candle was tested and hurt, so it is excluded

Patches BOTH parquets in place. A one-off <parquet>.bak is kept so the
originals stay recoverable; re-running is idempotent (every existing
gold_* column is dropped and rebuilt from the current selection -- so an
empty selection cleanly removes all gold columns).

Notes
-----
* Gold data: 15min/gold_data/XAU_<tf>_data.csv -- "Date;O;H;L;C;V",
  ';'-separated, dates "YYYY.MM.DD HH:MM". Read as UTC.
* Gold data ends 2026-01-30 -> BTC bars past that get gold = 0.0 (neutral,
  never NaN: a NaN feature would make the prediction path drop the row).
* Re-running modele_down/data.py rebuilds its parquet WITHOUT gold; re-run
  this script afterwards. modele_up has no data.py.

Usage:
    cd backtest/modele_updown/15min/modele_down
    python add_gold_features.py
    # then retrain both models: python train.py  (here and in ../modele_up)
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_15MIN = Path(__file__).resolve().parent.parent
GOLD_DIR = _15MIN / "gold_data"

INTERVAL_15M_MS = 15 * 60 * 1000

# Catalog of available gold timeframes: label -> (CSV filename tag, step ms).
GOLD_TF_CATALOG: dict[str, tuple[str, int]] = {
    "5m":  ("5m",  5 * 60 * 1000),
    "15m": ("15m", 15 * 60 * 1000),
    "1h":  ("1h",  60 * 60 * 1000),
    "4h":  ("4h",  4 * 60 * 60 * 1000),
    "1d":  ("1d",  24 * 60 * 60 * 1000),
}

# Per-model selection: which gold timeframes each parquet receives.
# Empty list -> that parquet gets NO gold feature (existing ones removed).
PARQUET_GOLD: dict[Path, list[str]] = {
    _15MIN / "modele_down" / "DB" / "btc_multi_tf_15m.parquet": [],
    _15MIN / "modele_up"   / "DB" / "btc_multi_tf_15m.parquet": ["15m"],
}


def load_gold_close_ret(tf_tag: str) -> tuple[np.ndarray, np.ndarray]:
    """Read XAU_<tf>_data.csv -> (timestamps_ms sorted asc, close_ret aligned).

    close_ret[i] = (close[i] - close[i-1]) / close[i-1] ; close_ret[0] = NaN.
    Timestamps are parsed naive and treated as UTC.
    """
    path = GOLD_DIR / f"XAU_{tf_tag}_data.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} introuvable.")
    df = pd.read_csv(path, sep=";", usecols=["Date", "Close"])
    ts = pd.to_datetime(df["Date"], format="%Y.%m.%d %H:%M")
    ts_ms = ts.to_numpy().astype("datetime64[ms]").astype("int64")
    close = df["Close"].astype(float).to_numpy()

    order = np.argsort(ts_ms, kind="stable")
    ts_ms, close = ts_ms[order], close[order]
    close_prev = np.concatenate([[np.nan], close[:-1]])
    close_ret = (close - close_prev) / close_prev
    return ts_ms, close_ret


def gold_feature(btc_times_ms: np.ndarray, gold_ts: np.ndarray,
                 gold_ret: np.ndarray, tf_step_ms: int) -> np.ndarray:
    """No-lookahead join. For each BTC 15m bar, return the close_ret of the
    last gold bar (of this TF) provably closed at T = open_15m + 15m. Bars
    past the end of the gold series get 0.0 (neutral) -- never NaN."""
    decision = btc_times_ms.astype(np.int64) + INTERVAL_15M_MS
    target = decision - tf_step_ms
    idx = np.searchsorted(gold_ts, target, side="right") - 1

    out = np.zeros(len(btc_times_ms), dtype=float)
    ok = idx >= 0
    out[ok] = gold_ret[np.clip(idx, 0, len(gold_ret) - 1)][ok]
    # past the last gold bar -> no fresh gold data -> neutral 0.0
    out[decision > (gold_ts[-1] + tf_step_ms)] = 0.0
    # residual NaN (gold bar 0 has no close_prev) -> 0.0
    out[np.isnan(out)] = 0.0
    return out


def patch_parquet(parquet_path: Path, tf_labels: list[str],
                  gold_cache: dict) -> None:
    if not parquet_path.exists():
        print(f"  [skip] {parquet_path} introuvable.")
        return

    bak = Path(str(parquet_path) + ".bak")
    if not bak.exists():
        shutil.copy2(parquet_path, bak)
        print(f"  [bak ] {bak.name} cree (original preserve).")

    df = pd.read_parquet(parquet_path)
    # idempotent : retire toute colonne gold deja presente
    df = df.drop(columns=[c for c in df.columns if c.startswith("gold_")])

    btc_times = df["timestamp_ms"].to_numpy(dtype=np.int64)
    for tf in tf_labels:
        gold_ts, gold_ret = gold_cache[tf]
        df[f"gold_close_ret_{tf}"] = gold_feature(
            btc_times, gold_ts, gold_ret, GOLD_TF_CATALOG[tf][1])

    pkg = parquet_path.parent.parent.name
    df.to_parquet(parquet_path, index=False)
    if tf_labels:
        names = ", ".join(f"gold_close_ret_{t}" for t in tf_labels)
        print(f"  [ok  ] {pkg:<12} -> {names}   ({len(df):,} rows)")
    else:
        print(f"  [ok  ] {pkg:<12} -> aucune feature or (toutes retirees)"
              f"   ({len(df):,} rows)")


def main() -> int:
    print("=== add_gold_features.py -- features OR (close_ret) ===\n")

    needed = sorted({tf for tfs in PARQUET_GOLD.values() for tf in tfs})
    gold_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for tf in needed:
        ts, ret = load_gold_close_ret(GOLD_TF_CATALOG[tf][0])
        gold_cache[tf] = (ts, ret)
        first = pd.Timestamp(int(ts[0]), unit="ms")
        last = pd.Timestamp(int(ts[-1]), unit="ms")
        print(f"  gold {tf:<3}: {len(ts):>9,} bougies   "
              f"{first:%Y-%m-%d} -> {last:%Y-%m-%d}")
    if needed:
        print()

    for pq, tfs in PARQUET_GOLD.items():
        patch_parquet(pq, tfs, gold_cache)

    print("\nTermine. Reentraine les deux modeles :")
    print("  cd modele_down && python train.py")
    print("  cd ../modele_up && python train.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
