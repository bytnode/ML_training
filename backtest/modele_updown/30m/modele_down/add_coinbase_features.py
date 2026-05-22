"""
Fetch Coinbase BTC-USD 1h OHLCV. Add cross-exchange features:
- cb_close_ret : Coinbase 1h close_ret (compared to Bitstamp's)
- cb_volume_ratio : Coinbase 1h volume_ratio
- cb_btc_spread : (Coinbase_close - Bitstamp_close) / Bitstamp_close
                  (cross-exchange spread, often direction-predictive)
- cb_volume_div : Coinbase_volume_ratio / Bitstamp_volume_ratio
                  (volume divergence, signal of arbitrage flow)

Coinbase API : 300 candles/request, paginated.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import time

import numpy as np
import pandas as pd
import requests


ROOT = Path(__file__).resolve().parent
PARQUET_PATH = ROOT / "DB" / "btc_multi_tf_1h.parquet"

CB_URL = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
GRANULARITY_1H = 3600
LIMIT = 300
INTERVAL_1H_MS = 3600 * 1000
DAYS_BACK = 5000
VOLUME_AVG_WINDOW = 20


def fetch_window(start_iso, end_iso):
    params = {"granularity": GRANULARITY_1H, "start": start_iso, "end": end_iso}
    for _ in range(5):
        try:
            r = requests.get(CB_URL, params=params, timeout=20)
            if r.status_code == 429:
                time.sleep(2)
                continue
            r.raise_for_status()
            data = r.json()
            if not isinstance(data, list):
                return []
            return data
        except Exception as e:
            time.sleep(1)
    return []


def fetch_all_cb():
    now = datetime.now(timezone.utc)
    # CB BTC-USD started ~2015. Start fetch from 2017 to get reasonable history.
    start = datetime(2017, 1, 1, tzinfo=timezone.utc)
    span_per_window = timedelta(seconds=LIMIT * GRANULARITY_1H)

    all_rows = []
    cur = start
    n_win = 0
    while cur < now:
        cur_end = min(cur + span_per_window, now)
        cur_iso = cur.strftime("%Y-%m-%dT%H:%M:%SZ")
        end_iso = cur_end.strftime("%Y-%m-%dT%H:%M:%SZ")
        rows = fetch_window(cur_iso, end_iso)
        if rows:
            for c in rows:
                all_rows.append({
                    "ts_sec": int(c[0]),
                    "low": float(c[1]), "high": float(c[2]),
                    "open": float(c[3]), "close": float(c[4]),
                    "volume": float(c[5]),
                })
        cur = cur_end
        n_win += 1
        time.sleep(0.3)  # avoid CB rate limit
        if n_win % 20 == 0:
            print(f"  CB: {n_win} windows, {len(all_rows):,} bars, cur={cur.isoformat()}")

    df = pd.DataFrame(all_rows).drop_duplicates(subset="ts_sec").sort_values("ts_sec").reset_index(drop=True)
    df["timestamp_ms"] = df["ts_sec"] * 1000
    return df


def main():
    if not PARQUET_PATH.exists():
        raise FileNotFoundError(f"{PARQUET_PATH} not found")
    print(f"Loading {PARQUET_PATH}...")
    btc = pd.read_parquet(PARQUET_PATH)
    print(f"  BTC: {len(btc):,} rows")

    print("\nFetching Coinbase BTC-USD 1h...")
    t0 = time.time()
    cb = fetch_all_cb()
    print(f"  {len(cb):,} CB bars in {time.time() - t0:.1f}s")
    if len(cb) < 1000:
        print("  Too few CB bars, exiting")
        return

    # CB features: close_ret + volume_ratio
    cb["close_prev"] = cb["close"].shift(1)
    cb["close_ret"] = (cb["close"] - cb["close_prev"]) / cb["close_prev"]
    cb["vol_avg"] = cb["volume"].rolling(VOLUME_AVG_WINDOW, min_periods=VOLUME_AVG_WINDOW).mean()
    cb["volume_ratio"] = cb["volume"] / cb["vol_avg"]

    # Match by timestamp_ms (1h aligned)
    times_btc = btc["timestamp_ms"].to_numpy(dtype=np.int64)
    times_cb = cb["timestamp_ms"].to_numpy(dtype=np.int64)
    # Use the CB bar whose open_time exactly matches BTC open_time
    indices = np.searchsorted(times_cb, times_btc, side="left")
    valid = (indices < len(times_cb)) & (times_cb[np.clip(indices, 0, len(times_cb) - 1)] == times_btc)

    cb_close_ret = np.full(len(btc), np.nan, dtype=float)
    cb_volume_ratio = np.full(len(btc), np.nan, dtype=float)
    cb_close = np.full(len(btc), np.nan, dtype=float)
    if valid.any():
        cb_close_ret[valid] = cb["close_ret"].to_numpy(dtype=float)[indices[valid]]
        cb_volume_ratio[valid] = cb["volume_ratio"].to_numpy(dtype=float)[indices[valid]]
        cb_close[valid] = cb["close"].to_numpy(dtype=float)[indices[valid]]

    btc["cb_close_ret"] = cb_close_ret
    btc["cb_volume_ratio"] = cb_volume_ratio
    # Spread = (CB close - Bitstamp close) / Bitstamp close
    # We don't have Bitstamp close in parquet (only close_ret). Reconstruct?
    # close_ret = (close - close_prev) / close_prev → close = close_prev * (1+close_ret)
    # Recursive. Use SQLite directly.
    import sqlite3
    db_path = ROOT / "DB" / "btc_multi_tf.db"
    conn = sqlite3.connect(db_path)
    try:
        bs_df = pd.read_sql_query(
            "SELECT timestamp_ms, close FROM candles WHERE timeframe='1h' ORDER BY timestamp_ms",
            conn,
        )
    finally:
        conn.close()
    bs_times = bs_df["timestamp_ms"].to_numpy(dtype=np.int64)
    bs_close = bs_df["close"].to_numpy(dtype=float)
    bs_indices = np.searchsorted(bs_times, times_btc, side="left")
    bs_valid = (bs_indices < len(bs_times)) & (bs_times[np.clip(bs_indices, 0, len(bs_times) - 1)] == times_btc)
    bs_close_aligned = np.full(len(btc), np.nan, dtype=float)
    if bs_valid.any():
        bs_close_aligned[bs_valid] = bs_close[bs_indices[bs_valid]]

    spread = (cb_close - bs_close_aligned) / bs_close_aligned
    btc["cb_btc_spread"] = spread

    print(f"\n  Added 3 CB features. Valid rows: {int(valid.sum()):,}")
    for c in ("cb_close_ret", "cb_volume_ratio", "cb_btc_spread"):
        v = btc[c].dropna()
        if len(v) > 0:
            print(f"    {c}: n_valid={len(v):,} min={v.min():.5f} max={v.max():.5f} mean={v.mean():.5f}")

    btc.to_parquet(PARQUET_PATH, index=False)
    print(f"\nWrote back {PARQUET_PATH}")


if __name__ == "__main__":
    main()
