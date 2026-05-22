"""
Fetch ETH/USD 1h from Bitstamp and add cross-pair features to parquet.

Features added :
  eth_close_ret           : ETH 1h return (most recent closed = at T+1h-1h)
  eth_close_ret_4h        : ETH 4h return
  eth_lead                : ETH return at the SAME 1h timestamp as BTC (i.e.
                            ETH ret during the bar that just closed at T+1h).
                            ETH often leads BTC by minutes.
  eth_btc_div             : ETH return - BTC return (divergence)
  eth_vol_ratio           : ETH volume_ratio at decision time

This adds CROSS-PAIR signal orthogonal to BTC's own price action.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
import time

import numpy as np
import pandas as pd
import requests


ROOT = Path(__file__).resolve().parent
PARQUET_PATH = ROOT / "DB" / "btc_multi_tf_1h.parquet"

BITSTAMP_URL = "https://www.bitstamp.net/api/v2/ohlc/ethusd/"
LIMIT = 1000
INTERVAL_1H_SEC = 3600
INTERVAL_1H_MS = 3600 * 1000
INTERVAL_4H_MS = 4 * INTERVAL_1H_MS
DAYS_BACK = 5000
WORKERS = 16

ATR_PERIOD = 14
VOLUME_AVG_WINDOW = 20


def fetch_window(start_sec, end_sec):
    params = {"step": INTERVAL_1H_SEC, "limit": LIMIT,
              "start": start_sec, "end": end_sec}
    for _ in range(5):
        try:
            r = requests.get(BITSTAMP_URL, params=params, timeout=20)
            if r.status_code in (429,) or r.status_code >= 500:
                time.sleep(2)
                continue
            r.raise_for_status()
            return r.json().get("data", {}).get("ohlc", [])
        except Exception:
            time.sleep(1)
    return []


def fetch_all_eth_1h():
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = int((datetime.now(timezone.utc) - timedelta(days=DAYS_BACK)).timestamp() * 1000)
    window_span_ms = INTERVAL_1H_MS * LIMIT
    windows = []
    w = start_ms
    while w < now_ms:
        windows.append((w // 1000, min(w + window_span_ms - 1, now_ms) // 1000))
        w += window_span_ms
    print(f"  ETH 1h: fetching {len(windows)} windows...")

    all_rows = []
    completed = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(fetch_window, ws, we) for ws, we in windows]
        for fut in as_completed(futs):
            for c in fut.result():
                all_rows.append({
                    "timestamp_ms": int(c["timestamp"]) * 1000,
                    "open": float(c["open"]), "high": float(c["high"]),
                    "low": float(c["low"]), "close": float(c["close"]),
                    "volume": float(c["volume"]),
                })
            completed += 1
            if completed % 20 == 0:
                print(f"    {completed}/{len(windows)} - {len(all_rows):,} bars")
    df = pd.DataFrame(all_rows).drop_duplicates(subset="timestamp_ms").sort_values("timestamp_ms").reset_index(drop=True)
    return df


def compute_eth_features(df):
    close_prev = df["close"].shift(1)
    out = pd.DataFrame({"timestamp_ms": df["timestamp_ms"].astype("int64")})
    out["close_ret"] = (df["close"] - close_prev) / close_prev
    out["close_ret_4h"] = (df["close"] - df["close"].shift(4)) / df["close"].shift(4)
    vol_avg = df["volume"].rolling(VOLUME_AVG_WINDOW, min_periods=VOLUME_AVG_WINDOW).mean()
    out["volume_ratio"] = df["volume"] / vol_avg
    return out


def main():
    if not PARQUET_PATH.exists():
        raise FileNotFoundError(f"{PARQUET_PATH} not found")
    print(f"Loading {PARQUET_PATH}...")
    btc = pd.read_parquet(PARQUET_PATH)
    print(f"  BTC: {len(btc):,} rows")

    print("\nFetching ETH/USD 1h from Bitstamp...")
    t0 = time.time()
    eth_raw = fetch_all_eth_1h()
    print(f"  {len(eth_raw):,} ETH bars in {time.time() - t0:.1f}s")

    eth_feat = compute_eth_features(eth_raw)
    print(f"  ETH features: {len(eth_feat.columns)} cols")

    times_btc = btc["timestamp_ms"].to_numpy(dtype=np.int64)
    times_eth = eth_feat["timestamp_ms"].to_numpy(dtype=np.int64)
    # decision_time = open_btc + 1h. Same 1h bar in ETH closes at decision_time.
    # ETH bar with open_time = decision_time - 1h = open_btc closes at decision time.
    target = times_btc + INTERVAL_1H_MS - INTERVAL_1H_MS  # = times_btc
    indices = np.searchsorted(times_eth, target, side="right") - 1
    valid = indices >= 0

    eth_close_ret = np.full(len(btc), np.nan, dtype=float)
    eth_close_ret_4h = np.full(len(btc), np.nan, dtype=float)
    eth_vol_ratio = np.full(len(btc), np.nan, dtype=float)
    if valid.any():
        eth_close_ret[valid] = eth_feat["close_ret"].to_numpy(dtype=float)[indices[valid]]
        eth_close_ret_4h[valid] = eth_feat["close_ret_4h"].to_numpy(dtype=float)[indices[valid]]
        eth_vol_ratio[valid] = eth_feat["volume_ratio"].to_numpy(dtype=float)[indices[valid]]

    btc["eth_close_ret"] = eth_close_ret
    btc["eth_close_ret_4h"] = eth_close_ret_4h
    btc["eth_vol_ratio"] = eth_vol_ratio
    # BTC-ETH divergence : BTC's close_ret already in btc
    btc["eth_btc_div"] = btc["eth_close_ret"] - btc["close_ret"]

    print(f"\n  Added 4 ETH cross-features. Valid rows : {int(valid.sum()):,}")
    for c in ("eth_close_ret", "eth_close_ret_4h", "eth_vol_ratio", "eth_btc_div"):
        v = btc[c].dropna()
        print(f"    {c}: n_valid={len(v):,} min={v.min():.5f} max={v.max():.5f} mean={v.mean():.5f}")

    btc.to_parquet(PARQUET_PATH, index=False)
    print(f"\nWrote back {PARQUET_PATH}")


if __name__ == "__main__":
    main()
