"""
Fetch Binance Futures BTCUSDT funding rate history and add as features.

Funding rate is published every 8h. We:
1. Fetch all funding history from Binance (public, no auth needed)
2. For each 1h bar in parquet, look up the most-recent funding rate KNOWN AT
   decision time (= open_1h + 1h)
3. Compute rolling stats : current, 24h avg, 24h slope
4. Add columns to parquet

Funding is a strong directional signal :
  - High positive funding (longs paying shorts) = often near tops (bear)
  - Negative funding = often near bottoms (bull)
"""
from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import pandas as pd
import requests


ROOT = Path(__file__).resolve().parent
PARQUET_PATH = ROOT / "DB" / "btc_multi_tf_1h.parquet"

OKX_FUNDING_URL = "https://www.okx.com/api/v5/public/funding-rate-history"
LIMIT = 100  # OKX max per request

INTERVAL_1H_MS = 60 * 60 * 1000


def fetch_funding_history(inst_id="BTC-USDT-SWAP", start_ms=None, end_ms=None):
    """Page through OKX funding rate history (newest first, page back)."""
    all_rows = []
    cur_after = None  # 'after' = older than this ms (page backwards)
    while True:
        params = {"instId": inst_id, "limit": LIMIT}
        if cur_after is not None:
            params["after"] = cur_after
        try:
            r = requests.get(OKX_FUNDING_URL, params=params, timeout=20)
            if r.status_code != 200:
                print(f"  HTTP {r.status_code}: {r.text[:200]}")
                if r.status_code == 429:
                    time.sleep(5)
                    continue
                break
            payload = r.json()
            data = payload.get("data", [])
        except Exception as e:
            print(f"  error: {e}")
            time.sleep(2)
            continue
        if not data:
            break
        # Filter by time range if requested
        for item in data:
            ts = int(item["fundingTime"])
            if start_ms is not None and ts < start_ms:
                continue
            if end_ms is not None and ts > end_ms:
                continue
            all_rows.append({"fundingTime": ts, "fundingRate": item["fundingRate"]})
        oldest_ts = int(data[-1]["fundingTime"])
        if start_ms is not None and oldest_ts < start_ms:
            break
        cur_after = oldest_ts
        time.sleep(0.15)
        if len(all_rows) % 2000 < LIMIT:
            print(f"  Fetched {len(all_rows):,} funding events, oldest={pd.to_datetime(oldest_ts, unit='ms', utc=True)}")
        if len(data) < LIMIT:
            break
    return all_rows


def main():
    if not PARQUET_PATH.exists():
        raise FileNotFoundError(f"{PARQUET_PATH} not found")
    print(f"Loading {PARQUET_PATH}...")
    df = pd.read_parquet(PARQUET_PATH)
    print(f"  {len(df):,} rows, {len(df.columns)} cols")

    # OKX BTC-USDT-SWAP funding exists since 2020-03 ish.
    start_ms = int(df["timestamp_ms"].min())
    end_ms = int(df["timestamp_ms"].max())
    okx_min_ms = int(pd.Timestamp("2020-01-01", tz="UTC").timestamp() * 1000)
    fetch_start = max(start_ms, okx_min_ms)

    print(f"Fetching OKX BTC-USDT-SWAP funding from {pd.to_datetime(fetch_start, unit='ms', utc=True)}...")
    raw = fetch_funding_history("BTC-USDT-SWAP", fetch_start, end_ms)
    print(f"  Got {len(raw):,} funding events")
    if not raw:
        print("  No funding data, exiting")
        return

    funding_df = pd.DataFrame([{
        "ts_ms": int(r["fundingTime"]),
        "rate": float(r["fundingRate"]) if r["fundingRate"] else np.nan,
    } for r in raw]).dropna().drop_duplicates(subset="ts_ms").sort_values("ts_ms").reset_index(drop=True)
    print(f"  Unique funding events: {len(funding_df):,}")
    print(f"  Rate: min={funding_df['rate'].min():.6f} max={funding_df['rate'].max():.6f} mean={funding_df['rate'].mean():.6f}")

    # For each 1h bar, find the MOST RECENT funding KNOWN at decision time.
    # decision_time = open_1h + 1h. We use funding events with ts <= decision_time.
    times_1h = df["timestamp_ms"].to_numpy(dtype=np.int64)
    decision_times = times_1h + INTERVAL_1H_MS
    fund_times = funding_df["ts_ms"].to_numpy(dtype=np.int64)
    fund_rates = funding_df["rate"].to_numpy(dtype=float)

    # searchsorted to find latest funding <= decision_time
    indices = np.searchsorted(fund_times, decision_times, side="right") - 1
    valid = indices >= 0

    funding_current = np.full(len(df), np.nan, dtype=float)
    funding_current[valid] = fund_rates[indices[valid]]

    # 24h average (last 3 funding events = 24h)
    funding_avg_24h = np.full(len(df), np.nan, dtype=float)
    funding_slope = np.full(len(df), np.nan, dtype=float)
    for i in range(len(df)):
        if not valid[i]:
            continue
        idx = indices[i]
        start_idx = max(0, idx - 2)
        recent_rates = fund_rates[start_idx:idx + 1]
        funding_avg_24h[i] = recent_rates.mean()
        if len(recent_rates) >= 2:
            funding_slope[i] = recent_rates[-1] - recent_rates[0]

    df["funding_current"] = funding_current
    df["funding_avg_24h"] = funding_avg_24h
    df["funding_slope_24h"] = funding_slope

    print(f"\n  Added 3 funding columns")
    for c in ("funding_current", "funding_avg_24h", "funding_slope_24h"):
        v = df[c].dropna()
        print(f"    {c}: n_valid={len(v):,} min={v.min():.6f} max={v.max():.6f} mean={v.mean():.6f}")

    df.to_parquet(PARQUET_PATH, index=False)
    print(f"\nWrote back {PARQUET_PATH}")


if __name__ == "__main__":
    main()
