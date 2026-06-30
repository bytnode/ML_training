"""fetch_1m.py — fetch raw 1min OHLCV from Bitstamp for the trailing-stop backtest.

Standalone: does NOT touch modele_down's SQLite/parquet. Writes RAW OHLCV
(open/high/low/close/volume, not normalized) to DB_1m/btc_1m.parquet so
trailing_backtest.py can walk the intra-1h path at 1min granularity.

    cd backtest/modele_updown/1h
    python fetch_1m.py
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

PAIR = "btcusd"
STEP_SEC = 1 * 60
STEP_MS = STEP_SEC * 1000
LIMIT = 1000
DAYS_BACK = 420                       # couvre la fenetre de test 365j + marge
OUT = Path(__file__).resolve().parent / "DB_1m" / "btc_1m.parquet"
URL = f"https://www.bitstamp.net/api/v2/ohlc/{PAIR}/"
MAX_RETRIES = 5


def fetch_window(start_sec: int, end_sec: int) -> list:
    params = {"step": STEP_SEC, "limit": LIMIT, "start": start_sec, "end": end_sec}
    last = None
    for a in range(MAX_RETRIES):
        try:
            r = requests.get(URL, params=params, timeout=20)
            if r.status_code == 429 or r.status_code >= 500:
                raise requests.HTTPError(f"HTTP {r.status_code}")
            r.raise_for_status()
            return r.json().get("data", {}).get("ohlc", [])
        except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as e:
            last = e
            if a == MAX_RETRIES - 1:
                break
            time.sleep(2.0 * (2 ** a))
    raise RuntimeError(f"fetch failed: {last}")


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = int((datetime.now(timezone.utc) - timedelta(days=DAYS_BACK)).timestamp() * 1000)
    span = STEP_MS * LIMIT
    total_win = (now_ms - start_ms) // span + 1
    print(f"=== fetch_1m.py : {DAYS_BACK}j de 1min BTC/USD (Bitstamp) ===")

    rows = {}
    w = start_ms
    n = 0
    while w < now_ms:
        we = min(w + span - 1, now_ms)
        data = fetch_window(w // 1000, we // 1000)
        for c in data:
            ts = int(c["timestamp"]) * 1000
            rows[ts] = (ts, float(c["open"]), float(c["high"]),
                        float(c["low"]), float(c["close"]), float(c["volume"]))
        n += 1
        if n % 10 == 0 or w + span >= now_ms:
            print(f"  {n}/{total_win} fenetres, {len(rows):,} bougies 1m")
        w += span
        time.sleep(0.1)

    df = pd.DataFrame(sorted(rows.values()),
                      columns=["timestamp_ms", "open", "high", "low", "close", "volume"])
    df.to_parquet(OUT, index=False)
    print(f"-> {OUT} : {len(df):,} bougies 1m  "
          f"[{pd.Timestamp(int(df['timestamp_ms'].iloc[0]), unit='ms', tz='UTC')} -> "
          f"{pd.Timestamp(int(df['timestamp_ms'].iloc[-1]), unit='ms', tz='UTC')}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
