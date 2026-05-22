"""Fetch altcoin 15m OHLCV from Bitstamp for an altcoin-breadth signal.
Caches each to optimize/data/<coin>_15m.parquet. Resumable.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import time

import pandas as pd
import requests

OUT = Path(__file__).resolve().parent / "data"
OUT.mkdir(exist_ok=True)
SEC15 = 15 * 60
MS15 = SEC15 * 1000
COINS = ["xrpusd", "ltcusd", "solusd", "adausd", "linkusd", "dogeusd"]
DAYS_BACK = 3200


def fetch_bitstamp_15m(pair, days_back=DAYS_BACK):
    url = f"https://www.bitstamp.net/api/v2/ohlc/{pair}/"
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = now_ms - days_back * 86400 * 1000
    span = MS15 * 1000
    windows = []
    w = start_ms
    while w < now_ms:
        windows.append((w // 1000, min(w + span, now_ms) // 1000))
        w += span

    def _one(ws, we):
        for _ in range(6):
            try:
                r = requests.get(url, params={"step": SEC15, "limit": 1000,
                                              "start": ws, "end": we}, timeout=20)
                if r.status_code == 429 or r.status_code >= 500:
                    time.sleep(2)
                    continue
                r.raise_for_status()
                return r.json().get("data", {}).get("ohlc", [])
            except Exception:
                time.sleep(1)
        return []

    rows = []
    with ThreadPoolExecutor(max_workers=16) as ex:
        futs = [ex.submit(_one, ws, we) for ws, we in windows]
        for fut in as_completed(futs):
            for c in fut.result():
                rows.append({"timestamp_ms": int(c["timestamp"]) * 1000,
                             "open": float(c["open"]), "high": float(c["high"]),
                             "low": float(c["low"]), "close": float(c["close"]),
                             "volume": float(c["volume"])})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).drop_duplicates("timestamp_ms").sort_values("timestamp_ms")
    return df.reset_index(drop=True)


def main():
    for coin in COINS:
        path = OUT / f"{coin}_15m.parquet"
        if path.exists():
            print(f"{coin}: cache exists", flush=True)
            continue
        print(f"Fetching {coin} 15m...", flush=True)
        t0 = time.time()
        df = fetch_bitstamp_15m(coin)
        if len(df) < 1000:
            print(f"  {coin}: too few bars ({len(df)}), skipped", flush=True)
            continue
        df.to_parquet(path, index=False)
        print(f"  {coin}: {len(df):,} bars, "
              f"{pd.to_datetime(df['timestamp_ms'].min(), unit='ms').date()} .. "
              f"{pd.to_datetime(df['timestamp_ms'].max(), unit='ms').date()} "
              f"({time.time()-t0:.0f}s)", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
