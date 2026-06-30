"""Fetch external data sources at 15m resolution and cache to optimize/data/.
  - ETH/USD 15m   from Bitstamp
  - Coinbase BTC-USD 15m
  - OKX BTC-USDT-SWAP funding rate history
Resumable: skips a source whose cache file already exists.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
import time

import numpy as np
import pandas as pd
import requests

OUT = Path(__file__).resolve().parent / "data"
OUT.mkdir(exist_ok=True)

MS15 = 15 * 60 * 1000
SEC15 = 15 * 60


# ---------------- ETH 15m (Bitstamp) ----------------
def fetch_bitstamp_15m(pair, days_back=5200):
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
        for _ in range(5):
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
    df = pd.DataFrame(rows).drop_duplicates("timestamp_ms").sort_values("timestamp_ms")
    return df.reset_index(drop=True)


# ---------------- Coinbase BTC-USD 15m ----------------
def fetch_coinbase_15m():
    url = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
    now = datetime.now(timezone.utc)
    cur = datetime(2016, 1, 1, tzinfo=timezone.utc)
    span = timedelta(seconds=300 * SEC15)
    rows = []
    n = 0
    while cur < now:
        ce = min(cur + span, now)
        ok = False
        for _ in range(5):
            try:
                r = requests.get(url, params={"granularity": SEC15,
                                              "start": cur.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                              "end": ce.strftime("%Y-%m-%dT%H:%M:%SZ")},
                                 timeout=20)
                if r.status_code == 429:
                    time.sleep(1.5)
                    continue
                r.raise_for_status()
                data = r.json()
                if isinstance(data, list):
                    for c in data:
                        rows.append({"timestamp_ms": int(c[0]) * 1000,
                                     "low": float(c[1]), "high": float(c[2]),
                                     "open": float(c[3]), "close": float(c[4]),
                                     "volume": float(c[5])})
                ok = True
                break
            except Exception:
                time.sleep(1)
        cur = ce
        n += 1
        time.sleep(0.27)
        if n % 200 == 0:
            print(f"  CB: {n} windows, {len(rows):,} bars, cur={cur.date()}", flush=True)
    df = pd.DataFrame(rows).drop_duplicates("timestamp_ms").sort_values("timestamp_ms")
    return df.reset_index(drop=True)


# ---------------- OKX funding ----------------
def fetch_okx_funding(inst="BTC-USDT-SWAP"):
    url = "https://www.okx.com/api/v5/public/funding-rate-history"
    rows = []
    after = None
    while True:
        params = {"instId": inst, "limit": 100}
        if after is not None:
            params["after"] = after
        try:
            r = requests.get(url, params=params, timeout=20)
            if r.status_code == 429:
                time.sleep(3)
                continue
            r.raise_for_status()
            data = r.json().get("data", [])
        except Exception as e:
            print(f"  funding err: {e}", flush=True)
            time.sleep(2)
            continue
        if not data:
            break
        for it in data:
            rows.append({"ts_ms": int(it["fundingTime"]),
                         "rate": float(it["fundingRate"]) if it["fundingRate"] else np.nan})
        after = int(data[-1]["fundingTime"])
        time.sleep(0.12)
        if len(data) < 100:
            break
    df = pd.DataFrame(rows).dropna().drop_duplicates("ts_ms").sort_values("ts_ms")
    return df.reset_index(drop=True)


def main():
    eth_p = OUT / "eth_15m.parquet"
    cb_p = OUT / "cb_15m.parquet"
    fund_p = OUT / "funding.parquet"

    if not eth_p.exists():
        print("Fetching ETH/USD 15m from Bitstamp...", flush=True)
        t0 = time.time()
        eth = fetch_bitstamp_15m("ethusd")
        eth.to_parquet(eth_p, index=False)
        print(f"  ETH: {len(eth):,} bars, {eth['timestamp_ms'].min()} .. "
              f"{eth['timestamp_ms'].max()} ({time.time()-t0:.0f}s)", flush=True)
    else:
        print(f"ETH cache exists ({eth_p})", flush=True)

    if not fund_p.exists():
        print("Fetching OKX funding...", flush=True)
        t0 = time.time()
        fund = fetch_okx_funding()
        fund.to_parquet(fund_p, index=False)
        print(f"  funding: {len(fund):,} events ({time.time()-t0:.0f}s)", flush=True)
    else:
        print(f"funding cache exists ({fund_p})", flush=True)

    if not cb_p.exists():
        print("Fetching Coinbase BTC-USD 15m (slow, ~8min)...", flush=True)
        t0 = time.time()
        cb = fetch_coinbase_15m()
        cb.to_parquet(cb_p, index=False)
        print(f"  CB: {len(cb):,} bars ({time.time()-t0:.0f}s)", flush=True)
    else:
        print(f"CB cache exists ({cb_p})", flush=True)

    print("DONE", flush=True)


if __name__ == "__main__":
    main()
