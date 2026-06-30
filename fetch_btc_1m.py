"""
Télécharge l'historique OHLCV BTC/USD en bougies 1 MINUTE depuis Bitstamp.

Par défaut : les 4 dernières années (~2,1 millions de bougies), sauvegardées en parquet.
Le fetch est parallélisé (ThreadPoolExecutor, comme backtest/.../modele_down/data.py)
car il faut ~2 100 requêtes (Bitstamp plafonne à 1000 bougies/requête).

Sortie (1 ligne par bougie 1m, triée par temps, dédupliquée) :
    timestamp_ms : int    unix millisecondes (convention interne du repo)
    datetime     : str    ISO8601 UTC, ex. "2022-06-16T00:00:00+00:00"
    open/high/low/close : float (USD)
    volume       : float  (BTC)

    python fetch_btc_1m.py                       # 4 ans -> btc_1m_4y.parquet
    python fetch_btc_1m.py --years 2 --out x.parquet
    python fetch_btc_1m.py --format csv          # -> btc_1m_4y.csv
    python fetch_btc_1m.py --pair ethusd --years 1
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Config Bitstamp (mêmes constantes que backtest/.../modele_down/data.py)
# ---------------------------------------------------------------------------
BITSTAMP_OHLC_URL = "https://www.bitstamp.net/api/v2/ohlc/{pair}/"
DEFAULT_PAIR = "btcusd"
STEP_SEC = 60               # bougies 1 minute
MAX_LIMIT = 1000            # max de bougies par requête Bitstamp
WORKERS = 8                 # requêtes parallèles
SLEEP_BETWEEN_CALLS = 0.1   # politesse réseau (par worker)
MAX_RETRIES = 5
RETRY_BACKOFF_BASE = 2.0


def fetch_window(pair: str, start_sec: int, end_sec: int) -> list[dict]:
    """Une fenêtre Bitstamp (<= 1000 bougies) avec retry exponentiel."""
    url = BITSTAMP_OHLC_URL.format(pair=pair)
    params = {"step": STEP_SEC, "limit": MAX_LIMIT,
              "start": start_sec, "end": end_sec}

    last_err: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 429 or resp.status_code >= 500:
                raise requests.HTTPError(f"HTTP {resp.status_code}: {resp.text[:200]}")
            resp.raise_for_status()
            time.sleep(SLEEP_BETWEEN_CALLS)
            return resp.json().get("data", {}).get("ohlc", [])
        except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as e:
            last_err = e
            if attempt == MAX_RETRIES - 1:
                break
            time.sleep(RETRY_BACKOFF_BASE * (2 ** attempt))

    raise RuntimeError(f"Échec après {MAX_RETRIES} essais : {last_err}")


def fetch_1m(pair: str, years: int) -> list[dict]:
    """Télécharge toutes les bougies 1m sur `years` ans, dédupliquées et triées."""
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=365 * years)
    start_sec = int(start.timestamp())
    now_sec = int(now.timestamp())

    window_span = STEP_SEC * MAX_LIMIT          # 1000 minutes ~ 16.7 h par fenêtre
    windows: list[tuple[int, int]] = []
    w_start = start_sec
    while w_start < now_sec:
        w_end = min(w_start + window_span - 1, now_sec)
        windows.append((w_start, w_end))
        w_start += window_span

    print(f"Source     : Bitstamp ({pair})")
    print(f"Timeframe  : 1m")
    print(f"Période    : {start:%Y-%m-%d} -> {now:%Y-%m-%d}  ({years} ans)")
    print(f"Fenêtres   : {len(windows)} requêtes ({WORKERS} workers parallèles)\n")

    candles: dict[int, dict] = {}      # timestamp_ms -> ligne OHLCV (dédup naturelle)
    completed = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = {ex.submit(fetch_window, pair, ws, we): (ws, we)
                   for ws, we in windows}
        for fut in as_completed(futures):
            for c in fut.result():
                ts_ms = int(c["timestamp"]) * 1000          # secondes -> ms
                candles[ts_ms] = {
                    "timestamp_ms": ts_ms,
                    "datetime": datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).isoformat(),
                    "open":   float(c["open"]),
                    "high":   float(c["high"]),
                    "low":    float(c["low"]),
                    "close":  float(c["close"]),
                    "volume": float(c["volume"]),
                }
            completed += 1
            if completed % 50 == 0 or completed == len(windows):
                pct = 100.0 * completed / len(windows)
                print(f"  {completed}/{len(windows)} fenêtres ({pct:.0f}%) "
                      f"— {len(candles):,} bougies uniques")

    rows = sorted(candles.values(), key=lambda r: r["timestamp_ms"])
    return rows


def save(rows: list[dict], out: Path, fmt: str) -> None:
    if not rows:
        print("Aucune bougie récupérée — rien à sauvegarder.")
        return

    if fmt == "parquet":
        import pandas as pd
        df = pd.DataFrame(rows)
        df.to_parquet(out, index=False)
    else:  # csv
        import csv
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    size_mb = out.stat().st_size / (1024 * 1024)
    first = rows[0]["datetime"]
    last = rows[-1]["datetime"]
    print(f"\n{'=' * 60}")
    print(f"  {len(rows):,} bougies 1m sauvegardées")
    print(f"  Fichier : {out}  ({size_mb:.1f} Mo)")
    print(f"  Plage   : {first}  ->  {last}")
    print("=" * 60)


def main() -> None:
    p = argparse.ArgumentParser(description="Fetch OHLCV BTC 1 minute depuis Bitstamp.")
    p.add_argument("--pair", default=DEFAULT_PAIR, help="Paire Bitstamp (défaut: btcusd).")
    p.add_argument("--years", type=int, default=4, help="Années en arrière (défaut: 4).")
    p.add_argument("--format", choices=["parquet", "csv"], default="parquet",
                   help="Format de sortie (défaut: parquet).")
    p.add_argument("--out", default=None, help="Chemin de sortie (défaut: btc_1m_<N>y.<ext>).")
    args = p.parse_args()

    out = Path(args.out) if args.out else Path(f"btc_1m_{args.years}y.{args.format}")
    rows = fetch_1m(args.pair, args.years)
    save(rows, out, args.format)


if __name__ == "__main__":
    main()
