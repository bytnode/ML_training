"""fetch_1m.py — fetch raw 1min OHLCV from Bitstamp for the intrabar exit engine.

Standalone: does NOT touch modele_down's SQLite/parquet. Writes RAW OHLCV
(open/high/low/close/volume, not normalized) to DB_1m/btc_1m.parquet so
main.py can walk the intra-1h path at 1min granularity (SL/TP/trailing/paliers).

RESTAURE 2026-08-11 (etait supprime depuis le 2026-07-09) et modernise :
  - DAYS_BACK = 750 (etait 420) — fenetre de test etendue + marge
  - INCREMENTAL : conserve les bougies deja presentes dans le parquet et ne
    telecharge QUE les plages manquantes (backfill ancien + rattrapage recent).
    Relancer le script est donc idempotent et rapide.
  - ecriture ATOMIQUE (tmp puis replace) : un Ctrl-C en plein run ne peut pas
    corrompre le parquet de production.

    cd backtest\\modele_updown\\1h\\modele_team
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
DAYS_BACK = 750                       # profondeur cible du fichier (jours), ancree sur maintenant
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


def _missing_ranges(existing_ts: set, start_ms: int, end_ms: int) -> list[tuple[int, int]]:
    """Plages [a, b) de minutes ABSENTES du parquet dans [start_ms, end_ms).
    Regroupe les minutes manquantes contigues pour minimiser les requetes."""
    ranges = []
    cur = None
    t = start_ms
    while t < end_ms:
        if t not in existing_ts:
            if cur is None:
                cur = t
        else:
            if cur is not None:
                ranges.append((cur, t))
                cur = None
        t += STEP_MS
    if cur is not None:
        ranges.append((cur, end_ms))
    return ranges


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    end_ms = int(now.timestamp() * 1000)                      # derniere minute close exclue naturellement
    start_ms = int((now - timedelta(days=DAYS_BACK)).timestamp() * 1000)
    print(f"=== fetch_1m.py : cible {DAYS_BACK}j de 1min BTC/USD (Bitstamp) ===")
    print(f"    [{pd.Timestamp(start_ms, unit='ms', tz='UTC')} -> {pd.Timestamp(end_ms, unit='ms', tz='UTC')}]")

    rows = {}
    if OUT.exists():
        old = pd.read_parquet(OUT)
        for r in old.itertuples(index=False):
            ts = int(r.timestamp_ms)
            if start_ms <= ts < end_ms:                        # trim hors fenetre 750j
                rows[ts] = (ts, float(r.open), float(r.high),
                            float(r.low), float(r.close), float(r.volume))
        print(f"    existant : {len(old):,} bougies, {len(rows):,} conservees dans la fenetre")

    todo = _missing_ranges(set(rows), start_ms, end_ms)
    # NB : Bitstamp a des minutes sans trade (pas de bougie) -> ces trous
    # reviendront a chaque run. On ne re-demande que par plage contigue, donc
    # le cout reste borne (~1 requete par plage de 1000 minutes manquantes).
    total_min = sum((b - a) // STEP_MS for a, b in todo)
    span = STEP_MS * LIMIT
    total_win = sum((b - a + span - 1) // span for a, b in todo)
    print(f"    a telecharger : {total_min:,} minutes en {total_win} fenetres "
          f"({len(todo)} plages manquantes)")

    n = 0
    for a, b in todo:
        w = a
        while w < b:
            we = min(w + span - 1, b - 1)
            data = fetch_window(w // 1000, we // 1000)
            for c in data:
                ts = int(c["timestamp"]) * 1000
                if start_ms <= ts < end_ms:
                    rows[ts] = (ts, float(c["open"]), float(c["high"]),
                                float(c["low"]), float(c["close"]), float(c["volume"]))
            n += 1
            if n % 25 == 0 or (w + span >= b and (a, b) == todo[-1]):
                print(f"  {n}/{total_win} fenetres, {len(rows):,} bougies au total")
            w += span
            time.sleep(0.1)

    df = pd.DataFrame(sorted(rows.values()),
                      columns=["timestamp_ms", "open", "high", "low", "close", "volume"])
    tmp = OUT.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(OUT)                                           # atomique
    print(f"-> {OUT} : {len(df):,} bougies 1m  "
          f"[{pd.Timestamp(int(df['timestamp_ms'].iloc[0]), unit='ms', tz='UTC')} -> "
          f"{pd.Timestamp(int(df['timestamp_ms'].iloc[-1]), unit='ms', tz='UTC')}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
