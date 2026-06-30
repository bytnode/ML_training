"""
Amplitude (évolution) moyenne des bougies BTC/USD sur 10 ans — 15m, source Bitstamp.

Récupère l'historique OHLC 15 minutes du BTC/USD sur les 10 dernières années
depuis l'API publique Bitstamp, puis calcule l'ÉVOLUTION MOYENNE des bougies.

Évolution d'une bougie = (close - open) / open, en %  (le corps, en relatif).
On l'exprime en % et non en $ : le BTC a été multiplié par ~150 en 10 ans,
donc une moyenne en dollars n'aurait aucun sens.

  - VERTES (close > open) : moyenne de (close-open)/open  -> gain moyen (positif)
  - ROUGES (close < open) : moyenne de (close-open)/open  -> perte moyenne (négatif)
  - GLOBAL (toutes)       : moyenne de (close-open)/open  -> dérive nette moyenne

Aucune dépendance lourde : seulement `requests` + la lib standard.

    python stats_bougies_btc.py
    python stats_bougies_btc.py --pair btcusd --years 10
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import requests

# ---------------------------------------------------------------------------
# Config Bitstamp (mêmes constantes que backtest/.../modele_down/data.py)
# ---------------------------------------------------------------------------
BITSTAMP_OHLC_URL = "https://www.bitstamp.net/api/v2/ohlc/{pair}/"
DEFAULT_PAIR = "btcusd"
TIMEFRAMES = {              # label -> pas Bitstamp en secondes
    "5m":  5 * 60,
    "15m": 15 * 60,
    "30m": 30 * 60,
    "1h":  60 * 60,
}
MAX_LIMIT = 1000            # max de bougies par requête Bitstamp
SLEEP_BETWEEN_CALLS = 0.1   # politesse réseau
MAX_RETRIES = 5
RETRY_BACKOFF_BASE = 2.0


def fetch_window(pair: str, step_sec: int, start_sec: int, end_sec: int) -> list[dict]:
    """Une fenêtre Bitstamp (<= 1000 bougies) avec retry exponentiel."""
    url = BITSTAMP_OHLC_URL.format(pair=pair)
    params = {"step": step_sec, "limit": MAX_LIMIT,
              "start": start_sec, "end": end_sec}

    last_err: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, timeout=20)
            if resp.status_code == 429 or resp.status_code >= 500:
                raise requests.HTTPError(f"HTTP {resp.status_code}: {resp.text[:200]}")
            resp.raise_for_status()
            return resp.json().get("data", {}).get("ohlc", [])
        except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as e:
            last_err = e
            if attempt == MAX_RETRIES - 1:
                break
            wait = RETRY_BACKOFF_BASE * (2 ** attempt)
            print(f"    {type(e).__name__} (essai {attempt + 1}/{MAX_RETRIES}), "
                  f"backoff {wait:.0f}s")
            time.sleep(wait)

    raise RuntimeError(f"Échec après {MAX_RETRIES} essais : {last_err}")


def fetch_all(pair: str, tf: str, years: int) -> dict[int, dict]:
    """Récupère toutes les bougies du timeframe `tf` sur `years` ans, dédupliquées."""
    step_sec = TIMEFRAMES[tf]
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=365 * years)
    start_sec = int(start.timestamp())
    now_sec = int(now.timestamp())

    window_span = step_sec * MAX_LIMIT          # 1000 bougies par fenêtre
    windows: list[tuple[int, int]] = []
    w_start = start_sec
    while w_start < now_sec:
        w_end = min(w_start + window_span - 1, now_sec)
        windows.append((w_start, w_end))
        w_start += window_span

    print(f"Source       : Bitstamp ({pair})")
    print(f"Timeframe    : {tf}")
    print(f"Période      : {start:%Y-%m-%d} -> {now:%Y-%m-%d}  ({years} ans)")
    print(f"Fenêtres     : {len(windows)} requêtes à ~{MAX_LIMIT} bougies\n")

    candles: dict[int, dict] = {}      # ts_sec -> bougie (dédup naturelle)
    for i, (ws, we) in enumerate(windows, 1):
        raw = fetch_window(pair, step_sec, ws, we)
        for c in raw:
            ts = int(c["timestamp"])
            candles[ts] = {"open": float(c["open"]), "close": float(c["close"])}
        if i % 25 == 0 or i == len(windows):
            print(f"  {i}/{len(windows)} fenêtres — {len(candles):,} bougies uniques")
        time.sleep(SLEEP_BETWEEN_CALLS)

    return candles


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def compute_stats(candles: dict[int, dict], tf: str = "") -> None:
    """Calcule et affiche l'évolution moyenne (%) verte / rouge / globale, par an + total."""
    if not candles:
        print("Aucune bougie récupérée.")
        return

    # Évolution en % par bougie, rangée par année et par couleur.
    per_year_green: dict[int, list[float]] = defaultdict(list)
    per_year_red: dict[int, list[float]] = defaultdict(list)
    per_year_all: dict[int, list[float]] = defaultdict(list)

    all_green: list[float] = []
    all_red: list[float] = []
    all_total: list[float] = []

    for ts, c in candles.items():
        o, cl = c["open"], c["close"]
        if o <= 0:
            continue
        ret = (cl - o) / o * 100.0      # évolution de la bougie en %
        year = datetime.fromtimestamp(ts, tz=timezone.utc).year

        all_total.append(ret)
        per_year_all[year].append(ret)
        if ret > 0:
            all_green.append(ret)
            per_year_green[year].append(ret)
        elif ret < 0:
            all_red.append(ret)
            per_year_red[year].append(ret)
        # ret == 0 (doji) : compté seulement dans le global

    # ----- Détail par année -----
    print("\n" + "=" * 64)
    print(f"{'Année':>6} | {'verte moy.':>11} | {'rouge moy.':>11} | {'globale moy.':>13}")
    print("-" * 64)
    for year in sorted(per_year_all):
        g = _avg(per_year_green[year])
        r = _avg(per_year_red[year])
        a = _avg(per_year_all[year])
        print(f"{year:>6} | {g:>+10.4f}% | {r:>+10.4f}% | {a:>+12.4f}%")

    # ----- Global sur toute la période -----
    g, r, a = _avg(all_green), _avg(all_red), _avg(all_total)
    print("=" * 64)
    print(f"\nEVOLUTION MOYENNE par bougie {tf or '?'} - {len(all_total):,} bougies")
    print(f"  VERTE  (close > open) : {g:>+8.4f} %   (amplitude {abs(g):.4f} %)")
    print(f"  ROUGE  (close < open) : {r:>+8.4f} %   (amplitude {abs(r):.4f} %)")
    print(f"  GLOBAL (toutes)       : {a:>+8.4f} %   (derive nette moyenne)")


def show_biggest(candles: dict[int, dict], tf: str, top: int) -> None:
    """Affiche les `top` plus grosses bougies à la hausse et à la baisse (par % de corps)."""
    rows = []
    for ts, c in candles.items():
        o, cl = c["open"], c["close"]
        if o <= 0:
            continue
        ret = (cl - o) / o * 100.0
        rows.append((ts, o, cl, ret))

    ups = sorted(rows, key=lambda r: r[3], reverse=True)[:top]
    downs = sorted(rows, key=lambda r: r[3])[:top]

    def line(rank: int, r: tuple) -> str:
        ts, o, cl, ret = r
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        return (f"  {rank:>2}. {dt:%Y-%m-%d %H:%M} UTC | "
                f"open {o:>11,.2f} -> close {cl:>11,.2f} | {ret:>+8.3f}%")

    print(f"\n{'=' * 72}")
    print(f"TOP {top} plus grosses bougies {tf} A LA HAUSSE (corps en %)")
    print("-" * 72)
    for i, r in enumerate(ups, 1):
        print(line(i, r))

    print(f"\nTOP {top} plus grosses bougies {tf} A LA BAISSE (corps en %)")
    print("-" * 72)
    for i, r in enumerate(downs, 1):
        print(line(i, r))


def main() -> None:
    p = argparse.ArgumentParser(description="Evolution moyenne des bougies BTC (Bitstamp).")
    p.add_argument("--pair", default=DEFAULT_PAIR, help="Paire Bitstamp (défaut: btcusd).")
    p.add_argument("--tf", default="15m", choices=list(TIMEFRAMES),
                   help="Timeframe des bougies (défaut: 15m).")
    p.add_argument("--years", type=int, default=10, help="Nombre d'années en arrière (défaut: 10).")
    p.add_argument("--top", type=int, default=0,
                   help="Affiche aussi les N plus grosses bougies hausse/baisse (0 = off).")
    args = p.parse_args()

    candles = fetch_all(args.pair, args.tf, args.years)
    compute_stats(candles, args.tf)
    if args.top > 0:
        show_biggest(candles, args.tf, args.top)


if __name__ == "__main__":
    main()
