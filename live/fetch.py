"""Fetch des bougies Bitstamp pour le mode live.

Reutilise les fonctions de backtest/data.py via un sys.path hack.
"""
import os
import sys
import time

# Ajoute backtest/ a sys.path AVANT d'importer ses modules. Cette ligne doit
# rester avant les imports backtest et avant tout import qui ferait
# 'from config import ...' (cf. note critique du plan).
_BACKTEST_DIR = os.path.join(os.path.dirname(__file__), "..", "backtest")
sys.path.insert(0, os.path.abspath(_BACKTEST_DIR))

from data import fetch_all_klines  # noqa: E402
from config import SYMBOL, MTF_INTERVALS, INTERVAL_MS  # noqa: E402

INTERVAL_15M = "15m"


def fetch_warmup_window(n_15m: int, n_mtf: int) -> dict:
    """Fetche assez de bougies pour warmup les indicateurs au demarrage.

    Returns: dict avec keys '15m', '30m', '1h', '4h', '1d'. Chaque value est
             une liste triee de [open_time_ms, o, h, l, c, v].
    """
    return _fetch_all(n_15m, n_mtf)


def fetch_recent_window(n_15m: int, n_mtf: int) -> dict:
    """Fetche les dernieres bougies a chaque cycle (typiquement 5 / 3).

    Same implementation que fetch_warmup_window ; semantique distincte pour
    clarifier l'intention au call site.
    """
    return _fetch_all(n_15m, n_mtf)


def _fetch_all(n_15m: int, n_mtf: int) -> dict:
    now_ms = int(time.time() * 1000)
    out = {}

    # 15m
    interval_15m_ms = INTERVAL_MS[INTERVAL_15M]
    start_15m = now_ms - (n_15m + 2) * interval_15m_ms  # +2 marge pour overlap
    klines_15m = fetch_all_klines(SYMBOL, start_15m, now_ms, INTERVAL_15M)
    out[INTERVAL_15M] = klines_15m[-n_15m:] if len(klines_15m) > n_15m else klines_15m

    # MTF
    for tf in MTF_INTERVALS:
        tf_ms = INTERVAL_MS[tf]
        start_tf = now_ms - (n_mtf + 2) * tf_ms
        klines_tf = fetch_all_klines(SYMBOL, start_tf, now_ms, tf)
        out[tf] = klines_tf[-n_mtf:] if len(klines_tf) > n_mtf else klines_tf

    return out


def merge_klines(existing: list, fresh: list) -> list:
    """Merge fresh dans existing en deduplicant par open_time_ms.

    fresh ecrase existing pour les open_time_ms en doublon (Bitstamp peut
    re-publier les dernieres bougies avec des valeurs revisees).
    """
    by_ts = {k[0]: k for k in existing}
    for k in fresh:
        by_ts[k[0]] = k
    return sorted(by_ts.values(), key=lambda x: x[0])


if __name__ == "__main__":
    # Smoke test : fetche 10 bougies recentes sur tous les TF
    print("Fetching warmup window (10 15m + 5 MTF each)...")
    result = fetch_warmup_window(10, 5)
    for tf, candles in result.items():
        latest_ts = candles[-1][0] if candles else None
        print(f"  {tf}: {len(candles)} candles, latest open_time_ms = {latest_ts}")
