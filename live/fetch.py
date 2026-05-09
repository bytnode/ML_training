"""Bitstamp fetch wrapper for the live daemon.

Reuses backtest/modele_15m/data.py helpers (fetch_all_klines, fetch_all_timeframes).
The only live-specific concerns are:
  - Warmup window sizing (WARMUP_CANDLES_15M / WARMUP_CANDLES_MTF)
  - Freshness retry: refetch until last candle is strictly newer than previous tick
"""
import time

import settings  # triggers sys.path injection

# Imports from backtest/modele_15m/data.py (via sys.path hack in settings.py)
from data import fetch_all_klines, fetch_all_timeframes  # noqa: E402

# Imports from backtest/modele_15m/config.py
from config import SYMBOL, INTERVAL, INTERVAL_MS, MTF_INTERVALS  # noqa: E402


def fetch_warmup_15m(end_ms: int | None = None) -> list[list]:
    """Fetch the last WARMUP_CANDLES_15M 15m candles for indicator warmup at boot."""
    if end_ms is None:
        end_ms = int(time.time() * 1000)
    span_ms = settings.WARMUP_CANDLES_15M * INTERVAL_MS["15m"]
    start_ms = end_ms - span_ms
    return fetch_all_klines(SYMBOL, start_ms, end_ms, interval="15m")


def fetch_warmup_mtf(end_ms: int | None = None) -> dict[str, list[list]]:
    """Fetch the last WARMUP_CANDLES_MTF bars per MTF interval at boot."""
    if end_ms is None:
        end_ms = int(time.time() * 1000)
    out = {}
    for iv in MTF_INTERVALS:
        span_ms = settings.WARMUP_CANDLES_MTF * INTERVAL_MS[iv]
        start_ms = end_ms - span_ms
        out[iv] = fetch_all_klines(SYMBOL, start_ms, end_ms, interval=iv)
    return out


def fetch_recent_15m(n: int = 5, end_ms: int | None = None) -> list[list]:
    """Fetch the last n 15m candles. Used per cycle to refresh the rolling buffer."""
    if end_ms is None:
        end_ms = int(time.time() * 1000)
    start_ms = end_ms - n * INTERVAL_MS["15m"]
    return fetch_all_klines(SYMBOL, start_ms, end_ms, interval="15m")


def fetch_recent_mtf(n: int = 3, end_ms: int | None = None) -> dict[str, list[list]]:
    """Fetch the last n bars per MTF interval per cycle."""
    if end_ms is None:
        end_ms = int(time.time() * 1000)
    out = {}
    for iv in MTF_INTERVALS:
        start_ms = end_ms - n * INTERVAL_MS[iv]
        out[iv] = fetch_all_klines(SYMBOL, start_ms, end_ms, interval=iv)
    return out


def merge_klines(buffer: list[list], fresh: list[list]) -> list[list]:
    """Dedup-merge a rolling buffer with freshly-fetched candles by open_time_ms."""
    by_ts = {int(k[0]): k for k in buffer}
    for k in fresh:
        by_ts[int(k[0])] = k
    return sorted(by_ts.values(), key=lambda k: int(k[0]))


def freshness_retry(last_known_open_ms: int, max_attempts: int | None = None,
                    delay_s: float | None = None) -> list[list]:
    """Refetch the last few 15m candles until the latest open_time is strictly > last_known_open_ms.

    Returns the freshly-fetched list of candles (not merged). Raises TimeoutError if
    the new candle isn't published within max_attempts retries.
    """
    if max_attempts is None:
        max_attempts = settings.FRESHNESS_RETRY_MAX
    if delay_s is None:
        delay_s = settings.FRESHNESS_RETRY_DELAY_S

    for attempt in range(max_attempts):
        fresh = fetch_recent_15m(n=5)
        if fresh and int(fresh[-1][0]) > last_known_open_ms:
            return fresh
        time.sleep(delay_s)
    raise TimeoutError(
        f"No fresh 15m candle after {max_attempts} attempts "
        f"(last_known={last_known_open_ms}, latest_seen={fresh[-1][0] if fresh else 'none'})"
    )
