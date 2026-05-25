"""Bitstamp public OHLC fetcher (no auth required).

Per TF, we ask the Bitstamp endpoint for the last N candles and convert to the
internal format used by features_up/features_down:
    klines = list of [open_time_ms, open, high, low, close, volume]

Usage:
    klines_15m = fetch_klines("btcusd", "15m", n_bars=3200)
    klines_5m  = fetch_klines("btcusd", "5m",  n_bars=200)

For multi-TF parallel fetch (warm or cold), see fetch_many_parallel below.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable

import requests

BITSTAMP_OHLC = "https://www.bitstamp.net/api/v2/ohlc/{pair}/"

TF_STEP_SEC = {
    "5m":  300,
    "15m": 900,
    "30m": 1800,
    "1h":  3600,
}

INTERVAL_MS = {tf: sec * 1000 for tf, sec in TF_STEP_SEC.items()}
MAX_LIMIT = 1000  # Bitstamp hard cap

REQUEST_TIMEOUT_S = 15            # per-request timeout (bumped from 10 for parallel)
REQUEST_RETRIES = 3               # retry on timeout / connection error
REQUEST_BACKOFF_S = 0.5           # exponential backoff base : 0.5, 1.0, 2.0, ...

# Module-level requests Session for HTTP keep-alive. Each thread sees the same
# Session but urllib3's pool gives each thread its own connection from the
# pool. With 5 concurrent threads we need a pool of >=5 per host to avoid
# blocking. requests.Session() defaults to pool_connections=10 which is enough.
# Without a Session, each call does a fresh TCP + TLS handshake (~300-500ms),
# completely dominating the call latency.
_SESSION = requests.Session()


def _request_chunk(pair: str, step_sec: int, limit: int, end_sec: int | None = None) -> list[list]:
    """One Bitstamp call. If end_sec is given, returns bars ending at or before that time.
    Returns list of [open_time_ms, open, high, low, close, volume].

    Retries up to REQUEST_RETRIES times on timeout / connection error with
    exponential backoff. Raises the last exception if all retries fail.
    """
    params = {"step": step_sec, "limit": limit}
    if end_sec is not None:
        params["end"] = end_sec
    url = BITSTAMP_OHLC.format(pair=pair)

    last_exc: Exception | None = None
    for attempt in range(REQUEST_RETRIES + 1):
        try:
            r = _SESSION.get(url, params=params, timeout=REQUEST_TIMEOUT_S)
            r.raise_for_status()
            data = r.json().get("data", {}).get("ohlc", [])
            return [
                [
                    int(c["timestamp"]) * 1000,
                    float(c["open"]),
                    float(c["high"]),
                    float(c["low"]),
                    float(c["close"]),
                    float(c["volume"]),
                ]
                for c in data
            ]
        except (requests.exceptions.Timeout,
                requests.exceptions.ConnectionError) as e:
            last_exc = e
            if attempt < REQUEST_RETRIES:
                time.sleep(REQUEST_BACKOFF_S * (2 ** attempt))
                continue
            raise
    if last_exc is not None:
        raise last_exc
    return []


def fetch_klines(pair: str, tf: str, n_bars: int) -> list[list]:
    """Fetch `n_bars` of the most recent candles for `pair` at timeframe `tf`.
    Concatenates multiple paginated requests when n_bars > 1000. Output is sorted
    ascending by open_time_ms and deduplicated.
    """
    if tf not in TF_STEP_SEC:
        raise ValueError(f"Unsupported timeframe: {tf}")
    step = TF_STEP_SEC[tf]

    out: dict[int, list] = {}
    remaining = n_bars
    end_sec: int | None = None
    safety_iter = 0
    while remaining > 0 and safety_iter < 20:
        safety_iter += 1
        limit = min(remaining, MAX_LIMIT)
        chunk = _request_chunk(pair, step, limit, end_sec=end_sec)
        if not chunk:
            break
        for k in chunk:
            out[k[0]] = k
        oldest_open = min(k[0] for k in chunk)
        if len(chunk) < limit:
            break
        # walk further back
        end_sec = oldest_open // 1000 - 1
        remaining -= len(chunk)
        time.sleep(0.15)  # gentle pacing
    klines = sorted(out.values(), key=lambda k: k[0])
    return klines[-n_bars:]


def fetch_all_warmups(pair: str = "btcusd", warmup_15m: int = 3200,
                      warmup_5m: int = 200, warmup_30m: int = 200,
                      warmup_1h: int = 200) -> dict[str, list[list]]:
    """Fetch the warmup buffers needed by features_up and features_down. Sizes
    are calibrated for: volume_rel_30d (2880 bars on 15m) + WARMUP margin.
    """
    return {
        "15m": fetch_klines(pair, "15m", warmup_15m),
        "5m":  fetch_klines(pair, "5m",  warmup_5m),
        "30m": fetch_klines(pair, "30m", warmup_30m),
        "1h":  fetch_klines(pair, "1h",  warmup_1h),
    }


def merge_klines(buffer: list[list], fresh: list[list]) -> list[list]:
    """Replace-or-append fresh bars into buffer by open_time_ms."""
    by_ts = {int(k[0]): k for k in buffer}
    for k in fresh:
        by_ts[int(k[0])] = k
    return sorted(by_ts.values(), key=lambda k: k[0])


def latest_open_ms(klines: list[list]) -> int:
    return int(klines[-1][0]) if klines else 0


def fetch_many_parallel(specs: list[tuple[str, str, int]]) -> dict[tuple[str, str], list[list]]:
    """Parallel fetch of multiple (pair, tf, n_bars) tuples in one shot.

    Each tuple becomes a separate ThreadPool task. Total wall time is ~ the
    SLOWEST single fetch (dominated by Bitstamp publish latency + 1 HTTP RTT),
    not the SUM. Replaces 5 serial fetch_klines calls (~2.5s) with one
    parallel batch (~0.3-0.7s).

    Returns a dict keyed by (pair, tf) -> klines list. Caller is responsible
    for merging into any pre-existing cached buffer.

    If any task raises, the exception propagates (sequential semantics on error).
    """
    if not specs:
        return {}
    with ThreadPoolExecutor(max_workers=len(specs)) as ex:
        futures = {
            (pair, tf): ex.submit(fetch_klines, pair, tf, n_bars)
            for (pair, tf, n_bars) in specs
        }
        return {key: fut.result() for key, fut in futures.items()}


if __name__ == "__main__":
    import json
    print("Fetching last 5 bars per TF as smoke test...")
    for tf in ["5m", "15m", "30m", "1h"]:
        kl = fetch_klines("btcusd", tf, n_bars=5)
        print(f"  {tf}: {len(kl)} bars")
        for k in kl[-2:]:
            import datetime as dt
            ts = dt.datetime.fromtimestamp(k[0] / 1000, tz=dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            print(f"    {ts}  O={k[1]} H={k[2]} L={k[3]} C={k[4]} V={k[5]}")
