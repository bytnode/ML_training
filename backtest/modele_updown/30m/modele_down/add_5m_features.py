"""
Fetch 5m BTC/USD bars from Bitstamp, compute scale-invariant features per 5m,
add LOOKBACK features for the last N 5m bars before decision time to the
existing parquet.

This gives the model microstructure context for the last 5-30 minutes BEFORE
the 1h close decision time.

Run from modele_down/ : python add_5m_features.py

Idempotent: re-running re-fetches + overwrites the 5m features.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
import time

import numpy as np
import pandas as pd
import requests


ROOT = Path(__file__).resolve().parent
PARQUET_PATH = ROOT / "DB" / "btc_multi_tf_1h.parquet"

# 5m data window: enough to cover all 1h bars in the parquet (~5000 days back).
DAYS_BACK = 5000

# How far back we look from decision time T = open_1h + 1h.
# The 5m bar closing at T is the one we MOST want.
# bar_idx 0 = closes at T, 1 = closes at T - 5min, etc.
LOOKBACK_5M_BARS = (0, 1, 2, 3, 5, 9)

INTERVAL_5M_MS = 5 * 60 * 1000
INTERVAL_1H_MS = 60 * 60 * 1000
INTERVAL_5M_SEC = 5 * 60

BITSTAMP_OHLC_URL = "https://www.bitstamp.net/api/v2/ohlc/btcusd/"
BITSTAMP_MAX_LIMIT = 1000
WORKERS = 16
MAX_RETRIES = 5
RETRY_BACKOFF_SEC = 2.0

# 5m feature periods (same as 30m in data.py)
ATR_PERIOD = 14
BODY_RATIO_AVG_WINDOW = 14
VOLUME_AVG_WINDOW = 20
OBV_PRESSURE_WINDOW = 10

COLS_5M_KEPT = ["close_ret", "volume_ratio", "body_ratio_close",
                "obv_pressure", "atr_pct"]


def fetch_5m_window(start_sec, end_sec):
    params = {"step": INTERVAL_5M_SEC, "limit": BITSTAMP_MAX_LIMIT,
              "start": start_sec, "end": end_sec}
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(BITSTAMP_OHLC_URL, params=params, timeout=20)
            if r.status_code == 429 or r.status_code >= 500:
                raise requests.HTTPError(f"HTTP {r.status_code}")
            r.raise_for_status()
            return r.json().get("data", {}).get("ohlc", [])
        except Exception as e:
            last_err = e
            if attempt == MAX_RETRIES - 1:
                break
            time.sleep(RETRY_BACKOFF_SEC * (2 ** attempt))
    raise RuntimeError(f"5m fetch failed: {last_err}")


def fetch_all_5m(days_back=DAYS_BACK):
    """Fetch all 5m bars across `days_back` days using parallel windows."""
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = int((datetime.now(timezone.utc) - timedelta(days=days_back)).timestamp() * 1000)
    window_span_ms = INTERVAL_5M_MS * BITSTAMP_MAX_LIMIT
    windows = []
    w = start_ms
    while w < now_ms:
        w_end = min(w + window_span_ms - 1, now_ms)
        windows.append((w // 1000, w_end // 1000))
        w += window_span_ms
    print(f"  5m: fetching {len(windows)} windows...")

    all_rows = []
    completed = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = [ex.submit(fetch_5m_window, ws, we) for ws, we in windows]
        for fut in as_completed(futures):
            try:
                raw = fut.result()
            except Exception as e:
                print(f"    [err] {e}")
                raw = []
            for c in raw:
                all_rows.append({
                    "timestamp_ms": int(c["timestamp"]) * 1000,
                    "open": float(c["open"]),
                    "high": float(c["high"]),
                    "low": float(c["low"]),
                    "close": float(c["close"]),
                    "volume": float(c["volume"]),
                })
            completed += 1
            if completed % 50 == 0 or completed == len(windows):
                print(f"    {completed}/{len(windows)} windows, "
                      f"{len(all_rows):,} bars")

    df = pd.DataFrame(all_rows).drop_duplicates(subset="timestamp_ms")
    df = df.sort_values("timestamp_ms").reset_index(drop=True)
    print(f"  5m: total {len(df):,} unique bars")
    return df


def compute_5m_features(df):
    """5 normalized features per 5m, scale-invariant."""
    df = df.copy()
    close_prev = df["close"].shift(1)
    out = pd.DataFrame({
        "timestamp_ms": df["timestamp_ms"].astype("int64"),
        "close_ret": (df["close"] - close_prev) / close_prev,
    })

    vol_avg = df["volume"].rolling(VOLUME_AVG_WINDOW, min_periods=VOLUME_AVG_WINDOW).mean()
    out["volume_ratio"] = df["volume"] / vol_avg

    rng = (df["high"] - df["low"]).replace(0, np.nan)
    body_close = ((df["close"] - df["open"]).abs() / rng).clip(0, 1)
    out["body_ratio_close"] = body_close

    direction = np.sign(df["close"].diff().fillna(0))
    obv = (direction * df["volume"]).cumsum()
    obv_diff = obv - obv.shift(OBV_PRESSURE_WINDOW)
    vol_sum = df["volume"].rolling(OBV_PRESSURE_WINDOW, min_periods=OBV_PRESSURE_WINDOW).sum()
    out["obv_pressure"] = obv_diff / vol_sum.clip(lower=1)

    high = df["high"]; low = df["low"]
    tr = pd.concat([
        (high - low).abs(),
        (high - close_prev).abs(),
        (low  - close_prev).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / ATR_PERIOD, adjust=False).mean()
    out["atr_pct"] = atr / df["close"]

    return out


def add_lookback_to_parquet(df_5m_feat):
    print(f"\n  Loading {PARQUET_PATH}...")
    df = pd.read_parquet(PARQUET_PATH)
    print(f"    {len(df):,} rows, {len(df.columns)} cols")

    times_1h = df["timestamp_ms"].to_numpy(dtype=np.int64)
    times_5m = df_5m_feat["timestamp_ms"].to_numpy(dtype=np.int64)
    decision_times = times_1h + INTERVAL_1H_MS

    new_cols = {}
    for back in LOOKBACK_5M_BARS:
        target_open = decision_times - INTERVAL_5M_MS * (back + 1)
        indices = np.searchsorted(times_5m, target_open, side="right") - 1
        valid = indices >= 0
        for col in COLS_5M_KEPT:
            new_col = f"{col}_5m_b{back}"
            values = np.full(len(df), np.nan, dtype=float)
            if valid.any():
                src = df_5m_feat[col].to_numpy(dtype=float)
                values[valid] = src[indices[valid]]
            new_cols[new_col] = values

    for col, vals in new_cols.items():
        df[col] = vals

    print(f"\n  Added {len(new_cols)} 5m lookback columns")
    valid_rows = (~df[list(new_cols.keys())].isna().any(axis=1)).sum()
    print(f"  Rows with all 5m features non-NaN: {valid_rows:,} / {len(df):,}")

    df.to_parquet(PARQUET_PATH, index=False)
    print(f"  Wrote back {PARQUET_PATH}")


def main():
    if not PARQUET_PATH.exists():
        raise FileNotFoundError(f"{PARQUET_PATH} not found.")

    print("Phase 1: fetching 5m raw bars from Bitstamp...")
    t0 = time.time()
    df_5m = fetch_all_5m()
    print(f"  Fetch time: {time.time() - t0:.1f}s")

    print("\nPhase 2: computing 5m features...")
    t1 = time.time()
    feat_5m = compute_5m_features(df_5m)
    print(f"  {len(feat_5m):,} feature rows, {time.time() - t1:.1f}s")

    print("\nPhase 3: adding lookback features to parquet...")
    add_lookback_to_parquet(feat_5m)

    print(f"\nTotal time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
