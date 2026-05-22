"""
Add 30m lookback features to the parquet.

The current parquet has a SINGLE 30m bar's features per 1h row (the one that
closes at open_1h + 1h). This script reads the SQLite 30m series and adds
features from the PREVIOUS 30m bars too (lookback of 1, 2, 3 bars back).

These give the model fine-grained context: instead of just "what did the last
30m do", we get "what did the last 2 hours of 30m do".

Idempotent: re-running overwrites existing columns.

Run from modele_down/ : python add_30m_lookback.py
"""
from __future__ import annotations

from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
PARQUET_PATH = ROOT / "DB" / "btc_multi_tf_1h.parquet"
DB_PATH = ROOT / "DB" / "btc_multi_tf.db"

# Lookback: 1 = the prev 30m bar (which closes 30min before decision time)
LOOKBACK_30M_BARS = (1, 2, 3)
COLS_30M = ["close_ret", "volume_ratio", "body_ratio_close",
            "obv_pressure", "atr_pct"]

INTERVAL_30M_MS = 30 * 60 * 1000
INTERVAL_1H_MS = 60 * 60 * 1000


def main():
    if not PARQUET_PATH.exists():
        raise FileNotFoundError(f"{PARQUET_PATH} not found.")
    if not DB_PATH.exists():
        raise FileNotFoundError(f"{DB_PATH} not found. Run data.py.")

    print(f"Loading {PARQUET_PATH}...")
    df = pd.read_parquet(PARQUET_PATH)
    print(f"  {len(df):,} rows, {len(df.columns)} cols")

    conn = sqlite3.connect(DB_PATH)
    try:
        df_30m = pd.read_sql_query(
            "SELECT timestamp_ms, " + ", ".join(COLS_30M) + " "
            "FROM candles WHERE timeframe = '30m' ORDER BY timestamp_ms",
            conn,
        )
    finally:
        conn.close()
    print(f"  30m table: {len(df_30m):,} rows")

    times_1h = df["timestamp_ms"].to_numpy(dtype=np.int64)
    times_30m = df_30m["timestamp_ms"].to_numpy(dtype=np.int64)
    # Decision time = open_1h + 1h. The 30m bar that closes AT decision time
    # has open_time = decision - 30m. searchsorted finds the index.
    decision_times = times_1h + INTERVAL_1H_MS

    new_cols = {}
    for back in LOOKBACK_30M_BARS:
        # The 30m bar `back` bars BEFORE the close-aligned one. Its open is at
        # decision_time - 30m * (back + 1).
        target_open = decision_times - INTERVAL_30M_MS * (back + 1)
        indices = np.searchsorted(times_30m, target_open, side="right") - 1
        valid = indices >= 0

        for col in COLS_30M:
            new_col = f"{col}_30m_b{back}"
            values = np.full(len(df), np.nan, dtype=float)
            if valid.any():
                src = df_30m[col].to_numpy(dtype=float)
                values[valid] = src[indices[valid]]
            new_cols[new_col] = values

    for col, vals in new_cols.items():
        df[col] = vals
    print(f"\n  Added {len(new_cols)} columns")
    for col in new_cols:
        v = df[col].dropna()
        if len(v) > 0:
            print(f"    {col:<28} min={v.min():>8.4f} max={v.max():>7.4f} "
                  f"mean={v.mean():>7.4f} std={v.std():>7.4f}")

    df.to_parquet(PARQUET_PATH, index=False)
    print(f"\nWrote back {PARQUET_PATH}")


if __name__ == "__main__":
    main()
