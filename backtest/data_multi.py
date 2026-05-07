"""Genere training_data_multi.csv : meme schema que training_data.csv mais
multi-paires (BTC + ETH + SOL + BNB + XRP) avec une colonne symbol_id en plus.

extract_features est appelee per-paire (jamais sur des klines melanges) pour
que la detection de streak reste correcte intra-paire. Les rows sont concatenees
puis triees par timestamp avant ecriture.

Usage : python backtest/data_multi.py
"""
import csv
import time

import pandas as pd

from config import (
    SYMBOLS_MULTI, SYMBOL_TO_ID, OUTPUT_CSV_MULTI,
    DAYS_BACK, INTERVAL, MTF_INTERVALS, MAX_SEQ_LEN,
)
from data import fetch_all_klines, fetch_all_timeframes, extract_features


def main():
    end = int(time.time() * 1000)
    start = end - (DAYS_BACK * 24 * 60 * 60 * 1000)

    all_rows = []
    per_pair_stats = []

    for symbol in SYMBOLS_MULTI:
        print(f"\n=== Fetching {symbol} ({DAYS_BACK} days max) ===", flush=True)
        klines_15m = fetch_all_klines(symbol, start, end, interval=INTERVAL)
        print(f"  {len(klines_15m):,} 15m klines.\n")

        print(f"  Fetching MTF for {symbol} : {MTF_INTERVALS}...")
        klines_mtf = fetch_all_timeframes(symbol, start, end, MTF_INTERVALS)

        print(f"  Extracting features for {symbol}...")
        rows = extract_features(klines_15m, klines_mtf, max_seq_len=MAX_SEQ_LEN)

        sym_id = SYMBOL_TO_ID[symbol]
        for r in rows:
            r["symbol_id"] = sym_id

        per_pair_stats.append((symbol, sym_id, len(rows)))
        all_rows.extend(rows)
        print(f"  {symbol}: {len(rows):,} feature rows (symbol_id={sym_id}).")

    print("\n=== Concat + sort + write ===")
    df = pd.DataFrame(all_rows)

    # symbol_id et label en derniere position (label tout dernier).
    cols = [c for c in df.columns if c not in ("symbol_id", "label")] + ["symbol_id", "label"]
    df = df[cols]
    df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)

    df.to_csv(OUTPUT_CSV_MULTI, index=False)

    print(f"\nWrote {len(df):,} rows ({len(df.columns)} cols) to {OUTPUT_CSV_MULTI}")
    print("\nPer-pair breakdown :")
    for symbol, sym_id, n in per_pair_stats:
        pct = n / len(df) * 100 if len(df) else 0
        print(f"  {symbol} (id={sym_id}): {n:>9,} rows ({pct:>5.1f}%)")


if __name__ == "__main__":
    main()
