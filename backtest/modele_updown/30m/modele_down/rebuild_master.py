"""
Rebuild the master 1h parquet from the existing SQLite candles table,
without re-fetching Bitstamp. Used when modifying feature definitions
in data.py and re-running just phase 4.

Usage : cd modele_down && python rebuild_master.py
"""
from __future__ import annotations
from pathlib import Path

from data import build_master_1h, DEFAULT_DB_PATH

if __name__ == "__main__":
    db_path = Path(DEFAULT_DB_PATH)
    if not db_path.exists():
        raise FileNotFoundError(f"{db_path} introuvable. Lance `python data.py` d'abord.")
    out_path = db_path.parent / "btc_multi_tf_1h.parquet"
    n_rows, n_cols = build_master_1h(db_path, out_path)
    print(f"\n=== Master parquet rebuilt : {n_rows:,} rows x {n_cols} cols ===")
