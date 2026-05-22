"""Detailed per-bar logs for the team backtest.

main.py calls this module to turn the two per-bar prediction frames (from each
package's evaluate_full_on_slice) into three CSVs inside 5m/logs/ :

    team_all_bars.csv  - one row per analyzed 5m bar (everything)
    team_skipped.csv   - bars where the team placed NO bet (team_decision PASS or CANCEL)
    team_taken.csv     - bars where the team placed a bet (team_decision UP or DOWN)

All three share one column schema (LOG_COLUMNS), so skipped + taken always
partition all_bars.

OHLC prices are reconstructed from the SQLite candles table built by
modele_down/data.py - the parquet itself stores only scale-invariant features.
If that DB is missing, the price columns are left blank and the backtest still
completes.
"""
from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

_5M_DIR = Path(__file__).resolve().parent
_OHLC_DB = _5M_DIR / "modele_down" / "DB" / "btc_multi_tf.db"
INTERVAL_5M_MS = 5 * 60 * 1000

# Canonical column order shared by all three CSVs.
LOG_COLUMNS = [
    "timestamp_ms", "datetime_utc",
    "open", "high", "low", "close",
    "volatility_state", "is_weekend", "hour",
    "p_up", "th_up", "take_up",
    "p_down", "th_down", "take_down",
    "team_decision", "bet_taken",
    "next_open", "next_close", "next_candle",
    "result",
]

# Filenames written into logs/. Order = how the run report lists them.
LOG_FILES = {
    "all": "team_all_bars.csv",
    "skipped": "team_skipped.csv",
    "taken": "team_taken.csv",
}


def load_ohlc(start_ms: int, end_ms: int) -> pd.DataFrame:
    """Reconstruct 5m OHLC for [start_ms, end_ms] from the SQLite candles table.

    The table stores raw `close` plus the four return ratios; data.py defines
    each ratio as (price - close_prev) / close_prev, so:
        close_prev = close / (1 + close_ret)
        open / high / low = (1 + *_ret) * close_prev

    Returns a DataFrame [timestamp_ms, open, high, low, close]. Empty (with the
    right columns) if the DB is missing - the caller treats prices as optional.
    """
    cols = ["timestamp_ms", "open", "high", "low", "close"]
    if not _OHLC_DB.exists():
        logging.warning("detailed_log: OHLC DB not found (%s); price columns "
                        "will be blank.", _OHLC_DB)
        return pd.DataFrame(columns=cols)

    # Pull one extra interval past end_ms so the last bar's next_* can be filled.
    query = (
        "SELECT timestamp_ms, close, open_ret, high_ret, low_ret, close_ret "
        "FROM candles WHERE timeframe = '5m' "
        "AND timestamp_ms >= ? AND timestamp_ms <= ? ORDER BY timestamp_ms"
    )
    conn = sqlite3.connect(_OHLC_DB)
    try:
        raw = pd.read_sql_query(query, conn, params=(start_ms, end_ms + INTERVAL_5M_MS))
    finally:
        conn.close()
    if raw.empty:
        return pd.DataFrame(columns=cols)

    close_prev = raw["close"] / (1.0 + raw["close_ret"])
    raw["open"] = (1.0 + raw["open_ret"]) * close_prev
    raw["high"] = (1.0 + raw["high_ret"]) * close_prev
    raw["low"] = (1.0 + raw["low_ret"]) * close_prev
    return raw[cols]


def _classify(take_up: pd.Series, take_down: pd.Series) -> pd.Series:
    """Per-bar team coherence rule (vectorized).

        up only   -> UP
        down only -> DOWN
        both      -> CANCEL  (one says UP, the other DOWN)
        neither   -> PASS

    Mirrors team_decision() in live/live.py and build_team_bets() in main.py.
    """
    u = take_up.astype(bool).to_numpy()
    d = take_down.astype(bool).to_numpy()
    out = np.where(u & d, "CANCEL", np.where(u, "UP", np.where(d, "DOWN", "PASS")))
    return pd.Series(out, index=take_up.index)


def _attach_ohlc(merged: pd.DataFrame, ohlc: pd.DataFrame) -> pd.DataFrame:
    """Add open/high/low/close (the analyzed bar) and next_open/next_close
    (the target bar one interval later) to `merged`, keyed on `timestamp`."""
    price_cols = ["open", "high", "low", "close", "next_open", "next_close"]
    if ohlc.empty:
        for col in price_cols:
            merged[col] = pd.NA
        return merged

    bar = ohlc.rename(columns={"timestamp_ms": "timestamp"})
    merged = merged.merge(bar[["timestamp", "open", "high", "low", "close"]],
                          on="timestamp", how="left")

    # The target bar is one interval after the analyzed bar: shift its key back.
    nxt = ohlc.rename(columns={"timestamp_ms": "timestamp",
                               "open": "next_open", "close": "next_close"})
    nxt["timestamp"] = nxt["timestamp"] - INTERVAL_5M_MS
    merged = merged.merge(nxt[["timestamp", "next_open", "next_close"]],
                          on="timestamp", how="left")
    return merged


def build_per_bar_table(up_full: pd.DataFrame, down_full: pd.DataFrame) -> pd.DataFrame:
    """Merge the two per-bar prediction frames into one unified log table.

    up_full / down_full are the outputs of each package's evaluate_full_on_slice
    (columns: timestamp, confidence, threshold, take, label, volatility_state,
    is_weekend, hour). Returns a DataFrame with exactly LOG_COLUMNS.
    """
    if up_full.empty or down_full.empty:
        return pd.DataFrame(columns=LOG_COLUMNS)

    up = up_full.rename(columns={
        "confidence": "p_up", "threshold": "th_up", "take": "take_up",
        "label": "label_up",
    })
    down = down_full.rename(columns={
        "confidence": "p_down", "threshold": "th_down", "take": "take_down",
        "label": "label_down",
    })[["timestamp", "p_down", "th_down", "take_down", "label_down"]]

    merged = up.merge(down, on="timestamp", how="inner")
    dropped = max(len(up_full), len(down_full)) - len(merged)
    if dropped > 0:
        logging.warning("detailed_log: %d bar(s) dropped - the up and down "
                        "sides did not cover identical timestamps.", dropped)
    if merged.empty:
        return pd.DataFrame(columns=LOG_COLUMNS)

    merged = merged.sort_values("timestamp").reset_index(drop=True)

    merged["team_decision"] = _classify(merged["take_up"], merged["take_down"])
    merged["bet_taken"] = merged["team_decision"].isin(["UP", "DOWN"]).astype(int)

    # The next candle's actual direction (modele_up's label == 1 means next is up).
    merged["next_candle"] = merged["label_up"].map({1: "green", 0: "red"})

    # result : only for taken bets. UP wins iff next is green; DOWN iff next red.
    won = (
        ((merged["team_decision"] == "UP") & (merged["label_up"] == 1))
        | ((merged["team_decision"] == "DOWN") & (merged["label_down"] == 1))
    )
    taken_mask = merged["bet_taken"] == 1
    merged["result"] = ""
    merged.loc[taken_mask, "result"] = won[taken_mask].map({True: "win", False: "loss"})

    # OHLC for the analyzed bar + the target bar.
    ohlc = load_ohlc(int(merged["timestamp"].min()), int(merged["timestamp"].max()))
    merged = _attach_ohlc(merged, ohlc)

    merged["timestamp_ms"] = merged["timestamp"].astype("int64")
    merged["datetime_utc"] = (
        pd.to_datetime(merged["timestamp_ms"], unit="ms", utc=True)
        .dt.strftime("%Y-%m-%d %H:%M")
    )
    return merged.reindex(columns=LOG_COLUMNS)


def write_detailed_logs(per_bar: pd.DataFrame, logs_dir: Path) -> dict[str, int]:
    """Write the three CSVs into logs_dir. Overwrites each run (a fresh snapshot
    of the latest backtest). Returns {filename: row_count} for the run report.
    """
    logs_dir.mkdir(parents=True, exist_ok=True)
    per_bar = per_bar.reindex(columns=LOG_COLUMNS)
    skipped = per_bar[per_bar["bet_taken"] == 0]
    taken = per_bar[per_bar["bet_taken"] == 1]

    per_bar.to_csv(logs_dir / LOG_FILES["all"], index=False)
    skipped.to_csv(logs_dir / LOG_FILES["skipped"], index=False)
    taken.to_csv(logs_dir / LOG_FILES["taken"], index=False)

    return {
        LOG_FILES["all"]: len(per_bar),
        LOG_FILES["skipped"]: len(skipped),
        LOG_FILES["taken"]: len(taken),
    }
