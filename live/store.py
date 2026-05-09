"""CSV append + outcome update for decisions.csv, bets.csv, wallet_history.csv.

Schemas defined in docs/superpowers/specs/2026-05-08-polymarket-live-trader-design.md.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

import settings

# Column order for the 42 features (must match training_data.csv exactly).
FEATURE_COLS_15M = [
    "seq_len", "seq_dir",
    "body_ratio_last", "body_ratio_avg", "upper_wick", "lower_wick",
    "volume_ratio_last", "volume_ratio_avg",
    "move_pct", "move_atr",
    "rsi", "rsi_change_5", "obv_slope_10", "obv_vs_ema",
    "atr_ratio", "volatility_state",
    "dist_ema20", "dist_ema50", "dist_ema200",
    "hour_utc", "day_of_week", "is_weekend",
]
FEATURE_COLS_MTF = [
    f"{kind}_{iv}"
    for iv in ["30m", "1h", "4h", "1d"]
    for kind in ["open", "high", "low", "close", "vol"]
]

DECISIONS_COLS = (
    ["timestamp"]
    + FEATURE_COLS_15M
    + FEATURE_COLS_MTF
    + [
        "bet_direction", "confidence", "effective_threshold",
        "bucket_v", "bucket_w",
        "taken", "is_chosen",
        "bet_id", "created_at_iso",
    ]
)

BETS_COLS = [
    "bet_id", "decision_timestamp", "created_at_iso",
    "target_window_start_ms", "target_window_end_ms",
    "polymarket_slug", "polymarket_condition_id", "polymarket_token_id",
    "bet_direction", "bet_size_usdc",
    "polymarket_order_id", "order_status",
    "fill_price_avg", "shares_filled", "usdc_spent", "fee_usdc",
    "outcome", "outcome_resolved_at_iso", "realized_pnl_usdc",
    "wallet_balance_pre_usdc", "wallet_balance_post_usdc",
]

WALLET_HISTORY_COLS = [
    "timestamp_iso", "cycle_action",
    "usdc_balance", "matic_balance",
    "pending_bets_n", "pending_bets_exposure_usdc",
]


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _ensure_header(path: Path, columns: list[str]) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(columns)


def _append_rows(path: Path, columns: list[str], rows: list[dict]) -> None:
    _ensure_header(path, columns)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        for r in rows:
            # Provide blanks for missing fields rather than raising.
            w.writerow({c: r.get(c, "") for c in columns})


def append_decisions(rows: list[dict]) -> None:
    """One row per qualifying setup. is_chosen is set True only on the chosen one (if any)."""
    if not rows:
        return
    for r in rows:
        r.setdefault("created_at_iso", _utc_iso())
    _append_rows(settings.DECISIONS_CSV, DECISIONS_COLS, rows)


def append_bet(row: dict) -> None:
    """One row per real Polymarket order placed (filled, rejected, or failed)."""
    row.setdefault("created_at_iso", _utc_iso())
    _append_rows(settings.BETS_CSV, BETS_COLS, [row])


def update_bet_outcome(bet_id: str, outcome: str, realized_pnl_usdc: float,
                      resolved_at_iso: str | None = None) -> bool:
    """Update outcome / pnl / resolved_at for the row whose bet_id matches.

    Returns True if a row was updated, False otherwise.
    """
    path = settings.BETS_CSV
    if not path.exists():
        return False

    rows: list[dict] = []
    updated = False
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r.get("bet_id") == bet_id:
                r["outcome"] = outcome
                r["realized_pnl_usdc"] = f"{realized_pnl_usdc:.6f}"
                r["outcome_resolved_at_iso"] = resolved_at_iso or _utc_iso()
                updated = True
            rows.append(r)

    if updated:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=BETS_COLS, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow({c: r.get(c, "") for c in BETS_COLS})
    return updated


def read_pending_bets() -> list[dict]:
    """Returns the list of rows in bets.csv where outcome == 'pending'."""
    path = settings.BETS_CSV
    if not path.exists():
        return []
    with open(path, "r", newline="", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r.get("outcome") == "pending"]


def append_wallet_snapshot(snapshot: dict) -> None:
    snapshot.setdefault("timestamp_iso", _utc_iso())
    _append_rows(settings.WALLET_HISTORY_CSV, WALLET_HISTORY_COLS, [snapshot])
