"""I/O CSV pour l'orchestrateur live.

Deux fichiers geres :
- training_data_live.csv : 44 cols, mirror du schema de backtest/training_data.csv
- decisions.csv : 53 cols, audit self-contained des decisions Aure

Updates retroactifs (label, outcome) atomiques via tmp + os.replace.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from settings import TRAINING_DATA_LIVE, DECISIONS_CSV, INTERVAL_15M_MS

# Ordre des colonnes (figer pour rester compatible avec backtest/training_data.csv).
TRAINING_DATA_COLUMNS = [
    "timestamp", "seq_len", "seq_dir", "body_ratio_last", "body_ratio_avg",
    "upper_wick", "lower_wick", "volume_ratio_last", "volume_ratio_avg",
    "move_pct", "move_atr", "rsi", "rsi_change_5", "obv_slope_10",
    "obv_vs_ema", "atr_ratio", "volatility_state", "dist_ema20", "dist_ema50",
    "dist_ema200", "hour_utc", "day_of_week", "is_weekend",
    "open_30m", "high_30m", "low_30m", "close_30m", "vol_30m",
    "open_1h", "high_1h", "low_1h", "close_1h", "vol_1h",
    "open_4h", "high_4h", "low_4h", "close_4h", "vol_4h",
    "open_1d", "high_1d", "low_1d", "close_1d", "vol_1d",
    "label",
]

# 42 features = TRAINING_DATA_COLUMNS sans timestamp et sans label.
FEATURE_COLUMNS = [c for c in TRAINING_DATA_COLUMNS if c not in ("timestamp", "label")]

DECISION_COLUMNS = (
    ["timestamp"]
    + FEATURE_COLUMNS
    + [
        "confidence", "effective_threshold", "bucket_v", "bucket_w",
        "bet_direction", "taken", "notified_at", "target_open_time",
        "outcome", "outcome_resolved_at",
    ]
)


def _utc_iso_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _atomic_write(df: pd.DataFrame, path: Path) -> None:
    """Write df to path atomiquement via tmp + os.replace.

    Necessaire pour les updates retroactifs (label, outcome) qui rewrite
    le fichier entier ; un crash a mi-write laisserait sinon un CSV tronque.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def append_training_row(features: dict) -> None:
    """Append une ligne dans training_data_live.csv. label=NaN au depart.

    features doit contenir 'timestamp' + les 42 features. La fonction
    rejette les autres clefs et reordonne selon TRAINING_DATA_COLUMNS.
    """
    row = {col: features.get(col) for col in TRAINING_DATA_COLUMNS}
    row["label"] = pd.NA  # marquera label comme nullable
    df = pd.DataFrame([row], columns=TRAINING_DATA_COLUMNS)
    _append_with_header_if_missing(df, TRAINING_DATA_LIVE)


def append_decision_row(features: dict, decision: dict) -> None:
    """Append une ligne dans decisions.csv.

    decision doit contenir : confidence, effective_threshold, bucket_v,
    bucket_w, bet_direction, taken, notified_at, target_open_time. outcome
    est mis a 'pending' et outcome_resolved_at a NaN par cette fonction.
    """
    row = {col: features.get(col) for col in FEATURE_COLUMNS}
    row["timestamp"] = features["timestamp"]
    row["target_open_time"] = features["target_open_time"]
    row.update({
        "confidence": decision["confidence"],
        "effective_threshold": decision["effective_threshold"],
        "bucket_v": decision["bucket_v"],
        "bucket_w": decision["bucket_w"],
        "bet_direction": decision["bet_direction"],
        "taken": decision["taken"],
        "notified_at": decision.get("notified_at"),
        "outcome": "pending",
        "outcome_resolved_at": None,
    })
    df = pd.DataFrame([row], columns=DECISION_COLUMNS)
    _append_with_header_if_missing(df, DECISIONS_CSV)


def _append_with_header_if_missing(df: pd.DataFrame, path: Path) -> None:
    """Append df a path. Ecrit le header seulement si le fichier n'existe pas."""
    if path.exists():
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        df.to_csv(path, mode="w", header=True, index=False)


def update_pending_labels(klines_15m: list) -> int:
    """Resoud les rows de training_data_live.csv ou label est NaN.

    Pour chaque row pendante :
      - target_open_time = timestamp + 15min
      - cherche la bougie 15m d'open_time == target_open_time dans klines_15m
      - si trouvee : label = 1 si direction inverse seq_dir, 0 sinon
      - si pas trouvee (target pas encore close) : laisse NaN

    Returns: nombre de rows mises a jour.
    """
    if not TRAINING_DATA_LIVE.exists():
        return 0
    df = pd.read_csv(TRAINING_DATA_LIVE)
    if df.empty or "label" not in df.columns:
        return 0

    pending_mask = df["label"].isna()
    if not pending_mask.any():
        return 0

    candle_by_ts = {int(k[0]): k for k in klines_15m}
    updated = 0

    for idx in df.index[pending_mask]:
        ts = int(df.at[idx, "timestamp"])
        target_ts = ts + INTERVAL_15M_MS
        candle = candle_by_ts.get(target_ts)
        if candle is None:
            continue
        # candle = [open_time, o, h, l, c, v] — values may be strings (Bitstamp JSON)
        c_open = float(candle[1])
        c_close = float(candle[4])
        next_dir_up = c_close > c_open
        next_dir_down = c_close < c_open
        seq_dir_up = bool(df.at[idx, "seq_dir"])

        if seq_dir_up and next_dir_down:
            label = 1
        elif (not seq_dir_up) and next_dir_up:
            label = 1
        else:
            label = 0  # flat (close==open) inclus dans loss

        df.at[idx, "label"] = label
        updated += 1

    if updated:
        # Restaure le type int sur label (pandas peut avoir promu en float a cause des NaN restants).
        # On garde nullable Int64 pour pouvoir retenir des NaN sur les rows non encore resolues.
        df["label"] = df["label"].astype("Int64")
        _atomic_write(df, TRAINING_DATA_LIVE)
    return updated


def update_pending_outcomes(klines_15m: list) -> int:
    """Resoud les rows de decisions.csv ou outcome=='pending'.

    Pour chaque row pendante :
      - cherche la bougie 15m d'open_time == target_open_time
      - si trouvee : outcome = 'win' si direction == bet_direction, 'loss' sinon
                     outcome_resolved_at = utc iso now
      - sinon : laisse 'pending'

    Returns: nombre de rows mises a jour.
    """
    if not DECISIONS_CSV.exists():
        return 0
    df = pd.read_csv(DECISIONS_CSV)
    if df.empty or "outcome" not in df.columns:
        return 0

    pending_mask = df["outcome"] == "pending"
    if not pending_mask.any():
        return 0

    # outcome_resolved_at is None for all pending rows and pandas reads it as
    # float64. Cast to object so that assigning an ISO string doesn't raise a
    # FutureWarning (and will raise an error in pandas >= 3.0).
    df["outcome_resolved_at"] = df["outcome_resolved_at"].astype(object)

    candle_by_ts = {int(k[0]): k for k in klines_15m}
    now_iso = _utc_iso_now()
    updated = 0

    for idx in df.index[pending_mask]:
        target_ts = int(df.at[idx, "target_open_time"])
        candle = candle_by_ts.get(target_ts)
        if candle is None:
            continue

        bet_direction = df.at[idx, "bet_direction"]
        c_open = float(candle[1])
        c_close = float(candle[4])
        candle_dir = "up" if c_close > c_open else ("down" if c_close < c_open else "flat")

        if candle_dir == bet_direction:
            outcome = "win"
        else:
            outcome = "loss"  # flat inclus dans loss (coherent avec extract_features)

        df.at[idx, "outcome"] = outcome
        df.at[idx, "outcome_resolved_at"] = now_iso
        updated += 1

    if updated:
        _atomic_write(df, DECISIONS_CSV)
    return updated


if __name__ == "__main__":
    # Smoke test : append une row factice dans chaque CSV puis affiche
    # les CSV pour visuel.
    print("Smoke test store.py")
    fake_features = {col: 0.0 for col in FEATURE_COLUMNS}
    fake_features.update({"timestamp": 1746000000000, "seq_len": 2, "seq_dir": 1,
                          "body_ratio_last": 0.7, "is_weekend": 0, "volatility_state": 1,
                          "target_open_time": 1746000000000 + INTERVAL_15M_MS})
    print("  Appending training row...")
    append_training_row(fake_features)
    print("  Appending decision row...")
    append_decision_row(fake_features, {
        "confidence": 0.65, "effective_threshold": 0.58,
        "bucket_v": 1, "bucket_w": 0, "bet_direction": "down",
        "taken": True, "notified_at": _utc_iso_now(),
    })
    print(f"  training_data_live.csv : {TRAINING_DATA_LIVE.stat().st_size} bytes")
    print(f"  decisions.csv : {DECISIONS_CSV.stat().st_size} bytes")
    print("  Done. Inspect CSVs manually if needed.")
