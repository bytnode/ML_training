"""Standardized inference API for the modele_15m model.

This is the contract between modele_15m and backtest/main.py (the team
backtest orchestrator). Each model package exposes:
    NAME : str
    evaluate_on_slice(start_ms: int, end_ms: int) -> pd.DataFrame

For modele_15m, this implementation reads logs/test_predictions.csv produced
by train.py, applies the adaptive threshold sidecar to filter taken bets,
and derives the bet direction from seq_dir.

Future models (e.g. modele_onchain) implement evaluate_on_slice however
they want (runtime fetch + predict, etc.). main.py only sees the
standardized DataFrame.
"""
import json
from pathlib import Path

import pandas as pd

from config import ADAPTIVE_TARGET_WR

NAME = "modele_15m"

_MODEL_DIR = Path(__file__).resolve().parent
_TEST_PRED_CSV = _MODEL_DIR / "logs" / "test_predictions.csv"
_SIDECAR_JSON = _MODEL_DIR / "btc_reversal_model.thresholds.json"


def _bucket_threshold(volatility_state: int, is_weekend: int, sidecar: dict) -> float:
    """Threshold effectif pour un bucket : ADAPTIVE_TARGET_WR + delta(bucket).
    La base vient de config.ADAPTIVE_TARGET_WR, pas du sidecar."""
    key = f"{int(volatility_state)}_{int(is_weekend)}"
    delta = sidecar.get("deltas", {}).get(key, 0.0)
    if delta is None:  # bucket fell back to insufficient_data at calibration
        delta = 0.0
    return ADAPTIVE_TARGET_WR + delta


def evaluate_on_slice(start_ms: int, end_ms: int) -> pd.DataFrame:
    """Renvoie 1 row par setup ou le modele aurait pris le pari dans la fenetre.

    Args:
        start_ms, end_ms: bornes inclusives sur la colonne `timestamp` (ms).

    Returns:
        DataFrame avec colonnes : timestamp (int ms), direction ('up'|'down'),
        confidence (float), label (int 0|1).
        DataFrame vide si aucun pari pris dans la fenetre.

    Raises:
        FileNotFoundError si test_predictions.csv ou le sidecar des seuils
        adaptatifs est absent (= modele jamais entraine, ou logs/sidecar supprime).
    """
    if not _TEST_PRED_CSV.exists():
        raise FileNotFoundError(
            f"{_TEST_PRED_CSV} introuvable. Lance d'abord 'cd modele_15m && python train.py' "
            f"pour generer test_predictions.csv."
        )
    if not _SIDECAR_JSON.exists():
        raise FileNotFoundError(
            f"{_SIDECAR_JSON} introuvable. Lance 'cd modele_15m && python train.py' "
            f"pour regenerer le sidecar des seuils adaptatifs."
        )

    df = pd.read_csv(_TEST_PRED_CSV)
    df = df[(df["timestamp"] >= start_ms) & (df["timestamp"] <= end_ms)].copy()
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "direction", "confidence", "label"])

    with open(_SIDECAR_JSON) as f:
        sidecar = json.load(f)

    df["_eff_th"] = df.apply(
        lambda r: _bucket_threshold(int(r["volatility_state"]), int(r["is_weekend"]), sidecar),
        axis=1,
    )

    df = df[df["confidence"] >= df["_eff_th"]].copy()
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "direction", "confidence", "label"])

    # seq_dir=1 = up-streak setup -> bet on reversal = next candle goes DOWN
    # seq_dir=0 = down-streak setup -> bet on reversal = next candle goes UP
    df["direction"] = df["seq_dir"].apply(lambda d: "down" if int(d) == 1 else "up")

    return df[["timestamp", "direction", "confidence", "label"]].reset_index(drop=True)


if __name__ == "__main__":
    # Smoke test : evalue sur tout test_predictions.csv
    import time
    end = int(time.time() * 1000)
    start = 0
    res = evaluate_on_slice(start, end)
    print(f"name={NAME} setups_taken={len(res)} wr={res['label'].mean():.4f}")
    print(res.head())
