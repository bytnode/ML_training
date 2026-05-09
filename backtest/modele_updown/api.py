"""Standardized inference API for the modele_updown model (DOWN-only).

Contrat :
    NAME : str
    evaluate_on_slice(start_ms: int, end_ms: int) -> pd.DataFrame

Le modèle est DOWN-only :
  - Sortie brute = P(next is down) ∈ [0, 1] (= "confidence")
  - Direction = toujours "down" pour les paris pris
  - On prend le pari ssi P(down) >= ADAPTIVE_TARGET_WR + delta(bucket)
  - Le label retourné = 1 ssi le pari aurait été correct
        (= la bougie suivante a effectivement été down)
"""
import json
from pathlib import Path

import pandas as pd

from config import ADAPTIVE_TARGET_WR

NAME = "modele_updown"

_MODEL_DIR = Path(__file__).resolve().parent
_TEST_PRED_CSV = _MODEL_DIR / "logs" / "test_predictions.csv"
_SIDECAR_JSON = _MODEL_DIR / "btc_updown_model.thresholds.json"


def _bucket_threshold(volatility_state: int, is_weekend: int, sidecar: dict) -> float:
    key = f"{int(volatility_state)}_{int(is_weekend)}"
    delta = sidecar.get("deltas", {}).get(key, 0.0)
    if delta is None:
        delta = 0.0
    return ADAPTIVE_TARGET_WR + delta


def evaluate_on_slice(start_ms: int, end_ms: int) -> pd.DataFrame:
    """Renvoie 1 row par setup où le modèle aurait pris un pari DOWN.

    Args:
        start_ms, end_ms: bornes inclusives sur la colonne `timestamp` (ms).

    Returns:
        DataFrame avec colonnes : timestamp (ms), direction='down',
        confidence (= P(down)), label (int 0|1 = pari correct).
    """
    if not _TEST_PRED_CSV.exists():
        raise FileNotFoundError(
            f"{_TEST_PRED_CSV} introuvable. Lance 'cd modele_updown && python train.py'."
        )
    if not _SIDECAR_JSON.exists():
        raise FileNotFoundError(
            f"{_SIDECAR_JSON} introuvable. Lance 'cd modele_updown && python train.py'."
        )

    df = pd.read_csv(_TEST_PRED_CSV)
    df = df[(df["timestamp"] >= start_ms) & (df["timestamp"] <= end_ms)].copy()
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "direction", "confidence", "label"])

    with open(_SIDECAR_JSON) as f:
        sidecar = json.load(f)

    # Seuil effectif par row (bucket vol × wknd)
    df["_eff_th"] = df.apply(
        lambda r: _bucket_threshold(int(r["volatility_state"]), int(r["is_weekend"]), sidecar),
        axis=1,
    )
    df = df[df["confidence"] >= df["_eff_th"]].copy()
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "direction", "confidence", "label"])

    # Tous les paris sont DOWN. Label brut (1 = next was down) = label du pari.
    df["direction"] = "down"
    df["label"] = df["label"].astype(int)

    return df[["timestamp", "direction", "confidence", "label"]].reset_index(drop=True)


if __name__ == "__main__":
    import time
    end = int(time.time() * 1000)
    res = evaluate_on_slice(0, end)
    if res.empty:
        print(f"name={NAME} setups_taken=0")
    else:
        print(f"name={NAME} setups_taken={len(res)} (all down) wr={res['label'].mean():.4f}")
        print(res.head())
