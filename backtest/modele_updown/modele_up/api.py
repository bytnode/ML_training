"""Standardized inference API for the modele_up model (UP-only).

Contrat :
    NAME : str
    evaluate_on_slice(start_ms: int, end_ms: int) -> pd.DataFrame

Le modèle est UP-only :
  - Sortie brute = P(next is up) ∈ [0, 1] (= "confidence")
  - Direction = toujours "up" pour les paris pris
  - On prend le pari ssi P(up) >= ADAPTIVE_TARGET_WR + delta(bucket)
  - Le label retourné = 1 ssi le pari aurait été correct
        (= la bougie suivante a effectivement été up)
"""
import json
from pathlib import Path

import pandas as pd

NAME = "modele_up"

_MODEL_DIR = Path(__file__).resolve().parent
_TEST_PRED_CSV = _MODEL_DIR / "logs" / "test_predictions.csv"
_SIDECAR_JSON = _MODEL_DIR / "btc_updown_model.thresholds.json"


def _hour_from_ts_ms(ts_ms) -> int:
    ts_ms = int(ts_ms)
    return int(((ts_ms // (15 * 60 * 1000)) % 96) // 4)


def _bucket_threshold(volatility_state: int, is_weekend: int, sidecar: dict,
                      ts_ms=None) -> float:
    """Supporte 2 formats sidecar : V1 (asym_wknd_excl) et V2 (asym_wknd_excl_hour_boost)."""
    strategy = sidecar.get("strategy")
    if strategy not in ("asym_wknd_excl", "asym_wknd_excl_hour_boost"):
        raise ValueError(f"Sidecar strategy={strategy!r} non supporte. "
                         f"Attendu : 'asym_wknd_excl' ou 'asym_wknd_excl_hour_boost'.")
    skip_cells = {(int(v), int(w)) for v, w in sidecar.get("skip_cells", [])}
    if (int(volatility_state), int(is_weekend)) in skip_cells:
        return 1.0
    th_map = sidecar.get("thresholds_per_wknd", {})
    T = float(th_map.get(str(int(is_weekend)), 1.0))
    if strategy == "asym_wknd_excl_hour_boost" and ts_ms is not None:
        hb = sidecar.get("hour_boost", {})
        good_hours = set(int(h) for h in hb.get("good_hours", []))
        delta = float(hb.get("delta", 0.0))
        if _hour_from_ts_ms(ts_ms) in good_hours:
            T -= delta
    return T


def evaluate_on_slice(start_ms: int, end_ms: int) -> pd.DataFrame:
    """Renvoie 1 row par setup où le modèle aurait pris un pari UP.

    Args:
        start_ms, end_ms: bornes inclusives sur la colonne `timestamp` (ms).

    Returns:
        DataFrame avec colonnes : timestamp (ms), direction='up',
        confidence (= P(up)), label (int 0|1 = pari correct).
    """
    if not _TEST_PRED_CSV.exists():
        raise FileNotFoundError(
            f"{_TEST_PRED_CSV} introuvable. Lance 'cd backtest/modele_updown/modele_up && python train.py'."
        )
    if not _SIDECAR_JSON.exists():
        raise FileNotFoundError(
            f"{_SIDECAR_JSON} introuvable. Lance 'cd backtest/modele_updown/modele_up && python train.py'."
        )

    df = pd.read_csv(_TEST_PRED_CSV)
    df = df[(df["timestamp"] >= start_ms) & (df["timestamp"] <= end_ms)].copy()
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "direction", "confidence", "label"])

    with open(_SIDECAR_JSON) as f:
        sidecar = json.load(f)

    # Seuil effectif par row (bucket vol × wknd, + hour-boost si strategy V2)
    df["_eff_th"] = df.apply(
        lambda r: _bucket_threshold(
            int(r["volatility_state"]), int(r["is_weekend"]), sidecar,
            ts_ms=int(r["timestamp"]),
        ),
        axis=1,
    )
    df = df[df["confidence"] >= df["_eff_th"]].copy()
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "direction", "confidence", "label"])

    # Tous les paris sont UP. Label brut (1 = next was up) = label du pari.
    df["direction"] = "up"
    df["label"] = df["label"].astype(int)

    return df[["timestamp", "direction", "confidence", "label"]].reset_index(drop=True)


if __name__ == "__main__":
    import time
    end = int(time.time() * 1000)
    res = evaluate_on_slice(0, end)
    if res.empty:
        print(f"name={NAME} setups_taken=0")
    else:
        print(f"name={NAME} setups_taken={len(res)} (all up) wr={res['label'].mean():.4f}")
        print(res.head())
