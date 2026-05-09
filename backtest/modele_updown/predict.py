"""
Inférence en ligne du modèle DOWN-only avec décision adaptative.

Le modèle est entraîné à prédire P(next is down). À l'inférence :
  - confidence = P(next is down) ∈ [0, 1]
  - take       = True ssi confidence >= base + delta(bucket)
  - direction  = TOUJOURS "down" (par construction du modèle)

API librairie :
    from predict import predict_proba, decide
    p_down = predict_proba(features)            # P(next is down)
    confidence, take = decide(features)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from config import MODEL_OUTPUT, ADAPTIVE_TARGET_WR

_model = None
_thresholds = None  # {"deltas": {...}, "metadata": {...}}


def get_model():
    global _model
    if _model is None:
        m = xgb.XGBClassifier()
        m.load_model(MODEL_OUTPUT)
        _model = m
    return _model


def get_thresholds():
    global _thresholds
    if _thresholds is None:
        sidecar = Path(MODEL_OUTPUT).with_suffix(".thresholds.json")
        if not sidecar.exists():
            raise FileNotFoundError(
                f"{sidecar} introuvable. Lance 'python train.py' pour le régénérer."
            )
        with open(sidecar) as f:
            _thresholds = json.load(f)
    return _thresholds


def _to_dataframe(features):
    if isinstance(features, pd.DataFrame):
        return features
    if isinstance(features, dict):
        return pd.DataFrame([features])
    if isinstance(features, list):
        return pd.DataFrame(features)
    raise TypeError(f"features doit être dict | list[dict] | DataFrame, reçu {type(features).__name__}")


def predict_proba(features):
    """P(next is down) pour chaque ligne (∈ [0, 1])."""
    model = get_model()
    df = _to_dataframe(features)
    feature_cols = model.get_booster().feature_names
    return model.predict_proba(df[feature_cols])[:, 1]


def _bucket_threshold(volatility_state, is_weekend, sidecar) -> float:
    """Threshold effectif pour un bucket = ADAPTIVE_TARGET_WR + delta(bucket)."""
    key = f"{int(volatility_state)}_{int(is_weekend)}"
    delta = sidecar.get("deltas", {}).get(key, 0.0)
    if delta is None:
        delta = 0.0
    return ADAPTIVE_TARGET_WR + delta


def decide(features):
    """
    Décision adaptative DOWN-only. Renvoie 2 arrays :
      confidence : P(next is down) ∈ [0, 1]
      take       : bool, True ssi confidence >= base + delta(bucket)

    Direction implicite : toujours "down".
    """
    confidence = np.asarray(predict_proba(features), dtype=float)
    df = _to_dataframe(features)
    sidecar = get_thresholds()

    deltas = sidecar.get("deltas", {})
    if "volatility_state" not in df.columns or "is_weekend" not in df.columns:
        eff_th = np.full_like(confidence, ADAPTIVE_TARGET_WR)
    else:
        vol = df["volatility_state"].astype(int).to_numpy()
        wknd = df["is_weekend"].astype(int).to_numpy()
        delta_per_row = np.array(
            [(deltas.get(f"{int(v)}_{int(w)}") or 0.0) for v, w in zip(vol, wknd)],
            dtype=float,
        )
        eff_th = ADAPTIVE_TARGET_WR + delta_per_row

    take = confidence >= eff_th
    return confidence, take


if __name__ == "__main__":
    test_pred_csv = Path(__file__).resolve().parent / "logs" / "test_predictions.csv"
    if not test_pred_csv.exists():
        raise FileNotFoundError(
            f"{test_pred_csv} introuvable. Lance d'abord 'python train.py'."
        )
    last = pd.read_csv(test_pred_csv).sort_values("timestamp").tail(1)
    p_down = float(predict_proba(last)[0])
    sidecar = get_thresholds()
    eff_th = _bucket_threshold(
        int(last["volatility_state"].iloc[0]),
        int(last["is_weekend"].iloc[0]),
        sidecar,
    )
    confidence, take = decide(last)
    print(f"Dernière bougie :")
    print(f"  P(next is down) = {p_down:.4f}")
    print(f"  seuil effectif  = {eff_th:.4f}")
    print(f"  direction       = down  (modèle DOWN-only)")
    print(f"  take            = {bool(take[0])}")
