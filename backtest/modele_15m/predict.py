import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from config import MODEL_OUTPUT, ADAPTIVE_TARGET_WR

_model = None
_thresholds = None  # sidecar JSON cache : {"deltas": {...}, "metadata": {...}}


def get_model():
    """Charge le modele XGBoost depuis le disque (lazy, mis en cache)."""
    global _model
    if _model is None:
        m = xgb.XGBClassifier()
        m.load_model(MODEL_OUTPUT)
        _model = m
    return _model


def get_thresholds():
    """
    Charge le sidecar JSON des deltas adaptatifs (lazy, mis en cache).
    Le sidecar est obligatoire : si absent, raise FileNotFoundError. Pas de fallback.
    """
    global _thresholds
    if _thresholds is None:
        sidecar = Path(MODEL_OUTPUT).with_suffix(".thresholds.json")
        if not sidecar.exists():
            raise FileNotFoundError(
                f"{sidecar} introuvable. Lance 'python train.py' pour le regenerer."
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
    raise TypeError(f"features doit etre dict | list[dict] | DataFrame, recu {type(features).__name__}")


def predict_proba(features):
    """
    Probabilite de retournement (classe 1) pour chaque ligne.

    features : dict (1 setup) | list[dict] (batch) | pd.DataFrame.
    Doit contenir au minimum les colonnes utilisees a l'entrainement.
    Les autres colonnes (timestamp, label, ...) sont ignorees.
    """
    model = get_model()
    df = _to_dataframe(features)
    feature_cols = model.get_booster().feature_names
    return model.predict_proba(df[feature_cols])[:, 1]


def _bucket_threshold(volatility_state, is_weekend, sidecar):
    """Renvoie le threshold effectif pour un bucket : ADAPTIVE_TARGET_WR + delta."""
    key = f"{int(volatility_state)}_{int(is_weekend)}"
    delta = sidecar.get("deltas", {}).get(key, 0.0)
    if delta is None:  # bucket fell back to insufficient_data at calibration
        delta = 0.0
    return ADAPTIVE_TARGET_WR + delta


def should_take_bet(features):
    """
    Decide quelles lignes prendre, sous la regle adaptative.

    Base = ADAPTIVE_TARGET_WR (config). Delta par bucket lu depuis le sidecar JSON.
    Le sidecar est obligatoire : raise FileNotFoundError si absent.
    Si volatility_state ou is_weekend manquent dans features, base seule (delta=0).
    """
    proba = predict_proba(features)
    df = _to_dataframe(features)
    sidecar = get_thresholds()

    if "volatility_state" not in df.columns or "is_weekend" not in df.columns:
        return proba >= ADAPTIVE_TARGET_WR

    deltas = sidecar.get("deltas", {})
    vol = df["volatility_state"].astype(int).to_numpy()
    wknd = df["is_weekend"].astype(int).to_numpy()
    deltas_per_row = np.array(
        [
            (deltas.get(f"{int(v)}_{int(w)}") or 0.0)
            for v, w in zip(vol, wknd)
        ],
        dtype=float,
    )
    return proba >= (ADAPTIVE_TARGET_WR + deltas_per_row)


if __name__ == "__main__":
    from config import CSV_PATH
    last = pd.read_csv(CSV_PATH).sort_values("timestamp").tail(1)
    p = float(predict_proba(last)[0])
    sidecar = get_thresholds()
    th_eff = _bucket_threshold(
        int(last["volatility_state"].iloc[0]),
        int(last["is_weekend"].iloc[0]),
        sidecar,
    )
    take = p >= th_eff
    print(f"Derniere bougie : proba = {p:.4f} | seuil adaptatif = {th_eff:.4f} | take = {take}")
