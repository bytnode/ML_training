import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from config import MODEL_OUTPUT, BREAKDOWN_THRESHOLD

_model = None
_thresholds = None  # sidecar JSON cache : {"base_threshold": ..., "deltas": {...}, "metadata": {...}}


def get_model():
    """Charge le modèle XGBoost depuis le disque (lazy, mis en cache)."""
    global _model
    if _model is None:
        m = xgb.XGBClassifier()
        m.load_model(MODEL_OUTPUT)
        _model = m
    return _model


def get_thresholds():
    """
    Charge le sidecar JSON des seuils adaptatifs (lazy, mis en cache).
    Renvoie None si le fichier n'existe pas (modele entraine sans adaptive threshold,
    ou fichier supprime). Dans ce cas, les callers doivent fallback sur BREAKDOWN_THRESHOLD.
    """
    global _thresholds
    if _thresholds is None:
        sidecar = Path(MODEL_OUTPUT).with_suffix(".thresholds.json")
        if sidecar.exists():
            with open(sidecar) as f:
                _thresholds = json.load(f)
        else:
            _thresholds = {}  # marker "deja cherche, pas trouve"
    return _thresholds or None


def _to_dataframe(features):
    if isinstance(features, pd.DataFrame):
        return features
    if isinstance(features, dict):
        return pd.DataFrame([features])
    if isinstance(features, list):
        return pd.DataFrame(features)
    raise TypeError(f"features doit être dict | list[dict] | DataFrame, reçu {type(features).__name__}")


def predict_proba(features):
    """
    Probabilité de retournement (classe 1) pour chaque ligne.

    features : dict (1 setup) | list[dict] (batch) | pd.DataFrame.
    Doit contenir au minimum les colonnes utilisées à l'entraînement.
    Les autres colonnes (timestamp, label, ...) sont ignorées.
    """
    model = get_model()
    df = _to_dataframe(features)
    feature_cols = model.get_booster().feature_names
    return model.predict_proba(df[feature_cols])[:, 1]


def _bucket_threshold(volatility_state, is_weekend, sidecar):
    """Renvoie le threshold effectif pour un bucket (base + delta)."""
    base = sidecar["base_threshold"]
    key = f"{int(volatility_state)}_{int(is_weekend)}"
    delta = sidecar.get("deltas", {}).get(key, 0.0)
    return base + delta


def should_take_bet(features, threshold=None):
    """
    Decide quelles lignes prendre.

    threshold = None (defaut) : utilise les seuils adaptatifs par bucket
        (volatility_state, is_weekend) lus depuis le sidecar JSON. Si le sidecar
        est absent, fallback sur BREAKDOWN_THRESHOLD pour toutes les lignes.

    threshold = float : utilise ce threshold fixe pour toutes les lignes
        (court-circuite l'adaptation, utile pour le mode legacy ou les tests).
    """
    proba = predict_proba(features)
    if threshold is not None:
        return proba >= threshold

    df = _to_dataframe(features)
    sidecar = get_thresholds()
    if sidecar is None:
        return proba >= BREAKDOWN_THRESHOLD

    if "volatility_state" not in df.columns or "is_weekend" not in df.columns:
        # Pas de bucket info dispo -> base threshold du sidecar
        return proba >= sidecar["base_threshold"]

    base = sidecar["base_threshold"]
    deltas = sidecar.get("deltas", {})
    vol = df["volatility_state"].astype(int).to_numpy()
    wknd = df["is_weekend"].astype(int).to_numpy()
    deltas_per_row = np.array(
        [deltas.get(f"{int(v)}_{int(w)}", 0.0) for v, w in zip(vol, wknd)],
        dtype=float,
    )
    return proba >= (base + deltas_per_row)


if __name__ == "__main__":
    from config import CSV_PATH
    last = pd.read_csv(CSV_PATH).sort_values("timestamp").tail(1)
    p = float(predict_proba(last)[0])
    sidecar = get_thresholds()
    if sidecar is not None and "volatility_state" in last.columns and "is_weekend" in last.columns:
        th_eff = _bucket_threshold(
            int(last["volatility_state"].iloc[0]),
            int(last["is_weekend"].iloc[0]),
            sidecar,
        )
        take = p >= th_eff
        print(f"Dernière bougie : proba = {p:.4f} | seuil adaptatif = {th_eff:.4f} | take = {take}")
    else:
        take = p >= BREAKDOWN_THRESHOLD
        print(f"Dernière bougie : proba = {p:.4f} | seuil fixe = {BREAKDOWN_THRESHOLD} | take = {take}")
