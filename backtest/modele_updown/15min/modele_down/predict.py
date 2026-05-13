"""
Inference en ligne du modele DOWN-only avec decision adaptative.

Le modele est entraine a predire P(next is down). A l'inference :
  - confidence = P(next is down) in [0, 1]
  - take       = True ssi confidence >= seuil_effectif(row)
  - direction  = TOUJOURS "down" (par construction du modele)

Sidecar V1/V2 (calibration par quantile) :
  - V1 ("asym_wknd_excl") : 1 seuil par axe weekday/weekend + skip cells
  - V2 ("asym_wknd_excl_hour_boost") : V1 + boost horaire

API librairie :
    from predict import predict_proba, decide
    p_down = predict_proba(features)
    confidence, take = decide(features)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from config import MODEL_OUTPUT

_model = None
_thresholds = None


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
    """P(next is down) pour chaque ligne (in [0, 1])."""
    model = get_model()
    df = _to_dataframe(features)
    feature_cols = model.get_booster().feature_names
    return model.predict_proba(df[feature_cols])[:, 1]


def _hour_from_ts_ms(ts_ms) -> int:
    """Renvoie l'heure UTC (0..23) pour un timestamp ms (single int/float)."""
    ts_ms = int(ts_ms)
    return int(((ts_ms // (15 * 60 * 1000)) % 96) // 4)


def _bucket_threshold(volatility_state, is_weekend, sidecar, ts_ms=None) -> float:
    """Threshold effectif pour une row. Supporte 2 formats sidecar :
      - V1 : {"strategy": "asym_wknd_excl", "thresholds_per_wknd": {...}, "skip_cells": [...]}
      - V2 : V1 + {"hour_boost": {"good_hours": [...], "delta": ...}}

    Pour V2, ts_ms est requis pour calculer l'heure UTC.
    """
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


def decide(features):
    """
    Decision adaptative DOWN-only. Renvoie 2 arrays :
      confidence : P(next is down) in [0, 1]
      take       : bool, True ssi confidence >= seuil(row)

    Direction implicite : toujours "down".
    """
    confidence = np.asarray(predict_proba(features), dtype=float)
    df = _to_dataframe(features)
    sidecar = get_thresholds()
    strategy = sidecar.get("strategy")
    if strategy not in ("asym_wknd_excl", "asym_wknd_excl_hour_boost"):
        raise ValueError(f"Sidecar strategy={strategy!r} non supporte. "
                         f"Attendu : 'asym_wknd_excl' ou 'asym_wknd_excl_hour_boost'. "
                         f"Re-run 'python train.py' pour regenerer le sidecar.")

    if "volatility_state" not in df.columns or "is_weekend" not in df.columns:
        raise ValueError("decide() requiert les colonnes 'volatility_state' et 'is_weekend'.")

    vol = df["volatility_state"].astype(int).to_numpy()
    wknd = df["is_weekend"].astype(int).to_numpy()
    skip_cells = {(int(v), int(w)) for v, w in sidecar.get("skip_cells", [])}
    th_map = sidecar.get("thresholds_per_wknd", {})
    eff_th = np.array([
        1.0 if (int(v), int(w)) in skip_cells else float(th_map.get(str(int(w)), 1.0))
        for v, w in zip(vol, wknd)
    ], dtype=float)
    if strategy == "asym_wknd_excl_hour_boost" and "timestamp" in df.columns:
        hb = sidecar.get("hour_boost", {})
        good_hours = set(int(h) for h in hb.get("good_hours", []))
        delta_boost = float(hb.get("delta", 0.0))
        ts = df["timestamp"].to_numpy()
        hour = ((ts // (15 * 60 * 1000)) % 96 // 4).astype(int)
        boost_mask = np.array([int(h) in good_hours for h in hour]) & (eff_th < 1.0)
        eff_th[boost_mask] -= delta_boost

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
        ts_ms=int(last["timestamp"].iloc[0]),
    )
    confidence, take = decide(last)
    print(f"Derniere bougie :")
    print(f"  P(next is down) = {p_down:.4f}")
    print(f"  seuil effectif  = {eff_th:.4f}")
    print(f"  direction       = down  (modele DOWN-only)")
    print(f"  take            = {bool(take[0])}")
