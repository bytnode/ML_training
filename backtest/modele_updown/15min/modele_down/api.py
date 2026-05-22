"""Standardized inference API for the modele_updown model (DOWN-only).

Contrat :
    NAME : str
    evaluate_on_slice(start_ms: int, end_ms: int) -> pd.DataFrame

Le modele est DOWN-only :
  - Sortie brute = P(next is down) in [0, 1] (= "confidence")
  - Direction = toujours "down" pour les paris pris
  - On prend le pari ssi P(down) >= seuil_effectif(row)
  - Le label retourne = 1 ssi le pari aurait ete correct
        (= la bougie suivante a effectivement ete down)

Sidecar V1/V2 (calibration par quantile) :
  - V1 ("asym_wknd_excl") : 1 seuil par axe weekday/weekend + skip cells
  - V2 ("asym_wknd_excl_hour_boost") : V1 + boost horaire

Cette version lit le PARQUET (~13.7 ans) et predit a la volee. Tu peux donc
evaluer n'importe quelle fenetre temporelle. Attention : les bougies utilisees
pendant le training apparaitront avec un WR optimiste (overfit). Le caller est
responsable de choisir une fenetre out-of-sample s'il veut un chiffre non biaise.
"""
import json
from pathlib import Path

import pandas as pd
import xgboost as xgb

NAME = "modele_updown"

_MODEL_DIR = Path(__file__).resolve().parent
_PARQUET_PATH = _MODEL_DIR / "DB" / "btc_multi_tf_15m.parquet"
_MODEL_PATH   = _MODEL_DIR / "btc_updown_model.json"
_SIDECAR_JSON = _MODEL_DIR / "btc_updown_model.thresholds.json"

# Module-level caches (loaded lazily, kept alive between calls).
_model: xgb.XGBClassifier | None = None
_sidecar: dict | None = None
_parquet: pd.DataFrame | None = None


def _load_model() -> xgb.XGBClassifier:
    global _model
    if _model is None:
        if not _MODEL_PATH.exists():
            raise FileNotFoundError(
                f"{_MODEL_PATH} introuvable. Lance 'cd 15min/modele_down && python train.py'."
            )
        m = xgb.XGBClassifier()
        m.load_model(str(_MODEL_PATH))
        _model = m
    return _model


def _load_sidecar() -> dict:
    global _sidecar
    if _sidecar is None:
        if not _SIDECAR_JSON.exists():
            raise FileNotFoundError(
                f"{_SIDECAR_JSON} introuvable. Lance 'cd 15min/modele_down && python train.py'."
            )
        with open(_SIDECAR_JSON) as f:
            _sidecar = json.load(f)
    return _sidecar


def _load_parquet() -> pd.DataFrame:
    global _parquet
    if _parquet is None:
        if not _PARQUET_PATH.exists():
            raise FileNotFoundError(
                f"{_PARQUET_PATH} introuvable. Lance 'cd 15min/modele_down && python data.py' "
                f"puis les patch scripts."
            )
        _parquet = pd.read_parquet(_PARQUET_PATH).sort_values("timestamp_ms").reset_index(drop=True)
    return _parquet


def _hour_from_ts_ms(ts_ms) -> int:
    ts_ms = int(ts_ms)
    return int(((ts_ms // (15 * 60 * 1000)) % 96) // 4)


def _bucket_threshold(volatility_state: int, is_weekend: int, sidecar: dict,
                      ts_ms=None) -> float:
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


# Per-bar prediction frame : partage par evaluate_on_slice et
# evaluate_full_on_slice pour que les vues taken-only et full ne divergent pas.
_FULL_COLUMNS = ["timestamp", "confidence", "threshold", "take", "label",
                 "volatility_state", "is_weekend", "hour"]


def _predict_all_bars(start_ms: int, end_ms: int) -> pd.DataFrame:
    """Predit chaque bougie de la fenetre [start_ms, end_ms] (bornes inclusives
    sur timestamp_ms).

    1 row par bougie ayant un label valide et des features completes. Colonnes =
    _FULL_COLUMNS : timestamp, confidence (= P(down)), threshold (seuil effectif
    de la bougie), take (1 ssi confidence >= threshold), label, volatility_state,
    is_weekend, hour (0-23 UTC).
    """
    empty = pd.DataFrame(columns=_FULL_COLUMNS)

    df = _load_parquet()
    df = df[(df["timestamp_ms"] >= start_ms) & (df["timestamp_ms"] <= end_ms)].copy()
    if df.empty:
        return empty

    df = df.dropna(subset=["label"])
    if df.empty:
        return empty

    model = _load_model()
    feature_cols = model.get_booster().feature_names

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"Parquet missing model features: {missing}")
    df = df.dropna(subset=feature_cols)
    if df.empty:
        return empty

    df["confidence"] = model.predict_proba(df[feature_cols])[:, 1]

    sidecar = _load_sidecar()
    df["threshold"] = df.apply(
        lambda r: _bucket_threshold(
            int(r["volatility_state"]), int(r["is_weekend"]), sidecar,
            ts_ms=int(r["timestamp_ms"]),
        ),
        axis=1,
    )
    df["take"] = (df["confidence"] >= df["threshold"]).astype(int)
    df["timestamp"] = df["timestamp_ms"].astype("int64")
    df["label"] = df["label"].astype(int)
    df["volatility_state"] = df["volatility_state"].astype(int)
    df["is_weekend"] = df["is_weekend"].astype(int)
    df["hour"] = df["timestamp_ms"].map(_hour_from_ts_ms).astype(int)
    return df[_FULL_COLUMNS].reset_index(drop=True)


def evaluate_on_slice(start_ms: int, end_ms: int) -> pd.DataFrame:
    """Renvoie 1 row par setup ou le modele aurait pris un pari DOWN, sur la
    fenetre [start_ms, end_ms] (bornes inclusives sur timestamp_ms).

    La fenetre peut couvrir n'importe quelle periode du parquet (~13.7 ans).
    Les bougies de train/val auront un WR optimiste (le modele les a deja vues).
    Choisis une plage out-of-sample pour un chiffre fiable.
    """
    empty_schema = pd.DataFrame(columns=["timestamp", "direction", "confidence", "label"])
    full = _predict_all_bars(start_ms, end_ms)
    taken = full[full["take"] == 1]
    if taken.empty:
        return empty_schema
    out = taken[["timestamp", "confidence", "label"]].copy()
    out["direction"] = "down"
    return out[["timestamp", "direction", "confidence", "label"]].reset_index(drop=True)


def evaluate_full_on_slice(start_ms: int, end_ms: int) -> pd.DataFrame:
    """Comme evaluate_on_slice, mais renvoie *toutes* les bougies de la fenetre
    (pas seulement celles ou un pari aurait ete pris).

    1 row par bougie ayant un label valide et des features completes.
    Colonnes : timestamp (int ms), confidence (= P(down)), threshold (seuil
    effectif de la bougie), take (int 0|1), label (int 0|1, 1 = bougie suivante
    rouge), volatility_state (int), is_weekend (int 0|1), hour (int 0-23 UTC).

    Utilise par 15min/main.py pour produire les logs detailles par bougie.
    Meme mise en garde out-of-sample que evaluate_on_slice.
    """
    return _predict_all_bars(start_ms, end_ms)


if __name__ == "__main__":
    import time
    end = int(time.time() * 1000)
    res = evaluate_on_slice(0, end)
    if res.empty:
        print(f"name={NAME} setups_taken=0")
    else:
        print(f"name={NAME} setups_taken={len(res)} (all down) wr={res['label'].mean():.4f}")
        print(res.head())
    full = evaluate_full_on_slice(0, end)
    taken = int(full["take"].sum()) if not full.empty else 0
    print(f"name={NAME} full_bars={len(full)} take_flagged={taken}")
