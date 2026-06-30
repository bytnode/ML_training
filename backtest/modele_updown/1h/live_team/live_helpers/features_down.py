"""features_down.py — constructeur de features live pour le modele DOWN (1h).

SKEW-CRITIQUE : ce fichier doit reproduire BYTE-A-BYTE les 6 features du
parquet d'entrainement (`modele_down/DB/btc_multi_tf_1h.parquet`) pour la
bougie 1h cible. Toute derive ici = serve skew = predictions fausses.

Les 6 features DOWN, dans l'ordre du modele
(`booster.feature_names` = stoch_k_14, hour_cos, ef_ret4, rsi_6, autocorr_12,
ef_upwick) :
  - stoch_k_14, rsi_6, autocorr_12 : base BTC 1h, via
    `add_engineered_features.compute_base_30m` (importe verbatim).
  - hour_cos : derive temporelle, via `data._add_tier1_derived` (importe).
  - ef_ret4, ef_upwick : cross-asset ETH, via
    `add_eth_features.compute_eth_features` (importe). L'ETH 15m brut est
    RESAMPLE en bougies 1h (bucket = ts//3600000*3600000 ; open=first,
    high=max, low=min, close=last, volume=sum) puis joint a la BTC 1h par
    TIMESTAMP EXACT (bougie ETH 1h concurrente, fermee a T = open_1h + 1h).

Aucun calcul de feature n'est re-implemente ici : on reutilise les helpers
d'entrainement par import (sys.path sur `live_team/modele_down`), exactement
comme l'ancien daemon 15min reutilisait `data.py`. Le seul code local est
l'assemblage des DataFrames OHLC depuis les buffers live et la selection de la
ligne cible.

Convention buffers : liste de bars `[open_ms, o, h, l, c, v]` triee ASC (open_ms
en MILLISECONDES). `build_features` recoit les buffers nommes via `predict.decide`.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Injection sys.path : on importe les helpers d'entrainement du package
# `modele_down` qui est un frere de `live_helpers/` dans ce dossier auto-contenu.
# __file__ = live_team/live_helpers/features_down.py
#   parents[0] = live_helpers/   parents[1] = live_team/
# ---------------------------------------------------------------------------
_MODELE_DOWN_DIR = str(Path(__file__).resolve().parents[1] / "modele_down")
if _MODELE_DOWN_DIR not in sys.path:
    sys.path.insert(0, _MODELE_DOWN_DIR)

# Imports apres injection sys.path (E402 attendu et assume).
from add_engineered_features import compute_base_30m            # noqa: E402
from add_eth_features import (                                  # noqa: E402
    resample_eth_to_30m,   # malgre son nom : resample en 1h (INTERVAL_1H_MS)
    compute_eth_features,
)
from data import _add_tier1_derived                             # noqa: E402


# Les 6 features DOWN attendues (ordre = contrat ; le predicteur reordonne de
# toute facon via booster.feature_names). On les expose pour la validation.
FEATURES_DOWN = ["stoch_k_14", "rsi_6", "autocorr_12", "hour_cos", "ef_ret4", "ef_upwick"]

INTERVAL_1H_MS = 60 * 60 * 1000


# ===========================================================================
# Helpers locaux : buffers -> DataFrame OHLC enrichi
# ===========================================================================
def _buf_to_ohlc_df(buf: Sequence) -> pd.DataFrame:
    """Buffer `[open_ms, o, h, l, c, v]` ASC -> DataFrame avec les colonnes
    OHLCV brutes + les rendements normalises vs close precedent.

    Le DataFrame produit satisfait les deux helpers d'entrainement :
      - `compute_base_30m` lit open/high/low/close/volume/close_ret ;
      - `_add_tier1_derived` lit timestamp_ms/close/open_ret/high_ret/low_ret/
        close_ret/volume.
    Les rendements sont reconstruits exactement comme `data.compute_normalized_
    features` (decalage vs close[i-1]), garantissant la parite avec le parquet.
    """
    if not buf:
        raise ValueError("features_down: buffer 1h vide.")
    df = pd.DataFrame(buf, columns=["timestamp_ms", "open", "high", "low", "close", "volume"])
    df["timestamp_ms"] = df["timestamp_ms"].astype(np.int64)
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = df[c].astype(float)
    df = df.sort_values("timestamp_ms").reset_index(drop=True)

    close_prev = df["close"].shift(1)
    df["open_ret"] = (df["open"] - close_prev) / close_prev
    df["high_ret"] = (df["high"] - close_prev) / close_prev
    df["low_ret"] = (df["low"] - close_prev) / close_prev
    df["close_ret"] = (df["close"] - close_prev) / close_prev
    return df


def _buf_eth_to_df(buf: Sequence) -> pd.DataFrame:
    """Buffer ETH 15m `[open_ms, o, h, l, c, v]` ASC -> DataFrame OHLCV brut
    (colonnes attendues par `resample_eth_to_30m`)."""
    df = pd.DataFrame(buf, columns=["timestamp_ms", "open", "high", "low", "close", "volume"])
    df["timestamp_ms"] = df["timestamp_ms"].astype(np.int64)
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = df[c].astype(float)
    return df.sort_values("timestamp_ms").reset_index(drop=True)


def _eth_feats_for_ts(buf_eth_15m, target_open_ms: int) -> dict:
    """ef_ret4 / ef_upwick pour la bougie ETH 1h dont l'open == target_open_ms.

    L'ETH 15m est resample en bougies 1h (bucket = ts//3600000*3600000), puis
    `compute_eth_features` calcule les ef_* ; on selectionne la ligne au
    TIMESTAMP EXACT. Donnees ETH absentes / bougie concurrente manquante -> NaN
    (XGBoost route les NaN nativement, comme le parquet pour les bars BTC
    anterieurs a l'historique ETH).
    """
    nan_row = {"ef_ret4": float("nan"), "ef_upwick": float("nan")}
    if not buf_eth_15m:
        return nan_row
    # Le bucket 1h CIBLE doit avoir ses 4 sous-bougies 15m (target, +15, +30, +45),
    # sinon close/high/low du bucket 1h resample sont FAUX (finis mais faux =
    # skew silencieux). Bucket incomplet -> NaN, que validate_features rejette
    # (-> retry puis annulation du cycle), au lieu de predire sur une valeur fausse.
    need = {int(target_open_ms) + k * (15 * 60 * 1000) for k in range(4)}
    have = {int(b[0]) for b in buf_eth_15m}
    if not need.issubset(have):
        return nan_row
    eth_1h = resample_eth_to_30m(_buf_eth_to_df(buf_eth_15m))   # -> bougies 1h
    if eth_1h.empty:
        return nan_row
    eth_feat = compute_eth_features(eth_1h)
    hit = eth_feat.loc[eth_feat["timestamp_ms"] == int(target_open_ms)]
    if len(hit) != 1:
        return nan_row
    r = hit.iloc[0]
    return {"ef_ret4": float(r["ef_ret4"]), "ef_upwick": float(r["ef_upwick"])}


# ===========================================================================
# API publique
# ===========================================================================
def build_features(buf_1h, buf_5m, buf_30m, buf_4h, buf_1d, buf_eth_15m,
                   target_open_ms: int) -> dict:
    """Construit les 6 features DOWN pour la bougie 1h dont l'open == target_open_ms.

    Parametres
    ----------
    buf_1h : buffer BTC 1h (base), liste de [open_ms,o,h,l,c,v] ASC. SEUL buffer
             reellement utilise par les 6 features DOWN (base 1h + temporelle).
    buf_5m, buf_30m, buf_4h, buf_1d : presents pour homogeneite de signature avec
             features_up ; non utilises ici (aucune des 6 features DOWN n'en depend).
    buf_eth_15m : buffer ETH/USD 15m (resample en 1h en interne) pour ef_*.
    target_open_ms : open_time (ms) de la bougie 1h cible (la bougie qui vient de
             fermer a T = target_open_ms + 1h).

    Retour : dict {stoch_k_14, rsi_6, autocorr_12, hour_cos, ef_ret4, ef_upwick}.
    """
    # buffers fins non utilises par DOWN ; referencement explicite pour la clarte.
    _ = (buf_5m, buf_30m, buf_4h, buf_1d)

    df_1h = _buf_to_ohlc_df(buf_1h)

    # Base BTC 1h : stoch_k_14, rsi_6, autocorr_12 (+ autres ignorees).
    base = compute_base_30m(df_1h)

    # Derive temporelle : _add_tier1_derived ecrit hour_cos in-place dans df_1h.
    _add_tier1_derived(df_1h)

    # Selection de la ligne cible par timestamp exact (jamais positionnel : on ne
    # depend pas de la presence de la bougie en cours dans le buffer).
    pos = np.where(df_1h["timestamp_ms"].to_numpy(dtype=np.int64) == int(target_open_ms))[0]
    if len(pos) == 0:
        raise ValueError(
            f"features_down: target_open_ms {target_open_ms} absent du buffer 1h."
        )
    i = int(pos[-1])

    out = {
        "stoch_k_14": float(base["stoch_k_14"][i]),
        "rsi_6": float(base["rsi_6"][i]),
        "autocorr_12": float(base["autocorr_12"][i]),
        "hour_cos": float(df_1h["hour_cos"].to_numpy()[i]),
    }
    # ETH : join par timestamp exact sur la bougie ETH 1h concurrente.
    out.update(_eth_feats_for_ts(buf_eth_15m, int(target_open_ms)))

    # Garantit l'ensemble exact des cles (detecte une derive de helper amont).
    missing = [c for c in FEATURES_DOWN if c not in out]
    if missing:
        raise KeyError(f"features_down: features manquantes apres pipeline : {missing}")
    return out


def validate_against_parquet(parquet_path, n_test_bars: int = 5, tol: float = 1e-6) -> None:
    """Verifie row-by-row que `build_features` reproduit le parquet d'entrainement.

    Reconstruit l'OHLC 1h BTC depuis la SQLite `candles` (meme source que
    `add_engineered_features`), rejoue le pipeline sur des lignes connues du
    parquet et compare les 6 features avec une tolerance `tol`.

    ef_ret4 / ef_upwick (ETH) sont compares uniquement si l'ETH 15m est
    disponible en parquet (`eth_data/eth_15m.parquet`) ; sinon on saute ces deux
    colonnes (le parquet peut porter des NaN sur les bars pre-historique ETH).
    """
    import sqlite3

    pq_path = Path(parquet_path)
    pq = pd.read_parquet(pq_path).sort_values("timestamp_ms").reset_index(drop=True)

    db_path = pq_path.parent / "btc_multi_tf.db"
    if not db_path.exists():
        print(f"features_down.validate: SQLite absente ({db_path}), validation sautee.")
        return

    con = sqlite3.connect(db_path)
    try:
        df = pd.read_sql(
            "SELECT timestamp_ms, close, volume, open_ret, high_ret, low_ret "
            "FROM candles WHERE timeframe='1h' ORDER BY timestamp_ms",
            con,
        )
    finally:
        con.close()
    close_prev = df["close"].shift(1)
    df["open"] = close_prev * (1.0 + df["open_ret"])
    df["high"] = close_prev * (1.0 + df["high_ret"])
    df["low"] = close_prev * (1.0 + df["low_ret"])
    sq_1h = (df[["timestamp_ms", "open", "high", "low", "close", "volume"]]
             .dropna().values.tolist())
    sq_1h = [[int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])]
             for r in sq_1h]

    # ETH 15m optionnel (meme parquet que le patch d'entrainement).
    eth_parquet = pq_path.parent.parent / "eth_data" / "eth_15m.parquet"
    eth_buf = None
    if eth_parquet.exists():
        edf = (pd.read_parquet(eth_parquet)
               .sort_values("timestamp_ms").reset_index(drop=True))
        eth_buf = edf[["timestamp_ms", "open", "high", "low", "close", "volume"]].values.tolist()
        eth_buf = [[int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])]
                   for r in eth_buf]

    # Choix de lignes de test eloignees du warmup et de la queue.
    lo = max(2880, len(pq) - 10000)
    hi = max(lo + 1, len(pq) - 100)
    test_idx = np.unique(np.linspace(lo, hi, n_test_bars).astype(int))

    btc_ts = np.array([k[0] for k in sq_1h], dtype=np.int64)
    base_cols = ["stoch_k_14", "rsi_6", "autocorr_12", "hour_cos"]
    eth_cols = ["ef_ret4", "ef_upwick"]

    failures = []
    print(f"features_down validation: {len(test_idx)} bars, tol={tol}...")
    for idx in test_idx:
        ts = int(pq.iloc[idx]["timestamp_ms"])
        # Buffer 1h = toutes les bougies dont l'open <= ts (la cible incluse).
        cut = int(np.searchsorted(btc_ts, ts, side="right"))
        if cut == 0 or btc_ts[cut - 1] != ts:
            failures.append((ts, "cible absente de la SQLite 1h"))
            continue
        buf_1h = sq_1h[:cut]
        try:
            feats = build_features(buf_1h, None, None, None, None, eth_buf, ts)
        except Exception as e:  # noqa: BLE001
            failures.append((ts, f"build_features a leve : {e}"))
            continue

        row = pq.iloc[idx]
        cols = base_cols + (eth_cols if eth_buf is not None else [])
        for c in cols:
            if c not in row.index or pd.isna(row[c]):
                continue
            v_live, v_pq = float(feats[c]), float(row[c])
            if pd.isna(v_live):
                continue
            diff = abs(v_live - v_pq)
            rel = diff / max(abs(v_pq), 1e-9)
            if diff > tol and rel > tol:
                failures.append((ts, f"{c}: live={v_live:.6e} pq={v_pq:.6e} diff={diff:.2e}"))

    if failures:
        msg = f"features_down.validate: {len(failures)} ecarts\n"
        for ts, err in failures[:20]:
            msg += f"  ts={ts}: {err}\n"
        raise AssertionError(msg)
    print(f"  OK: {len(test_idx)} bars conformes (tol={tol}).")


if __name__ == "__main__":
    # Validateur dev : pointe sur le parquet du package d'entrainement frere.
    _PARQUET = Path(__file__).resolve().parents[1] / "modele_down" / "DB" / "btc_multi_tf_1h.parquet"
    if _PARQUET.exists():
        validate_against_parquet(_PARQUET, n_test_bars=5, tol=1e-6)
    else:
        print(f"Parquet introuvable : {_PARQUET} (validation dev sautee).")
