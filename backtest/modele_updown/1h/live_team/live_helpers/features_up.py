"""features_up.py — builder runtime des 6 features UP pour le daemon live 1h.

Construit, pour la bougie 1h dont l'open == ``target_open_ms``, le dict des 6
features exactement dans l'ordre attendu par le modèle UP :

    stoch_k_14, rsi_6, hour_cos, ema_compression_20, bars_since_low_50, ret_skew_24

SKEW-CRITIQUE — solidité #1
---------------------------
Les valeurs doivent être *byte-identiques* à celles produites par la chaîne
d'entraînement (sinon train≠serve → dégradation silencieuse du modèle). Pour
l'éliminer par construction, on RÉUTILISE les helpers d'entraînement importés
directement depuis le package ``modele_up`` (via sys.path), au lieu de les
réécrire :

  - ``add_engineered_features.compute_base_30m``  → stoch_k_14, rsi_6
        (la base « 30m » de ce module est en réalité la base 1h, cf. data.py)
  - ``add_alpha_features._ema``                   → réutilisé tel quel
  - ``data._add_tier1_derived``                   → hour_cos

Les 3 features alpha (``ema_compression_20``, ``bars_since_low_50``,
``ret_skew_24``) proviennent de ``add_alpha_features.compute_alpha``. On ne peut
pas appeler ``compute_alpha`` tel quel en live : il lit le parquet ETH
(``eth_data/eth_15m.parquet``, absent du dossier runtime) uniquement pour
``btc_eth_corr_48`` — une colonne dont on n'a PAS besoin ici. On extrait donc
*verbatim* les 3 lignes de calcul concernées depuis ``compute_alpha`` et on
réutilise son helper ``_ema``. Ces formules sont recopiées à l'identique (mêmes
fenêtres, mêmes ``min_periods``, mêmes ``ddof``) — voir ``_alpha_three`` ci-dessous.

Conventions (cf. CONTRAT)
-------------------------
- Tout en millisecondes en interne ; seul le fetch voit les secondes.
- Les buffers sont des listes de barres ``[open_ms, o, h, l, c, v]`` triées ASC.
- ``build_features`` cible la bougie 1h dont l'open == target_open_ms (la
  bougie qui vient de fermer à T = target_open_ms + 1h = decision time).
- On reconstruit les returns normalisés (open_ret/high_ret/low_ret/close_ret)
  depuis l'OHLC brut EXACTEMENT comme data.py phase 2
  (``(X - close_prev) / close_prev``), puisque les helpers d'entraînement les
  attendent en entrée.

API publique
------------
    build_features(buf_1h, buf_5m, buf_30m, buf_4h, buf_1d,
                   target_open_ms) -> dict
    validate_against_parquet(parquet_path, n_test_bars=12, tol=1e-6,
                             sqlite_path=None, verbose=True) -> None
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Import des helpers d'entraînement (pattern sys.path, comme l'ancien builder) #
# --------------------------------------------------------------------------- #
# On ajoute le package runtime modele_up sur sys.path pour importer ses modules
# comme des modules top-level (mêmes noms qu'à l'entraînement). C'est le SEUL
# moyen de garantir l'identité byte-à-byte des formules.
_MODELE_UP_DIR = (Path(__file__).resolve().parent.parent / "modele_up")
if str(_MODELE_UP_DIR) not in sys.path:
    sys.path.insert(0, str(_MODELE_UP_DIR))

import add_engineered_features as _eng   # compute_base_30m (stoch_k_14, rsi_6)
import add_alpha_features as _alpha       # _ema (réutilisé pour ema_compression_20)
import data as _data                      # _add_tier1_derived (hour_cos)


# Ordre EXACT attendu par le modèle (= booster.feature_names). Source de vérité :
# btc_updown_model.json ; predict.py ré-ordonne aussi via feature_names, mais on
# émet déjà dans le bon ordre par sécurité.
FEATURES = [
    "stoch_k_14",
    "rsi_6",
    "hour_cos",
    "ema_compression_20",
    "bars_since_low_50",
    "ret_skew_24",
]

INTERVAL_1H_MS = 60 * 60 * 1000


# --------------------------------------------------------------------------- #
# Reconstruction du DataFrame OHLC 1h (avec returns normalisés)                #
# --------------------------------------------------------------------------- #

def _buf_to_df_1h(buf_1h: Sequence) -> pd.DataFrame:
    """[open_ms, o, h, l, c, v] (ASC) -> DataFrame 1h avec les colonnes attendues
    par les helpers d'entraînement.

    Reconstruit open_ret/high_ret/low_ret/close_ret depuis l'OHLC brut, à
    l'identique de data.py phase 2 (``compute_normalized_features``) :
        X_ret = (X - close_prev) / close_prev.
    """
    if not buf_1h or len(buf_1h) < 2:
        raise ValueError("buf_1h trop court (au moins 2 barres requises).")

    df = pd.DataFrame(buf_1h, columns=["timestamp_ms", "open", "high", "low",
                                       "close", "volume"])
    df["timestamp_ms"] = df["timestamp_ms"].astype(np.int64)
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = df[col].astype(float)
    df = df.sort_values("timestamp_ms").reset_index(drop=True)

    close_prev = df["close"].shift(1)
    df["open_ret"] = (df["open"] - close_prev) / close_prev
    df["high_ret"] = (df["high"] - close_prev) / close_prev
    df["low_ret"] = (df["low"] - close_prev) / close_prev
    df["close_ret"] = (df["close"] - close_prev) / close_prev
    return df


# --------------------------------------------------------------------------- #
# Les 3 features alpha — recopie VERBATIM des lignes de compute_alpha          #
# --------------------------------------------------------------------------- #

def _alpha_three(df: pd.DataFrame) -> dict:
    """Calcule ema_compression_20, bars_since_low_50, ret_skew_24.

    Lignes recopiées à l'identique depuis
    ``add_alpha_features.compute_alpha`` (mêmes fenêtres, mêmes min_periods),
    en réutilisant son helper ``_alpha._ema``. On n'appelle pas compute_alpha
    directement car il lit le parquet ETH (pour btc_eth_corr_48, non utilisé ici)
    qui est absent du dossier runtime.
    """
    c = df["close"].astype(float)
    l = df["low"]
    cr = df["close_ret"].astype(float)
    out = {}

    # --- Forme de distribution : ret_skew_24 ---
    out["ret_skew_24"] = cr.rolling(24, min_periods=24).skew().to_numpy()

    # --- Temps depuis extreme (fraction de la fenetre) : bars_since_low_50 ---
    win = 50
    bsl = l.rolling(win, min_periods=win).apply(
        lambda x: (win - 1 - np.argmin(x)), raw=True)
    out["bars_since_low_50"] = (bsl / win).to_numpy()

    # --- Structure de tendance : ema_compression_20 ---
    e10, e20, e50 = _alpha._ema(c, 10), _alpha._ema(c, 20), _alpha._ema(c, 50)
    emax = pd.concat([e10, e20, e50], axis=1).max(axis=1)
    emin = pd.concat([e10, e20, e50], axis=1).min(axis=1)
    out["ema_compression_20"] = ((emax - emin) / c).to_numpy()

    return out


# --------------------------------------------------------------------------- #
# API publique                                                                 #
# --------------------------------------------------------------------------- #

def build_features(buf_1h: Sequence, buf_5m: Sequence | None = None,
                   buf_30m: Sequence | None = None, buf_4h: Sequence | None = None,
                   buf_1d: Sequence | None = None,
                   target_open_ms: int | None = None) -> dict:
    """Construit les 6 features UP pour la bougie 1h dont l'open == target_open_ms.

    Les 6 features sont toutes dérivées de la SEULE série 1h (base). Les buffers
    MTF (5m/30m/4h/1d) sont acceptés pour respecter la signature du contrat mais
    ne sont pas utilisés par le modèle UP (aucune feature MTF dans ses 6 colonnes).

    Paramètres
    ----------
    buf_1h         : liste de barres [open_ms, o, h, l, c, v] ASC, gap-free.
                     Profondeur ≥ ~200 barres recommandée (warmup des rolling :
                     EMA50, rolling 50/48/24 ; warmup réel ≈ 100 barres).
    target_open_ms : open_ms (int) de la bougie 1h cible (celle qui vient de
                     fermer). Doit être présent dans buf_1h.

    Retour : dict {feature: float} dans l'ordre du modèle (FEATURES).

    Lève ValueError si target_open_ms est absent du buffer.
    """
    if target_open_ms is None:
        raise ValueError("target_open_ms est requis (open_ms de la bougie cible).")

    df = _buf_to_df_1h(buf_1h)

    # Localise la bougie cible (open exact). Robuste : on n'utilise PAS un index
    # positionnel (la bougie en cours peut ou non être présente dans le buffer).
    pos = np.where(df["timestamp_ms"].to_numpy() == int(target_open_ms))[0]
    if len(pos) == 0:
        raise ValueError(
            f"target_open_ms {target_open_ms} introuvable dans buf_1h "
            f"(range {int(df['timestamp_ms'].iloc[0])}..{int(df['timestamp_ms'].iloc[-1])})."
        )
    last_idx = int(pos[-1])

    # --- stoch_k_14 + rsi_6 via le helper d'entraînement (verbatim) ---
    base = _eng.compute_base_30m(df)   # dict {col: np.ndarray} aligné sur df

    # --- hour_cos via data._add_tier1_derived (verbatim) ---
    # _add_tier1_derived mute df en place et ajoute hour_cos (+ d'autres colonnes
    # qu'on ignore). Il a besoin de high_ret/low_ret/open_ret/close_ret + close,
    # tous déjà présents.
    df_hc = df.copy()
    _data._add_tier1_derived(df_hc)
    hour_cos = df_hc["hour_cos"].to_numpy()

    # --- les 3 features alpha (verbatim depuis compute_alpha) ---
    alpha = _alpha_three(df)

    out = {
        "stoch_k_14":         float(base["stoch_k_14"][last_idx]),
        "rsi_6":              float(base["rsi_6"][last_idx]),
        "hour_cos":           float(hour_cos[last_idx]),
        "ema_compression_20": float(alpha["ema_compression_20"][last_idx]),
        "bars_since_low_50":  float(alpha["bars_since_low_50"][last_idx]),
        "ret_skew_24":        float(alpha["ret_skew_24"][last_idx]),
    }
    # Émis explicitement dans l'ordre du modèle (FEATURES).
    return {k: out[k] for k in FEATURES}


def validate_against_parquet(parquet_path, n_test_bars: int = 12,
                             tol: float = 1e-6, sqlite_path=None,
                             verbose: bool = True) -> None:
    """Re-calcule les 6 features pour n_test_bars barres historiques via le
    pipeline LIVE et les compare row-by-row au parquet d'entraînement.

    Reconstruit l'OHLC 1h depuis la SQLite sœur (le parquet ne stocke que des
    features scale-invariantes, pas l'OHLC brut), exactement comme l'ancien
    builder 15m. Mode strict : lève AssertionError si un écart dépasse `tol`.

    Paramètres
    ----------
    parquet_path : chemin du master parquet (modele_up/DB/btc_multi_tf_1h.parquet).
    sqlite_path  : SQLite sœur (défaut : <parquet_dir>/btc_multi_tf.db).
    """
    import sqlite3

    pq_path = Path(parquet_path)
    if sqlite_path is None:
        sqlite_path = pq_path.parent / "btc_multi_tf.db"

    pq = pd.read_parquet(pq_path).sort_values("timestamp_ms").reset_index(drop=True)

    con = sqlite3.connect(str(sqlite_path))
    try:
        sq_1h = pd.read_sql(
            "SELECT timestamp_ms, close, volume, open_ret, high_ret, low_ret "
            "FROM candles WHERE timeframe='1h' ORDER BY timestamp_ms",
            con,
        )
    finally:
        con.close()

    # Reconstruit l'OHLC brut 1h : X = close_prev * (1 + X_ret).
    close_prev = sq_1h["close"].shift(1)
    sq_1h["open"] = close_prev * (1 + sq_1h["open_ret"])
    sq_1h["high"] = close_prev * (1 + sq_1h["high_ret"])
    sq_1h["low"] = close_prev * (1 + sq_1h["low_ret"])
    sq_1h = sq_1h[["timestamp_ms", "open", "high", "low", "close", "volume"]].dropna()
    buf_full = sq_1h.values.tolist()

    # Barres de test loin du warmup (dans les dernières lignes du parquet).
    hi = len(pq) - 100
    lo = max(200, len(pq) - 10000)
    if hi <= lo:
        lo = 200
        hi = max(lo + 1, len(pq) - 1)
    test_indices = np.linspace(lo, hi, n_test_bars).astype(int)

    failures: list[tuple[int, str]] = []
    if verbose:
        print(f"Validation features_up : {len(test_indices)} barres (tol={tol})…")

    for i in test_indices:
        target_ts = int(pq.iloc[i]["timestamp_ms"])
        # Tronque le buffer jusqu'à la bougie cible incluse (no-lookahead).
        cut = [b for b in buf_full if int(b[0]) <= target_ts]
        if not cut or int(cut[-1][0]) != target_ts:
            failures.append((target_ts, "bougie cible absente de la SQLite"))
            continue
        try:
            feats = build_features(cut, target_open_ms=target_ts)
        except Exception as exc:  # noqa: BLE001 — on collecte, on ne stoppe pas
            failures.append((target_ts, f"build_features a levé : {exc}"))
            continue

        row_pq = pq.iloc[i]
        for col in FEATURES:
            if col not in row_pq.index:
                continue
            v_pq = row_pq[col]
            if pd.isna(v_pq):
                continue
            v_live = feats[col]
            diff = abs(v_live - v_pq)
            rel = diff / max(abs(v_pq), 1e-9)
            if diff > tol and rel > tol:
                failures.append((
                    target_ts,
                    f"{col}: live={v_live:.6e} parquet={v_pq:.6e} diff={diff:.2e}",
                ))

    if failures:
        msg = (f"features_up.validate_against_parquet : "
               f"{len(failures)} écarts\n")
        for ts, err in failures[:20]:
            msg += f"  ts={ts}: {err}\n"
        raise AssertionError(msg)

    if verbose:
        print(f"  OK : {len(test_indices)} barres concordent avec le parquet "
              f"(tol={tol}).")


if __name__ == "__main__":
    # Validateur de dev : nécessite le parquet + la SQLite côté entraînement.
    _PARQUET = (Path(__file__).resolve().parents[2]
                / "modele_up" / "DB" / "btc_multi_tf_1h.parquet")
    validate_against_parquet(_PARQUET, n_test_bars=12, tol=1e-6)
