"""filters.py — logique de decision partagee entre le backtest (main.py) et le
daemon live (live/).

COPIE INDEPENDANTE (2026-08-03, campagne autonomie complete live/) : ce
fichier est desormais DUPLIQUE, pas partage -- modele_team/filters.py (utilise
par main.py) et live/filters.py (cette copie, utilisee par live.py/position.py/
market_data.py) sont deux fichiers distincts sur disque, embarques pour que
live/ soit deployable SEUL sur un VPS (aucune dependance a modele_team/ ni a
1h/). AVANT ce changement, un seul fichier partage garantissait la symetrie
backtest/live automatiquement ; MAINTENANT, un changement de regle/seuil ici
NE SE PROPAGE PLUS tout seul vers modele_team/filters.py (ou l'inverse) --
recopie manuellement apres toute modification, meme tradeoff deja accepte
pour live/modele_up|down/regime/ (cf live/CLAUDE.md).

Extrait de main.py (campagne symetrie OKX 2026-07-09) : regle de coherence,
filtre DMI, les 3 filtres d'entree UP, le wrapper regime.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

# AUCUNE insertion sys.path ICI (retire 2026-07-10, campagne autonomie live/) :
# filters.py est importe par main.py (backtest, pousse 1h/ sur sys.path AVANT
# d'importer filters -- modele_down/regime resolvent vers les packages 1h/) ET
# par live.py/market_data.py/position.py (live, poussent live/ AVANT d'importer
# filters -- modele_down/regime resolvent vers les copies EMBARQUEES live/modele_down/,
# live/regime/). Si filters.py forcait 1h/ lui-meme, le cache sys.modules figerait
# `modele_down`/`regime` sur 1h/ pour TOUT le process live -- meme quand live.py
# essaie ensuite d'importer SES propres copies (Python ne re-resout jamais un nom
# de package deja present dans sys.modules, quel que soit sys.path apres coup).
# Le CALLER doit donc TOUJOURS avoir configure sys.path avant `import filters`.
from modele_down import api as api_down
from regime import api as api_regime
from regime import config_regime as cfg_regime

from config_main import (
    DMI_FILTER_ENABLED, DMI_LENGTH, DMI_TIMEFRAME, DMI_SIDES,
    UP_CONF_FILTER_ENABLED, UP_CONF_MIN,
    UP_XMODEL_FILTER_ENABLED, UP_XMODEL_MAX_PDOWN,
    DOWN_CONF_DELTA,
)


# ----------------------------------------------------------------------------
#  Regle de coherence d'equipe
# ----------------------------------------------------------------------------
def build_team_bets(up_bets: pd.DataFrame, down_bets: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Applique la regle de coherence. Renvoie (team_bets, nb_annules).

        up parie UP   + down s'abstient  -> UP
        down parie DN + up s'abstient    -> DOWN
        les deux parient                  -> CANCEL
        aucun                             -> PASS
    """
    up_ts = set(up_bets["timestamp"]) if not up_bets.empty else set()
    down_ts = set(down_bets["timestamp"]) if not down_bets.empty else set()

    cancelled = up_ts & down_ts
    coherent_up_ts = up_ts - down_ts
    coherent_down_ts = down_ts - up_ts

    kept_up = up_bets[up_bets["timestamp"].isin(coherent_up_ts)]
    kept_down = down_bets[down_bets["timestamp"].isin(coherent_down_ts)]

    team = pd.concat([kept_up, kept_down], ignore_index=True)
    if not team.empty:
        team = team.sort_values("timestamp").reset_index(drop=True)
    return team, len(cancelled)


# ----------------------------------------------------------------------------
#  Filtre DMI (Directional Movement Index) Wilder sur le TF DMI_TIMEFRAME
# ----------------------------------------------------------------------------
def _wilder_dmi(high, low, close, n: int):
    """+DI / -DI de Wilder (longueur n), alignes bougie a bougie sur les entrees.

    +DI[i] / -DI[i] n'utilisent que les bougies <= i (high/low/close de i et i-1,
    puis lissage recursif) -> connus a la cloture de la bougie i. Renvoie 2 arrays
    (NaN sur les premieres bougies / le warmup). On ne compare que le SIGNE de
    (+DI - -DI), donc le seed exact du lissage est sans incidence apres rodage.
    """
    h = np.asarray(high, dtype=float)
    l = np.asarray(low, dtype=float)
    cl = np.asarray(close, dtype=float)
    m = len(h)
    plus = np.full(m, np.nan)
    minus = np.full(m, np.nan)
    if m < 2:
        return plus, minus
    up_move = h[1:] - h[:-1]
    down_move = l[:-1] - l[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0.0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0.0), down_move, 0.0)
    prev_close = cl[:-1]
    tr = np.maximum.reduce([h[1:] - l[1:],
                            np.abs(h[1:] - prev_close),
                            np.abs(l[1:] - prev_close)])

    def _rma(x):  # lissage de Wilder = EMA recursive alpha=1/n (adjust=False)
        return pd.Series(x).ewm(alpha=1.0 / n, adjust=False).mean().to_numpy()

    atr = _rma(tr)
    s_plus = _rma(plus_dm)
    s_minus = _rma(minus_dm)
    with np.errstate(divide="ignore", invalid="ignore"):
        plus[1:] = 100.0 * s_plus / atr
        minus[1:] = 100.0 * s_minus / atr
    return plus, minus


def _candles_db_path() -> Path:
    down_dir = Path(api_down.__file__).resolve().parent
    db_path = next((p for p in (down_dir / "db" / "btc_multi_tf.db",
                                down_dir / "DB" / "btc_multi_tf.db") if p.exists()), None)
    if db_path is None:
        raise FileNotFoundError(
            f"SQLite candles introuvable sous {down_dir} (db/ ou DB/). Lance data.py."
        )
    return db_path


def load_dmi_map() -> dict[int, tuple[float, float]]:
    """Map open-time (ms) -> (+DI, -DI) du DMI Wilder(DMI_LENGTH) sur les bougies
    DMI_TIMEFRAME. OHLC reconstruit depuis la table `candles` du package
    modele_down : close est absolu, et *_ret = (val - close_prev)/close_prev, donc
        close_prev = close / (1 + close_ret)
        high = close_prev*(1+high_ret) ; low = close_prev*(1+low_ret).
    Couvre tout l'historique (la fenetre de test/live est un sous-ensemble).

    NON UTILISE PAR live/ (le live calcule le DMI par-trade directement depuis
    son buffer Bitstamp en RAM via market_data.py::dmi_blocks -> dmi_blocks_direction,
    jamais via cette fonction ni via _candles_db_path -- live/ ne touche aucune
    SQLite). Reste ici uniquement pour le backtest (apply_dmi_filter, batch)."""
    con = sqlite3.connect(str(_candles_db_path()))
    try:
        c = pd.read_sql_query(
            "SELECT timestamp_ms, close, close_ret, high_ret, low_ret "
            "FROM candles WHERE timeframe=? ORDER BY timestamp_ms",
            con, params=(DMI_TIMEFRAME,))
    finally:
        con.close()
    c = c.dropna(subset=["close", "close_ret", "high_ret", "low_ret"]).reset_index(drop=True)
    close = c["close"].to_numpy(dtype=float)
    close_prev = close / (1.0 + c["close_ret"].to_numpy(dtype=float))
    high = close_prev * (1.0 + c["high_ret"].to_numpy(dtype=float))
    low = close_prev * (1.0 + c["low_ret"].to_numpy(dtype=float))
    plus_di, minus_di = _wilder_dmi(high, low, close, DMI_LENGTH)
    ts = c["timestamp_ms"].astype("int64").to_numpy()
    return {int(t): (float(p), float(mi)) for t, p, mi in zip(ts, plus_di, minus_di)}


def dmi_blocks_direction(direction: str, dmi_map: dict[int, tuple[float, float]],
                         decision_ts: int) -> bool:
    """True si le DMI de la bougie de DECISION `decision_ts` bloque un pari
    `direction` ('up'/'down'), False sinon (garde le pari). Bougie inconnue ou
    warmup (NaN) ou egalite +DI==-DI -> False (on garde). Utilise par live/live.py
    pour une decision bar-par-bar (equivalent single-row de apply_dmi_filter)."""
    if not DMI_FILTER_ENABLED:
        return False
    di = dmi_map.get(int(decision_ts))
    if di is None:
        return False
    pdi, mdi = di
    if np.isnan(pdi) or np.isnan(mdi):
        return False
    if direction == "down" and "down" in DMI_SIDES and pdi > mdi:
        return True
    if direction == "up" and "up" in DMI_SIDES and mdi > pdi:
        return True
    return False


def apply_dmi_filter(team_bets: pd.DataFrame,
                     dmi_map: dict[int, tuple[float, float]] | None = None) -> tuple[pd.DataFrame, int, int]:
    """Annule les paris dont le sens contredit le DMI de la bougie de DECISION
    (batch, backtest). dmi_map precalcule (optionnel) evite de recharger/recalculer
    le DMI a chaque appel -- utilise par compare_regime_variants.py (memes
    predictions, plusieurs variantes de regime testees). None (defaut) = calcule ici.
    Renvoie (team_bets_filtre, n_removed_up, n_removed_down)."""
    if not DMI_FILTER_ENABLED or team_bets.empty:
        return team_bets, 0, 0
    if dmi_map is None:
        dmi_map = load_dmi_map()
    keep = np.ones(len(team_bets), dtype=bool)
    removed_up = removed_down = 0
    for idx, b in enumerate(team_bets.itertuples(index=False)):
        if dmi_blocks_direction(b.direction, dmi_map, int(b.timestamp)):
            keep[idx] = False
            if b.direction == "down":
                removed_down += 1
            else:
                removed_up += 1
    return team_bets[keep].reset_index(drop=True), removed_up, removed_down


# ----------------------------------------------------------------------------
#  Wrapper de regime (optionnel) : timestamp -> regime confirme, et seuils
#  effectifs par pari qui en decoulent. regime_map=None PARTOUT = comportement
#  strictement inchange (c'est le defaut, cfg_regime.REGIME_WRAPPER_ENABLED=False).
# ----------------------------------------------------------------------------
def load_regime_map() -> dict[int, str]:
    """timestamp (ms, bougie de decision) -> regime confirme ('bear'/'neutral'/
    'bull'), calcule sur TOUT l'historique 1h disponible (independant de la
    fenetre de test/live -- la validation du regime est volontairement plus
    large pour couvrir plusieurs vrais cycles bear/neutral/bull)."""
    df = api_regime.evaluate_regime_full()
    return dict(zip(df["timestamp"].astype("int64"), df["regime"]))


def current_regime(regime_map: dict[int, str] | None, ts: int) -> str:
    """Regime confirme au timestamp `ts` (bar par bar, pour le live). 'neutral'
    si regime_map est None ou si ts est absent (warmup/hors-historique)."""
    if regime_map is None:
        return "neutral"
    return regime_map.get(int(ts), "neutral")


def effective_down_delta(timestamps, regime_map: dict[int, str] | None) -> np.ndarray:
    """DOWN_CONF_DELTA effectif par bar : le scalaire de config_main.py, sauf
    override du regime actif a ce timestamp (regime_map=None -> DOWN_CONF_DELTA
    partout, comportement actuel inchange)."""
    if regime_map is None:
        return np.full(len(timestamps), DOWN_CONF_DELTA)
    out = np.empty(len(timestamps))
    for i, ts in enumerate(timestamps):
        regime = regime_map.get(int(ts), "neutral")
        override = cfg_regime.REGIME_OVERRIDES.get(regime, {}).get("down_conf_delta")
        out[i] = DOWN_CONF_DELTA if override is None else override
    return out


def effective_up_conf_min(timestamps, regime_map: dict[int, str] | None) -> np.ndarray:
    """UP_CONF_MIN effectif par pari : meme principe que effective_down_delta."""
    if regime_map is None:
        return np.full(len(timestamps), UP_CONF_MIN)
    out = np.empty(len(timestamps))
    for i, ts in enumerate(timestamps):
        regime = regime_map.get(int(ts), "neutral")
        override = cfg_regime.REGIME_OVERRIDES.get(regime, {}).get("up_conf_min")
        out[i] = UP_CONF_MIN if override is None else override
    return out


def effective_up_conf_min_1(ts: int, regime_map: dict[int, str] | None) -> float:
    """Version scalaire (bar unique) de effective_up_conf_min, pour le live."""
    return float(effective_up_conf_min(np.array([ts]), regime_map)[0])


def effective_down_delta_1(ts: int, regime_map: dict[int, str] | None) -> float:
    """Version scalaire (bar unique) de effective_down_delta, pour le live."""
    return float(effective_down_delta(np.array([ts]), regime_map)[0])


def apply_conf_filter(team_bets: pd.DataFrame,
                      regime_map: dict[int, str] | None = None) -> tuple[pd.DataFrame, int]:
    """Annule les paris UP dont la confiance P(up) < seuil effectif (UP_CONF_MIN, ou
    son override regime si regime_map est fourni -- cf effective_up_conf_min).
    DOWN inchange. regime_map=None (defaut) = comportement identique a avant (seuil
    scalaire UP_CONF_MIN pour tous les bars). Renvoie (team_bets_filtre, n_removed_up)."""
    if not UP_CONF_FILTER_ENABLED or team_bets.empty:
        return team_bets, 0
    eff_min = effective_up_conf_min(team_bets["timestamp"].to_numpy(), regime_map)
    drop = (team_bets["direction"] == "up").to_numpy() & (team_bets["confidence"].to_numpy() < eff_min)
    removed = int(drop.sum())
    return team_bets[~drop].reset_index(drop=True), removed


def apply_xmodel_filter(up_bets: pd.DataFrame, down_full: pd.DataFrame) -> pd.DataFrame:
    """Filtre CROSS-MODEL UP (Strategie 2, campagne 500%/cote 2026-07-01) : ne
    garde un LONG que si le modele DOWN est confiant qu'il n'y aura PAS de chute :
    P(down) au meme bar < UP_XMODEL_MAX_PDOWN. down_full["confidence"] = P(down)
    par bar (batch, backtest). Voir up_xmodel_blocks pour l'equivalent bar-unique."""
    if not UP_XMODEL_FILTER_ENABLED or up_bets.empty:
        return up_bets
    pdown = dict(zip(down_full["timestamp"].astype("int64"), down_full["confidence"]))
    keep = up_bets["timestamp"].map(lambda t: pdown.get(int(t), 0.0) < UP_XMODEL_MAX_PDOWN)
    return up_bets[keep].reset_index(drop=True)


def up_xmodel_blocks(p_down: float) -> bool:
    """True si le filtre cross-model bloque un LONG etant donne P(down) au meme
    bar (bar-unique, pour le live). Miroir de apply_xmodel_filter."""
    if not UP_XMODEL_FILTER_ENABLED:
        return False
    return p_down >= UP_XMODEL_MAX_PDOWN


def effective_down_exit_params(ts: int, regime_map: dict[int, str] | None) -> tuple[int | None, float | None]:
    """(max_steps, trail_gap) effectifs pour un trade DOWN entre a la bougie de
    DECISION `ts` (jamais revu apres -- un trade ouvert n'est jamais modifie),
    selon REGIME_EXIT_OVERRIDES. regime_map=None -> (None, None) : le caller
    retombe sur DOWN_MAX_STEPS/DOWN_TRAIL_GAP de config_main.py, comportement
    inchange. `max_steps` est en pas de DOWN_CHECK_MIN minutes (backtest) ; le
    live doit multiplier par DOWN_CHECK_MIN pour obtenir des minutes."""
    if regime_map is None:
        return None, None
    from config_main import DOWN_MAX_STEPS, DOWN_TRAIL_GAP
    regime = regime_map.get(int(ts), "neutral")
    ov = cfg_regime.REGIME_EXIT_OVERRIDES.get(regime, {})
    hold_mult = ov.get("down_hold_mult", 1.0)
    trail_mult = ov.get("down_trail_gap_mult", 1.0)
    max_steps = int(DOWN_MAX_STEPS * hold_mult) if hold_mult != 1.0 else None
    trail_gap = DOWN_TRAIL_GAP * trail_mult if trail_mult != 1.0 else None
    return max_steps, trail_gap


# ----------------------------------------------------------------------------
#  Stop-loss temporel (decroissant) -- partage exit management backtest/live
# ----------------------------------------------------------------------------
def sl_threshold(j, max_steps, start_dec, end_dec, floor_dec):
    """SL TEMPOREL lineaire : seuil prix (decimal neg) de start_dec (large, j=0) vers
    end_dec (serre, j=max_steps), jamais plus serre (plus proche de 0) que floor_dec.
    `j` : nombre de pas de CHECK_MIN minutes ecoules depuis l'entree (backtest :
    index de boucle ; live : minutes_ecoulees // CHECK_MIN)."""
    if max_steps <= 0:
        return start_dec
    s = start_dec + (end_dec - start_dec) * (j / max_steps)
    return min(s, floor_dec)
