"""filters.py — logique de decision partagee entre le backtest (main.py) et le
daemon live (live/). Extrait de main.py (campagne symetrie OKX 2026-07-09) :
regle de coherence, filtre DMI, les 3 filtres d'entree UP, le wrapper regime.

CE FICHIER EST LE BACKTEST-SEULEMENT : main.py l'importe directement. DEPUIS
2026-08-03 (campagne autonomie complete live/), `live/filters.py` est une
COPIE INDEPENDANTE de ce fichier (plus un import partage) -- live/ peut
tourner seul sur un VPS sans aucune dependance a modele_team/. Si tu changes
un seuil ou une regle ici, recopie manuellement le fichier dans live/filters.py
pour que le live suive -- ca ne se propage plus tout seul (cf live/CLAUDE.md).
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
    CHOP_ER_DAYS, CHOP_ER_MAX, CHOP_SMA_DAYS, CHOP_BEAR_AGE_MAX_H,
    CHOP_UP_TP, MBULL_UP_TP, MBULL_UP_TP_ENABLED, CHOP_DOWN_CONF_DELTA_BUMP,
    CHOP_DOWN_RATCHET_STEP, CHOP_UP_CONF_MIN_BUMP, CHOP_DOWN_ENABLED,
    EXPOSURE,
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
    Couvre tout l'historique (la fenetre de test/live est un sous-ensemble)."""
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
#  Contexte de tendance longue (campagne TOUT-TERRAIN 2026-08-11) : ER de
#  Kaufman + age du bear sous SMA + direction macro, par barre 1h. Consomme par
#  main.simulate_trades pour le MODE DEFENSIF CHOP (config CHOP_* / MBULL_*).
#  AUCUN lookahead : la valeur au ts de decision n'utilise que les closes <= ts
#  (la barre de decision vient de cloturer quand on decide, meme convention que
#  le filtre DMI). trend_ctx=None partout = comportement strictement inchange.
# ----------------------------------------------------------------------------
_TREND_CTX_CACHE: dict[int, tuple[float, float, bool]] | None = None


def load_trend_context() -> dict[int, tuple[float, float, bool]]:
    """Map open-time 1h (ms) -> (er, age_bear_h, is_mbull), tout l'historique.
    er : Efficiency Ratio de Kaufman sur CHOP_ER_DAYS jours en PAS QUOTIDIENS
    (24 barres 1h), rafraichi a chaque barre. age_bear_h : heures consecutives
    passees sous la SMA CHOP_SMA_DAYS jours. is_mbull : close > cette SMA.
    Mis en cache module (une seule lecture SQLite par process)."""
    global _TREND_CTX_CACHE
    if _TREND_CTX_CACHE is not None:
        return _TREND_CTX_CACHE
    con = sqlite3.connect(str(_candles_db_path()))
    try:
        c = pd.read_sql_query(
            "SELECT timestamp_ms, close FROM candles WHERE timeframe='1h' ORDER BY timestamp_ms",
            con)
    finally:
        con.close()
    c = c.dropna(subset=["close"]).reset_index(drop=True)
    ts = c["timestamp_ms"].astype("int64").to_numpy()
    cl = c["close"].to_numpy(dtype=float)
    n, step = CHOP_ER_DAYS, 24
    er = np.full(len(cl), np.nan)
    den = np.zeros(len(cl))
    idx = np.arange(step * n, len(cl))
    num = np.abs(cl[idx] - cl[idx - step * n])
    for k in range(1, n + 1):
        den[idx] += np.abs(cl[idx - step * (k - 1)] - cl[idx - step * k])
    er[idx] = np.where(den[idx] > 0, num / den[idx], np.nan)
    sma = pd.Series(cl).rolling(CHOP_SMA_DAYS * 24).mean().to_numpy()
    age = np.zeros(len(cl))
    for i in range(1, len(cl)):
        if not np.isnan(sma[i]) and cl[i] < sma[i]:
            age[i] = age[i - 1] + 1
    out = {}
    for i in range(len(ts)):
        if np.isnan(er[i]) or np.isnan(sma[i]):
            continue                      # warmup -> absent = jamais en mode scalp
        out[int(ts[i])] = (float(er[i]), float(age[i]), bool(cl[i] > sma[i]))
    _TREND_CTX_CACHE = out
    return out


def chop_mode(trend_ctx: dict | None, decision_ts: int) -> bool:
    """True si la barre de decision est en CHOP scalpable : ER bas ET bear pas
    installe (jeune correction / range, jamais un bear de plusieurs mois)."""
    if trend_ctx is None:
        return False
    e = trend_ctx.get(int(decision_ts))
    return e is not None and e[0] < CHOP_ER_MAX and e[1] <= CHOP_BEAR_AGE_MAX_H


def dn_ratchet_extra(trend_ctx: dict | None, decision_ts: int) -> tuple | None:
    """Palier de cliquet DOWN supplementaire (campagne WR-DOWN 2026-08-11) :
    CHOP_DOWN_RATCHET_STEP quand la barre de decision est en chop scalpable OU
    en macro-bull — c.-a-d. partout SAUF bear installe (age > 30j), ou les
    trend-rides gardent tout leur espace. None = pas de palier supplementaire.
    CHOP_DOWN_ENABLED=False (retour DOWN au commit ecb04f8, 2026-08-14) ->
    toujours None."""
    if trend_ctx is None or not CHOP_DOWN_ENABLED:
        return None
    e = trend_ctx.get(int(decision_ts))
    if e is None:
        return None
    in_chop = e[0] < CHOP_ER_MAX and e[1] <= CHOP_BEAR_AGE_MAX_H
    if in_chop or e[2]:
        return CHOP_DOWN_RATCHET_STEP
    return None


def up_tp_override(trend_ctx: dict | None, decision_ts: int) -> float | None:
    """TP UP effectif de la barre : CHOP_UP_TP en mode scalp, MBULL_UP_TP en
    macro-bull hors chop, None (= UP_TAKE_PROFIT config) sinon."""
    if trend_ctx is None:
        return None
    e = trend_ctx.get(int(decision_ts))
    if e is None:
        return None
    if e[0] < CHOP_ER_MAX and e[1] <= CHOP_BEAR_AGE_MAX_H:
        return CHOP_UP_TP
    if MBULL_UP_TP_ENABLED and e[2]:
        return MBULL_UP_TP
    return None


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


def effective_down_delta(timestamps, regime_map: dict[int, str] | None,
                         trend_ctx: dict | None = None) -> np.ndarray:
    """DOWN_CONF_DELTA effectif par bar : le scalaire de config_main.py, sauf
    override du regime actif a ce timestamp (regime_map=None -> DOWN_CONF_DELTA
    partout, comportement actuel inchange). trend_ctx (campagne DOWN-ANNEE-1
    2026-08-11) : sur les barres en chop scalpable, on AJOUTE
    CHOP_DOWN_CONF_DELTA_BUMP — un short en chop doit etre plus sur qu'un short
    en tendance (les 34 SL DOWN chop de l'annee 1 etaient les shorts marginaux).
    trend_ctx=None (defaut, harnais/sweeps) = aucun bump, comportement inchange."""
    if regime_map is None and trend_ctx is None:
        return np.full(len(timestamps), DOWN_CONF_DELTA)
    out = np.empty(len(timestamps))
    for i, ts in enumerate(timestamps):
        regime = regime_map.get(int(ts), "neutral") if regime_map else "neutral"
        override = cfg_regime.REGIME_OVERRIDES.get(regime, {}).get("down_conf_delta")
        out[i] = DOWN_CONF_DELTA if override is None else override
        # CHOP_DOWN_ENABLED=False (retour DOWN au commit ecb04f8, 2026-08-14) :
        # plus aucun bump -- delta plat DOWN_CONF_DELTA comme au commit.
        if CHOP_DOWN_ENABLED and trend_ctx is not None and chop_mode(trend_ctx, int(ts)):
            out[i] += CHOP_DOWN_CONF_DELTA_BUMP
    return out


def effective_up_conf_min(timestamps, regime_map: dict[int, str] | None,
                          trend_ctx: dict | None = None) -> np.ndarray:
    """UP_CONF_MIN effectif par pari : meme principe que effective_down_delta.
    trend_ctx (campagne UP-PERTES 2026-08-11) : sur les barres en chop scalpable,
    on AJOUTE CHOP_UP_CONF_MIN_BUMP — les 21 SL UP chop de l'annee 1 etaient les
    longs marginaux. trend_ctx=None (harnais/sweeps) = aucun bump, inchange."""
    if regime_map is None and trend_ctx is None:
        return np.full(len(timestamps), UP_CONF_MIN)
    out = np.empty(len(timestamps))
    for i, ts in enumerate(timestamps):
        regime = regime_map.get(int(ts), "neutral") if regime_map else "neutral"
        override = cfg_regime.REGIME_OVERRIDES.get(regime, {}).get("up_conf_min")
        out[i] = UP_CONF_MIN if override is None else override
        if trend_ctx is not None and chop_mode(trend_ctx, int(ts)):
            out[i] += CHOP_UP_CONF_MIN_BUMP
    return out


def effective_up_conf_min_1(ts: int, regime_map: dict[int, str] | None) -> float:
    """Version scalaire (bar unique) de effective_up_conf_min, pour le live."""
    return float(effective_up_conf_min(np.array([ts]), regime_map)[0])


def effective_down_delta_1(ts: int, regime_map: dict[int, str] | None) -> float:
    """Version scalaire (bar unique) de effective_down_delta, pour le live."""
    return float(effective_down_delta(np.array([ts]), regime_map)[0])


def apply_conf_filter(team_bets: pd.DataFrame,
                      regime_map: dict[int, str] | None = None,
                      trend_ctx: dict | None = None) -> tuple[pd.DataFrame, int]:
    """Annule les paris UP dont la confiance P(up) < seuil effectif (UP_CONF_MIN, ou
    son override regime si regime_map est fourni, + CHOP_UP_CONF_MIN_BUMP en chop
    si trend_ctx est fourni -- cf effective_up_conf_min). DOWN inchange.
    regime_map=None et trend_ctx=None (defauts) = comportement identique a avant
    (seuil scalaire UP_CONF_MIN pour tous les bars). Renvoie (team_bets_filtre,
    n_removed_up)."""
    if not UP_CONF_FILTER_ENABLED or team_bets.empty:
        return team_bets, 0
    eff_min = effective_up_conf_min(team_bets["timestamp"].to_numpy(), regime_map, trend_ctx)
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


# ----------------------------------------------------------------------------
#  Cliquet de gain (trailing a paliers) -- partage exit management backtest/live
# ----------------------------------------------------------------------------
def ratchet_floor(peak, steps, prop=None):
    """CLIQUET DE GAIN : plancher VERROUILLE par paliers en fonction du pic de
    profit deja atteint. Complete le trailing classique (plancher = pic - gap),
    qui ne verrouille RIEN tant que pic < gap -- trou structurel : cote UP le gap
    (1.00%) depasse le TP (0.60%), donc le trailing ne peut mathematiquement
    jamais verrouiller un gain ; cote DOWN la zone morte couvre les 75 premiers
    pourcents du chemin vers le TP.

    `steps` : suite de (peak_dec, locked_dec) en DECIMAL de mouvement de prix,
    sans levier (meme unite que sl_threshold / UP_TAKE_PROFIT / *_TRAIL_GAP).
    Des que le pic atteint `peak_dec`, le plancher ne peut plus descendre sous
    `locked_dec`. Renvoie None si aucun palier n'est franchi (-> le caller garde
    le seul plancher du trailing classique, comportement strictement inchange).

    `prop` : (arm_dec, frac) OPTIONNEL -- palier PROPORTIONNEL. Des que le pic
    atteint arm_dec, le plancher ne peut plus descendre sous frac * pic : on
    garantit de conserver une FRACTION du meilleur gain atteint, au lieu d'un
    niveau absolu. Generalise strictement les paliers plats (une echelle de
    paliers plats approxime une droite) et exprime directement ce que le trailing
    classique rate : un giveback PROPORTIONNEL au mouvement deja capture, la ou
    `pic - gap` autorise un giveback CONSTANT quelle que soit la taille du gain.
    frac=0 -> breakeven garanti au-dela de arm_dec.

    Le caller combine le tout : floor = max(pic - gap, ratchet_floor(pic)).
    `peak` etant monotone croissant et toutes les composantes etant croissantes
    en `peak`, le plancher resultant est monotone -- la garantie du trailing
    existant est preservee.
    """
    best = None
    if steps:
        for peak_dec, locked_dec in steps:
            if peak >= peak_dec and (best is None or locked_dec > best):
                best = locked_dec
    if prop is not None:
        arm_dec, frac = prop
        if peak >= arm_dec:
            v = frac * peak
            if best is None or v > best:
                best = v
    return best


def build_ratchet(steps_pct):
    """(peak_pct, locked_pct) en POURCENT -> tuple de (peak_dec, locked_dec) en
    decimal, trie par pic croissant. Miroir des lignes "derives, ne pas editer"
    de config_main.py : on n'edite que des _PCT, le moteur consomme des decimales."""
    if not steps_pct:
        return ()
    return tuple(sorted(((p / 100.0, f / 100.0) for p, f in steps_pct),
                        key=lambda s: s[0]))
