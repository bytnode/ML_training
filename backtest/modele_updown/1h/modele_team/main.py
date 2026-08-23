"""Team TRADING backtest (marche reel, P/L net de frais).

Difference avec l'ancien backtest WR : on ne compte plus seulement les paris
gagnants/perdants, on SIMULE le trade reel et on calcule le benefice/perte :

    1. les 2 modeles predisent (equipe : regle de coherence UP/DOWN/PASS/CANCEL)
    2. sur chaque pari pris, on ENTRE une position au debut de la bougie suivante
       (entree = open de la bougie i+1 ~ cloture de la bougie de decision)
         - pari UP   -> LONG  (on profite si la bougie monte)
         - pari DOWN -> SHORT (on profite si la bougie baisse)
    3. on ATTEND l'evolution de la bougie i+1
    4. on VEND/CLOTURE a la cloture de la bougie i+1
    5. on calcule le P/L : rendement directionnel du corps de bougie
         body_ret = (close[i+1] - open[i+1]) / open[i+1]
         gross    = +body_ret (UP)  ou  -body_ret (DOWN)
       moins les FRAIS : 0.05 % a l'achat + 0.05 % a la vente (= 0.10 % A/R).

Le rendement du corps de bougie est reconstruit depuis la table SQLite `candles`
(timeframe 30m) du package modele_down : body = (1+close_ret)/(1+open_ret) - 1,
sans prix absolus (scale-invariant). Aucune dependance a un flux de prix brut.

Working-directory contract :
    cd backtest/modele_updown/1h/modele_team
    python main.py
"""
from __future__ import annotations

import sys

# Force UTF-8 stdout on legacy Windows terminals (cp1252 can't encode box-drawing
# / arrows). Must happen before rich.Console is instantiated.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# main.py lives at backtest/modele_updown/1h/modele_team/main.py. The sibling
# packages modele_up / modele_down live ONE LEVEL UP (in 1h/). Inject that
# grandparent dir on sys.path so they import as top-level (1h/ starts with a
# digit and can't be a package; modele_team/ holds main.py + its helpers).
_PKG_PARENT = str(Path(__file__).resolve().parent.parent)
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from modele_up import api as api_up
from modele_down import api as api_down
from regime import config_regime as cfg_regime
import detailed_log
import full_logs
import filters
import okx_fees

ROOT = Path(__file__).resolve().parent
LOGS_DIR = ROOT / "logs"
TRADING_RUNS_CSV = LOGS_DIR / "trading_runs.csv"
TRADES_CSV = LOGS_DIR / "trades.csv"

# Toute la config editable du backtest vit dans config_main.py.
from config_main import (
    TEAM_TEST_DAYS, DAY_MS, INTERVAL_1H_MS,
    FEE_PER_SIDE, INITIAL_CAPITAL, POSITION_FRACTION, LEVERAGE, EXPOSURE,
    UP_MGMT_ENABLED, UP_MAX_STEPS, UP_STEP_MS, UP_CHECK_MIN,
    UP_TRAIL_ACTIVATE, UP_TRAIL_GAP, UP_TAKE_PROFIT,
    UP_SL_START_DEC, UP_SL_END_DEC, UP_SL_FLOOR_DEC,
    UP_SL_START_PCT, UP_SL_END_PCT, UP_SL_FLOOR_PCT,
    DOWN_MGMT_ENABLED, DOWN_MAX_STEPS, DOWN_STEP_MS,
    DOWN_TRAIL_ACTIVATE, DOWN_TRAIL_GAP, DOWN_TAKE_PROFIT,
    DOWN_SL_START_DEC, DOWN_SL_END_DEC, DOWN_SL_FLOOR_DEC,
    DOWN_SL_START_PCT, DOWN_SL_END_PCT, DOWN_SL_FLOOR_PCT,
    MAKER, MAKER_FEE_PER_SIDE, MAKER_FILL_WINDOW_MIN,
    DMI_FILTER_ENABLED, DMI_LENGTH, DMI_TIMEFRAME, DMI_SIDES,
    UP_CONF_FILTER_ENABLED, UP_CONF_MIN, UP_DROP_HOURS, DOWN_CONF_DELTA,
    UP_XMODEL_FILTER_ENABLED, UP_XMODEL_MAX_PDOWN,
    MAX_CONCURRENT_POSITIONS, MAX_CONCURRENT_POSITIONS_PER_SIDE,
    WIDEN_ON_PROFIT_ENABLED,
    UP_WIDEN_TP_MULT, UP_WIDEN_TRAIL_GAP_MULT, UP_WIDEN_MAX_HOLD_MULT,
    DOWN_WIDEN_TP_MULT, DOWN_WIDEN_TRAIL_GAP_MULT, DOWN_WIDEN_MAX_HOLD_MULT,
    UP_RATCHET_STEPS, DOWN_RATCHET_STEPS, RATCHET_SCALES_WITH_WIDEN,
    UP_RATCHET_PROP, DOWN_RATCHET_PROP,
    DOWN_TPEXT_ENABLED, DOWN_TPEXT_REGIMES, DOWN_TPEXT_MAX_ER,
    DOWN_TPEXT_SWING_P, DOWN_TPEXT_HARD_GUARD_DEC,
    CHOP_SCALP_ENABLED, CHOP_DOWN_TP, CHOP_DOWN_MAX_STEPS, CHOP_DOWN_TRAIL_GAP,
    CHOP_DOWN_TRAIL_ACTIVATE, CHOP_UP_SL_START_DEC, CHOP_UP_SL_FLOOR_DEC,
    CHOP_DOWN_ENABLED, DOWN_WIDEN_ENABLED,
)


def _read_csv_timestamps(api_module) -> pd.Series:
    """Read the timestamp column from a package's test_predictions.csv."""
    csv_path = Path(api_module.__file__).resolve().parent / "logs" / "test_predictions.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} introuvable. Lance d'abord `python train.py` dans le package."
        )
    return pd.read_csv(csv_path, usecols=["timestamp"])["timestamp"]


def compute_window() -> tuple[int, int]:
    """Anchor end to the most recent timestamp covered by *both* packages."""
    up_ts = _read_csv_timestamps(api_up)
    down_ts = _read_csv_timestamps(api_down)
    common_end = int(min(up_ts.max(), down_ts.max()))
    start = common_end - TEAM_TEST_DAYS * DAY_MS
    return start, common_end


def _taken_bets(full: pd.DataFrame, direction: str) -> pd.DataFrame:
    """Derive the 4-column taken-bets frame from a full per-bar frame."""
    cols = ["timestamp", "direction", "confidence", "label"]
    if full.empty:
        return pd.DataFrame(columns=cols)
    taken = full[full["take"] == 1].copy()
    if taken.empty:
        return pd.DataFrame(columns=cols)
    taken["direction"] = direction
    return taken[cols].reset_index(drop=True)


build_team_bets = filters.build_team_bets


# ----------------------------------------------------------------------------
#  Rendements de bougie (corps) depuis la SQLite candles 30m
# ----------------------------------------------------------------------------
def _load_body_returns() -> dict[int, float]:
    """Map open-time (ms) -> body return (close-open)/open de la bougie 1h.

    body = (1+close_ret)/(1+open_ret) - 1  (scale-invariant, pas de prix brut).
    Source : table `candles` (timeframe='1h') du package modele_down.
    """
    down_dir = Path(api_down.__file__).resolve().parent
    db_path = next((p for p in (down_dir / "db" / "btc_multi_tf.db",
                                down_dir / "DB" / "btc_multi_tf.db") if p.exists()), None)
    if db_path is None:
        raise FileNotFoundError(
            f"SQLite candles introuvable sous {down_dir} (db/ ou DB/). Lance data.py."
        )
    con = sqlite3.connect(str(db_path))
    try:
        c = pd.read_sql_query(
            "SELECT timestamp_ms, open_ret, close_ret FROM candles WHERE timeframe='1h'", con)
    finally:
        con.close()
    c = c.dropna(subset=["open_ret", "close_ret"])
    body = (1.0 + c["close_ret"].to_numpy()) / (1.0 + c["open_ret"].to_numpy()) - 1.0
    return dict(zip(c["timestamp_ms"].astype("int64").to_numpy(), body))


def _load_intrabar_prices():
    """Renvoie (open_by_ts, close_by_ts) des sous-bougies pour les sorties gerees
    UP/DOWN (TP/trailing/SL temporel). Source : DB_<m>m/btc_<m>m.parquet (OHLCV
    brut, m = UP_CHECK_MIN = DOWN_CHECK_MIN = 1). None si absent (sortie classique
    a la cloture 1h)."""
    m = UP_CHECK_MIN
    fp = ROOT / f"DB_{m}m" / f"btc_{m}m.parquet"
    if not fp.exists():
        return None
    df = pd.read_parquet(fp)
    ts = df["timestamp_ms"].astype("int64").to_numpy()
    open_by = dict(zip(ts, df["open"].astype(float).to_numpy()))
    close_by = dict(zip(ts, df["close"].astype(float).to_numpy()))
    return open_by, close_by


def _load_intrabar_extremes():
    """Renvoie (low_by, high_by) des sous-bougies 1-min pour le test de remplissage
    MAKER (une limite a P0 est remplie si le prix vient la toucher). Meme source que
    _load_intrabar_prices : DB_1m/btc_1m.parquet. None si absent."""
    m = UP_CHECK_MIN
    fp = ROOT / f"DB_{m}m" / f"btc_{m}m.parquet"
    if not fp.exists():
        return None
    df = pd.read_parquet(fp)
    ts = df["timestamp_ms"].astype("int64").to_numpy()
    low_by = dict(zip(ts, df["low"].astype(float).to_numpy()))
    high_by = dict(zip(ts, df["high"].astype(float).to_numpy()))
    return low_by, high_by


def _load_swing_pivots(p: int):
    """Pivots HAUTS confirmes du 1-min (p barres de chaque cote, confirme p barres
    plus tard -- AUCUN lookahead : au moment t on ne connait que les pivots deja
    confirmes). Renvoie {ts_ms: (niveau_px, ts_naissance_ms)} pour la sortie
    structure de l'extension TP DOWN (config DOWN_TPEXT_*). None si pas de 1-min."""
    m = UP_CHECK_MIN
    fp = ROOT / f"DB_{m}m" / f"btc_{m}m.parquet"
    if not fp.exists():
        return None
    df = pd.read_parquet(fp)
    ts = df["timestamp_ms"].astype("int64").to_numpy()
    hi = df["high"].astype(float).to_numpy()
    hh = pd.Series(hi)
    left = hh.rolling(p).max().shift(1).to_numpy()
    right = hh.shift(-p).rolling(p).max().to_numpy()
    isp = (hi > np.nan_to_num(left, nan=-1.0)) & (hi >= np.nan_to_num(right, nan=-1.0))
    out = {}
    cur_lvl, cur_born = float("nan"), -1
    for i in range(len(ts)):
        b = i - p
        if b >= 0 and isp[b]:
            cur_lvl, cur_born = hi[b], int(ts[b])
        out[int(ts[i])] = (cur_lvl, cur_born)
    return out


_sl_threshold = filters.sl_threshold
_ratchet_floor = filters.ratchet_floor

# Cliquet de gain (campagne SECURISATION 2026-08-11) : paliers EFFECTIFS lus a
# CHAQUE appel des sorties gerees. Variables de MODULE (pas des constantes
# importees) pour que les sweeps puissent les surcharger sans recharger
# config_main -- cf optimize/sweep_ratchet.py. Valeurs par defaut = config.
UP_RATCHET = UP_RATCHET_STEPS
DOWN_RATCHET = DOWN_RATCHET_STEPS
UP_RATCHET_P = UP_RATCHET_PROP
DOWN_RATCHET_P = DOWN_RATCHET_PROP

# Pivots hauts 1-min pour l'extension TP DOWN (charges une fois au premier
# simulate_trades ; None si DB_1m absent ou extension desactivee).
_SWING_PIVOTS: dict | None = None


def _widened_ratchet(steps, prop, tp_mult):
    """Paliers apres elargissement sur profit. RATCHET_SCALES_WITH_WIDEN=False
    (defaut) -> inchanges : le gain deja verrouille le reste au meme niveau.
    Le palier PROPORTIONNEL est intrinsequement relatif au pic -- seul son seuil
    d'armement est mis a l'echelle, jamais la fraction."""
    if not RATCHET_SCALES_WITH_WIDEN:
        return steps, prop
    steps = tuple((p * tp_mult, f * tp_mult) for p, f in steps) if steps else steps
    prop = (prop[0] * tp_mult, prop[1]) if prop is not None else None
    return steps, prop


def _up_managed_exit(prices_intrabar, traded_ts, widen_candidates: set | None = None,
                     take_profit: float | None = None,
                     sl_override: tuple[float, float] | None = None):
    # Sortie geree UNIFORME d'un trade UP (long) : take-profit + trailing 'let-run'
    # (verrouillent le gain -> remontent le WR) + stop-loss dur large (cape la perte
    # sans whipsaw). Verifie chaque UP_CHECK_MIN min sur au plus UP_MAX_STEPS pas
    # (ELARGI a UP_MAX_STEPS*UP_WIDEN_MAX_HOLD_MULT si elargissement declenche,
    # cf widen_candidates ci-dessous).
    # Renvoie (gross, reason, exit_ts, widened_at_ts) ; (None, "close", traded_ts+1h,
    # None) si pas de 1min.
    # exit_ts (campagne symetrie 2026-07-09) sert au plafond de concurrence de
    # simulate_trades (MAX_CONCURRENT_POSITIONS) -- meme role que la fermeture de
    # position cote live/ (voir live/position.py).
    #
    # widen_candidates (test 2026-07-10) : ensemble des traded_ts (ms) des futurs
    # signaux UP bloques par MAX_CONCURRENT_POSITIONS_PER_SIDE pendant que CETTE
    # position est ouverte. Des qu'un de ces instants est atteint par la marche ET
    # que le profit courant est POSITIF, on elargit UNE SEULE FOIS take_profit /
    # trail_gap / max_steps (SL jamais elargi -- reste borne a l'horizon ORIGINAL
    # UP_MAX_STEPS, cf sl_j ci-dessous). None/vide/WIDEN_ON_PROFIT_ENABLED=False
    # -> comportement exactement inchange.
    open_ib, close_ib = prices_intrabar
    entry = open_ib.get(traded_ts)
    if entry is None:
        return None, "close", traded_ts + INTERVAL_1H_MS, None
    peak = -9.0
    floor = None
    last = None
    last_j = 0
    # take_profit (param, MODE DEFENSIF CHOP 2026-08-11) : TP override de la barre
    # (chop scalp / macro-bull), None = UP_TAKE_PROFIT config. base_tp sert de
    # reference au widen (le TP elargi suit le TP effectif, pas la constante).
    base_tp = take_profit if take_profit is not None else UP_TAKE_PROFIT
    take_profit, trail_gap, max_steps = base_tp, UP_TRAIL_GAP, UP_MAX_STEPS
    ratchet, ratchet_p = UP_RATCHET, UP_RATCHET_P
    # sl_override (MODE DEFENSIF CHOP, campagne POINTS-FAIBLES 2026-08-11) :
    # (start_dec, floor_dec) du SL temporel en chop ; END jamais modifie.
    sl_start, sl_floor = sl_override if sl_override is not None \
        else (UP_SL_START_DEC, UP_SL_FLOOR_DEC)
    widened_at_ts = None
    j = 0
    while j < max_steps:
        t = traded_ts + j * UP_STEP_MS
        cl = close_ib.get(t)
        if cl is None:
            break
        pp = cl / entry - 1.0
        last = pp
        last_j = j
        if (WIDEN_ON_PROFIT_ENABLED and widened_at_ts is None and widen_candidates
                and pp > 0.0 and t in widen_candidates):
            take_profit = base_tp * UP_WIDEN_TP_MULT
            trail_gap = UP_TRAIL_GAP * UP_WIDEN_TRAIL_GAP_MULT
            max_steps = int(UP_MAX_STEPS * UP_WIDEN_MAX_HOLD_MULT)
            ratchet, ratchet_p = _widened_ratchet(ratchet, ratchet_p, UP_WIDEN_TP_MULT)
            widened_at_ts = t
        sl_j = min(j, UP_MAX_STEPS - 1)     # SL jamais elargi -- horizon ORIGINAL toujours
        sl_th = _sl_threshold(sl_j, UP_MAX_STEPS, sl_start, UP_SL_END_DEC, sl_floor)
        if pp <= sl_th:
            return pp, "stop_loss", t, widened_at_ts
        if pp >= take_profit:
            return pp, "take_profit", t, widened_at_ts
        if pp > peak:
            peak = pp
        # Plancher effectif = max(trailing classique, cliquet de gain). Les deux
        # sont monotones croissants (peak l'est), donc le plancher l'est aussi.
        if peak >= UP_TRAIL_ACTIVATE:
            floor = peak - trail_gap
        rf = _ratchet_floor(peak, ratchet, ratchet_p)
        if rf is not None and (floor is None or rf > floor):
            floor = rf
        if floor is not None and pp < floor:
            return pp, "trailing", t, widened_at_ts
        j += 1
    if last is None:
        return None, "close", traded_ts + INTERVAL_1H_MS, widened_at_ts
    return last, "timestop", traded_ts + last_j * UP_STEP_MS, widened_at_ts


_effective_down_exit_params = filters.effective_down_exit_params


def _down_managed_exit(prices_intrabar, traded_ts, max_steps=None, trail_gap=None,
                       widen_candidates: set | None = None, regime: str = "neutral",
                       swing_pivots: dict | None = None,
                       take_profit: float | None = None,
                       trail_activate: float | None = None,
                       ratchet_extra: tuple | None = None):
    # Sortie geree UNIFORME d'un trade DOWN (short), calque exact de _up_managed_exit
    # mais sur le profit SHORT pp = 1 - close/entry (un short gagne quand le prix baisse).
    # take-profit + trailing 'let-run' (verrouillent le gain) + stop-loss dur large
    # (cape la perte sans whipsaw). Verifie chaque DOWN_CHECK_MIN min sur <= max_steps pas.
    # max_steps/trail_gap (parametres d'entree) : override regime-conditionnel
    # (REGIME_EXIT_OVERRIDES, cf _effective_down_exit_params) ; None (defaut) =
    # DOWN_MAX_STEPS/DOWN_TRAIL_GAP de config_main.py, comportement inchange. Ce
    # sont ces valeurs EFFECTIVES (deja regime-ajustees) qui servent de BASE a
    # l'elargissement sur profit ci-dessous, pas les constantes brutes.
    # Renvoie (gross_short, reason, exit_ts, widened_at_ts) ; (None, "close",
    # traded_ts+1h, None) si pas de 1min.
    # exit_ts (campagne symetrie 2026-07-09) sert au plafond de concurrence de
    # simulate_trades (MAX_CONCURRENT_POSITIONS).
    #
    # widen_candidates (test 2026-07-10) : cf _up_managed_exit, meme mecanisme
    # cote DOWN. SL jamais elargi -- reste borne a l'horizon EFFECTIF (regime-
    # ajuste mais non-widened) sl_max_steps ci-dessous.
    sl_max_steps = max_steps if max_steps is not None else DOWN_MAX_STEPS
    if max_steps is None:
        max_steps = DOWN_MAX_STEPS
    if trail_gap is None:
        trail_gap = DOWN_TRAIL_GAP
    # take_profit (param, MODE DEFENSIF CHOP 2026-08-11) : TP override de la barre
    # (chop scalp), None = DOWN_TAKE_PROFIT config. base_tp sert de reference au
    # widen ET au plancher dur tpext (le TP effectif, pas la constante).
    base_tp = take_profit if take_profit is not None else DOWN_TAKE_PROFIT
    take_profit = base_tp
    # trail_activate (MODE DEFENSIF CHOP, campagne POINTS-FAIBLES 2026-08-11) :
    # seuil d'armement du trailing en chop (le short doit prouver un profit
    # avant que le trailing puisse le sortir) ; None = DOWN_TRAIL_ACTIVATE config.
    trail_act = trail_activate if trail_activate is not None else DOWN_TRAIL_ACTIVATE
    # ratchet_extra (campagne WR-DOWN 2026-08-11) : palier supplementaire hors
    # bear installe (cf filters.dn_ratchet_extra), fusionne avec les paliers de
    # securite ; None = paliers production inchanges.
    ratchet = DOWN_RATCHET if ratchet_extra is None \
        else tuple(sorted(DOWN_RATCHET + (ratchet_extra,)))
    ratchet_p = DOWN_RATCHET_P
    # Extension conditionnelle du TP en bear (campagne ADAPTATIVITE 2026-08-11,
    # cf bloc DOWN_TPEXT_* de config_main.py) : etat + suivi du chemin pour le
    # gate ER (grinders only).
    tpext_armed = (DOWN_TPEXT_ENABLED and regime in DOWN_TPEXT_REGIMES
                   and swing_pivots is not None)
    extended = False
    ext_t = None
    path_len = 0.0
    prev_pp = 0.0
    open_ib, close_ib = prices_intrabar
    entry = open_ib.get(traded_ts)
    if entry is None:
        return None, "close", traded_ts + INTERVAL_1H_MS, None
    peak = -9.0
    floor = None
    last = None
    last_j = 0
    widened_at_ts = None
    j = 0
    while j < max_steps:
        t = traded_ts + j * DOWN_STEP_MS
        cl = close_ib.get(t)
        if cl is None:
            break
        pp = 1.0 - cl / entry            # profit directionnel d'un short
        if j > 0:
            path_len += abs(pp - prev_pp)
        prev_pp = pp
        last = pp
        last_j = j
        if (WIDEN_ON_PROFIT_ENABLED and widened_at_ts is None and widen_candidates
                and pp > 0.0 and t in widen_candidates):
            take_profit = base_tp * DOWN_WIDEN_TP_MULT
            trail_gap = trail_gap * DOWN_WIDEN_TRAIL_GAP_MULT
            max_steps = int(max_steps * DOWN_WIDEN_MAX_HOLD_MULT)
            ratchet, ratchet_p = _widened_ratchet(ratchet, ratchet_p, DOWN_WIDEN_TP_MULT)
            widened_at_ts = t
            extended = False               # le widen (TP x4) prime sur l'extension
        sl_j = min(j, sl_max_steps - 1)   # SL jamais elargi -- horizon EFFECTIF (non-widened)
        sl_th = _sl_threshold(sl_j, sl_max_steps, DOWN_SL_START_DEC, DOWN_SL_END_DEC, DOWN_SL_FLOOR_DEC)
        if pp <= sl_th:
            return pp, "stop_loss", t, widened_at_ts
        # --- Armement de l'extension TP (bear + grinder) au franchissement du TP ---
        if (not extended and tpext_armed and widened_at_ts is None
                and pp >= take_profit):
            er = pp / max(path_len, 1e-9)
            if er <= DOWN_TPEXT_MAX_ER:
                extended = True
                ext_t = t                  # les pivots nes avant ne comptent pas
            else:
                return pp, "take_profit", t, widened_at_ts
        if not extended and pp >= take_profit:
            return pp, "take_profit", t, widened_at_ts
        if pp > peak:
            peak = pp
        # Plancher effectif = max(trailing classique, cliquet de gain), cf _up_managed_exit.
        if peak >= trail_act:
            floor = peak - trail_gap
        rf = _ratchet_floor(peak, ratchet, ratchet_p)
        if rf is not None and (floor is None or rf > floor):
            floor = rf
        if extended:
            # Sortie STRUCTURE : cassure du dernier pivot haut confirme forme
            # PENDANT l'extension ; plancher dur TP - HARD_GUARD sinon.
            hard = base_tp - DOWN_TPEXT_HARD_GUARD_DEC
            sfloor = hard
            piv = swing_pivots.get(t)
            if piv is not None and piv[1] >= ext_t:
                lvl_pp = 1.0 - piv[0] / entry
                if lvl_pp > sfloor:
                    sfloor = lvl_pp
            if floor is None or sfloor > floor:
                floor = sfloor
        if floor is not None and pp < floor:
            return pp, "trailing", t, widened_at_ts
        j += 1
    if last is None:
        return None, "close", traded_ts + INTERVAL_1H_MS, widened_at_ts
    return last, "timestop", traded_ts + last_j * DOWN_STEP_MS, widened_at_ts


def simulate_trades(team_bets: pd.DataFrame, body_map: dict[int, float],
                    prices_intrabar: tuple | None = None,
                    up_meta: dict[int, tuple] | None = None,
                    extremes: tuple | None = None,
                    regime_map: dict[int, str] | None = None,
                    fee_override: dict | None = None,
                    trend_ctx: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Un trade par pari pris. Entree open(i+1), sortie close(i+1).
    gross = +body (UP/long) ou -body (DOWN/short) ; net = gross - (entry_fee + exit_fee).
    L'ENTREE suit MAKER/TAKER (limite a P0, simulee via le fill intrabar) ; la SORTIE
    (close/take_profit/trailing/stop_loss) est TOUJOURS TAKER : impossible de garantir
    un fill maker sur une sortie declenchee par une condition (execution immediate requise).

    Si prices_intrabar est fourni, la sortie est geree minute par minute par
    _up_managed_exit/_down_managed_exit (TP / trailing / SL temporel / timestop) ;
    sinon sortie a la cloture 1h (exit_reason='close').

    fee_override (campagne symetrie 2026-07-09, cf okx_fees.py) : {"maker": float,
    "taker": float} recupere en LIVE depuis l'API OKX (taux reel du compte, pas
    une estimation) ; None (defaut) = comportement inchange, MAKER_FEE_PER_SIDE/
    FEE_PER_SIDE de config_main.py.

    trend_ctx (MODE DEFENSIF CHOP, campagne TOUT-TERRAIN 2026-08-11) : map de
    filters.load_trend_context(). Quand la barre de decision est en chop
    (filters.chop_mode), les SORTIES passent en scalp mean-reversion : TP UP/DOWN
    courts, gap/hold DOWN serres (config CHOP_*), tpext et override regime bear
    coupes ; hors chop, TP UP serre en macro-bull (MBULL_UP_TP). Les ENTREES ne
    changent jamais. None (defaut) = comportement strictement inchange."""
    cols = ["timestamp", "ts_utc", "traded_ts", "direction", "label",
            "body_ret", "gross_ret", "net_ret", "win_net", "exit_reason", "exit_ts"]
    maker_fee_rate = fee_override["maker"] if fee_override else MAKER_FEE_PER_SIDE
    taker_fee_rate = fee_override["taker"] if fee_override else FEE_PER_SIDE
    entry_fee = maker_fee_rate if MAKER else taker_fee_rate    # entree : maker si rempli, sinon taker
    exit_fee = taker_fee_rate                                    # sortie : toujours taker
    low_by, high_by = extremes if (MAKER and extremes is not None) else (None, None)
    open_ib0 = prices_intrabar[0] if prices_intrabar is not None else None
    n_would_be = n_missed = missed_up = missed_down = 0
    n_concurrency_skipped = up_concurrency_skipped = down_concurrency_skipped = 0
    n_widened_up = n_widened_down = 0
    if team_bets.empty:
        return (pd.DataFrame(columns=cols),
                {"maker": MAKER, "n_would_be": 0, "n_missed": 0, "missed_up": 0,
                 "missed_down": 0, "fill_rate": 1.0, "entry_fee": entry_fee, "exit_fee": exit_fee,
                 "n_concurrency_skipped": 0, "up_concurrency_skipped": 0, "down_concurrency_skipped": 0,
                 "n_widened_up": 0, "n_widened_down": 0})
    global _SWING_PIVOTS
    if DOWN_TPEXT_ENABLED and _SWING_PIVOTS is None and prices_intrabar is not None:
        _SWING_PIVOTS = _load_swing_pivots(DOWN_TPEXT_SWING_P)
    rows = []
    # Elargissement sur profit (test 2026-07-10) : traded_ts (ms) de TOUS les paris
    # UP/DOWN de team_bets, tries -- sert a batir, pour chaque position ouverte, la
    # liste des futurs signaux du meme sens qui pourraient la declencher (cf
    # widen_candidates dans _up_managed_exit/_down_managed_exit). Un seul passage
    # np.searchsorted par position ouverte, pas de cout notable.
    _up_ts_all = np.sort((team_bets.loc[team_bets["direction"] == "up", "timestamp"].to_numpy()
                          + INTERVAL_1H_MS)) if WIDEN_ON_PROFIT_ENABLED else None
    _down_ts_all = np.sort((team_bets.loc[team_bets["direction"] == "down", "timestamp"].to_numpy()
                            + INTERVAL_1H_MS)) if WIDEN_ON_PROFIT_ENABLED else None

    def _later_same_dir_candidates(traded_ts_now: int, up_side: bool) -> set | None:
        if not WIDEN_ON_PROFIT_ENABLED:
            return None
        arr = _up_ts_all if up_side else _down_ts_all
        idx = np.searchsorted(arr, traded_ts_now, side="right")
        return set(int(t) for t in arr[idx:])
    # exit_ts (ms) des positions actuellement "ouvertes", purge chronologiquement au
    # fil des trades (team_bets est trie par timestamp -> traded_ts est croissant).
    # Symetrie avec live_team (position.py::n_open + config_team.MAX_CONCURRENT_POSITIONS,
    # has_open_side retire) : plafond sur le compte TOTAL, tous sens confondus.
    open_slots: list[int] = []
    # Plafond PAR SENS (symetrie live/, campagne OKX 2026-07-10) : mirroir de la
    # contrainte OKX mode HEDGE (1 seule position suivable par sens, cf
    # MAX_CONCURRENT_POSITIONS_PER_SIDE dans config_main.py).
    up_open_slots: list[int] = []
    down_open_slots: list[int] = []
    for _, b in team_bets.iterrows():
        ts = int(b["timestamp"])
        traded_ts = ts + INTERVAL_1H_MS          # bougie i+1 = celle qu'on trade
        body = body_map.get(traded_ts)
        if body is None:
            continue                               # pas de bougie suivante connue -> skip
        up = b["direction"] == "up"
        n_would_be += 1
        # --- Mode MAKER : limite postee a P0 (open i+1) ; remplie SEULEMENT si le prix
        #     revient la toucher (long: un low 1-min < P0 ; short: un high 1-min > P0).
        #     Sinon = opportunite perdue (on ne prend pas le trade). ---
        if MAKER and low_by is not None and open_ib0 is not None:
            entry0 = open_ib0.get(traded_ts)
            lo = hi = None
            if entry0 is not None:
                for j in range(MAKER_FILL_WINDOW_MIN):
                    t = traded_ts + j * 60_000
                    lv = low_by.get(t)
                    hv = high_by.get(t)
                    if lv is not None:
                        lo = lv if lo is None else min(lo, lv)
                    if hv is not None:
                        hi = hv if hi is None else max(hi, hv)
            if entry0 is not None and lo is not None:
                filled = (lo < entry0) if up else (hi > entry0)
                if not filled:
                    n_missed += 1
                    if up:
                        missed_up += 1
                    else:
                        missed_down += 1
                    continue

        # --- Plafond de concurrence (MAX_CONCURRENT_POSITIONS, tous sens confondus,
        #     PUIS MAX_CONCURRENT_POSITIONS_PER_SIDE, par sens) : purge d'abord les
        #     slots deja clos avant CETTE entree (traded_ts), puis n'ouvre que si un
        #     slot est libre des DEUX cotes. Seuls les trades REELLEMENT REMPLIS
        #     (apres le filtre MAKER ci-dessus) consomment un slot -- symetrie avec
        #     live/ ou seul un fill reel ouvre une Position. ---
        open_slots = [e for e in open_slots if e > traded_ts]
        up_open_slots = [e for e in up_open_slots if e > traded_ts]
        down_open_slots = [e for e in down_open_slots if e > traded_ts]
        side_slots = up_open_slots if up else down_open_slots
        if (len(open_slots) >= MAX_CONCURRENT_POSITIONS
                or len(side_slots) >= MAX_CONCURRENT_POSITIONS_PER_SIDE):
            n_concurrency_skipped += 1
            if up:
                up_concurrency_skipped += 1
            else:
                down_concurrency_skipped += 1
            continue

        gross = body if up else -body
        exit_reason = "close"                      # defaut : sortie a la cloture 1h
        exit_ts = traded_ts + INTERVAL_1H_MS
        # --- Gestion de sortie geree par sens (let-run + TP + SL backstop) ; sinon SL simple ---
        # MODE DEFENSIF CHOP (2026-08-11) : sorties scalp quand la barre de
        # decision est en chop scalpable (ER bas + bear non installe).
        _chop = trend_ctx is not None and filters.chop_mode(trend_ctx, ts)
        if up and UP_MGMT_ENABLED and prices_intrabar is not None:
            widen_cands = _later_same_dir_candidates(traded_ts, True)
            _up_tp = filters.up_tp_override(trend_ctx, ts) if trend_ctx is not None else None
            _up_sl = (CHOP_UP_SL_START_DEC, CHOP_UP_SL_FLOOR_DEC) if _chop else None
            g2, exit_reason, exit_ts, widened_at = _up_managed_exit(
                prices_intrabar, traded_ts, widen_cands, take_profit=_up_tp,
                sl_override=_up_sl)
            if g2 is not None:
                gross = g2
            if widened_at is not None:
                n_widened_up += 1
        elif (not up) and DOWN_MGMT_ENABLED and prices_intrabar is not None:
            # Gates per-side (2026-08-14) : DOWN_WIDEN_ENABLED et CHOP_DOWN_ENABLED
            # (config_main.py fait foi) controlent respectivement le widen DOWN et
            # les sorties scalp DOWN -- False = coupes, True = actifs.
            widen_cands = (_later_same_dir_candidates(traded_ts, False)
                           if DOWN_WIDEN_ENABLED else None)
            if _chop and CHOP_DOWN_ENABLED:
                # scalp : TP/hold/gap serres, trailing arme sur profit prouve,
                # tpext et override regime bear coupes
                max_steps_eff, trail_gap_eff = CHOP_DOWN_MAX_STEPS, CHOP_DOWN_TRAIL_GAP
                _regime, _dn_tp = "neutral", CHOP_DOWN_TP
                _dn_act = CHOP_DOWN_TRAIL_ACTIVATE
            else:
                max_steps_eff, trail_gap_eff = _effective_down_exit_params(ts, regime_map)
                _regime = regime_map.get(ts, "neutral") if regime_map else "neutral"
                _dn_tp = _dn_act = None
            _dn_rx = filters.dn_ratchet_extra(trend_ctx, ts) if trend_ctx is not None else None
            g2, exit_reason, exit_ts, widened_at = _down_managed_exit(
                prices_intrabar, traded_ts, max_steps_eff, trail_gap_eff, widen_cands,
                _regime, _SWING_PIVOTS, take_profit=_dn_tp, trail_activate=_dn_act,
                ratchet_extra=_dn_rx)
            if g2 is not None:
                gross = g2
            if widened_at is not None:
                n_widened_down += 1

        open_slots.append(exit_ts)
        side_slots.append(exit_ts)
        net = gross - (entry_fee + exit_fee)
        rows.append({
            "timestamp": ts,
            "ts_utc": pd.Timestamp(ts, unit="ms", tz="UTC").strftime("%Y-%m-%d %H:%M"),
            "traded_ts": traded_ts,
            "direction": b["direction"],
            "label": int(b["label"]) if pd.notna(b["label"]) else None,
            "body_ret": body,
            "gross_ret": gross,
            "net_ret": net,
            "win_net": bool(net > 0),
            "exit_reason": exit_reason,
            # exit_ts (2026-08-11) : deja calcule pour le plafond de concurrence ;
            # persiste dans trades.csv (duree reelle de chaque position).
            "exit_ts": int(exit_ts),
        })
    fill_rate = (1.0 - n_missed / n_would_be) if n_would_be else 1.0
    maker_info = {"maker": MAKER, "n_would_be": n_would_be, "n_missed": n_missed,
                  "missed_up": missed_up, "missed_down": missed_down,
                  "fill_rate": fill_rate, "entry_fee": entry_fee, "exit_fee": exit_fee,
                  "n_concurrency_skipped": n_concurrency_skipped,
                  "up_concurrency_skipped": up_concurrency_skipped,
                  "down_concurrency_skipped": down_concurrency_skipped,
                  "n_widened_up": n_widened_up, "n_widened_down": n_widened_down}
    return pd.DataFrame(rows, columns=cols), maker_info


def trading_stats(trades: pd.DataFrame, initial: float = INITIAL_CAPITAL) -> dict:
    """Agrege le P/L : equity composee, ROI net, frais, WR, drawdown, par sens."""
    if trades.empty:
        return {"n_trades": 0}
    # Levier : exposition au mouvement de prix = POSITION_FRACTION * LEVERAGE (= EXPOSURE).
    # P/L ET frais (deja inclus dans net_ret) portent sur le notionnel -> x EXPOSURE.
    net = trades["net_ret"].to_numpy(dtype=float) * EXPOSURE
    gross = trades["gross_ret"].to_numpy(dtype=float) * EXPOSURE
    equity = initial * np.cumprod(1.0 + net)
    final = float(equity[-1])
    peak = np.maximum.accumulate(equity)
    max_dd = float(((equity - peak) / peak).min())

    # check d'alignement : le pari etait-il directionnellement correct ssi gross>0 ?
    lbl = trades["label"]
    chk = trades[lbl.notna() & (trades["body_ret"].abs() > 1e-12)]
    align = float(((chk["gross_ret"] > 0).to_numpy() == (chk["label"] == 1).to_numpy()).mean()) \
        if len(chk) else float("nan")

    def _dir(d):
        sub = trades[trades["direction"] == d]
        if sub.empty:
            return {"n": 0, "wr": float("nan"), "net_pct": 0.0}
        net = sub["net_ret"].to_numpy() * EXPOSURE
        wins = net[net > 0]
        losses = net[net < 0]
        wr = float((net > 0).mean())
        lr = float((net < 0).mean())
        aw_mag = float(wins.mean() * 100) if len(wins) else 0.0      # AvgWin magnitude (%)
        al_mag = float(-losses.mean() * 100) if len(losses) else 0.0  # AvgLoss magnitude (%)
        return {"n": len(sub),
                "wr": wr,
                "net_pct": float((np.prod(1.0 + net) - 1.0) * 100),
                "worst_pct": float(net.min() * 100),
                "best_pct": float(net.max() * 100),
                "avg_win_pct": float(wins.mean() * 100) if len(wins) else float("nan"),
                "avg_loss_pct": float(losses.mean() * 100) if len(losses) else float("nan"),
                "ev_pct": wr * aw_mag - lr * al_mag}  # EV = WR*AvgWin - LR*AvgLoss (%/trade)

    # --- P/L mensuel (USDC) : decoupe l'equity composee par mois calendaire ---
    # Les P/L $ mensuels se telescopent (somme = P/L total), donc moyenne = total / nb mois.
    tr_ts = trades["traded_ts"].to_numpy(dtype="int64")
    months = pd.to_datetime(tr_ts, unit="ms", utc=True).strftime("%Y-%m")
    equity_prev = np.concatenate(([initial], equity[:-1]))
    m_start, m_end = {}, {}
    for m, e0, e1 in zip(months, equity_prev, equity):
        if m not in m_start:
            m_start[m] = e0
        m_end[m] = e1
    month_usdc = {m: m_end[m] - m_start[m] for m in m_start}
    month_pct = {m: (m_end[m] / m_start[m] - 1.0) * 100 for m in m_start}
    n_months = len(m_start)
    total_pnl = final - initial
    best_m = max(month_usdc, key=month_usdc.get)
    worst_m = min(month_usdc, key=month_usdc.get)
    monthly_stats = {
        "usdc_total": total_pnl,
        "usdc_avg_per_month": total_pnl / n_months if n_months else 0.0,
        "n_months": n_months,
        "avg_month_pct": float(np.mean(list(month_pct.values()))),
        "best_month": best_m,
        "best_month_usdc": month_usdc[best_m],
        "best_month_pct": month_pct[best_m],
        "worst_month": worst_m,
        "worst_month_usdc": month_usdc[worst_m],
        "worst_month_pct": month_pct[worst_m],
    }

    # --- EV (esperance) par trade : EV = WR*avg_win - LR*avg_loss (net de frais, x
    #     levier). Mathematiquement = moyenne nette par trade ; on l'expose
    #     decomposee (formule utilisateur) pour l'afficher sur le headline. ---
    wins_all = net[net > 0]
    losses_all = net[net < 0]
    n_all = len(net)
    wr_all = len(wins_all) / n_all if n_all else 0.0
    lr_all = len(losses_all) / n_all if n_all else 0.0
    avg_win_all = float(wins_all.mean()) if len(wins_all) else 0.0
    avg_loss_all = float(-losses_all.mean()) if len(losses_all) else 0.0  # magnitude > 0
    ev_per_trade_pct = (wr_all * avg_win_all - lr_all * avg_loss_all) * 100

    return {
        "n_trades": len(net),
        **monthly_stats,
        "gross_total_pct": float((np.prod(1.0 + gross) - 1.0) * 100),
        "net_total_pct": float((final / initial - 1.0) * 100),
        "final_equity": final,
        "pnl": final - initial,
        "fee_drag_pct": float((gross - net).sum() * 100),  # frais cumules reels (x levier, maker/taker) en pts
        "avg_net_per_trade_pct": float(net.mean() * 100),
        "ev_per_trade_pct": float(ev_per_trade_pct),
        "avg_win_pct_all": float(avg_win_all * 100),    # AvgWin magnitude (%) tous trades
        "avg_loss_pct_all": float(avg_loss_all * 100),  # AvgLoss magnitude (%) tous trades
        "avg_gross_per_trade_pct": float(gross.mean() * 100),
        "wr_net": float((net > 0).mean()),
        "wr_dir": float((trades["label"] == 1).mean()),
        "best_pct": float(net.max() * 100),
        "worst_pct": float(net.min() * 100),
        "max_drawdown_pct": max_dd * 100,
        "align_check": align,
        "n_stop_loss": int((trades["exit_reason"] == "stop_loss").sum()) if "exit_reason" in trades else 0,
        "n_take_profit": int((trades["exit_reason"] == "take_profit").sum()) if "exit_reason" in trades else 0,
        "n_trailing": int((trades["exit_reason"] == "trailing").sum()) if "exit_reason" in trades else 0,
        "up": _dir("up"),
        "down": _dir("down"),
        "equity_final": final,
    }


def _agg_side(net: np.ndarray, gross: np.ndarray, contrib: np.ndarray) -> dict:
    """Metriques du bloc "Profit / Perte" sur un sous-ensemble de trades
    (une tranche, un sens, ou les deux). `contrib` = P/L $ reel de chaque trade
    dans l'equity globale (equity - equity_prev), de sorte que la somme
    UP + DOWN d'une tranche retombe exactement sur le P/L $ de la tranche."""
    if not len(net):
        return dict(n=0, gross_pct=0.0, net_pct=0.0, wr=float("nan"),
                    avg_win_pct=0.0, avg_loss_pct=0.0, ev_pct=0.0,
                    avg_gross_pct=0.0, best_pct=0.0, worst_pct=0.0,
                    dd_pct=0.0, fee_pts=0.0, pnl_usd=0.0)
    eq = np.cumprod(1.0 + net)
    peak = np.maximum.accumulate(eq)
    wins, losses = net[net > 0], net[net < 0]
    wr = float((net > 0).mean())
    lr = float((net < 0).mean())
    aw = float(wins.mean()) if len(wins) else 0.0
    al = float(-losses.mean()) if len(losses) else 0.0
    return dict(n=int(len(net)),
                gross_pct=float((np.prod(1.0 + gross) - 1.0) * 100),
                net_pct=float((eq[-1] - 1.0) * 100),
                wr=wr * 100,
                avg_win_pct=aw * 100,
                avg_loss_pct=al * 100,
                ev_pct=(wr * aw - lr * al) * 100,
                avg_gross_pct=float(gross.mean() * 100),
                best_pct=float(net.max() * 100),
                worst_pct=float(net.min() * 100),
                dd_pct=float(((eq - peak) / peak).min() * 100),
                fee_pts=float((gross - net).sum() * 100),
                pnl_usd=float(contrib.sum()))


def yearly_stats(trades: pd.DataFrame, start_ms: int, end_ms: int,
                 initial: float = INITIAL_CAPITAL) -> list[dict]:
    """Decoupe la fenetre en tranches de 365 jours depuis le DEBUT (la derniere
    tranche prend le reliquat) et agrege chaque tranche : n, net compose (%),
    WR, EV, pire trade, drawdown intra-tranche, P/L $ (equity courante --
    les P/L $ des tranches se telescopent, somme = P/L total).
    Ajoute 2026-08-11 (demande user apres le test 720j) : rend visible la
    dependance a l'annee de calibration au lieu d'un seul chiffre compose."""
    if trades.empty:
        return []
    year_ms = 365 * DAY_MS
    bounds = []
    t0 = start_ms
    while t0 < end_ms:
        t1 = min(t0 + year_ms, end_ms)
        if end_ms - t1 < 30 * DAY_MS:          # reliquat < 30j -> fusionne
            t1 = end_ms
        bounds.append((t0, t1))
        t0 = t1
    tr_ts = trades["traded_ts"].to_numpy(dtype="int64")
    dir_all = trades["direction"].to_numpy()
    net_all = trades["net_ret"].to_numpy(dtype=float) * EXPOSURE
    gross_all = trades["gross_ret"].to_numpy(dtype=float) * EXPOSURE
    equity = initial * np.cumprod(1.0 + net_all)
    equity_prev = np.concatenate(([initial], equity[:-1]))
    contrib_all = equity - equity_prev      # P/L $ de chaque trade dans l'equity globale
    out = []
    for k, (a, b) in enumerate(bounds, 1):
        m = (tr_ts >= a) & (tr_ts < b)
        net = net_all[m]
        gross = gross_all[m]
        # capital courant a la fin de la tranche (equity globale, meme si la
        # tranche est vide : on reprend la derniere equity anterieure)
        before = tr_ts < b
        eq_end = float(equity[before][-1]) if before.any() else float(initial)
        d = {"idx": k, "start": _fmt_ms(a), "end": _fmt_ms(b),
             "days": round((b - a) / DAY_MS), "n": int(m.sum())}
        # perfs par sens de la tranche (memes formules, trades du sens seulement)
        for _s in ("up", "down"):
            _ms = m & (dir_all == _s)
            d[_s] = _agg_side(net_all[_ms], gross_all[_ms], contrib_all[_ms])
        if not len(net):
            d.update(net_pct=0.0, wr=float("nan"), ev_pct=0.0, worst_pct=0.0,
                     dd_pct=0.0, pnl_usd=0.0, avg_win_pct=0.0, avg_loss_pct=0.0,
                     gross_pct=0.0, fee_pts=0.0, avg_gross_pct=0.0,
                     best_pct=0.0, eq_end=eq_end)
            out.append(d)
            continue
        eq = np.cumprod(1.0 + net)
        peak = np.maximum.accumulate(eq)
        wins, losses = net[net > 0], net[net < 0]
        wr = float((net > 0).mean())
        lr = float((net < 0).mean())
        aw = float(wins.mean()) if len(wins) else 0.0
        al = float(-losses.mean()) if len(losses) else 0.0
        d.update(net_pct=float((eq[-1] - 1.0) * 100),
                 wr=wr * 100,
                 avg_win_pct=aw * 100,
                 avg_loss_pct=al * 100,
                 ev_pct=(wr * aw - lr * al) * 100,
                 worst_pct=float(net.min() * 100),
                 dd_pct=float(((eq - peak) / peak).min() * 100),
                 pnl_usd=float(equity[m][-1] - equity_prev[m][0]),
                 # miroir du bloc "Profit / Perte" par tranche (memes formules
                 # que trading_stats, restreintes aux trades de la tranche)
                 gross_pct=float((np.prod(1.0 + gross) - 1.0) * 100),
                 fee_pts=float((gross - net).sum() * 100),
                 avg_gross_pct=float(gross.mean() * 100),
                 best_pct=float(net.max() * 100),
                 eq_end=eq_end)
        out.append(d)
    return out


def _fmt_ms(ts_ms: int) -> str:
    return pd.Timestamp(ts_ms, unit="ms", tz="UTC").strftime("%Y-%m-%d %H:%M UTC")


def append_run(stats: dict) -> None:
    """Append one summary row to logs/trading_runs.csv (append-only history)."""
    flat = {k: v for k, v in stats.items() if not isinstance(v, (dict, list, np.ndarray))}
    new_row = pd.DataFrame([flat])
    if TRADING_RUNS_CSV.exists():
        existing = pd.read_csv(TRADING_RUNS_CSV)
        all_cols = list(existing.columns) + [c for c in new_row.columns if c not in existing.columns]
        existing = existing.reindex(columns=all_cols)
        new_row = new_row.reindex(columns=all_cols)
        combined = pd.concat([existing, new_row], ignore_index=True)
    else:
        combined = new_row
    combined.to_csv(TRADING_RUNS_CSV, index=False)


def _prepare_predictions() -> dict:
    """Partie couteuse et INDEPENDANTE du regime : predictions des 2 modeles +
    tout ce qui ne varie jamais selon le regime (UP_DROP_HOURS, DMI map, body
    returns, prix intrabar). Reutilisable telle quelle par compare_regime_variants.py
    pour eviter de re-predire/recharger a chaque variante de regime testee."""
    start_ms, end_ms = compute_window()

    up_full = api_up.evaluate_full_on_slice(start_ms, end_ms)
    down_full = api_down.evaluate_full_on_slice(start_ms, end_ms)
    # --- UP : drop des heures fragiles (campagne 500%/cote) : pas de long a ces heures UTC ---
    if UP_DROP_HOURS:
        up_full = up_full.copy()
        up_full.loc[up_full["hour"].isin(UP_DROP_HOURS), "take"] = 0

    dmi_map = filters.load_dmi_map() if DMI_FILTER_ENABLED else None
    body_map = _load_body_returns()
    prices_intrabar = _load_intrabar_prices()
    extremes = _load_intrabar_extremes() if MAKER else None

    return {
        "start_ms": start_ms, "end_ms": end_ms,
        "up_full": up_full, "down_full": down_full,
        "dmi_map": dmi_map, "body_map": body_map,
        "prices_intrabar": prices_intrabar, "extremes": extremes,
    }


def run_backtest(prepared: dict, regime_map: dict[int, str] | None = None,
                 fee_override: dict | None = None):
    """Partie legere et regime-CONDITIONNELLE : selectivite DOWN/UP -> coherence ->
    filtre DMI -> filtre de confiance -> simulation -> stats. Reutilise les
    predictions deja calculees dans `prepared` (cf _prepare_predictions).
    regime_map=None (defaut) = comportement EXACTEMENT identique a avant ce refactor
    (seuils scalaires DOWN_CONF_DELTA / UP_CONF_MIN de config_main.py, partout).
    fee_override=None (defaut) = frais statiques config_main.py, inchange ; sinon
    taux reel OKX (cf okx_fees.py). Renvoie (summary, stats, trades, coverage, per_bar)."""
    up_full = prepared["up_full"]
    down_full = prepared["down_full"]

    # MODE DEFENSIF CHOP (2026-08-11) : contexte de tendance longue (ER + age
    # bear + macro), cache module dans filters -> une seule lecture SQLite.
    # Charge ICI (avant la selectivite DOWN) car le bump de delta en chop
    # (campagne DOWN-ANNEE-1, meme jour) en a besoin des l'entree.
    trend_ctx = filters.load_trend_context() if CHOP_SCALP_ENABLED else None

    # --- DOWN : selectivite (campagne 500%/cote), regime-conditionnelle si regime_map fourni,
    #     renforcee en chop (CHOP_DOWN_CONF_DELTA_BUMP) si trend_ctx fourni ---
    eff_delta = filters.effective_down_delta(down_full["timestamp"].to_numpy(), regime_map,
                                             trend_ctx)
    if regime_map is not None or DOWN_CONF_DELTA:
        down_full = down_full.copy()
        down_full["take"] = (down_full["confidence"] >= down_full["threshold"] + eff_delta).astype(int)
    up_bets = _taken_bets(up_full, "up")
    # --- UP : filtre CROSS-MODEL (Strategie 2) : long seulement si P(down) < UP_XMODEL_MAX_PDOWN
    #     (le modele DOWN confirme l'absence de chute). down_full["confidence"] = P(down) par bar. ---
    up_bets = filters.apply_xmodel_filter(up_bets, down_full)
    down_bets = _taken_bets(down_full, "down")
    team_bets, cancelled = build_team_bets(up_bets, down_bets)

    # --- Filtre DMI 1h : annule les paris dont le sens contredit le DMI de la
    #     bougie de decision (+DI vs -DI ; pas de seuil ADX). ---
    team_bets, dmi_removed_up, dmi_removed_down = filters.apply_dmi_filter(team_bets, prepared.get("dmi_map"))

    # --- Filtre de confiance UP : ne garde un pari UP que si P(up) >= seuil effectif
    #     (+ CHOP_UP_CONF_MIN_BUMP sur les barres en chop, campagne UP-PERTES 2026-08-11). ---
    team_bets, conf_removed_up = filters.apply_conf_filter(team_bets, regime_map, trend_ctx)

    # Meta par bar UP (volatility_state, is_weekend) pour le filtre d'entree UP.
    _uf = up_full[up_full["take"] == 1]
    up_meta = {int(r["timestamp"]):
               (int(r["volatility_state"]) if pd.notna(r["volatility_state"]) else -1,
                int(r["is_weekend"]) if pd.notna(r["is_weekend"]) else -1)
               for _, r in _uf.iterrows()}

    # Logs detailles par bougie (inchanges : team_all_bars / team_skipped / team_taken).
    per_bar = detailed_log.build_per_bar_table(up_full, down_full)

    # --- Simulation de trading --- (trend_ctx charge plus haut, avant la selectivite)
    trades, maker_info = simulate_trades(team_bets, prepared["body_map"], prepared["prices_intrabar"],
                                         up_meta, prepared["extremes"], regime_map, fee_override,
                                         trend_ctx)
    coverage = (maker_info["n_would_be"] / len(team_bets)) if len(team_bets) else 0.0

    # (Le gouverneur d'equity qui filtrait ici les trades a ete supprime le
    # 2026-08-14 — decision user, parite prod. Tous les candidats s'executent.)
    stats = trading_stats(trades)
    stats["years"] = yearly_stats(trades, prepared["start_ms"], prepared["end_ms"])
    for _y in stats["years"]:
        _k = f"y{_y['idx']}"
        stats[f"{_k}_net_pct"] = _y["net_pct"]
        stats[f"{_k}_n"] = _y["n"]
        stats[f"{_k}_wr"] = _y["wr"]
        stats[f"{_k}_avg_win_pct"] = _y["avg_win_pct"]
        stats[f"{_k}_avg_loss_pct"] = _y["avg_loss_pct"]
        stats[f"{_k}_ev_pct"] = _y["ev_pct"]
        stats[f"{_k}_dd_pct"] = _y["dd_pct"]
        stats[f"{_k}_pnl_usd"] = _y["pnl_usd"]
        stats[f"{_k}_gross_pct"] = _y["gross_pct"]
        stats[f"{_k}_fee_pts"] = _y["fee_pts"]
        stats[f"{_k}_avg_gross_pct"] = _y["avg_gross_pct"]
        stats[f"{_k}_best_pct"] = _y["best_pct"]
        stats[f"{_k}_eq_end"] = _y["eq_end"]
        for _s, _tag in (("up", "up"), ("down", "dn")):
            stats[f"{_k}_{_tag}_n"] = _y[_s]["n"]
            stats[f"{_k}_{_tag}_net_pct"] = _y[_s]["net_pct"]
            stats[f"{_k}_{_tag}_wr"] = _y[_s]["wr"]
            stats[f"{_k}_{_tag}_ev_pct"] = _y[_s]["ev_pct"]
            stats[f"{_k}_{_tag}_pnl_usd"] = _y[_s]["pnl_usd"]
    # perfs par sens sur la fenetre entiere (lignes TOTAL UP / TOTAL DOWN de la
    # table par tranche) -- memes formules _agg_side que les tranches
    if not trades.empty:
        _dirs = trades["direction"].to_numpy()
        _net = trades["net_ret"].to_numpy(dtype=float) * EXPOSURE
        _gross = trades["gross_ret"].to_numpy(dtype=float) * EXPOSURE
        _eq = INITIAL_CAPITAL * np.cumprod(1.0 + _net)
        _contrib = _eq - np.concatenate(([INITIAL_CAPITAL], _eq[:-1]))
        for _s in ("up", "down"):
            _m = _dirs == _s
            stats[f"total_{_s}"] = _agg_side(_net[_m], _gross[_m], _contrib[_m])
    summary = {
        "run_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "window_days": TEAM_TEST_DAYS,
        "window_start": _fmt_ms(prepared["start_ms"]),
        "window_end": _fmt_ms(prepared["end_ms"]),
        "entry_fee_pct": maker_info["entry_fee"] * 100,
        "exit_fee_pct": maker_info["exit_fee"] * 100,
        "fee_source": "okx_api" if fee_override else "config_fallback",
        "initial_capital": INITIAL_CAPITAL,
        "position_fraction": POSITION_FRACTION,
        "leverage": LEVERAGE,
        "exposure": EXPOSURE,
        "maker": maker_info["maker"],
        "fill_rate": round(maker_info["fill_rate"], 4),
        "n_would_be": maker_info["n_would_be"],
        "n_missed": maker_info["n_missed"],
        "missed_up": maker_info["missed_up"],
        "missed_down": maker_info["missed_down"],
        "team_bets": len(team_bets),
        "cancelled": cancelled,
        "dmi_filter": DMI_FILTER_ENABLED,
        "dmi_removed_up": dmi_removed_up,
        "dmi_removed_down": dmi_removed_down,
        "conf_filter": UP_CONF_FILTER_ENABLED,
        "conf_min": UP_CONF_MIN,
        "conf_removed_up": conf_removed_up,
        "max_concurrent_positions": MAX_CONCURRENT_POSITIONS,
        "max_concurrent_positions_per_side": MAX_CONCURRENT_POSITIONS_PER_SIDE,
        "concurrency_skipped": maker_info["n_concurrency_skipped"],
        "concurrency_skipped_up": maker_info["up_concurrency_skipped"],
        "concurrency_skipped_down": maker_info["down_concurrency_skipped"],
        "widen_on_profit": WIDEN_ON_PROFIT_ENABLED,
        "widened_up": maker_info["n_widened_up"],
        "widened_down": maker_info["n_widened_down"],
        "coverage": round(coverage, 4),
        "regime_wrapper": regime_map is not None,
        **{k: (round(v, 5) if isinstance(v, float) else v)
           for k, v in stats.items() if not isinstance(v, dict)},
    }
    return summary, stats, trades, coverage, per_bar


def main() -> None:
    prepared = _prepare_predictions()
    regime_map = filters.load_regime_map() if cfg_regime.REGIME_WRAPPER_ENABLED else None

    # Frais reels OKX (campagne symetrie 2026-07-09) : jamais silencieux -- si
    # OKX_API_KEY/OKX_API_SECRET/OKX_API_PASSPHRASE sont absentes ou l'appel echoue,
    # on retombe sur les constantes de config_main.py mais on le dit clairement (peut etre perime).
    fee_override = okx_fees.fetch_real_fee_rate()
    if fee_override:
        print(f"  frais OKX reels (API) : maker={fee_override['maker']*100:.4f}%  "
              f"taker={fee_override['taker']*100:.4f}%")
    else:
        print(f"  frais OKX reels indisponibles (OKX_API_KEY/OKX_API_SECRET/OKX_API_PASSPHRASE "
              f"absentes de l'environnement, ou appel API en echec) -> fallback config_main.py : "
              f"maker={MAKER_FEE_PER_SIDE*100:.4f}%  taker={FEE_PER_SIDE*100:.4f}%  "
              f"(PEUT ETRE PERIME, verifie contre https://www.okx.com/fees)")

    summary, stats, trades, coverage, per_bar = run_backtest(prepared, regime_map, fee_override)

    LOGS_DIR.mkdir(exist_ok=True)
    detailed_log.write_detailed_logs(per_bar, LOGS_DIR)
    full_logs.write_full_logs(per_bar, api_up, api_down, LOGS_DIR)
    trades.to_csv(TRADES_CSV, index=False)
    append_run(summary)

    _render_report(summary, stats, coverage)


# ----------------------------------------------------------------------------
#  Rapport Rich
# ----------------------------------------------------------------------------
def _pnl_color(pct: float) -> str:
    if pd.isna(pct):
        return "dim"
    return "bold bright_green" if pct > 0 else "red" if pct < 0 else "yellow"


def _render_report(summary: dict, stats: dict, coverage: float) -> None:
    console = Console()

    header = Text.assemble(
        ("TEAM ", "bold cyan"),
        ("modele_updown", "bold magenta"),
        ("  -  TRADING backtest (P/L net de frais)", "dim"),
    )
    window = Text.assemble(
        ("window  ", "dim"),
        (f"{summary['window_start']}  ->  {summary['window_end']}", "white"),
        (f"   ({summary['window_days']} days)", "dim"),
    )
    round_trip_fee_pct = summary['entry_fee_pct'] + summary['exit_fee_pct']
    fees = Text.assemble(
        ("frais   ", "dim"),
        (f"entree {summary['entry_fee_pct']:.3f}%", "white"),
        (" (maker)" if summary.get("maker") else " (taker)", "dim"),
        (f"  /  sortie {summary['exit_fee_pct']:.3f}% (taker)", "white"),
        (f"  (= {round_trip_fee_pct:.3f}% aller-retour)", "dim"),
        ("    capital initial ", "dim"),
        (f"{summary['initial_capital']:.0f}$", "white"),
    )
    sizing = Text.assemble(
        ("sizing  ", "dim"),
        (f"{summary['position_fraction']*100:.0f}% du capital", "white"),
        (f"  x levier {summary['leverage']:.0f}", "white"),
        (f"  = exposition {summary['exposure']:.1f}x", "bold yellow"),
        (" le mouvement de prix", "dim"),
    )
    # expand=False : les panneaux epousent leur contenu au lieu de s'etirer
    # sur toute la largeur du terminal (demande user 2026-08-14).
    console.print(Panel(Text("\n").join([header, window, fees, sizing]),
                        border_style="cyan", padding=(0, 2), expand=False))

    if stats.get("n_trades", 0) == 0:
        console.print(Panel(Text("Aucun trade sur la fenetre.", style="red"),
                            border_style="red"))
        return

    # --- Bloc trades / WR ---
    t = Table(title="Trades", title_style="bold cyan", border_style="cyan",
              header_style="bold", expand=False)
    t.add_column("Metrique", style="bold")
    t.add_column("Valeur", justify="right")
    t.add_row("Trades pris", f"{stats['n_trades']}")
    t.add_row("Couverture bougie i+1", f"{coverage*100:.1f}%")
    t.add_row("WR directionnel", f"{stats['wr_dir']*100:.2f}%")
    t.add_row("WR net (apres frais)", f"{stats['wr_net']*100:.2f}%")
    t.add_row("Mode d'entree",
              Text("MAKER (limite @ P0)", style="bold magenta") if summary.get("maker")
              else "TAKER (marche @ open)")
    if summary.get("maker"):
        t.add_row("Opportunites perdues (non remplies)",
                  Text(f"{(1-summary['fill_rate'])*100:.1f}%   "
                       f"({summary['n_missed']}/{summary['n_would_be']} ; "
                       f"UP {summary['missed_up']} / DOWN {summary['missed_down']})",
                       style="bold red"))
    t.add_row("Check alignement signe/label",
              "n/a" if pd.isna(stats["align_check"]) else f"{stats['align_check']*100:.1f}%")
    if summary.get("dmi_filter"):
        t.add_row(f"Filtre DMI {DMI_TIMEFRAME} (len {DMI_LENGTH}) - paris annules",
                  Text(f"{summary.get('dmi_removed_up', 0)} UP / "
                       f"{summary.get('dmi_removed_down', 0)} DOWN", style="yellow"))
    if summary.get("conf_filter"):
        t.add_row(f"Filtre confiance UP (p_up >= {UP_CONF_MIN}) - paris annules",
                  Text(f"{summary.get('conf_removed_up', 0)} UP", style="yellow"))
    t.add_row(f"Plafond concurrence (max {summary.get('max_concurrent_positions', MAX_CONCURRENT_POSITIONS)} positions tous sens / "
              f"max {summary.get('max_concurrent_positions_per_side', MAX_CONCURRENT_POSITIONS_PER_SIDE)} par sens) - signaux sautes",
              Text(f"{summary.get('concurrency_skipped', 0)} "
                   f"({summary.get('concurrency_skipped_up', 0)} UP / "
                   f"{summary.get('concurrency_skipped_down', 0)} DOWN)", style="yellow"))
    if summary.get("widen_on_profit"):
        t.add_row(f"Elargissement sur profit (UP x{UP_WIDEN_TP_MULT:.1f}/{UP_WIDEN_TRAIL_GAP_MULT:.1f}/{UP_WIDEN_MAX_HOLD_MULT:.1f} "
                  f"DOWN x{DOWN_WIDEN_TP_MULT:.1f}/{DOWN_WIDEN_TRAIL_GAP_MULT:.1f}/{DOWN_WIDEN_MAX_HOLD_MULT:.1f}) - positions elargies",
                  Text(f"{summary.get('widened_up', 0)} UP / {summary.get('widened_down', 0)} DOWN",
                       style="bold magenta"))
    t.add_row(f"Gestion UP (TP {UP_TAKE_PROFIT*100:.2f}% / SL temporel -{UP_SL_START_PCT:.2f}%->-{UP_SL_END_PCT:.2f}% / trail {UP_TRAIL_GAP*100:.2f}% / max {UP_MAX_STEPS}min)",
              Text(f"SL temporel -{UP_SL_START_PCT:.2f}% -> -{UP_SL_END_PCT:.2f}% (floor -{UP_SL_FLOOR_PCT:.2f}%), % du prix, levier hors-jeu", style="yellow"))
    t.add_row(f"Gestion DOWN (TP {DOWN_TAKE_PROFIT*100:.2f}% / SL temporel -{DOWN_SL_START_PCT:.2f}%->-{DOWN_SL_END_PCT:.2f}% / trail {DOWN_TRAIL_GAP*100:.2f}% / max {DOWN_MAX_STEPS}min)",
              Text(f"SL temporel -{DOWN_SL_START_PCT:.2f}% -> -{DOWN_SL_END_PCT:.2f}% (floor -{DOWN_SL_FLOOR_PCT:.2f}%), % du prix, levier hors-jeu", style="yellow"))
    t.add_row("Sorties gerees (tous sens : TP / SL / trail)",
              f"{stats.get('n_take_profit', 0)} / {stats.get('n_stop_loss', 0)} / {stats.get('n_trailing', 0)}")
    console.print(t)

    # --- Bloc P/L ---
    p = Table(title="Profit / Perte", title_style="bold cyan", border_style="cyan",
              header_style="bold", expand=False)
    p.add_column("Metrique", style="bold")
    p.add_column("Valeur", justify="right")
    p.add_row("Rendement BRUT (hors frais)",
              Text(f"{stats['gross_total_pct']:+.2f}%", style=_pnl_color(stats['gross_total_pct'])))
    p.add_row("Rendement NET (compose)",
              Text(f"{stats['net_total_pct']:+.2f}%", style=_pnl_color(stats['net_total_pct'])))
    p.add_row("Capital final", f"{stats['final_equity']:.2f}")
    p.add_row("P/L", Text(f"{stats['pnl']:+.2f}", style=_pnl_color(stats['pnl'])))
    p.add_row("Frais cumules (drag)", Text(f"-{stats['fee_drag_pct']:.2f} pts", style="red"))
    p.add_row("Moyenne / trade (net)",
              Text(f"{stats['avg_net_per_trade_pct']:+.4f}%",
                   style=_pnl_color(stats['avg_net_per_trade_pct'])))
    p.add_row("Moyenne / trade (brut)", f"{stats['avg_gross_per_trade_pct']:+.4f}%")
    p.add_row("Meilleur / pire trade",
              f"{stats['best_pct']:+.2f}% / {stats['worst_pct']:+.2f}%")
    p.add_row("Max drawdown", Text(f"{stats['max_drawdown_pct']:.2f}%", style="red"))
    console.print(p)

    # --- Bloc mensuel (USDC) ---
    # --- Par tranche de 365 jours (TOUJOURS affichee, demande user 2026-08-14 :
    #     meme sur une fenetre <= 365j on veut la tranche unique + les lignes
    #     UP/DOWN + TOTAL ; avant elle n'apparaissait que si fenetre > 1 an) ---
    years = stats.get("years") or []
    if years:
        # colonnes = les metriques du bloc "Profit / Perte" ventilees par tranche
        # (ajout 2026-08-11 a la demande du user), en plus des colonnes WR/EV.
        ty = Table(title="Par tranche de 365 jours", show_lines=True)
        ty.add_column("Periode")
        ty.add_column("Trades", justify="right")
        ty.add_column("Brut", justify="right")
        ty.add_column("Net compose", justify="right")
        ty.add_column("WR net", justify="right")
        ty.add_column("AvgWin", justify="right")
        ty.add_column("AvgLoss", justify="right")
        ty.add_column("EV/trade", justify="right")
        ty.add_column("Moy brut/tr", justify="right")
        ty.add_column("Meilleur", justify="right")
        ty.add_column("Pire trade", justify="right")
        ty.add_column("Max DD", justify="right")
        ty.add_column("Frais", justify="right")
        ty.add_column("P/L $", justify="right")
        ty.add_column("Capital fin", justify="right")
        def _side_row(label, s):
            # ligne UP ou DOWN (tranche ou total) : memes colonnes, "Capital fin"
            # vide (l'equity est globale, elle ne se decompose pas par sens)
            return [
                Text(label, style="bold blue" if "UP" in label else "bold magenta"),
                str(s["n"]),
                Text(f"{s['gross_pct']:+.2f}%", style=_pnl_color(s["gross_pct"])),
                Text(f"{s['net_pct']:+.2f}%", style=_pnl_color(s["net_pct"])),
                f"{s['wr']:.2f}%" if s["n"] else "n/a",
                f"{s['avg_win_pct']:.3f}%",
                f"{s['avg_loss_pct']:.3f}%",
                Text(f"{s['ev_pct']:+.4f}%", style=_pnl_color(s["ev_pct"])),
                f"{s['avg_gross_pct']:+.4f}%",
                f"{s['best_pct']:+.2f}%",
                f"{s['worst_pct']:+.2f}%",
                f"{s['dd_pct']:+.2f}%",
                Text(f"-{s['fee_pts']:.2f} pts", style="red"),
                Text(f"{s['pnl_usd']:+,.2f}$", style=_pnl_color(s["pnl_usd"])),
                "",
            ]

        for y in years:
            ty.add_row(
                f"annee {y['idx']}  {y['start'][:10]} -> {y['end'][:10]} ({y['days']}j)",
                str(y["n"]),
                Text(f"{y['gross_pct']:+.2f}%", style=_pnl_color(y["gross_pct"])),
                Text(f"{y['net_pct']:+.2f}%", style=_pnl_color(y["net_pct"])),
                f"{y['wr']:.2f}%",
                f"{y['avg_win_pct']:.3f}%",
                f"{y['avg_loss_pct']:.3f}%",
                Text(f"{y['ev_pct']:+.4f}%", style=_pnl_color(y["ev_pct"])),
                f"{y['avg_gross_pct']:+.4f}%",
                f"{y['best_pct']:+.2f}%",
                f"{y['worst_pct']:+.2f}%",
                f"{y['dd_pct']:+.2f}%",
                Text(f"-{y['fee_pts']:.2f} pts", style="red"),
                Text(f"{y['pnl_usd']:+,.2f}$", style=_pnl_color(y["pnl_usd"])),
                f"{y['eq_end']:.2f}",
            )
            ty.add_row(*_side_row(f"  annee {y['idx']} UP", y["up"]))
            ty.add_row(*_side_row(f"  annee {y['idx']} DOWN", y["down"]))
        # ligne TOTAL : memes definitions que le rapport global (stats), pour
        # comparer chaque annee a l'ensemble sans changer de table.
        ty.add_row(
            Text("TOTAL", style="bold"),
            Text(str(stats["n_trades"]), style="bold"),
            Text(f"{stats['gross_total_pct']:+.2f}%",
                 style=f"bold {_pnl_color(stats['gross_total_pct'])}"),
            Text(f"{stats['net_total_pct']:+.2f}%",
                 style=f"bold {_pnl_color(stats['net_total_pct'])}"),
            Text(f"{stats['wr_net'] * 100:.2f}%", style="bold"),
            Text(f"{stats['avg_win_pct_all']:.3f}%", style="bold"),
            Text(f"{stats['avg_loss_pct_all']:.3f}%", style="bold"),
            Text(f"{stats['ev_per_trade_pct']:+.4f}%",
                 style=f"bold {_pnl_color(stats['ev_per_trade_pct'])}"),
            Text(f"{stats['avg_gross_per_trade_pct']:+.4f}%", style="bold"),
            Text(f"{stats['best_pct']:+.2f}%", style="bold"),
            Text(f"{stats['worst_pct']:+.2f}%", style="bold"),
            Text(f"{stats['max_drawdown_pct']:+.2f}%", style="bold"),
            Text(f"-{stats['fee_drag_pct']:.2f} pts", style="bold red"),
            Text(f"{stats['pnl']:+,.2f}$", style=f"bold {_pnl_color(stats['pnl'])}"),
            Text(f"{stats['final_equity']:.2f}", style="bold"),
        )
        if stats.get("total_up") and stats.get("total_down"):
            ty.add_row(*_side_row("  TOTAL UP", stats["total_up"]))
            ty.add_row(*_side_row("  TOTAL DOWN", stats["total_down"]))
        console.print(ty)

    # (Tables 'Par mois (USDC)', 'Par sens' et 'EV par trade' SUPPRIMEES du
    #  rapport 2026-08-11 a la demande du user -- les stats sous-jacentes restent
    #  calculees (trading_runs.csv + g.py) ; seule la table par tranche de 365j
    #  et le bloc P/L subsistent a l'affichage.)


    # --- Headline ---
    net = stats["net_total_pct"]
    headline = Text.assemble(
        ("P/L NET  ", "bold"),
        (f"{net:+.2f}%", f"bold {_pnl_color(net)}"),
        (f"   ({stats['pnl']:+.2f} sur {summary['initial_capital']:.0f})", "dim"),
        ("   sur ", "dim"),
        (f"{stats['n_trades']}", "bold"),
        (" trades, frais ", "dim"),
        (f"{(summary['entry_fee_pct'] + summary['exit_fee_pct']):.2f}% A/R", "dim"),
        ("   EV ", "dim"),
        (f"{stats['ev_per_trade_pct']:+.4f}%/trade", f"bold {_pnl_color(stats['ev_per_trade_pct'])}"),
    )
    console.print(Panel(headline, border_style=_pnl_color(net), padding=(0, 2),
                        expand=False))

    console.print(Text.assemble(("  written  ", "dim"), ("logs/trades.csv", "white"),
                                (f"   ({stats['n_trades']} trades)", "dim")))
    console.print(Text.assemble(("  logged   ", "dim"),
                                (f"{TRADING_RUNS_CSV.relative_to(ROOT)}", "white"),
                                ("   (+1 line)", "dim")))


if __name__ == "__main__":
    main()
