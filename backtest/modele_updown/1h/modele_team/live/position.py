"""position.py -- PositionManager : gestion des positions ouvertes et des
entrees maker en cours, persistance JSON entre cycles.

PARITE AVEC LE BACKTEST (ALIGNEMENT COMPLET 2026-08-14, puis ALIGNEMENT GRILLE
1-MIN 2026-08-23) : `_up_managed_exit` / `_down_managed_exit` de ce module sont
un PORT LIGNE A LIGNE de leurs homonymes dans main.py -- memes seuils
(config_main.py, importe du parent, PAS une copie), meme `filters.sl_threshold`
(SL toujours time-decay), meme cliquet `filters.ratchet_floor`, memes overrides
chop/macro-bull figes a la barre de decision (`_decision_ctx`, depuis le
contexte SMA/ER de market_data), meme palier DOWN chop/mbull, memes
multiplicateurs widen appliques au TP EFFECTIF, et desormais la MEME extension
conditionnelle du TP DOWN (`DOWN_TPEXT_*`, pivots 1-min via
`swing_pivots_from`). `pp` suit la meme formule que le backtest
(`cl/entry - 1` LONG, `1 - cl/entry` SHORT).

LA GRILLE EST LA MEME DES DEUX COTES (2026-08-23). Le backtest deroule les
CLOTURES des sous-bougies 1-min sur t = traded_ts + j*60s ; ce module lit
exactement les memes objets chez OKX (`api_okx.get_candles_range`, cle =
open-time) et rejoue la boucle DEPUIS j=0 a chaque tick. Consequences :
  - plus d'echantillon ponctuel de markPx pris a un instant quelconque de la
    minute, et plus d'index `j` decale d'un cran ;
  - DRY_RUN et mode reel lisent la MEME source (l'ancien ecart 'ticker.last vs
    markPx' n'existe plus) ;
  - un tick manque, un redemarrage ou une minute de retard sont rattrapes
    automatiquement : la marche re-parcourt toutes les bougies depuis l'entree.
`peak_pp`/`worst_pp`/`floor_pp` restent persistes dans
`config_live.POSITIONS_STATE_JSON` mais ne servent plus qu'au REPORTING -- ils
sont recalcules a chaque tick et ne pilotent aucune decision. Ce qui doit
IMPERATIVEMENT survivre a un redemarrage : `avg_px`, `traded_ts_ms`, le
contexte de decision fige (`chop` / `up_tp_override` / `dn_ratchet_extra` /
`regime` / `down_*_override`), `widen_candidates`, et l'ordre maker en cours.

CONTRAINTE HEDGE OKX : une position fusionnee par (instId, posSide) -> 1 SLOT
PAR SENS (long, short) = 2 au total. Le backtest applique EXACTEMENT le meme
cap depuis 2026-07-10 (MAX_CONCURRENT_POSITIONS_PER_SIDE=1) -- plus aucun
ecart entre les deux systemes sur ce point.

ELARGISSEMENT SUR PROFIT (symetrie main.py, campagne 2026-07-10, re-aligne
2026-08-23) : quand un signal du meme sens arrive alors qu'un slot est deja
occupe, live.py appelle `register_widen_candidate` qui MEMORISE son traded_ts
dans `entry["widen_candidates"]` -- exactement le `widen_candidates` que
main.py construit avec tous les futurs signaux du meme sens. C'est ensuite LA
MARCHE de sortie qui elargit, au premier candidat atteint EN PROFIT (`pp > 0.0
and t in widen_candidates`), UNE SEULE FOIS, sans jamais toucher le SL ni son
horizon. L'ancienne version tranchait au moment du signal, sur le prix de
l'instant, et perdait definitivement un candidat dont le pp etait <= 0 a cette
seconde-la.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("live.position")

_LIVE_DIR = Path(__file__).resolve().parent
_TEAM_DIR = _LIVE_DIR.parent
if str(_TEAM_DIR) not in sys.path:
    sys.path.insert(0, str(_TEAM_DIR))
if str(_LIVE_DIR) not in sys.path:
    sys.path.insert(0, str(_LIVE_DIR))

import filters  # noqa: E402
# ALIGNEMENT LIVE/BACKTEST 2026-08-14 : les imports UP_STOP_LOSS_DEC /
# UP_SL_TIME_DECAY / DOWN_STOP_LOSS_DEC / DOWN_SL_TIME_DECAY ont ete RETIRES
# (supprimes de config_main.py le 2026-08-11 — ils faisaient crasher le daemon
# a l'import). Le SL est TOUJOURS time-decay, comme au backtest.
from config_main import (  # noqa: E402
    UP_TAKE_PROFIT, UP_TRAIL_ACTIVATE, UP_TRAIL_GAP,
    UP_SL_START_DEC, UP_SL_END_DEC, UP_SL_FLOOR_DEC,
    UP_STEP_MS, UP_MAX_STEPS,
    DOWN_TAKE_PROFIT, DOWN_TRAIL_ACTIVATE, DOWN_TRAIL_GAP,
    DOWN_SL_START_DEC, DOWN_SL_END_DEC, DOWN_SL_FLOOR_DEC,
    DOWN_STEP_MS, DOWN_MAX_STEPS,
    DOWN_TPEXT_ENABLED, DOWN_TPEXT_REGIMES, DOWN_TPEXT_MAX_ER,
    DOWN_TPEXT_SWING_P, DOWN_TPEXT_HARD_GUARD_DEC,
    MAKER, MAKER_FILL_WINDOW_MIN,
    WIDEN_ON_PROFIT_ENABLED, DOWN_WIDEN_ENABLED,
    UP_WIDEN_TP_MULT, UP_WIDEN_TRAIL_GAP_MULT, UP_WIDEN_MAX_HOLD_MULT,
    DOWN_WIDEN_TP_MULT, DOWN_WIDEN_TRAIL_GAP_MULT, DOWN_WIDEN_MAX_HOLD_MULT,
    UP_RATCHET_STEPS, UP_RATCHET_PROP, DOWN_RATCHET_STEPS, DOWN_RATCHET_PROP,
    RATCHET_SCALES_WITH_WIDEN,
    CHOP_UP_SL_START_DEC, CHOP_UP_SL_FLOOR_DEC, CHOP_DOWN_ENABLED,
    CHOP_DOWN_TP, CHOP_DOWN_MAX_STEPS, CHOP_DOWN_TRAIL_GAP, CHOP_DOWN_TRAIL_ACTIVATE,
)
# NB : ENTRY_REPRICE_EVERY_S n'est plus importe (repricing maker SUPPRIME le
# 2026-08-14 -- la limite reste statique au prix de depart, semantique backtest).
from config_live import (  # noqa: E402
    POSITIONS_STATE_JSON, OKX_INST_ID,
    NATIVE_BACKSTOP_ENABLED, CATASTROPHE_SL_UP_PCT, CATASTROPHE_SL_DOWN_PCT,
    BACKSTOP_TRIGGER_PX_TYPE, DESYNC_CONFIRM_TICKS,
)


# ----------------------------------------------------------------------------
#  Persistance de l'etat (long/short/pending_entry)
# ----------------------------------------------------------------------------
def load_state() -> dict:
    if POSITIONS_STATE_JSON.exists():
        try:
            return json.loads(POSITIONS_STATE_JSON.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            logger.error(f"positions_state.json illisible ({exc}) -- repart d'un etat vide.")
    return {"long": None, "short": None, "pending_entry": None}


def save_state(state: dict) -> None:
    # Ecriture ATOMIQUE (tmp + os.replace, alignement 2026-08-14) : ce fichier
    # est ecrit a CHAQUE tick ; un crash/kill mi-ecriture corrompait le JSON et
    # load_state repartait d'un etat vide (peak/floor/widen/overrides perdus,
    # trailing rearme a zero apres reconcile).
    POSITIONS_STATE_JSON.parent.mkdir(parents=True, exist_ok=True)
    tmp = POSITIONS_STATE_JSON.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2))
    os.replace(tmp, POSITIONS_STATE_JSON)


# ----------------------------------------------------------------------------
#  Moteur de sortie -- PORT LIGNE A LIGNE de main.py::_up_managed_exit /
#  _down_managed_exit (campagne ALIGNEMENT GRILLE 1-MIN, 2026-08-23)
# ----------------------------------------------------------------------------
# Le backtest ne regarde JAMAIS un prix instantane : il deroule les CLOTURES des
# sous-bougies 1-min (DB_1m/btc_1m.parquet, cle = open-time) sur la grille
#     t = traded_ts + j * STEP_MS,  j = 0 .. max_steps-1
# Le live lisait jusqu'ici un echantillon ponctuel de markPx une fois par minute :
# meme CADENCE apparente, mais (a) un prix different (instant quelconque de la
# minute, jamais la cloture), (b) un index `j` decale d'un cran (elapsed_min=1 au
# premier tick alors que le backtest y est encore a j=0), et (c) aucun rattrapage
# si un tick etait manque. On rejoue desormais EXACTEMENT la meme boucle sur les
# MEMES objets : les bougies 1-min FERMEES d'OKX, indexees par leur open-time
# (api_okx.get_candles_range) -- meme grille, meme j, memes prix.
#
# La boucle est REJOUEE INTEGRALEMENT depuis j=0 a chaque tick : peak / floor /
# path_len / extension TP se reconstruisent de facon deterministe a partir des
# seules bougies, donc un redemarrage, un tick manque ou une minute de retard ne
# peuvent plus faire deriver l'etat (l'ancien suivi incremental persistait un pic
# qui n'etait plus recalculable). Cout : quelques milliers d'iterations triviales
# par tick. peak_pp / worst_pp / floor_pp restent persistes, mais pour le SEUL
# reporting -- ils ne pilotent plus aucune decision.
#
# Sortie de boucle : `cl is None` signifie ici "la bougie n'existe pas ENCORE"
# (le futur n'est pas arrive), la ou le backtest ne rencontre ce cas qu'a la fin
# de son parquet. On renvoie donc reason=None (position toujours ouverte) tant
# que j < max_steps -- le `timestop` n'est emis QUE quand la boucle a reellement
# consomme ses max_steps pas, exactement comme au backtest.
def _widened_ratchet(steps: tuple, prop, tp_mult: float) -> tuple:
    """Paliers de cliquet apres elargissement -- MEME regle que main.py::
    _widened_ratchet : RATCHET_SCALES_WITH_WIDEN=False (defaut) = paliers
    ABSOLUS inchanges ; True = seuils multiplies par le facteur TP."""
    if not RATCHET_SCALES_WITH_WIDEN:
        return steps, prop
    steps = tuple((p * tp_mult, f * tp_mult) for p, f in steps)
    prop = None if prop is None else (prop[0] * tp_mult, prop[1])
    return steps, prop


def swing_pivots_from(ts: list, hi: list, p: int) -> dict:
    """Pivots HAUTS confirmes du 1-min -- port de main.py::_load_swing_pivots
    (meme definition, meme absence de lookahead : le pivot d'indice b n'est
    CONNU qu'a l'indice b+p). Renvoie {ts_ms: (niveau_px, ts_naissance_ms)}.

    left[i]  = max(hi[i-p .. i-1])   (p bougies AVANT, i exclu)
    right[i] = max(hi[i+1 .. i+p])   (p bougies APRES)
    pivot(i) <=> hi[i] > left[i] ET hi[i] >= right[i]

    Bords : le backtest calcule ces deux bornes avec des `rolling(p)` pandas
    (`min_periods=p` par defaut), qui rendent NaN des que la fenetre est
    incomplete -- puis `nan_to_num(..., nan=-1.0)`, et une comparaison a -1 est
    toujours vraie. On reproduit ce -1 A L'IDENTIQUE plutot qu'un max partiel :
      left[b]  valide ssi b >= p                  (rolling(p).max().shift(1))
      right[b] valide ssi p-1 <= b <= n-p-1       (shift(-p).rolling(p).max())
    Sans ca les premieres barres du buffer live seraient classees autrement que
    dans le backtest (verifie : 68 840 barres, 0 ecart)."""
    n = len(hi)
    out: dict[int, tuple] = {}
    cur_lvl, cur_born = float("nan"), -1
    for i in range(n):
        b = i - p
        if b >= 0:
            left = max(hi[b - p:b]) if b >= p else -1.0
            right = max(hi[b + 1:b + p + 1]) if (p - 1 <= b <= n - p - 1) else -1.0
            if hi[b] > left and hi[b] >= right:
                cur_lvl, cur_born = hi[b], int(ts[b])
        out[int(ts[i])] = (cur_lvl, cur_born)
    return out


def _res(reason, pp, peak, worst, floor, step_ts, widened_at_ts) -> dict:
    """Resultat d'une marche de sortie. `pp` est None tant qu'aucune bougie 1-min
    n'a encore ete fermee depuis l'entree (position tout juste ouverte)."""
    return {"reason": reason, "pp": pp,
            "peak": None if peak <= -8.0 else peak,
            "worst": None if worst >= 8.0 else worst,
            "floor": floor, "step_ts": step_ts, "widened_at_ts": widened_at_ts}


def _up_managed_exit(close_by: dict, traded_ts: int, entry_px: float,
                     widen_candidates: set | None = None,
                     take_profit: float | None = None,
                     sl_override: tuple[float, float] | None = None) -> dict:
    """Port EXACT de main.py::_up_managed_exit (long). Seules differences : la
    source des `cl` (bougies OKX 1-min fermees au lieu du parquet DB_1m) et le
    fait qu'une bougie absente = futur non encore arrive (reason=None)."""
    peak = -9.0
    worst = 9.0
    floor = None
    last = None
    last_j = 0
    base_tp = take_profit if take_profit is not None else UP_TAKE_PROFIT
    take_profit, trail_gap, max_steps = base_tp, UP_TRAIL_GAP, UP_MAX_STEPS
    ratchet, ratchet_p = UP_RATCHET_STEPS, UP_RATCHET_PROP
    sl_start, sl_floor = sl_override if sl_override is not None \
        else (UP_SL_START_DEC, UP_SL_FLOOR_DEC)
    widened_at_ts = None
    j = 0
    while j < max_steps:
        t = traded_ts + j * UP_STEP_MS
        cl = close_by.get(t)
        if cl is None:
            break                            # bougie pas encore fermee -> on s'arrete la
        pp = cl / entry_px - 1.0
        last = pp
        last_j = j
        if pp < worst:
            worst = pp
        if (WIDEN_ON_PROFIT_ENABLED and widened_at_ts is None and widen_candidates
                and pp > 0.0 and t in widen_candidates):
            take_profit = base_tp * UP_WIDEN_TP_MULT
            trail_gap = UP_TRAIL_GAP * UP_WIDEN_TRAIL_GAP_MULT
            max_steps = int(UP_MAX_STEPS * UP_WIDEN_MAX_HOLD_MULT)
            ratchet, ratchet_p = _widened_ratchet(ratchet, ratchet_p, UP_WIDEN_TP_MULT)
            widened_at_ts = t
        sl_j = min(j, UP_MAX_STEPS - 1)      # SL jamais elargi -- horizon ORIGINAL toujours
        sl_th = filters.sl_threshold(sl_j, UP_MAX_STEPS, sl_start, UP_SL_END_DEC, sl_floor)
        if pp <= sl_th:
            return _res("stop_loss", pp, peak, worst, floor, t, widened_at_ts)
        if pp >= take_profit:
            return _res("take_profit", pp, peak, worst, floor, t, widened_at_ts)
        if pp > peak:
            peak = pp
        if peak >= UP_TRAIL_ACTIVATE:
            floor = peak - trail_gap
        rf = filters.ratchet_floor(peak, ratchet, ratchet_p)
        if rf is not None and (floor is None or rf > floor):
            floor = rf
        if floor is not None and pp < floor:
            return _res("trailing", pp, peak, worst, floor, t, widened_at_ts)
        j += 1
    reason = "timestop" if (j >= max_steps and last is not None) else None
    return _res(reason, last, peak, worst, floor,
                traded_ts + last_j * UP_STEP_MS, widened_at_ts)


def _down_managed_exit(close_by: dict, traded_ts: int, entry_px: float,
                       max_steps=None, trail_gap=None,
                       widen_candidates: set | None = None, regime: str = "neutral",
                       swing_pivots: dict | None = None,
                       take_profit: float | None = None,
                       trail_activate: float | None = None,
                       ratchet_extra: tuple | None = None) -> dict:
    """Port EXACT de main.py::_down_managed_exit (short), extension conditionnelle
    du TP (DOWN_TPEXT_*) COMPRISE -- meme gate ER 'grinder', meme sortie structure
    sur cassure du dernier pivot haut confirme NE PENDANT l'extension, meme
    plancher dur TP - HARD_GUARD. (DOWN_TPEXT_ENABLED=False cote backtest a ce
    jour : le code est present des deux cotes pour que basculer le flag ne
    desynchronise plus rien.)"""
    sl_max_steps = max_steps if max_steps is not None else DOWN_MAX_STEPS
    if max_steps is None:
        max_steps = DOWN_MAX_STEPS
    if trail_gap is None:
        trail_gap = DOWN_TRAIL_GAP
    base_tp = take_profit if take_profit is not None else DOWN_TAKE_PROFIT
    take_profit = base_tp
    trail_act = trail_activate if trail_activate is not None else DOWN_TRAIL_ACTIVATE
    ratchet = DOWN_RATCHET_STEPS if ratchet_extra is None \
        else tuple(sorted(DOWN_RATCHET_STEPS + (ratchet_extra,)))
    ratchet_p = DOWN_RATCHET_PROP
    tpext_armed = (DOWN_TPEXT_ENABLED and regime in DOWN_TPEXT_REGIMES
                   and swing_pivots is not None)
    extended = False
    ext_t = None
    path_len = 0.0
    prev_pp = 0.0
    peak = -9.0
    worst = 9.0
    floor = None
    last = None
    last_j = 0
    widened_at_ts = None
    j = 0
    while j < max_steps:
        t = traded_ts + j * DOWN_STEP_MS
        cl = close_by.get(t)
        if cl is None:
            break                            # bougie pas encore fermee -> on s'arrete la
        pp = 1.0 - cl / entry_px             # profit directionnel d'un short
        if j > 0:
            path_len += abs(pp - prev_pp)
        prev_pp = pp
        last = pp
        last_j = j
        if pp < worst:
            worst = pp
        if (WIDEN_ON_PROFIT_ENABLED and widened_at_ts is None and widen_candidates
                and pp > 0.0 and t in widen_candidates):
            take_profit = base_tp * DOWN_WIDEN_TP_MULT
            trail_gap = trail_gap * DOWN_WIDEN_TRAIL_GAP_MULT
            max_steps = int(max_steps * DOWN_WIDEN_MAX_HOLD_MULT)
            ratchet, ratchet_p = _widened_ratchet(ratchet, ratchet_p, DOWN_WIDEN_TP_MULT)
            widened_at_ts = t
            extended = False                 # le widen (TP x4) prime sur l'extension
        sl_j = min(j, sl_max_steps - 1)      # SL jamais elargi -- horizon EFFECTIF (non-widened)
        sl_th = filters.sl_threshold(sl_j, sl_max_steps, DOWN_SL_START_DEC,
                                     DOWN_SL_END_DEC, DOWN_SL_FLOOR_DEC)
        if pp <= sl_th:
            return _res("stop_loss", pp, peak, worst, floor, t, widened_at_ts)
        # --- Armement de l'extension TP (bear + grinder) au franchissement du TP ---
        if (not extended and tpext_armed and widened_at_ts is None
                and pp >= take_profit):
            er = pp / max(path_len, 1e-9)
            if er <= DOWN_TPEXT_MAX_ER:
                extended = True
                ext_t = t                    # les pivots nes avant ne comptent pas
            else:
                return _res("take_profit", pp, peak, worst, floor, t, widened_at_ts)
        if not extended and pp >= take_profit:
            return _res("take_profit", pp, peak, worst, floor, t, widened_at_ts)
        if pp > peak:
            peak = pp
        if peak >= trail_act:
            floor = peak - trail_gap
        rf = filters.ratchet_floor(peak, ratchet, ratchet_p)
        if rf is not None and (floor is None or rf > floor):
            floor = rf
        if extended:
            # Sortie STRUCTURE : cassure du dernier pivot haut confirme forme
            # PENDANT l'extension ; plancher dur TP - HARD_GUARD sinon.
            hard = base_tp - DOWN_TPEXT_HARD_GUARD_DEC
            sfloor = hard
            piv = swing_pivots.get(t)
            if piv is not None and piv[1] >= ext_t:
                lvl_pp = 1.0 - piv[0] / entry_px
                if lvl_pp > sfloor:
                    sfloor = lvl_pp
            if floor is None or sfloor > floor:
                floor = sfloor
        if floor is not None and pp < floor:
            return _res("trailing", pp, peak, worst, floor, t, widened_at_ts)
        j += 1
    reason = "timestop" if (j >= max_steps and last is not None) else None
    return _res(reason, last, peak, worst, floor,
                traded_ts + last_j * DOWN_STEP_MS, widened_at_ts)


def evaluate_exit(pos_side: str, entry: dict, close_by: dict,
                  swing_pivots: dict | None = None) -> dict:
    """Rejoue la marche de sortie du backtest pour la position `entry` sur les
    bougies 1-min fermees `close_by` ({open_time_ms: close}). Mute `entry` en
    place (peak_pp / worst_pp / floor_pp / widened -- REPORTING uniquement, plus
    aucune decision n'en depend). Renvoie le dict de `_res`.

    Tous les parametres de sortie sont ceux FIGES A LA BARRE DE DECISION
    (`_decision_ctx` + `_finalize_entry`), exactement comme main.py::
    simulate_trades les choisit une fois par trade sans jamais les reviser."""
    avg_px = float(entry["avg_px"])
    traded_ts = int(entry["traded_ts_ms"])
    chop = bool(entry.get("chop", False))
    cands = entry.get("widen_candidates") or ()
    widen_cands = {int(t) for t in cands} if WIDEN_ON_PROFIT_ENABLED else None

    if pos_side == "long":
        tp_ov = entry.get("up_tp_override")
        res = _up_managed_exit(
            close_by, traded_ts, avg_px, widen_cands,
            take_profit=float(tp_ov) if tp_ov is not None else None,
            sl_override=(CHOP_UP_SL_START_DEC, CHOP_UP_SL_FLOOR_DEC) if chop else None)
    else:
        # Gates per-side : main.py::simulate_trades coupe le widen DOWN via
        # DOWN_WIDEN_ENABLED et les sorties scalp DOWN via CHOP_DOWN_ENABLED
        # (ce dernier deja applique dans _decision_ctx -> entry["chop"]).
        if not DOWN_WIDEN_ENABLED:
            widen_cands = None
        if chop:
            max_steps_eff, trail_gap_eff = CHOP_DOWN_MAX_STEPS, CHOP_DOWN_TRAIL_GAP
            regime, dn_tp, dn_act = "neutral", CHOP_DOWN_TP, CHOP_DOWN_TRAIL_ACTIVATE
        else:
            max_steps_eff = entry.get("down_max_steps_override")
            trail_gap_eff = entry.get("down_trail_gap_override")
            regime = str(entry.get("regime") or "neutral")
            dn_tp = dn_act = None
        rx = entry.get("dn_ratchet_extra")
        res = _down_managed_exit(
            close_by, traded_ts, avg_px, max_steps_eff, trail_gap_eff, widen_cands,
            regime, swing_pivots, take_profit=dn_tp, trail_activate=dn_act,
            ratchet_extra=(float(rx[0]), float(rx[1])) if rx is not None else None)

    # REPORTING SEUL : main.py met a jour `peak` APRES les tests SL/TP (donc le
    # pic rendu exclut la bougie de sortie -- fidele, mais on afficherait un pic
    # inferieur au pp d'une sortie en take-profit). On reporte le max ici, ce
    # qui ne touche AUCUNE decision (celles-ci comparent toujours `pp` brut aux
    # seuils, et `floor` est calcule dans la marche avec le vrai `peak`).
    peak = res["peak"]
    if res["pp"] is not None and (peak is None or res["pp"] > peak):
        peak = res["pp"]
    if peak is not None:
        entry["peak_pp"] = peak
    if res["worst"] is not None:
        entry["worst_pp"] = res["worst"]
    entry["floor_pp"] = res["floor"]
    entry["widened"] = res["widened_at_ts"] is not None
    return res


# ----------------------------------------------------------------------------
#  PositionManager
# ----------------------------------------------------------------------------
class PositionManager:
    def __init__(self, client, inst_id: str = OKX_INST_ID):
        self.client = client
        self.inst_id = inst_id
        self.state = load_state()
        self._instrument: dict | None = None
        # Buffer de bougies 1-min FERMEES (open_time_ms -> [ts,o,h,l,c,vol,confirm]),
        # source unique du moteur de sortie (cf ALIGNEMENT GRILLE 1-MIN en tete de
        # module). Maintenu incrementalement par _refresh_candles ; jamais persiste
        # (un redemarrage le refetche, et la marche etant rejouee depuis j=0 l'etat
        # se reconstruit a l'identique).
        self._candles: dict[int, list] = {}
        self._pivots_cache: tuple[int, dict] | None = None
        self._sanitize_state_for_mode()

    # -- Purge de l'etat papier au passage en reel (point #2 audit 2026-08-03) --
    def _sanitize_state_for_mode(self) -> None:
        """En mode REEL, purge tout etat cree sous la semantique DRY_RUN (ordId/
        algoId prefixe 'DRYRUN-'). Sans ca, un `pending_entry` papier residuel
        dans positions_state.json au moment du passage DRY_RUN=false
        declencherait un ordre reel involontaire (cf get_order/point #2), et une
        position papier porterait un backstop_algo_id fictif. Les positions
        papier ouvertes, elles, sont nettoyees ensuite par reconcile_positions
        (aucune position OKX correspondante)."""
        if self.client.dry_run:
            return
        pending = self.state.get("pending_entry")
        if pending and str(pending.get("ord_id", "") or "").startswith("DRYRUN-"):
            logger.warning("Etat REEL : pending_entry papier (ordId DRYRUN-) au chargement "
                           "-> purge (aucun ordre reel ne lui correspond).")
            self.state["pending_entry"] = None
        for pos_side in ("long", "short"):
            entry = self.state.get(pos_side)
            if entry and str(entry.get("backstop_algo_id", "") or "").startswith("DRYRUN-"):
                entry["backstop_algo_id"] = None   # re-pose/reconcilie par reconcile_positions
        save_state(self.state)

    # -- Helpers instrument (tickSz pour arrondir le prix du filet) ----------
    def _load_instrument(self) -> dict:
        if self._instrument is None:
            self._instrument = self.client.get_instrument(self.inst_id)
        return self._instrument

    def _round_px(self, px: float) -> str:
        """Arrondit un prix a l'increment tickSz de l'instrument (OKX rejette un
        px non aligne). Repli robuste sur 1 decimale si l'instrument est
        illisible -- le filet ne doit jamais bloquer une entree."""
        try:
            tick_str = str(self._load_instrument().get("tickSz", "0.1"))
            tick = float(tick_str)
            if tick <= 0:
                raise ValueError
            decimals = len(tick_str.split(".")[1]) if "." in tick_str else 0
            return f"{round(px / tick) * tick:.{decimals}f}"
        except Exception:  # noqa: BLE001
            return f"{px:.1f}"

    def _fmt_sz(self, sz: float) -> str:
        """Taille en decimal fixe (jamais de notation scientifique), meme
        convention que live.compute_size_contracts."""
        return f"{sz:.8f}".rstrip("0").rstrip(".")

    # -- Filet catastrophe natif OKX (point #1 audit 2026-08-03) -------------
    def _place_backstop(self, pos_side: str, avg_px: float, sz: str) -> tuple[str | None, float | None]:
        """Pose le stop STATIQUE de securite sur OKX (order-algo conditionnel).
        Renvoie (algo_id, trigger_px) ou (None, None) si desactive/echec.
        JAMAIS bloquant : si la pose echoue, la position reste ouverte et le SL
        DYNAMIQUE du daemon reste actif -- on alerte seulement (logger.error ->
        Telegram) car on perd le filet 'daemon mort'."""
        if not NATIVE_BACKSTOP_ENABLED:
            return None, None
        try:
            pct = CATASTROPHE_SL_UP_PCT if pos_side == "long" else CATASTROPHE_SL_DOWN_PCT
            raw = avg_px * (1.0 - pct) if pos_side == "long" else avg_px * (1.0 + pct)
            trigger_px = self._round_px(raw)
            res = self.client.place_algo_order(self.inst_id, pos_side, sz,
                                               sl_trigger_px=trigger_px,
                                               trigger_px_type=BACKSTOP_TRIGGER_PX_TYPE)
            algo_id = res.get("algoId")
            logger.info(f"Filet catastrophe {pos_side} pose : trigger={trigger_px} "
                        f"(-{pct:.1%}) sz={sz} algoId={algo_id}")
            return algo_id, float(trigger_px)
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Echec pose filet catastrophe {pos_side} : {exc} -- position ouverte "
                         f"SANS filet natif (SL dynamique du daemon toujours actif).")
            return None, None

    def _cancel_backstop(self, entry: dict | None) -> None:
        """Annule le filet natif d'une position (a sa fermeture normale, ou lors
        d'une desync/reconciliation). Best-effort : un algo deja declenche/
        annule renvoie une erreur, sans consequence."""
        algo_id = entry.get("backstop_algo_id") if entry else None
        if not algo_id:
            return
        try:
            self.client.cancel_algos(self.inst_id, algo_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Annulation filet {algo_id} echouee (deja declenche/annule ?) : {exc}")
        entry["backstop_algo_id"] = None

    # -- Monitoring des positions ouvertes (appele chaque minute) ------------
    # -- Buffer de bougies 1-min : la GRILLE DE REFERENCE du moteur de sortie --
    # (cf ALIGNEMENT GRILLE 1-MIN en tete de module). Le backtest lit
    # DB_1m/btc_1m.parquet ; le live lit les memes objets chez OKX
    # (api_okx.get_candles_range), meme cle (open-time), meme champ (close).
    def _oldest_needed_ts(self) -> int | None:
        """Plus ancien open-time 1-min necessaire pour rejouer les marches en
        cours : le traded_ts de la plus vieille position ouverte -- recule de
        2p bougies cote short si l'extension TP DOWN est active (un pivot
        confirme exige p bougies AVANT lui et p apres, cf swing_pivots_from)."""
        need = []
        for pos_side in ("long", "short"):
            e = self.state.get(pos_side)
            if e is None:
                continue
            t = int(e["traded_ts_ms"])
            if pos_side == "short" and DOWN_TPEXT_ENABLED:
                t -= 2 * DOWN_TPEXT_SWING_P * DOWN_STEP_MS
            need.append(t)
        return min(need) if need else None

    def _refresh_candles(self, now_ms: int) -> None:
        """Maintient self._candles = bougies 1-min FERMEES couvrant
        [_oldest_needed_ts(), now]. Fetch complet la premiere fois (ou apres une
        longue absence / un besoin d'historique plus ancien), incremental
        ensuite (une requete publique par tick)."""
        need_from = self._oldest_needed_ts()
        if need_from is None:
            self._candles = {}
            self._pivots_cache = None
            return
        have = self._candles
        for t in [t for t in have if t < need_from]:
            del have[t]                       # purge de ce qui ne sert plus
        full = (not have) or min(have) > need_from \
            or (now_ms - max(have)) > 240 * 60_000
        if full:
            rows = self.client.get_candles_range(self.inst_id, need_from, now_ms, bar="1m")
            self._candles = {int(r[0]): r for r in rows}
        else:
            n = int((now_ms - max(have)) // 60_000) + 2
            rows = self.client.get_candles(self.inst_id, bar="1m", limit=max(2, min(n, 300)))
            for r in rows:
                if str(r[-1]) == "1" and int(r[0]) >= need_from:
                    self._candles[int(r[0])] = r
        self._heal_candle_gaps(need_from, now_ms)
        self._pivots_cache = None

    def _heal_candle_gaps(self, need_from: int, now_ms: int) -> None:
        """Un TROU interne dans le buffer arreterait la marche de sortie net (le
        backtest ne rencontre ce cas qu'a la fin de son parquet). On re-fetche
        la plage une fois ; si OKX n'a vraiment rien pour cette minute (aucune
        transaction -- quasi impossible sur BTC-USDT-SWAP), on reporte la
        cloture precedente : un prix inchange est la seule lecture defendable,
        et la marche continue au lieu de se figer."""
        last_closed = ((now_ms // 60_000) * 60_000) - 60_000
        if last_closed <= need_from or not self._candles:
            return
        end = min(last_closed, max(self._candles))
        missing = [t for t in range(need_from, end + 1, 60_000) if t not in self._candles]
        if not missing:
            return
        logger.warning(f"Buffer 1-min : {len(missing)} bougie(s) manquante(s) entre "
                       f"{missing[0]} et {missing[-1]} -- re-fetch de la plage.")
        try:
            for r in self.client.get_candles_range(self.inst_id, missing[0],
                                                   missing[-1] + 60_000, bar="1m"):
                self._candles[int(r[0])] = r
        except Exception as exc:  # noqa: BLE001 -- le report ci-dessous prend le relais
            logger.warning(f"re-fetch des bougies manquantes a echoue : {exc}")
        still = [t for t in missing if t not in self._candles]
        if not still:
            return
        logger.error(f"{len(still)} bougie(s) 1-min introuvables chez OKX "
                     f"({still[0]}..{still[-1]}) -- cloture precedente reportee.")
        for t in still:                        # ascendant : un trou de plusieurs
            prev = self._candles.get(t - 60_000)   # minutes se comble de proche en proche
            if prev is not None:
                self._candles[t] = [t, prev[4], prev[4], prev[4], prev[4], 0.0, "1"]
        # Cas du trou EN TETE (aucune bougie anterieure a reporter -- typiquement
        # la bougie de traded_ts elle-meme) : sans ca la marche resterait bloquee
        # a j=0 et la position ne serait JAMAIS evaluee. On remonte alors depuis
        # la premiere bougie connue APRES le trou, en reportant son open.
        for t in sorted((t for t in still if t not in self._candles), reverse=True):
            nxt = self._candles.get(t + 60_000)
            if nxt is not None:
                o = float(nxt[1])
                self._candles[t] = [t, o, o, o, o, 0.0, "1"]

    def _close_by(self) -> dict[int, float]:
        """{open_time_ms: close} -- l'exact equivalent du `close_ib` que
        main.py passe a _up_managed_exit / _down_managed_exit."""
        return {t: float(r[4]) for t, r in self._candles.items()}

    def _swing_pivots(self) -> dict | None:
        """Pivots hauts 1-min pour l'extension TP DOWN, calcules sur le buffer
        courant (None si DOWN_TPEXT_ENABLED=False, comme au backtest)."""
        if not DOWN_TPEXT_ENABLED or not self._candles:
            return None
        key = max(self._candles)
        if self._pivots_cache is not None and self._pivots_cache[0] == key:
            return self._pivots_cache[1]
        ts = sorted(self._candles)
        piv = swing_pivots_from(ts, [float(self._candles[t][2]) for t in ts],
                                DOWN_TPEXT_SWING_P)
        self._pivots_cache = (key, piv)
        return piv

    def check_positions(self, now_ms: int) -> list[dict]:
        """Rejoue, pour chaque position ouverte, la marche de sortie du backtest
        sur les bougies 1-min fermees (SL temporel / TP / trailing+cliquet /
        timestop, extension TP DOWN comprise) et ferme au marche (taker -- une
        sortie declenchee exige une execution immediate) des qu'un seuil est
        franchi. Renvoie la liste des evenements du tick.

        ALIGNEMENT GRILLE 1-MIN (2026-08-23) : le prix de reference n'est plus
        un echantillon de markPx pris a un instant quelconque de la minute, mais
        la CLOTURE de la bougie 1-min d'open-time traded_ts + j*60s -- le meme
        objet, au meme index j, que main.py. Le DRY_RUN et le mode reel lisent
        desormais la MEME source (l'ecart 'ticker.last vs markPx' documente
        jusqu'ici disparait).

        BUG FIXE (2026-07-15) : en DRY_RUN, `place_order` ne poste JAMAIS de
        vrai ordre (stubbe, cf api_okx.py) -- OKX ne peut donc physiquement
        JAMAIS montrer la position papier dans `get_positions()`. En DRY_RUN
        l'etat local EST la verite : on ne consulte pas `get_positions()`."""
        events = []
        if self.state.get("long") is None and self.state.get("short") is None:
            self._candles = {}
            return events

        try:
            self._refresh_candles(now_ms)
        except Exception as exc:  # noqa: BLE001 -- jamais bloquant : on retente au tick suivant
            logger.warning(f"Rafraichissement des bougies 1-min echoue : {exc} -- "
                           f"tick de surveillance saute (buffer conserve).", exc_info=True)
            return events
        close_by = self._close_by()
        pivots = self._swing_pivots()

        live_positions = {}
        if not self.client.dry_run:
            live_positions = {p["posSide"]: p for p in self.client.get_positions(self.inst_id)}

        for pos_side in ("long", "short"):
            entry = self.state.get(pos_side)
            if entry is None:
                continue
            live_pos = live_positions.get(pos_side)
            if not self.client.dry_run and live_pos is None:
                # ANTI-REBOND (point #4 audit 2026-08-03) : une SEULE lecture
                # get_positions() vide (maintenance/propagation/rate-limit OKX)
                # ne doit JAMAIS orpheliner une position reelle encore ouverte.
                entry["missing_count"] = int(entry.get("missing_count", 0)) + 1
                if entry["missing_count"] < DESYNC_CONFIRM_TICKS:
                    logger.warning(f"{pos_side} absent chez OKX (lecture {entry['missing_count']}/"
                                   f"{DESYNC_CONFIRM_TICKS}) -- anti-rebond, etat conserve.")
                    save_state(self.state)
                    continue
                logger.warning(f"{pos_side} absent chez OKX apres {DESYNC_CONFIRM_TICKS} lectures "
                               f"-- desync confirmee (liquidation/manuel/filet ?). Etat efface.")
                self._cancel_backstop(entry)   # filet eventuellement orphelin
                events.append({"type": "desync_closed", "pos_side": pos_side, "entry": entry})
                self.state[pos_side] = None
                continue

            if not self.client.dry_run and entry.get("missing_count"):
                entry["missing_count"] = 0   # position revue -> reset anti-rebond

            was_widened = bool(entry.get("widened"))
            res = evaluate_exit(pos_side, entry, close_by, pivots)
            reason, pp = res["reason"], res["pp"]

            # L'elargissement est decide PAR LA MARCHE (au premier candidat
            # atteint avec pp > 0), exactement comme main.py -- pas au moment ou
            # live.py voit le signal bloque. On ne notifie qu'a la transition.
            if entry.get("widened") and not was_widened:
                logger.info(f"{pos_side.upper()} ELARGIE a "
                            f"{datetime.fromtimestamp(res['widened_at_ts'] / 1000, tz=timezone.utc):%H:%M} "
                            f"UTC (TP/trailing/hold etendus, SL inchange).")
                events.append({"type": "widened", "pos_side": pos_side})

            if pp is None:
                # Aucune bougie 1-min encore fermee depuis l'entree : rien a
                # evaluer (le backtest est au meme point avant son premier pas).
                save_state(self.state)
                continue

            mark_px = float(self._candles[res["step_ts"]][4])

            if reason is None:
                save_state(self.state)   # persiste peak_pp/floor_pp (reporting)
                now_dt = datetime.fromtimestamp(now_ms / 1000, tz=timezone.utc)
                if now_dt.minute % 15 == 0:
                    elapsed_min = (now_ms - int(entry["traded_ts_ms"])) // 60000
                    peak = entry.get("peak_pp", pp)
                    floor = entry.get("floor_pp")
                    floor_txt = f"{floor:+.3%}" if floor is not None else "n/a"
                    logger.info(f"[POSITION] {pos_side.upper()} entree={entry['avg_px']:.2f} "
                               f"cloture_1min={mark_px:.2f} pp={pp:+.3%} peak={peak:+.3%} "
                               f"floor_trailing={floor_txt} ecoule={elapsed_min}min")
                continue

            if self.client.dry_run:
                # `or` couvre absent ET None (etat corrompu/edite a la main).
                sz = entry.get("sz") or "0"
            else:
                # OKX peut renvoyer `pos` SIGNE selon posSide (cf get_positions) --
                # place_order attend TOUJOURS une taille POSITIVE (le sens vient de
                # side+posSide). lstrip("-") plutot qu'un round-trip float pour ne
                # jamais perdre de precision decimale.
                sz = live_pos["pos"].lstrip("-")
            try:
                close = self.client.close_position_market(self.inst_id, pos_side, sz)
            except Exception as exc:  # noqa: BLE001 -- ne jamais crasher la boucle sur une erreur d'ordre
                logger.warning(f"Echec fermeture {pos_side} ({reason}) : {exc}", exc_info=True)
                events.append({"type": "close_failed", "pos_side": pos_side,
                               "reason": reason, "pp": pp, "error": str(exc)})
                continue

            # Fermeture reussie -> annule le filet catastrophe natif (point #1) :
            # sans ca il resterait un ordre algo orphelin cote OKX, susceptible
            # de fermer une eventuelle position re-ouverte du meme sens plus tard.
            self._cancel_backstop(entry)

            # Stats WR cumulatives depuis le lancement -- persistees dans le meme
            # positions_state.json, survivent aux redemarrages.
            stats = self.state.setdefault("stats", {"wins": 0, "losses": 0})
            if pp > 0:
                stats["wins"] += 1
            else:
                stats["losses"] += 1
            try:
                balance = self.client.get_balance("USDT")
            except Exception as exc:  # noqa: BLE001 -- notif/CSV degradee, jamais bloquant
                logger.warning(f"get_balance() a echoue a la fermeture de {pos_side} : {exc}")
                balance = None

            events.append({"type": "closed", "pos_side": pos_side, "reason": reason,
                           "pp": pp, "avg_px": entry["avg_px"], "mark_px": mark_px,
                           "entry_ts_ms": entry["traded_ts_ms"], "exit_ts_ms": now_ms,
                           "peak_pp": float(entry.get("peak_pp", pp)),
                           "worst_pp": float(entry.get("worst_pp", pp)),
                           "notional_usd": float(entry.get("notional_usd") or 0.0),
                           "wins": stats["wins"], "losses": stats["losses"],
                           "balance": balance, "order": close})
            self.state[pos_side] = None

        save_state(self.state)
        return events


    # -- Slot libre ? ----------------------------------------------------------
    def slot_free(self, pos_side: str) -> bool:
        """Un slot est libre si aucune position locale N'EST ouverte et aucune
        entree maker N'EST en cours pour ce sens (cf limite HEDGE en tete de
        module : jamais 2 positions independantes du meme sens)."""
        if self.state.get(pos_side) is not None:
            return False
        pending = self.state.get("pending_entry")
        if pending is not None and pending.get("pos_side") == pos_side:
            return False
        return True

    # -- Elargissement sur profit (symetrie main.py, campagne 2026-07-10) ------
    def register_widen_candidate(self, pos_side: str, traded_ts_ms: int) -> bool:
        """A appeler par live.py quand un signal `pos_side` arrive alors que
        `slot_free(pos_side)` est False -- AVANT d'abandonner le signal comme
        "bloque". Enregistre son `traded_ts` comme CANDIDAT d'elargissement sur
        la position deja ouverte ; renvoie True si le candidat a ete retenu.

        ALIGNEMENT GRILLE 1-MIN (2026-08-23) : c'est desormais la MARCHE de
        sortie qui decide (comme main.py::_up_managed_exit : `pp > 0.0 and t in
        widen_candidates`, au premier candidat atteint EN PROFIT). L'ancienne
        version tranchait ici, sur le prix de l'instant, et perdait
        definitivement un candidat dont le pp etait <= 0 a cette seconde precise
        -- alors que le backtest garde TOUS les candidats et elargit au premier
        qui tombe en profit. `widen_candidates` est persiste avec l'entree.

        DOWN_WIDEN_ENABLED : gate per-side du widen cote short, symetrie avec
        main.py::simulate_trades (widen_cands=None quand le flag est False)."""
        if not WIDEN_ON_PROFIT_ENABLED:
            return False
        if pos_side == "short" and not DOWN_WIDEN_ENABLED:
            return False
        entry = self.state.get(pos_side)
        if entry is None or entry.get("widened"):
            return False
        cands = entry.setdefault("widen_candidates", [])
        t = int(traded_ts_ms)
        if t in cands:
            return False
        cands.append(t)
        save_state(self.state)
        return True

    # -- Entree maker (non bloquant, avance d'un pas par appel) ---------------
    @staticmethod
    def _decision_ctx(pos_side: str, decision_ts_ms: int, trend_ctx: dict | None) -> dict:
        """Etat de marche fige a la barre de DECISION (alignement 2026-08-14,
        symetrie main.py::simulate_trades : les parametres de sortie sont
        choisis UNE FOIS a l'entree, jamais revises en cours de trade) :
        chop (mode scalp), TP override UP (chop/macro-bull), palier de cliquet
        DOWN chop/mbull. trend_ctx=None (warmup buffer, ou CHOP_SCALP_ENABLED
        =False cote live.py) -> defauts trend."""
        chop = filters.chop_mode(trend_ctx, decision_ts_ms) if trend_ctx is not None else False
        # Gate per-side CHOP_DOWN_ENABLED (finding revue adversariale
        # 2026-08-14) : main.py:simulate_trades exige `_chop and
        # CHOP_DOWN_ENABLED` pour les sorties scalp DOWN -- meme regle ici,
        # sinon un flag False cote backtest laisserait le live scalper.
        if pos_side == "short" and not CHOP_DOWN_ENABLED:
            chop = False
        ctx = {"chop": chop}
        if pos_side == "long":
            tp_ov = filters.up_tp_override(trend_ctx, decision_ts_ms) if trend_ctx is not None else None
            ctx["up_tp_override"] = float(tp_ov) if tp_ov is not None else None
        else:
            rx = filters.dn_ratchet_extra(trend_ctx, decision_ts_ms) if trend_ctx is not None else None
            ctx["dn_ratchet_extra"] = [float(rx[0]), float(rx[1])] if rx is not None else None
        return ctx

    def start_entry(self, pos_side: str, sz: str, price_hint: float,
                    decision_ts_ms: int, traded_ts_ms: int,
                    regime_map: dict | None, maker: bool = MAKER,
                    notional_usd: float = 0.0,
                    trend_ctx: dict | None = None) -> dict | None:
        """Demarre une tentative d'entree. Poste un ordre post-only (MAKER) ou
        market (TAKER) et enregistre l'etat 'pending_entry'. Ne bloque JAMAIS --
        `advance_pending_entry` fait progresser la tentative a chaque tick
        suivant (cf `live.py`). `notional_usd` (exposition USD demandee, cf
        live.py::run_hourly_cycle) est memorise sur l'entree pour convertir
        pp (%) en USDT dans les notifications de sortie (send_bet_resolved).
        trend_ctx (alignement 2026-08-14) : contexte SMA/ER de market_data,
        fige ici en decision_ctx (chop / TP override / palier DOWN)."""
        if not self.slot_free(pos_side):
            return None
        decision_ctx = self._decision_ctx(pos_side, decision_ts_ms, trend_ctx)
        side = "buy" if pos_side == "long" else "sell"
        if not maker:
            order = self.client.place_order(self.inst_id, side=side, pos_side=pos_side,
                                            ord_type="market", sz=sz)
            fill_px = price_hint
            if not self.client.dry_run:
                # BUG FIXE (2026-07-15, audit complet) : la reponse immediate
                # POST /trade/order NE CONTIENT PAS avgPx pour un ordre market
                # (OKX ne le rapporte que via une lecture ulterieure) --
                # price_hint (instantane ticker D'AVANT l'ordre) etait utilise a
                # tort comme prix d'entree, masquant tout slippage reel dans le
                # pp/SL/TP suivis ensuite. Poll get_order() une fois juste apres
                # (best-effort, jamais bloquant -- fallback price_hint si la
                # lecture echoue, une position DOIT toujours etre tracee).
                try:
                    confirmed = self.client.get_order(self.inst_id, order.get("ordId", ""))
                    fill_px = float(confirmed.get("avgPx") or price_hint)
                except Exception as exc:  # noqa: BLE001
                    # BUG FIXE (2026-07-15, audit erreurs silencieuses) : le
                    # repli sur price_hint (moins precis, pre-slippage) partait
                    # sans AUCUNE trace qu'on operait en precision degradee sur
                    # cette position. Comportement de repli inchange.
                    logger.warning(f"get_order() post-fill taker a echoue pour {pos_side} "
                                   f"(repli sur price_hint={price_hint}) : {exc}")
            self._finalize_entry(pos_side, fill_px, decision_ts_ms, traded_ts_ms,
                                 regime_map, notional_usd, sz, decision_ctx)
            return {"type": "opened", "pos_side": pos_side, "mode": "taker", "order": order}

        # ALIGNEMENT BACKTEST : limite postee UNE FOIS a P0 et laissee STATIQUE
        # pendant MAKER_FILL_WINDOW_MIN -- aucun repricing (2026-08-14).
        #
        # 2026-08-23 : (a) P0 est desormais le VRAI P0 du backtest -- l'OPEN de
        # la bougie 1-min d'open-time traded_ts (= open_ib.get(traded_ts) dans
        # simulate_trades), fourni par live.py, et non plus un instantane
        # bid/ask ; (b) ord_type passe de "post_only" a "limit". Raison : le
        # backtest considere la limite REMPLIE des qu'un low 1-min descend sous
        # P0 (long) -- y compris quand le marche est DEJA sous P0 au moment de
        # poster. Un post_only serait alors REJETE par OKX ("would take
        # liquidity") et ferait perdre un trade que le backtest prend. Une
        # limite GTC a P0 couvre les deux cas : elle se remplit immediatement
        # au prix courant (>= P0, donc jamais pire) s'il croise, sinon elle
        # repose et se remplit en maker quand le prix revient la toucher.
        px = self._round_px(price_hint)
        order = self.client.place_order(self.inst_id, side=side, pos_side=pos_side,
                                        ord_type="limit", sz=sz, px=px)
        self.state["pending_entry"] = {
            "pos_side": pos_side, "sz": sz, "px": price_hint,
            "ord_id": order.get("ordId", ""), "decision_ts_ms": decision_ts_ms,
            "traded_ts_ms": traded_ts_ms, "posted_at_ms": traded_ts_ms,
            "notional_usd": notional_usd, "decision_ctx": decision_ctx,
            "deadline_ms": traded_ts_ms + MAKER_FILL_WINDOW_MIN * 60_000,
        }
        save_state(self.state)
        return {"type": "entry_posted", "pos_side": pos_side, "mode": "maker", "order": order}

    def _simulate_maker_fill(self, pending: dict, now_ms: int) -> dict:
        """Simulation DRY_RUN du fill d'une limite maker -- port du test MAKER de
        main.py::simulate_trades : la limite a P0 est REMPLIE ssi, sur les
        bougies 1-min de la fenetre, un `low` passe STRICTEMENT sous P0 (long)
        ou un `high` STRICTEMENT au-dessus (short). Fill a P0 exactement (le
        backtest entre toujours a P0)."""
        traded_ts = int(pending["traded_ts_ms"])
        p0 = float(pending["px"])
        try:
            rows = self.client.get_candles_range(self.inst_id, traded_ts,
                                                 min(now_ms, int(pending["deadline_ms"])),
                                                 bar="1m")
        except Exception as exc:  # noqa: BLE001 -- non bloquant : on retente au tick suivant
            logger.warning(f"Simulation de fill maker : lecture des bougies 1-min echouee : {exc}")
            return {"state": "live"}
        long_side = pending["pos_side"] == "long"
        for r in rows:
            if (float(r[3]) < p0) if long_side else (float(r[2]) > p0):
                return {"state": "filled", "avgPx": p0}
        return {"state": "live"}

    def advance_pending_entry(self, now_ms: int, regime_map: dict | None) -> dict | None:
        """A appeler CHAQUE tick. Fait progresser l'entree maker en cours :
        verifie le fill de la limite STATIQUE (plus aucun repricing depuis
        l'alignement 2026-08-14) ; fenetre MAKER_FILL_WINDOW_MIN ecoulee sans
        fill = opportunite RATEE (entry_missed, semantique backtest) -- plus
        JAMAIS de bascule taker. Un fill partiel au deadline est garde tel
        quel. La branche deadline passe AVANT la lecture de l'ordre : des
        lectures en echec repetees ne peuvent plus retarder l'annulation
        au-dela de la fenetre (finding revue adversariale 2026-08-14)."""
        pending = self.state.get("pending_entry")
        if pending is None:
            return None
        pos_side = pending["pos_side"]
        ord_id = pending["ord_id"]

        if now_ms < pending["deadline_ms"]:
            try:
                if self.client.dry_run:
                    # get_order() en DRY_RUN repond TOUJOURS "live" -> on simule
                    # le fill. ALIGNEMENT GRILLE 1-MIN (2026-08-23) : on applique
                    # EXACTEMENT le test du backtest (simulate_trades, bloc
                    # MAKER) -- rempli ssi un LOW 1-min passe STRICTEMENT sous P0
                    # (long) / un HIGH 1-min passe STRICTEMENT au-dessus (short),
                    # sur les bougies deja fermees de la fenetre. L'ancienne
                    # version comparait le ticker instantane, qui rate tout
                    # passage intra-minute.
                    order = self._simulate_maker_fill(pending, now_ms)
                else:
                    order = self.client.get_order(self.inst_id, ord_id)
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Impossible de lire l'ordre {ord_id} ({pos_side}) : {exc}", exc_info=True)
                return {"type": "entry_check_failed", "pos_side": pos_side, "error": str(exc)}

            if order.get("state") == "filled":
                avg_px = float(order.get("avgPx") or pending["px"])
                self.state["pending_entry"] = None
                self._finalize_entry(pos_side, avg_px, pending["decision_ts_ms"],
                                     pending["traded_ts_ms"], regime_map,
                                     pending.get("notional_usd", 0.0), pending["sz"],
                                     pending.get("decision_ctx"))
                return {"type": "opened", "pos_side": pos_side, "mode": "maker", "avg_px": avg_px}
            return None

        if now_ms >= pending["deadline_ms"]:
            # ALIGNEMENT BACKTEST 2026-08-14 : fenetre ecoulee = opportunite
            # RATEE (le backtest compte un "missed" et n'ouvre RIEN). L'ancienne
            # bascule taker est SUPPRIMEE -- avec elle disparait le bug de
            # sur-exposition (lecture du fill partiel echouee -> "suppose non
            # rempli" -> market PLEINE taille par-dessus un fill partiel reel,
            # jusqu'a x1.8 la taille voulue sous levier).
            if self.client.dry_run:
                self.state["pending_entry"] = None
                save_state(self.state)
                return {"type": "entry_missed", "pos_side": pos_side, "filled_sz": 0.0}

            if not pending.get("cancel_sent"):
                try:
                    self.client.cancel_order(self.inst_id, ord_id)
                    # Marque seulement un appel ABOUTI (finding revue 2026-08-14 :
                    # l'ancienne version marquait "envoye" meme sur timeout reseau
                    # -- l'ordre pouvait rester VIVANT cote OKX apres la purge).
                    pending["cancel_sent"] = True
                except Exception as exc:  # noqa: BLE001 -- deja annule/expire cote OKX, OU reseau :
                    # dans les deux cas, la lecture d'etat ci-dessous tranche.
                    logger.warning(f"cancel_order maker {pos_side} : {exc} -- l'etat lu ci-dessous decide.")
                save_state(self.state)

            # FILL PARTIEL (point #3 audit 2026-08-03, durci 2026-08-14) : on
            # garde SEULEMENT ce qui est reellement rempli. FAIL-SAFE : tant que
            # la lecture du fill echoue, on ne poste RIEN et on ne jette RIEN --
            # nouvelle lecture au tick suivant (un fill partiel reel doit finir
            # trace comme position, jamais complete a l'aveugle).
            try:
                prev = self.client.get_order(self.inst_id, ord_id)
                maker_filled_sz = float(prev.get("accFillSz") or 0.0)
                maker_avg_px = float(prev.get("avgPx") or 0.0)
            except Exception as exc:  # noqa: BLE001
                n = pending["confirm_fail_count"] = int(pending.get("confirm_fail_count", 0)) + 1
                log = logger.error if n >= 3 else logger.warning
                log(f"Fill maker {pos_side} INCONNU apres annulation (lecture {n} echouee : "
                    f"{exc}) -- aucun ordre poste, nouvelle lecture au prochain tick.")
                save_state(self.state)
                return {"type": "entry_check_failed", "pos_side": pos_side, "error": str(exc)}

            # L'ordre doit etre TERMINAL avant toute purge/finalisation
            # (finding revue 2026-08-14) : si l'annulation n'est pas encore
            # effective ("live"/"partially_filled"), un accFillSz partiel lu
            # maintenant n'est pas final -- et purger laisserait un ordre
            # VIVANT non suivi cote OKX. On re-tente le cancel au tick suivant.
            state_str = str(prev.get("state") or "").lower()
            if state_str not in ("canceled", "cancelled", "filled"):
                pending["cancel_sent"] = False
                save_state(self.state)
                return {"type": "entry_check_failed", "pos_side": pos_side,
                        "error": f"ordre non terminal ({state_str or 'inconnu'}) apres annulation"}

            total_sz = float(pending["sz"])
            notional = float(pending.get("notional_usd") or 0.0)
            self.state["pending_entry"] = None
            if maker_filled_sz <= 0:
                save_state(self.state)
                return {"type": "entry_missed", "pos_side": pos_side, "filled_sz": 0.0}
            mode = "maker" if maker_filled_sz >= total_sz else "maker_partial"
            if mode == "maker_partial":
                notional = notional * (maker_filled_sz / total_sz)  # reporting pp->USDT au prorata
                logger.info(f"Fill partiel maker {pos_side} : {maker_filled_sz}/{total_sz} rempli "
                            f"a {maker_avg_px} -- position gardee a cette taille, PAS de completion taker.")
            self._finalize_entry(pos_side, maker_avg_px or float(pending["px"]),
                                 pending["decision_ts_ms"], pending["traded_ts_ms"],
                                 regime_map, notional, self._fmt_sz(maker_filled_sz),
                                 pending.get("decision_ctx"))
            return {"type": "opened", "pos_side": pos_side, "mode": mode,
                    "avg_px": maker_avg_px or float(pending["px"])}

        return None

    def _finalize_entry(self, pos_side: str, avg_px: float, decision_ts_ms: int,
                        traded_ts_ms: int, regime_map: dict | None,
                        notional_usd: float = 0.0, sz: str = "0",
                        decision_ctx: dict | None = None) -> None:
        entry = {"avg_px": avg_px, "decision_ts_ms": decision_ts_ms,
                 "traded_ts_ms": traded_ts_ms, "peak_pp": -9.0, "worst_pp": 9.0,
                 "floor_pp": None, "widened": False, "widen_candidates": [],
                 "notional_usd": notional_usd,
                 "sz": sz, "missing_count": 0, "backstop_algo_id": None,
                 "backstop_trigger_px": None,
                 # Regime FIGE a la barre de decision : main.py le passe une fois
                 # a _down_managed_exit (gate de l'extension TP DOWN) et ne le
                 # revise jamais en cours de trade.
                 "regime": filters.current_regime(regime_map, decision_ts_ms)}
        # Contexte de decision (alignement 2026-08-14) : chop / TP override UP /
        # palier DOWN, figes a la barre de decision (cf _decision_ctx). Absent
        # (position adoptee par reconcile, vieux pending) -> defauts trend,
        # choix conservateur symetrique du down_*_override=None ci-dessous.
        entry.update(decision_ctx or {"chop": False})
        entry.setdefault("chop", False)
        if pos_side == "long":
            entry.setdefault("up_tp_override", None)
        else:
            entry.setdefault("dn_ratchet_extra", None)
        if pos_side == "short":
            if entry["chop"]:
                # En chop, main.py court-circuite l'override regime
                # (regime="neutral") : les parametres scalp CHOP_DOWN_* sont
                # appliques par evaluate_exit via entry["chop"].
                entry["down_max_steps_override"] = None
                entry["down_trail_gap_override"] = None
            else:
                max_steps_eff, trail_gap_eff = filters.effective_down_exit_params(decision_ts_ms, regime_map)
                entry["down_max_steps_override"] = max_steps_eff
                entry["down_trail_gap_override"] = trail_gap_eff
        # Filet catastrophe natif OKX (point #1) -- pose des que l'avg_px reel est
        # connu, pour que la position ait sa protection 'daemon mort' immediate.
        algo_id, trigger_px = self._place_backstop(pos_side, avg_px, sz)
        entry["backstop_algo_id"] = algo_id
        entry["backstop_trigger_px"] = trigger_px
        self.state[pos_side] = entry
        save_state(self.state)

    # -- Reconciliation au demarrage (point #5 audit 2026-08-03) -------------
    def reconcile_positions(self, now_ms: int) -> list[dict]:
        """A appeler UNE FOIS au demarrage (mode reel). Aligne l'etat local sur
        la realite OKX :
          - position reelle ouverte SANS entree locale -> ADOPTEE (avg_px/taille/
            cTime depuis OKX) pour ne jamais laisser une orpheline sans suivi
            SL/TP (ex. state perdu/corrompu, position ouverte hors bande) ;
          - entree locale SANS position reelle -> effacee (fermee pendant l'arret,
            ou etat papier residuel apres passage en reel) ;
          - garantit un filet catastrophe pour chaque position suivie (adopte un
            algo existant plutot que d'en dupliquer un).
        No-op en DRY_RUN (aucune position reelle a reconcilier)."""
        if self.client.dry_run:
            return []
        events: list[dict] = []
        try:
            live_positions = {p["posSide"]: p for p in self.client.get_positions(self.inst_id)}
        except Exception as exc:  # noqa: BLE001
            logger.error(f"reconcile_positions : get_positions a echoue : {exc} -- reconciliation "
                         f"sautee (l'etat local est conserve tel quel).")
            return events
        try:
            pending_algos = self.client.get_algo_pending(self.inst_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"reconcile_positions : get_algo_pending a echoue : {exc}")
            pending_algos = []
        pending_algo_ids = {a.get("algoId") for a in pending_algos}

        # Pending maker survivant a un ARRET (finding revue adversariale
        # 2026-08-14) : la fenetre n'est plus garantie, et laisser
        # advance_pending_entry le re-finaliser apres l'adoption ci-dessous
        # ecraserait l'entree adoptee (double filet catastrophe, suivi perdu).
        # On annule l'ordre (sans effet s'il est deja rempli/annule) et on
        # purge -- s'il s'est rempli pendant l'arret, la position existe cote
        # OKX et est ADOPTEE par la boucle ci-dessous, avec son filet.
        pending = self.state.get("pending_entry")
        if pending is not None:
            try:
                self.client.cancel_order(self.inst_id, pending.get("ord_id", ""))
            except Exception as exc:  # noqa: BLE001 -- deja rempli/annule : sans consequence
                logger.warning(f"reconcile : annulation du pending {pending.get('ord_id')} : {exc}")
            logger.warning("reconcile : pending_entry residuel d'avant l'arret -> purge "
                           "(fenetre maker non garantie ; position eventuelle adoptee ci-dessous).")
            self.state["pending_entry"] = None
            events.append({"type": "reconciled_closed", "pos_side": pending.get("pos_side", "?")})

        for pos_side in ("long", "short"):
            live_pos = live_positions.get(pos_side)
            entry = self.state.get(pos_side)

            if live_pos is not None and entry is None:
                avg_px = float(live_pos.get("avgPx") or 0.0)
                sz = str(live_pos.get("pos", "0")).lstrip("-")
                c_time = int(live_pos.get("cTime") or now_ms)
                # ALIGNEMENT GRILLE 1-MIN (2026-08-23) : traded_ts_ms EST la cle
                # de la grille des bougies 1-min (t = traded_ts + j*60s). cTime
                # est un instant quelconque -> on le plancher sur la minute,
                # sinon aucune bougie ne correspondrait et la marche de sortie
                # ne demarrerait jamais.
                c_time = (c_time // 60_000) * 60_000
                entry = {"avg_px": avg_px, "decision_ts_ms": c_time, "traded_ts_ms": c_time,
                         "peak_pp": -9.0, "worst_pp": 9.0, "floor_pp": None, "widened": False,
                         "widen_candidates": [],
                         "notional_usd": 0.0, "sz": sz, "missing_count": 0,
                         "backstop_algo_id": None, "backstop_trigger_px": None, "adopted": True,
                         # Pas de contexte de decision fiable pour une position
                         # adoptee -> defauts trend (chop=False, regime neutre),
                         # meme choix conservateur que les overrides ci-dessous.
                         "chop": False, "regime": "neutral"}
                if pos_side == "long":
                    entry["up_tp_override"] = None
                else:
                    entry["dn_ratchet_extra"] = None
                if pos_side == "short":
                    # Pas de regime_map fiable au demarrage -> params DOWN par defaut
                    # (config_main), jamais l'override regime (choix conservateur).
                    entry["down_max_steps_override"] = None
                    entry["down_trail_gap_override"] = None
                self.state[pos_side] = entry
                logger.warning(f"reconcile : position {pos_side} ouverte sur OKX mais absente de "
                               f"l'etat local -> ADOPTEE (avg_px={avg_px} sz={sz}). Suivi SL/TP repris.")
                events.append({"type": "adopted", "pos_side": pos_side, "avg_px": avg_px})
            elif live_pos is None and entry is not None:
                logger.warning(f"reconcile : entree locale {pos_side} sans position OKX -> effacee "
                               f"(fermee pendant l'arret / etat residuel).")
                self._cancel_backstop(entry)
                self.state[pos_side] = None
                events.append({"type": "reconciled_closed", "pos_side": pos_side})

            # Garantit un filet pour toute position desormais suivie.
            entry = self.state.get(pos_side)
            if entry is not None and NATIVE_BACKSTOP_ENABLED:
                if entry.get("backstop_algo_id") not in pending_algo_ids:
                    existing = next((a for a in pending_algos if a.get("posSide") == pos_side), None)
                    if existing is not None:
                        entry["backstop_algo_id"] = existing.get("algoId")
                        trig = existing.get("slTriggerPx")
                        entry["backstop_trigger_px"] = float(trig) if trig else None
                        logger.info(f"reconcile : filet {pos_side} existant adopte "
                                    f"(algoId={existing.get('algoId')}).")
                    else:
                        algo_id, trig = self._place_backstop(pos_side, float(entry["avg_px"]),
                                                             entry.get("sz") or "0")
                        entry["backstop_algo_id"] = algo_id
                        entry["backstop_trigger_px"] = trig
        save_state(self.state)
        return events
