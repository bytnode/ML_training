"""position.py -- PositionManager : gestion des positions ouvertes et des
entrees maker en cours, persistance JSON entre cycles.

DIFFERENCE STRUCTURELLE avec le backtest : `main.py::_up_managed_exit` / `_down_
managed_exit` rejouent tout l'historique intrabar d'un coup (une seule boucle
Python voit tout le futur simule). Ici, `live.py` appelle ce module UNE FOIS
PAR MINUTE avec seulement le prix COURANT -- peak/floor (trailing) et l'etat
d'un ordre maker en cours DOIVENT donc etre persistes entre les appels
(`config_live.POSITIONS_STATE_JSON`), sinon un redemarrage perdrait le pic deja
atteint et re-armerait le trailing a zero.

PARITE AVEC LE BACKTEST : `evaluate_exit` utilise EXACTEMENT les memes seuils
(config_main.py) et la meme fonction `filters.sl_threshold` (SL temporel) que
`main.py`. Le prix courant vient de `avgPx`/`markPx` OKX (jamais de `uplRatio`,
dont la convention de signe/definition exacte n'est pas garantie identique a la
formule backtest) -- `pp` est recalcule ICI avec la MEME formule que le backtest
(`cl/entry - 1` LONG, `1 - cl/entry` SHORT), pour une parite exacte.

LIMITE DELIBEREE (mode HEDGE OKX) : contrairement au backtest qui autorise 2
positions simultanees MEME SENS (2x DOWN par ex., suivies independamment), OKX
en mode HEDGE ne montre qu'UNE position fusionnee par (instId, posSide) --
une 2e entree sur le meme sens re-blend l'avgPx et casserait le suivi SL/TP/
trailing/timestop de la position deja ouverte. Ce module cape donc a 1 SLOT PAR
SENS (long, short) = 2 au total, plus strict que le backtest sur le cas rare
"2x meme sens" mais jamais incorrect -- ecart documente, pas un bug.

ELARGISSEMENT SUR PROFIT (symetrie main.py, campagne 2026-07-10) : quand un
signal du meme sens arrive alors qu'un slot est deja occupe (`try_widen`,
appele par live.py juste avant d'abandonner un signal bloque), si la position
ouverte est ACTUELLEMENT en profit (pp > 0) et n'a jamais ete elargie, on
elargit UNE SEULE FOIS son take-profit/trailing-gap/hold max (SL jamais
touche) -- meme mecanisme et memes multiplicateurs (WIDEN_ON_PROFIT_ENABLED,
config_main.py) que main.py::_up_managed_exit/_down_managed_exit.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("live.position")

_LIVE_DIR = Path(__file__).resolve().parent
if str(_LIVE_DIR) not in sys.path:
    sys.path.insert(0, str(_LIVE_DIR))

import filters  # noqa: E402
from config_main import (  # noqa: E402
    UP_TAKE_PROFIT, UP_TRAIL_ACTIVATE, UP_TRAIL_GAP,
    UP_SL_START_DEC, UP_SL_END_DEC, UP_SL_FLOOR_DEC,
    UP_MAX_HOLD_MIN, UP_CHECK_MIN,
    DOWN_TAKE_PROFIT, DOWN_TRAIL_ACTIVATE, DOWN_TRAIL_GAP,
    DOWN_SL_START_DEC, DOWN_SL_END_DEC, DOWN_SL_FLOOR_DEC,
    DOWN_MAX_HOLD_MIN, DOWN_CHECK_MIN,
    MAKER, MAKER_FILL_WINDOW_MIN,
    WIDEN_ON_PROFIT_ENABLED,
    UP_WIDEN_TP_MULT, UP_WIDEN_TRAIL_GAP_MULT, UP_WIDEN_MAX_HOLD_MULT,
    DOWN_WIDEN_TP_MULT, DOWN_WIDEN_TRAIL_GAP_MULT, DOWN_WIDEN_MAX_HOLD_MULT,
)
from config_live import POSITIONS_STATE_JSON, OKX_INST_ID, ENTRY_REPRICE_EVERY_S


def _new_cl_ord_id(pos_side: str) -> str:
    """Identifiant client unique pour une tentative d'ordre (OKX dedupe sur ce
    champ) -- BUG FIXE (2026-08-03, audit idempotence) : sans lui, une reponse
    HTTP perdue apres qu'OKX a deja accepte l'ordre (timeout reseau sur
    `_request`, cf api_okx.py) ne laissait aucun moyen de faire le lien entre
    l'ordre reellement passe et notre tentative -- l'ordre devenait orphelin
    tant que la reconciliation (`PositionManager.check_positions`) ne le
    rattachait pas. Prefixe alphanumerique pur, <=32 caracteres (contrainte OKX)."""
    return f"live{pos_side}{int(time.time() * 1000)}"[:32]


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
    POSITIONS_STATE_JSON.parent.mkdir(parents=True, exist_ok=True)
    POSITIONS_STATE_JSON.write_text(json.dumps(state, indent=2))


# ----------------------------------------------------------------------------
#  Evaluation de sortie (SL / TP / trailing / timestop) -- meme axe que main.py
# ----------------------------------------------------------------------------
def evaluate_exit(pos_side: str, entry: dict, mark_px: float, now_ms: int) -> tuple[str | None, float]:
    """Renvoie (exit_reason|None, pp). Mute `entry` en place (peak_pp/floor_pp).

    Si entry["widened"] est True (cf PositionManager.try_widen), take_profit/
    trail_gap/max_hold_min sont MULTIPLIES par les facteurs UP_WIDEN_*/
    DOWN_WIDEN_* -- le SL (sl_start/sl_end/sl_floor ET son horizon
    sl_max_hold_min) reste TOUJOURS sur la base non-elargie, symetrie exacte
    avec main.py (sl_j y est aussi borne a l'horizon ORIGINAL/effectif, jamais
    au horizon widened)."""
    avg_px = float(entry["avg_px"])
    widened = bool(entry.get("widened", False))
    if pos_side == "long":
        pp = mark_px / avg_px - 1.0
        trail_activate = UP_TRAIL_ACTIVATE
        sl_start, sl_end, sl_floor = UP_SL_START_DEC, UP_SL_END_DEC, UP_SL_FLOOR_DEC
        check_min = UP_CHECK_MIN
        sl_max_hold_min = UP_MAX_HOLD_MIN            # SL jamais elargi -- horizon ORIGINAL toujours
        if widened:
            take_profit = UP_TAKE_PROFIT * UP_WIDEN_TP_MULT
            trail_gap = UP_TRAIL_GAP * UP_WIDEN_TRAIL_GAP_MULT
            max_hold_min = UP_MAX_HOLD_MIN * UP_WIDEN_MAX_HOLD_MULT
        else:
            take_profit, trail_gap, max_hold_min = UP_TAKE_PROFIT, UP_TRAIL_GAP, UP_MAX_HOLD_MIN
    else:
        pp = 1.0 - mark_px / avg_px
        trail_activate = DOWN_TRAIL_ACTIVATE
        sl_start, sl_end, sl_floor = DOWN_SL_START_DEC, DOWN_SL_END_DEC, DOWN_SL_FLOOR_DEC
        check_min = DOWN_CHECK_MIN
        trail_gap_override = entry.get("down_trail_gap_override")
        base_trail_gap = DOWN_TRAIL_GAP if trail_gap_override is None else trail_gap_override
        max_steps_override = entry.get("down_max_steps_override")
        base_max_hold_min = DOWN_MAX_HOLD_MIN if max_steps_override is None else max_steps_override * check_min
        sl_max_hold_min = base_max_hold_min           # SL jamais elargi -- horizon EFFECTIF (regime, non-widened)
        if widened:
            take_profit = DOWN_TAKE_PROFIT * DOWN_WIDEN_TP_MULT
            trail_gap = base_trail_gap * DOWN_WIDEN_TRAIL_GAP_MULT
            max_hold_min = base_max_hold_min * DOWN_WIDEN_MAX_HOLD_MULT
        else:
            take_profit, trail_gap, max_hold_min = DOWN_TAKE_PROFIT, base_trail_gap, base_max_hold_min

    elapsed_min = max(0, (now_ms - int(entry["traded_ts_ms"])) // 60000)
    max_steps = max(1, int(max_hold_min) // check_min)
    sl_max_steps = max(1, int(sl_max_hold_min) // check_min)
    # backtest boucle `for j in range(max_steps)` -> j max reellement atteint =
    # max_steps-1 (jamais max_steps) ; meme borne ici pour un seuil identique
    # au dernier pas surveille avant le fallback timestop. sl_j utilise TOUJOURS
    # l'horizon SL (jamais elargi), meme si max_steps (TP/trail/hold) l'est.
    j = min(elapsed_min // check_min, max_steps - 1)
    sl_j = min(elapsed_min // check_min, sl_max_steps - 1)

    # peak_pp/worst_pp sont mis a jour AVANT tout `return` (y compris le tick qui
    # declenche SL/TP) pour que le reporting Telegram (plus haut profit/plus
    # forte perte atteints) reflete bien l'extremum reel de la position, y
    # compris sur son tout dernier tick -- ca ne change AUCUNE decision de
    # sortie (celles-ci comparent toujours `pp` brut aux seuils).
    if pp > entry.get("peak_pp", -9.0):
        entry["peak_pp"] = pp
    if pp < entry.get("worst_pp", 9.0):
        entry["worst_pp"] = pp

    sl_th = filters.sl_threshold(sl_j, sl_max_steps, sl_start, sl_end, sl_floor)
    if pp <= sl_th:
        return "stop_loss", pp
    if pp >= take_profit:
        return "take_profit", pp
    if entry["peak_pp"] >= trail_activate:
        entry["floor_pp"] = entry["peak_pp"] - trail_gap
    if entry.get("floor_pp") is not None and pp < entry["floor_pp"]:
        return "trailing", pp
    if elapsed_min >= max_hold_min:
        return "timestop", pp
    return None, pp


# ----------------------------------------------------------------------------
#  PositionManager
# ----------------------------------------------------------------------------
class PositionManager:
    def __init__(self, client, inst_id: str = OKX_INST_ID):
        self.client = client
        self.inst_id = inst_id
        self.state = load_state()

    # -- Monitoring des positions ouvertes (appele chaque minute) ------------
    def check_positions(self, now_ms: int, regime_map: dict | None = None) -> list[dict]:
        """Evalue chaque position ouverte (selon l'etat local) contre SL/TP/
        trailing/timestop. Ferme au marche (taker, cf CLAUDE.md : la sortie est
        TOUJOURS taker) si un seuil est franchi. Renvoie la liste des evenements
        de sortie (dicts) survenus ce tick.

        BUG FIXE (2026-07-15) : en DRY_RUN, `place_order` ne poste JAMAIS de
        vrai ordre (stubbe, cf api_okx.py) -- OKX ne peut donc physiquement
        JAMAIS montrer la position "papier" dans `get_positions()` (methode de
        LECTURE, jamais stubbee). L'ancien code traitait ca comme une desync
        (liquidation/fermeture manuelle) des le tick suivant l'ouverture,
        effacant l'etat local a chaque fois -- aucune position papier n'etait
        donc JAMAIS suivie ni fermee. En DRY_RUN, l'etat local EST la verite :
        on ne consulte plus `get_positions()` du tout, le prix courant vient du
        ticker (lecture reelle et independante de tout ordre).

        BUG FIXE (2026-08-03, audit reconciliation) : la reconciliation OKX vers
        l'etat local etait UNIDIRECTIONNELLE -- une position locale absente chez
        OKX etait bien detectee (desync ci-dessous), mais l'inverse (position
        REELLE chez OKX, absente de l'etat local -- ex. positions_state.json
        tronque par un crash pendant `save_state`, ou crash entre un
        `place_order` taker et `_finalize_entry`) n'etait JAMAIS adoptee ni
        surveillee : une position a levier tournait alors sans AUCUN SL/TP/
        trailing/timestop, invisible du daemon. `_adopt_position` comble ce
        trou a CHAQUE tick (pas seulement au demarrage), avant la boucle
        d'evaluation ci-dessous -- une position adoptee ce tick est donc deja
        surveillee ce meme tick."""
        events = []
        live_positions = {}
        if not self.client.dry_run:
            live_positions = {p["posSide"]: p for p in self.client.get_positions(self.inst_id)}
            for pos_side, live_pos in live_positions.items():
                if self.state.get(pos_side) is None:
                    events.append(self._adopt_position(pos_side, live_pos, regime_map, now_ms))

        for pos_side in ("long", "short"):
            entry = self.state.get(pos_side)
            if entry is None:
                continue
            live_pos = live_positions.get(pos_side)
            if not self.client.dry_run and live_pos is None:
                # Position fermee hors de notre controle (liquidation, manuel...) :
                # on desynchronise l'etat local et on notifie -- jamais silencieux.
                # (mode reel UNIQUEMENT -- en DRY_RUN il n'y a jamais de live_pos
                # a comparer, cf note ci-dessus.)
                logger.warning(f"{pos_side} attendu ouvert mais absent chez OKX -- "
                               f"desync (liquidation/manuel ?). Etat local efface.")
                events.append({"type": "desync_closed", "pos_side": pos_side, "entry": entry})
                self.state[pos_side] = None
                continue

            if self.client.dry_run:
                ticker = self.client.get_ticker(self.inst_id)
                mark_px = float(ticker["last"])
            else:
                mark_px = float(live_pos.get("markPx") or live_pos.get("last") or entry["avg_px"])
            reason, pp = evaluate_exit(pos_side, entry, mark_px, now_ms)
            if reason is None:
                save_state(self.state)   # persiste peak_pp/floor_pp mis a jour
                # NOUVEAU (2026-07-15, logs lisibles) : visibilite periodique de
                # l'evolution d'une position ouverte, toutes les 15 minutes,
                # visible en direct via `docker compose logs -f live-okx`.
                now_dt = datetime.fromtimestamp(now_ms / 1000, tz=timezone.utc)
                if now_dt.minute % 15 == 0:
                    elapsed_min = (now_ms - int(entry["traded_ts_ms"])) // 60000
                    peak = entry.get("peak_pp", pp)
                    floor = entry.get("floor_pp")
                    floor_txt = f"{floor:+.3%}" if floor is not None else "n/a"
                    logger.info(f"[POSITION] {pos_side.upper()} entree={entry['avg_px']:.2f} "
                               f"actuel={mark_px:.2f} pp={pp:+.3%} peak={peak:+.3%} "
                               f"floor_trailing={floor_txt} ecoule={elapsed_min}min")
                continue

            if self.client.dry_run:
                # BUG FIXE (2026-07-15, audit complet) : .get("sz", "0") ne
                # substitue le defaut QUE si la cle est ABSENTE, pas si elle
                # vaut explicitement None -- un state corrompu/edite a la main
                # avec "sz": null aurait fait planter ce tick. `or` couvre les
                # deux cas (absent ET None).
                sz = entry.get("sz") or "0"
            else:
                # BUG FIXE (audit 2026-07-10) : OKX peut renvoyer `pos` SIGNE selon
                # posSide (cf docstring get_positions dans api_okx.py) -- place_order
                # attend TOUJOURS une taille POSITIVE (le sens vient de side+posSide,
                # jamais du signe de sz). lstrip("-") plutot qu'un round-trip float
                # pour ne jamais perdre de precision decimale sur la taille exacte.
                sz = live_pos["pos"].lstrip("-")
            try:
                close = self.client.close_position_market(
                    self.inst_id, pos_side, sz, cl_ord_id=_new_cl_ord_id(pos_side))
            except Exception as exc:  # noqa: BLE001 -- ne jamais crasher la boucle sur une erreur d'ordre
                logger.warning(f"Echec fermeture {pos_side} ({reason}) : {exc}", exc_info=True)
                events.append({"type": "close_failed", "pos_side": pos_side,
                               "reason": reason, "pp": pp, "error": str(exc)})
                continue

            # Stats WR cumulatives depuis le lancement -- persistees dans le meme
            # positions_state.json (pas de fichier separe), survivent aux
            # redemarrages exactement comme le reste de l'etat des positions.
            stats = self.state.setdefault("stats", {"wins": 0, "losses": 0})
            if pp > 0:
                stats["wins"] += 1
            else:
                stats["losses"] += 1
            try:
                balance = self.client.get_balance("USDT")
            except Exception as exc:  # noqa: BLE001 -- notif/CSV degradee, jamais bloquant
                # BUG FIXE (2026-07-15, audit erreurs silencieuses) : balance
                # tombait a None sans AUCUN log -- la notif Telegram affichait
                # juste "capital : n/a" sans jamais dire pourquoi (vraie panne
                # OKX ? autre chose ?). Log trace la cause racine ; le
                # comportement de repli (None -> "n/a") reste inchange.
                logger.warning(f"get_balance() a echoue a la fermeture de {pos_side} : {exc}")
                balance = None

            events.append({"type": "closed", "pos_side": pos_side, "reason": reason,
                           "pp": pp, "avg_px": entry["avg_px"], "mark_px": mark_px,
                           "entry_ts_ms": entry["traded_ts_ms"], "exit_ts_ms": now_ms,
                           "peak_pp": float(entry.get("peak_pp", pp)),
                           "worst_pp": float(entry.get("worst_pp", pp)),
                           # BUG FIXE (2026-07-15, audit complet) : meme raison
                           # que le "sz" ci-dessus -- `or` couvre absent ET None.
                           "notional_usd": float(entry.get("notional_usd") or 0.0),
                           "wins": stats["wins"], "losses": stats["losses"],
                           "balance": balance, "order": close})
            self.state[pos_side] = None

        save_state(self.state)
        return events

    # -- Adoption d'une position OKX inconnue de l'etat local -----------------
    def _adopt_position(self, pos_side: str, live_pos: dict, regime_map: dict | None,
                        now_ms: int) -> dict:
        """Position REELLEMENT ouverte chez OKX mais absente de l'etat local
        (cf BUG FIXE 2026-08-03 dans check_positions) -- ADOPTEE au lieu de
        rester invisible et non surveillee. L'heure d'entree reelle n'est pas
        reconstructible avec certitude depuis `get_positions()` : l'horizon SL
        (time-decay)/max-hold/timestop redemarre a `now_ms` (choix conservateur
        -- ca donne au pire un horizon PLUS LONG que la duree de vie reelle
        restante, jamais une sortie prematuree ; le risque reel qu'on corrige
        est "aucune surveillance du tout", pas "horizon legerement trop long")."""
        avg_px = float(live_pos["avgPx"])
        sz = live_pos["pos"].lstrip("-")
        try:
            instrument = self.client.get_instrument(self.inst_id)
            ct_val = float(instrument["ctVal"])
            notional_usd = abs(float(live_pos["pos"])) * ct_val * avg_px
        except Exception as exc:  # noqa: BLE001 -- notional degrade (PnL USDT approx), jamais bloquant
            logger.warning(f"get_instrument() a echoue en adoptant {pos_side} "
                           f"(notional_usd approxime a 0) : {exc}")
            notional_usd = 0.0
        self._finalize_entry(pos_side, avg_px, decision_ts_ms=now_ms, traded_ts_ms=now_ms,
                             regime_map=regime_map, notional_usd=notional_usd, sz=sz)
        logger.warning(f"{pos_side.upper()} ouverte chez OKX mais absente de l'etat local -- "
                       f"ADOPTEE (avg_px={avg_px}, sz={sz}) : surveillance SL/TP/trailing/"
                       f"timestop reprend a partir de maintenant.")
        return {"type": "adopted", "pos_side": pos_side, "avg_px": avg_px, "sz": sz}

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
    def try_widen(self, pos_side: str) -> bool:
        """A appeler par live.py quand un signal `pos_side` arrive alors que
        `slot_free(pos_side)` est False -- AVANT d'abandonner le signal comme
        "bloque". Si la position ouverte de ce sens est ACTUELLEMENT en profit
        (pp > 0, meme formule avgPx/markPx que evaluate_exit) et n'a jamais ete
        elargie, elargit UNE SEULE FOIS (WIDEN_ON_PROFIT_ENABLED) et renvoie
        True. False si desactive, aucune position ouverte, deja elargie, ou pas
        en profit -- le signal reste alors un simple "bloque, ignore"."""
        if not WIDEN_ON_PROFIT_ENABLED:
            return False
        entry = self.state.get(pos_side)
        if entry is None or entry.get("widened"):
            return False
        avg_px = float(entry["avg_px"])
        if self.client.dry_run:
            # BUG FIXE (2026-07-15) : meme raison que check_positions -- en
            # DRY_RUN, get_positions() ne montrera JAMAIS la position papier.
            ticker = self.client.get_ticker(self.inst_id)
            mark_px = float(ticker["last"])
        else:
            live_positions = {p["posSide"]: p for p in self.client.get_positions(self.inst_id)}
            live_pos = live_positions.get(pos_side)
            if live_pos is None:
                return False
            mark_px = float(live_pos.get("markPx") or live_pos.get("last") or avg_px)
        pp = (mark_px / avg_px - 1.0) if pos_side == "long" else (1.0 - mark_px / avg_px)
        if pp <= 0.0:
            return False
        entry["widened"] = True
        save_state(self.state)
        return True

    # -- Entree maker (non bloquant, avance d'un pas par appel) ---------------
    def start_entry(self, pos_side: str, sz: str, price_hint: float,
                    decision_ts_ms: int, traded_ts_ms: int,
                    regime_map: dict | None, maker: bool = MAKER,
                    notional_usd: float = 0.0) -> dict | None:
        """Demarre une tentative d'entree. Poste un ordre post-only (MAKER) ou
        market (TAKER) et enregistre l'etat 'pending_entry'. Ne bloque JAMAIS --
        `advance_pending_entry` fait progresser la tentative a chaque tick
        suivant (cf `live.py`). `notional_usd` (exposition USD demandee, cf
        live.py::run_hourly_cycle) est memorise sur l'entree pour convertir
        pp (%) en USDT dans les notifications de sortie (send_bet_resolved)."""
        if not self.slot_free(pos_side):
            return None
        side = "buy" if pos_side == "long" else "sell"
        if not maker:
            order = self.client.place_order(self.inst_id, side=side, pos_side=pos_side,
                                            ord_type="market", sz=sz,
                                            cl_ord_id=_new_cl_ord_id(pos_side))
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
                                 regime_map, notional_usd, sz)
            return {"type": "opened", "pos_side": pos_side, "mode": "taker", "order": order}

        px = f"{price_hint:.2f}"
        order = self.client.place_order(self.inst_id, side=side, pos_side=pos_side,
                                        ord_type="post_only", sz=sz, px=px,
                                        cl_ord_id=_new_cl_ord_id(pos_side))
        self.state["pending_entry"] = {
            "pos_side": pos_side, "sz": sz, "px": price_hint,
            "ord_id": order.get("ordId", ""), "decision_ts_ms": decision_ts_ms,
            "traded_ts_ms": traded_ts_ms, "posted_at_ms": traded_ts_ms,
            "last_reprice_ms": traded_ts_ms, "notional_usd": notional_usd,
            "deadline_ms": traded_ts_ms + MAKER_FILL_WINDOW_MIN * 60_000,
        }
        save_state(self.state)
        return {"type": "entry_posted", "pos_side": pos_side, "mode": "maker", "order": order}

    def advance_pending_entry(self, now_ms: int, regime_map: dict | None) -> dict | None:
        """A appeler CHAQUE tick. Fait progresser l'entree maker en cours (s'il
        y en a une) : verifie le fill, repost au prix courant toutes les
        ENTRY_REPRICE_EVERY_S secondes, bascule en taker au marche si la
        fenetre MAKER_FILL_WINDOW_MIN est ecoulee sans fill (= opportunite
        ratee en mode maker pur, mais on NE LAISSE JAMAIS un signal valide sans
        position -- contrairement au backtest qui compte juste le "missed")."""
        pending = self.state.get("pending_entry")
        if pending is None:
            return None
        pos_side = pending["pos_side"]
        ord_id = pending["ord_id"]

        try:
            if self.client.dry_run:
                # BUG FIXE (2026-07-15, audit complet) : get_order() en DRY_RUN
                # repond TOUJOURS "live" (jamais "filled", cf api_okx.py) -- une
                # entree maker papier tombait donc TOUJOURS en fallback taker
                # apres le delai, rendant le taux de fill maker observe en
                # DRY_RUN totalement non-representatif (0% de fill maker, quel
                # que soit le marche -- exactement le pattern "(maker_fallback_
                # taker)" vu sur 100% des ouvertures dans les logs Telegram).
                # Simule un fill realiste : un ordre post_only BUY (LONG) remplit
                # quand le marche descend AU NIVEAU ou EN DESSOUS du prix poste
                # (ordre passif attendant d'etre touche) ; SELL (SHORT) l'inverse.
                ticker = self.client.get_ticker(self.inst_id)
                last = float(ticker["last"])
                posted_px = float(pending["px"])
                filled = (last <= posted_px) if pos_side == "long" else (last >= posted_px)
                order = {"state": "filled", "avgPx": posted_px} if filled else {"state": "live"}
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
                                 pending.get("notional_usd", 0.0), pending["sz"])
            return {"type": "opened", "pos_side": pos_side, "mode": "maker", "avg_px": avg_px}

        if now_ms >= pending["deadline_ms"]:
            # Fenetre ecoulee, pas (ou seulement partiellement) rempli -> annule
            # le reliquat et bascule taker au marche pour SEULEMENT ce qui
            # manque. BUG FIXE (2026-08-03, audit fill partiel) : avant ce fix,
            # un post-only PARTIELLEMENT rempli (ex. 0.3/0.8 contrat) etait
            # quand meme suivi d'un market order pour pending["sz"] PLEIN (0.8)
            # -- position reelle = 0.3 (maker) + 0.8 (taker) = 1.1 au lieu de
            # 0.8 (sur-exposition ; pnl_usdt sous-estime, calcule sur la taille
            # voulue et pas la taille reelle). On relit l'etat FINAL de l'ordre
            # (le cancel peut echouer si l'ordre vient de finir de se remplir
            # entre notre dernier `get_order` et cette requete -- race) pour
            # calculer le reliquat EXACT a market-er.
            try:
                self.client.cancel_order(self.inst_id, ord_id)
            except Exception:  # noqa: BLE001 -- l'ordre peut deja etre annule/expire/filled cote OKX
                pass
            try:
                final_order = self.client.get_order(self.inst_id, ord_id)
            except Exception as exc:  # noqa: BLE001 -- lecture degradee, retombe sur le dernier etat connu
                logger.warning(f"get_order() post-cancel a echoue pour {pos_side} "
                               f"(repli sur le dernier etat connu) : {exc}")
                final_order = order

            if final_order.get("state") == "filled":
                # Rempli integralement pendant la fenetre d'annulation (race) --
                # aucun taker necessaire, traite comme un fill maker normal.
                avg_px = float(final_order.get("avgPx") or pending["px"])
                self.state["pending_entry"] = None
                self._finalize_entry(pos_side, avg_px, pending["decision_ts_ms"],
                                     pending["traded_ts_ms"], regime_map,
                                     pending.get("notional_usd", 0.0), pending["sz"])
                return {"type": "opened", "pos_side": pos_side, "mode": "maker", "avg_px": avg_px}

            requested_sz = float(pending["sz"])
            filled_sz = float(final_order.get("accFillSz") or 0.0)
            remaining_sz = round(requested_sz - filled_sz, 8)

            if remaining_sz <= 0:
                # Deja entierement rempli (a l'arrondi pres) sans que l'API ne
                # l'ait encore marque "filled" -- pas de taker a poster.
                avg_px = float(final_order.get("avgPx") or pending["px"])
                self.state["pending_entry"] = None
                self._finalize_entry(pos_side, avg_px, pending["decision_ts_ms"],
                                     pending["traded_ts_ms"], regime_map,
                                     pending.get("notional_usd", 0.0),
                                     f"{filled_sz:.8f}".rstrip("0").rstrip("."))
                return {"type": "opened", "pos_side": pos_side, "mode": "maker", "avg_px": avg_px}

            sz_for_taker = f"{remaining_sz:.8f}".rstrip("0").rstrip(".")
            try:
                order_taker = self.client.place_order(
                    self.inst_id, side=("buy" if pos_side == "long" else "sell"),
                    pos_side=pos_side, ord_type="market", sz=sz_for_taker,
                    cl_ord_id=_new_cl_ord_id(pos_side))
            except Exception as exc:  # noqa: BLE001
                # BUG FIXE (2026-08-03) : un rejet du taker (ex. reliquat trop
                # petit) ne doit pas laisser pending_entry bloque pour toujours
                # (le prochain tick retenterait cancel_order -- no-op -- puis le
                # meme place_order voue au meme rejet, indefiniment). On accepte
                # la portion deja remplie (filled_sz, potentiellement 0) comme
                # position finale plutot que de crash-looper.
                logger.warning(f"Fallback taker ({sz_for_taker}) echoue pour {pos_side} "
                               f"apres fill partiel {filled_sz} : {exc}")
                self.state["pending_entry"] = None
                if filled_sz <= 0:
                    return {"type": "entry_abandoned", "pos_side": pos_side, "error": str(exc)}
                avg_px = float(final_order.get("avgPx") or pending["px"])
                self._finalize_entry(pos_side, avg_px, pending["decision_ts_ms"],
                                     pending["traded_ts_ms"], regime_map,
                                     pending.get("notional_usd", 0.0),
                                     f"{filled_sz:.8f}".rstrip("0").rstrip("."))
                return {"type": "opened", "pos_side": pos_side, "mode": "maker_partial_only",
                        "avg_px": avg_px}

            fill_px = pending["px"]
            if not self.client.dry_run:
                # BUG FIXE (2026-07-15, audit complet) : meme raison que le
                # taker direct plus haut -- price posted (pending["px"]) n'est
                # pas le prix reellement rempli par ce fallback market.
                try:
                    confirmed = self.client.get_order(self.inst_id, order_taker.get("ordId", ""))
                    fill_px = float(confirmed.get("avgPx") or pending["px"])
                except Exception as exc:  # noqa: BLE001
                    # BUG FIXE (2026-07-15, audit erreurs silencieuses) : meme
                    # raison que le taker direct -- le repli sur pending["px"]
                    # partait sans trace. Comportement de repli inchange.
                    logger.warning(f"get_order() post-fill maker_fallback_taker a echoue pour "
                                   f"{pos_side} (repli sur px={pending['px']}) : {exc}")

            # Prix d'entree final = moyenne ponderee par taille entre le fill
            # maker partiel (partial_px, s'il y en a un) et le fill taker du
            # reliquat (fill_px) -- BUG FIXE (2026-08-03) : avant ce fix,
            # `fill_px` seul etait utilise comme si TOUT avait rempli au prix
            # taker, faussant pp/SL/TP des le premier tick suivant sur la
            # portion deja remplie a un (generalement meilleur) prix maker.
            total_sz = filled_sz + remaining_sz
            if filled_sz > 0:
                partial_px = float(final_order.get("avgPx") or pending["px"])
                avg_px = (filled_sz * partial_px + remaining_sz * fill_px) / total_sz
            else:
                avg_px = fill_px

            self.state["pending_entry"] = None
            self._finalize_entry(pos_side, avg_px, pending["decision_ts_ms"],
                                 pending["traded_ts_ms"], regime_map,
                                 pending.get("notional_usd", 0.0),
                                 f"{total_sz:.8f}".rstrip("0").rstrip("."))
            return {"type": "opened", "pos_side": pos_side, "mode": "maker_fallback_taker",
                    "order": order_taker, "partial_filled_sz": filled_sz}

        if now_ms - pending["last_reprice_ms"] >= ENTRY_REPRICE_EVERY_S * 1000:
            ticker = self.client.get_ticker(self.inst_id)
            new_px = float(ticker["bidPx"] if pos_side == "long" else ticker["askPx"])
            try:
                self.client.amend_order(self.inst_id, ord_id, new_px=f"{new_px:.2f}")
                pending["px"] = new_px
                pending["last_reprice_ms"] = now_ms
                save_state(self.state)
                return {"type": "entry_repriced", "pos_side": pos_side, "px": new_px}
            except Exception as exc:  # noqa: BLE001 -- reprice best-effort, retente au tick suivant
                logger.warning(f"Reprice {pos_side} echoue ({exc}), retente au prochain tick.")
        return None

    def _finalize_entry(self, pos_side: str, avg_px: float, decision_ts_ms: int,
                        traded_ts_ms: int, regime_map: dict | None,
                        notional_usd: float = 0.0, sz: str = "0") -> None:
        entry = {"avg_px": avg_px, "decision_ts_ms": decision_ts_ms,
                 "traded_ts_ms": traded_ts_ms, "peak_pp": -9.0, "worst_pp": 9.0,
                 "floor_pp": None, "widened": False, "notional_usd": notional_usd,
                 "sz": sz}
        if pos_side == "short":
            max_steps_eff, trail_gap_eff = filters.effective_down_exit_params(decision_ts_ms, regime_map)
            entry["down_max_steps_override"] = max_steps_eff
            entry["down_trail_gap_override"] = trail_gap_eff
        self.state[pos_side] = entry
        save_state(self.state)
