"""position.py — gestion des positions ouvertes du daemon live MEXC.

Ce module est le miroir LIVE de la logique de sortie du backtest
(`modele_team/main.py` : `_sl_threshold`, `_up_managed_exit`,
`_down_managed_exit`). La difference fondamentale : le backtest deroule toute la
bougie d'un coup (boucle interne sur les sous-bougies 1-min connues a l'avance),
alors qu'en live on recoit UN seul close 1-min a la fois. L'etat necessaire
(`peak`) est donc PERSISTE dans la `Position` entre deux appels.

Pieces exposees :
    - sl_threshold(...)           : copie EXACTE de _sl_threshold de main.py.
    - Position (dataclass)        : etat d'une position ouverte, serialisable JSON.
    - PositionManager             : check d'exit minute par minute, execution de la
                                    sortie (politique maker->taker), persistance
                                    JSON de l'etat, reconcile() au boot, log CSV.

Conventions dures (cf contrat) :
    - Tout en MILLISECONDES en interne.
    - EXPOSURE n'intervient JAMAIS dans les seuils de sortie ; les seuils SL
      (cfg.*_SL_*_DEC, cfg.*_STOP_LOSS_DEC) sont DEJA en decimal-prix (= % capital
      / EXPOSURE), calcules dans config_team. On compare `pp` (rendement prix
      directionnel) directement a ces seuils.
    - Fermeture hedge MEXC : up (long) -> side 4 (close long) ;
      down (short) -> side 2 (close short).

FICHIER CRITIQUE (argent reel) : priorite #1 = solidite. Toute erreur d'I/O
(persistance, CSV) est capturee et journalisee sans faire tomber la boucle ;
toute erreur API remonte au caller (qui retentera au prochain tick).
"""
from __future__ import annotations

import csv
import json
import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from live_helpers.logsetup import log_event

logger = logging.getLogger("live_team.position")

# Pas temporel des sous-bougies en live = 1 minute (comme DB_1m du backtest).
_MINUTE_MS = 60_000

# Colonnes du journal logs/positions.csv (ordre fige, cf contrat).
POSITIONS_COLUMNS = [
    "pos_id", "open_ts", "side", "entry", "vol", "leverage", "anchor_ts",
    "exit_ts", "exit_price", "exit_reason", "pp",
    "mfe_pp", "mae_pp", "mfe_usdc", "mae_usdc",
    "fees", "pnl_usdc",
    "pnl_pct_equity", "balance_post", "external_oid",
]


# ----------------------------------------------------------------------------
#  Seuil de stop-loss temporel — COPIE EXACTE de _sl_threshold (main.py)
# ----------------------------------------------------------------------------
def sl_threshold(j, max_steps, start_dec, end_dec, floor_dec):
    """SL TEMPOREL lineaire : seuil prix (decimal neg) de start_dec (large, j=0) vers
    end_dec (serre, j=max_steps), jamais plus serre (plus proche de 0) que floor_dec."""
    if max_steps <= 0:
        return start_dec
    s = start_dec + (end_dec - start_dec) * (j / max_steps)
    return min(s, floor_dec)


# ----------------------------------------------------------------------------
#  Etat d'une position ouverte (serialisable JSON)
# ----------------------------------------------------------------------------
@dataclass
class Position:
    """Position ouverte sur MEXC, telle que suivie par le daemon.

    side          : 'up' (long) ou 'down' (short).
    entry         : prix d'entree (open de la bougie i+1 / prix de fill).
    vol           : nombre de contrats (entier).
    anchor_ts     : open (ms) de la bougie 1h de DECISION ; sert de t0 pour le SL
                    temporel et le timestop (j = (now - anchor_ts) // 60000). C'est
                    aussi l'ancre du backtest (traded_ts = open de la bougie i+1,
                    mais la gestion compte les minutes depuis l'entree).
    peak          : meilleur profit directionnel observe depuis l'entree (PERSISTE
                    entre deux ticks ; init -9.0 comme le backtest).
    mexc_position_id : positionId MEXC (None tant qu'inconnu).
    external_oid  : identifiant d'ordre cote daemon (idempotence).
    open_ts       : timestamp (ms) d'ouverture effective de la position.
    status        : 'open' tant qu'active ; passe a 'closed' apres sortie.
    """

    pos_id: str
    side: str
    entry: float
    vol: int
    anchor_ts: int
    peak: float = -9.0
    # MFE / MAE de la position (excursions latentes en rendement-prix directionnel
    # pp, relatives a l'entree = pp 0). PUREMENT INFORMATIF (jamais lu par
    # check_exit) : `mfe` = plus haut profit atteint, `mae` = plus forte perte
    # atteinte pendant la vie de la position. Persistes (survivent au redemarrage).
    mfe: float = 0.0
    mae: float = 0.0
    mexc_position_id: int | None = None
    external_oid: str = ""
    open_ts: int = 0
    status: str = "open"
    # Plancher de trailing arme (None tant que le pic n'a pas atteint *_TRAIL_ACTIVATE).
    # Persiste, comme peak, pour repliquer la boucle backtest minute par minute.
    trail_floor: float | None = None
    leverage: int = 0
    # True si l'entree a ete remplie en maker (post-only) ; sert au calcul des
    # frais aller-retour dans execute_exit. L'entree live est post-only maker.
    entry_is_maker: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Position":
        """Reconstruit une Position depuis un dict JSON, tolerant aux champs
        absents (compat ascendante du fichier d'etat)."""
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        clean = {k: v for k, v in d.items() if k in known}
        return cls(**clean)


# ----------------------------------------------------------------------------
#  Gestionnaire de positions
# ----------------------------------------------------------------------------
class PositionManager:
    """Suit les positions ouvertes, decide/execute leur sortie, persiste l'etat.

    Parametres :
        mexc       : MexcClient (le seul a toucher l'API).
        cfg        : module config_team (constantes UP_/DOWN_ + frais + symbol...).
        notify     : module/objet notify (send_position_closed, ...) ; peut etre None.
        logger     : logger dedie (defaut : logger module).
        state_path : chemin du JSON d'etat (survie au redemarrage).
        fee_taker  : frais taker par cote (decimal) ; defaut cfg.TAKER_FEE.
        fee_maker  : frais maker par cote (decimal) ; defaut cfg.MAKER_FEE.

    Le journal CSV (logs/positions.csv) est ecrit a cote du state_path (dossier
    logs/) et protege par un lock (ecriture thread-safe).
    """

    def __init__(self, mexc, cfg, notify=None, logger: logging.Logger | None = None,
                 state_path: str | Path | None = None,
                 fee_taker: float | None = None, fee_maker: float | None = None):
        self.mexc = mexc
        self.cfg = cfg
        self.notify = notify
        self.log = logger or logging.getLogger("live_team.position")

        # Chemins : state JSON + CSV dans le meme dossier logs/.
        if state_path is None:
            state_path = Path(__file__).resolve().parent / "logs" / "positions_state.json"
        self.state_path = Path(state_path)
        self.csv_path = self.state_path.parent / "positions.csv"
        self.state_path.parent.mkdir(parents=True, exist_ok=True)

        self.fee_taker = float(fee_taker) if fee_taker is not None else float(getattr(cfg, "TAKER_FEE", 0.0005))
        self.fee_maker = float(fee_maker) if fee_maker is not None else float(getattr(cfg, "MAKER_FEE", 0.0001))

        # contract_size + price_unit, renseignes par live.py au boot (cache contrat).
        # Defaut prudent BTC_USDT si jamais non fourni.
        self.contract_size = float(getattr(cfg, "CONTRACT_SIZE", 0.0001) or 0.0001)
        self.price_unit = float(getattr(cfg, "PRICE_UNIT", 0.0) or 0.0)

        self._lock = threading.RLock()      # protege l'etat ET le CSV
        self.positions: dict[str, Position] = {}
        # Open (ms) de la derniere bougie 1-min DEJA traitee par
        # process_min1_bars. None = aucun traitement encore -> au 1er appel on ne
        # rejoue PAS tout le buffer historique (cf. process_min1_bars).
        self._last_min1_open_ms: int | None = None
        self._load_state()

    # ------------------------------------------------------------------
    #  Persistance JSON de l'etat
    # ------------------------------------------------------------------
    def _load_state(self) -> None:
        """Charge l'etat persiste (positions ouvertes) au demarrage."""
        if not self.state_path.exists():
            return
        try:
            raw = json.loads(self.state_path.read_text(encoding="utf-8"))
        except Exception as e:                              # noqa: BLE001
            self.log.error("Lecture etat positions impossible (%s) : %s — on repart vide.",
                           self.state_path, e)
            return
        items = raw.get("positions", raw) if isinstance(raw, dict) else raw
        if isinstance(items, dict):
            items = items.values()
        loaded = 0
        for d in items or []:
            try:
                pos = Position.from_dict(d)
                if pos.status == "open":
                    self.positions[pos.pos_id] = pos
                    loaded += 1
            except Exception as e:                          # noqa: BLE001
                self.log.error("Position d'etat ignoree (corrompue) : %s — %s", d, e)
        # Curseur des bougies 1-min deja traitees : restaure pour qu'un REDEMARRAGE
        # rejoue les minutes manquees (SL/timestop) au lieu d'un cold-start qui
        # ne traiterait que la derniere bougie.
        if isinstance(raw, dict):
            lm = raw.get("last_min1_open_ms")
            if isinstance(lm, int):
                self._last_min1_open_ms = lm
        if loaded:
            self.log.info("Etat positions charge : %d position(s) ouverte(s).", loaded)

    def _save_state(self) -> None:
        """Persiste l'etat courant (ecriture atomique : tmp puis replace)."""
        try:
            payload = {"positions": [p.to_dict() for p in self.positions.values()],
                       "last_min1_open_ms": self._last_min1_open_ms,
                       "saved_at": int(time.time() * 1000)}
            tmp = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
            tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            tmp.replace(self.state_path)
            self.log.debug("Etat persiste : %d position(s), last_min1_open_ms=%s.",
                           len(self.positions), self._last_min1_open_ms)
        except Exception as e:                              # noqa: BLE001
            # Solidite : un echec de persistance ne doit jamais tuer la boucle.
            self.log.error("Sauvegarde etat positions impossible : %s", e)

    # ------------------------------------------------------------------
    #  Parametres de sortie par sens (lus dans config_team)
    # ------------------------------------------------------------------
    def _exit_params(self, side: str) -> dict[str, Any]:
        """Regroupe les constantes de gestion du bon sens (prefixe UP_/DOWN_).

        Toutes les valeurs SL (*_SL_*_DEC, *_STOP_LOSS_DEC) sont DEJA en decimal
        prix (= % capital / EXPOSURE, calcule dans config_team) -> on les compare
        directement a pp, EXPOSURE n'entre pas en jeu.
        """
        cfg = self.cfg
        p = "UP" if side == "up" else "DOWN"
        return {
            "enabled": bool(getattr(cfg, f"{p}_MGMT_ENABLED", True)),
            "max_steps": int(getattr(cfg, f"{p}_MAX_STEPS")),
            "trail_activate": float(getattr(cfg, f"{p}_TRAIL_ACTIVATE")),
            "trail_gap": float(getattr(cfg, f"{p}_TRAIL_GAP")),
            "take_profit": float(getattr(cfg, f"{p}_TAKE_PROFIT")),
            "sl_time_decay": bool(getattr(cfg, f"{p}_SL_TIME_DECAY")),
            "sl_start_dec": float(getattr(cfg, f"{p}_SL_START_DEC")),
            "sl_end_dec": float(getattr(cfg, f"{p}_SL_END_DEC")),
            "sl_floor_dec": float(getattr(cfg, f"{p}_SL_FLOOR_DEC")),
            "stop_loss_dec": float(getattr(cfg, f"{p}_STOP_LOSS_DEC")),
        }

    # ------------------------------------------------------------------
    #  Decision de sortie sur UN close (replique _up/_down_managed_exit)
    # ------------------------------------------------------------------
    def check_exit(self, pos: Position, close_price: float, now_ms: int) -> tuple[str | None, float]:
        """Replique l'ordre SL > TP > trailing > timestop du backtest, mais sur
        UN seul close 1-min a la fois. L'etat (peak / trail_floor) persiste dans
        la Position pour reconstituer la boucle minute par minute.

        Retourne (reason, pp) :
            reason in {"stop_loss","take_profit","trailing","timestop", None}
            pp = rendement directionnel prix a ce close.
            reason is None tant qu'aucun seuil n'est franchi et qu'on n'a pas
            atteint le timestop (la position reste ouverte).

        Calque de _up_managed_exit / _down_managed_exit :
            up   : pp = close/entry - 1
            down : pp = 1 - close/entry  (profit directionnel d'un short)
        L'index temporel j (en pas de 1 min) = (now_ms - anchor_ts) // 60000, borne
        a max_steps-1 pour le SL temporel ; le timestop se declenche a j>=max_steps-1
        (derniere minute autorisee), comme la boucle `for j in range(MAX_STEPS)`
        du backtest dont la derniere iteration retourne "timestop".
        """
        params = self._exit_params(pos.side)
        max_steps = params["max_steps"]

        if pos.entry <= 0:
            return None, 0.0
        if pos.side == "up":
            pp = close_price / pos.entry - 1.0
        else:
            pp = 1.0 - close_price / pos.entry

        # MFE / MAE : suivi des excursions latentes de la position. MAJ AVANT tout
        # `return` ci-dessous pour capter aussi la bougie de sortie (SL/TP/timestop).
        # Purement informatif (jamais relu par la decision de sortie) -> aucun
        # impact sur la parite backtest.
        if pp > pos.mfe:
            pos.mfe = pp
        if pp < pos.mae:
            pos.mae = pp

        # j = nombre de minutes ecoulees depuis l'ancre (open bougie 1h de decision).
        j = int((now_ms - pos.anchor_ts) // _MINUTE_MS)
        if j < 0:
            j = 0
        # j_eff borne pour l'evaluation du SL temporel (le backtest n'itere jamais
        # au-dela de max_steps-1).
        j_eff = min(j, max_steps - 1) if max_steps > 0 else 0

        # --- 1) STOP-LOSS (priorite max) ---
        if params["sl_time_decay"]:
            sl_th = sl_threshold(j_eff, max_steps, params["sl_start_dec"],
                                 params["sl_end_dec"], params["sl_floor_dec"])
        else:
            sl_th = params["stop_loss_dec"]

        # Trace exhaustive de l'evaluation minute-par-minute (fichier DEBUG).
        self.log.debug(
            "check_exit %s [%s] close=%.2f entry=%.2f pp=%+.4f%% j=%d/%d "
            "sl_th=%+.4f%% tp=%+.4f%% peak=%+.4f%% trail_floor=%s",
            pos.pos_id, pos.side, close_price, pos.entry, pp * 100, j, max_steps,
            sl_th * 100, params["take_profit"] * 100, pos.peak * 100,
            f"{pos.trail_floor * 100:+.4f}%" if pos.trail_floor is not None else "None")

        if pp <= sl_th:
            self.log.info("check_exit %s -> STOP_LOSS (pp=%+.4f%% <= sl_th=%+.4f%%, j=%d).",
                          pos.pos_id, pp * 100, sl_th * 100, j)
            return "stop_loss", pp

        # --- 2) TAKE-PROFIT ---
        if pp >= params["take_profit"]:
            self.log.info("check_exit %s -> TAKE_PROFIT (pp=%+.4f%% >= tp=%+.4f%%).",
                          pos.pos_id, pp * 100, params["take_profit"] * 100)
            return "take_profit", pp

        # --- 3) TRAILING ('let-run') : maj du pic puis du plancher arme ---
        if pp > pos.peak:
            pos.peak = pp
        if pos.peak >= params["trail_activate"]:
            pos.trail_floor = pos.peak - params["trail_gap"]
        if pos.trail_floor is not None and pp < pos.trail_floor:
            self.log.info("check_exit %s -> TRAILING (pp=%+.4f%% < floor=%+.4f%%, "
                          "peak=%+.4f%%).", pos.pos_id, pp * 100,
                          pos.trail_floor * 100, pos.peak * 100)
            return "trailing", pp

        # --- 4) TIMESTOP : detention max atteinte -> sortie a la cloture ---
        if max_steps > 0 and j >= max_steps - 1:
            self.log.info("check_exit %s -> TIMESTOP (j=%d >= max_steps-1=%d, pp=%+.4f%%).",
                          pos.pos_id, j, max_steps - 1, pp * 100)
            return "timestop", pp

        return None, pp

    # ------------------------------------------------------------------
    #  Execution de la sortie (politique maker -> taker)
    # ------------------------------------------------------------------
    def _close_side(self, side: str) -> int:
        """side MEXC de FERMETURE (mode hedge) : up->4 (close long), down->2 (close short)."""
        return 4 if side == "up" else 2

    def _round_price(self, price: float) -> float | None:
        """Arrondit un prix au price_unit du contrat (si connu). None si invalide."""
        if price is None or price <= 0:
            return None
        if self.price_unit and self.price_unit > 0:
            steps = round(price / self.price_unit)
            return round(steps * self.price_unit, 12)
        return price

    def execute_exit(self, pos: Position, reason: str, pp: float, market_price: float) -> dict:
        """Ferme la position selon la politique maker->taker.

        Politique :
            - pp > 0 (sortie gagnante : take_profit / trailing positif / timestop
              positif) -> on tente jusqu'a EXIT_MAKER_ATTEMPTS ordres POST-ONLY
              reduce-only legerement passifs (offset EXIT_MAKER_OFFSET du bon cote),
              chacun laisse EXIT_MAKER_WAIT_SEC pour se remplir. Si toujours pas
              rempli -> bascule MARKET reduce-only (taker).
            - pp <= 0 (perte / stop_loss) -> MARKET reduce-only direct (taker), on
              ne flane pas sur une position perdante.

        Cote de fermeture (hedge) : up->4, down->2.
        Lit le fill (get_order puis get_open_positions en repli), calcule le pnl,
        ecrit logs/positions.csv, notifie, retire la position de l'etat.

        Retourne un dict de synthese (toujours, meme en cas d'erreur partielle) :
            {pos_id, side, reason, pp, exit_price, vol_closed, fee_side, fees,
             pnl_usdc, balance_post, order_id, closed}
        """
        cfg = self.cfg
        symbol = getattr(cfg, "SYMBOL", "BTC_USDT")
        close_side = self._close_side(pos.side)
        vol = int(pos.vol)
        exit_price = float(market_price) if market_price else float(pos.entry)
        fee_side = self.fee_taker          # par defaut taker ; maker si fill maker
        order_id: str | None = None
        filled = False

        maker_offset = float(getattr(cfg, "EXIT_MAKER_OFFSET", 0.0001))
        maker_attempts = int(getattr(cfg, "EXIT_MAKER_ATTEMPTS", 3))
        maker_wait = float(getattr(cfg, "EXIT_MAKER_WAIT_SEC", 20))

        self.log.info("execute_exit %s [%s] reason=%s pp=%+.4f%% market=%.2f "
                      "-> politique=%s (close_side=%d, vol=%d).",
                      pos.pos_id, pos.side, reason, pp * 100, exit_price,
                      "maker->taker" if pp > 0 else "taker_direct", close_side, vol)
        log_event("exit_start", pos_id=pos.pos_id, side=pos.side, reason=reason,
                  pp=round(pp, 6), market=round(exit_price, 2), vol=vol,
                  policy="maker_first" if pp > 0 else "taker_direct")

        try:
            if pp > 0:
                # --- Sortie gagnante : maker passif d'abord (economie de frais) ---
                for attempt in range(maker_attempts):
                    ticker = self._safe_market_price(symbol, exit_price)
                    # Limite passive : pour fermer un long on VEND -> on poste au-dessus
                    # du prix (le marche doit monter nous chercher) ; pour fermer un
                    # short on RACHETE -> on poste en dessous. Post-only => maker garanti.
                    if pos.side == "up":
                        limit = ticker * (1.0 + maker_offset)
                    else:
                        limit = ticker * (1.0 - maker_offset)
                    limit_r = self._round_price(limit) or limit
                    res = self.mexc.place_order(
                        symbol=symbol, side=close_side, vol=vol, price=limit_r,
                        order_type=2,  # post-only (maker garanti)
                        open_type=int(getattr(cfg, "OPEN_TYPE", 1)),
                        reduce_only=True,
                        external_oid=f"{pos.external_oid}-exit-m{attempt}",
                        position_id=pos.mexc_position_id,
                    )
                    order_id = str(res.get("order_id"))
                    self.log.debug("exit %s : maker post-only #%d @ %.2f (oid=%s).",
                                   pos.pos_id, attempt + 1, limit_r, order_id)
                    if res.get("dry_run"):
                        # En DRY_RUN aucun fill reel : on considere la sortie maker
                        # comme effective au prix limite (pas de POST).
                        exit_price = limit_r
                        fee_side = self.fee_maker
                        filled = True
                        break
                    deal = self._poll_fill(order_id, vol, maker_wait)
                    if deal is not None:
                        exit_price = deal[0] or limit_r
                        fee_side = self.fee_maker
                        filled = True
                        self.log.info("exit %s : fill MAKER @ %.2f a l'essai #%d.",
                                      pos.pos_id, exit_price, attempt + 1)
                        break
                    # Pas rempli dans la fenetre -> annuler avant le prochain essai.
                    self.log.debug("exit %s : maker #%d non rempli en %.0fs -> annule.",
                                   pos.pos_id, attempt + 1, maker_wait)
                    self._safe_cancel(order_id)
                if not filled:
                    # Repli taker : market reduce-only.
                    self.log.warning("exit %s : %d essais maker sans fill -> repli TAKER (market).",
                                     pos.pos_id, maker_attempts)
                    filled, exit_price, order_id = self._market_close(
                        symbol, close_side, vol, pos, exit_price)
                    fee_side = self.fee_taker
            else:
                # --- Sortie perdante / SL : market direct (taker) ---
                self.log.info("exit %s : sortie perdante/SL -> market TAKER direct.",
                              pos.pos_id)
                filled, exit_price, order_id = self._market_close(
                    symbol, close_side, vol, pos, exit_price)
                fee_side = self.fee_taker
        except Exception as e:                              # noqa: BLE001
            # Solidite : on journalise, on NE retire PAS la position (sera retentee
            # au prochain on_minute). Le caller voit closed=False.
            self.log.exception("execute_exit %s a echoue : %s", pos.pos_id, e)
            log_event("exit_error", pos_id=pos.pos_id, side=pos.side,
                      reason=reason, error=str(e))
            return {"pos_id": pos.pos_id, "side": pos.side, "reason": reason,
                    "pp": pp, "exit_price": exit_price, "vol_closed": 0,
                    "fee_side": fee_side, "fees": 0.0, "pnl_usdc": 0.0,
                    "balance_post": None, "order_id": order_id, "closed": False}

        # --- Recalage du pp sur le prix de sortie reellement obtenu ---
        if pos.entry > 0 and exit_price > 0:
            pp = (exit_price / pos.entry - 1.0) if pos.side == "up" else (1.0 - exit_price / pos.entry)

        # --- P/L en USDC (notionnel = vol * contract_size * prix) ---
        notional_entry = vol * self.contract_size * pos.entry
        # Frais aller-retour : entree (maker si fill post-only, sinon taker) +
        # sortie (fee_side selon le mode de fill de la sortie).
        fee_entry = self.fee_maker if pos.entry_is_maker else self.fee_taker
        fees = notional_entry * (fee_entry + fee_side)
        # P/L brut directionnel = pp * notionnel d'entree (pp est deja le rendement
        # prix du bon signe : positif si gagnant). pnl net = brut - frais.
        pnl_gross = pp * notional_entry
        pnl_usdc = pnl_gross - fees
        self.log.debug("exit %s : pnl breakdown notional_entry=%.2f pp=%+.4f%% "
                       "gross=%+.4f fees=%.4f (entry_maker=%s, exit_%s) net=%+.4f USDC",
                       pos.pos_id, notional_entry, pp * 100, pnl_gross, fees,
                       pos.entry_is_maker,
                       "maker" if fee_side == self.fee_maker else "taker", pnl_usdc)

        # --- Solde apres cloture + P/L en % de l'equity (best-effort) ---
        balance_post = None
        pnl_pct_equity = None
        try:
            assets = self.mexc.get_assets("USDT")
            balance_post = float(assets.get("available_balance"))
            equity = float(assets.get("equity") or balance_post)
            if equity:
                pnl_pct_equity = pnl_usdc / equity * 100.0
        except Exception as e:                              # noqa: BLE001
            self.log.warning("get_assets apres cloture indisponible : %s", e)

        now_ms = int(time.time() * 1000)
        pos.status = "closed"

        # --- Excursions latentes (MFE/MAE) en USDC brut (= pp_excursion x notionnel
        #     d'entree, hors frais ; hypothetique « si on avait clos a ce point »). ---
        mfe_usdc = pos.mfe * notional_entry
        mae_usdc = pos.mae * notional_entry

        # --- Journalisation CSV + notification + retrait de l'etat ---
        self._append_csv_row({
            "pos_id": pos.pos_id,
            "open_ts": pos.open_ts,
            "side": pos.side,
            "entry": pos.entry,
            "vol": vol,
            "leverage": pos.leverage or getattr(cfg, "MEXC_LEVERAGE", ""),
            "anchor_ts": pos.anchor_ts,
            "exit_ts": now_ms,
            "exit_price": round(exit_price, 6),
            "exit_reason": reason,
            "pp": round(pp, 8),
            "mfe_pp": round(pos.mfe, 8),
            "mae_pp": round(pos.mae, 8),
            "mfe_usdc": round(mfe_usdc, 6),
            "mae_usdc": round(mae_usdc, 6),
            "fees": round(fees, 6),
            "pnl_usdc": round(pnl_usdc, 6),
            "pnl_pct_equity": round(pnl_pct_equity, 6) if pnl_pct_equity is not None else "",
            "balance_post": round(balance_post, 6) if balance_post is not None else "",
            "external_oid": pos.external_oid,
        })

        self._notify_closed(pos, reason, exit_price, pp, pnl_usdc, balance_post,
                            mfe_usdc, mae_usdc)

        with self._lock:
            self.positions.pop(pos.pos_id, None)
            self._save_state()

        self.log.info("Position %s fermee (%s) pp=%.4f%% pnl=%.2f USDC fill=%s",
                      pos.pos_id, reason, pp * 100, pnl_usdc,
                      "maker" if fee_side == self.fee_maker else "taker")
        log_event("position_closed", pos_id=pos.pos_id, side=pos.side,
                  reason=reason, entry=round(pos.entry, 2),
                  exit_price=round(exit_price, 2), pp=round(pp, 6),
                  mfe_pp=round(pos.mfe, 6), mae_pp=round(pos.mae, 6),
                  mfe_usdc=round(mfe_usdc, 4), mae_usdc=round(mae_usdc, 4),
                  vol=vol, fees=round(fees, 4), pnl_usdc=round(pnl_usdc, 4),
                  fill="maker" if fee_side == self.fee_maker else "taker",
                  balance_post=round(balance_post, 2) if balance_post is not None else None,
                  order_id=order_id)

        return {"pos_id": pos.pos_id, "side": pos.side, "reason": reason, "pp": pp,
                "exit_price": exit_price, "vol_closed": vol, "fee_side": fee_side,
                "fees": fees, "pnl_usdc": pnl_usdc, "balance_post": balance_post,
                "order_id": order_id, "closed": True}

    # ------------------------------------------------------------------
    #  Helpers d'execution (market close / poll fill / cancel / prix)
    # ------------------------------------------------------------------
    def _market_close(self, symbol: str, close_side: int, vol: int, pos: Position,
                      fallback_price: float) -> tuple[bool, float, str | None]:
        """Ferme au MARCHE (taker) reduce-only. Retourne (filled, exit_price, order_id)."""
        res = self.mexc.place_order(
            symbol=symbol, side=close_side, vol=vol, price=None,
            order_type=5,  # market
            open_type=int(getattr(self.cfg, "OPEN_TYPE", 1)),
            reduce_only=True,
            external_oid=f"{pos.external_oid}-exit-mkt",
            position_id=pos.mexc_position_id,
        )
        order_id = str(res.get("order_id"))
        if res.get("dry_run"):
            return True, fallback_price, order_id
        # Le market se remplit quasi-instantanement ; on lit le prix moyen reel.
        deal = self._poll_fill(order_id, vol, timeout_sec=5.0)
        if deal is not None and deal[0]:
            return True, deal[0], order_id
        # Repli : prix marche courant (le fill a eu lieu mais get_order tarde).
        return True, self._safe_market_price(symbol, fallback_price), order_id

    def _poll_fill(self, order_id: str, vol: int, timeout_sec: float) -> tuple[float, float] | None:
        """Sonde get_order jusqu'a remplissage (state=3) ou timeout.

        Retourne (deal_avg_price, deal_vol) si rempli (meme partiellement), None
        si toujours en cours / annule a l'expiration. Robuste aux erreurs API
        transitoires (on continue de sonder dans la fenetre)."""
        if not order_id or order_id.startswith("DRYRUN-"):
            return None
        deadline = time.time() + max(0.0, timeout_sec)
        poll_interval = 1.0
        last: tuple[float, float] | None = None
        while True:
            try:
                od = self.mexc.get_order(order_id)
                state = int(od.get("state", 0))
                deal_vol = float(od.get("deal_vol", 0.0))
                deal_px = float(od.get("deal_avg_price", 0.0))
                if deal_vol > 0:
                    last = (deal_px, deal_vol)
                if state == 3:                  # completed
                    return (deal_px, deal_vol)
                if state == 4:                  # cancelled
                    return last if (last and last[1] > 0) else None
            except Exception as e:              # noqa: BLE001
                self.log.warning("get_order(%s) pendant le poll : %s", order_id, e)
            if time.time() >= deadline:
                # Timeout : retourne un fill partiel s'il y en a un, sinon None.
                return last if (last and last[1] > 0) else None
            time.sleep(poll_interval)

    def _safe_cancel(self, order_id: str | None) -> None:
        """Annule un ordre, en ignorant l'erreur (l'ordre a pu se remplir/expirer)."""
        if not order_id or order_id.startswith("DRYRUN-"):
            return
        try:
            self.mexc.cancel_orders([order_id])
        except Exception as e:                              # noqa: BLE001
            self.log.warning("Annulation ordre %s impossible : %s", order_id, e)

    def _safe_market_price(self, symbol: str, fallback: float) -> float:
        """Dernier prix marche (ticker) ; fallback si l'API echoue."""
        try:
            t = self.mexc.get_ticker(symbol)
            px = float(t.get("last_price") or t.get("fair_price") or 0.0)
            if px > 0:
                return px
        except Exception as e:                              # noqa: BLE001
            self.log.warning("get_ticker(%s) indisponible : %s", symbol, e)
        return float(fallback)

    def _notify_closed(self, pos: Position, reason: str, exit_price: float, pp: float,
                       pnl_usdc: float, balance_post: float | None,
                       mfe_usdc: float = 0.0, mae_usdc: float = 0.0) -> None:
        """Envoie la notification de cloture (best-effort, jamais bloquant)."""
        if self.notify is None:
            return
        send = getattr(self.notify, "send_position_closed", None)
        if send is None:
            return
        try:
            send(side=pos.side, exit_reason=reason, entry=pos.entry,
                 exit_price=exit_price, pp=pp, pnl_usdc=pnl_usdc,
                 balance=balance_post if balance_post is not None else 0.0,
                 mfe_pp=pos.mfe, mae_pp=pos.mae,
                 mfe_usdc=mfe_usdc, mae_usdc=mae_usdc)
        except Exception as e:                              # noqa: BLE001
            self.log.warning("notify.send_position_closed a echoue : %s", e)

    # ------------------------------------------------------------------
    #  Boucle minute : check + execution pour chaque position ouverte
    # ------------------------------------------------------------------
    def on_minute(self, close_min1: float, now_ms: int) -> list[dict]:
        """Pour chaque position ouverte : check_exit sur le close 1-min qui vient
        de fermer ; si un motif de sortie est arme, execute_exit immediatement.

        Retourne la liste des dict de synthese des positions effectivement
        traitees (utile au logging de la boucle live)."""
        results: list[dict] = []
        # Copie de la liste : execute_exit mute self.positions.
        for pos in list(self.positions.values()):
            if pos.status != "open":
                continue
            try:
                reason, pp = self.check_exit(pos, float(close_min1), int(now_ms))
            except Exception as e:                          # noqa: BLE001
                self.log.exception("check_exit %s a echoue : %s", pos.pos_id, e)
                continue
            # peak/trail_floor ont pu changer -> persiste meme sans sortie.
            self._save_state()
            if reason is not None:
                res = self.execute_exit(pos, reason, pp, market_price=float(close_min1))
                results.append(res)
        return results

    def process_min1_bars(self, bars: list[list]) -> list[dict]:
        """Rejoue TOUTES les bougies 1-min fermees depuis le dernier traitement.

        Remplace l'evaluation du seul dernier close (on_minute) : si une minute a
        saute (redemarrage, stall reseau, kline vide), le SL/TP qui aurait du se
        declencher sur une bougie intermediaire est tout de meme evalue (en
        differe mais sur le BON close).

        `bars` = liste ASC de bougies `[open_ms, o, h, l, c, v]` (format MEXC
        get_klines, deja en ms). On ne doit recevoir que des bougies REELLEMENT
        FERMEES (le caller exclut la bougie en formation).

        Alignement backtest : pour chaque bougie rejouee on appelle
        `check_exit(pos, close=bar[4], now_ms=bar_open_ms)` -> le OPEN de la
        bougie sert d'horloge pour que j = (open - anchor_ts)//60000 corresponde
        EXACTEMENT a l'index `for j in range(MAX_STEPS): cl = close_ib[traded_ts
        + j*60000]` du backtest (la sous-bougie d'index j a pour open
        anchor_ts + j*60000 et on utilise SON close).

        1er appel (`_last_min1_open_ms is None`) : on ne traite QUE la derniere
        bougie close (on n'a pas suivi l'historique avant le boot) ; _last est
        alors place a l'avant-derniere open pour ne pas rejouer tout le buffer.

        Retourne la liste des dict de synthese des positions effectivement
        fermees (utile au logging de la boucle live)."""
        results: list[dict] = []
        if not bars:
            return results

        n_open = sum(1 for p in self.positions.values() if p.status == "open")
        if n_open == 0:
            # Aucune position : rien a evaluer, on avance juste le curseur plus bas.
            self.log.debug("process_min1_bars : %d bougie(s), aucune position ouverte.",
                           len(bars))

        # Bougies triees ASC par open et dedupliquees (robustesse au format API).
        try:
            asc = sorted(bars, key=lambda b: int(b[0]))
        except Exception as e:                              # noqa: BLE001
            self.log.warning("process_min1_bars : bars illisibles (%s) — ignore.", e)
            return results

        if self._last_min1_open_ms is None:
            # Cold start : ne rejoue pas le buffer historique. On amorce le
            # curseur a l'avant-derniere bougie pour que SEULE la derniere close
            # soit traitee ci-dessous.
            if len(asc) >= 2:
                self._last_min1_open_ms = int(asc[-2][0])
            else:
                # Une seule bougie : on la traitera puis on cale le curseur juste
                # avant elle pour qu'elle soit prise une fois.
                self._last_min1_open_ms = int(asc[-1][0]) - _MINUTE_MS

        # Rejoue chaque bougie strictement posterieure au dernier open traite.
        for bar in asc:
            try:
                bar_open_ms = int(bar[0])
            except Exception:                               # noqa: BLE001
                continue
            if bar_open_ms <= self._last_min1_open_ms:
                continue
            try:
                close_px = float(bar[4])
            except Exception:                               # noqa: BLE001
                # Bougie corrompue : on avance quand meme le curseur pour ne pas
                # boucler dessus indefiniment.
                self._last_min1_open_ms = bar_open_ms
                continue

            # check_exit + execute_exit pour chaque position ouverte, sur CE close.
            open_now = [p for p in self.positions.values() if p.status == "open"]
            if open_now:
                self.log.debug("process_min1_bars : rejeu bougie open=%d close=%.2f "
                               "sur %d position(s).", bar_open_ms, close_px, len(open_now))
            for pos in open_now:
                if pos.status != "open":
                    continue
                try:
                    reason, pp = self.check_exit(pos, close_px, bar_open_ms)
                except Exception as e:                      # noqa: BLE001
                    self.log.exception("check_exit %s a echoue : %s", pos.pos_id, e)
                    continue
                if reason is not None:
                    res = self.execute_exit(pos, reason, pp, market_price=close_px)
                    results.append(res)

            # Curseur + persistance de l'etat (peak/trail_floor ont pu bouger)
            # apres chaque bougie traitee.
            self._last_min1_open_ms = bar_open_ms
            self._save_state()

        return results

    # ------------------------------------------------------------------
    #  Ouverture d'une position
    # ------------------------------------------------------------------
    def open_position(self, side: str, entry: float, vol: int, anchor_ts: int,
                      mexc_position_id: int | None, external_oid: str) -> Position:
        """Cree + persiste une nouvelle Position (apres fill de l'ordre d'entree),
        notifie l'ouverture. Renvoie la Position creee."""
        now_ms = int(time.time() * 1000)
        pos_id = external_oid or f"pos-{now_ms}"
        pos = Position(
            pos_id=pos_id,
            side=side,
            entry=float(entry),
            vol=int(vol),
            anchor_ts=int(anchor_ts),
            peak=-9.0,
            mexc_position_id=mexc_position_id,
            external_oid=external_oid or pos_id,
            open_ts=now_ms,
            status="open",
            trail_floor=None,
            leverage=int(getattr(self.cfg, "MEXC_LEVERAGE", 0) or 0),
        )
        with self._lock:
            self.positions[pos.pos_id] = pos
            self._save_state()

        notional = vol * self.contract_size * float(entry)
        self.log.info("Position %s ouverte : %s entry=%.2f vol=%d notional=%.2f",
                      pos.pos_id, side, entry, vol, notional)
        log_event("position_opened", pos_id=pos.pos_id, side=side,
                  entry=round(float(entry), 2), vol=int(vol),
                  notional=round(notional, 2), anchor_ts=int(anchor_ts),
                  mexc_position_id=mexc_position_id)
        if self.notify is not None:
            send = getattr(self.notify, "send_position_opened", None)
            if send is not None:
                try:
                    send(side=side, entry=float(entry), vol=int(vol), notional=notional)
                except Exception as e:                      # noqa: BLE001
                    self.log.warning("notify.send_position_opened a echoue : %s", e)
        return pos

    # ------------------------------------------------------------------
    #  Reconciliation au boot
    # ------------------------------------------------------------------
    def reconcile(self, symbol: str) -> None:
        """Aligne l'etat persiste avec les positions reellement ouvertes sur MEXC.

        Au boot :
          - toute position MEXC inconnue de l'etat est ADOPTEE (anchor_ts estime =
            open de l'heure d'ouverture courante, faute de mieux) ;
          - toute position de l'etat qui n'existe plus cote MEXC (fermee hors
            daemon, liquidee...) est retiree de l'etat.

        Best-effort : si l'API echoue, on garde l'etat persiste tel quel.
        """
        try:
            open_positions = self.mexc.get_open_positions(symbol)
        except Exception as e:                              # noqa: BLE001
            self.log.error("reconcile : get_open_positions a echoue (%s) — etat persiste conserve.", e)
            return

        # Index des positions MEXC vivantes (hold_vol > 0) par positionId.
        live_by_id: dict[int, dict] = {}
        for p in open_positions:
            if float(p.get("hold_vol", 0.0)) <= 0:
                continue
            pid = int(p.get("position_id", 0))
            if pid:
                live_by_id[pid] = p

        now_ms = int(time.time() * 1000)
        hour_open_ms = (now_ms // 3_600_000) * 3_600_000   # open de l'heure courante

        with self._lock:
            # 1) Retirer de l'etat les positions qui n'ont plus de pendant MEXC.
            known_ids = {p.mexc_position_id for p in self.positions.values()
                         if p.mexc_position_id}
            to_drop = []
            for pos_id, pos in self.positions.items():
                if pos.mexc_position_id and pos.mexc_position_id not in live_by_id:
                    self.log.warning("reconcile : position %s (mexc_id=%s) absente cote MEXC "
                                     "-> retiree de l'etat.", pos_id, pos.mexc_position_id)
                    log_event("reconcile_drop", pos_id=pos_id,
                              mexc_position_id=pos.mexc_position_id)
                    to_drop.append(pos_id)
            for pos_id in to_drop:
                self.positions.pop(pos_id, None)

            # 2) Adopter les positions MEXC inconnues du daemon.
            for pid, p in live_by_id.items():
                if pid in known_ids:
                    continue
                side = "up" if int(p.get("position_type", 0)) == 1 else "down"
                vol = int(round(float(p.get("hold_vol", 0.0))))
                entry = float(p.get("open_avg_price", 0.0))
                oid = f"adopted-{pid}"
                pos = Position(
                    pos_id=oid,
                    side=side,
                    entry=entry,
                    vol=vol,
                    anchor_ts=hour_open_ms,
                    peak=-9.0,
                    mexc_position_id=pid,
                    external_oid=oid,
                    open_ts=now_ms,
                    status="open",
                    trail_floor=None,
                    leverage=int(p.get("leverage", 0) or 0),
                )
                self.positions[oid] = pos
                self.log.warning("reconcile : position MEXC inconnue adoptee %s "
                                 "(side=%s vol=%d entry=%.2f anchor=heure courante).",
                                 oid, side, vol, entry)
                log_event("reconcile_adopt", pos_id=oid, side=side, vol=vol,
                          entry=round(entry, 2), mexc_position_id=pid)

            self._save_state()

        self.log.info("reconcile termine : %d position(s) ouverte(s) suivie(s).",
                      len(self.positions))
        log_event("reconcile_done", n_open=len(self.positions),
                  pos_ids=list(self.positions.keys()))

    # ------------------------------------------------------------------
    #  Divers
    # ------------------------------------------------------------------
    def n_open(self) -> int:
        """Nombre de positions actuellement ouvertes (status='open')."""
        with self._lock:
            return sum(1 for p in self.positions.values() if p.status == "open")

    def has_open_side(self, side: str) -> bool:
        """True si une position du sens `side` ('up'/'down') est deja ouverte.
        Sert a interdire une 2e entree du MEME sens : en mode hedge MEXC agrege
        tout le meme sens en une seule position, qu'on suivrait alors en double."""
        with self._lock:
            return any(p.side == side and p.status == "open"
                       for p in self.positions.values())

    # ------------------------------------------------------------------
    #  Ecriture CSV thread-safe (logs/positions.csv)
    # ------------------------------------------------------------------
    def _append_csv_row(self, row: dict[str, Any]) -> None:
        """Ajoute une ligne a logs/positions.csv (header ecrit si fichier neuf).
        Protege par lock ; toute erreur d'I/O est journalisee sans interrompre."""
        with self._lock:
            try:
                # Rotation de schema : si un positions.csv existe avec un en-tete
                # different (ajout de colonnes), on l'archive en .old et on repart
                # sur un fichier neuf au nouvel en-tete (comme cycles.csv.old).
                if self.csv_path.exists():
                    with self.csv_path.open("r", encoding="utf-8") as f:
                        header = f.readline().strip()
                    if header and header != ",".join(POSITIONS_COLUMNS):
                        old = self.csv_path.with_suffix(".csv.old")
                        self.csv_path.replace(old)
                        self.log.warning("positions.csv : schema change -> ancien "
                                         "fichier archive en %s, nouvel en-tete ecrit.", old.name)
                new_file = not self.csv_path.exists()
                with self.csv_path.open("a", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=POSITIONS_COLUMNS,
                                       extrasaction="ignore")
                    if new_file:
                        w.writeheader()
                    # Complete les colonnes manquantes par "" pour un CSV regulier.
                    w.writerow({c: row.get(c, "") for c in POSITIONS_COLUMNS})
            except Exception as e:                          # noqa: BLE001
                self.log.error("Ecriture %s impossible : %s", self.csv_path, e)
