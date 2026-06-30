"""live.py — daemon de trading live MEXC Futures pour l'equipe 1h (up + down).

Boucle 1-minute (REST-polling) qui integre tous les modules deja ecrits :

    mexc.py        -> MexcClient : le seul fichier qui touche l'API MEXC.
    position.py    -> PositionManager : gestion/sortie des positions ouvertes
                      (SL > TP > trailing > timestop), persistance, reconcile.
    config_team.py -> constantes recopiees du backtest + params live (env).
    live_helpers/  -> fetch (Bitstamp OHLC), predict (decide), notify (Telegram).

Architecture du cycle :
    - Le daemon tourne en continu. A CHAQUE minute (frontiere xx:xx:00 UTC) :
        1. lit le close 1-min qui vient de fermer (MEXC) -> PositionManager.on_minute
           (gestion intrabar des positions : SL/TP/trailing/timestop).
        2. EntryManager.tick : avance les cycles d'entree maker en cours.
        3. si on est au TOP DE L'HEURE (xx:00:00) : freshness Bitstamp 1h, refresh
           des buffers, predict.decide, append cycles.csv ; si l'equipe dit UP/DOWN
           et qu'un slot de position est libre -> EntryManager.start_entry.
        4. rendu Rich + log de timing.

Decision d'implementation cle :
    - La DECISION se prend au top de l'heure sur la bougie 1h qui vient de FERMER.
      On TRADE la bougie 1h qui vient d'OUVRIR (= open == target_open_ms). C'est
      l'ancre (anchor_ts) de la position : le SL temporel / timestop comptent les
      minutes depuis cet open, exactement comme le backtest (traded_ts = open i+1).
    - L'entree est MAKER (post-only) re-pricee 3 fois (1 essai / ENTRY_ATTEMPT_SEC),
      puis abandon si toujours pas remplie (opportunite ratee, notifiee).
    - Tout en MILLISECONDES en interne ; seul le fetch voit les secondes.

Lancement :
    cd backtest/modele_updown/1h/live_team
    python live.py            # mode reel ou DRY_RUN selon .env (DRY_RUN=true|false)

Prerequis (mode reel) : compte MEXC Futures KYC + permission trading Futures +
VPS hors-zone-bloquee (cf CLAUDE.md). DRY_RUN=true exerce toute la chaine sans
poster aucun ordre.
"""
from __future__ import annotations

import json
import logging
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Force la sortie UTF-8 pour que les box-drawing de Rich s'affichent sur les
# terminaux Windows cp1252 (et dans les logs Docker).
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

# --------------------------------------------------------------------------- #
# sys.path : ce dossier (live_team/) doit etre importable a plat pour que      #
# `import config_team`, `import mexc`, `import position` et le package          #
# live_helpers se resolvent quel que soit le cwd.                              #
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent           # live_team/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Chargement optionnel du .env (python-dotenv) avant de lire la config / l'env.
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv(ROOT / ".env")
except Exception:  # noqa: BLE001 - dotenv absent en dev : on lit l'env systeme.
    pass

import pandas as pd  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.panel import Panel  # noqa: E402
from rich.table import Table  # noqa: E402
from rich.text import Text  # noqa: E402

import config_team as cfg  # noqa: E402
from mexc import MexcClient, MexcError  # noqa: E402
from position import PositionManager  # noqa: E402
from live_helpers import fetch  # noqa: E402
from live_helpers import notify  # noqa: E402
from live_helpers import predict  # noqa: E402

logger = logging.getLogger("live_team")

# --------------------------------------------------------------------------- #
# Chemins de logs                                                              #
# --------------------------------------------------------------------------- #
LOGS_DIR = ROOT / "logs"
CYCLES_CSV = LOGS_DIR / "cycles.csv"
# Heure deja decidee, PERSISTEE : un redemarrage dans la 1re minute de l'heure ne
# doit pas rejouer la decision et ouvrir une 2e entree pour la meme heure.
DECISION_STATE = LOGS_DIR / "last_decision.json"


def _load_last_decided() -> int:
    """Open (ms) de la derniere heure deja decidee, ou -1 si absent/illisible."""
    try:
        return int(json.loads(DECISION_STATE.read_text(encoding="utf-8"))["last_decided_hour_open"])
    except Exception:  # noqa: BLE001
        return -1


def _save_last_decided(hour_open_ms: int) -> None:
    try:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        DECISION_STATE.write_text(
            json.dumps({"last_decided_hour_open": int(hour_open_ms)}), encoding="utf-8")
    except Exception as e:  # noqa: BLE001
        logger.warning("Persistance last_decided echouee : %s", e)

# --------------------------------------------------------------------------- #
# Constantes de la boucle                                                      #
# --------------------------------------------------------------------------- #
INTERVAL_1H_MS = int(getattr(cfg, "INTERVAL_1H_MS", 3_600_000))
SYMBOL = getattr(cfg, "SYMBOL", "BTC_USDT")

# Profondeur de warmup (barres) par TF. Le 1h doit clairement couvrir les
# fenetres longues du calcul de buckets (SMA 100 de l'atr_pct) + features alpha
# (bars_since_low_50, ret_skew_24...) -> >= 3000 barres 1h. L'ETH 15m est
# resample en 1h cote features_down, donc il faut ~4x plus de barres 15m.
WARMUP = {
    ("btcusd", 300):   500,    # 5m
    ("btcusd", 1800):  500,    # 30m
    ("btcusd", 3600):  3200,   # 1h (>= 3000 pour SMA100 atr + alpha)
    ("btcusd", 14400): 500,    # 4h
    ("btcusd", 86400): 500,    # 1d
    ("ethusd", 900):   3200,   # 15m ETH (resample 1h -> ~800 bougies 1h)
}

# Plancher dur de warmup 1h : en dessous, le calcul de bucket / features serait
# degrade (NaN sur la SMA longue). On refetch puis on echoue bruyamment.
MIN_WARMUP_1H_BARS = 3000
WARMUP_1H_FETCH_RETRIES = 4

# Rafraichissement incremental (queue uniquement) a chaque cycle horaire.
INCREMENTAL_REFRESH_N_BARS = 20

# Mapping (pair, step_s) -> cle de buffer attendue par predict.decide.
_BUFFER_KEYS = {
    ("btcusd", 3600):  "1h",
    ("btcusd", 300):   "5m",
    ("btcusd", 1800):  "30m",
    ("btcusd", 14400): "4h",
    ("btcusd", 86400): "1d",
    ("ethusd", 900):   "eth_15m",
}

console = Console()


# ============================================================
#  Helpers temporels (frontieres minute / heure UTC)
# ============================================================
def next_minute_boundary_s(now_s: float | None = None) -> int:
    """Prochaine frontiere de minute (xx:xx:00 UTC) en secondes unix."""
    if now_s is None:
        now_s = time.time()
    return int((int(now_s) // 60 + 1) * 60)


def wait_until(ts_s: int) -> None:
    """Dort jusqu'a l'instant `ts_s` (secondes unix), par pas <= 5 s pour rester
    reactif a un KeyboardInterrupt."""
    while True:
        delta = ts_s - time.time()
        if delta <= 0:
            return
        time.sleep(min(delta, 5))


def floor_hour_ms(ts_ms: int) -> int:
    """Open (ms) de l'heure contenant ts_ms."""
    return (int(ts_ms) // INTERVAL_1H_MS) * INTERVAL_1H_MS


# ============================================================
#  Buffers Bitstamp (warmup + refresh incremental)
# ============================================================
_CACHED_BUFFERS: dict[tuple[str, int], list[list]] = {}


def warmup_buffers() -> dict[str, list[list]]:
    """Cold-start : fetch parallele du warmup complet des 6 series Bitstamp.

    Garde dur sur le 1h : un buffer 1h trop court ne pourra jamais etre comble
    par le refresh incremental (queue seulement) et le daemon tournerait degrade
    (NaN sur la SMA longue) -> on refetch puis on echoue bruyamment.
    """
    global _CACHED_BUFFERS
    specs = [(pair, step, n) for (pair, step), n in WARMUP.items()]
    out = fetch.fetch_many_parallel(specs)

    key_1h = ("btcusd", 3600)
    n1h = len(out.get(key_1h, []))
    attempt = 0
    while n1h < MIN_WARMUP_1H_BARS and attempt < WARMUP_1H_FETCH_RETRIES:
        attempt += 1
        logger.warning("Warmup 1h court : %d < %d barres (refetch %d/%d).",
                       n1h, MIN_WARMUP_1H_BARS, attempt, WARMUP_1H_FETCH_RETRIES)
        out[key_1h] = fetch.fetch_klines("btcusd", 3600, WARMUP[key_1h])
        n1h = len(out[key_1h])
    if n1h < MIN_WARMUP_1H_BARS:
        raise RuntimeError(
            f"Warmup 1h seulement {n1h} barres apres {WARMUP_1H_FETCH_RETRIES} "
            f"refetch ; besoin de >= {MIN_WARMUP_1H_BARS}. Refus de demarrer "
            f"degrade — verifier la disponibilite OHLC Bitstamp."
        )

    _CACHED_BUFFERS = dict(out)
    logger.info("Warmup buffers OK : %s", ", ".join(
        f"{p}/{s}s={len(v)}" for (p, s), v in out.items()))
    return _named_buffers()


def refresh_buffers_tail() -> dict[str, list[list]]:
    """Cycle chaud : ne refetch que la queue de chaque TF (parallele) puis merge
    dans les buffers caches (taille bornee au warmup)."""
    specs = [(pair, step, INCREMENTAL_REFRESH_N_BARS) for (pair, step) in WARMUP]
    fresh = fetch.fetch_many_parallel(specs)
    for key, new_bars in fresh.items():
        merged = fetch.merge_klines(_CACHED_BUFFERS.get(key, []), new_bars)
        _CACHED_BUFFERS[key] = merged[-WARMUP[key]:]
    return _named_buffers()


def _named_buffers() -> dict[str, list[list]]:
    """Convertit le cache (pair, step) en dict de cles '1h','5m',... attendu par
    predict.decide."""
    return {_BUFFER_KEYS[key]: _CACHED_BUFFERS.get(key, []) for key in WARMUP}


# ============================================================
#  cycles.csv — une ligne par cycle horaire (decision team)
# ============================================================
CYCLES_COLUMNS = [
    "cycle_n", "ts", "bar_close_utc", "o", "h", "l", "c",
    "p_up", "th_up", "take_up", "p_dn", "th_dn", "take_dn",
    "volatility_state", "is_weekend", "hour", "team",
    "n_open", "run_cycles", "run_entries",
    "data_valid", "invalid_reason",
]


def append_cycle_row(row: dict) -> None:
    """Ajoute une ligne a cycles.csv (header si fichier neuf). Best-effort : une
    erreur d'I/O ne doit jamais tuer la boucle."""
    try:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        full = {c: row.get(c, "") for c in CYCLES_COLUMNS}
        write_header = not CYCLES_CSV.exists()
        pd.DataFrame([full], columns=CYCLES_COLUMNS).to_csv(
            CYCLES_CSV, mode="a", header=write_header, index=False)
    except Exception as e:  # noqa: BLE001
        logger.error("Ecriture %s impossible : %s", CYCLES_CSV, e)


# ============================================================
#  EntryManager — entree MAKER 3x re-pricee + sizing
# ============================================================
class EntryManager:
    """Gere l'entree d'une position via des ordres MAKER (post-only) re-prices.

    Un seul cycle d'entree actif a la fois (l'equipe ne prend qu'une decision par
    heure et MAX_CONCURRENT_POSITIONS borne le total). Le cycle :
      - poste un ordre post-only au prix marche (offset ENTRY_MAKER_OFFSET passif),
      - le laisse vivre ENTRY_ATTEMPT_SEC ; a chaque `tick`, si l'echeance est
        atteinte et l'ordre non rempli : annule, re-price, repost (jusqu'a
        ENTRY_ATTEMPTS essais),
      - sur fill -> PositionManager.open_position(...) avec anchor_ts = open de la
        bougie 1h tradee,
      - apres ENTRY_ATTEMPTS sans fill -> abandon (opportunite ratee, notifiee).
    """

    def __init__(self, mexc: MexcClient, cfg_mod, position_manager: PositionManager,
                 notify_mod, contract_size: float, price_unit: float):
        self.mexc = mexc
        self.cfg = cfg_mod
        self.pm = position_manager
        self.notify = notify_mod
        self.contract_size = float(contract_size) if contract_size else 0.0001
        self.price_unit = float(price_unit) if price_unit else 0.0

        # Etat du cycle d'entree en cours (None = aucun).
        self._active: dict | None = None
        self.entries_started = 0   # compteur cumulatif (stats / cycles.csv)

    # --- helpers ------------------------------------------------------------ #
    def is_busy(self) -> bool:
        """True si un cycle d'entree est en cours (un ordre maker vit)."""
        return self._active is not None

    def _round_price(self, price: float) -> float | None:
        if price is None or price <= 0:
            return None
        if self.price_unit and self.price_unit > 0:
            steps = round(price / self.price_unit)
            return round(steps * self.price_unit, 12)
        return price

    def _market_price(self, fallback: float = 0.0) -> float:
        """Dernier prix marche (ticker) ; fallback si l'API echoue."""
        try:
            t = self.mexc.get_ticker(self.cfg.SYMBOL)
            px = float(t.get("last_price") or t.get("fair_price") or 0.0)
            if px > 0:
                return px
        except Exception as e:  # noqa: BLE001
            logger.warning("get_ticker indisponible (entree) : %s", e)
        return float(fallback)

    def _compute_vol(self, price: float) -> int:
        """Sizing : marge = BET_FRACTION * solde dispo ; notionnel = marge x levier ;
        vol = max(MIN_VOL, floor(notionnel / (prix x contract_size)))."""
        try:
            bal = float(self.mexc.get_assets("USDT").get("available_balance", 0.0))
        except Exception as e:  # noqa: BLE001
            # Echec de lecture du solde -> on ABANDONNE l'entree (return 0 =>
            # start_entry annule), au lieu de retomber sur MIN_VOL et placer un
            # ordre reel sur un simple echec de fetch.
            logger.exception("get_assets a echoue pour le sizing : %s — entree abandonnee.", e)
            return 0
        margin = float(getattr(self.cfg, "BET_FRACTION", 0.5)) * bal
        notional = margin * int(getattr(self.cfg, "MEXC_LEVERAGE", 10))
        min_vol = int(getattr(self.cfg, "MIN_VOL", 1))
        denom = price * self.contract_size
        if denom <= 0:
            return 0
        return max(min_vol, int(math.floor(notional / denom)))

    def _open_side(self, side: str) -> int:
        """side MEXC d'OUVERTURE (mode hedge) : up->1 (open long), down->3 (open short)."""
        return 1 if side == "up" else 3

    def _maker_limit(self, side: str, market_price: float) -> float:
        """Prix limite post-only PASSIF : pour ouvrir un long on poste SOUS le marche
        (le prix doit redescendre nous toucher) ; pour ouvrir un short on poste
        AU-DESSUS. Post-only => maker garanti (rejete s'il croiserait le book)."""
        offset = float(getattr(self.cfg, "ENTRY_MAKER_OFFSET", 0.0001))
        if side == "up":
            limit = market_price * (1.0 - offset)
        else:
            limit = market_price * (1.0 + offset)
        return self._round_price(limit) or limit

    # --- cycle de vie ------------------------------------------------------- #
    def start_entry(self, side: str, anchor_ts: int, decision_ctx: dict) -> None:
        """Demarre un cycle d'entree maker pour `side` ('up'|'down').

        anchor_ts : open (ms) de la bougie 1h TRADEE (= bougie qui vient d'ouvrir).
        decision_ctx : contexte de la decision (proba/seuils/bucket) pour le log.
        """
        if self.is_busy():
            logger.warning("start_entry ignore : un cycle d'entree est deja actif.")
            return

        market = self._market_price()
        if market <= 0:
            logger.error("start_entry %s : prix marche indisponible, entree annulee.", side)
            return
        vol = self._compute_vol(market)
        if vol <= 0:
            logger.error("start_entry %s : vol calcule <= 0 (solde insuffisant ?), "
                         "entree annulee.", side)
            return

        now_ms = int(time.time() * 1000)
        base_oid = f"entry-{side}-{anchor_ts}"
        self._active = {
            "side": side,
            "anchor_ts": int(anchor_ts),
            "vol": vol,
            "attempt": 0,                # nombre d'ordres deja postes
            "order_id": None,
            "limit": None,
            "deadline": now_ms,          # tick() postera tout de suite (echeance atteinte)
            "base_oid": base_oid,
            "ctx": decision_ctx,
        }
        self.entries_started += 1
        logger.info("Cycle d'entree %s demarre (anchor=%s vol=%d).", side, anchor_ts, vol)
        # On poste le premier ordre immediatement.
        self._place_attempt()

    def _place_attempt(self) -> None:
        """Poste un ordre maker (post-only) pour le cycle actif et arme l'echeance."""
        a = self._active
        if a is None:
            return
        attempts_max = int(getattr(self.cfg, "ENTRY_ATTEMPTS", 3))
        if a["attempt"] >= attempts_max:
            self._abandon()
            return

        market = self._market_price(a.get("limit") or 0.0)
        if market <= 0:
            logger.warning("_place_attempt : prix marche indisponible, on reessaiera.")
            a["deadline"] = int(time.time() * 1000) + 5_000
            return
        limit = self._maker_limit(a["side"], market)
        oid = f"{a['base_oid']}-a{a['attempt']}"
        try:
            res = self.mexc.place_order(
                symbol=self.cfg.SYMBOL,
                side=self._open_side(a["side"]),
                vol=int(a["vol"]),
                price=limit,
                order_type=2,  # post-only (maker garanti)
                open_type=int(getattr(self.cfg, "OPEN_TYPE", 1)),
                leverage=int(getattr(self.cfg, "MEXC_LEVERAGE", 10)),
                reduce_only=False,
                external_oid=oid,
            )
        except MexcError as e:
            logger.error("place_order (entree %s) a echoue : %s", a["side"], e)
            # On retentera au prochain essai apres l'echeance.
            a["attempt"] += 1
            a["deadline"] = int(time.time() * 1000) + int(
                getattr(self.cfg, "ENTRY_ATTEMPT_SEC", 300)) * 1000
            return

        a["order_id"] = str(res.get("order_id"))
        a["limit"] = limit
        a["attempt"] += 1
        a["deadline"] = int(time.time() * 1000) + int(
            getattr(self.cfg, "ENTRY_ATTEMPT_SEC", 300)) * 1000
        logger.info("Entree %s : ordre maker poste #%d @ %.2f (vol=%d, oid=%s).",
                    a["side"], a["attempt"], limit, a["vol"], a["order_id"])

        # En DRY_RUN l'ordre n'est pas reellement poste : on considere l'entree
        # comme remplie immediatement au prix limite (pas de poll possible).
        if res.get("dry_run"):
            self._on_filled(limit, int(a["vol"]), mexc_position_id=None)

    def tick(self, now_ms: int) -> None:
        """Avance le cycle d'entree en cours : verifie le fill, re-price/repost si
        l'echeance est atteinte, abandonne apres ENTRY_ATTEMPTS essais."""
        a = self._active
        if a is None:
            return

        # 1) Verifie un fill eventuel de l'ordre courant.
        last_deal_vol = 0.0
        last_deal_px = 0.0
        oid = a.get("order_id")
        if oid and not str(oid).startswith("DRYRUN-"):
            try:
                od = self.mexc.get_order(oid)
            except MexcError as e:
                logger.warning("get_order(%s) pendant l'entree : %s", oid, e)
                od = None
            if od is not None:
                state = int(od.get("state", 0))
                deal_vol = float(od.get("deal_vol", 0.0))
                deal_px = float(od.get("deal_avg_price", 0.0))
                last_deal_vol = deal_vol
                last_deal_px = deal_px
                if deal_vol > 0 and (state == 3 or state == 2):
                    # Rempli (totalement state=3, ou partiellement en cours) : on
                    # adopte le fill. Pour un partiel state==2 on attend la fin de
                    # fenetre ; on ne cloture le cycle qu'a fill complet.
                    if state == 3:
                        self._on_filled(deal_px or a["limit"], int(round(deal_vol)),
                                        mexc_position_id=self._lookup_position_id(a["side"]))
                        return

        # 2) Echeance atteinte sans fill complet.
        if now_ms >= int(a["deadline"]):
            # Annule l'ordre en cours d'abord (libere le vol restant non rempli).
            self._cancel_active_order()
            # Fill PARTIEL a l'expiration : on ADOPTE la portion deja remplie comme
            # position et on ARRETE le cycle d'entree. Re-poster le vol PLEIN
            # sur-exposerait (la portion deja remplie reste ouverte cote MEXC).
            filled_vol = int(round(last_deal_vol))
            if filled_vol > 0:
                logger.warning("Entree %s : fill PARTIEL %.4f/%d a l'expiration "
                               "-> adoption de la portion remplie (pas de repost).",
                               a["side"], last_deal_vol, int(a["vol"]))
                self._on_filled(last_deal_px or a["limit"], filled_vol,
                                mexc_position_id=self._lookup_position_id(a["side"]))
                return
            # Aucun fill -> re-price + repost (comportement inchange).
            self._place_attempt()

    def _on_filled(self, entry_price: float, vol: int,
                   mexc_position_id: int | None) -> None:
        """Le cycle d'entree a un fill : ouvre la position et clot le cycle."""
        a = self._active
        if a is None:
            return
        try:
            self.pm.open_position(
                side=a["side"],
                entry=float(entry_price),
                vol=int(vol),
                anchor_ts=int(a["anchor_ts"]),
                mexc_position_id=mexc_position_id,
                external_oid=a["base_oid"],
            )
        except Exception as e:  # noqa: BLE001
            logger.exception("open_position apres fill a echoue : %s", e)
        finally:
            self._active = None

    def _abandon(self) -> None:
        """Abandonne le cycle d'entree (aucun fill apres ENTRY_ATTEMPTS) et notifie."""
        a = self._active
        self._active = None
        if a is None:
            return
        self._cancel_order_id(a.get("order_id"))
        logger.warning("Entree %s ABANDONNEE apres %d essais (non remplie).",
                       a["side"], a["attempt"])
        send = getattr(self.notify, "send_entry_missed", None)
        if send is not None:
            try:
                send(side=a["side"])
            except Exception as e:  # noqa: BLE001
                logger.warning("notify.send_entry_missed a echoue : %s", e)

    def _cancel_active_order(self) -> None:
        if self._active is not None:
            self._cancel_order_id(self._active.get("order_id"))
            self._active["order_id"] = None

    def _cancel_order_id(self, order_id) -> None:
        if not order_id or str(order_id).startswith("DRYRUN-"):
            return
        try:
            self.mexc.cancel_orders([str(order_id)])
        except Exception as e:  # noqa: BLE001
            logger.warning("Annulation ordre entree %s impossible : %s", order_id, e)

    def _lookup_position_id(self, side: str) -> int | None:
        """Retrouve le positionId MEXC du sens venant d'etre ouvert (best-effort)."""
        try:
            want = 1 if side == "up" else 2
            for p in self.mexc.get_open_positions(self.cfg.SYMBOL):
                if int(p.get("position_type", 0)) == want and float(p.get("hold_vol", 0.0)) > 0:
                    return int(p.get("position_id", 0)) or None
        except Exception as e:  # noqa: BLE001
            logger.warning("lookup positionId (%s) a echoue : %s", side, e)
        return None


# ============================================================
#  Rendu Rich
# ============================================================
def _dir_text(team: str) -> Text:
    if team == "UP":
        return Text("UP", style="bright_blue")
    if team == "DOWN":
        return Text("DOWN", style="bright_red")
    if team == "CANCEL":
        return Text("CANCEL", style="yellow")
    return Text("PASS", style="dim")


def render_cycle(cycle_n: int, just_closed_open: int, dec: dict, n_open: int,
                 stats: dict, bar: list | None) -> None:
    """Affiche un panneau Rich resumant la decision horaire."""
    closed_at = pd.Timestamp(just_closed_open + INTERVAL_1H_MS, unit="ms", tz="UTC")
    head = Text.assemble(
        ("LIVE TEAM ", "bold cyan"),
        ("1h MEXC", "bold magenta"),
        (f"  -  cycle #{cycle_n}", "dim"),
        ("\n", ""),
        ("bougie 1h fermee a ", "dim"),
        (closed_at.strftime("%Y-%m-%d %H:%M UTC"), "white"),
    )
    console.print(Panel(head, border_style="cyan", padding=(0, 2)))

    pt = Table(title="Prediction equipe", title_style="bold cyan",
               border_style="cyan", header_style="bold", expand=False)
    pt.add_column("Modele", style="bold")
    pt.add_column("p", justify="right")
    pt.add_column("seuil", justify="right")
    pt.add_column("prend ?", justify="right")
    for name, p, th, take, color in [
        ("modele_up",   dec["p_up"], dec["th_up"], dec["take_up"], "bright_blue"),
        ("modele_down", dec["p_dn"], dec["th_dn"], dec["take_dn"], "bright_red"),
    ]:
        th_cell = "skip" if th >= 1.0 else f"{th*100:6.2f}%"
        pt.add_row(Text(name, style=color), f"{p*100:6.2f}%", th_cell,
                   Text("oui", style="bold green") if take else Text("non", style="dim"))
    console.print(pt)

    headline = Text.assemble(("decision equipe : ", "bold"), _dir_text(dec["team"]))
    console.print(Panel(headline, border_style=_dir_text(dec["team"]).style or "white",
                        padding=(0, 2)))

    line = Text.assemble(
        ("etat  ", "dim"),
        (f"{n_open} position(s) ouverte(s)  |  ", "white"),
        (f"vol_state={dec['volatility_state']} wknd={dec['is_weekend']} h={dec['hour']}  |  ", "dim"),
        (f"{stats['cycles']} cycles, {stats['entries']} entrees", "white"),
    )
    console.print(line)
    console.print()


# ============================================================
#  Bootstrap
# ============================================================
def bootstrap() -> tuple[MexcClient, PositionManager, EntryManager]:
    """Initialise tout : error-handler Telegram, client MEXC, levier hedge,
    cache contrat, reconcile des positions, warmup des buffers, chargement des
    modeles, notif de demarrage. Renvoie (mexc, position_manager, entry_manager).
    """
    # 1) Error-handler Telegram EN PREMIER : tout logger.error/exception qui suit
    #    part vers le bot d'erreur (s'il est configure).
    notify.install_error_handler()

    # 2) Client MEXC (le seul a toucher l'API).
    mexc = MexcClient.from_env()
    logger.info("MexcClient pret (host=%s, dry_run=%s).", mexc.rest_host, mexc.dry_run)

    # 3) Levier hedge sur les DEUX sens (long ET short). Best-effort : un echec
    #    (deja regle, pas de permission) ne doit pas empecher le demarrage.
    leverage = int(getattr(cfg, "MEXC_LEVERAGE", 10))
    open_type = int(getattr(cfg, "OPEN_TYPE", 1))
    for position_type in (1, 2):  # 1=long, 2=short
        try:
            mexc.set_leverage(SYMBOL, leverage, open_type, position_type)
            logger.info("Levier regle a x%d (positionType=%d).", leverage, position_type)
        except MexcError as e:
            logger.warning("set_leverage(positionType=%d) a echoue : %s", position_type, e)

    # 4) Cache des details du contrat (contract_size / price_unit). Indispensable
    #    pour le sizing et l'arrondi de prix.
    contract_size = float(getattr(cfg, "CONTRACT_SIZE", 0.0001) or 0.0001)
    price_unit = float(getattr(cfg, "PRICE_UNIT", 0.0) or 0.0)
    try:
        detail = mexc.get_contract_detail(SYMBOL)
        contract_size = float(detail.get("contract_size") or contract_size)
        price_unit = float(detail.get("price_unit") or price_unit)
        logger.info("Contrat %s : contract_size=%s price_unit=%s min_vol=%s.",
                    SYMBOL, contract_size, price_unit, detail.get("min_vol"))
    except MexcError as e:
        logger.warning("get_contract_detail a echoue (valeurs par defaut) : %s", e)

    # 5) PositionManager + reconcile au boot (aligne l'etat persiste avec MEXC).
    pm = PositionManager(mexc, cfg, notify=notify, logger=logger)
    pm.contract_size = contract_size
    pm.price_unit = price_unit
    pm.reconcile(SYMBOL)

    # 6) Warmup des buffers Bitstamp (5 TF BTC + ETH 15m).
    warmup_buffers()

    # 7) Chargement des modeles + sidecars + config.py (cache module-level).
    predict.load()
    logger.info("Modeles charges : UP(%d feats) DOWN(%d feats).",
                len(predict.FEATURES_UP), len(predict.FEATURES_DOWN))

    # 8) EntryManager.
    em = EntryManager(mexc, cfg, pm, notify, contract_size, price_unit)

    # 9) Notif de demarrage (solde dispo si lisible).
    balance = None
    try:
        balance = float(mexc.get_assets("USDT").get("available_balance", 0.0))
    except Exception as e:  # noqa: BLE001
        logger.warning("get_assets au boot indisponible : %s", e)
    mode = "DRY_RUN" if mexc.dry_run else "REAL"
    send_startup = getattr(notify, "send_startup", None)
    if send_startup is not None:
        try:
            send_startup(balance=balance, symbol=SYMBOL, mode=mode)
        except Exception as e:  # noqa: BLE001
            logger.warning("notify.send_startup a echoue : %s", e)

    return mexc, pm, em


# ============================================================
#  Boucle principale (tick 1-minute)
# ============================================================
def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    mexc, pm, em = bootstrap()

    stats = {"cycles": 0, "entries": 0}
    # Open de la derniere heure deja traitee par la branche horaire (anti double
    # decision si plusieurs ticks tombent dans la meme heure). PERSISTE : un
    # redemarrage dans la 1re minute de l'heure ne rejoue pas la decision.
    last_decided_hour_open = _load_last_decided()

    console.print(Panel(
        Text(f"daemon live demarre — symbol={SYMBOL} mode={'DRY_RUN' if mexc.dry_run else 'REAL'}",
             style="bold green"),
        border_style="green"))

    while True:
        # Frontiere minute suivante (xx:xx:00 UTC).
        boundary_s = next_minute_boundary_s()
        wait_until(boundary_s)

        t_start = time.perf_counter()
        try:
            now_ms = int(time.time() * 1000)

            # --- 1) Gestion intrabar : close 1-min qui vient de fermer --------
            try:
                kl_min1 = mexc.get_klines(SYMBOL, "Min1")
            except MexcError as e:
                logger.error("get_klines(Min1) a echoue : %s", e)
                kl_min1 = []
            if kl_min1:
                # On ne rejoue QUE les bougies 1-min reellement FERMEES : la
                # derniere barre renvoyee par get_klines est souvent celle en
                # formation (open == minute courante non terminee) -> on l'exclut.
                cur_min_open = (now_ms // 60_000) * 60_000
                closed = [b for b in kl_min1 if int(b[0]) < cur_min_open]
                try:
                    # Rejeu de toutes les bougies fermees depuis le dernier
                    # traitement (rattrape une minute sautee : redemarrage, stall,
                    # kline vide) ; chaque close est evalue avec j aligne sur le
                    # backtest (now_ms = OPEN de la bougie, cf. process_min1_bars).
                    pm.process_min1_bars(closed)
                except Exception as e:  # noqa: BLE001
                    logger.exception("PositionManager.process_min1_bars a echoue : %s", e)

            # --- 2) Avance le cycle d'entree maker en cours -------------------
            try:
                em.tick(now_ms)
            except Exception as e:  # noqa: BLE001
                logger.exception("EntryManager.tick a echoue : %s", e)

            # --- 3) Top de l'heure : decision team ----------------------------
            cur_hour_open = floor_hour_ms(now_ms)
            # On declenche la decision dans la 1re minute de l'heure (xx:00) et une
            # seule fois par heure.
            at_hour_top = (now_ms - cur_hour_open) < 90_000  # < 90 s apres xx:00
            if at_hour_top and cur_hour_open != last_decided_hour_open:
                last_decided_hour_open = cur_hour_open
                _save_last_decided(cur_hour_open)
                _run_hourly_decision(mexc, pm, em, stats, cur_hour_open, t_start)

        except KeyboardInterrupt:
            console.print("\n[bold red]Arret demande.[/bold red]")
            break
        except Exception as e:  # noqa: BLE001
            logger.exception("Tick echoue : %s", e)
            time.sleep(5)


def _run_hourly_decision(mexc: MexcClient, pm: PositionManager, em: EntryManager,
                         stats: dict, cur_hour_open: int, t_start: float) -> None:
    """Branche horaire (top de l'heure) : freshness Bitstamp 1h, refresh des
    buffers, predict.decide sur la bougie qui vient de fermer, log cycles.csv et
    eventuellement demarrage d'un cycle d'entree.

    La bougie qui vient de FERMER a pour open `just_closed_open = cur_hour_open -
    1h` ; c'est ELLE qui porte les features de decision. On TRADE la bougie qui
    vient d'OUVRIR (open == cur_hour_open) -> anchor_ts = cur_hour_open.
    """
    just_closed_open = cur_hour_open - INTERVAL_1H_MS

    # 1) Confirme que Bitstamp a bien publie la bougie 1h qui vient de fermer.
    buf_1h_key = ("btcusd", 3600)
    try:
        updated = fetch.freshness_retry(
            "btcusd", 3600, just_closed_open,
            _CACHED_BUFFERS.get(buf_1h_key, []), max_tries=12)
        _CACHED_BUFFERS[buf_1h_key] = fetch.merge_klines(
            _CACHED_BUFFERS.get(buf_1h_key, []), updated)[-WARMUP[buf_1h_key]:]
    except TimeoutError as e:
        logger.error("freshness_retry 1h a echoue (%s) — cycle ignore.", e)
        return

    # 2+3) Refresh + Decide avec VALIDATION des donnees et re-essais.
    #    predict.decide valide les donnees AVANT toute prediction (OHLC corrompu /
    #    troue / buffer trop court / bougie cible absente / feature NaN-inf, ETH
    #    manquant compris). Si invalides -> dec["data_valid"]=False et AUCUNE
    #    prediction n'a ete faite : on re-fetch et on re-essaie jusqu'a
    #    VALIDATION_MAX_RETRIES, puis on ANNULE le cycle (aucun pari).
    max_retries = max(1, int(getattr(cfg, "VALIDATION_MAX_RETRIES", 3)))
    retry_sleep = float(getattr(cfg, "VALIDATION_RETRY_SLEEP_SEC", 5))
    buffers, dec = None, None
    for attempt in range(1, max_retries + 1):
        if attempt > 1:
            time.sleep(retry_sleep)
        try:
            buffers = refresh_buffers_tail()
        except Exception as e:  # noqa: BLE001
            logger.warning("refresh_buffers_tail essai %d/%d a echoue : %s",
                           attempt, max_retries, e)
            dec = None
            continue
        try:
            dec = predict.decide(buffers, just_closed_open)
        except Exception as e:  # noqa: BLE001
            logger.warning("predict.decide essai %d/%d a leve : %s",
                           attempt, max_retries, e)
            dec = None
            continue
        if dec.get("data_valid", True):
            break  # donnees valides -> on poursuit le cycle normalement
        logger.warning("Donnees invalides essai %d/%d : %s", attempt, max_retries,
                       "; ".join(dec.get("invalid_reasons", [])) or "?")
    t_after_fetch = time.perf_counter()
    t_after_predict = t_after_fetch

    # Donnees toujours invalides apres tous les essais -> ANNULE le cycle.
    # logger.error route automatiquement vers le bot Telegram d'erreur (notify).
    if dec is None or not dec.get("data_valid", True):
        reason = ("; ".join(dec.get("invalid_reasons", [])) if dec
                  else "fetch/predict en echec")
        bar_close = pd.Timestamp(just_closed_open + INTERVAL_1H_MS, unit="ms",
                                 tz="UTC").strftime("%Y-%m-%d %H:%M UTC")
        logger.error("Cycle %s ANNULE : donnees invalides apres %d essais — "
                     "AUCUNE prediction. Raison: %s", bar_close, max_retries, reason)
        stats["cycles"] += 1
        append_cycle_row({
            "cycle_n": stats["cycles"],
            "ts": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "bar_close_utc": bar_close,
            "team": "PASS", "data_valid": 0, "invalid_reason": reason,
            "n_open": pm.n_open(), "run_cycles": stats["cycles"],
            "run_entries": stats["entries"],
        })
        return

    stats["cycles"] += 1
    n_open = pm.n_open()

    # 4) Log cycles.csv (une ligne par cycle horaire).
    jc_bar = _find_bar(buffers.get("1h", []), just_closed_open)
    append_cycle_row({
        "cycle_n": stats["cycles"],
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "bar_close_utc": pd.Timestamp(just_closed_open + INTERVAL_1H_MS, unit="ms",
                                      tz="UTC").strftime("%Y-%m-%d %H:%M UTC"),
        "o": round(float(jc_bar[1]), 2) if jc_bar else "",
        "h": round(float(jc_bar[2]), 2) if jc_bar else "",
        "l": round(float(jc_bar[3]), 2) if jc_bar else "",
        "c": round(float(jc_bar[4]), 2) if jc_bar else "",
        "p_up": round(dec["p_up"], 6), "th_up": round(dec["th_up"], 6),
        "take_up": int(dec["take_up"]),
        "p_dn": round(dec["p_dn"], 6), "th_dn": round(dec["th_dn"], 6),
        "take_dn": int(dec["take_dn"]),
        "volatility_state": dec["volatility_state"],
        "is_weekend": dec["is_weekend"], "hour": dec["hour"],
        "team": dec["team"], "n_open": n_open,
        "run_cycles": stats["cycles"], "run_entries": stats["entries"],
        "data_valid": 1, "invalid_reason": "",
    })

    # 5) Si l'equipe prend une position et qu'un slot est libre -> entree maker.
    #    La bougie TRADEE est celle qui vient d'ouvrir : anchor_ts = cur_hour_open.
    max_pos = int(getattr(cfg, "MAX_CONCURRENT_POSITIONS", 2))
    if dec["team"] in ("UP", "DOWN"):
        if n_open >= max_pos:
            logger.info("Decision %s ignoree : %d/%d positions deja ouvertes.",
                        dec["team"], n_open, max_pos)
        elif em.is_busy():
            logger.info("Decision %s ignoree : un cycle d'entree est deja en cours.",
                        dec["team"])
        elif pm.has_open_side("up" if dec["team"] == "UP" else "down"):
            # En mode hedge, MEXC agrege tout le meme sens en UNE position : ouvrir
            # un 2e long (ou 2e short) ferait suivre 2 Position pour 1 reelle.
            logger.info("Decision %s ignoree : une position de meme sens est deja ouverte.",
                        dec["team"])
        else:
            side = "up" if dec["team"] == "UP" else "down"
            em.start_entry(side, anchor_ts=cur_hour_open, decision_ctx=dec)
            stats["entries"] = em.entries_started

    # 6) Rendu Rich + timing.
    render_cycle(stats["cycles"], just_closed_open, dec, pm.n_open(), stats, jc_bar)
    logger.info("cycle horaire : fetch=%.2fs predict=%.2fs total=%.2fs | team=%s n_open=%d",
                t_after_fetch - t_start, t_after_predict - t_after_fetch,
                time.perf_counter() - t_start, dec["team"], pm.n_open())


def _find_bar(buf: list[list], open_ms: int) -> list | None:
    """Trouve une barre par son open_ms (None si absente)."""
    for b in reversed(buf):
        if int(b[0]) == int(open_ms):
            return b
    return None


if __name__ == "__main__":
    main()
