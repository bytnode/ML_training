"""config_live.py -- constantes SPECIFIQUES au daemon live OKX.

Ne redefinit RIEN de ce qui vit deja dans modele_team/config_main.py (seuils
DMI/UP-filters/SL/TP/trailing/timestop/levier/exposure) -- ces valeurs sont
IMPORTEES depuis config_main.py partout ou elles sont necessaires (symetrie
backtest/live par construction, cf CLAUDE.md racine et filters.py). Ce fichier
ne contient que ce qui n'a pas d'equivalent backtest : identifiant d'instrument
OKX, cadence de la boucle, chemins de logs, noms des variables d'env attendues.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# live/config_live.py -> parent = live/, parent.parent = modele_team/. On pousse
# modele_team/ sur sys.path pour importer config_main / filters / okx_fees comme
# modules top-level (meme trick que main.py pousse 1h/ pour modele_up/modele_down).
_TEAM_DIR = str(Path(__file__).resolve().parent.parent)
if _TEAM_DIR not in sys.path:
    sys.path.insert(0, _TEAM_DIR)

import logging

from config_main import LEVERAGE as _BACKTEST_LEVERAGE  # noqa: E402
from config_main import POSITION_FRACTION as _BACKTEST_POSITION_FRACTION  # noqa: E402

_logger = logging.getLogger("live.config_live")


def _env_int(name: str, default: int) -> int:
    """Lit une variable d'env entiere avec fallback sur `default` si absente/
    vide/malformee -- ne leve JAMAIS (une .env corrompue ne doit jamais
    empecher le daemon de demarrer), mais loggue un avertissement si elle
    etait presente mais invalide plutot que d'echouer silencieusement (cf
    audit erreurs silencieuses 2026-07-15)."""
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        _logger.warning(f"{name}={raw!r} invalide (entier attendu) -- fallback sur {default}.")
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        _logger.warning(f"{name}={raw!r} invalide (nombre attendu) -- fallback sur {default}.")
        return default


# ============================================================
#  Instrument OKX
# ============================================================
OKX_INST_ID = os.getenv("OKX_INST_ID", "BTC-USDT-SWAP")   # SWAP perpetuel USDT-M
OKX_MGN_MODE = "isolated"                                   # marge ISOLATED (cf design)
OKX_POS_MODE = "long_short_mode"                             # mode HEDGE (long + short simultanes)

# ============================================================
#  Parametres financiers -- SURCHARGEABLES via .env (2026-07-15, demande
#  explicite operateur : ".env = fichier de config pour le sensible")
# ============================================================
# ATTENTION -- ceci s'ECARTE DELIBEREMENT du principe de symetrie backtest/
# live (config_main.py = seule source de verite partagee, cf CLAUDE.md racine
# et live/CLAUDE.md). Si OKX_LEVERAGE/OKX_POSITION_FRACTION sont ABSENTES du
# .env, le comportement par defaut reste symetrique (retombe exactement sur
# config_main.LEVERAGE/POSITION_FRACTION, ce que le backtest a valide). Des
# qu'elles sont renseignees, le live peut trader avec un risque REEL different
# de ce que le backtest a mesure -- a l'operateur de le savoir.
OKX_LEVERAGE = _env_int("OKX_LEVERAGE", _BACKTEST_LEVERAGE)
OKX_POSITION_FRACTION = _env_float("OKX_POSITION_FRACTION", _BACKTEST_POSITION_FRACTION)

# ============================================================
#  Filet de securite "catastrophe" natif OKX (audit 2026-08-03, point #1)
# ============================================================
# Un stop-loss STATIQUE pose cote OKX (order-algo conditionnel) a l'ouverture de
# CHAQUE position reelle -- il s'execute sur le matching engine d'OKX MEME SI le
# daemon est mort/fige/deconnecte. Ce n'est PAS une replique du SL dynamique du
# daemon (SL temporel/trailing/timestop/regime restent 100% dans position.py) :
# c'est un unique plancher fixe par sens, pose DELIBEREMENT PLUS LARGE que la
# valeur la plus large que le SL dynamique atteint jamais (sa valeur a l'entree,
# j=0 : UP_SL_START=-1.5%, DOWN_SL_START=-1%), donc il ne se declenche JAMAIS
# avant le SL du daemon en fonctionnement normal -- il ne sert que si le daemon
# est aveugle, pour eviter la liquidation (~ -9% a 10x isole). Invariant : le
# widen ne touche jamais le SL, le regime ne touche que max_hold/trail_gap ->
# ces bornes -1.5%/-1% sont fixes, un seul chiffre par sens suffit.
NATIVE_BACKSTOP_ENABLED = _env_int("NATIVE_BACKSTOP_ENABLED", 1) == 1
CATASTROPHE_SL_UP_PCT = _env_float("CATASTROPHE_SL_UP_PCT", 0.030)     # long : -3.0% prix (2x le SL le plus large -1.5%)
CATASTROPHE_SL_DOWN_PCT = _env_float("CATASTROPHE_SL_DOWN_PCT", 0.025)  # short : -2.5% prix (2.5x le SL le plus large -1%)
# ⚠️ PIEGE D'UNITES (audit 2026-08-14) : malgre le suffixe _PCT, ces deux
# valeurs sont des FRACTIONS decimales (0.030 = 3.0%), PAS des pourcents comme
# la convention _PCT de config_main.py (2026-08-11). Ne JAMAIS "harmoniser" a
# 3.0/2.5 ni mettre CATASTROPHE_SL_UP_PCT=3 dans .env : le filet se poserait a
# -300% (= jamais) ou pire. position.py les consomme en decimal tel quel.
# Declenchement sur le MARK price (coherent avec le pp reel de evaluate_exit et
# avec la logique de liquidation OKX). Valeurs OKX : "last" / "index" / "mark".
BACKSTOP_TRIGGER_PX_TYPE = "mark"

# ============================================================
#  Anti-rebond desync (audit 2026-08-03, point #4)
# ============================================================
# Nb de lectures get_positions() CONSECUTIVES montrant la position absente avant
# de declarer une desync et d'effacer l'etat local. Une seule lecture vide
# transitoire (maintenance/propagation OKX) ne doit JAMAIS orpheliner une
# position reelle encore ouverte (= perte de tout suivi SL/TP).
DESYNC_CONFIRM_TICKS = _env_int("DESYNC_CONFIRM_TICKS", 3)

# ============================================================
#  Sizing : marge disponible + garde-fous (audit 2026-08-03, points #6/#9)
# ============================================================
# Fraction de la marge DISPONIBLE (availEq) reellement engageable -- laisse un
# coussin pour les frais taker + funding, pour qu'un 2e trade (sens oppose) ne
# soit jamais rejete par OKX faute de marge (code 51008). Dimensionne sur
# availEq (pas eq) : une position deja ouverte gele sa marge, la suivante ne
# doit pas re-supposer le capital plein.
SIZING_SAFETY_BUFFER = _env_float("SIZING_SAFETY_BUFFER", 0.98)
# Tolerance de sur-taille au plancher minSz : si le notionnel voulu est plus de
# (1+tol)x en-dessous du notionnel minimum d'OKX, on SAUTE le trade plutot que
# de poster un ordre plus gros que voulu (petit solde).
MIN_SIZE_OVERSHOOT_TOL = _env_float("MIN_SIZE_OVERSHOOT_TOL", 0.10)

# ============================================================
#  Horloge : tolerance de derive vs serveur OKX (audit 2026-08-03, point #11)
# ============================================================
# OKX rejette une requete signee a plus de ~30s de son heure serveur. On
# synchronise un offset au demarrage (public/time) et on alerte au-dela de ce
# seuil -- un daemon a l'horloge derivee echouerait SILENCIEUSEMENT toute
# ecriture (order/cancel) une fois en reel.
MAX_CLOCK_SKEW_S = _env_float("MAX_CLOCK_SKEW_S", 20.0)

# Re-synchronisation PERIODIQUE de l'offset horloge dans la boucle (audit
# 2026-08-13) : le sync unique au boot derive sur un daemon tournant des jours
# (NTP par pas, VM suspend) et finit par declencher des rejets 50102 "Timestamp
# request expired". On re-cale l'offset toutes les N minutes -- best-effort,
# n'echoue jamais la boucle. `_request` conserve en plus un self-heal (resync +
# retry) sur 50102 pour rattraper un decalage soudain entre deux resyncs.
CLOCK_RESYNC_MIN = _env_int("CLOCK_RESYNC_MIN", 30)

# ============================================================
#  Cadence de la boucle
# ============================================================
TICK_INTERVAL_S = 60          # 1 tick / minute (monitoring positions)
INTERVAL_1H_MS = 60 * 60 * 1000

# ============================================================
#  Entree MAKER : re-pricing / fenetre de remplissage
# ============================================================
# MAKER_FILL_WINDOW_MIN vient de config_main.py (deja importe la ou necessaire) ;
# ici seulement la cadence de re-pricing PENDANT cette fenetre (pas d'equivalent
# backtest -- le backtest simule un fill binaire, le live doit activement gerer
# l'ordre post-only pendant la fenetre).
# ENTRY_REPRICE_EVERY_S : SUPPRIME (alignement backtest 2026-08-14) -- la
# limite maker reste STATIQUE au prix de depart pendant toute la fenetre
# (semantique backtest "fill seulement si le marche revient toucher P0").

# ============================================================
#  Chemins de logs
# ============================================================
ROOT = Path(__file__).resolve().parent
LOGS_DIR = ROOT / "logs"
CYCLES_CSV = LOGS_DIR / "cycles.csv"
POSITIONS_CSV = LOGS_DIR / "positions.csv"
POSITIONS_STATE_JSON = LOGS_DIR / "positions_state.json"
LIVE_LOG_FILE = LOGS_DIR / "live.log"

# ============================================================
#  Variables d'environnement attendues (documentees ici, lues via os.getenv
#  directement la ou necessaire -- api_okx.py / live.py)
# ============================================================
# OKX_API_KEY, OKX_API_SECRET, OKX_API_PASSPHRASE : credentials OKX (lecture +
#   trading -- portee complete Futures/SWAP necessaire pour place_order).
# DRY_RUN : "true"/"false" -- true = cycle complet mais AUCUN appel d'ecriture OKX.
# TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID : notifications operationnelles.
# TELEGRAM_ERROR_BOT_TOKEN / TELEGRAM_ERROR_CHAT_ID (optionnel) : bot separe
#   pour les logger.error/logger.exception (cf notify.py).


def is_dry_run() -> bool:
    # BUG FIXE (2026-07-15, audit complet) : sans .strip(), une valeur .env avec
    # un espace/CRLF parasite (ex. "DRY_RUN=true " ou un artefact copier-coller
    # Windows) ne matchait AUCUNE des valeurs de la liste -- basculait donc
    # SILENCIEUSEMENT en mode REEL (False) sans la moindre erreur, alors que
    # l'operateur croit etre en paper trading. Aucun crash, aucun log -- juste
    # de l'argent reel engage par surprise. Trouvaille la plus severe de l'audit.
    return os.getenv("DRY_RUN", "true").strip().lower() in ("1", "true", "yes", "y", "on")
