"""config_live.py -- constantes SPECIFIQUES au daemon live OKX.

Ne redefinit RIEN de ce qui vit deja dans config_main.py (seuils DMI/UP-filters/
SL/TP/trailing/timestop/levier/exposure) -- ces valeurs sont IMPORTEES depuis
config_main.py partout ou elles sont necessaires. Ce fichier ne contient que ce
qui n'a pas d'equivalent backtest : identifiant d'instrument OKX, cadence de la
boucle, chemins de logs, noms des variables d'env attendues.

COPIE INDEPENDANTE (2026-08-03, campagne autonomie complete live/) : `config_main.py`
est desormais une copie EMBARQUEE dans live/ (live/config_main.py), plus le
fichier partage de modele_team/ -- voir l'entete de live/config_main.py pour le
tradeoff (recopie manuelle apres un changement backtest).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# live/config_live.py -> parent = live/. On pousse live/ sur sys.path pour
# importer config_main comme module top-level (config_main.py est une copie
# EMBARQUEE dans live/ depuis 2026-08-03, plus modele_team/ -- meme trick que
# main.py pousse 1h/ pour modele_up/modele_down, ou live.py pousse live/ pour
# ses propres copies modele_up/modele_down/regime).
_LIVE_DIR = str(Path(__file__).resolve().parent)
if _LIVE_DIR not in sys.path:
    sys.path.insert(0, _LIVE_DIR)

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
ENTRY_REPRICE_EVERY_S = 60    # reprice l'ordre post-only toutes les 60s si non rempli

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
