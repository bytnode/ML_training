"""Configuration locale de l'orchestrateur Aure (live mode).

Important : ce module s'appelle settings.py et non config.py pour eviter
la collision de namespace avec backtest/config.py quand sys.path inclut
backtest/. Voir docs/superpowers/plans/2026-05-07-aure-live-orchestrator.md
section 'Note d'architecture critique'.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
LOGS_DIR = ROOT / "logs"

TRAINING_DATA_LIVE = LOGS_DIR / "training_data_live.csv"
DECISIONS_CSV = LOGS_DIR / "decisions.csv"
ORCHESTRATOR_LOG = LOGS_DIR / "orchestrator.log"

MODEL_PATH = REPO_ROOT / "aure.json"
THRESHOLDS_PATH = REPO_ROOT / "aure.thresholds.json"
THRESHOLDS_SOURCE = REPO_ROOT / "backtest" / "btc_reversal_model.thresholds.json"

# Cycle parameters
WARMUP_CANDLES_15M = 200
WARMUP_CANDLES_MTF = 50
RECENT_FETCH_15M = 5
RECENT_FETCH_MTF = 3
FETCH_RETRIES = 3
FETCH_BACKOFF_BASE = 2  # seconds, exponential

# Telegram
TELEGRAM_BOT_TOKEN_ENV = "AURE_TELEGRAM_BOT_TOKEN"
TELEGRAM_CHAT_ID_ENV = "AURE_TELEGRAM_CHAT_ID"
TELEGRAM_TIMEOUT = 10  # seconds

# 15min en millisecondes (utilise pour calculer next boundary et target_open_time)
INTERVAL_15M_MS = 15 * 60 * 1000

# Marge apres frontiere 15min avant de fetch (laisse le temps a Bitstamp de publier)
CYCLE_FUDGE_SECONDS = 2
