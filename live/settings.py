"""Live trader configuration. All knobs live here.

Note: this is settings.py (not config.py) because we sys.path.insert backtest/modele_15m/
in the imports below, and that module already exports a top-level `config` we don't
want to shadow.
"""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
LOGS_DIR = ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# --- Load .env: live/.env takes priority, then repo-root .env fills in missing vars ---
# (load_dotenv with override=False only sets vars not already in os.environ.)
load_dotenv(ROOT / ".env", override=False)
load_dotenv(REPO_ROOT / ".env", override=False)

# --- Logs ---
DECISIONS_CSV = LOGS_DIR / "decisions.csv"
BETS_CSV = LOGS_DIR / "bets.csv"
WALLET_HISTORY_CSV = LOGS_DIR / "wallet_history.csv"
ORCHESTRATOR_LOG = LOGS_DIR / "orchestrator.log"

# --- Cycle ---
SEQ_LIVE = [2, 3, 4]                    # seq_len values the daemon fires on
FRESHNESS_RETRY_MAX = 12                # 12 retries x 1s = 12s of slack for Bitstamp publishing
FRESHNESS_RETRY_DELAY_S = 1
WARMUP_CANDLES_15M = 250                # need enough history for EMA-200 + obv_slope_10
WARMUP_CANDLES_MTF = 50

# --- Bet sizing ---
BET_FRACTION = 0.10                     # 10% of free USDC per window
BET_MIN_USDC = 1.00                     # Polymarket minimum order
BET_MAX_FRACTION = 0.95                 # safety: refuse to bet > 95% of balance

# --- Settlement reconciliation ---
SETTLEMENT_STUCK_AFTER_CYCLES = 6       # ~90min after window close

# --- Polymarket / Polygon (env vars are mandatory; no defaults for keys) ---
POLYGON_PRIVATE_KEY = os.environ.get("POLYGON_PRIVATE_KEY")
POLYGON_EOA_ADDRESS = os.environ.get("POLYGON_EOA_ADDRESS")
POLYMARKET_PROXY_ADDRESS = os.environ.get("POLYMARKET_PROXY_ADDRESS")
POLYGON_RPC_URL = os.environ.get("POLYGON_RPC_URL", "https://polygon-bor-rpc.publicnode.com")
CLOB_HOST = os.environ.get("CLOB_HOST", "https://clob.polymarket.com")
GAMMA_HOST = os.environ.get("GAMMA_HOST", "https://gamma-api.polymarket.com")
POLYMARKET_API_TIMEOUT = 10
MATIC_LOW_THRESHOLD = 0.5

# --- Telegram (optional; if missing, daemon runs but doesn't notify) ---
TELEGRAM_BOT_TOKEN = os.environ.get("AURE_TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("AURE_TELEGRAM_CHAT_ID")
TELEGRAM_TIMEOUT = 10

# --- Model paths (absolute, used by orchestrator to inject into predict module) ---
MODEL_PATH = REPO_ROOT / "backtest" / "modele_15m" / "btc_reversal_model.json"
SIDECAR_PATH = REPO_ROOT / "backtest" / "modele_15m" / "btc_reversal_model.thresholds.json"

# --- sys.path injection so we can import data.py / predict.py from backtest/modele_15m/ ---
_BACKTEST_MODEL_DIR = str(REPO_ROOT / "backtest" / "modele_15m")
if _BACKTEST_MODEL_DIR not in sys.path:
    sys.path.insert(0, _BACKTEST_MODEL_DIR)


def assert_env_ready():
    """Raise on startup if mandatory env vars are missing."""
    missing = [
        name for name, val in [
            ("POLYGON_PRIVATE_KEY", POLYGON_PRIVATE_KEY),
            ("POLYGON_EOA_ADDRESS", POLYGON_EOA_ADDRESS),
            ("POLYMARKET_PROXY_ADDRESS", POLYMARKET_PROXY_ADDRESS),
        ]
        if not val
    ]
    if missing:
        raise RuntimeError(
            f"Missing required env vars in .env: {missing}. "
            f"See docs/superpowers/specs/2026-05-08-polymarket-live-trader-design.md."
        )
