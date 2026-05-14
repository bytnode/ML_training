# Live trading container for the modele_updown team (15min timeframe).
#
# Build context = backtest/modele_updown/15min/ (one level above this Dockerfile).
# Compose pins this via `build: { context: .., dockerfile: live/Dockerfile }`
# so that COPY can reach the sibling modele_up/ and modele_down/ directories.
# Run from backtest/modele_updown/15min/live/:
#   docker compose build
#   docker compose up -d
#   docker compose logs -f live-team
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Python deps (no requirements.txt by repo convention - kept explicit here).
# --timeout 300 + --retries 5 because xgboost / pandas / scipy wheels are big
# and PyPI can be slow.
#
# The last 3 lines (py-clob-client / web3 / eth-account) are only USED when
# LIVE_MODE=true, but they're installed unconditionally so the same image works
# in both modes. Lazy imports in live.py mean paper mode never loads them.
RUN pip install --no-cache-dir --timeout 300 --retries 5 \
        requests==2.32.* \
        pandas==2.2.* \
        numpy==2.1.* \
        xgboost==2.1.* \
        scikit-learn==1.5.* \
        scipy==1.14.* \
        rich==13.* \
        py-clob-client==0.21.* \
        web3==7.* \
        eth-account==0.13.*

# --- Application code ---
# Paths are relative to the build context (backtest/modele_updown/15min/).
# We mirror the host layout inside /app/15min/ so features_down.py's parents[2]
# trick (sys.path -> modele_down/) resolves identically in both environments.
COPY live/live.py        ./15min/live/live.py
COPY live/polymarket.py  ./15min/live/polymarket.py
COPY live/live_helpers/  ./15min/live/live_helpers/

# --- Model artifacts + minimal package code needed at runtime ---
# Sources are siblings of live/ in the 15min/ build context, no need for a
# leading "15min/" segment here. Destinations DO carry the 15min/ prefix to
# preserve symmetry with the host layout (so features_down's parents[2] hop
# resolves to /app/15min/, with modele_down/ as a sibling).
# modele_up : config + model + sidecar
COPY modele_up/config.py                       ./15min/modele_up/config.py
COPY modele_up/btc_updown_model.json           ./15min/modele_up/btc_updown_model.json
COPY modele_up/btc_updown_model.thresholds.json ./15min/modele_up/btc_updown_model.thresholds.json

# modele_down : config + data.py (used by features_down via sys.path) + model + sidecar
COPY modele_down/config.py                      ./15min/modele_down/config.py
COPY modele_down/data.py                        ./15min/modele_down/data.py
COPY modele_down/btc_updown_model.json          ./15min/modele_down/btc_updown_model.json
COPY modele_down/btc_updown_model.thresholds.json ./15min/modele_down/btc_updown_model.thresholds.json

# Logs directory (mounted as a host volume in docker-compose.yml).
# Matches live.py's LOGS_DIR = ROOT / "logs"  (ROOT = /app/15min/live/).
RUN mkdir -p /app/15min/live/logs

# Non-root user for safety
RUN useradd -m -u 1000 live && chown -R live:live /app
USER live

CMD ["python", "15min/live/live.py"]
