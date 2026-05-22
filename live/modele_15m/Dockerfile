# Live trading container for the modele_updown 15min team.
#
# This folder (live/modele_15m/) is self-contained: it carries the live code,
# the model artifacts, and everything needed to run. The build context is THIS
# directory - no parent-directory hack. Run from here:
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
# The last 3 lines (py-clob-client-v2 / web3 / eth-account) are only USED when
# LIVE_MODE=true, but they're installed unconditionally so the same image works
# in both modes. Lazy imports in live.py mean paper mode never loads them.
#
# py-clob-client-v2 (NOT the v1 py-clob-client): Polymarket migrated to CLOB V2
# on 2026-04-28; the v1 SDK returns `order_version_mismatch` on every order
# against production. V2 is mandatory.
RUN pip install --no-cache-dir --timeout 300 --retries 5 \
        requests==2.32.* \
        pandas==2.2.* \
        numpy==2.1.* \
        xgboost==2.1.* \
        scikit-learn==1.5.* \
        scipy==1.14.* \
        rich==13.* \
        py-clob-client-v2==1.0.* \
        web3==7.* \
        eth-account==0.13.*

# --- Application code + model artifacts ---
# Build context IS this folder, so the layout is copied flat into /app/. live.py
# resolves ROOT = /app and the models at /app/modele_up|down/; features_down.py
# resolves modele_down/ via parents[1] = /app. Host and container match.
COPY live.py        ./live.py
COPY polymarket.py  ./polymarket.py
COPY live_helpers/  ./live_helpers/
COPY modele_up/     ./modele_up/
COPY modele_down/   ./modele_down/

# Logs directory (mounted as a host volume in docker-compose.yml).
# Matches live.py's LOGS_DIR = ROOT / "logs"  (ROOT = /app).
RUN mkdir -p /app/logs

# Non-root user for safety
RUN useradd -m -u 1000 live && chown -R live:live /app
USER live

CMD ["python", "live.py"]
