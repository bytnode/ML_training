# Live paper-trading container for the modele_updown team.
#
# Build context = backtest/modele_updown/. Run from there:
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
RUN pip install --no-cache-dir --timeout 300 --retries 5 \
        requests==2.32.* \
        pandas==2.2.* \
        numpy==2.1.* \
        xgboost==2.1.* \
        scikit-learn==1.5.* \
        scipy==1.14.* \
        rich==13.*

# --- Application code ---
COPY live.py ./live.py
COPY live_helpers/ ./live_helpers/

# --- Model artifacts + minimal package code needed at runtime ---
# The two sister packages live under 15min/. The directory name starts with a
# digit (not a valid Python identifier) so consumers add 15min/ to sys.path.
# modele_up : config + model + sidecar
COPY 15min/modele_up/config.py                       ./15min/modele_up/config.py
COPY 15min/modele_up/btc_updown_model.json           ./15min/modele_up/btc_updown_model.json
COPY 15min/modele_up/btc_updown_model.thresholds.json ./15min/modele_up/btc_updown_model.thresholds.json

# modele_down : config + data.py (used by features_down via sys.path) + model + sidecar
COPY 15min/modele_down/config.py                      ./15min/modele_down/config.py
COPY 15min/modele_down/data.py                        ./15min/modele_down/data.py
COPY 15min/modele_down/btc_updown_model.json          ./15min/modele_down/btc_updown_model.json
COPY 15min/modele_down/btc_updown_model.thresholds.json ./15min/modele_down/btc_updown_model.thresholds.json

# Logs directory (mounted as a host volume in docker-compose.yml)
RUN mkdir -p /app/logs

# Non-root user for safety
RUN useradd -m -u 1000 live && chown -R live:live /app
USER live

CMD ["python", "live.py"]
