# Live Polymarket trader — image Docker.
#
# Build context = racine du repo (cf. infra/docker-compose.yml: context: ..).
# Le code de live/ et backtest/modele_15m/ est copie dans l'image.
# Le modele (btc_reversal_model.json), le sidecar (...thresholds.json) et les logs
# sont fournis a l'execution via des bind-mounts geres par docker-compose.

FROM python:3.10-slim

WORKDIR /app

# Deps Python (pas de requirements.txt par convention du projet — listees inline).
RUN pip install --no-cache-dir \
    requests \
    pandas \
    numpy \
    xgboost \
    scikit-learn \
    py-clob-client \
    web3 \
    eth-account \
    python-dotenv

# Code uniquement (le modele et les CSVs viennent par volumes).
COPY backtest/ /app/backtest/
COPY live/ /app/live/

WORKDIR /app/live

CMD ["python", "orchestrator.py"]
