# Aure live orchestrator — image Docker.
#
# Build context = racine du repo (cf. infra/docker-compose.yml: context: ..).
# Le code de live/ et backtest/ est copie dans l'image. Le modele (aure.json),
# le sidecar (aure.thresholds.json) et les logs sont fournis a l'execution
# via des bind-mounts geres par docker-compose.

FROM python:3.10-slim

WORKDIR /app

# Deps Python (pas de requirements.txt par convention du projet — listees inline).
RUN pip install --no-cache-dir \
    requests \
    pandas \
    numpy \
    xgboost \
    scikit-learn

# Code uniquement (le modele et les CSVs viennent par volumes).
COPY backtest/ /app/backtest/
COPY live/ /app/live/

WORKDIR /app/live

CMD ["python", "orchestrator.py"]
