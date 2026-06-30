# Daemon live MEXC futures — team 1h. Build context = ce dossier (self-contained).
FROM python:3.12-slim

# Logs non bufferisés (visibles immédiatement dans `docker logs`) ; horloge UTC
# (le daemon raisonne en epoch ms, mais UTC explicite évite toute ambiguïté).
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=UTC

WORKDIR /app

# xgboost ÉPINGLÉ 3.2.* (parité base_score avec l'env d'entraînement ; skew connu sinon).
RUN pip install --no-cache-dir \
    requests pandas numpy "xgboost==3.2.*" scikit-learn scipy rich python-dotenv

# Copie à plat (live.py, mexc.py, position.py, config_team.py, live_helpers/, modele_up/, modele_down/).
# Le .dockerignore (Dockerfile.dockerignore) exclut logs/, .env, parquet/db ; les
# artefacts modèle (btc_updown_model.json + .thresholds.json) sont bien copiés.
COPY . /app/

# Dossier de logs (monté par docker-compose ; créé ici pour un run sans mount).
RUN mkdir -p /app/logs

# Rich rend ses panneaux correctement avec un tty (docker-compose : tty: true).
CMD ["python", "live.py"]
