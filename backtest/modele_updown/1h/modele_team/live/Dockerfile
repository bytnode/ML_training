# Build context = live/ lui-meme -- AUTONOMIE COMPLETE (2026-08-03) : live/ n'a
# plus AUCUNE dependance a modele_team/ ni a 1h/ -- config_main.py et filters.py
# sont maintenant des copies embarquees (comme modele_up/, modele_down/, regime/
# deja depuis 2026-07-10). Deploiement VPS = copier ce dossier (live/) seul.
FROM python:3.12-slim

WORKDIR /app

# xgboost EPINGLE (3.2.*) : un mismatch de version entre l'entrainement (host)
# et ce container casse la parite des predictions (format de serialisation
# base_score change entre versions majeures/mineures de xgboost) -- meme
# contrainte que l'ancien live_team documentait deja. Verifie/ajuste cette
# version si tu reentraines les modeles avec un xgboost different.
RUN pip install --no-cache-dir \
    pandas numpy "xgboost==3.2.*" scikit-learn scipy requests

# Tout est deja dans live/ (contexte = ce dossier) : live.py, position.py,
# config_main.py, filters.py, modele_up/, modele_down/, regime/, etc.
COPY . /app/

# .env n'est PAS copie (secrets -- charges via env_file de compose).
RUN rm -f /app/.env

CMD ["python", "live.py"]
