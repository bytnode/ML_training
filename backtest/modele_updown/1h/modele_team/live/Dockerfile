# Build context = modele_team/ (voir docker-compose.yml) -- AUTONOMIE (2026-07-10) :
# live/ embarque ses propres copies de modele_up/, modele_down/, regime/
# (live/modele_up/, live/modele_down/, live/regime/, cf leurs docstrings),
# donc ce Dockerfile n'a plus besoin de remonter jusqu'a 1h/ pour les COPY --
# un seul dossier (modele_team/) suffit a un deploiement complet.
FROM python:3.12-slim

WORKDIR /app

# xgboost EPINGLE (3.2.*) : un mismatch de version entre l'entrainement (host)
# et ce container casse la parite des predictions (format de serialisation
# base_score change entre versions majeures/mineures de xgboost) -- meme
# contrainte que l'ancien live_team documentait deja. Verifie/ajuste cette
# version si tu reentraines les modeles avec un xgboost different.
RUN pip install --no-cache-dir \
    pandas numpy "xgboost==3.2.*" scikit-learn scipy requests

# config_main.py/filters.py/okx_fees.py sont sibling de live/ (dans modele_team/,
# racine du contexte) ; live/ lui-meme embarque tout le reste (modele_up/,
# modele_down/, regime/ inclus).
COPY config_main.py filters.py okx_fees.py /app/modele_team/
COPY live/ /app/modele_team/live/

# live/.env n'est PAS copie (secrets -- charges via env_file de compose).
RUN rm -f /app/modele_team/live/.env

WORKDIR /app/modele_team/live
CMD ["python", "live.py"]
