# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in `live/`.

## Commands

```powershell
# Required env vars for Telegram notifs (sinon le daemon tourne mais ne notifie pas)
$env:AURE_TELEGRAM_BOT_TOKEN = "123:abc..."
$env:AURE_TELEGRAM_CHAT_ID = "-100..."

cd live

# Daemon long-running (mode normal)
python orchestrator.py

# Un seul cycle puis exit (test fonctionnel)
python orchestrator.py --dry-run

# Pas de notif Telegram (test sans spammer le chat)
python orchestrator.py --no-notify

# --dry-run + --no-notify combinables
python orchestrator.py --dry-run --no-notify
```

### Lancement dockerisé (recommandé pour la prod)

Pre-requis : `aure.json` + `aure.thresholds.json` doivent exister à la racine du repo (Docker bind-mount des fichiers, ne crée pas si absent). Le `.env` à la racine fournit les credentials Telegram via `env_file:` de compose.

Les commandes marchent **depuis `live/` OU depuis `infra/`** indifféremment — `live/docker-compose.yml` est un pointeur (`include:`) vers `infra/docker-compose.yml`. Source unique de vérité = `infra/docker-compose.yml`.

```powershell
# Depuis live/ (le plus pratique vu que tu y es deja)
docker compose up -d --build      # build + start en arriere-plan
docker compose logs -f aure-live  # suivre les logs
docker compose restart aure-live  # apres avoir remplace aure.json sur l'host
docker compose down               # stop + remove
```

Modèle / sidecar / `live/logs/` sont montés en volume depuis l'host : pas de rebuild quand tu retraines, juste un `docker compose restart`.

## Architecture

Daemon Python long-running aligné sur les frontières 15min UTC. Chaque cycle :
1. Sleep jusqu'à la prochaine frontière 15m pile (`CYCLE_FUDGE_SECONDS = 0`, ex: 12:15:00 UTC).
2. Boucle de freshness retry : refetch jusqu'à obtenir une bougie 15m strictement plus récente que la précédente (preuve que Bitstamp a bien publié la nouvelle bougie qui vient de fermer). 1s entre essais, max 30 tentatives. Si épuisé, cycle skippé (log WARN) et on attend la frontière suivante.
3. Fetch les 5 dernières bougies 15m + 3 par MTF (30m/1h/4h/1d) Bitstamp.
4. Resolve les rows pendantes (`label=NaN` dans `training_data_live.csv`, `outcome=pending` dans `decisions.csv`) si la bougie cible est maintenant fermée.
5. Détecte les setups qualifiants au dernier index 15m (streak 1-4 same-dir + `body_ratio_last ≥ 0.6`), puis applique le filtre `SEQ_LIVE` (cf. ci-dessous).
6. Pour chaque setup retenu : extrait 42 features, append `training_data_live.csv` (label=NaN), prédit avec Aure, applique le seuil adaptatif du sidecar, append `decisions.csv`, envoie Telegram si `taken=True`.

## Filtre `SEQ_LIVE`

`live/settings.py:SEQ_LIVE` contrôle sur quelles longueurs de streak Aure tire :
- `SEQ_LIVE = [2, 3, 4]` (défaut, recommandé) — ne tire qu'à partir de 2 bougies same-dir consécutives. Cohérent avec `backtest/config.py:SEQ_TEST` quand on calibre live ↔ backtest.
- `SEQ_LIVE = [1]` — test isolé sur les setups single-candle (instable, beaucoup plus de paris).
- `SEQ_LIVE = None` ou `[]` — pas de filtre, toutes les seq_len 1-4 tirent.

Changement à chaud : édite `live/settings.py` puis `python orchestrator.py` redémarre. **En mode dockerisé, settings.py est copié dans l'image au build** — il faut `docker compose up -d --build` (un simple `restart` ne suffit pas, c'est intentionnel pour ne pas qu'un settings.py modifié à chaud parte en prod sans review).

## Module collision avec backtest/

`live/settings.py` (et **non `config.py`**) parce qu'on insère `backtest/` dans `sys.path` pour réutiliser `backtest/data.py`, ce qui shadowerait `live/config.py` aux imports `from config import ...` dans le reste de `live/`. Voir le commentaire en tête de `live/settings.py`.

## Files produits

- `live/logs/training_data_live.csv` — 44 cols, schema identique à `backtest/training_data.csv`. Réutilisable pour ré-entraîner Aure (drop pas si tu veux).
- `live/logs/decisions.csv` — 53 cols, audit self-contained des décisions (features + confidence + threshold + outcome).
- `live/logs/orchestrator.log` — log texte un événement par ligne. Lisible en `Get-Content -Wait`.

## Sidecar threshold

`aure.thresholds.json` à la racine est créé au 1er run en copiant `backtest/btc_reversal_model.thresholds.json`. Si tu réentraînes Aure, copie le nouveau sidecar à la main au-dessus, ou supprime `aure.thresholds.json` pour forcer la recopie au prochain démarrage. Le daemon lit le sidecar UNE FOIS au démarrage (cache mémoire) ; pas de hot reload.

## Restart / outage

Au démarrage le daemon fetch ~200 dernières bougies 15m + 50 MTF. Pendings dont la bougie cible est dans cette fenêtre sont résolus automatiquement. Au-delà (daemon down >50h) les pendings restent `pending`/`NaN` indéfiniment — relancer `backtest/data.py` pour régénérer si besoin.
