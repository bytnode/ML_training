# Design — équipe modele_updown 30m (clone strict du 15min)

**Date** : 2026-05-14
**Statut** : design validé, en attente du plan d'implémentation

## Objectif

Créer un dossier `backtest/modele_updown/30m/` qui est un clone strict de `backtest/modele_updown/15min/`, adapté pour prédire la direction de la prochaine bougie 30 minutes (au lieu de 15 minutes). Les deux équipes (15min et 30m) cohabitent indépendamment.

Périmètre : modèles + team backtest uniquement. Pas de live (paper ou polymarket) dans cette passe — sera traité dans une 2ème passe une fois validé que les WR sont bons sur 30m.

## Sémantique du modèle 30m

- Pas de temps de base : bougies 30 minutes BTC/USD Bitstamp.
- Label de `modele_up` : `1` si la prochaine bougie 30m est verte (`close[i+1] > open[i+1]`), `0` sinon.
- Label de `modele_down` : `1` si la prochaine bougie 30m est rouge (`close[i+1] < open[i+1]`), `0` sinon.
- Boucle de décision théorique : fin de bougie 30m → prédiction sur la prochaine.

## Arborescence finale

```
backtest/modele_updown/
  15min/                    INTACT, on ne touche à rien
    modele_up/
    modele_down/
  30m/                      NOUVEAU (équipe 30m, miroir exact de 15min/)
    modele_up/              clone de 15min/modele_up/
    modele_down/            clone de 15min/modele_down/
    main.py                 clone de modele_updown/main.py, pointe vers 30m/
    logs/                   créé par main.py (team_bets.csv + team_runs.csv)
  main.py                   INTACT (orchestre toujours l'équipe 15min)
  live/                     INTACT
```

Chaque équipe est auto-contenue dans son dossier. Pour lancer le team backtest 30m :
`cd backtest\modele_updown\30m && python main.py`.

## Substitutions à appliquer dans le clone

Quatre catégories de changements, tout le reste est strictement copié-collé.

| Catégorie | 15min | 30m |
|---|---|---|
| INTERVAL de base | `"15m"` | `"30m"` |
| DB_PATH | `DB/btc_multi_tf_15m.parquet` | `DB/btc_multi_tf_30m.parquet` |
| MTF (higher TFs) | `["30m", "1h", "4h", "1d"]` | `["1h", "4h", "1d"]` (on enlève le 30m qui devient le pas de base, sinon lookahead trivial) |
| Contexte fin (modele_up `data.py` + modele_down `fetch_and_add_5m.py`) | `5m` | `15m` (la TF immédiatement plus fine que le pas de base, exactement comme `5m` joue ce rôle pour 15m) |

Le label, la coherence rule, le sidecar shape, la stratégie adaptative
(`asym_wknd_excl_hour_boost`), les hyperparamètres XGBoost, le warmup, les buckets
de volatilité, les patches `add_*.py` : tout est identique mot pour mot.

## Composants à cloner — liste exhaustive

> **État réel du repo (working tree)** : les patches `add_*.py`, `patch_body_signed.py`, `fetch_and_add_5m.py`, `add_tier1_features.py`, `drop_redundant_features.py`, `asymmetry_diagnostic.py` mentionnés dans les CLAUDE.md package-scoped **n'existent pas dans le repo**. Ils ont été consolidés dans `15min/modele_down/data.py` (docstring : "Ce fichier remplace l'ancienne chaîne…"). De plus, `15min/modele_up/` n'a **pas de `data.py`** : son parquet est une copie de celui de `modele_down` avec le label flippé (mécanisme documenté dans le trap section de `15min/modele_up/CLAUDE.md`).

### `30m/modele_up/` — clone intégral de `15min/modele_up/` (8 fichiers)

- `CLAUDE.md` — adapté : substitution textuelle "15m"/"15min" → "30m" dans le texte.
- `config.py` — substitutions de la table ci-dessus (INTERVAL, DB_PATH, MTF_INTERVALS).
- `features.py` — runtime feature builder, lit le parquet. Aucune logique métier ne change, juste le nom du parquet.
- `train.py` — strictement copié, lit `config.py`.
- `predict.py` — strictement copié.
- `api.py` — strictement copié, expose le même contrat `NAME` + `evaluate_on_slice(start_ms, end_ms)`. Les messages d'erreur qui mentionnent `python data.py` deviennent `python ../modele_down/data.py` (puisque modele_up n'a pas son propre `data.py`).
- `analyse/analyze_features.py` — strictement copié.
- `analyse/check_new_features.py` — strictement copié.

**Modèle de remplissage du parquet de `30m/modele_up/`** : exactement comme côté 15min, après que `30m/modele_down/data.py` ait produit son parquet, on le copie dans `30m/modele_up/DB/` et on inverse le label (`df["label"] = 1 - df["label"]`). Pas de `data.py` à créer dans `30m/modele_up/`.

### `30m/modele_down/` — clone intégral de `15min/modele_down/` (24 fichiers)

- `CLAUDE.md` — adapté avec substitutions textuelles. Supprimer la section "Build pipeline — order matters" qui décrit la chaîne legacy de patches obsolètes, et la remplacer par la pipeline mono-script actuelle (`python data.py` puis `python train.py`).
- `config.py` — substitutions de la table.
- `data.py` — **le seul fichier du pipeline data, consolidé**. Changements :
  - TIMEFRAMES dict : passe de `[5m, 15m, 30m, 1h, 4h, 1d]` (6 TFs) à `[15m, 30m, 1h, 4h, 1d]` (5 TFs) — on drop le 5m.
  - Base TF dans la construction du master : `15m` → `30m`. Toutes les références à "master 15m", "5m close-aligned", etc. dans le docstring et le code sont mises à jour.
  - Suffixes de colonnes : `_5m` → `_15m` (fine context), `_30m` disparaît (devient la base, pas de suffixe), MTF reste `[1h, 4h, 1d]`.
  - Sortie : `DB/btc_multi_tf_30m.parquet` (au lieu de `_15m.parquet`).
  - Le label cible passe à la prochaine bougie 30m au lieu de 15m.
- `features.py` — strictement copié, lit le parquet 30m.
- `train.py` — strictement copié, lit `config.py`.
- `predict.py` — strictement copié.
- `api.py` — strictement copié.
- `analyse/00_synthese_finale.md` — copié tel quel (référence historique 15m).
- `analyse/01..08_*.{py,md}` (8 paires) — copiés tels quels. Leurs conclusions sont sur le 15m, à re-runner sur 30m une fois entraîné si on veut une re-validation.
- `analyse/analyze_features.py`, `analyse/check_new_features.py`, `analyse/compare_runs.py` — strictement copiés.

### `30m/main.py` — clone de `modele_updown/main.py`

Quatre lignes changent :

- `_15MIN_DIR = Path(__file__).resolve().parent / "15min"` devient simplement
  `_30M_DIR = Path(__file__).resolve().parent`. `main.py` étant maintenant DANS
  `30m/`, son parent IS `30m/` — pas de sous-dossier à descendre.
- Les imports `from modele_up import api as api_up` et `from modele_down import
  api as api_down` restent identiques (l'astuce sys.path fonctionne pareil).
- Le panel Rich affiche `"TEAM modele_updown 30m"` au lieu de
  `"TEAM modele_updown"` (cosmétique, optionnel mais utile pour distinguer les
  rapports des deux équipes).
- `LOGS_DIR = ROOT / "logs"` reste pareil — les logs vont dans `30m/logs/`.

## Le piège du `sys.path` trick

`30m/` commence par un chiffre exactement comme `15min/`, donc
`from 30m.modele_up import api` est aussi un SyntaxError Python. Le trick
`sys.path.insert(0, str(30m_dir))` puis `from modele_up import api` continue de
marcher.

Conséquence importante : si on ajoute un jour `15min/main.py` (n'existe pas
aujourd'hui) ET qu'on a `30m/main.py` qui font tous les deux
`from modele_up import api`, ils résolvent vers des paquets différents selon
quel sys.path est inséré en premier. Pas de problème aujourd'hui parce qu'on
lance UN seul orchestrateur à la fois (`cd 30m && python main.py`), mais à
garder en tête. **Ne pas pré-importer les deux dans un même processus.**

## Flux de données

```
30m/modele_down/
  python data.py        → DB/btc_multi_tf.db + DB/btc_multi_tf_30m.parquet
                          (cold fetch 5 TFs ~3-5 min, tout consolidé en un script)
  python train.py       → btc_updown_model.json + .thresholds.json
                          + logs/runs.csv + logs/test_predictions.csv

30m/modele_up/
  (pas de data.py — on copie le parquet du frère et on inverse le label)
  copy 30m/modele_down/DB/btc_multi_tf_30m.parquet → 30m/modele_up/DB/
  flip label : df["label"] = 1 - df["label"]
  python train.py       → btc_updown_model.json + sidecar + logs/

30m/
  python main.py        → lit les test_predictions.csv des deux paquets,
                          calcule la fenêtre commune, applique la coherence
                          rule, écrit logs/team_bets.csv + logs/team_runs.csv,
                          imprime le panel Rich
```

## Points de friction prévus

1. **Substitution des suffixes de colonnes dans `data.py`** : c'est le seul
   travail "non-trivial" du clone. Tous les `_5m` deviennent `_15m`. Toutes
   les références à `_30m` disparaissent (le 30m est maintenant la base, pas
   suffixé). Le set MTF passe de 4 TFs à 3 TFs (`[1h, 4h, 1d]`). Vérifier
   que tous les sites où `"5m"`, `"15m"`, `"30m"` apparaissent comme chaînes
   ou clés dans `TIMEFRAMES` / boucles sont bien mis à jour.

2. **WARMUP** : reste à 500 bars. Sur 30m c'est 10.4 jours au lieu de 5.2 —
   toujours raisonnable pour calibrer un `ATR_LONG_PERIOD=100`. Pas de
   modification.

3. **TRAIN_DAYS / TEST_DAYS** : on les laisse aux valeurs actuelles (3600 / 366
   pour up, 4000 / 365 pour down). Sur 30m on a deux fois moins de bars par
   jour, donc l'entraînement aura ~moitié moins d'exemples, mais la couverture
   temporelle reste identique. C'est le comportement attendu d'un "clone à
   l'identique".

4. **`logs/runs.csv` legacy columns** : les nouveaux fichiers sont vierges, pas
   de cohabitation V1/V2 à gérer. C'est plus simple que le 15min sur ce point.

5. **CLAUDE.md package-scoped legacy** : les CLAUDE.md de `15min/modele_up/`
   et `15min/modele_down/` mentionnent des scripts (`add_*.py`,
   `patch_body_signed.py`, `fetch_and_add_5m.py`, etc.) qui n'existent plus
   dans le repo. Quand on copie ces CLAUDE.md vers `30m/`, on profite de
   l'occasion pour nettoyer ces références obsolètes — sans toucher au
   contenu du `15min/` lui-même.

6. **Fichiers gitignored** : `.gitignore` racine couvre déjà `*.csv`,
   `*.parquet`, `*.db`, `btc_updown_model.json`, `btc_updown_model.thresholds.json`
   — donc rien à ajouter.

## Validation

Après le premier run end-to-end (data.py + train.py × 2 paquets + main.py) :

- `30m/modele_up/logs/runs.csv` contient au moins une ligne avec
  `adaptive_total_taken > 0` et `adaptive_total_wr > baseline_wr_up`.
- `30m/modele_down/logs/runs.csv` même chose côté DOWN.
- `30m/logs/team_runs.csv` contient une ligne avec `team_wr` cohérent.
- Le panel Rich de `30m/main.py` s'affiche sans erreur.
- Aucun fichier de `15min/` n'a été modifié (`git status` propre côté `15min/`).

## Non-objectifs explicites

- Pas de live (paper ou polymarket) pour le 30m dans cette passe.
- Pas de modification de `modele_updown/main.py` ni de `modele_updown/live/`.
- Pas d'optimisation des hyperparamètres XGBoost pour le 30m — on garde les
  mêmes valeurs que le 15m comme baseline, à tuner ensuite si les WR sont
  décevants.
- Pas de re-audit des features sur 30m via `analyse/check_new_features.py` —
  on garde le set issu du 15m comme baseline, à auditer plus tard si les WR
  sont décevants.
- Pas de refactor symétrique (déplacer `modele_updown/main.py` dans
  `15min/main.py`). Le 15min reste à l'emplacement actuel, l'asymétrie est
  assumée.
