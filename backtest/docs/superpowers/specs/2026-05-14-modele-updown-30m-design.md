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

### `30m/modele_up/` — clone intégral de `15min/modele_up/`

- `config.py` — substitutions de la table ci-dessus.
- `data.py` — cold fetch + features. Affecté : `INTERVAL`, `MTF_INTERVALS`, le set
  de TFs téléchargées de Bitstamp (passe de [5m, 15m, 30m, 1h, 4h, 1d] à
  [15m, 30m, 1h, 4h, 1d]), la TF de contexte fin (5m → 15m), le nom du parquet
  final.
- `features.py` — runtime feature builder, lit le parquet. Aucune logique métier
  ne change, juste le nom du parquet.
- `train.py` — strictement copié, lit `config.py`.
- `predict.py` — strictement copié.
- `api.py` — strictement copié, expose le même contrat `NAME` +
  `evaluate_on_slice(start_ms, end_ms)`.
- `analyse/` — strictement copié (`check_new_features.py`, `analyze_features.py`,
  `compare_runs.py`).
- `CLAUDE.md` — adapté : substitution textuelle "15m"/"15min" → "30m" dans le
  texte, garde la structure exacte.
- Les `add_*.py` patches strictement copiés :
  `add_acceleration.py`, `add_adx.py`, `add_autocorr.py`, `add_range_expansion.py`,
  `add_structural_pressure.py`, `add_upside_velocity.py`, `add_up_down_ratio.py`,
  `add_downside_velocity.py`.

### `30m/modele_down/` — clone intégral de `15min/modele_down/`

Pipeline CSV-based (pas parquet) avec les patches en chaîne :

- `data.py` — cold fetch + CSVs (un par TF) + SQLite. Côté 15min il fetch
  5 TFs `[15m, 30m, 1h, 4h, 1d]` (le 5m est ajouté plus tard par
  `fetch_and_add_5m.py`). Côté 30m, il fetch 4 TFs `[30m, 1h, 4h, 1d]` (le
  15m sera ajouté plus tard par `fetch_and_add_15m.py`). INTERVAL passe à
  `"30m"`, le CSV de base devient `db/btc_multi_tf_30m.csv`, et le MTF passe
  à `[1h, 4h, 1d]`.
- `patch_body_signed.py` — strictement copié.
- `fetch_and_add_5m.py` → renommé en `fetch_and_add_15m.py`. Toutes les
  occurrences `"5m"` → `"15m"` dans le code et dans les noms de colonnes
  suffixées (`_5m` → `_15m`).
- `add_tier1_features.py` — strictement copié, mais les colonnes MTF jointes
  changent (passent de `[30m, 1h, 4h, 1d]` à `[1h, 4h, 1d]`).
- `drop_redundant_features.py` — strictement copié. À auditer après le premier
  `train.py` 30m parce que la liste de features droppées (5 dups, 6 corrélations
  fortes, 9 weak) a été établie sur le set 15m — certaines décisions peuvent
  différer sur 30m. **Note d'implémentation** : copier tel quel d'abord, ne pas
  ajuster la liste avant d'avoir un premier run et un `analyze_features.py`
  exécuté sur le parquet 30m.
- `train.py`, `predict.py`, `api.py`, `features.py`, `config.py` — strictement
  copiés (avec les substitutions de la table).
- `analyse/` (les 8 fichiers d'audit `01_*` à `08_*` + `00_synthese_finale.md`) —
  copiés tels quels. Leurs conclusions sont sur le 15m, on les garde comme
  référence historique mais ils ne devront PAS être consultés comme vérité
  pour le 30m tant qu'on ne les a pas re-runnés.
- `asymmetry_diagnostic.py`, `analyze_features.py` — strictement copiés.
- `CLAUDE.md` — adapté avec substitutions textuelles.

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
30m/modele_up/
  python data.py        → DB/btc_multi_tf_30m.parquet (cold fetch ~3-5 min)
  python add_*.py × 8   → enrichit le parquet avec les features patches
  python train.py       → btc_updown_model.json + .thresholds.json
                          + logs/runs.csv + logs/test_predictions.csv

30m/modele_down/
  python data.py        → DB/btc_multi_tf.db + 5 CSVs (un par TF)
  python patch_body_signed.py
  python fetch_and_add_15m.py    (≡ fetch_and_add_5m.py côté 15min)
  python add_tier1_features.py
  python drop_redundant_features.py
  python train.py       → btc_updown_model.json + sidecar + logs/

30m/
  python main.py        → lit les test_predictions.csv des deux paquets,
                          calcule la fenêtre commune, applique la coherence
                          rule, écrit logs/team_bets.csv + logs/team_runs.csv,
                          imprime le panel Rich
```

## Points de friction prévus

1. **Bitstamp 15m fetch dans modele_down** : aujourd'hui `fetch_and_add_5m.py`
   ajoute 5m comme TF de contexte plus fine. Pour le 30m, ce sera
   `fetch_and_add_15m.py` ajoutant 15m. C'est un rename mécanique de
   variables + chaîne `"5m"` → `"15m"`. Vérifier que les colonnes suffixées
   `_5m` deviennent bien `_15m` partout et que `add_tier1_features.py`
   (qui consomme ces colonnes) est mis à jour en cohérence.

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

5. **`drop_redundant_features.py` côté modele_down** : sa liste de features à
   dropper a été calibrée sur le set 15m. La copier telle quelle, mais marquer
   un TODO dans le `CLAUDE.md` du paquet : "la liste de features droppées a
   été héritée du 15m, à re-auditer après le premier `train.py` 30m via
   `analyze_features.py`".

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
- Pas de re-calibration de la liste de features droppées dans
  `drop_redundant_features.py` côté modele_down — héritée du 15m, à auditer
  plus tard.
- Pas de refactor symétrique (déplacer `modele_updown/main.py` dans
  `15min/main.py`). Le 15min reste à l'emplacement actuel, l'asymétrie est
  assumée.
