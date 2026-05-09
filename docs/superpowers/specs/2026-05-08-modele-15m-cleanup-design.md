# Spec : Cleanup `backtest/modele_15m/` — adaptive-threshold only

**Date** : 2026-05-08
**Statut** : approuvé

## Problème

Le package `backtest/modele_15m/` contient trois catégories de code :

1. **Runtime** (utilisé pour produire et servir le modèle) : `config.py`, `data.py`, `train.py`, `predict.py`, `api.py`.
2. **Tooling indépendant** (jamais importé par le runtime) : `sweep.py`, `analyze_runs.py`, `clean_data/analyze.py`, `clean_data/clean.py`.
3. **Artefacts/docs** : `README.md`, `DOCUMENTATION.md`, `clean_data/analysis_report.{md,json}`, `clean_data/cleaning_log.md`.

Par ailleurs, le code de décision (`predict.py`, `api.py`) et les sorties d'évaluation (`train.py`, `logs/runs.csv`) supportent **deux** chemins de seuil : un seuil fixe legacy (`BREAKDOWN_THRESHOLD`) et un seuil adaptatif par bucket `(volatility_state, is_weekend)`. Le seuil fixe sert de fallback silencieux quand le sidecar JSON est absent, et alimente une grosse partie de l'output d'éval (`THRESHOLDS` sweep table, `breakdown_total_*`, colonnes `wr_at_<th>` dans `runs.csv`).

Objectif : ramener le package à du runtime pur, avec un seul chemin de décision (adaptatif), et des docs qui ne décrivent que ce qui existe.

## Décisions clés

| # | Sujet | Choix |
|---|---|---|
| 1 | Tooling indépendant | Suppression complète : `sweep.py`, `analyze_runs.py`, dossier `clean_data/` |
| 2 | Profondeur du cut adaptatif | Adaptive-only partout : décision **et** évaluation |
| 3 | Docs package | Réécriture slim de `README.md` et `DOCUMENTATION.md` |
| 4 | Sidecar manquant | `FileNotFoundError` (plus de fallback silencieux) |
| 5 | `runs.csv` | Schéma breaking → fichier supprimé, régénéré au prochain run |

## Périmètre

### Suppressions complètes

```
backtest/modele_15m/
├── sweep.py                            ❌ delete
├── analyze_runs.py                     ❌ delete
├── clean_data/                         ❌ delete (whole dir)
│   ├── analyze.py
│   ├── clean.py
│   ├── analysis_report.json
│   ├── analysis_report.md
│   └── cleaning_log.md
├── logs/
│   └── runs.csv                        ❌ delete (regenerable, schéma breaking)
└── __pycache__/                        ❌ delete (regenerable, références modules supprimés)
```

### Fichiers édités (runtime)

#### `config.py`

**Supprimer** :
- `THRESHOLDS = [0.50, 0.52, ...]`
- `BREAKDOWN_THRESHOLD = 0.6`
- `ADAPTIVE_THRESHOLD = True` (le flag on/off — l'adaptatif est toujours actif)

**Conserver** : toute la SECTION 1 — DATA (consommée par `data.py`) ; dans la SECTION 2, garder `CSV_PATH`, `MODEL_OUTPUT`, `TRAIN_DAYS`, `TEST_DAYS`, `VAL_FRACTION`, `SEQ_TEST`, la famille `ADAPTIVE_*` sauf le flag, et les `XGB_*`.

#### `predict.py`

**Supprimer** :
- Import de `BREAKDOWN_THRESHOLD`.
- Branche `if threshold is not None: return proba >= threshold` (le paramètre `threshold` disparaît).
- Branche `if sidecar is None: return proba >= BREAKDOWN_THRESHOLD` → remplacée par `raise FileNotFoundError(...)` avec un message qui pointe vers `python train.py`.
- La signature devient `should_take_bet(features) -> np.ndarray[bool]`.

**Conserver** : `predict_proba`, `get_model`, `get_thresholds`, le cache module-level `_model`/`_thresholds`, la lecture des features depuis `model.get_booster().feature_names`.

**Smoke test `__main__`** : réécrit pour afficher uniquement le cas adaptatif (proba + seuil effectif + take). Plus de branche legacy.

#### `api.py`

**Supprimer** :
- Import de `BREAKDOWN_THRESHOLD`.
- `_FALLBACK_THRESHOLD = BREAKDOWN_THRESHOLD`.
- Branche `else: df["_eff_th"] = _FALLBACK_THRESHOLD` (sidecar manquant) → remplacée par `raise FileNotFoundError(...)`.

**Conserver** : la signature `evaluate_on_slice(start_ms, end_ms) -> pd.DataFrame`, la dérivation `direction = "down" if seq_dir==1 else "up"`, le filtre `df["confidence"] >= df["_eff_th"]`.

#### `train.py`

**Supprimer** :
- Imports : `THRESHOLDS`, `BREAKDOWN_THRESHOLD`, `ADAPTIVE_THRESHOLD`.
- `LOG_THRESHOLDS = [0.50, 0.55, 0.58, 0.60, 0.62]`.
- La boucle qui produit le tableau « Seuil / Paris pris / Win rate / % volume conservé » et son print.
- La section « Breakdown par longueur de séquence (seuil fixe) » qui utilise `BREAKDOWN_THRESHOLD`.
- Toutes les colonnes `wr_at_<th>`, `n_at_<th>`, `pct_at_<th>` écrites dans `runs.csv`.
- Toutes les colonnes `breakdown_total_taken`, `breakdown_total_wr` (et leurs équivalents par seq_len fixés sur le seuil fixe).
- Tous les `if ADAPTIVE_THRESHOLD:` guards (le bloc devient inconditionnel).

**Conserver** :
- Split temporel `TRAIN_DAYS` / `TEST_DAYS` / `VAL_FRACTION`, filtre `SEQ_TEST` côté test.
- Baseline (`y_test.mean()`) — non thread-pendant des seuils, sert de référence.
- Tout le bloc adaptatif : `compute_adaptive_thresholds`, `_compute_bucket_threshold`, `apply_adaptive_threshold`, `save_thresholds_sidecar`, le tableau de calibration par bucket imprimé en console.
- Colonnes `runs.csv` : `adaptive_total_taken`, `adaptive_total_wr`, `delta_v{0,1,2}_w{0,1}`, `adaptive_calib_days`, `adaptive_target_wr`, `wr_seq<n>` / `n_seq<n>` (sémantique repurposée — calculées sous décision adaptative au lieu de seuil fixe), plus tous les `hp_*`, durée, sizes de splits, `seq_test`, `val_logloss_best`, `best_iteration`.
- `logs/test_predictions.csv` (consommé par `api.py`) — schéma inchangé.

**Ajout** : un breakdown par `seq_len` calculé **sous la règle adaptative** (un seul tableau, plus deux). Une row par `seq_len` présent dans le test après filtre `SEQ_TEST`, avec : `n_taken`, `wr` sous décision adaptative. Précédé d'une ligne `TOTAL` égale à `adaptive_total_*`. Les colonnes correspondantes dans `runs.csv` deviennent `wr_seq<n>` / `n_seq<n>` (réutilisation des noms existants, mais sémantique adaptative au lieu de fixée).

**Base du seuil** : la base devient strictement `ADAPTIVE_TARGET_WR` (c'est déjà ce que `predict.py` et `api.py` consomment). `BREAKDOWN_THRESHOLD` disparaît partout — y compris en tant que base théorique pour les `delta` du sidecar : la métadonnée `metadata.target_wr` du sidecar reste pertinente, mais on ne stocke plus de `base_threshold`.

#### `data.py`

Aucun changement. Toute la SECTION 1 — DATA reste utilisée.

### Réécriture des docs

#### `README.md` (~1500 mots cible)

Sections retenues :
- Pitch (binary classifier, body_ratio filter, reversal label).
- Stack + install (`pip install requests pandas numpy xgboost scikit-learn`, retire `scipy` qui ne servait qu'à `clean_data/`).
- Quick start : `python data.py` → `python train.py` → `python predict.py`.
- Architecture : 5 fichiers (`config.py`, `data.py`, `train.py`, `predict.py`, `api.py`).
- Configuration : pointe vers `config.py` pour les knobs, mention rapide de la convention `False` pour la régularisation.
- Adaptive threshold contract : 1 paragraphe + lien vers `DOCUMENTATION.md` pour le détail.
- Conventions et pièges connus : rows non indépendantes, Bitstamp seconds vs ms, indicateurs hand-rolled / `WARMUP`, `BODY_RATIO_MIN` filtre métier.

Sections supprimées : tout ce qui concerne `clean_data/`, `sweep.py`, `analyze_runs.py`, le tableau de seuils sweep, l'API `should_take_bet(threshold=…)`.

#### `DOCUMENTATION.md` (~3000-4000 mots cible)

Plan resserré (vs. l'actuel à 24k caractères) :
1. Vue d'ensemble (problème, cas d'usage, métrique d'intérêt).
2. Source de données Bitstamp (seconds vs ms, retry policy, dédup, normalisation Binance kline).
3. Indicateurs hand-rolled + `WARMUP`.
4. Logique de labeling (séquence, body_ratio, label, non-indépendance des rows).
5. Les 42 features (22 15m + 20 MTF) avec table.
6. Entraînement : split temporel `TRAIN_DAYS`/`TEST_DAYS`/`VAL_FRACTION`, filtre `SEQ_TEST` côté test, hyperparams XGBoost, convention `False`.
7. Adaptive threshold : buckets, recette de calibration, fallback cascade, sidecar JSON, base = `ADAPTIVE_TARGET_WR`.
8. Évaluation produite par `train.py` : baseline, calibration table par bucket, breakdown par `seq_len` sous décision adaptative, top-10 feature importance.
9. `predict.py` API : lazy load, sidecar mandatory, smoke test.
10. `api.py` : contrat `NAME` + `evaluate_on_slice(start_ms, end_ms)`, format de sortie, conditions d'erreur (sidecar manquant, `test_predictions.csv` manquant).
11. `runs.csv` : schéma adaptatif, append behavior.
12. Limites connues (concept drift, signal faible, `obv_slope_10` instable, sensibilité à `BODY_RATIO_MIN`).

Sections supprimées : `clean_data/` (chapitre entier), `sweep.py` / `analyze_runs.py`, le chapitre « Évaluation : tableau de seuils » qui décrit le sweep, la section sur `ADAPTIVE_THRESHOLD = False` comme bypass.

## Comportements modifiés (breaking)

| # | Avant | Après |
|---|---|---|
| 1 | `predict.should_take_bet(features)` fallback silencieux sur `BREAKDOWN_THRESHOLD` si sidecar absent | `FileNotFoundError` |
| 2 | `predict.should_take_bet(features, threshold=0.58)` accepté | Argument `threshold` supprimé (`TypeError` si appelé) |
| 3 | `api.evaluate_on_slice` fallback silencieux | `FileNotFoundError` |
| 4 | `ADAPTIVE_THRESHOLD = False` désactive la calibration | Flag supprimé, calibration toujours active |
| 5 | `runs.csv` contient `wr_at_<th>`, `breakdown_total_*`, etc. | Schéma adaptatif uniquement ; ancien fichier supprimé |
| 6 | Console `train.py` : tableau de seuils + breakdown fixe + breakdown adaptatif | Tableau de calibration adaptatif + breakdown adaptatif uniquement |

Ces breaking changes sont voulus. Les anciens artefacts (`runs.csv`, modèles entraînés sans sidecar) ne sont pas migrés.

## Ce qui reste après la coupe

```
backtest/modele_15m/
├── README.md                            (slim, ~1.5k mots)
├── DOCUMENTATION.md                     (slim, ~3-4k mots)
├── config.py                            (DATA + TRAINING slim)
├── data.py                              (unchanged)
├── train.py                             (adaptive-only)
├── predict.py                           (adaptive-only)
├── api.py                               (adaptive-only)
├── training_data.csv                    (regen par data.py, gitignored)
├── btc_reversal_model.json              (regen par train.py, gitignored)
├── btc_reversal_model.thresholds.json   (regen par train.py, mandatory)
└── logs/
    └── test_predictions.csv             (regen par train.py, consommé par api.py)
```

## Vérifications post-cleanup

À exécuter pour valider que rien n'est cassé :

1. `cd backtest/modele_15m && python data.py` — produit `training_data.csv` (44 colonnes, ~290k rows).
2. `cd backtest/modele_15m && python train.py` — produit `btc_reversal_model.json`, `btc_reversal_model.thresholds.json`, `logs/runs.csv`, `logs/test_predictions.csv`. Console : baseline + tableau de calibration adaptatif + breakdown par seq_len adaptatif + top-10 features. Pas d'erreur, pas de tableau de seuils legacy.
3. `cd backtest/modele_15m && python predict.py` — affiche le smoke test adaptatif sur la dernière row.
4. `cd backtest/modele_15m && python api.py` — smoke test : `evaluate_on_slice(0, now_ms)` renvoie un DataFrame non vide avec les 4 colonnes attendues.
5. `grep -r "BREAKDOWN_THRESHOLD\|THRESHOLDS\|ADAPTIVE_THRESHOLD\|LOG_THRESHOLDS\|sweep\|clean_data\|analyze_runs" backtest/modele_15m/` — aucune référence (à part dans des commentaires d'historique éventuels, mais on les retire aussi).
6. `python -c "from predict import should_take_bet; should_take_bet({}, threshold=0.5)"` — `TypeError` (preuve que le paramètre legacy a disparu).

## Hors-scope explicite

- **Pas** de mise à jour de `infra/Dockerfile` ni `infra/docker-compose.yml` (ils référencent `live/` qui est déjà cassé pour d'autres raisons — sortir de ce périmètre).
- **Pas** de mise à jour du `CLAUDE.md` racine (sera fait dans un commit séparé après la cleanup, pour éviter de mélanger refactor de code et refactor de docs cross-package).
- **Pas** de modification de `data.py` ni du schéma de `training_data.csv` ni du schéma de `test_predictions.csv` (le contrat `api.py` dépend de ce dernier).
- **Pas** de migration des anciennes rows `runs.csv` — fichier supprimé, on regénère.
- **Pas** de tests unitaires ajoutés (le projet n'en a pas, par convention répétée dans `CLAUDE.md`).
