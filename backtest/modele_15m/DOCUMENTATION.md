# Documentation technique — BTC Reversal Classifier

Document de présentation complet du projet. Couvre l'intégralité du pipeline : fetch → features → modèle → calibration adaptative → prédiction → API.

Pour une vue rapide, voir `README.md`. Ce document est la référence pour les invariants, les contrats inter-modules et les pièges connus.

---

## 1. Vue d'ensemble

### 1.1 Problème résolu

Classifieur binaire qui répond à la question :

> *« Étant donné une séquence de N bougies 15m de même direction qui vient de se terminer par une bougie à fort body, la bougie suivante va-t-elle inverser la tendance ? »*

Ce n'est **pas** un modèle de prédiction de prix. C'est un **filtre de setups** : il classe des opportunités de trading déjà sélectionnées par une règle métier déterministe, en estimant la probabilité que le retournement attendu se produise.

### 1.2 Cas d'usage

À la clôture de chaque bougie 15m, un bot ou un moteur de backtest :

1. Vérifie si la séquence actuelle constitue un setup valide (1 à 4 bougies de même sens, dernière à `body_ratio ≥ 0.6`).
2. Si oui, calcule les features.
3. Appelle `predict.predict_proba(features)` pour obtenir la probabilité.
4. Décide de prendre ou non le pari via `predict.should_take_bet(features)` — décision sous **seuil adaptatif par bucket**, lue dans le sidecar JSON.

### 1.3 Métrique d'intérêt

La métrique principale n'est **pas** l'accuracy globale, mais le **win rate (WR) sous décision adaptative**, comparé à la baseline du filtre body_ratio seul. Le modèle est utile s'il permet de relever significativement le WR au prix d'une réduction du volume de paris pris.

---

## 2. Architecture

```
backtest/modele_15m/
├── config.py                            Hyperparams (DATA + TRAINING)
├── data.py                              Étape 1 : fetch Bitstamp → CSV de features
├── train.py                             Étape 2 : entraîne le modèle + calibre les deltas
├── predict.py                           Module librairie : inférence depuis l'extérieur
├── api.py                               Contrat avec backtest/main.py (NAME, evaluate_on_slice)
├── training_data.csv                    Dataset généré par data.py
├── btc_reversal_model.json              Modèle généré par train.py
├── btc_reversal_model.thresholds.json   Sidecar des deltas adaptatifs (généré par train.py)
└── logs/
    ├── runs.csv                         Journal des runs
    └── test_predictions.csv             Test set + confidence (consommé par api.py)
```

Pipeline strictement linéaire :

```
config.py
    │
    ├──> data.py ──> training_data.csv
    │                       │
    └──> train.py ──────────┴──> btc_reversal_model.json
                                  │
                                  ├──> btc_reversal_model.thresholds.json (sidecar)
                                  ├──> logs/test_predictions.csv ──> api.py
                                  │
                                  └──> predict.py (chargement lazy)
```

---

## 3. La source de données

### 3.1 Bitstamp public OHLC API

Endpoint : `https://www.bitstamp.net/api/v2/ohlc/{pair}/`. Pas de clé requise. Code paire en lowercase (`btcusd`, `btcusdt`, `ethusd`, …).

### 3.2 Différence d'unités temporelles

Bitstamp travaille en **secondes** pour `start`, `end`, `step`. Le reste du pipeline en **millisecondes** (convention Binance kline). La conversion est confinée à `fetch_chunk` :

- En sortie : `start_sec = start_ms // 1000`, `end_sec = end_ms // 1000`.
- En entrée : `timestamp_ms = bitstamp_timestamp * 1000`.

**Aucune fonction en aval de `fetch_chunk` ne voit jamais des secondes.** Si vous ajoutez une nouvelle granularité, mettez à jour `INTERVAL_MS` dans `config.py`.

### 3.3 Pagination, parallélisation, dedup

- `fetch_all_klines` paginé par fenêtres de `KLINES_LIMIT = 1000` bougies, parallélisé via `ThreadPoolExecutor` (`FETCH_WORKERS = 8`).
- Dedup par `open_time` : nécessaire car les fenêtres Bitstamp adjacentes peuvent se chevaucher d'une bougie aux frontières (`start`/`end` inclusifs côté API).

### 3.4 Retry policy

Backoff exponentiel (`2 ** attempt`s), max `MAX_RETRIES = 3`, **uniquement** sur `Timeout` / `ConnectionError`. Les erreurs HTTP (4xx/5xx) sont raise immédiatement — un 429 ou un 500 indique un problème qui ne sera pas résolu par un retry court.

### 3.5 MTF (multi-timeframe)

`fetch_all_timeframes` récupère 30m / 1h / 4h / 1d en plus du 15m principal. Chaque TF est fetché séquentiellement (les fenêtres internes restent parallélisées par `fetch_all_klines`).

---

## 4. Indicateurs hand-rolled

Tous implémentés en Python pur, sans `talib` ni `pandas-ta` : zéro dépendance binaire fragile, comportement contrôlé.

| Indicateur | Période défaut | Notes |
|---|---|---|
| OBV | — | Cumul signé du volume selon la direction du close |
| EMA close | 20, 50, 200 | Init = première valeur, α = 2/(period+1) |
| EMA OBV | 20 | Même formule |
| RSI | 14 | Wilder smoothing après init en moyenne simple |
| ATR | 14 (court), 100 (long) | Wilder smoothing, init = SMA sur les `period` premiers TR |
| Volume avg | 20 | SMA glissante |

**Warmup** : `WARMUP = 200`. Une EMA initialisée sur la première valeur converge vers la vraie EMA en ~3-5 fois sa période — 200 bougies couvrent largement l'EMA-200. Si vous ajoutez un indicateur avec un lookback supérieur à 200, augmentez `WARMUP` ou les premières rows seront contaminées.

---

## 5. Logique de labeling

### 5.1 Direction et séquence

Direction d'une bougie :
- `up` si `close > open`,
- `down` si `close < open`,
- `flat` si `close == open` (ignorée).

**Séquence de longueur `seq_len`** : `seq_len` bougies consécutives ayant toutes la même direction. On itère sur `seq_len ∈ {1, 2, 3, 4}` — la même bougie de départ peut donc générer jusqu'à 4 rows.

### 5.2 Filtre body_ratio (filtre métier)

```
body_ratio_last = |close - open| / (high - low) ≥ BODY_RATIO_MIN (= 0.6)
```

C'est un filtre d'**élan**. **Ce seuil n'est pas une feature apprise** — il définit qui *entre* dans le dataset. Le modifier change la nature du problème, pas juste la performance ; la WR baseline change mécaniquement avec lui.

### 5.3 Définition du label

Soit `last_idx = i + seq_len - 1`. La bougie suivante est `last_idx + 1`.

```
label = 1  si  direction(last_idx + 1) == retournement_attendu
label = 0  sinon
```

Retournement attendu = `down` si la séquence est `up`, et inversement. Une bougie suivante `flat` exclut la row du dataset.

### 5.4 Conséquence : rows non indépendantes

1. La même position de départ produit potentiellement 4 rows.
2. Deux rows consécutives partagent la majorité de leur historique d'indicateurs.

Implication directe : **toute cross-validation random est invalide**. `TimeSeriesSplit` ou walk-forward, jamais `KFold` ni `train_test_split(shuffle=True)`.

---

## 6. Les 42 features

22 features 15m + 20 features MTF (5 OHLCV × 4 TFs). Toutes calculées à `last_idx`, au moment où la décision serait prise — aucune ne regarde dans le futur.

### 6.1 Features 15m (22)

| Catégorie | Features |
|---|---|
| Structure de séquence | `seq_len`, `seq_dir` |
| Anatomie bougie | `body_ratio_last`, `body_ratio_avg`, `upper_wick`, `lower_wick` |
| Volume relatif | `volume_ratio_last`, `volume_ratio_avg` |
| Mouvement séquence | `move_pct`, `move_atr` |
| Momentum | `rsi`, `rsi_change_5`, `obv_slope_10`, `obv_vs_ema` |
| Volatilité | `atr_ratio`, `volatility_state` (catégoriel ∈ {0,1,2}) |
| Tendance EMA | `dist_ema20`, `dist_ema50`, `dist_ema200` |
| Contexte temporel | `hour_utc`, `day_of_week`, `is_weekend` |

Conventions :
- `seq_dir` : 1 = up, 0 = down.
- `volatility_state` : 2 si `atr_ratio > VOLATILITY_HIGH`, 0 si `< VOLATILITY_LOW`, sinon 1.
- Wicks et body en proportion du range `(high - low)` → invariant : `body_ratio_last + upper_wick + lower_wick == 1`.
- `move_atr` invariant à la volatilité absolue, `move_pct` non.
- `obv_slope_10` peut exploser quand `obv[last_idx - 10]` est proche de zéro (premières années du marché). Connu, on vit avec.

### 6.2 Features MTF (20)

Pour chaque interval `iv ∈ {30m, 1h, 4h, 1d}`, 5 colonnes : `open_<iv>`, `high_<iv>`, `low_<iv>`, `close_<iv>`, `vol_<iv>`. Ce sont les valeurs OHLCV brutes de la dernière bougie MTF closed à `T`.

### 6.3 No-lookahead MTF

Le timestamp de décision est `T = times[last_idx] + 15m` (la fin de la séquence vient de fermer). Pour chaque MTF, `last_closed_index(open_times_sorted, T, INTERVAL_MS[iv])` retourne l'index de la dernière bar telle que `open_time + interval_ms ≤ T` — i.e. la dernière bougie MTF provably closed à T. Si aucune bougie MTF n'est encore closed (début d'historique), la row est skippée (`skipped_mtf` counter).

**Ne bypassez jamais `last_closed_index`** : utiliser `closes[i]` directement sur un MTF leakerait des informations futures dans la row.

### 6.4 Format CSV final

`training_data.csv` : 44 colonnes = `timestamp` + 22 (15m) + 20 (MTF) + `label`. Une ligne par couple `(i, seq_len)` valide. Pour BTC/USD avec `DAYS_BACK = 5000`, ~290k lignes.

---

## 7. Entraînement

### 7.1 Split temporel

`train.py` charge le CSV, le trie par timestamp, puis le découpe **sans shuffle** :

- **Test** : les `TEST_DAYS` derniers jours du CSV.
- **Train+Val** : les `TRAIN_DAYS` jours qui précèdent le cutoff de test (`TRAIN_DAYS = None` → tout l'historique).
- **Val** : les derniers `VAL_FRACTION` du bloc train+val.
- **Train** : ce qui reste du bloc train+val.

Calcul exact :

```python
ms_per_day = 86_400_000
test_cutoff_ts = df["timestamp"].max() - TEST_DAYS * ms_per_day
test_start = df["timestamp"].searchsorted(test_cutoff_ts, side="left")
train_val_cutoff_ts = test_cutoff_ts - TRAIN_DAYS * ms_per_day
train_val_start = df["timestamp"].searchsorted(train_val_cutoff_ts, side="left")
val_size = int((test_start - train_val_start) * VAL_FRACTION)
train_end = test_start - val_size
```

### 7.2 Pourquoi pas de shuffle ?

Les rows adjacentes dans le temps partagent le même régime de marché. Un shuffle introduit du **leakage temporel** : une row du futur peut se retrouver dans le train et révéler de l'information sur des rows quasi-jumelles présentes dans le test, ce qui gonfle artificiellement les métriques.

### 7.3 `SEQ_TEST` — filtrage du test set

`SEQ_TEST = [2, 3, 4]` (défaut) restreint le test set aux setups de longueur 2, 3 ou 4. **Train et val ne sont jamais filtrés** — le modèle voit toutes les seq_len pendant le fit. Le filtre s'applique juste après le split, avant la calibration adaptative et le breakdown.

`SEQ_TEST = None` ou `[]` désactive le filtre.

### 7.4 Modèle XGBoost

`XGBClassifier` standard avec :

| Paramètre | Défaut | Rôle |
|---|---|---|
| `n_estimators` | 1000 | Borne sup ; early stopping coupe avant en pratique |
| `max_depth` | 5 | Profondeur des arbres |
| `learning_rate` | 0.05 | Pas de boosting |
| `min_child_weight` | 5 | Régul par poids minimum dans une feuille |
| `subsample` | 0.8 | Sous-échantillonnage stochastique des rows |
| `colsample_bytree` | 0.8 | Sous-échantillonnage stochastique des features |
| `early_stopping_rounds` | 50 | Surveille la val ; coupe si pas de progrès |
| `objective` | `binary:logistic` | Probabilité en sortie |
| `eval_metric` | `logloss` | Métrique surveillée par early stopping |

### 7.5 Convention `False` pour la régularisation

`XGB_REG_LAMBDA`, `XGB_REG_ALPHA`, `XGB_GAMMA` peuvent valoir **`False`** (et **pas** `0` ou `None`) pour signifier *« ne pas passer cet argument à XGBClassifier »* — XGBoost utilise alors son défaut natif (1.0, 0.0, 0.0). Mettre une valeur numérique active la régularisation. Le test côté `train.py` est `if XGB_REG_LAMBDA is not False:` — l'identité, pas la véracité — parce que `0` est une valeur valide signifiant *« force la régularisation à zéro explicitement »*.

### 7.6 Early stopping

Early stopping observe la `logloss` sur la slice **val**. La slice **test** n'est jamais touchée pendant l'entraînement — elle est utilisée pour l'évaluation finale (baseline + breakdown adaptatif). Cette discipline évite le data snooping.

---

## 8. Seuil adaptatif par bucket

C'est l'**unique règle de décision** de ce package. Pas de seuil fixe en parallèle, pas de fallback.

### 8.1 Buckets

`(volatility_state ∈ {0, 1, 2}, is_weekend ∈ {0, 1})` = 6 buckets fixes. Les deux features existent déjà dans le CSV.

### 8.2 Recette de calibration

Effectuée par `train.py` après `model.fit`, sur la slice **val** uniquement :

1. Slicer val aux **`ADAPTIVE_CALIB_FRACTION × TEST_DAYS` derniers jours** (exemple : `0.75 × 365 ≈ 273` jours). La fenêtre de calib suit l'horizon d'évaluation — si `TEST_DAYS` rétrécit, la calib aussi.
2. Pour chaque bucket, sweep des seuils candidats sur `[0.45, 0.75]` step 0.005, en gardant uniquement les candidats avec ≥ `ADAPTIVE_MIN_N_AT_THRESHOLD` paris (défaut 30) — anti-overfit du delta.
3. Choisir le **plus petit** seuil dont la WR réalisée ≥ `ADAPTIVE_TARGET_WR`. Si aucun candidat n'atteint la cible, fallback : seuil avec la WR maximale.
4. `delta = chosen_th - ADAPTIVE_TARGET_WR`, clampé à `ADAPTIVE_DELTA_CLAMP` (défaut `(-0.05, 0.08)`) — rail de sécurité.
5. Cascade de fallback par bucket (les buckets sparses comme high-vol-weekend sont fréquents en BTC) :

| Condition | Source | Confiance |
|---|---|---|
| `n_calib ≥ ADAPTIVE_MIN_N_VAL` | `calib_window` | high |
| `n_full ≥ ADAPTIVE_MIN_N_VAL` | `fallback_full_val` | high (récence sacrifiée) |
| `n_full ≥ ADAPTIVE_MIN_N_VAL_SOFT` | `low_conf_full_val` | low |
| sinon | `insufficient_data` | `delta = 0` |

Le clamp + le floor `MIN_N_AT_THRESHOLD` + le fallback max-WR à l'intérieur de `_compute_bucket_threshold` sont ce qui empêche un delta bruyant d'être dangereux dans le tier low-confidence. Ne les desserrez pas sans réflexion.

### 8.3 Sidecar JSON

`btc_reversal_model.thresholds.json` :

```json
{
  "deltas": {"0_0": -0.025, "0_1": 0.0, "1_0": 0.0, "1_1": 0.015, "2_0": 0.04, "2_1": 0.06},
  "metadata": {
    "trained_at": "2026-05-08T...",
    "calib_fraction": 0.75,
    "calib_days_resolved": 273,
    "test_days": 365,
    "target_wr": 0.58,
    "min_n_val": 200,
    "min_n_at_threshold": 30,
    "clamp": [-0.05, 0.08]
  }
}
```

Clés `deltas` : `"<vol>_<wknd>"`. **Pas de `base_threshold`** dans le sidecar — la base vient toujours de `config.ADAPTIVE_TARGET_WR` côté lecteur (`predict.py`, `api.py`). C'est volontaire : une seule source de vérité, et la base reste tunable côté config sans réécrire les sidecars.

Le sidecar est écrit uniquement quand `train.py` tourne avec `save_model=True`.

### 8.4 Contrat `predict.py`

```python
proba = predict.predict_proba(features)        # array[float] ∈ [0, 1]
take  = predict.should_take_bet(features)      # array[bool], decision adaptative
```

- `predict_proba` accepte `dict`, `list[dict]`, `pd.DataFrame`. Les colonnes en trop (`timestamp`, `label`, …) sont ignorées (les colonnes attendues sont lues depuis `model.get_booster().feature_names`).
- `should_take_bet` lit le sidecar lazy au premier appel. **Si le sidecar est absent : `FileNotFoundError`**. Pas de fallback silencieux.
- Si `volatility_state` ou `is_weekend` ne sont pas dans `features`, `should_take_bet` applique uniquement la base (`delta = 0` partout).
- Modèle et sidecar sont cachés en singletons module-level (`_model`, `_thresholds`) — chargés une fois, réutilisés.
- Smoke-test : `python predict.py` affiche la décision adaptative sur la dernière row du CSV.

---

## 9. Évaluation produite par `train.py`

Trois blocs d'output destinés à un opérateur humain.

### 9.1 Baseline

```
Baseline (filtre body_ratio seul) : 53.48% WR sur 4123 paris
```

= `y_test.mean() * 100`. C'est la barre à battre. Battre 50% n'a aucun sens dans ce projet ; battre la baseline en a.

### 9.2 Breakdown par seq_len sous décision adaptative

```
--- Breakdown par longueur de sequence (decision adaptative) ---
  TOTAL paris pris par l'IA : 1517 (36.79% volume), WR = 59.92%
  Seq 2b :   612 paris, WR = 58.66%
  Seq 3b :   568 paris, WR = 60.74%
  Seq 4b :   337 paris, WR = 60.83%
```

Ligne **TOTAL** = chiffre principal. Le détail par seq_len permet de vérifier que la performance n'est pas concentrée sur une seule longueur. Les seq_len absents de `SEQ_TEST` sont silencieusement omis (pas de ligne à 0).

### 9.3 Calibration table

```
======================================================================
  CALIBRATION SEUIL ADAPTATIF (sur 273j de val = 75% de TEST_DAYS=365, target_wr=0.58)
======================================================================
  (vol, wknd)   n_calib   n_used    th_pick   wr@th     delta     source
  --------------------------------------------------------------------------------
  (0, 0)        ...       ...       0.565     0.5832    -0.015    calib_window:target_hit
  (0, 1)        ...       ...       0.530     0.5810    -0.050    calib_window:target_hit
  ...
```

Watch `source_flag` : si la majorité des buckets read `insufficient_data`, c'est que `VAL_FRACTION` est trop petit ou `ADAPTIVE_MIN_N_VAL` trop strict.

### 9.4 Top-10 feature importance

Diagnostic, pas métrique d'évaluation. Aide à repérer si une feature inattendue domine.

---

## 10. `api.py` — contrat avec le team backtest

```python
NAME : str = "modele_15m"

def evaluate_on_slice(start_ms: int, end_ms: int) -> pd.DataFrame:
    """Renvoie 1 row par setup ou le modele aurait pris le pari dans la fenetre.
    Colonnes : timestamp (int ms), direction ('up'|'down'), confidence (float), label (int 0|1).
    DataFrame vide si aucun pari pris.
    Raises FileNotFoundError si test_predictions.csv ou le sidecar est absent.
    """
```

Implémentation : lit `logs/test_predictions.csv` (régénéré à chaque `python train.py`), filtre par fenêtre temporelle, applique les deltas du sidecar bucket par bucket, et dérive `direction` depuis `seq_dir` (`seq_dir=1` → bet `down` ; `seq_dir=0` → bet `up`).

Smoke-test : `python api.py` évalue sur tout `test_predictions.csv` et affiche `setups_taken` + WR global.

C'est le contrat d'inférence standardisé que `backtest/main.py` (team orchestrator) consomme. Les futurs packages modèle (ex : `modele_onchain/`) implémentent `evaluate_on_slice` librement (runtime fetch + predict, etc.) — `main.py` ne voit que le DataFrame standardisé.

---

## 11. `runs.csv` — schéma

Append-only, écrit par `train.py` et `train_and_evaluate(log_run=True)`. Schéma flat by design. `append_to_runs_csv` fait une union de colonnes en cas d'évolution de schéma — le fichier reste petit (< 1000 runs cible).

Colonnes principales :

| Catégorie | Colonnes |
|---|---|
| Meta | `timestamp_iso`, `run_label`, `duration_sec`, `csv_path` |
| Split | `train_days`, `test_days`, `val_fraction`, `seq_test`, `n_features`, `n_train`, `n_val`, `n_test` |
| Train | `best_iteration`, `val_logloss_best`, tous les `hp_*` |
| Décision adaptative | `baseline_wr`, `adaptive_total_taken`, `adaptive_total_wr`, `wr_seq{1..4}`, `n_seq{1..4}` |
| Calibration | `adaptive_calib_fraction`, `adaptive_calib_days`, `adaptive_target_wr`, `delta_v{0,1,2}_w{0,1}` |

`wr_seq<n>` / `n_seq<n>` sont calculés sous la décision adaptative — pas sous un seuil fixe. `delta_v<vol>_w<wknd>` peut être `None` si le bucket est tombé en `insufficient_data` à la calibration.

---

## 12. Limites connues et pièges

### 12.1 Rows non indépendantes (rappel)

- ❌ `train_test_split(shuffle=True)` — leakage temporel direct.
- ❌ `KFold` random — leakage par voisinage.
- ✅ Split temporel par timestamp (déjà en place).
- ✅ `TimeSeriesSplit` ou walk-forward pour CV.

### 12.2 Pouvoir prédictif intrinsèquement faible

Corrélation Spearman max d'une feature individuelle avec le label ≈ **0.04**. Mutual information max ≈ **0.04**. Conséquence : le modèle gagne par **agrégation** de signaux faibles, pas par exploitation d'une feature dominante. Tout bruit non géré (outliers, drift) coûte cher en proportion.

### 12.3 Concept drift

Le marché BTC a profondément changé entre 2012 et aujourd'hui : volume, profondeur de carnet, structure des participants. Test KS > 0.15 sur plusieurs features entre la 1ère et la 2ème moitié temporelle, WR baseline qui passe de ~52% à ~55%. Le modèle entraîné en split temporel apprend implicitement à mélanger ces régimes ; `TRAIN_DAYS` permet de couper l'historique ancien si nécessaire.

### 12.4 `obv_slope_10` instable sur l'historique ancien

La feature divise par `|obv[t-10]|`. Sur 2012-2014, `obv` est souvent proche de zéro, ce qui produit des valeurs extrêmes (z-score > 100). XGBoost gère mieux que la plupart des modèles, mais ces aberrations introduisent du bruit dans les splits.

### 12.5 Sensibilité au filtre `BODY_RATIO_MIN`

Modifier `BODY_RATIO_MIN` change **mécaniquement** : le nombre de rows, la WR baseline, l'interprétation des deltas adaptatifs (la calibration cible une WR donnée — si la baseline change, les deltas changent aussi). Un modèle entraîné avec `BODY_RATIO_MIN = 0.4` n'est **pas comparable** à un modèle entraîné avec `0.6` — ils résolvent des problèmes différents.

### 12.6 Conventions Bitstamp à préserver

- Ne propagez jamais des **secondes** au-delà de `fetch_chunk`.
- Si vous ajoutez une nouvelle granularité, mettez à jour `INTERVAL_MS` dans `config.py`.
- La déduplication par timestamp dans `fetch_all_klines` est nécessaire — ne la retirez pas même si elle semble redondante.

### 12.7 Sidecar obligatoire

`predict.py` et `api.py` lèvent `FileNotFoundError` si le sidecar est absent. C'est volontaire : la décision adaptative est l'unique règle de décision, pas de fallback silencieux possible. Si le sidecar manque (modèle ancien, suppression accidentelle), regénérez-le par `python train.py`.
