# Documentation technique — BTC Reversal Classifier

Document de présentation complet du projet. Lecture estimée : 15 min.
Couvre l'intégralité du pipeline d'entraînement (fetch → features → modèle → prédiction).

---

## 1. Vue d'ensemble

### 1.1 Problème résolu

Le projet produit un classifieur binaire qui répond à la question :

> *« Étant donné une séquence de N bougies 15m de même direction qui vient de se terminer par une bougie à fort body, la bougie suivante va-t-elle inverser la tendance ? »*

Ce n'est **pas** un modèle de prédiction de prix. C'est un **filtre de setups** : il classe des opportunités de trading déjà sélectionnées par une règle métier déterministe, en estimant la probabilité que le retournement attendu se produise réellement.

### 1.2 Cas d'usage

Le modèle est conçu pour être branché sur un moteur de backtest ou un bot de trading. À la clôture de chaque bougie 15m, le bot :

1. Vérifie si la séquence actuelle constitue un setup valide (1 à 4 bougies de même sens, dernière à `body_ratio ≥ 0.6`).
2. Si oui, calcule les 22 features de la situation actuelle.
3. Appelle `predict.predict_proba(features)` pour obtenir la probabilité de retournement.
4. Décide de prendre ou non le pari selon un seuil configurable.

### 1.3 Métrique d'intérêt

La métrique principale n'est **pas** l'accuracy globale, mais le **win rate (WR) à seuil de probabilité fixe**, comparé à la baseline du filtre body_ratio seul. Le modèle est utile s'il permet de relever significativement le WR au prix d'une réduction du volume de paris pris — c'est-à-dire qu'il est capable de distinguer les bons setups des mauvais parmi ceux qui passent déjà le filtre déterministe.

---

## 2. Architecture du système

Le projet est volontairement minimaliste : trois fichiers Python racine, un fichier de configuration, un CSV intermédiaire, un fichier de modèle.

```
xgboost_training/
├── config.py                   Hyperparamètres centralisés (DATA + TRAINING)
├── data.py                     Étape 1 : fetch Bitstamp → CSV de features
├── train.py                    Étape 2 : entraîne le modèle XGBoost
├── predict.py                  Module librairie : inférence depuis l'extérieur
├── training_data.csv           Dataset généré par data.py
└── btc_reversal_model.json     Modèle généré par train.py
```

Le pipeline est strictement linéaire et reproductible :

```
config.py
    │
    ├──> data.py  ──> training_data.csv
    │                       │
    └──> train.py  ─────────┴──> btc_reversal_model.json
                                        │
                                        └──> predict.py (chargement lazy)
```

Aucune base de données, aucun cache, aucun service tiers en dehors de l'API publique Bitstamp.

---

## 3. La source de données

### 3.1 Bitstamp public OHLC API

Endpoint : `https://www.bitstamp.net/api/v2/ohlc/{pair}/`
Pas de clé API requise. Le code paire est en lowercase (`btcusd`, `btcusdt`, `ethusd`, ...).

### 3.2 Différence d'unités temporelles

Bitstamp travaille en **secondes** pour les paramètres `start`, `end`, `step`, mais le reste du pipeline travaille en **millisecondes** (convention Binance kline). La conversion est encapsulée dans `fetch_chunk` :

- En sortie : `start_sec = start_ms // 1000`, `end_sec = end_ms // 1000`.
- En entrée : `timestamp_ms = bitstamp_timestamp * 1000`.

C'est volontaire pour ne pas avoir à toucher au reste du pipeline si on change de provider plus tard. **Aucune fonction en aval de `fetch_chunk` ne voit jamais des secondes.**

### 3.3 Pagination et parallélisation

La récupération de `DAYS_BACK` jours d'historique est découpée en fenêtres de `KLINES_LIMIT = 1000` bougies (la limite Bitstamp par requête). Les fenêtres sont récupérées en parallèle via un `ThreadPoolExecutor` à `FETCH_WORKERS = 8` threads.

Les fenêtres adjacentes Bitstamp peuvent se chevaucher d'une bougie aux frontières (les paramètres `start`/`end` sont inclusifs côté API). `fetch_all_klines` déduplique par timestamp avant de retourner la série.

### 3.4 Stratégie de retry

`fetch_chunk` retente jusqu'à `MAX_RETRIES = 3` fois avec backoff exponentiel (`2 ** attempt` secondes), **uniquement** sur `requests.exceptions.Timeout` et `requests.exceptions.ConnectionError`. Toute erreur HTTP (4xx, 5xx) lève immédiatement l'exception — par design, parce qu'un 429 ou un 500 indique un problème qui ne sera pas résolu par un retry court.

### 3.5 Normalisation au format Binance kline

Chaque bougie est normalisée en `[open_time_ms, open, high, low, close, volume]` (6 colonnes), pour que le code d'extraction de features reste agnostique de la source.

---

## 4. Les indicateurs techniques

Tous les indicateurs sont **implémentés à la main en Python pur**, sans `talib` ni `pandas-ta`. C'est intentionnel : zéro dépendance binaire fragile, comportement strictement contrôlé, facile à auditer.

### 4.1 Liste

| Indicateur | Fonction | Période défaut | Notes |
|---|---|---|---|
| OBV (On-Balance Volume) | `compute_obv` | — | Cumul signé du volume selon la direction du close |
| EMA | `compute_ema` | 20, 50, 200 (close) ; 20 (OBV) | Init = première valeur, α = 2/(period+1) |
| RSI | `compute_rsi` | 14 | Wilder smoothing après init en moyenne simple |
| ATR | `compute_atr` | 14 (court), 100 (long) | Wilder smoothing, init = moyenne simple sur les `period` premiers TR |
| Volume avg | `compute_volume_avg` | 20 | Moyenne glissante simple |

### 4.2 Warmup

`WARMUP = 200` constante. C'est le nombre de bougies ignorées au début de la série pour laisser les indicateurs (notamment EMA-200) se stabiliser. Une EMA initialisée sur la première valeur converge vers la vraie EMA en ~3-5 fois sa période — 200 bougies couvrent donc largement le besoin.

> **Si vous ajoutez un indicateur** avec un lookback supérieur à 200 sur 15m, augmentez `WARMUP` ou les premières rows seront contaminées par des valeurs d'indicateurs non stabilisées.

---

## 5. La logique de labeling

C'est le cœur conceptuel du projet. À comprendre avant d'interpréter quoi que ce soit.

### 5.1 Séquence et direction

Pour chaque bougie, sa **direction** est :
- `up` si `close > open`,
- `down` si `close < open`,
- `flat` si `close == open` (ignorée totalement).

Une **séquence de longueur `seq_len`** est `seq_len` bougies consécutives ayant toutes la même direction (toutes `up` ou toutes `down`). Le code itère sur `seq_len ∈ {1, 2, 3, 4}`, ce qui signifie que **la même bougie de départ peut générer jusqu'à 4 rows** dans le dataset (une par longueur de séquence valide).

### 5.2 Filtre body_ratio (filtre métier d'entrée)

Une séquence est admise dans le dataset uniquement si la **dernière bougie** de la séquence vérifie :

```
body_ratio = |close - open| / (high - low) ≥ BODY_RATIO_MIN  (= 0.6)
```

C'est-à-dire : le corps de la dernière bougie représente au moins 60% de son range total. C'est un filtre d'**élan** — on ne s'intéresse qu'aux fins de séquence où l'impulsion est visible.

> **Important** : ce seuil n'est pas une feature apprise. Il définit *qui entre dans le dataset*. Le modifier change la nature du problème, pas juste la performance du modèle. La WR baseline change mécaniquement avec lui.

### 5.3 Définition du label

Soit `last_idx = i + seq_len - 1` l'index de la dernière bougie de la séquence. La bougie suivante est `last_idx + 1 = i + seq_len`.

```
label = 1  si  direction(i + seq_len) == retournement_attendu
label = 0  sinon
```

Avec `retournement_attendu = "down"` si la séquence est `up`, et inversement. Une bougie suivante `flat` exclut la row du dataset.

### 5.4 Conséquence sur l'indépendance des observations

**Les rows ne sont pas indépendantes** :

1. La même bougie de départ produit potentiellement 4 rows (une par `seq_len`).
2. Deux rows consécutives partagent la majorité de leur historique d'indicateurs.

Implication directe : **toute cross-validation random est invalide**. Si vous avez besoin de CV, utilisez `TimeSeriesSplit` ou du walk-forward — pas `KFold`.

---

## 6. Les 22 features

Toutes les features sont calculées à `last_idx` (la dernière bougie de la séquence), au moment où la décision serait prise. Aucune feature ne regarde dans le futur.

### 6.1 Structure de séquence (2 features)

| Feature | Type | Définition |
|---|---|---|
| `seq_len` | int ∈ {1,2,3,4} | Nombre de bougies dans la séquence |
| `seq_dir` | int ∈ {0,1} | 1 si séquence `up`, 0 si `down` |

### 6.2 Anatomie de bougie (4 features)

Toutes en proportion du range `(high - low)`, donc dans `[0, 1]`.

| Feature | Définition |
|---|---|
| `body_ratio_last` | `body_ratio` de la dernière bougie (≥ 0.6 par construction) |
| `body_ratio_avg` | Moyenne du `body_ratio` sur les `seq_len` bougies de la séquence |
| `upper_wick` | `(high - max(open, close)) / range` sur la dernière bougie |
| `lower_wick` | `(min(open, close) - low) / range` sur la dernière bougie |

Invariant : `body_ratio_last + upper_wick + lower_wick == 1` (à l'erreur d'arrondi près).

### 6.3 Volume relatif (2 features)

| Feature | Définition |
|---|---|
| `volume_ratio_last` | `volume[last_idx] / vol_avg_20[last_idx]` |
| `volume_ratio_avg` | `mean(volume[i:i+seq_len]) / vol_avg_20[last_idx]` |

Capture l'intensité de la séquence par rapport au régime de volume récent.

### 6.4 Mouvement de la séquence (2 features)

| Feature | Définition |
|---|---|
| `move_pct` | `(closes[last_idx] - opens[i]) / opens[i] * 100` (signé) |
| `move_atr` | `\|closes[last_idx] - opens[i]\| / atr_14[last_idx]` (en multiples d'ATR) |

`move_atr` est invariant à la volatilité absolue ; `move_pct` ne l'est pas.

### 6.5 Momentum (4 features)

| Feature | Définition |
|---|---|
| `rsi` | RSI(14) à `last_idx` |
| `rsi_change_5` | `rsi[last_idx] - rsi[last_idx - 5]` |
| `obv_slope_10` | `(obv[last_idx] - obv[last_idx - 10]) / \|obv[last_idx - 10]\|` |
| `obv_vs_ema` | 1 si `obv[last_idx] > ema_obv_20[last_idx]`, 0 sinon |

`obv_slope_10` peut exploser quand `obv[last_idx - 10]` est proche de zéro (premières années du marché). C'est un point connu.

### 6.6 Volatilité (2 features)

| Feature | Définition |
|---|---|
| `atr_ratio` | `atr_14[last_idx] / atr_100[last_idx]` |
| `volatility_state` | Catégoriel : 2 si `atr_ratio > 1.5`, 0 si `< 0.7`, 1 sinon |

### 6.7 Tendance (3 features)

Distance signée du prix aux EMAs, en %.

| Feature | Définition |
|---|---|
| `dist_ema20` | `(close - ema_20) / ema_20 * 100` |
| `dist_ema50` | idem avec EMA-50 |
| `dist_ema200` | idem avec EMA-200 |

### 6.8 Contexte temporel (3 features)

| Feature | Définition |
|---|---|
| `hour_utc` | Heure UTC du `last_idx` (0..23) |
| `day_of_week` | Lundi=0, Dimanche=6 |
| `is_weekend` | 1 si `day_of_week >= 5`, sinon 0 |

`is_weekend` est techniquement redondante avec `day_of_week`, mais simplifie la prise de décision dans les arbres.

### 6.9 Format final du CSV

`training_data.csv` contient 24 colonnes : `timestamp` + les 22 features ci-dessus + `label`. Une ligne par couple `(i, seq_len)` valide. Le CSV typique pour BTC/USD avec `DAYS_BACK = 5000` fait environ **290k lignes**.

---

## 7. L'entraînement

### 7.1 Split temporel

`train.py` charge le CSV, le trie par timestamp, puis le découpe **sans shuffle** :

- **Test**  : les `TEST_DAYS` derniers jours du CSV (défaut : 365 jours).
- **Validation** : les derniers `VAL_FRACTION = 15%` du segment hors-test (donc le morceau le plus récent du segment d'entraînement).
- **Train** : tout ce qui précède.

Le calcul exact :

```python
ms_per_day = 86_400_000
test_cutoff_ts = df["timestamp"].max() - TEST_DAYS * ms_per_day
test_start = df["timestamp"].searchsorted(test_cutoff_ts, side="left")
val_size = int(test_start * VAL_FRACTION)
train_end = test_start - val_size
```

### 7.2 Pourquoi pas de shuffle ?

Les rows adjacentes dans le temps partagent le même régime de marché (même volatilité, même tendance macro, mêmes corrélations). Un shuffle introduit du **leakage temporel** : une row du futur peut se retrouver dans le train et révéler de l'information sur des rows quasi-jumelles présentes dans le test, ce qui gonfle artificiellement les métriques d'évaluation.

> **Ne passez jamais à `train_test_split(shuffle=True)` ni à `KFold` sans réfléchir.** Si vous voulez de la cross-validation, utilisez `TimeSeriesSplit` ou un walk-forward manuel.

### 7.3 Modèle XGBoost

Le classifieur est un `XGBClassifier` standard avec les hyperparamètres suivants (modifiables dans `config.py`) :

| Paramètre | Défaut | Rôle |
|---|---|---|
| `n_estimators` | 1000 | Borne supérieure ; l'early stopping coupe en pratique avant |
| `max_depth` | 4 | Profondeur des arbres ; faible pour limiter l'overfit sur ~290k rows |
| `learning_rate` | 0.05 | Pas de boosting |
| `min_child_weight` | 5 | Régularisation par poids minimum dans une feuille |
| `subsample` | 0.8 | Sous-échantillonnage stochastique des rows |
| `colsample_bytree` | 0.8 | Sous-échantillonnage stochastique des features |
| `early_stopping_rounds` | 50 | Coupe si la val ne progresse pas pendant 50 itérations |
| `random_state` | 42 | Reproductibilité |
| `objective` | `"binary:logistic"` | Sortie sous forme de probabilité |
| `eval_metric` | `"logloss"` | Métrique surveillée pour l'early stopping |

### 7.4 Convention `False` pour la régularisation

Trois hyperparamètres suivent une convention spéciale :

```python
XGB_REG_LAMBDA = False   # désactivé → pas passé à XGBoost (défaut natif = 1.0)
XGB_REG_ALPHA  = False   # désactivé → pas passé à XGBoost (défaut natif = 0.0)
XGB_GAMMA      = False   # désactivé → pas passé à XGBoost (défaut natif = 0.0)
```

Mettre `False` (et **pas** `0` ou `None`) signifie *« n'inclus pas cet argument dans l'appel à `XGBClassifier` »*. Le test côté `train.py` est `if XGB_REG_LAMBDA is not False:` — l'identité, pas la véracité — parce que `0` est une valeur valide et signifie *« force la régularisation à zéro explicitement »*. C'est un mécanisme d'opt-in propre pour activer/désactiver des hyperparamètres optionnels.

### 7.5 Early stopping sur la slice validation

L'early stopping observe la métrique `logloss` sur la slice `val` (pas `train`). La slice `test` n'est jamais touchée pendant l'entraînement — elle est intégralement réservée pour le balayage de seuils final. Cette discipline évite le data snooping sur le test set.

---

## 8. L'évaluation

`train.py` produit trois blocs de sortie destinés à un opérateur humain.

### 8.1 Baseline

```
Baseline (filtre body_ratio seul) : 53.48% WR sur 4123 paris
```

C'est `y_test.mean() * 100` : le WR qu'on aurait en prenant **tous** les setups du test set (ceux qui ont passé le filtre body_ratio). C'est la barre à battre. Battre 50% n'a aucun sens dans ce projet ; battre la baseline en a.

### 8.2 Tableau de seuils

```
Seuil   Paris pris    Win rate    % volume conservé
-------------------------------------------------------
0.50    3801          54.21%      92.18%
0.55    2517          56.87%      61.04%
0.58    1482          58.71%      35.94%
0.60    873           61.40%      21.18%
0.62    412           63.83%       9.99%
```

Chaque ligne lit : *« si je prends uniquement les paris où la probabilité prédite est ≥ seuil, j'en prends N, avec un WR de X%, ce qui représente Y% du volume total disponible. »*

L'arbitrage est trading-style : on échange du volume contre du WR. Plus le seuil est haut, moins on prend de paris, mais plus chacun est confiant.

### 8.3 Breakdown par `seq_len`

```
--- Breakdown par longueur de séquence (seuil=0.58) ---
  TOTAL paris pris par l'IA : 1482 (WR = 58.71%)
  Seq 1b :  712 paris, WR = 57.30%
  Seq 2b :  401 paris, WR = 59.85%
  Seq 3b :  238 paris, WR = 60.50%
  Seq 4b :  131 paris, WR = 61.07%
```

La ligne **TOTAL** est l'agrégat — c'est le chiffre principal à regarder. Le détail par `seq_len` permet de vérifier que la performance n'est pas concentrée sur une seule longueur de séquence (ce qui indiquerait un mode d'échec spécifique).

### 8.4 Feature importance

Le top 10 des features par importance native XGBoost (gain) est imprimé en fin de sortie. C'est un diagnostic, pas une métrique d'évaluation : il aide à repérer si une feature inattendue domine ou si la distribution est saine.

---

## 9. Le module de prédiction

`predict.py` n'est pas un script du pipeline — c'est un **module librairie** importé par les scripts de backtest et de live.

### 9.1 API publique

```python
from predict import predict_proba, should_take_bet

# Trois formes acceptées en entrée
proba = predict_proba({"seq_len": 2, "rsi": 65.4, ...})       # un seul setup
proba = predict_proba([{...}, {...}, {...}])                  # batch
proba = predict_proba(df)                                     # DataFrame

# Décision booléenne au seuil voulu
take = should_take_bet({...}, threshold=0.58)
```

### 9.2 Lazy loading et cache

Le modèle est chargé depuis le disque au **premier appel uniquement**, et caché dans un singleton de module (`_model`). Les appels suivants ne retouchent jamais le filesystem. C'est important pour les bots live qui peuvent faire des dizaines d'appels par minute.

### 9.3 Tolérance aux colonnes en trop

L'extraction des features attendues se fait via le booster lui-même :

```python
feature_cols = model.get_booster().feature_names
return model.predict_proba(df[feature_cols])[:, 1]
```

Cela signifie que vous pouvez passer un DataFrame contenant `timestamp`, `label`, ou n'importe quelle colonne supplémentaire — elles sont ignorées automatiquement, dans l'ordre correct des features attendues. Pas besoin de présélectionner les colonnes côté appelant.

### 9.4 Smoke-test direct

Lancer `python predict.py` directement effectue un test minimal sur la dernière ligne du CSV :

```
Dernière bougie : proba retournement = 0.5612  | seuil = 0.58  | take = False
```

Sert principalement à vérifier que le modèle se charge correctement après réentraînement.

---

## 10. Configuration centralisée

`config.py` regroupe **toutes** les constantes du projet, en deux sections explicitement labellisées.

### 10.1 Section 1 — DATA (consommée par `data.py`)

Les constantes qui définissent **quel dataset on construit** :

| Catégorie | Constantes |
|---|---|
| Source | `SYMBOL`, `INTERVAL`, `DAYS_BACK`, `OUTPUT_CSV` |
| API Bitstamp | `API_URL`, `KLINES_LIMIT`, `REQUEST_TIMEOUT`, `MAX_RETRIES`, `FETCH_WORKERS`, `INTERVAL_MS` |
| Filtres dataset | `WARMUP`, `MAX_SEQ_LEN`, `BODY_RATIO_MIN` |
| Indicateurs | `EMA_OBV_PERIOD`, `EMA_CLOSE_PERIODS`, `RSI_PERIOD`, `ATR_PERIOD`, `ATR_LONG_PERIOD`, `VOL_AVG_PERIOD` |
| Fenêtres features | `OBV_SLOPE_WINDOW`, `RSI_CHANGE_WINDOW` |
| Seuils volatilité | `VOLATILITY_HIGH`, `VOLATILITY_LOW` |

### 10.2 Section 2 — TRAINING (consommée par `train.py` et `predict.py`)

Les constantes qui définissent **comment on entraîne et évalue** :

| Catégorie | Constantes |
|---|---|
| I/O | `CSV_PATH`, `MODEL_OUTPUT` |
| Split temporel | `TEST_DAYS`, `VAL_FRACTION` |
| Évaluation | `THRESHOLDS`, `BREAKDOWN_THRESHOLD` |
| XGBoost | `XGB_N_ESTIMATORS`, `XGB_MAX_DEPTH`, `XGB_LEARNING_RATE`, `XGB_MIN_CHILD_WEIGHT`, `XGB_SUBSAMPLE`, `XGB_COLSAMPLE_BYTREE`, `XGB_REG_LAMBDA`, `XGB_REG_ALPHA`, `XGB_GAMMA`, `XGB_EARLY_STOPPING_ROUNDS`, `XGB_RANDOM_STATE` |

> **Règle** : aucun magic number en dur dans `data.py` ou `train.py`. Toute valeur tunable doit vivre dans `config.py`. Cette discipline est ce qui permet de comparer plusieurs runs en ne changeant qu'un fichier.

---

## 11. Limites connues et pièges

Section à lire avant toute modification du code.

### 11.1 Rows non indépendantes

Comme expliqué en §5.4, la même bougie de départ génère plusieurs rows (une par `seq_len`), et les rows consécutives partagent leur historique. Conséquence pour l'évaluation :

- ❌ `train_test_split(shuffle=True)` — leakage temporel direct.
- ❌ `KFold` random — leakage par voisinage.
- ✅ Split temporel par timestamp (déjà en place).
- ✅ `TimeSeriesSplit` ou walk-forward pour CV.

### 11.2 Pouvoir prédictif intrinsèquement faible

La corrélation Spearman maximale entre une feature individuelle et le label est de l'ordre de **0.04**. La mutual information maximale est de l'ordre de **0.04** également. Cela signifie deux choses :

1. Le problème est intrinsèquement difficile — c'est attendu, c'est de la prédiction de marché à court terme.
2. Le modèle gagne par **agrégation** de signaux faibles, pas par exploitation d'une feature dominante. Tout bruit non géré (outliers, drift) coûte donc cher en proportion.

### 11.3 Concept drift

Le marché BTC a profondément changé entre 2012 et aujourd'hui : volume, profondeur de carnet, structure des participants. Une analyse statistique du dataset montre un **drift significatif** entre la première et la seconde moitié temporelle (test KS > 0.15 sur plusieurs features, WR baseline qui passe de ~52% à ~55%). Le modèle entraîné en split temporel apprend implicitement à mélanger ces régimes.

### 11.4 `obv_slope_10` instable sur l'historique ancien

La feature `obv_slope_10` divise par `|obv[t-10]|`. Sur les premières années du dataset (2012-2014), `obv` est souvent proche de zéro, ce qui produit des valeurs extrêmes (z-score > 100). XGBoost gère mieux que la plupart des modèles, mais ces aberrations introduisent du bruit dans les splits.

### 11.5 Sensibilité au filtre `BODY_RATIO_MIN`

Modifier `BODY_RATIO_MIN` change **mécaniquement** :
- Le nombre de rows dans le dataset,
- La WR baseline,
- L'interprétation de tous les seuils du tableau d'évaluation.

Un modèle entraîné avec `BODY_RATIO_MIN = 0.4` n'est **pas comparable** à un modèle entraîné avec `BODY_RATIO_MIN = 0.6` — ils résolvent des problèmes différents.

### 11.6 Conventions Bitstamp à préserver

- Ne propagez jamais des **secondes** au-delà de `fetch_chunk`. Tout en aval est en millisecondes.
- Si vous ajoutez une nouvelle granularité (1m, 1h, ...), mettez à jour `INTERVAL_MS` dans `config.py`.
- La déduplication par timestamp dans `fetch_all_klines` est nécessaire — ne la retirez pas même si elle semble redondante.

---

## 12. Récapitulatif des sorties

Pour orienter un nouvel arrivant : voici exactement ce qui existe sur disque après un run complet.

| Fichier | Producteur | Consommateur | Contenu |
|---|---|---|---|
| `training_data.csv` | `data.py` | `train.py` | 24 colonnes, ~290k lignes (selon `DAYS_BACK`) |
| `btc_reversal_model.json` | `train.py` | `predict.py` | Modèle XGBoost sérialisé (format natif XGBoost) |
| Sortie console de `train.py` | `train.py` | Opérateur humain | Tableau de seuils + breakdown + feature importance |

---

## 13. Pour aller plus loin

Pistes naturelles d'évolution non implémentées dans ce repo :

- **Multi-timeframe** : ajouter des features dérivées des bougies 1h/4h/1d en plus du 15m, en respectant le no-lookahead par bisect sur les timestamps.
- **Walk-forward backtest** : module séparé qui rejoue le modèle décision par décision sur la slice test, avec gestion de position et calcul de PnL net de frais.
- **Tuning des hyperparamètres** : `optuna` sur la slice val avec `TimeSeriesSplit`, en tenant compte que `n_estimators` est de toute façon plafonné par l'early stopping.
- **Calibration de probabilités** : `CalibratedClassifierCV` ou Platt scaling sur la val, si on veut que les probabilités soient utilisables comme telles (pour de la gestion de taille de position par exemple).

Aucune de ces pistes ne nécessite de changer l'architecture actuelle — elles s'ajoutent en périphérie.
