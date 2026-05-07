# BTC Reversal Classifier

Modèle XGBoost binaire qui prédit, à la clôture d'une séquence de N bougies de même direction, la probabilité que la **bougie suivante inverse** la tendance.

Conçu comme un filtre de setups de trading : il ne prédit pas le prix, il classe des opportunités déjà filtrées par une règle métier (`body_ratio ≥ 0.6` sur la dernière bougie de la séquence).

---

## Stack

- Python 3.10+ (testé sur 3.14)
- `requests`, `pandas`, `numpy`, `xgboost`, `scikit-learn`, `scipy`
- Données : Bitstamp public OHLC API (aucune clé requise)

Pas de `requirements.txt` à ce stade — installer manuellement :

```bash
pip install requests pandas numpy xgboost scikit-learn scipy
```

---

## Quick start

```bash
# 1. Récupérer 5000 jours de bougies 15m BTC/USD et générer le CSV d'entraînement
python data.py

# 2. (Optionnel mais recommandé) Nettoyer le dataset
python clean_data/clean.py

# 3. Entraîner le modèle
python train.py

# 4. Smoke-test du modèle sur la dernière ligne du CSV
python predict.py
```

---

## Ce que prédit le modèle

Pour chaque position `i` du flux de bougies 15m et chaque longueur `seq_len ∈ {1, 2, 3, 4}` :

- **Conditions d'entrée du dataset** :
  - Les `seq_len` bougies à partir de `i` ont toutes la même direction (toutes haussières ou toutes baissières),
  - La dernière bougie de la séquence a un `body_ratio ≥ 0.6`,
  - Les bougies `flat` (open == close) sont ignorées.

- **Label** : `1` si la bougie `i + seq_len` inverse la direction de la séquence, `0` sinon.

Le modèle apprend donc à distinguer, parmi les setups qui passent déjà le filtre body_ratio, ceux qui ont une probabilité élevée de retournement.

---

## Architecture

Trois fichiers Python racine + un dossier de data cleaning.

```
xgboost_training/
├── config.py                  # tous les hyperparamètres (DATA + TRAINING)
├── data.py                    # fetch Bitstamp + indicateurs + extraction features → training_data.csv
├── train.py                   # split temporel + XGBoost + évaluation par seuil → btc_reversal_model.json
├── predict.py                 # module librairie : predict_proba(features), should_take_bet(features)
├── training_data.csv          # dataset brut (généré par data.py)
├── training_data_clean.csv    # dataset nettoyé (généré par clean_data/clean.py)
├── btc_reversal_model.json    # modèle entraîné (généré par train.py)
├── CLAUDE.md                  # spec interne pour assistance IA
└── clean_data/
    ├── analyze.py             # étape 1+2 : compréhension + détection des points faibles
    ├── clean.py               # étape 3 : corrections (winsorization, weights, features)
    ├── analysis_report.md     # rapport lisible des points faibles détectés
    ├── analysis_report.json   # mêmes findings en JSON
    └── cleaning_log.md        # journal des corrections appliquées
```

---

## Le pipeline en détail

### `data.py` — récupération et features

- Source : `https://www.bitstamp.net/api/v2/ohlc/{pair}/`, sans clé API.
- Convention : Bitstamp travaille en **secondes** (`start`, `end`, `step`), le reste du pipeline en **millisecondes**. La conversion est faite dans `fetch_chunk`.
- Récupère `DAYS_BACK` jours d'historique (défaut : 5000), paginé par fenêtres de `KLINES_LIMIT=1000` bougies, parallélisé via `FETCH_WORKERS=8`.
- Indicateurs **hand-rolled** (pas de pandas-ta / talib) : OBV, EMA(20/50/200), RSI(14), ATR(14), ATR(100), volume avg(20).
- Extrait 22 features par row (cf. table ci-dessous), écrit `training_data.csv`.

**Features écrites dans le CSV** (24 colonnes au total : `timestamp` + 22 features + `label`) :

| Catégorie | Features |
|---|---|
| Structure de séquence | `seq_len`, `seq_dir` |
| Anatomie bougie | `body_ratio_last`, `body_ratio_avg`, `upper_wick`, `lower_wick` |
| Volume | `volume_ratio_last`, `volume_ratio_avg` |
| Mouvement | `move_pct`, `move_atr` |
| Momentum | `rsi`, `rsi_change_5`, `obv_slope_10`, `obv_vs_ema` |
| Volatilité | `atr_ratio`, `volatility_state` |
| Tendance EMA | `dist_ema20`, `dist_ema50`, `dist_ema200` |
| Contexte temporel | `hour_utc`, `day_of_week`, `is_weekend` |

### `train.py` — entraînement et évaluation

Split **temporel strict, sans shuffle** :

- `test`  = les `TEST_DAYS` derniers jours du CSV (défaut : 365),
- `val`   = `VAL_FRACTION = 15%` du segment hors-test (le plus récent du segment train),
- `train` = tout ce qui précède.

`early_stopping_rounds = XGB_EARLY_STOPPING_ROUNDS` (défaut : 50) utilise la slice `val` ; la slice `test` n'est utilisée qu'à la fin pour le balayage de seuils.

**Évaluation produite** :

- WR baseline (= taux de positifs si on prend tous les setups qui ont passé le filtre body_ratio).
- Tableau WR pour chaque seuil de probabilité dans `THRESHOLDS`, avec nombre de paris pris et % de volume conservé.
- Breakdown par `seq_len` au seuil `BREAKDOWN_THRESHOLD` (défaut : 0.58), précédé d'une ligne **TOTAL** (nombre de paris + WR agrégé).
- Top 10 feature importance.

> **Pourquoi pas de shuffle ?** Les rows adjacentes partagent le même régime de marché. Mélanger ferait fuiter de l'information du futur dans le train. Si vous voulez de la cross-validation, utilisez `TimeSeriesSplit` ou du walk-forward.

### `predict.py` — module librairie

Pas un script de pipeline : **importé** par les scripts de backtest / live.

```python
from predict import predict_proba, should_take_bet

# accepte dict, list[dict] ou DataFrame
proba = predict_proba({...})           # array[float]
take  = should_take_bet({...}, threshold=0.58)  # array[bool]
```

- Le modèle est chargé en lazy-init et caché dans une variable de module (`_model`).
- Les colonnes attendues sont lues directement depuis le booster (`model.get_booster().feature_names`), donc les colonnes en trop (`timestamp`, `label`) sont ignorées automatiquement.
- Lancer `python predict.py` directement effectue un smoke-test sur la dernière ligne du CSV.

---

## Configuration

Tous les hyperparamètres vivent dans `config.py`, en deux sections :

**Section 1 — DATA** (consommée par `data.py`)

| Paramètre | Description | Défaut |
|---|---|---|
| `SYMBOL` | Code Bitstamp lowercase | `"btcusd"` |
| `INTERVAL` | Granularité | `"15m"` |
| `DAYS_BACK` | Historique en jours | `5000` |
| `BODY_RATIO_MIN` | Filtre d'entrée du dataset | `0.6` |
| `MAX_SEQ_LEN` | Longueur max de séquence testée | `4` |
| `WARMUP` | Bougies ignorées en début (stabilisation indicateurs) | `200` |
| `EMA_CLOSE_PERIODS` | Périodes EMA | `(20, 50, 200)` |
| `RSI_PERIOD`, `ATR_PERIOD`, `VOL_AVG_PERIOD` | Périodes indicateurs | 14, 14, 20 |
| `VOLATILITY_HIGH`, `VOLATILITY_LOW` | Seuils état volatilité | 1.5, 0.7 |

**Section 2 — TRAINING** (consommée par `train.py` et `predict.py`)

| Paramètre | Description | Défaut |
|---|---|---|
| `CSV_PATH` | Source d'entraînement | `"training_data.csv"` |
| `MODEL_OUTPUT` | Destination du modèle | `"btc_reversal_model.json"` |
| `TEST_DAYS` | Fenêtre de test | `365` |
| `VAL_FRACTION` | Fraction validation hors-test | `0.15` |
| `THRESHOLDS` | Seuils probabilité évalués | `[0.50 .. 0.62]` |
| `BREAKDOWN_THRESHOLD` | Seuil pour le breakdown détaillé | `0.58` |
| `XGB_*` | Hyperparamètres XGBoost | cf. `config.py` |

> **Convention `False` pour la régularisation** : `XGB_REG_LAMBDA`, `XGB_REG_ALPHA`, `XGB_GAMMA` peuvent valoir `False` (au lieu de `0` ou `None`) pour signifier "ne pas passer ce paramètre à XGBoost", et donc utiliser le défaut natif. Mettre une valeur numérique active la régularisation correspondante. Le test est `is not False`, ce qui est intentionnel : `0` est une valeur valide signifiant "désactive explicitement cette régularisation".

---

## Module de data cleaning (`clean_data/`)

Pipeline en trois étapes pour améliorer la qualité du dataset avant entraînement.

### Étape 1 + 2 : `analyze.py`

Produit un audit complet du CSV brut sans rien modifier :

- Vue d'ensemble (taille, plage temporelle, types).
- Détection NaN / Inf.
- Doublons exacts, doublons sur features, **conflits** (mêmes features → labels différents).
- Distribution du label : globale, par `seq_len`, par `seq_dir`.
- Stats descriptives par feature (mean, std, percentiles, skewness, kurtosis).
- **Outliers** : % hors IQR (k=1.5 et k=3), % hors [p0.1, p99.9], max |z|.
- Features quasi-constantes.
- **Multi-colinéarité** : paires avec |r| ≥ 0.85.
- Cohérence métier : `body_ratio_last ≥ 0.6`, anatomie bougie, ranges des catégorielles.
- **Concept drift** : test KS (1ère moitié vs 2ème moitié temporelle).
- Mutual information avec le label.
- Densité temporelle (gaps, lignes par année).

Sortie : `analysis_report.md` (lisible) + `analysis_report.json` (machine-readable).

```bash
python clean_data/analyze.py
```

### Étape 3 : `clean.py`

Applique des corrections **non destructives** (aucune ligne supprimée) à `training_data.csv` et écrit `training_data_clean.csv` :

| Correction | Détail | Effet vérifié |
|---|---|---|
| **Winsorization** sur 10 features | Clip aux quantiles p0.5 / p99.5 | max \|z\| réduit de ~200 à ~7 sur `obv_slope_10`, `volume_ratio_*` |
| **`era_year`** ajouté | Année comme feature | Permet à XGBoost d'apprendre la non-stationarité du marché |
| **`sample_weight`** exponentiel | `exp(-α × age_years)`, α=0.15, normalisé à mean=1 | Row 2026 pèse ~2.24, row 2012 pèse ~0.29 |
| **7 features dérivées** | `volume_ratio_*_log`, `wick_imbalance`, `dist_ema_aligned`, `rsi_extreme`, `move_signed_atr`, `dist_ema20_vs_ema200` | Capture du signal non linéaire / non capturé par XGBoost |

```bash
python clean_data/clean.py
```

Sortie : `training_data_clean.csv` (racine, 33 colonnes) + `clean_data/cleaning_log.md` (journal détaillé des corrections).

### Utilisation du dataset nettoyé dans `train.py`

```python
# Pointer config.CSV_PATH vers training_data_clean.csv

df = pd.read_csv(CSV_PATH)
# Exclure timestamp, label, ET sample_weight des features
feature_cols = [c for c in df.columns
                if c not in ("timestamp", "label", "sample_weight")]

# era_year RESTE dans X — c'est une feature volontaire
model.fit(
    X_train, y_train,
    sample_weight=train["sample_weight"],
    eval_set=[(X_val, y_val)],
    sample_weight_eval_set=[val["sample_weight"]],
)
```

---

## Conventions et pièges connus

- **Rows non indépendantes** : la même position de départ `i` génère jusqu'à 4 rows (une par `seq_len`). Important si vous ajoutez de la cross-validation : utilisez `GroupKFold` sur `timestamp` ou `TimeSeriesSplit`, pas `KFold` random.
- **Bitstamp seconds vs internal ms** : ne propagez jamais les secondes au-delà de `fetch_chunk`. Si vous ajoutez une nouvelle granularité, mettez à jour `INTERVAL_MS` dans `config.py`.
- **Indicateurs hand-rolled** : `WARMUP = 200` est calibré pour stabiliser EMA-200 sur 15m. Si vous ajoutez un indicateur avec un lookback plus long, augmentez `WARMUP` ou les premières rows seront polluées.
- **Filtre body_ratio = règle métier, pas feature** : il définit qui *entre* dans le dataset. Le modifier change la nature du problème, pas seulement la performance — la "WR baseline" change avec lui.
- **WR baseline ≠ 50%** : le baseline pertinent est le `y_test.mean()` du dataset filtré. Battre 50% n'a aucun sens ; battre le baseline en est un.

---

## Format des sorties d'évaluation

Exemple de sortie de `train.py` :

```
Seuil   Paris pris    Win rate    % volume conservé
-------------------------------------------------------
0.5     12453         53.21%      85.34%
0.55    7821          55.87%      53.61%
0.58    4012          58.42%      27.50%
0.60    1893          61.04%      12.97%

--- Breakdown par longueur de séquence (seuil=0.58) ---
  TOTAL paris pris par l'IA : 4012 (WR = 58.42%)
  Seq 1b :  1845 paris, WR = 57.10%
  Seq 2b :  1102 paris, WR = 59.21%
  Seq 3b :   612 paris, WR = 60.34%
  Seq 4b :   453 paris, WR = 60.92%
```

Lecture : à seuil `0.58`, le modèle prend **27.5%** des setups disponibles avec un WR de **58.42%** (vs ~53.5% baseline) — gain ~5pp en échange d'une réduction du volume de paris.

---

## Licence

Privé — non publié.
