# BTC Reversal Classifier

Modèle XGBoost binaire qui prédit, à la clôture d'une séquence de N bougies de même direction, la probabilité que la **bougie suivante inverse** la tendance.

Filtre de setups, pas prédiction de prix : il classe des opportunités déjà filtrées par une règle métier (`body_ratio ≥ 0.6` sur la dernière bougie de la séquence). La décision take/skip se fait sous une règle de **seuil adaptatif par bucket** `(volatility_state, is_weekend)`, calibrée sur la slice de validation et persistée dans un sidecar JSON à côté du modèle.

---

## Stack

- Python 3.10+ (testé sur 3.14)
- `requests`, `pandas`, `numpy`, `xgboost`, `scikit-learn`
- Données : Bitstamp public OHLC API (aucune clé requise)

Pas de `requirements.txt` à ce stade — installer manuellement :

```bash
pip install requests pandas numpy xgboost scikit-learn
```

---

## Quick start

Tous les scripts utilisent des chemins relatifs (`config.CSV_PATH`, `config.MODEL_OUTPUT`) — exécuter avec `backtest/modele_15m/` comme cwd :

```powershell
cd backtest\modele_15m

# 1. Récupérer 5000 jours de bougies 15m BTC/USD + MTF (30m/1h/4h/1d) → CSV de features
python data.py

# 2. Entraîner le modèle + calibrer les seuils adaptatifs
python train.py

# 3. Smoke-test du modèle sur la dernière ligne du CSV
python predict.py
```

`python train.py` produit :
- `btc_reversal_model.json` — le modèle XGBoost.
- `btc_reversal_model.thresholds.json` — sidecar des deltas par bucket. **Obligatoire** pour `predict.py` et `api.py` ; sa régénération fait partie du run.
- `logs/test_predictions.csv` — test set + colonne `confidence`. Consommé par `api.py`.
- `logs/runs.csv` — journal des runs (un row par appel de `train_and_evaluate(log_run=True)`).

---

## Ce que prédit le modèle

Pour chaque position `i` du flux de bougies 15m et chaque longueur `seq_len ∈ {1, 2, 3, 4}` :

- **Conditions d'entrée du dataset** :
  - Les `seq_len` bougies à partir de `i` ont toutes la même direction (toutes haussières ou toutes baissières),
  - La dernière bougie de la séquence a un `body_ratio ≥ 0.6`,
  - Les bougies `flat` (open == close) sont ignorées.

- **Label** : `1` si la bougie `i + seq_len` inverse la direction de la séquence, `0` sinon.

Le modèle apprend à distinguer, parmi les setups qui passent déjà le filtre body_ratio, ceux qui ont une probabilité élevée de retournement.

---

## Architecture

```
backtest/modele_15m/
├── config.py                            # hyperparams (DATA + TRAINING)
├── data.py                              # fetch Bitstamp + indicateurs + features → training_data.csv
├── train.py                             # split temporel + XGBoost + calibration adaptative → modele + sidecar
├── predict.py                           # module librairie : predict_proba(...), should_take_bet(...)
├── api.py                               # contrat NAME + evaluate_on_slice(start_ms, end_ms) pour le team backtest
├── training_data.csv                    # généré par data.py (44 cols, ~290k rows)
├── btc_reversal_model.json              # généré par train.py
├── btc_reversal_model.thresholds.json   # généré par train.py (deltas par bucket + metadata)
└── logs/
    ├── runs.csv                         # un row par run (hyperparams + metrics adaptatives + deltas)
    └── test_predictions.csv             # test set + confidence (consommé par api.py)
```

---

## Décision adaptative

Source unique de vérité pour la décision take/skip : `confidence ≥ ADAPTIVE_TARGET_WR + delta(bucket)`.

- **Buckets** : `(volatility_state ∈ {0,1,2}, is_weekend ∈ {0,1})` = 6 buckets fixes. Les deux features existent déjà dans le CSV.
- **Base** : `config.ADAPTIVE_TARGET_WR` (= 0.58), source unique de vérité — pas de `BREAKDOWN_THRESHOLD` ni de seuil fixe en parallèle.
- **Calibration** (effectuée par `train.py`) : pour chaque bucket, on sweep des seuils candidats sur les `ADAPTIVE_CALIB_FRACTION × TEST_DAYS` derniers jours de val ; on garde le plus petit seuil qui atteint `ADAPTIVE_TARGET_WR` (avec ≥ `ADAPTIVE_MIN_N_AT_THRESHOLD` paris). `delta = chosen_th − base`, clampé à `ADAPTIVE_DELTA_CLAMP`. Cascade de fallback documentée dans `DOCUMENTATION.md`.
- **Sidecar JSON** (`btc_reversal_model.thresholds.json`) : `{"deltas": {"<vol>_<wknd>": float, ...}, "metadata": {...}}`. Les clés sont `"vol_wknd"` (ex : `"2_0"`).
- **`predict.should_take_bet(features)`** charge le sidecar lazy, applique `proba ≥ base + delta(bucket)` row par row. Si le sidecar est absent → `FileNotFoundError`. Mêmes garanties pour `api.evaluate_on_slice`.

Pour les détails de la cascade de fallback (`calib_window` / `fallback_full_val` / `low_conf_full_val` / `insufficient_data`) et le format du sidecar, voir `DOCUMENTATION.md`.

---

## Configuration

Tous les hyperparamètres vivent dans `config.py`, en deux sections :

**Section 1 — DATA** (consommée par `data.py`) : `SYMBOL`, `INTERVAL`, `DAYS_BACK`, paramètres API Bitstamp, `INTERVAL_MS`, `MTF_INTERVALS`, périodes des indicateurs, `BODY_RATIO_MIN`, `WARMUP`, `MAX_SEQ_LEN`, seuils de volatilité.

**Section 2 — TRAINING** (consommée par `train.py` et `predict.py`) : `CSV_PATH`, `MODEL_OUTPUT`, `TRAIN_DAYS`, `TEST_DAYS`, `VAL_FRACTION`, `SEQ_TEST` (filtre du test set par seq_len), la famille `ADAPTIVE_*` (target WR + paramètres de calibration), tous les `XGB_*`.

> **Convention `False` pour la régularisation** : `XGB_REG_LAMBDA`, `XGB_REG_ALPHA`, `XGB_GAMMA` peuvent valoir `False` (au lieu de `0` ou `None`) pour signifier "ne pas passer ce paramètre à XGBoost", et donc utiliser le défaut natif. Mettre une valeur numérique active la régularisation correspondante. Le test est `is not False`, c'est intentionnel : `0` est une valeur valide signifiant "désactive explicitement cette régularisation".

---

## Conventions et pièges connus

- **Rows non indépendantes** : la même position de départ `i` génère jusqu'à 4 rows (une par `seq_len`). Pas de `train_test_split(shuffle=True)` ni de `KFold` random — utilisez `TimeSeriesSplit` ou du walk-forward si vous ajoutez de la cross-validation.
- **Bitstamp seconds vs ms internes** : la conversion est confinée à `data.fetch_chunk` (paramètres `start`/`end`/`step` en secondes côté API, tout downstream en millisecondes). Ne propagez jamais les secondes au-delà.
- **MTF features no-lookahead** : `last_closed_index(open_times_sorted, T, INTERVAL_MS[iv])` retourne le dernier bar provably closed à `T = times[last_idx] + 15m`. Ne le bypassez pas — utiliser `closes[i]` directement sur un MTF leakerait le futur dans la row.
- **Indicateurs hand-rolled** : OBV, EMA, RSI, ATR, vol avg en Python pur. `WARMUP = 200` est calibré pour stabiliser EMA-200 sur 15m. Si vous ajoutez un indicateur avec un lookback plus long, augmentez `WARMUP` ou les premières rows seront polluées.
- **Filtre `BODY_RATIO_MIN` = règle métier, pas feature** : il définit qui *entre* dans le dataset. Le modifier change la nature du problème, pas seulement la performance — la baseline WR change avec lui.
- **WR baseline ≠ 50%** : le baseline pertinent est le `y_test.mean()` du dataset filtré. Battre 50% n'a aucun sens ; battre la baseline en a.

---

## Format des sorties d'évaluation

Exemple de sortie de `train.py` :

```
======================================================================
  EVALUATION SUR LE TEST SET (out-of-sample, decision adaptative)
======================================================================

Baseline (filtre body_ratio seul) : 53.48% WR sur 4123 paris

--- Breakdown par longueur de sequence (decision adaptative) ---
  TOTAL paris pris par l'IA : 1517 (36.79% volume), WR = 59.92%
  Seq 2b :   612 paris, WR = 58.66%
  Seq 3b :   568 paris, WR = 60.74%
  Seq 4b :   337 paris, WR = 60.83%

--- Feature importance (top 10) ---
...

======================================================================
  CALIBRATION SEUIL ADAPTATIF (sur 273j de val = 75% de TEST_DAYS=365, target_wr=0.58)
======================================================================
  (vol, wknd)   n_calib   n_used    th_pick   wr@th     delta     source
  --------------------------------------------------------------------------------
  (0, 0)        ...       ...       0.565     0.5832    -0.015    calib_window:target_hit
  (0, 1)        ...       ...       0.530     0.5810    -0.050    calib_window:target_hit
  ...
```

Lecture : la décision adaptative prend **36.79%** des setups disponibles avec un WR de **59.92%** (vs ~53.5% baseline) — gain ~6pp en échange d'une réduction de volume.

---

## Licence

Privé — non publié.
