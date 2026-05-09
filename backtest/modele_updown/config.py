# ============================================================
#  SECTION 1 : DATA / FEATURES (utilisé par data.py + features.py)
# ============================================================

# --- Source de données ---
SYMBOL = "btcusd"
INTERVAL = "15m"
DAYS_BACK = 5000
DB_PATH = "db/btc_multi_tf_15m.parquet"  # CSV 15m lu par features.py + train.py

# --- Higher timeframes ajoutés à chaque ligne du dataset (no-lookahead via
# last_closed_index : OHLCV de la dernière bougie closed avant le timestamp
# de décision T = times[i] + 15m).
MTF_INTERVALS = ["30m", "1h", "4h", "1d"]

INTERVAL_MS = {
    "15m": 15 * 60 * 1000,
    "30m": 30 * 60 * 1000,
    "1h":  60 * 60 * 1000,
    "4h":  4 * 60 * 60 * 1000,
    "1d":  24 * 60 * 60 * 1000,
}

# --- Extraction des features ---
WARMUP = 500            # bars 15m ignorées en début (warmup ATR long)

# --- Volatility state (seul calcul dérivé en plus des colonnes DB) ---
# ratio = atr_15m / rolling_mean(atr_15m, ATR_LONG_PERIOD)
# bucket : 0 = low (< VOLATILITY_LOW), 1 = mid, 2 = high (> VOLATILITY_HIGH)
ATR_LONG_PERIOD = 100
VOLATILITY_HIGH = 1.5
VOLATILITY_LOW = 0.7


# ============================================================
#  SECTION 2 : TRAINING (utilisé par train.py)
# ============================================================

# --- Fichiers ---
MODEL_OUTPUT = "btc_updown_model.json"

TRAIN_DAYS = 4000
TEST_DAYS = 365
VAL_FRACTION = 0.1

# --- Seuil adaptatif (deux-côtés : un delta UP + un delta DOWN par bucket) ---
# Le seuil utilisé pour décider :
#   bet UP   si proba >= ADAPTIVE_TARGET_WR + delta_up(bucket)
#   bet DOWN si proba <= 1 - (ADAPTIVE_TARGET_WR + delta_down(bucket))
#   sinon skip
# bucket = (volatility_state, is_weekend) — 6 buckets au total.

ADAPTIVE_TARGET_WR = 0.5675
ADAPTIVE_CALIB_FRACTION = 0.5
ADAPTIVE_MIN_N_VAL = 200
ADAPTIVE_MIN_N_VAL_SOFT = 30
ADAPTIVE_MIN_N_AT_THRESHOLD = 30
ADAPTIVE_DELTA_CLAMP = (-0.0425,0.08)

# --- Hyperparamètres XGBoost (identiques à modele_15m) ---
XGB_N_ESTIMATORS = 1000
XGB_MAX_DEPTH = 5
XGB_LEARNING_RATE = 0.05
XGB_MIN_CHILD_WEIGHT = 50
XGB_SUBSAMPLE = 0.8
XGB_COLSAMPLE_BYTREE = 0.8
XGB_REG_LAMBDA = False
XGB_REG_ALPHA = False
XGB_GAMMA = 1.5
XGB_EARLY_STOPPING_ROUNDS = 50
XGB_RANDOM_STATE = 42

# --- Rééquilibrage de classe (DOWN vs UP) ---
# Le label=1 (DOWN) est légèrement minoritaire (~49.1 % du dataset).
# scale_pos_weight = poids appliqué à la classe positive pendant l'entraînement.
#   False  -> pas passé à XGBoost (= 1.0, comportement par défaut, asymétrie subie)
#   "auto" -> calculé à l'entraînement = n_up / n_down (équilibrage exact)
#   float  -> valeur fixe (ex 1.05 pour un upweight DOWN manuel)
XGB_SCALE_POS_WEIGHT = False
