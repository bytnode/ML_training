# ============================================================
#  SECTION 1 : DATA / FEATURES (utilisé par data.py + features.py)
# ============================================================

# --- Source de données ---
SYMBOL = "btcusd"
INTERVAL = "1h"
DAYS_BACK = 5000
DB_PATH = "DB/btc_multi_tf_1h.parquet"

# --- Higher timeframes ajoutés à chaque ligne du dataset (no-lookahead via
# last_closed_index : OHLCV de la dernière bougie closed avant le timestamp
# de décision T = times[i] + 1h).
MTF_INTERVALS = ["4h", "1d"]

INTERVAL_MS = {
    "30m": 30 * 60 * 1000,
    "1h":  60 * 60 * 1000,
    "4h":  4 * 60 * 60 * 1000,
    "1d":  24 * 60 * 60 * 1000,
}

# --- Extraction des features ---
WARMUP = 500            # bars 1h ignorées en début (warmup ATR long)

# --- Volatility state (seul calcul dérivé en plus des colonnes DB) ---
# ratio = atr_1h / rolling_mean(atr_1h, ATR_LONG_PERIOD)
# bucket : 0 = low (< VOLATILITY_LOW), 1 = mid, 2 = high (> VOLATILITY_HIGH)
ATR_LONG_PERIOD = 100
VOLATILITY_HIGH = 1.5
VOLATILITY_LOW = 0.7


# ============================================================
#  SECTION 2 : TRAINING (utilisé par train.py)
# ============================================================

# --- Fichiers ---
MODEL_OUTPUT = "btc_updown_model.json"

TRAIN_DAYS = 3600
TEST_DAYS = 366
VAL_FRACTION = 0.1

# ============================================================
#  SECTION 2b : Strategie de calibration adaptative (V1 / V2)
# ============================================================
# Decision UP : take ssi P(up) >= seuil_effectif(row).
#
# CALIBRATION PAR QUANTILE (depuis 2026-05-11) :
#   T_axis = val_pool_axis_quantile(1 - ASYM_VOL_FRAC_axis)
#   = on garde le top ASYM_VOL_FRAC_axis de la val non-skip de l'axe.
#
# Ce design remplace le "smallest T s.t. WR_val(p>=T) >= target_wr" qui etait
# instable cross-window (le seuil weekend s'effondrait de 0.555 a 0.520 selon
# la periode de val, faisant exploser le volume de 6% a 17%).
# Le quantile assure une STABILITE du volume par construction, immune au
# baseline-drift de WR_val.
#
# "asym_wknd_excl"            : V1.
#                               1 seuil par axe (weekday / weekend) calibre
#                               par quantile sur val. Cellules dans
#                               ASYM_SKIP_CELLS = jamais de pari.
#                               Sidecar : {strategy, thresholds_per_wknd, skip_cells}.
# "asym_wknd_excl_hour_boost" : V2 = V1 + boost horaire.
#                               Pour HH(UTC) in ASYM_HOUR_BOOST_GOOD_HOURS,
#                               le seuil de la bougie est abaisse de
#                               ASYM_HOUR_BOOST_DELTA (skip cells non boostees).
#                               Sidecar : V1 + {hour_boost: {good_hours, delta}}.
ADAPTIVE_STRATEGY = "asym_wknd_excl_hour_boost"

# Parametres asym (V1 + V2 partagent le meme bloc) :
# Fraction de val non-skip a garder par axe. Determinee a partir du sweet-spot
# trouve sur TEST_DAYS=365 (V1=1761 paris @ 59.23%):
#   - weekday : T_wd=0.594 = val_weekday_pool quantile_98.7 (275/21706 = 1.27%)
#   - weekend : T_we=0.555 = val_weekend_pool quantile_83.3 (1499/8960 = 16.72%)
ASYM_VOL_FRAC_WEEKDAY = 0.018       # top 1.3% du val weekday non-skip
ASYM_VOL_FRAC_WEEKEND = 0.18       # top 16.7% du val weekend non-skip
ASYM_SKIP_CELLS = ((2, 0), (2, 1))  # (volatility_state, is_weekend) exclus
ASYM_USE_FULL_VAL = True            # True = full val, False = derniers 55j

# Parametres specifiques V2 ("asym_wknd_excl_hour_boost") :
# Pour les bougies ouvertes a une heure UTC dans ASYM_HOUR_BOOST_GOOD_HOURS,
# le seuil de la bougie est abaisse de ASYM_HOUR_BOOST_DELTA. Les cellules
# skippees (ASYM_SKIP_CELLS) ne sont jamais boostees.
ASYM_HOUR_BOOST_GOOD_HOURS = (20, 21, 22)
ASYM_HOUR_BOOST_DELTA = 0.020

# --- Hyperparamètres XGBoost ---
XGB_N_ESTIMATORS = 1000
XGB_MAX_DEPTH = 5
XGB_LEARNING_RATE = 0.05
XGB_MIN_CHILD_WEIGHT = 20
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
XGB_SCALE_POS_WEIGHT = 1.03
