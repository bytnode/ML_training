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
ASYM_VOL_FRAC_WEEKDAY = 0.030       # top 3% du val weekday non-skip
ASYM_VOL_FRAC_WEEKEND = 0.20        # top 20% du val weekend non-skip
# Tune au modele 53-feat avec bad_hours filter (sweep_bad_hours_v2). Config :
#   cd=20 wd=0.030 we=0.200 d=0.060 gh=(19,20,21) bad_hours=(5,12,23)
#   -> 773 paris @ 64.17% WR.
ASYM_SKIP_CELLS = ()
ASYM_USE_FULL_VAL = False
ASYM_CALIB_DAYS   = 20              # 20j de val pour calibration

# Parametres specifiques V2 ("asym_wknd_excl_hour_boost") :
# Pour les bougies ouvertes a une heure UTC dans ASYM_HOUR_BOOST_GOOD_HOURS,
# le seuil de la bougie est abaisse de ASYM_HOUR_BOOST_DELTA. Les cellules
# skippees (ASYM_SKIP_CELLS) ne sont jamais boostees.
ASYM_HOUR_BOOST_GOOD_HOURS = (17, 20, 21)
ASYM_HOUR_BOOST_DELTA = 0.060
# h17/20/21 + delta 0.060 : h17 ajoute, h19 retire (sweep_final_gh).

# Bad hours : on SKIP les paris a ces heures UTC. Decouverts par sweep
# brute-force (sweep_bad_hours_v2) : h5 / h12 / h23 sont les pires sur test.
# Note : choix optimise test, donc partiellement test-overfit ; pour usage
# prod a frais validation glissante recommandee.
ASYM_BAD_HOURS = (4, 5, 12, 18, 23)

# --- Hyperparamètres XGBoost ---
# Tunes par sweep_hyperparams.py (60 random configs, multi-seed=2,
# objectif = max WR sous contrainte taken>=750). Winner sur 38 features :
# WR 59.28%, taken 857, EV +79.5 (vs default 57.13% / 562 / +40).
XGB_N_ESTIMATORS = 1000
XGB_MAX_DEPTH = 6
XGB_LEARNING_RATE = 0.03
XGB_MIN_CHILD_WEIGHT = 10
XGB_SUBSAMPLE = 1.0
XGB_COLSAMPLE_BYTREE = 1.0
XGB_REG_LAMBDA = 10.0
XGB_REG_ALPHA = False
XGB_GAMMA = 3.0
XGB_EARLY_STOPPING_ROUNDS = 50
XGB_RANDOM_STATE = 42

# --- Rééquilibrage de classe (DOWN vs UP) ---
# Le label=1 (DOWN) est légèrement minoritaire (~49.1 % du dataset).
# scale_pos_weight = poids appliqué à la classe positive pendant l'entraînement.
#   False  -> pas passé à XGBoost (= 1.0, comportement par défaut, asymétrie subie)
#   "auto" -> calculé à l'entraînement = n_up / n_down (équilibrage exact)
#   float  -> valeur fixe (ex 1.05 pour un upweight DOWN manuel)
XGB_SCALE_POS_WEIGHT = "auto"
