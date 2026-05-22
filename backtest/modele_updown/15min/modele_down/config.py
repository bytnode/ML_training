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

# ============================================================
#  SECTION 2b : Strategie de calibration adaptative (V1 / V2)
# ============================================================
# Decision DOWN : take ssi P(down) >= seuil_effectif(row).
#
# CALIBRATION PAR QUANTILE :
#   T_axis = val_pool_axis_quantile(1 - ASYM_VOL_FRAC_axis)
#   = on garde le top ASYM_VOL_FRAC_axis de la val non-skip de l'axe.
#
# Ce design remplace l'ancienne logique "smallest T s.t. WR_val(p>=T) >=
# target_wr" qui etait instable cross-window. Le quantile assure une
# STABILITE du volume par construction.
#
# "asym_wknd_excl"            : V1.
#                               1 seuil par axe (weekday/weekend) calibre par
#                               quantile sur val non-skip. Cellules dans
#                               ASYM_SKIP_CELLS = jamais de pari.
#                               Sidecar : {strategy, thresholds_per_wknd, skip_cells}.
# "asym_wknd_excl_hour_boost" : V2 = V1 + boost horaire.
#                               Pour HH(UTC) in ASYM_HOUR_BOOST_GOOD_HOURS,
#                               le seuil de la bougie est abaisse de
#                               ASYM_HOUR_BOOST_DELTA (skip cells non boostees).
#                               Sidecar : V1 + {hour_boost: {good_hours, delta}}.
ADAPTIVE_STRATEGY = "asym_wknd_excl_hour_boost"

# Parametres asym (V1 + V2 partagent le meme bloc) :
# Fraction de val non-skip a garder par axe. Retunes par audit walk-forward
# 6 fenetres sur le modele DOWN + features ETH (cf optimize/opt_calib.py).
ASYM_VOL_FRAC_WEEKDAY = 0.0225     # top 2.25% du val weekday non-skip
ASYM_VOL_FRAC_WEEKEND = 0.1        # top 10% du val weekend non-skip
ASYM_SKIP_CELLS = ()                # aucune cellule exclue : avec les features
                                    # ETH, inclure vol2 ameliore la WR honnete
                                    # (walk-forward +0.34pt) -- skip vol2 retire
ASYM_USE_FULL_VAL = True            # True = full val, False = derniers 55j

# Parametres specifiques V2 ("asym_wknd_excl_hour_boost") :
# Pour les bougies ouvertes a une heure UTC dans ASYM_HOUR_BOOST_GOOD_HOURS,
# le seuil de la bougie est abaisse de ASYM_HOUR_BOOST_DELTA. Les cellules
# skippees (ASYM_SKIP_CELLS) ne sont jamais boostees.
# Note : les heures 20-22 sont l'effet identifie pour UP (US close).
# Pour DOWN, l'effet microstructure pourrait differer - a auditer.
ASYM_HOUR_BOOST_GOOD_HOURS = (20, 21, 22)
ASYM_HOUR_BOOST_DELTA = 0.020

# --- Hyperparamètres XGBoost (identiques à modele_15m) ---
XGB_N_ESTIMATORS = 1000
XGB_MAX_DEPTH = 5
XGB_LEARNING_RATE = 0.05
XGB_MIN_CHILD_WEIGHT = 50   # 20 -> 50 : les 14 features ETH ajoutent de la
                            # capacite ; +regularisation = +0.6pt WR honnete
                            # (audit walk-forward, cf optimize/opt_model.py)
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
XGB_SCALE_POS_WEIGHT = "auto"
