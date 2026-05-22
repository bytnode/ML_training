# ============================================================
#  SECTION 1 : DATA / FEATURES (utilisé par data.py + features.py)
# ============================================================

# --- Source de données ---
SYMBOL = "btcusd"
INTERVAL = "1h"
DAYS_BACK = 5000
DB_PATH = "db/btc_multi_tf_1h.parquet"

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
# Fraction de val non-skip a garder par axe (= top X% du val pool).
# Point de depart heritee du modele_up qui atteint 64.54% / 753 bets. A retuner
# par sweep specifique DOWN (microstructure & directionalite differente).
ASYM_VOL_FRAC_WEEKDAY = 0.100       # top 10% du val weekday non-skip
ASYM_VOL_FRAC_WEEKEND = 0.20        # top 20% du val weekend non-skip
ASYM_SKIP_CELLS = ()                # pas de skip cell (perd trop de volume)
ASYM_USE_FULL_VAL = False           # False = derniers ASYM_CALIB_DAYS de val
ASYM_CALIB_DAYS   = 180             # fenetre de calibration recente (jours)

# Parametres specifiques V2 ("asym_wknd_excl_hour_boost") :
# Pour les bougies ouvertes a une heure UTC dans ASYM_HOUR_BOOST_GOOD_HOURS,
# le seuil de la bougie est abaisse de ASYM_HOUR_BOOST_DELTA. Les cellules
# skippees (ASYM_SKIP_CELLS) ne sont jamais boostees.
# Note : les heures UP (US close 19-21) pourraient etre les pires pour DOWN.
# Point de depart copie du UP, a re-sweeper specifiquement pour DOWN.
ASYM_HOUR_BOOST_GOOD_HOURS = (2, 9, 20, 22)
ASYM_HOUR_BOOST_DELTA = 0.030

# Bad hours : heures UTC ou on SKIP les paris (heures historiquement mauvaises
# pour DOWN). Découvertes par sweep_extreme : h1, h8, h10, h13, h14, h17, h21.
ASYM_BAD_HOURS = (1, 8, 13, 14, 17, 21)

# Filtres d'inférence : on SKIP les paris quand une feature dépasse un cutoff.
# Format : {feature_name: (cutoff, "<" | ">")}
# "<" = on garde si feature < cutoff ; ">" = on garde si feature >= cutoff.
# Découverts par analyze_mistakes.py + sweep_aggressive : volume_ratio_30m et
# return_3 sont les meilleurs discriminateurs win/loss parmi les top P(down).
# BASE_FILTERS sont appliqués A LA FOIS pendant la calibration du seuil ET
# pendant l'inference. Ces filtres definissent le sous-espace de marche
# tradable du modele (low/mid volume, momentum non extreme).
BASE_INFERENCE_FILTERS = {
    "volume_ratio_30m": (1.5, "<"),
    "return_3":         (0.030, "<"),
}

# EXTRA_FILTERS sont appliqués UNIQUEMENT a l'inference (pas pendant la
# calibration). Ils servent a affiner la selection des paris parmi ceux qui
# passent BASE_FILTERS + le seuil. Decouverts par greedy filter sweep.
EXTRA_INFERENCE_FILTERS = {
    "lower_wick":       (0.003795, ">"),    # skip si bougie sans lower wick
    "mfi_norm":         (-0.523159, ">"),   # skip oversold extreme
    "body_ratio_avg":   (0.287662, ">"),    # skip si bougies recentes ont peu de body
    "close_ret":        (-0.003506, ">"),   # skip si bougie courante deja tres rouge
}

# Aggrege des deux dicts pour le code legacy.
INFERENCE_FILTERS = {**BASE_INFERENCE_FILTERS, **EXTRA_INFERENCE_FILTERS}

# Calibration filter-aware : si True, applique BASE_INFERENCE_FILTERS (PAS
# les EXTRA) au pool de val AVANT le calcul du seuil quantile.
CALIB_USE_FILTERS = True

# --- Hyperparamètres XGBoost ---
# Point de depart : config du modele_up apres sweep (depth=6, lr=0.03, mcw=10,
# subsample/colsample=1.0, lambda=10, gamma=3). A re-sweeper pour DOWN.
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
