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
ATR_LONG_PERIOD = 500
VOLATILITY_HIGH = 1.5
VOLATILITY_LOW = 0.7


# ============================================================
#  SECTION 2 : TRAINING (utilisé par train.py)
# ============================================================

# --- Fichiers ---
MODEL_OUTPUT = "btc_updown_model.json"

TRAIN_DAYS = 3600
TEST_DAYS = 365
VAL_FRACTION = 0.3

# Bridge (embargo) : nombre de JOURS de pause entre la fin de (train+val) et le
# debut du test. Les bougies de cet intervalle ne sont ni en val ni en test.
# 0 = pas de pause (val colle au test = comportement historique). Sert a couper
# la fuite des features a lookback (ex: return_96, volume_rel_30d) dont la
# fenetre chevaucherait le debut du test.
BRIDGE_DAYS = 0

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
ASYM_VOL_FRAC_WEEKDAY = 0.0225    # ignored: weekday is fully skipped below
                                   # (kept for the runs.csv schema only).
ASYM_VOL_FRAC_WEEKEND = 0.07     # top 16.7% du val weekend non-skip
ASYM_SKIP_CELLS = ((2, 0), (2, 1))  # (volatility_state, is_weekend) exclus
ASYM_USE_FULL_VAL = True            # True = full val, False = derniers 55j

# --- Seuil UNANIME weekday/weekend, HARDCODE (toggle d'experimentation) ---
# True  -> UN SEUL seuil pour weekday ET weekend = la valeur que tu ecris dans
#          ASYM_UNANIMOUS_THRESHOLD, prise TELLE QUELLE (PAS un quantile,
#          PAS un %). Desactive completement le mode par axe : la boucle
#          weekday/weekend (ASYM_VOL_FRAC_WEEKDAY / ASYM_VOL_FRAC_WEEKEND) n'est
#          jamais executee. Le sidecar ecrit {"0": T, "1": T}, donc api.py /
#          predict.py / live.py l'appliquent sans aucun changement de code.
# False -> comportement par defaut (deux seuils distincts par axe, calibres
#          par quantile sur ASYM_VOL_FRAC_WEEKDAY / ASYM_VOL_FRAC_WEEKEND).
# Les skip_cells, bad_hours, bad_days et le hour_boost restent actifs PAR
# DESSUS : seul le split weekday/weekend du seuil de base est neutralise.
ASYM_UNANIMOUS_WKND = True
# Seuil de confiance P(up) ecrit en dur. Ex: 0.58 -> on prend le pari UP ssi
# P(up) >= 0.58, weekday comme weekend. C'est exactement le nombre du sidecar.
ASYM_UNANIMOUS_THRESHOLD = 0.575

# Parametres specifiques V2 ("asym_wknd_excl_hour_boost") :
# Pour les bougies ouvertes a une heure UTC dans ASYM_HOUR_BOOST_GOOD_HOURS,
# le seuil de la bougie est abaisse de ASYM_HOUR_BOOST_DELTA. Les cellules
# skippees (ASYM_SKIP_CELLS) ne sont jamais boostees.
ASYM_HOUR_BOOST_GOOD_HOURS = (00 ,10, 12, 21)
ASYM_HOUR_BOOST_DELTA = 0.015

# --- Bad-hours exclusion (V2 extension) ---
# UTC hours where we NEVER bet, regardless of confidence.
# Empirical losers on weekday+weekend mix (cf optimize/deep_analysis.py) :
#   hour 2  (35 % WR weekday, 68 % weekend)
#   hour 4  (22 % WR weekday, 83 % weekend - small)  -> dropped, weekend rescues it
#   hour 8  (20 % WR weekday, 58 % weekend)
#   hour 11 (46 % WR weekday, 56 % weekend)
#   hour 23 (56 % WR weekday, 52 % weekend)
# Hour 23 was even in the OLD hour_boost (= MORE bets there) which was
# clearly hurting. Removed from boost above + added to bad_hours.
ASYM_BAD_HOURS = (2,3,4,8,14,15,16,17,19,23,6,9)

# --- Bad-days threshold boost ---
# UTC isoweekday(s) where the bucket threshold is RAISED by ADAPTIVE_WR_BAD_DAY
# (= fewer, more selective bets on these days). isoweekday: 1=Monday..7=Sunday.
# Empirically Friday (=5) shows the weakest team WR in the per-weekday
# breakdown (cf main.py "Share of taken bets per weekday" table). Boosting the
# threshold there forces the model to pick only its highest-confidence Friday
# bets. Read at runtime by api.py._bucket_threshold (no retrain needed).
# Empty tuple disables the filter.
ADAPTIVE_BAD_DAYS = (4,5)
ADAPTIVE_WR_BAD_DAY = 0.02

# --- Bad-hours soft boost (mirror of ASYM_HOUR_BOOST_GOOD_HOURS) ---
# UTC hours where the bucket threshold is RAISED by ASYM_BAD_HOURS_DELTA.
# Same mechanism as ASYM_HOUR_BOOST_GOOD_HOURS / ASYM_HOUR_BOOST_DELTA but the
# delta is ADDED to T instead of subtracted (= the model becomes MORE selective
# on these hours, fewer bets, higher confidence required).
# Unlike ASYM_BAD_HOURS which is a HARD skip (T=1.0), this lets the model still
# bet if its confidence is high enough. Read at runtime by
# api.py._bucket_threshold (no retrain needed). Empty tuple disables.
ADAPTIVE_BAD_HOURS = (13,3,15,5,9,00,1,11, 22,5)
ASYM_BAD_HOURS_DELTA = 0.01

# --- Cross-model filter (V3 extension) ---
# Pour chaque bar pris par modele_up, on lit p_down du modele_down (meme bar,
# join par timestamp) et on n'engage le pari UP que si p_down < ASYM_MAX_P_DOWN.
# Rationale (cf optimize/deep_analysis.py) : sur les 1159 bets actuels,
# p_down in [0.47, 0.50) donne 51.2 % WR vs 63.7 % pour p_down in [0.45, 0.47).
# La frontiere est nette autour de 0.47.
# 1.0 desactive le filtre (= use p_down_xmodel as a model feature instead).
ASYM_MAX_P_DOWN = 0.530
# 2026-05-25 : retuned from 0.470 -> 0.530 because modele_down's p_down distribution
# was shifted upward (mean 0.50 -> 0.55) by the spw=1.30 fix that aligned its
# probability distribution with UP (cf project_modele_down_15min_high_wr). The
# old 0.47 cutoff rejected 100 % of UP bets after the shift; 0.530 reproduces
# the same SEMANTIC filter (bars where modele_down is "above its own mean").
ASYM_MODELE_DOWN_PREDS = "../modele_down/logs/test_predictions.csv"

# --- Hyperparamètres XGBoost (identiques à modele_15m) ---
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
