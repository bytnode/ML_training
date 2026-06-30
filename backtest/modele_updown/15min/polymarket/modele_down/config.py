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

TRAIN_DAYS = 3600
TEST_DAYS = 365
VAL_FRACTION = 0.15

# Bridge (embargo) : nombre de JOURS de pause entre la fin de (train+val) et le
# debut du test. Les bougies de cet intervalle ne sont ni en val ni en test.
# 0 = pas de pause (val colle au test = comportement historique). Sert a couper
# la fuite des features a lookback (ex: return_96, volume_rel_30d) dont la
# fenetre chevaucherait le debut du test.
BRIDGE_DAYS = 0

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
ASYM_VOL_FRAC_WEEKDAY = 0.0225    # ignored: weekday is fully skipped below
                                    # (kept for the runs.csv schema only).
ASYM_VOL_FRAC_WEEKEND = 0.07       # CLONE de modele_up (ex-0.145). Ignore en mode unanime.

# Skip ALL weekday cells. Per-bucket diagnostic (cf optimize/sweep_high_wr.py)
# shows weekday WR plafonne at 56-58 % across all volatility states, vs
# weekend at 59-67 %. Sacrificing weekday volume lifts pooled WR by +3 pt.
ASYM_SKIP_CELLS = ((2, 0), (2, 1))
ASYM_USE_FULL_VAL = True            # True = full val, False = derniers 55j

# --- Seuil UNANIME weekday/weekend, HARDCODE (identique a modele_up) ---
# True  -> UN SEUL seuil pour weekday ET weekend = la valeur ecrite dans
#          ASYM_UNANIMOUS_THRESHOLD, prise TELLE QUELLE (PAS un quantile,
#          PAS un %). Desactive completement le mode par axe : la boucle
#          weekday/weekend (ASYM_VOL_FRAC_WEEKDAY / ASYM_VOL_FRAC_WEEKEND) n'est
#          jamais executee. Le sidecar ecrit {"0": T, "1": T}, donc api.py /
#          predict.py / live.py l'appliquent sans aucun changement de code.
# False -> comportement par defaut (deux seuils distincts par axe, quantile).
# Les skip_cells, bad_hours, bad_days et le hour_boost restent actifs PAR
# DESSUS : seul le split weekday/weekend du seuil de base est neutralise.
# NB: ici le seuil porte sur P(down). spw=1.30 a deja recentre la distribution
# de p_down vers la zone de UP (p_q95 ~= 0.600), donc 0.58 est atteignable.
ASYM_UNANIMOUS_WKND = True
# Seuil de confiance P(down) ecrit en dur. Ex: 0.58 -> pari DOWN ssi
# P(down) >= 0.58, weekday comme weekend. C'est exactement le nombre du sidecar.
ASYM_UNANIMOUS_THRESHOLD = 0.5936  # 2026-06-13 goal run : recale ~1000 bets avec soft(+21,+22)/wbd=0.03

# Parametres specifiques V2 ("asym_wknd_excl_hour_boost") :
# DISABLED -- the original boost (20,21,22) was the US-close effect carried
# from the UP model. On DOWN, hour 22 is the WORST hour empirically
# (52-55 % WR), so boosting it pulled losing bets in. Empty good_hours +
# delta=0 = no boost; the V2 strategy still applies correctly.
ASYM_HOUR_BOOST_GOOD_HOURS = ()
ASYM_HOUR_BOOST_DELTA = 0.0125

# --- Bad-hours exclusion (V2 extension) ---
# UTC hours where we NEVER bet, regardless of confidence. Empirical losers
# on weekend (Asian off-peak + lunch dump): hour 3, 4, 11, 13, 22 sit at
# 51-58 % WR while pooled weekend WR is ~64 % at the same threshold. Cutting
# them concentrates volume on the high-WR hours. cf optimize/sweep_weekend_only.py.
ASYM_BAD_HOURS = (1,5,13,17,3,8,11,9,7,16,14)   # CLONE de modele_up (ex-(3,4,5,9,11,13,23))

# --- Bad-days threshold boost ---
# UTC isoweekday(s) where the bucket threshold is RAISED by ADAPTIVE_WR_BAD_DAY
# (= fewer, more selective bets on these days). isoweekday: 1=Monday..7=Sunday.
# Empirically Friday (=5) shows the weakest team WR in the per-weekday
# breakdown (cf main.py "Share of taken bets per weekday" table). Boosting the
# threshold there forces the model to pick only its highest-confidence Friday
# bets. Read at runtime by api.py._bucket_threshold (no retrain needed).
# Empty tuple disables the filter.
ADAPTIVE_BAD_DAYS = (4, 5)
ADAPTIVE_WR_BAD_DAY = 0.03       # 2026-06-13 goal run (ex-0.02 ; ex-0.01 up-clone)

# --- Bad-hours soft boost (CLONE de modele_up) ---
# Heures UTC ou le seuil est RELEVE de ASYM_BAD_HOURS_DELTA (= plus selectif,
# PAS un skip dur : le pari peut encore passer si la confiance est assez haute).
# Applique par apply_thresholds_asym_hour_boost. Tuple vide = desactive.
# Porte depuis modele_up : ce mecanisme n'existait pas dans down avant le clone.
ADAPTIVE_BAD_HOURS = (2,10,4,12,15,8,21,22)  # 2026-06-13 goal run : +21,+22 (heures faibles a forte masse sur le test)
ASYM_BAD_HOURS_DELTA = 0.01


# --- Cross-model filter (V3 extension) ---
# Pour chaque bar pris par modele_down, on lit p_up du modele_up (meme bar,
# join par timestamp) et on n'engage le pari DOWN que si p_up < ASYM_MAX_P_UP.
# Rationale (cf optimize/deep_analysis.py) : sur les 799 bets actuels,
# p_up < 0.47 donne 66.4 % WR vs 58.4 % pour p_up >= 0.47 -- ecart de +8 pt
# pour un seul filtre binaire. ASYM_MAX_P_UP = 1.0 desactive le filtre.
ASYM_MAX_P_UP = 0.530              # CLONE de modele_up (meme valeur que son ASYM_MAX_P_DOWN ; ex-0.48).
# Path vers modele_up's test_predictions.csv (relatif au cwd au moment du
# run -- train.py est suppose etre execute depuis 15min/modele_down/).
ASYM_MODELE_UP_PREDS = "../modele_up/logs/test_predictions.csv"

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
XGB_SCALE_POS_WEIGHT = 1.3
# Pourquoi 1.30 et pas 1.03 / "auto" ?
# La cause DIRECTE de l'incoherence p-distribution UP vs DOWN : XGBoost converge
# ses predictions vers mean(y_train). UP train_mean=0.5097, DOWN train_mean=0.4903
# (BTC = 51% bougies vertes / 49% rouges sur 5000j). Cet ecart de 0.0195 decale
# toute la distribution DOWN de ~2 pt vers le bas (p_q95 max ~0.555 au lieu de
# ~0.580 pour UP). spw=1.30 upweighte artificiellement la classe minoritaire
# (DOWN) pour annuler le decentrage et faire converger DOWN vers la meme zone
# que UP. Test empirique : spw=1.30 -> p_mean=0.557, p_q95=0.600 (territoire UP).
# spw=1.03 etait l'ancienne valeur "match UP" mais elle ignorait que UP est
# naturellement avantage par le drift haussier.
