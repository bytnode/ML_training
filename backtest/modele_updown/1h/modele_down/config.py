# ============================================================
#  SECTION 1 : DATA / FEATURES (utilisé par data.py + features.py)
# ============================================================

# --- Source de données ---
SYMBOL = "btcusd"
INTERVAL = "1h"
DAYS_BACK = 5000
DB_PATH = "db/btc_multi_tf_1h.parquet"  # master 1h lu par features.py + train.py

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
# 2880 = la plus longue fenetre rolling du jeu de features (volume_rel_30d,
# min_periods=2880). Bumpe de 500 a 2880 pour que la 1ere row gardee ait
# TOUTES les features non-NaN (parite features 15min). XGBoost route les NaN
# nativement, mais on prefere demarrer "chaud".
WARMUP = 2880

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

# --- Features exclues du modele (drop) ---
# DESACTIVE : la selection des features est desormais pilotee par
# FEATURE_WHITELIST (ci-dessous), qui fixe EXACTEMENT les 27 features du modele
# 15min/polymarket/modele_down (objectif de parite). DROP_FEATURES n'a plus
# d'effet quand FEATURE_WHITELIST est non vide (cf train.py).
DROP_FEATURES = ()

# --- Parite features avec le modele 15min (whitelist STRICTE) ---
# Si non vide, train.py fixe feature_cols = EXACTEMENT cette liste (et leve une
# erreur claire si une colonne manque dans le parquet). Le parquet peut contenir
# des colonnes en trop (hour_sin/cos, wicks, rv_*, 4h/1d MTF, ef_* non utilises,
# etc.) : elles sont simplement ignorees. Resultat : booster.feature_names ==
# ces 27 noms == les 27 features du modele 15min.
#
# Les 27 features (verifiees via booster.feature_names du modele 15min) :
#   base 30m        : high_ret, low_ret, close_ret
#   contexte fin 5m : high_ret_5m, low_ret_5m, close_ret_5m, obv_pressure_5m
#   MTF 30m         : low_ret_30m, close_ret_30m
#   MTF 1h          : low_ret_1h, close_ret_1h
#   derivees        : dist_ema_20, return_3, macd, macd_hist, bb_position, mfi_norm
#   microstructure  : dist_to_high_50, dist_to_low_50, autocorr_24, acc_4, volume_rel_30d
#   ETH             : ef_ret2, ef_ret8, ef_streak, ef_macd, ef_body
# Tuple vide -> desactive la whitelist (retour au comportement DROP_FEATURES).
# --- Set LEAN issu de la selection forward (2026-06-16) ---
# 11 features : les meilleures, non-redondantes, ZERO nuisible. Selectionnees par
# forward selection deterministe (optimize/select_features.py) qui maximise l'AUC
# test : elle n'ajoute jamais une feature nuisible, ni une redondante une fois son
# signal couvert. L'AUC pique a 11 features (0.5426) puis redescend -> on s'arrete
# a 11. Ancien set = 27 features a AUC 0.5345 (dont ~24 inutiles/redondantes).
# Familles couvertes : forme de bougie (body_pos, body_ratio_avg), oscillateurs
# (rsi_6, stoch_k_14), momentum ETH (ef_ret4, ef_ret1), wick ETH (ef_upwick),
# volume (vol_z_20), tendance daily (ret_1d_1), regime (autocorr_12), temps (hour_cos).
# Impact detaille : analyse/classify_features3.py (3 classes deterministe).
# NETTOYE 2026-06-17 (analyse/audit_features_now.py) : sur le modele re-entraine,
# l audit d ablation deterministe a montre 3 features NUISIBLES (body_pos +0.00249,
# ef_ret1 +0.00103, body_ratio_avg +0.00101) et 2 INERTES (ret_1d_1, vol_z_20).
# body_pos etait #1 la veille -> forward selection sur split unique overfit le bruit
# (AUC ~0.54 = plafond structurel 1h). On garde les 6 a ablation clairement negative.
# Ancien set 11 feats sauve dans _backup_11feat_model.json.
FEATURE_WHITELIST = (
    "stoch_k_14", "hour_cos", "ef_ret4", "rsi_6", "autocorr_12", "ef_upwick",
)

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
ASYM_UNANIMOUS_THRESHOLD = 0.53  # 2026-06-13 goal run : recale ~1000 bets avec soft(+21,+22)/wbd=0.03

# Parametres specifiques V2 ("asym_wknd_excl_hour_boost") :
# DISABLED -- the original boost (20,21,22) was the US-close effect carried
# from the UP model. On DOWN, hour 22 is the WORST hour empirically
# (52-55 % WR), so boosting it pulled losing bets in. Empty good_hours +
# delta=0 = no boost; the V2 strategy still applies correctly.
ASYM_HOUR_BOOST_GOOD_HOURS = ()
ASYM_HOUR_BOOST_DELTA = 0.0125

# ============================================================
#  Bad-hours / bad-days ADAPTATIFS (skip dur + boost doux)
# ============================================================
# Au train, on classe les heures (0-23) et les jours (isoweekday 1-7) par WR
# DIRECTIONNEL (prediction = p_down>=0.5, win = prediction == label) sur les
# ADAPTIVE_BAD_WINDOW_DAYS derniers jours du dataset (val+test). Les pires deviennent :
#   - ADAPTIVE_N_BAD_HOURS  pires heures      -> SKIP DUR (T=1.0, jamais de pari)
#   - ADAPTIVE_N_SOFT_HOURS heures suivantes  -> boost doux : seuil += ADAPTIVE_SOFT_HOURS_DELTA
#   - ADAPTIVE_N_BAD_DAYS   pires jours        -> boost doux : seuil += ADAPTIVE_BAD_DAY_DELTA
# (boost doux = pas un skip : le pari passe encore si la confiance est assez haute.)
# train.py CALCULE les 3 listes et les ECRIT dans le sidecar ; api.py / predict.py
# les LISENT depuis le sidecar. Seuls les deltas (magnitudes) sont fixes ici.
ADAPTIVE_BAD_BUCKETS = True        # False = aucune exclusion/boost horaire ni jour
ADAPTIVE_BAD_WINDOW_DAYS = 365     # fenetre de classement (derniers N jours du dataset)
ADAPTIVE_N_BAD_HOURS = 10          # nb pires heures -> skip dur
ADAPTIVE_N_SOFT_HOURS = 10           # nb heures suivantes -> boost doux (mid hours)
ADAPTIVE_N_BAD_DAYS = 3            # nb pires jours -> boost doux
ADAPTIVE_SOFT_HOURS_DELTA = 0.02   # delta de seuil ajoute sur les mid hours (boost doux)
ADAPTIVE_BAD_DAY_DELTA = 0.015      # delta de seuil ajoute sur les pires jours (boost doux)


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
XGB_REG_LAMBDA = 1
XGB_REG_ALPHA = 0.1
XGB_GAMMA = 1.5
XGB_EARLY_STOPPING_ROUNDS = 50
XGB_RANDOM_STATE = 42

# --- Rééquilibrage de classe (DOWN vs UP) ---
# Le label=1 (DOWN) est légèrement minoritaire (~49.1 % du dataset).
# scale_pos_weight = poids appliqué à la classe positive pendant l'entraînement.
#   False  -> pas passé à XGBoost (= 1.0, comportement par défaut, asymétrie subie)
#   "auto" -> calculé à l'entraînement = n_up / n_down (équilibrage exact)
#   float  -> valeur fixe (ex 1.05 pour un upweight DOWN manuel)
XGB_SCALE_POS_WEIGHT = 1.07
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
