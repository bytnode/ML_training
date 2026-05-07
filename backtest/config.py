# ============================================================
#  SECTION 1 : DATA (utilisé par data.py)
# ============================================================

# --- Source de données ---
SYMBOL = "btcusd"        # Bitstamp pair code (minuscules : btcusd, btcusdt, ethusd, ...)
INTERVAL = "15m"
DAYS_BACK = 5000
OUTPUT_CSV = "training_data.csv"

# --- Multi-pair dataset (consume par data_multi.py uniquement) ---
SYMBOLS_MULTI = ["btcusd", "ethusd", "solusd", "bnbusd", "xrpusd"]
SYMBOL_TO_ID = {s: i for i, s in enumerate(SYMBOLS_MULTI)}  # btcusd=0, ethusd=1, solusd=2, bnbusd=3, xrpusd=4
OUTPUT_CSV_MULTI = "training_data_multi.csv"

# --- API Bitstamp ---
API_URL = "https://www.bitstamp.net/api/v2/ohlc/{pair}/"
KLINES_LIMIT = 1000      # max autorisé par Bitstamp
REQUEST_TIMEOUT = 30
MAX_RETRIES = 3
FETCH_WORKERS = 8

INTERVAL_MS = {
    "15m": 15 * 60 * 1000,
    "30m": 30 * 60 * 1000,
    "1h":  60 * 60 * 1000,
    "4h":  4 * 60 * 60 * 1000,
    "1d":  24 * 60 * 60 * 1000,
}

# Higher timeframes ajoutees a chaque ligne du dataset (OHLCV de la derniere bougie
# closed avant le timestamp de decision T = times[last_idx] + 15m).
MTF_INTERVALS = ["30m", "1h", "4h", "1d"]

# --- Extraction des features ---
WARMUP = 200          # bars ignorées en début pour stabiliser les indicateurs
MAX_SEQ_LEN = 4       # longueurs de streak testées : 1..MAX_SEQ_LEN
BODY_RATIO_MIN = 0.6  # body ratio min de la dernière bougie pour entrer dans le dataset

# --- Périodes des indicateurs ---
EMA_OBV_PERIOD = 20
EMA_CLOSE_PERIODS = (20, 50, 200)
RSI_PERIOD = 14
ATR_PERIOD = 14
ATR_LONG_PERIOD = 100
VOL_AVG_PERIOD = 20

# --- Fenêtres de calcul des features ---
OBV_SLOPE_WINDOW = 10
RSI_CHANGE_WINDOW = 5

# --- Seuils de l'état de volatilité (ratio ATR court / ATR long) ---
VOLATILITY_HIGH = 1.5
VOLATILITY_LOW = 0.7


# ============================================================
#  SECTION 2 : TRAINING (utilisé par train.py)
# ============================================================

# --- Fichiers ---
CSV_PATH = "training_data.csv"
MODEL_OUTPUT = "btc_reversal_model.json"

TRAIN_DAYS   = 4000     # 1825 jours = 5 ans  (None = utilise tout ce qui reste)
TEST_DAYS    = 365      # 365 jours = 1 an
VAL_FRACTION = 0.1   # 15% du bloc train+val devient le set de validation (la partie la plus recente du bloc)

# Filtre du test set : liste des seq_len a evaluer.
#   None ou liste vide = toutes les seq_len presentes dans le CSV (comportement par defaut).
#   [2, 3]             = n'evalue que sur les rows seq_len in {2, 3}.
# N'affecte PAS train/val : le modele est toujours entraine sur tout.
SEQ_TEST = [2, 3, 4]

# --- Évaluation ---
THRESHOLDS = [0.50, 0.52, 0.55,0.56,0.57, 0.58,0.59, 0.60,0.61, 0.62]
BREAKDOWN_THRESHOLD = 0.6

# --- Seuil adaptatif ---
# Le seuil utilise pour decider take/skip est BREAKDOWN_THRESHOLD + delta(bucket),
# ou bucket = (volatility_state, is_weekend). Les deltas sont calcules sur les
# derniers ADAPTIVE_CALIB_FRACTION * TEST_DAYS jours de la slice de validation et
# sauves dans un sidecar JSON a cote du modele.
ADAPTIVE_THRESHOLD = True               # False = utilise BREAKDOWN_THRESHOLD partout (comportement legacy)
ADAPTIVE_TARGET_WR = 0.58               # WR cible par bucket (egal a la base = "lisse les regimes")
ADAPTIVE_CALIB_FRACTION = 0.75          # fenetre de calib = ADAPTIVE_CALIB_FRACTION * TEST_DAYS (0.15 * 365 = ~54j)
ADAPTIVE_MIN_N_VAL = 200                # seuil minimum de rows par bucket pour calibration "high confidence"
ADAPTIVE_MIN_N_VAL_SOFT = 30            # plancher "low confidence" : si calib + full val sont sous MIN_N_VAL mais                               # qu'on a au moins ce nombre dans full val, on calibre quand meme (flag low_conf)
ADAPTIVE_MIN_N_AT_THRESHOLD = 30        # seuil minimum de bets a un threshold candidat (anti-overfit du delta)
ADAPTIVE_DELTA_CLAMP = (-0.05, 0.08)    # rail de securite : le delta par bucket reste dans ces bornes

# --- Hyperparamètres XGBoost ---
XGB_N_ESTIMATORS = 1000
XGB_MAX_DEPTH = 5
XGB_LEARNING_RATE = 0.05
XGB_MIN_CHILD_WEIGHT = 5
XGB_SUBSAMPLE = 0.8
XGB_COLSAMPLE_BYTREE = 0.8
XGB_REG_LAMBDA = False    # désactivé (False = utilise le défaut XGBoost = 1.0)
XGB_REG_ALPHA = False     # désactivé (False = utilise le défaut XGBoost = 0.0)
XGB_GAMMA = False        # désactivé (False = utilise le défaut XGBoost = 0.0)
XGB_EARLY_STOPPING_ROUNDS = 50
XGB_RANDOM_STATE = 42