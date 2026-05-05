# ============================================================
#  SECTION 1 : DATA (utilisé par data.py)
# ============================================================

# --- Source de données ---
SYMBOL = "btcusd"        # Bitstamp pair code (minuscules : btcusd, btcusdt, ethusd, ...)
INTERVAL = "15m"
DAYS_BACK = 5000
OUTPUT_CSV = "training_data.csv"

# --- API Bitstamp ---
API_URL = "https://www.bitstamp.net/api/v2/ohlc/{pair}/"
KLINES_LIMIT = 1000      # max autorisé par Bitstamp
REQUEST_TIMEOUT = 30
MAX_RETRIES = 3
FETCH_WORKERS = 8

INTERVAL_MS = {"15m": 15 * 60 * 1000}

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

# --- Split temporel (pas de shuffle) ---
TEST_DAYS = 3600        # le test set = les TEST_DAYS derniers jours du CSV (730=2ans, 1825=5ans)
VAL_FRACTION = 0.15     # 15% du segment hors-test (= train+val) devient le set de validation

# --- Évaluation ---
THRESHOLDS = [0.50, 0.52, 0.55,0.56,0.57, 0.58, 0.60,]
BREAKDOWN_THRESHOLD = 0.6

# --- Hyperparamètres XGBoost ---
XGB_N_ESTIMATORS = 1000
XGB_MAX_DEPTH = 4
XGB_LEARNING_RATE = 0.05
XGB_MIN_CHILD_WEIGHT = 5
XGB_SUBSAMPLE = 0.8
XGB_COLSAMPLE_BYTREE = 0.8
XGB_REG_LAMBDA = False    # désactivé (False = utilise le défaut XGBoost = 1.0)
XGB_REG_ALPHA = False     # désactivé (False = utilise le défaut XGBoost = 0.0)
XGB_GAMMA = False         # désactivé (False = utilise le défaut XGBoost = 0.0)
XGB_EARLY_STOPPING_ROUNDS = 50
XGB_RANDOM_STATE = 42