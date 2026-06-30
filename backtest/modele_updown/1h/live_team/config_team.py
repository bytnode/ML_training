"""config_team.py — configuration du daemon live MEXC `live_team`.

Miroir des constantes de trading de `modele_team/config_main.py` (SL/TP/trailing/
timestop, levier, exposure) + le bloc de paramètres propres au live MEXC.

Les seuils de gestion de sortie sont identiques au backtest : TP/trailing ET SL
tous exprimés en **décimal de PRIX** (sur la position, levier hors-jeu). EXPOSURE
n'intervient JAMAIS dans les seuils de sortie — uniquement dans le sizing (cf live.py).
"""

import os

# ============================================================
#  Frais / capital / levier  (miroir config_main.py)
# ============================================================
INITIAL_CAPITAL = 5000.0          # référence (le live compose sur le solde réel)
POSITION_FRACTION = 0.5           # part du capital engagée comme marge par position
LEVERAGE = 10                     # levier appliqué sur la marge
EXPOSURE = POSITION_FRACTION * LEVERAGE   # = 5.0 ; multiplicateur effectif du rendement

# Frais MEXC futures (barème API) : maker 0.01%, taker 0.05%.
MAKER_FEE = 0.0001
TAKER_FEE = 0.0005

# ============================================================
#  Constantes temps (ms)
# ============================================================
DAY_MS = 24 * 3600 * 1000
INTERVAL_1H_MS = 60 * 60 * 1000

# ============================================================
#  Gestion de sortie cote UP (long)  — miroir config_main.py
# ============================================================
UP_MGMT_ENABLED = True
UP_MAX_HOLD_MIN = 90
UP_TRAIL_ACTIVATE = 0.0035        # trailing armé quand le pic atteint +0.35%
UP_TRAIL_GAP = 0.008              # sort si le profit retombe de 0.80% sous le pic
UP_TAKE_PROFIT = 0.012            # take-profit +1.20% (décimal de prix)
UP_STOP_LOSS = 0.60               # SL en % du PRIX (sur la position, levier hors-jeu ; utilisé si time-decay OFF)
UP_CHECK_MIN = 1                  # intervalle de vérif (min) = monitoring 1-min
# dérivés (ne pas éditer)
UP_STEP_MS = UP_CHECK_MIN * 60 * 1000
UP_MAX_STEPS = UP_MAX_HOLD_MIN // UP_CHECK_MIN
UP_STOP_LOSS_DEC = -(UP_STOP_LOSS / 100.0)
# SL temporel décroissant (START large -> END serré, borné par FLOOR), en % PRIX (sur la position)
UP_SL_TIME_DECAY = True
UP_SL_START = 1                   # SL à l'entrée (large), en % prix
UP_SL_END = 0.60                  # SL à la clôture forcée (serré), en % prix
UP_SL_FLOOR = 0.30                # SL le plus serré autorisé (plancher), en % prix
UP_SL_START_DEC = -(UP_SL_START / 100.0)
UP_SL_END_DEC = -(UP_SL_END / 100.0)
UP_SL_FLOOR_DEC = -(UP_SL_FLOOR / 100.0)

# ============================================================
#  Gestion de sortie cote DOWN (short)  — miroir config_main.py
# ============================================================
DOWN_MGMT_ENABLED = True
DOWN_MAX_HOLD_MIN = 105
DOWN_TRAIL_ACTIVATE = 0.0025      # trailing short armé à +0.25%
DOWN_TRAIL_GAP = 0.0050           # sort si le profit short retombe de 0.50% sous le pic
DOWN_TAKE_PROFIT = 0.003          # take-profit short +0.30% (levier principal du WR)
DOWN_STOP_LOSS = 0.60             # SL en % du PRIX (sur la position, levier hors-jeu ; utilisé si time-decay OFF)
DOWN_CHECK_MIN = 1
# dérivés (ne pas éditer)
DOWN_STEP_MS = DOWN_CHECK_MIN * 60 * 1000
DOWN_MAX_STEPS = DOWN_MAX_HOLD_MIN // DOWN_CHECK_MIN
DOWN_STOP_LOSS_DEC = -(DOWN_STOP_LOSS / 100.0)
DOWN_SL_TIME_DECAY = True
DOWN_SL_START = 0.80              # SL à l'entrée (large), en % prix
DOWN_SL_END = 0.30               # SL à la clôture forcée (serré), en % prix
DOWN_SL_FLOOR = 0.15             # SL le plus serré autorisé (plancher), en % prix
DOWN_SL_START_DEC = -(DOWN_SL_START / 100.0)
DOWN_SL_END_DEC = -(DOWN_SL_END / 100.0)
DOWN_SL_FLOOR_DEC = -(DOWN_SL_FLOOR / 100.0)

# ============================================================
#  Validation des données AVANT prédiction (garde-fou anti mauvaise donnée)
# ============================================================
# Couche de contrôle exécutée dans predict.decide juste avant predict_proba : si
# une donnée est douteuse, AUCUNE prédiction n'est faite. live.py re-fetch et
# ré-essaie VALIDATION_MAX_RETRIES fois ; si toujours invalide -> cycle ANNULÉ
# (aucun pari) + alerte Telegram. Met VALIDATION_ENABLED=False pour revenir à
# l'ancien comportement (zéro validation).
VALIDATION_ENABLED = True          # False = aucune validation (ancien comportement)
VALIDATION_MAX_RETRIES = 3         # nb de re-fetch si donnée incomplète avant d'annuler le cycle
VALIDATION_RETRY_SLEEP_SEC = 5     # pause (s) entre deux essais (laisse Bitstamp publier la bougie)
VALIDATION_MIN_1H_BARS = 200       # barres 1h minimum (warmup des rolling : EMA50, rolling 50/48/24, ATR_LONG 100)
VALIDATION_GAP_WINDOW = 150        # fenêtre récente (barres) où l'OHLC 1h doit être intègre ET sans trou

# ============================================================
#  Paramètres live MEXC  (overridables par variables d'env)
# ============================================================
SYMBOL = os.getenv("MEXC_SYMBOL", "BTC_USDT")   # perpétuel USDT-M
MEXC_LEVERAGE = LEVERAGE                          # levier réglé sur l'exchange
OPEN_TYPE = 1                                     # 1 = isolated, 2 = cross
POSITION_MODE = 1                                 # 1 = hedge (2 positions possibles), 2 = one-way

# Sizing : marge = BET_FRACTION * solde_futures_courant (composé) ; notionnel = marge * levier.
BET_FRACTION = float(os.getenv("BET_FRACTION", str(POSITION_FRACTION)))   # défaut 0.5
MIN_VOL = int(os.getenv("MIN_VOL", "1"))          # contrats minimum (validation micro-volume)
MAX_CONCURRENT_POSITIONS = 2                      # mathématiquement ≤2 (max_hold < 2h)

# Entrée maker : 3 tentatives de 5 min, re-pricée au marché courant à 0.01% passif.
ENTRY_MAKER_OFFSET = 0.0001                       # 0.01%
ENTRY_ATTEMPTS = 3
ENTRY_ATTEMPT_SEC = 300                           # 5 min par tentative

# Sortie maker (uniquement en profit) : 3 tentatives post-only de 20 s puis taker.
EXIT_MAKER_ATTEMPTS = 3
EXIT_MAKER_WAIT_SEC = 20
EXIT_MAKER_OFFSET = 0.0001                        # 0.01% passif

# Réseau MEXC
# ⚠️ FUTURES = contract.mexc.com (api.mexc.com = SPOT). À VÉRIFIER par un appel
# public (get_contract_detail BTC_USDT) sur ton VPS avant tout mode réel : si la
# migration vers api.mexc.com est effective, surcharge MEXC_REST_HOST dans .env.
REST_HOST = os.getenv("MEXC_REST_HOST", "https://contract.mexc.com")
WS_URL = os.getenv("MEXC_WS_URL", "wss://contract.mexc.com/edge")
RECV_WINDOW = int(os.getenv("MEXC_RECV_WINDOW", "10000"))

# Mode test : signe/prépare les ordres mais NE POSTE PAS (validation connexion/signature).
DRY_RUN = os.getenv("DRY_RUN", "false").lower() in ("1", "true", "yes", "y", "on")
