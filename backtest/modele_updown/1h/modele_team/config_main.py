"""config_main.py — configuration centralisee du backtest trading (main.py).

Tous les parametres editables de main.py vivent ici ; main.py les importe.
(Les chemins de fichiers restent dans main.py car derives de __file__.)
"""

# ============================================================
#  Fenetre de backtest
# ============================================================
TEAM_TEST_DAYS = 365          # taille de la fenetre de test (jours), ancree sur la
                              # derniere bougie commune aux 2 modeles

# ============================================================
#  Frais / capital
# ============================================================
FEE_PER_SIDE = -0.000        # frais par cote (achat ET vente) ; aller-retour = 2x
                              # NB : les frais portent sur le NOTIONNEL -> amplifies par le levier
INITIAL_CAPITAL = 5000.0      # capital de depart ($)

# --- Sizing / levier (futures-style) ---
# Chaque trade engage POSITION_FRACTION du capital comme MARGE, avec un LEVIER dessus.
# Exposition effective au mouvement de prix = POSITION_FRACTION * LEVERAGE (= EXPOSURE).
# Ex : 50% du capital x levier 10 = 5.0x -> un mouvement +0.30% rapporte +1.50% sur l'equity.
# P/L ET FRAIS sont tous deux multiplies par EXPOSURE (notionnel = EXPOSURE * equity).
POSITION_FRACTION = 0.5      # part du capital engagee par trade (marge)
LEVERAGE = 10              # levier applique sur la marge
EXPOSURE = POSITION_FRACTION * LEVERAGE   # multiplicateur effectif du rendement par trade (= 5.0)

# --- Mode MAKER (ordres limites) vs TAKER (ordres marche) ---
# TAKER (defaut) : on entre AU MARCHE a l'open de la bougie i+1 (= P0), frais taker.
# MAKER : on poste une LIMITE a P0 (le + proche possible du prix actuel) et on n'est
# rempli QUE si le prix revient nous toucher pendant la fenetre d'entree (1-min) :
#   - LONG  (achat) : rempli si un low 1-min < P0   (sinon la bougie verte part sans nous)
#   - SHORT (vente) : rempli si un high 1-min > P0   (sinon la bougie rouge part sans nous)
# Les ordres NON remplis = opportunites perdues (comptees, % affiche dans le rapport).
# Un trade rempli a le MEME P0 et la meme gestion qu'en taker (P&L identique), mais
# paie les frais MAKER. -> on isole exactement le cout des opportunites ratees.
MAKER = True                 # True = simule des ordres makers (limites) au lieu de taker
MAKER_FEE_PER_SIDE = -0.0002      # frais maker par cote (souvent 0 ou negatif/rebate ; mets le taux de ta plateforme)
MAKER_FILL_WINDOW_MIN = 15    # duree (min) ou la limite reste postee avant annulation (= opportunite perdue)

# ============================================================
#  Stop-loss intra-bougie
# ============================================================
# Le SL est verifie a chaque sous-bougie de STOP_LOSS_CHECK_MIN minutes, a
# l'interieur de la bougie 1h tradee. Si le profit directionnel casse STOP_LOSS_PCT
# a un de ces pas, on sort a cette sous-bougie (sinon sortie classique a la
# cloture 1h). Source des sous-bougies : DB_<m>m/btc_<m>m.parquet (m = check_min).
STOP_LOSS_ENABLED = False      # False = aucun SL (sortie classique a la cloture 1h)
STOP_LOSS_PCT = -0.006       # SL FIXE a -0.75% (profit <= ce seuil => sortie)
STOP_LOSS_SIDES = ()     # sens auxquels le SL s'applique.
                              #   ("up",)        = UP (long) seulement  <-- actuel
                              #   ("up", "down") = les deux
                              #   ()             = aucun
                              # Rationale : les pertes UP tombent sur les chutes
                              # violentes (rouge ~0.37% vs 0.30% typique) ; les
                              # pertes DOWN sur des hausses molles (~0.27%). Le SL
                              # n'a de valeur que cote UP.
STOP_LOSS_CHECK_MIN = 1       # intervalle de verification du SL, en MINUTES (1 = 1min)

# ============================================================
#  Constantes temps (ms)
# ============================================================
DAY_MS = 24 * 3600 * 1000
INTERVAL_1H_MS = 60 * 60 * 1000

# ============================================================
#  Derives (ne pas editer a la main)
# ============================================================
STOP_LOSS_STEP_MS = STOP_LOSS_CHECK_MIN * 60 * 1000        # pas de verif (ms)
STOP_LOSS_N_STEPS = INTERVAL_1H_MS // STOP_LOSS_STEP_MS    # nb de checks dans 1h (60 si 1min)

# ============================================================
#  Gestion gain/perte cote UP (trailing 'let-run' + stop-loss)
# ============================================================
# But : GESTION DE SORTIE pure (AUCUN filtre d'entree -> 724 trades intacts) cote
# LONG (up), pour remonter le WR >= 56% et tenir la perte moyenne <= 0.35%
# (optimisee plus bas si possible). DOWN reste totalement inchange.
# Diagnostic (sweeps sweep_up_loss*.py) : le let-run tenait 2h et rendait des gains
# de fin de bougie 1h -> WR 54.4%. La cloture 1h + un TAKE-PROFIT et un trailing
# serre VERROUILLENT le gain (remontent le WR), un SL large cape la queue sans
# whipsaw. Verifie chaque UP_CHECK_MIN min sur DB_1m.
#
# CHOIX user = "gain & net max" (gain >= 0.30, WR >= 56, perte la plus basse possible).
# MUR PROUVE (sweeps 7/8) : perte <= 0.30 et WR >= 56 s'EXCLUENT -> caper la perte a
# 0.30 whipsaw le WR a ~50%. Perte minimale a WR>=56 & gain>=0.30 = ~-0.35%.
# Reglages (change TP/trail/SL) — SL = % du CAPITAL REEL (independant du levier) :
# les SL ci-dessous sont en % PRIX (base sweep, invariant au levier) ; champ a saisir = ce% x EXPOSURE.
#   DEFAUT          max90 TP=0.0080 trail 0.0035/0.0050 SL=0.90 prix -> gain 0.31/WR 57/perte -0.36/net +14
#   net max         idem mais SL=5.0 prix (off)                      -> gain 0.31/WR 57/perte -0.36/net +17
#   perte serree    max75 TP=0.0030 trail 0.0020/0.0030 SL=0.60 prix -> gain 0.24/WR 59/perte -0.31/net +9
UP_MGMT_ENABLED = True
UP_MAX_HOLD_MIN = 90          # detention max (min) ; laisse courir ~30min apres la cloture 1h
UP_TRAIL_ACTIVATE = 0.0035    # le trailing s'arme quand le pic atteint +0.35%
UP_TRAIL_GAP = 0.008       # sort si le profit retombe de 0.80% sous le pic (sweep 2026-06-18 : trailing actif)
UP_TAKE_PROFIT = 0.012     # take-profit +1.20% (sweep conjoint 2026-06-18 : meilleur net que 2.0% avec SL temporel)
UP_STOP_LOSS = 0.60     # SL en % du PRIX (sur la position), levier hors-jeu : 0.60 = sortie
                              # quand le prix bouge de 0.60% contre nous. (TP/trail aussi en % prix.)
                              # Inactif tant que UP_SL_TIME_DECAY=True (SL temporel ci-dessous).
UP_CHECK_MIN = 1              # intervalle de verif (min) -> DB_1m
UP_SKIP_BUCKETS = ()          # AUCUN filtre d'entree : selection intacte (724 trades)
# derives (ne pas editer)
UP_STEP_MS = UP_CHECK_MIN * 60 * 1000
UP_MAX_STEPS = UP_MAX_HOLD_MIN // UP_CHECK_MIN
UP_STOP_LOSS_DEC = -(UP_STOP_LOSS / 100.0)   # % prix -> stop PRIX (decimal neg), directement
# --- SL TEMPOREL (decroissant) : le SL se resserre lineairement de START (large, a l'entree)
#     vers END (serre, a max_hold), borne par FLOOR (jamais plus serre). Toggle False = SL
#     constant UP_STOP_LOSS ci-dessus. Tout en % du PRIX (sur la position, levier hors-jeu). ---
UP_SL_TIME_DECAY = True     # ON : sweep 2026-06-18 (premisse : long perdant en fin de bougie = 0% retournement)
UP_SL_START = 1          # SL a l'entree (large), en % prix (sweep 2026-06-18)
UP_SL_END   = 0.60          # SL a la cloture forcee (serre), en % prix
UP_SL_FLOOR = 0.30          # SL le plus serre autorise (plancher), en % prix
UP_SL_START_DEC = -(UP_SL_START / 100.0)
UP_SL_END_DEC   = -(UP_SL_END   / 100.0)
UP_SL_FLOOR_DEC = -(UP_SL_FLOOR / 100.0)

# ============================================================
#  Gestion gain/perte cote DOWN (SHORT) — calque exact de UP
# ============================================================
# Meme gestion de SORTIE pure (AUCUN filtre d'entree -> 754 trades DOWN intacts),
# appliquee au SHORT. Profit directionnel short a la minute j = 1 - close/entry
# (symetrique du long close/entry - 1). DOWN sortait brut a la cloture 1h :
# gain 0.271 / perte -0.318 / net +12.44 / WR 56.76.
# Sweeps (sweep_down_loss*.py : 1800+1400 configs, valides sur 2 moities) -> 3 reglages.
# Change les 5 lignes ci-dessous pour basculer — SL = % du PRIX (sur la position, levier hors-jeu) :
# les valeurs sweep ci-dessous sont directement le % PRIX a saisir.
#   DEFAUT (WR+net) max105 SL=2.5cap(0.50% prix) TP=0.0030 trail 0.0025/0.0050   (sweep 2026-06-18 maker+levier)
#                    -> DOWN WR 62.2% / gain 0.276 / perte -0.352 / DOWN net +232% / TEAM net +772%
#                    (le WR DOWN se pilote par le TP, pas le SL ; SL large 0.80% prix = WR 57 / TEAM +866)
#   3-AXES+          max80  SL=0.70 prix TP=0.0050 trail 0.0025/0.0045
#                    -> gain 0.299 / perte -0.310 / net +24.9 / WR 55.8  (ameliore perte ET gain ET net vs base)
#   PERTE MIN        max60  SL=0.70 prix TP=0.0080 trail 0.0025/0.0045
#                    -> gain 0.261 / perte -0.273 / net +21.0 / WR 56.1  (perte la plus basse)
DOWN_MGMT_ENABLED = True
DOWN_MAX_HOLD_MIN = 105       # detention max (min) ; laisse courir le short ~45min apres la cloture 1h
DOWN_TRAIL_ACTIVATE = 0.0025  # le trailing s'arme quand le pic short atteint +0.25% (sweep 2026-06-18)
DOWN_TRAIL_GAP = 0.0050       # sort si le profit short retombe de 0.50% sous le pic
DOWN_TAKE_PROFIT = 0.003     # take-profit short +0.30% : LEVIER PRINCIPAL du WR DOWN — baisser le TP
                              # verrouille le gain avant reversal -> WR 55->62% a SL 2.5 (sweep 2026-06-18).
DOWN_STOP_LOSS = 0.60      # SL en % du PRIX (sur la position), levier hors-jeu. Le WR se pilote par
                              # le TAKE-PROFIT (au-dessus), PAS par le SL : TP 0.30 -> WR DOWN 62% /
                              # TEAM +772% (sweep 2026-06-18). Inactif tant que DOWN_SL_TIME_DECAY=True.
DOWN_CHECK_MIN = 1            # intervalle de verif (min) -> DB_1m
# derives (ne pas editer)
DOWN_STEP_MS = DOWN_CHECK_MIN * 60 * 1000
DOWN_MAX_STEPS = DOWN_MAX_HOLD_MIN // DOWN_CHECK_MIN
DOWN_STOP_LOSS_DEC = -(DOWN_STOP_LOSS / 100.0)   # % prix -> stop PRIX (decimal neg), directement
# --- SL TEMPOREL (decroissant) cote DOWN : meme principe que UP (cf bloc UP ci-dessus). ---
DOWN_SL_TIME_DECAY = True    # ON : sweep 2026-06-18 (premisse : short perdant en fin de bougie = 0% retournement)
DOWN_SL_START = 0.80         # SL a l'entree (large), en % prix (sweep 2026-06-18)
DOWN_SL_END   = 0.30         # SL a la cloture forcee (serre), en % prix (sweep 2026-06-18)
DOWN_SL_FLOOR = 0.15         # SL le plus serre autorise (plancher), en % prix
DOWN_SL_START_DEC = -(DOWN_SL_START / 100.0)
DOWN_SL_END_DEC   = -(DOWN_SL_END   / 100.0)
DOWN_SL_FLOOR_DEC = -(DOWN_SL_FLOOR / 100.0)
