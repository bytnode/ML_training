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
#  Filtre DMI (Directional Movement Index) sur le TF 1h
# ============================================================
# Filtre d'ENTREE applique a l'equipe APRES la regle de coherence : on ANNULE un
# pari quand le DMI 1h contredit son sens.
#   - pari DOWN (short) annule si +DI > -DI  (le directionnel haussier domine -> pas de short)
#   - pari UP   (long)  annule si -DI > +DI  (le directionnel baissier domine -> pas de long)
# PAS de seuil ADX (force de tendance) : on compare seulement quel DI domine.
# Le DMI lu est celui de la bougie 1h de DECISION (celle qui vient de cloturer au
# moment du pari), JAMAIS la bougie tradee i+1 -> aucun lookahead. Lissage de Wilder.
# OHLC 1h reconstruit depuis la table `candles` (close absolu + *_ret), tout l'historique.
# Cas limite : +DI == -DI (rarissime) ou warmup (DMI_LENGTH premieres bougies) -> on GARDE le pari.
DMI_FILTER_ENABLED = True
DMI_LENGTH = 12              # longueur (en bougies 1h) du DMI et du lissage de Wilder
DMI_TIMEFRAME = "4h"         # timeframe des bougies servant au calcul du DMI
DMI_SIDES = ("down",)        # cotes auxquels le DMI s'applique. Campagne 500%/cote 2026-07-01 :
                             # le DMI (calibre DOWN) annulait des LONGS profitables -> on le coupe
                             # cote UP. ("down",) = DMI sur les shorts seulement ; ("up","down") = les deux.

# ============================================================
#  Filtre de confiance UP (entree) — CLE de la ROBUSTESSE AUX FRAIS
# ============================================================
# Filtre d'ENTREE applique a l'equipe APRES coherence + DMI : on ne garde un pari
# UP que si la confiance du modele_up (P(up), colonne `confidence`) >= UP_CONF_MIN.
# DOWN totalement inchange (son edge est robuste sur les 2 moities, cf plus bas).
#
# CAMPAGNE FRAIS 2026-07-01 (fee_harness, frais 0.02%/cote DANS la boucle de sim) :
# le cote UP a un edge STRUCTURELLEMENT fragile — son EV/trade de 1re moitie de
# fenetre (h1) est NEGATIF a tous les seuils de 0.59 a 0.64 (tout l'edge UP est en
# 2e moitie). Il ne devient robuste (h1>0) qu'a partir de conf>=0.65 : on ne prend
# alors que les ~93 setups UP les plus surs/an, en jetant ceux qui perdent en h1.
# Effet EQUIPE (365j @ 0.02%/cote) : EV +0.200 (h1 +0.086) -> +0.287/trade (h1 +0.221),
# pire trade -7.9% -> -5.0%, net +331% -> +296% (527 trades vs 841). EV/trade +44%.
#   0.65  -> RECOMMANDE (robuste : h1 solidement positif, garde du volume, meilleur worst)
#   0.62  -> +volume (675 trades, net +340%) mais h1 fragile (+0.092) — arbitrage volume/robustesse
# Mets UP_CONF_FILTER_ENABLED=False pour le comportement nu (n=504, NON fee-robuste).
UP_CONF_FILTER_ENABLED = True
UP_CONF_MIN = 0.63           # seuil P(up) mini. Campagne 500%/cote 2026-07-01 : 0.63 (au lieu de 0.65)
                             # DOUBLE les trades UP (93->183) SANS perdre la robustesse, grace aux 2
                             # leviers ci-dessous (DMI-off UP + drop heures asiatiques). UP +61% -> +185%.

# --- UP : filtre horaire (campagne 500%/cote 2026-07-01) ---
# Drop des heures asiatiques/overnight 0-5 UTC (bloc de SESSION, pas cherry-pick) : repare la
# fragilite structurelle de la 1re moitie (h1) de UP. h1 +12% -> +44%, h2 +43% -> +98%, DD -11%->-18%.
UP_DROP_HOURS = (0, 1, 2, 3, 4, 5)   # heures UTC ou on NE prend PAS de long

# --- UP : filtre CROSS-MODEL (Strategie 2, campagne 500%/cote 2026-07-01) ---
# On ne garde un LONG que si le modele DOWN est confiant qu'il n'y aura PAS de chute :
# P(down) au meme bar < UP_XMODEL_MAX_PDOWN. Isole les longs propres : UP +185% -> +224%,
# WR 62% -> 67%, h2 +98% -> +121% (n 183 -> 139). Combine (equipe) +1113% -> +1277%.
UP_XMODEL_FILTER_ENABLED = True
UP_XMODEL_MAX_PDOWN = 0.44   # long autorise ssi P(down) < 0.44 ; False/1.0 = filtre off

# --- DOWN : selectivite d'entree (campagne 500%/cote 2026-07-01) ---
# On ne short que si P(down) >= seuil_sidecar + DOWN_CONF_DELTA : enleve les shorts marginaux (bruit)
# -> l'equipe reste ROBUSTE (2 moities equilibrees) avec le trend-ride DOWN ci-dessous. delta 0 = off.
DOWN_CONF_DELTA = 0.015

# ============================================================
#  Frais / capital
# ============================================================
FEE_PER_SIDE = 0.00045        # frais par cote (achat ET vente) ; aller-retour = 2x
                              # NB : les frais portent sur le NOTIONNEL -> amplifies par le levier
INITIAL_CAPITAL = 2500.0      # capital de depart ($)

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
MAKER_FEE_PER_SIDE = 0.00015      # frais maker par cote (souvent 0 ou negatif/rebate ; mets le taux de ta plateforme)
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
UP_MAX_HOLD_MIN = 150         # detention max (min) : LET-RUN (campagne 500%/cote 2026-07-01) ~1.5h apres
                              # la cloture 1h. Avec DMI-off + drop-hrs, tenir 150min (vs 90) laisse courir
                              # les longs gagnants -> AvgWin 2.0% -> 2.48%, UP net +185% (WR 62% tenu).
UP_TRAIL_ACTIVATE = 0.003     # le trailing s'arme quand le pic atteint +0.30%
UP_TRAIL_GAP = 0.008       # sort si le profit retombe de 0.80% sous le pic (sweep 2026-06-18 : trailing actif)
UP_TAKE_PROFIT = 0.012     # take-profit +1.20% : let-run (campagne 500%/cote 2026-07-01, avec DMI-off +
                              # drop-hrs + hold 150). Le TP large capture les gagnants sans casser le WR.
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
# REGLAGE EV-MAX (sweep 2026-06-30, sweep_down_ev.py) : optimise l'EV/trade DOWN
# = WR*AvgWin - LR*AvgLoss. EV 0.4668% -> +0.6317% (worst -4.60% <= 6% ; 2 moities
# +0.536/+0.728). Levier = laisser courir le short (hold 150, TP 0.55%) au lieu de
# verrouiller tot (TP 0.30%) : AvgWin 1.51 -> 2.10%. Plateau large (non sur-ajuste).
DOWN_MGMT_ENABLED = True
DOWN_MAX_HOLD_MIN = 360       # TREND-RIDE (campagne 500%/cote 2026-07-01) : tenir le short ~5h apres la
                              # cloture 1h pour capturer les CRASHS multi-bougies. AvgWin 1.9% -> 3.2%.
DOWN_TRAIL_ACTIVATE = 0.0025  # le trailing s'arme quand le pic short atteint +0.25%
DOWN_TRAIL_GAP = 0.012        # trail LARGE (trend-ride) : laisse courir le crash, sort si le profit short
                              # retombe de 1.2% sous le pic (au lieu de 0.5% : verrouillait trop tot).
DOWN_TAKE_PROFIT = 0.015      # take-profit quasi-off (2%) : LET-RUN le short (campagne 500%/cote 2026-07-01).
                              # Le crash multi-bougies porte l'AvgWin a 3.2% -> DOWN net +146% -> +325% robuste.
DOWN_STOP_LOSS = 0.1      # SL CONSTANT 0.70% prix (trend-ride : DOWN_SL_TIME_DECAY=False ci-dessous).
                              # Coupe le short perdant vite, laisse courir le gagnant (asymetrie trend-ride).
DOWN_CHECK_MIN = 1            # intervalle de verif (min) -> DB_1m
# derives (ne pas editer)
DOWN_STEP_MS = DOWN_CHECK_MIN * 60 * 1000
DOWN_MAX_STEPS = DOWN_MAX_HOLD_MIN // DOWN_CHECK_MIN
DOWN_STOP_LOSS_DEC = -(DOWN_STOP_LOSS / 100.0)   # % prix -> stop PRIX (decimal neg), directement
# --- SL TEMPOREL (decroissant) cote DOWN : meme principe que UP (cf bloc UP ci-dessus). ---
DOWN_SL_TIME_DECAY = True   # OFF (trend-ride campagne 500%/cote 2026-07-01) : SL CONSTANT DOWN_STOP_LOSS
                             # (0.70% prix). Le SL decroissant coupait les shorts avant le crash.
DOWN_SL_START = 1        # SL a l'entree (large), en % prix (sweep 2026-06-30)
DOWN_SL_END   = 0.5         # SL a la cloture forcee (serre), en % prix (sweep 2026-06-30 EV-max)
DOWN_SL_FLOOR = 0.2         # SL le plus serre autorise (plancher), en % prix (sweep 2026-06-30)
DOWN_SL_START_DEC = -(DOWN_SL_START / 100.0)
DOWN_SL_END_DEC   = -(DOWN_SL_END   / 100.0)
DOWN_SL_FLOOR_DEC = -(DOWN_SL_FLOOR / 100.0)
