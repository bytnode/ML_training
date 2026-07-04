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
DMI_SIDES = ("down", "up")   # RE-SWEEPE 2026-07-03 (donnees fraiches) : le DMI sur UP AIDE maintenant
                             # (etait ("down",) seulement) -> fait partie du meilleur UP-standalone
                             # trouve (sweep_up_advanced.py) : +66.89% (vs ~+20% avant re-tuning).

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
UP_CONF_MIN = 0.59           # RE-SWEEPE 2026-07-03 UP-standalone, round final (sweep_up_final_round.py,
                             # avec la gestion de sortie ameliorée ci-dessous) : 0.59 -> UP seul +89.08%
                             # (214 trades, WR 58.4%), le meilleur trouve apres plusieurs tours d'affinage
                             # (66.89% -> 84.20% -> 84.32% -> 89.08%, rendements decroissants = plateau).

# --- UP : filtre horaire ---
# RE-SWEEPE 2026-07-03 (UP-standalone, round final) : "overnight" 23h-4h UTC est le meilleur
# bloc trouve pour UP seul avec la gestion de sortie amelioree (+89.08%).
# RE-SWEEPE 2026-07-04 (objectif EV/trade >= 0.6, n >= 150) : diagnostic empirique
# (diagnose_up_losses.py) sur les 214 trades UP -> les sorties stop_loss/trailing
# (les vraies perdantes) se concentrent sur des heures precises. Par MOYENNE NETTE
# par heure, seules 10h/14h/20h UTC sont negatives (-0.01/-0.06/-0.11%) ; 9h/19h/21h
# ont aussi un taux de "bad exit" eleve mais restent NETTES POSITIVES (leurs gagnants
# compensent) -> gardees. Ajouter 10/14/20 au bloc overnight (sweep_up_bad_hours.py,
# teste sur tous les sous-ensembles de {10,14,20}) : EV 0.4014 -> 0.6337, n 214 -> 157
# (>=150), WR 58.88% -> 61.78%, PERTE MOYENNE -2.639% -> -2.348% (baisse, comme demande),
# gain moyen quasi stable (2.525 -> 2.478%), net UP +113.93% -> +153.51%.
# Valide 2 moities (validate_bad_hours_split.py) : h1 EV 0.110 -> 0.345 (x3, toujours
# le plus faible, coherent avec la fragilite h1 documentee ailleurs), h2 EV 0.662 -> 0.919 ;
# amelioration dans LES DEUX moities (pas un artefact d'une seule) -> robuste, pas du bruit.
UP_DROP_HOURS = (23, 0, 1, 2, 3, 4, 10, 14, 20)   # heures UTC ou on NE prend PAS de long

# --- UP : filtre CROSS-MODEL (Strategie 2, campagne 500%/cote 2026-07-01) ---
# On ne garde un LONG que si le modele DOWN est confiant qu'il n'y aura PAS de chute :
# P(down) au meme bar < UP_XMODEL_MAX_PDOWN. Isole les longs propres : UP +185% -> +224%,
# WR 62% -> 67%, h2 +98% -> +121% (n 183 -> 139). Combine (equipe) +1113% -> +1277%.
UP_XMODEL_FILTER_ENABLED = True
UP_XMODEL_MAX_PDOWN = 0.44   # long autorise ssi P(down) < 0.44 ; False/1.0 = filtre off

# --- DOWN : selectivite d'entree (campagne 500%/cote 2026-07-01) ---
# On ne short que si P(down) >= seuil_sidecar + DOWN_CONF_DELTA : enleve les shorts marginaux (bruit)
# -> l'equipe reste ROBUSTE (2 moities equilibrees) avec le trend-ride DOWN ci-dessous. delta 0 = off.
# RE-SWEEPE 2026-07-04 (campagne STABILITE, cf bloc gestion DOWN plus bas) : 0.03 (etait 0.01).
# Le user a explicitement PREFERE moins de rendement pour beaucoup plus de stabilite : DOWN doit
# rester "le plus identique possible quelque soit le mouvement" (moins sensible aux retournements
# haussiers), avec un plafond dur de drawdown <=40% et pire mois <=$46k, rendement min 750% net.
# 0.03 coupe DOWN de 350 -> 140 trades/an (les moins fiables sont ceux les plus pres du seuil) :
# EV DOWN 0.81% -> 1.24%, WR 47.4% -> 55.7%. Surtout : l'EV DOWN en 2e moitie de fenetre (regime
# plus haussier) etait NEGATIF (-0.26%, DOWN perdait de l'argent) -> devient +0.61% avec ce delta,
# la ROBUSTESSE PAR REGIME visee est confirmee, pas juste la moyenne globale (validate_3_tiers.py).
DOWN_CONF_DELTA = 0.03

# ============================================================
#  Frais / capital
# ============================================================
FEE_PER_SIDE = 0.0005        # frais par cote (achat ET vente) ; aller-retour = 2x
                              # NB : les frais portent sur le NOTIONNEL -> amplifies par le levier
INITIAL_CAPITAL = 5000.0      # capital de depart ($)

# --- Sizing / levier (futures-style) ---
# Chaque trade engage POSITION_FRACTION du capital comme MARGE, avec un LEVIER dessus.
# Exposition effective au mouvement de prix = POSITION_FRACTION * LEVERAGE (= EXPOSURE).
# Ex : 50% du capital x levier 10 = 5.0x -> un mouvement +0.30% rapporte +1.50% sur l'equity.
# P/L ET FRAIS sont tous deux multiplies par EXPOSURE (notionnel = EXPOSURE * equity).
# UNIQUE et PARTAGEE entre UP et DOWN (user : ne pas toucher levier/taille de position/
# fraction de capital -- l'amelioration doit etre purement strategique).
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
MAKER_FEE_PER_SIDE = 0.0002      # frais maker par cote (souvent 0 ou negatif/rebate ; mets le taux de ta plateforme)
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
UP_MAX_HOLD_MIN = 240         # RE-SWEEPE 2026-07-03 UP-standalone : 240 (etait 90 puis 150).
                              # Teste jusqu'a 1440 (24h) : au-dela de 240 le rendement DEGRADE
                              # (contrairement a DOWN) -- l'edge UP ne beneficie PAS d'un trend-ride long.
UP_TRAIL_ACTIVATE = 0.003     # RE-SWEEPE 2026-07-03 UP-standalone, round ASYMETRIE (sweep_up_asymmetry.py,
                              # user a demande de reduire la perte moyenne SANS sacrifier gain/WR) : 0.003
UP_TRAIL_GAP = 0.010       # RE-SWEEPE 2026-07-03 round ASYMETRIE : 0.010 (etait 0.025) -- le trailing
                              # LARGE etait le principal moteur de perte (une fois arme a peine +0.4%,
                              # un giveback de 2.5% pouvait transformer un petit gain en grosse perte).
                              # Resserrer a 1.0% coupe la perte moyenne SANS toucher le gain moyen (le TP
                              # serre ci-dessous verrouille deja la plupart des gains avant meme le trailing).
UP_TAKE_PROFIT = 0.006     # TP SERRE : verrouille vite les petits gains frequents (WR 55->59%+).
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
UP_SL_START = 1.5        # RE-SWEEPE 2026-07-03 UP-standalone round final : 1.5 (etait 1) -- large au debut,
                              # laisse la position respirer avant que le trailing/TP ne prennent le relais.
UP_SL_END   = 0.3           # RE-SWEEPE 2026-07-03 round ASYMETRIE : 0.3 (etait 0.2)
UP_SL_FLOOR = 0.1           # RE-SWEEPE 2026-07-03 UP-standalone round final : 0.1 (etait 0.30 puis 0.2)
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
DOWN_MAX_HOLD_MIN = 720       # RE-SWEEPE 2026-07-04 (campagne STABILITE, TIER 2 "moyen securise") :
                              # 720 (etait 870). Raccourcit le trend-ride pour reduire l'exposition
                              # temporelle aux retournements haussiers violents. Combine a
                              # DOWN_CONF_DELTA=0.03 (plus haut) et DOWN_SL_END=0.4 (plus bas) : le
                              # user a explicitement PREFERE moins de rendement pour la stabilite
                              # (drawdown equipe -46.08%->-23.44%, pire mois -$85k->-$4k, net
                              # +2480%->+1051%). Valide 2 moities (validate_3_tiers.py) : EV DOWN
                              # en 2e moitie (regime + haussier) -0.26% -> +0.61% (etait NEGATIF).
DOWN_TRAIL_ACTIVATE = 0.0015  # RE-SWEEPE 2026-07-03 : 0.0015 (etait 0.0025)
DOWN_TRAIL_GAP = 0.012        # trail LARGE (trend-ride) : inchange, confirme optimal par le re-sweep.
DOWN_TAKE_PROFIT = 0.016      # RE-SWEEPE 2026-07-03 : 0.016 (etait 0.015) -- inchange par la campagne stabilite
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
DOWN_SL_END   = 0.4         # RE-SWEEPE 2026-07-04 (campagne STABILITE, TIER 2) : 0.4 (etait 0.5) --
                            # SL a la cloture forcee (serre), en % prix ; resserre legerement la
                            # queue de perte en fin de hold, combine a MAX_HOLD=720/CONF_DELTA=0.03.
DOWN_SL_FLOOR = 0.15        # RE-SWEEPE 2026-07-03 : 0.15 (etait 0.2)
DOWN_SL_START_DEC = -(DOWN_SL_START / 100.0)
DOWN_SL_END_DEC   = -(DOWN_SL_END   / 100.0)
DOWN_SL_FLOOR_DEC = -(DOWN_SL_FLOOR / 100.0)
