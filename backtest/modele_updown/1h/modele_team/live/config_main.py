"""config_main.py — configuration centralisee du backtest trading (main.py).

Tous les parametres editables de main.py vivent ici ; main.py les importe.
(Les chemins de fichiers restent dans main.py car derives de __file__.)

COPIE INDEPENDANTE (2026-08-03, campagne autonomie complete live/) : ceci est
une copie FIGEE de modele_team/config_main.py, embarquee dans live/ pour que
le dossier live/ soit deployable SEUL (VPS, aucune dependance a modele_team/
ni a 1h/). Meme tradeoff deja accepte pour live/modele_up|down/regime/ : apres
un changement de seuil/regle dans le config_main.py DU BACKTEST, recopie ce
fichier ici A LA MAIN -- rien ne le fait automatiquement, et une copie perimee
echoue silencieusement (meme risque documente pour modele_up/modele_down/regime
dans live/CLAUDE.md).

CONVENTION D'UNITES (2026-08-11, AUCUN changement de comportement, alignee sur
le config_main.py du backtest) : tous les seuils de prix EDITABLES portent le
suffixe _PCT et sont en POURCENT du mouvement de PRIX, SANS levier (0.6 = 0.60%).
Le moteur (position.py, filters.py) consomme des fractions decimales DERIVEES
(UP_TAKE_PROFIT, *_DEC, FEE_PER_SIDE, ... -- lignes "derives, ne pas editer",
calculees ici via /100) : n'edite QUE les champs _PCT. Le levier n'existe qu'a
UN seul endroit : EXPOSURE (bloc Sizing) -- jamais dans un seuil.
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
FEE_PER_SIDE_PCT = 0.05      # frais taker par cote en % du notionnel (0.05 = 0.05%) ; aller-retour = 2x
FEE_PER_SIDE = FEE_PER_SIDE_PCT / 100.0   # (derive, ne pas editer)
                              # NB : les frais portent sur le NOTIONNEL -> amplifies par le levier
INITIAL_CAPITAL = 5000.0      # capital de depart ($)

# --- Plafond de concurrence (symetrie live_team, campagne symetrie 2026-07-09) ---
# Nombre max de positions ouvertes simultanement, TOUS SENS CONFONDUS (2x DOWN ou
# 2x UP autorise, pas seulement 1 UP + 1 DOWN). Miroir EXACT de
# live_team/config_team.py::MAX_CONCURRENT_POSITIONS + position.py (has_open_side
# retire, seul le compte total est verifie) -- decision explicite du user : garder
# le plafond live ET l'ajouter au backtest (pas l'inverse, qui aurait augmente
# l'exposition reelle simultanee -- hors scope, cf regle "jamais toucher au
# levier/position/capital"). simulate_trades() saute un signal (ne l'ouvre pas)
# si MAX_CONCURRENT_POSITIONS positions sont deja ouvertes a l'entree (traded_ts) ;
# le suivi est purement chronologique (entry/exit_ts de chaque trade DEJA accepte),
# jamais de lookahead.
MAX_CONCURRENT_POSITIONS = 2

# --- Plafond de concurrence PAR SENS (symetrie live/, campagne OKX 2026-07-10) ---
# Nombre max de positions ouvertes simultanement pour UN MEME sens (UP ou DOWN
# separement). Mirroir EXACT de la contrainte OKX mode HEDGE : une 2e entree sur
# le meme sens (instId, posSide) FUSIONNE avec la position deja ouverte (avgPx
# reblende) au lieu d'ouvrir une position independante -- live/position.py ne
# peut donc jamais suivre 2 positions du meme sens separement (cf live/CLAUDE.md).
# Avec MAX_CONCURRENT_POSITIONS_PER_SIDE=1 et MAX_CONCURRENT_POSITIONS=2, le
# plafond total devient mathematiquement redondant (1 UP + 1 DOWN au plus = 2 au
# plus) mais reste verifie en plus pour rester defensif si l'un des deux change.
MAX_CONCURRENT_POSITIONS_PER_SIDE = 1

# --- Elargissement sur profit (test 2026-07-10) ---
# Quand un signal du meme sens arrive alors qu'une position de ce sens est DEJA
# ouverte (donc bloque par MAX_CONCURRENT_POSITIONS_PER_SIDE, jamais ouvert comme
# position independante), on l'utilise comme CONFIRMATION de tendance : SI la
# position ouverte est ACTUELLEMENT en profit (pp > 0) et n'a jamais ete elargie,
# on elargit UNE SEULE FOIS (pas cumulatif) son take-profit / trailing-gap / hold
# max, pour la laisser absorber plus du mouvement au lieu de verrouiller le gain
# tot. Le SL (temporel ou constant) n'est PAS elargi -- reste sur l'horizon
# ORIGINAL (protection inchangee, seule la partie "laisser courir" est etendue).
# Trouvaille motivant ce test (audit symetrie backtest/live 2026-07-10) : les
# signaux bloques par le plafond par sens etaient en moyenne 2x plus rentables
# que la moyenne (WR 60.5% vs 58.5%, net/trade 0.45% vs 0.22%) -- ils arrivent
# structurellement pendant les tendances, exactement quand elargir a du sens.
WIDEN_ON_PROFIT_ENABLED = True

# SWEEP 2026-07-10 (grille UP 5x4x4=80 + DOWN 5x4x4=80, sur la fenetre actuelle) :
# UP et DOWN sont DECOMPOSABLES (le rendement compose total est un produit de
# facteurs disjoints par sens -- elargir UP ne change AUCUN trade DOWN et vice
# versa), donc balayes SEPAREMENT (l'autre sens neutralise a x1.0/x1.0/x1.0).
# ⚠️ ECHANTILLON MINUSCULE : sur la fenetre testee, seulement 7 positions UP et
# 3 positions DOWN sont effectivement elargies (nb de fois ou un signal du meme
# sens arrive alors qu'une position est deja ouverte ET en profit). Le repo exige
# n>=150 pour valider un seuil ailleurs (cf UP_CONF_MIN/UP_DROP_HOURS) -- ici on
# est TRES loin de ce seuil. Le plateau DOWN (net% continue de monter jusqu'a
# tp_mult~4-6/hold_mult~4-6 puis se tasse) est un vrai signal (les 3 trades
# sortent bien en "take_profit", pas un artefact d'extrapolation), mais reste
# la generalisation de SEULEMENT 3 trades -- NE PAS traiter ces valeurs comme
# definitives. Re-sweeper des que plus de positions se sont fait elargir.
# UP (sweep quasi plat : tp=1.0 et tp=3.0 quasi identiques, hold=2.0 domine
# clairement) -> garde le point de depart, deja proche de l'optimum du sweep.
UP_WIDEN_TP_MULT = 2.0          # UP_TAKE_PROFIT x2 une fois elargi
UP_WIDEN_TRAIL_GAP_MULT = 1.5   # UP_TRAIL_GAP x1.5
UP_WIDEN_MAX_HOLD_MULT = 2.0    # UP_MAX_HOLD_MIN x2

# DOWN (sweep en plateau reel autour de tp=4-6/hold=4-6 ; milieu du plateau
# retenu plutot que le bord teste (tp=6/hold=6, net 1273%) pour eviter de
# sur-ajuster sur les 3 trades qui composent tout l'echantillon) :
DOWN_WIDEN_TP_MULT = 4.0        # DOWN_TAKE_PROFIT x4 une fois elargi (etait 2.0)
DOWN_WIDEN_TRAIL_GAP_MULT = 2.0 # DOWN_TRAIL_GAP (deja effectif, cf regime) x2 (etait 1.5)
DOWN_WIDEN_MAX_HOLD_MULT = 4.0  # DOWN_MAX_HOLD_MIN (deja effectif, cf regime) x4 (etait 2.0)
# --- Sizing / levier (futures-style) ---
# Chaque trade engage POSITION_FRACTION du capital comme MARGE, avec un LEVIER dessus.
# Exposition effective au mouvement de prix = POSITION_FRACTION * LEVERAGE (= EXPOSURE).
# Ex : 50% du capital x levier 10 = 5.0x -> un mouvement +0.30% rapporte +1.50% sur l'equity.
# P/L ET FRAIS sont tous deux multiplies par EXPOSURE (notionnel = EXPOSURE * equity).
# UNIQUE et PARTAGEE entre UP et DOWN (user : ne pas toucher levier/taille de position/
# fraction de capital -- l'amelioration doit etre purement strategique).
# ⚠️ SEUL endroit du fichier ou le levier existe : sert au sizing du notionnel live
# (live.py) et au reporting. AUCUN seuil TP/SL/trail ci-dessous n'en depend --
# ils sont tous compares au mouvement de prix NU.
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
MAKER_FEE_PER_SIDE_PCT = 0.02    # frais maker par cote en % du notionnel (0.02 = 0.02% ; souvent 0 ou
                                  # negatif/rebate -- mets le taux de ta plateforme)
MAKER_FEE_PER_SIDE = MAKER_FEE_PER_SIDE_PCT / 100.0   # (derive, ne pas editer)
MAKER_FILL_WINDOW_MIN = 15    # duree (min) ou la limite reste postee avant annulation (= opportunite perdue)

# ============================================================
#  Constantes temps (ms)
# ============================================================
# (Le bloc legacy "Stop-loss intra-bougie" -- STOP_LOSS_ENABLED/PCT/SIDES/CHECK_MIN
# et ses derives -- a ete SUPPRIME 2026-08-11 : desactive depuis longtemps, remplace
# par les sorties gerees UP/DOWN ci-dessous.)
DAY_MS = 24 * 3600 * 1000
INTERVAL_1H_MS = 60 * 60 * 1000

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
# Reglages (change TP/trail/SL) — UNITE : POURCENT du mouvement de PRIX, SANS levier
# (0.6 = 0.60% du prix), la meme pour TP, trailing ET SL.
#   DEFAUT          max90 TP=0.80% trail 0.35%/0.50% SL=0.90% -> gain 0.31/WR 57/perte -0.36/net +14
#   net max         idem mais SL=5% (off)                     -> gain 0.31/WR 57/perte -0.36/net +17
#   perte serree    max75 TP=0.30% trail 0.20%/0.30% SL=0.60% -> gain 0.24/WR 59/perte -0.31/net +9
UP_MGMT_ENABLED = True
UP_MAX_HOLD_MIN = 240         # RE-SWEEPE 2026-07-03 UP-standalone : 240 (etait 90 puis 150).
                              # Teste jusqu'a 1440 (24h) : au-dela de 240 le rendement DEGRADE
                              # (contrairement a DOWN) -- l'edge UP ne beneficie PAS d'un trend-ride long.
UP_TRAIL_ACTIVATE_PCT = 0.3   # trailing arme des +0.30% de profit prix. RE-SWEEPE 2026-07-03
                              # UP-standalone, round ASYMETRIE (sweep_up_asymmetry.py, user a
                              # demande de reduire la perte moyenne SANS sacrifier gain/WR)
UP_TRAIL_GAP_PCT = 1.0        # giveback de 1.00% une fois arme. RE-SWEEPE 2026-07-03 round ASYMETRIE
                              # (etait 2.5%) -- le trailing LARGE etait le principal moteur de perte
                              # (une fois arme a peine +0.4%, un giveback de 2.5% pouvait transformer
                              # un petit gain en grosse perte). Resserrer a 1.0% coupe la perte moyenne
                              # SANS toucher le gain moyen (le TP serre ci-dessous verrouille deja la
                              # plupart des gains avant meme le trailing).
UP_TAKE_PROFIT_PCT = 0.6      # TP SERRE a +0.60% : verrouille vite les petits gains frequents (WR 55->59%+).
UP_CHECK_MIN = 1              # intervalle de verif (min) -> DB_1m
# derives (ne pas editer) -- fractions decimales consommees par le moteur
UP_STEP_MS = UP_CHECK_MIN * 60 * 1000
UP_MAX_STEPS = UP_MAX_HOLD_MIN // UP_CHECK_MIN
UP_TRAIL_ACTIVATE = UP_TRAIL_ACTIVATE_PCT / 100.0
UP_TRAIL_GAP = UP_TRAIL_GAP_PCT / 100.0
UP_TAKE_PROFIT = UP_TAKE_PROFIT_PCT / 100.0
# --- SL TEMPOREL (decroissant, TOUJOURS actif -- sweep 2026-06-18) : le SL se resserre
#     lineairement de START (large, a l'entree) vers END (serre, a max_hold), borne par
#     FLOOR (jamais plus serre). Tout en POURCENT du prix, sans levier. (Le SL constant
#     UP_STOP_LOSS_PCT, son toggle UP_SL_TIME_DECAY et UP_SKIP_BUCKETS -- tous inactifs --
#     ont ete SUPPRIMES 2026-08-11.) ---
UP_SL_START_PCT = 1.5    # SL a l'entree : -1.50% prix (large). RE-SWEEPE 2026-07-03 UP-standalone
                              # round final (etait 1.0) -- laisse la position respirer avant que
                              # trailing/TP prennent le relais.
UP_SL_END_PCT   = 0.3       # SL a max_hold : -0.30% prix. RE-SWEEPE 2026-07-03 round ASYMETRIE (etait 0.2)
UP_SL_FLOOR_PCT = 0.1       # jamais plus serre que -0.10% prix. RE-SWEEPE 2026-07-03 UP-standalone
                              # round final (etait 0.3 puis 0.2).
                              # ⚠️ le config_main.py du BACKTEST a lui 0.7 (=0.70% prix) : divergence REELLE.
# derives (ne pas editer)
UP_SL_START_DEC = -(UP_SL_START_PCT / 100.0)
UP_SL_END_DEC   = -(UP_SL_END_PCT   / 100.0)
UP_SL_FLOOR_DEC = -(UP_SL_FLOOR_PCT / 100.0)

# ============================================================
#  Gestion gain/perte cote DOWN (SHORT) — calque exact de UP
# ============================================================
# Meme gestion de SORTIE pure (AUCUN filtre d'entree -> 754 trades DOWN intacts),
# appliquee au SHORT. Profit directionnel short a la minute j = 1 - close/entry
# (symetrique du long close/entry - 1). DOWN sortait brut a la cloture 1h :
# gain 0.271 / perte -0.318 / net +12.44 / WR 56.76.
# Sweeps (sweep_down_loss*.py : 1800+1400 configs, valides sur 2 moities) -> 3 reglages.
# Change les 5 lignes ci-dessous pour basculer — UNITE : POURCENT du mouvement de
# PRIX, sans levier (0.5 = 0.50% du prix), la meme que TP/trail.
#   DEFAUT (WR+net) max105 SL=0.50% TP=0.30% trail 0.25%/0.50%   (sweep 2026-06-18 maker ;
#                    l'ancienne note "2.5cap" = 2.5% du CAPITAL etait ce meme 0.50% prix x EXPOSURE 5 --
#                    notation capital abandonnee, on n'ecrit plus que le % prix nu)
#                    -> DOWN WR 62.2% / gain 0.276 / perte -0.352 / DOWN net +232% / TEAM net +772%
#                    (le WR DOWN se pilote par le TP, pas le SL ; SL large 0.80% = WR 57 / TEAM +866)
#   3-AXES+          max80  SL=0.70% TP=0.50% trail 0.25%/0.45%
#                    -> gain 0.299 / perte -0.310 / net +24.9 / WR 55.8  (ameliore perte ET gain ET net vs base)
#   PERTE MIN        max60  SL=0.70% TP=0.80% trail 0.25%/0.45%
#                    -> gain 0.261 / perte -0.273 / net +21.0 / WR 56.1  (perte la plus basse)
# REGLAGE EV-MAX (sweep 2026-06-30, sweep_down_ev.py) : optimise l'EV/trade DOWN
# = WR*AvgWin - LR*AvgLoss. EV 0.4668% -> +0.6317% (worst -4.60% <= 6% ; 2 moities
# +0.536/+0.728). Levier = laisser courir le short (hold 150, TP 0.55%) au lieu de
# verrouiller tot (TP 0.30%) : AvgWin 1.51 -> 2.10%. Plateau large (non sur-ajuste).
DOWN_MGMT_ENABLED = True
DOWN_MAX_HOLD_MIN = 720       # RE-SWEEPE 2026-07-04 (campagne STABILITE, TIER 2 "moyen securise") :
                              # 720 (etait 870). Raccourcit le trend-ride pour reduire l'exposition
                              # temporelle aux retournements haussiers violents. Combine a
                              # DOWN_CONF_DELTA=0.03 (plus haut) et DOWN_SL_END=0.004 (plus bas) : le
                              # user a explicitement PREFERE moins de rendement pour la stabilite
                              # (drawdown equipe -46.08%->-23.44%, pire mois -$85k->-$4k, net
                              # +2480%->+1051%). Valide 2 moities (validate_3_tiers.py) : EV DOWN
                              # en 2e moitie (regime + haussier) -0.26% -> +0.61% (etait NEGATIF).
DOWN_TRAIL_ACTIVATE_PCT = 0.0 # trailing arme des l'entree (0%). NB : valeur en place 0.0 ; le
                              # commentaire re-sweep 2026-07-03 disait 0.15% (etait 0.25%) -- la valeur fait foi.
DOWN_TRAIL_GAP_PCT = 1.2      # giveback 1.20% : trail LARGE (trend-ride), confirme optimal par le re-sweep.
DOWN_TAKE_PROFIT_PCT = 1.6    # TP a +1.60%. RE-SWEEPE 2026-07-03 (etait 1.5%) -- inchange par la campagne stabilite
DOWN_CHECK_MIN = 1            # intervalle de verif (min) -> DB_1m
# derives (ne pas editer) -- fractions decimales consommees par le moteur
DOWN_STEP_MS = DOWN_CHECK_MIN * 60 * 1000
DOWN_MAX_STEPS = DOWN_MAX_HOLD_MIN // DOWN_CHECK_MIN
DOWN_TRAIL_ACTIVATE = DOWN_TRAIL_ACTIVATE_PCT / 100.0
DOWN_TRAIL_GAP = DOWN_TRAIL_GAP_PCT / 100.0
DOWN_TAKE_PROFIT = DOWN_TAKE_PROFIT_PCT / 100.0
# --- SL TEMPOREL (decroissant, TOUJOURS actif) cote DOWN : meme principe que UP.
#     (Le SL constant DOWN_STOP_LOSS_PCT et son toggle DOWN_SL_TIME_DECAY -- inactifs --
#     ont ete SUPPRIMES 2026-08-11.) ---
DOWN_SL_START_PCT = 1.0  # SL a l'entree : -1.00% prix (large) (sweep 2026-06-30)
DOWN_SL_END_PCT   = 0.4     # SL a la cloture forcee : -0.40% prix. RE-SWEEPE 2026-07-04 (campagne
                            # STABILITE, TIER 2 ; etait 0.5) -- resserre la queue de perte en fin
                            # de hold, combine a MAX_HOLD=720/CONF_DELTA=0.03.
DOWN_SL_FLOOR_PCT = 0.15    # jamais plus serre que -0.15% prix. RE-SWEEPE 2026-07-03 (etait 0.2)
# derives (ne pas editer)
DOWN_SL_START_DEC = -(DOWN_SL_START_PCT / 100.0)
DOWN_SL_END_DEC   = -(DOWN_SL_END_PCT   / 100.0)
DOWN_SL_FLOOR_DEC = -(DOWN_SL_FLOOR_PCT / 100.0)
