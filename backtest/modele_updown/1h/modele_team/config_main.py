"""config_main.py — configuration centralisee du backtest trading (main.py).

Tous les parametres editables de main.py vivent ici ; main.py les importe.
(Les chemins de fichiers restent dans main.py car derives de __file__.)

DEPUIS 2026-08-03 (campagne autonomie complete live/), `live/config_main.py`
est une COPIE INDEPENDANTE de ce fichier (plus un import partage) -- si tu
changes un seuil ici, recopie-le manuellement dans live/config_main.py pour
que le live suive (cf live/CLAUDE.md).

CONVENTION D'UNITES (2026-08-11, AUCUN changement de comportement) : tous les
seuils de prix EDITABLES portent le suffixe _PCT et sont en POURCENT du mouvement
de PRIX, SANS levier : 0.6 = 0.60%, 1.5 = 1.50%, 0.05 = 0.05%. Ce qu'on lit est
ce que ca vaut en %. Le moteur, lui, consomme des fractions decimales DERIVEES
(UP_TAKE_PROFIT, *_DEC, FEE_PER_SIDE, ... -- lignes "derives, ne pas editer",
calculees ici via /100) : n'edite QUE les champs _PCT. Le levier n'existe qu'a
UN seul endroit : EXPOSURE (bloc Sizing), applique une seule fois au
reporting/equity dans trading_stats() -- il n'entre JAMAIS dans un seuil.
"""

# ============================================================
#  Fenetre de backtest
# ============================================================
TEAM_TEST_DAYS = 720          # taille de la fenetre de test (jours), ancree sur la
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
                              # NB : les frais portent sur le NOTIONNEL -> amplifies par le levier
FEE_PER_SIDE = FEE_PER_SIDE_PCT / 100.0   # (derive, ne pas editer)
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
# ALIGNEMENT PROD (decision user 2026-08-14, remplace le "retour commit" du
# matin) : la prod deployee (Desktop/modele_team, etat mi-juillet) utilise le
# widen DOWN x4/x2/x4 -- reactive pour que le backtest DOWN = prod DOWN.
DOWN_WIDEN_ENABLED = True
# --- Sizing / levier (futures-style) ---
# Chaque trade engage POSITION_FRACTION du capital comme MARGE, avec un LEVIER dessus.
# Exposition effective au mouvement de prix = POSITION_FRACTION * LEVERAGE (= EXPOSURE).
# Ex : 50% du capital x levier 10 = 5.0x -> un mouvement +0.30% rapporte +1.50% sur l'equity.
# P/L ET FRAIS sont tous deux multiplies par EXPOSURE (notionnel = EXPOSURE * equity).
# UNIQUE et PARTAGEE entre UP et DOWN (user : ne pas toucher levier/taille de position/
# fraction de capital -- l'amelioration doit etre purement strategique).
# ⚠️ SEUL endroit du fichier ou le levier existe : EXPOSURE n'est applique QU'A la
# toute fin, dans trading_stats() (equity / rapport / logs). AUCUN seuil TP/SL/trail
# ci-dessous n'en depend -- ils sont tous compares au mouvement de prix NU.
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
#  Cliquet de gain (trailing a paliers) — campagne SECURISATION 2026-08-11
# ============================================================
# PROBLEME STRUCTUREL corrige ici : le trailing classique pose son plancher a
# `pic - gap`. Il ne verrouille donc un gain POSITIF que si le pic a deja
# depasse le gap. Deux zones mortes, vraies par construction :
#   - UP   : gap 1.00% > TP 0.60% -> le TP sort TOUJOURS avant que le trailing
#            puisse verrouiller quoi que ce soit. AUCUN gain intermediaire n'est
#            jamais securise : un trade monte a +0.55% peut finir a -0.45%.
#   - DOWN : verrouillage positif seulement au-dela d'un pic de 1.20%, soit 75%
#            du chemin vers le TP parcouru sans aucune securisation.
# Cas reel (live OKX) : short, pic +1.13% (+63.86 USDT), plancher a -0.12%,
# sortie -0.16% -- 70% du chemin vers le TP capture, rien garde.
#
# CORRECTIF : plancher effectif = max(pic - gap, palier(pic)). Des que le pic
# franchit un seuil, le plancher ne redescend plus sous le niveau verrouille.
# N'entre en jeu QUE sur les trades ayant deja prouve un mouvement ; le gap
# reste large sous le premier palier -> le trend-ride DOWN est intact.
# Le SL temporel et son horizon sont STRICTEMENT inchanges (jamais elargis,
# jamais resserres par le cliquet).
#
# Format : suite de (pic_pct, plancher_pct) en POURCENT du mouvement de prix,
# sans levier -- meme unite que TP / trailing / SL. () = cliquet desactive de ce
# cote (comportement d'avant la campagne, bit-identique).
#
# RESULTAT DE LA CAMPAGNE (sweeps optimize/sweep_ratchet*.py, ~25 000 configs) :
#   1. Le cliquet SEUL, a TP/gap constants, est MONOTONEMENT DESTRUCTEUR : plus il
#      mord, plus le net baisse (DOWN +502% -> +4% a arm=0.2% ; UP +125% -> +27% a
#      arm=0.10%). Les trades CLIPPES coutent bien plus que les SAUVES. Le trou de
#      verrouillage est donc bien "le prix du trend-ride" -- A TP/GAP CONSTANTS.
#   2. Cote DOWN : 3150 configs balayees (TP 1.0-3.5%, gap 0.6-9.0%, cliquets plats
#      et proportionnels) -- AUCUNE ne bat la baseline, meme sur le net total seul.
#      L'optimum (TP 1.60 / gap 1.20 / pas de cliquet) est INTERIEUR a la grille.
#      -> DOWN reste STRICTEMENT INCHANGE. Ne pas y remettre de cliquet.
#   3. Cote UP : le cliquet devient PAYANT une fois le TP releve. A TP 0.60% le TP
#      sortait avant tout verrouillage possible ; a TP 1.55% les trades respirent et
#      le cliquet capte la retombee. Apport PROPRE du cliquet, TP/gap identiques :
#      UP +160.75% -> +181.99% (net total +1469.76% -> +1626.48%, soit +157 pts).
#      Comptabilite sauves/clippes : le cliquet ne sauve pas PLUS de trades (25 vs 26)
#      -- il REDUIT LE COUT DES CLIPPES (-74.93 -> -66.52 pts). C'est la sa valeur.
RATCHET_ENABLED = True
# CAMPAGNE SECURITE 2026-08-11 (3e goal du jour, optimize/sweep_security*.py) :
# le user exige qu'une position montee a ~1% de profit prix ne puisse plus JAMAIS
# finir perdante (cas vecu : pic +1.13% -> perte). Paliers PLATS reactives comme
# GARANTIE DE SECURITE (pas comme optimisation de rendement -- en tant que telle,
# la campagne cliquet du matin reste valable : ils COUTENT du net) :
#   UP   (0.80 -> 0.05)  : quasi gratuit (+2 pts), complete le proportionnel
#   DOWN (1.10 -> 0.25)  : LA garantie -- pic >= 1.10% => plancher >= +0.25%
#     (0.25 = verrou minimal NET-POSITIF : l'overshoot de cloture 1-min sous le
#      declencheur laisse ~0.10-0.20 brut, encore > 0.07 de frais A/R).
# COUT ASSUME (decision user : "la securite est la grande priorite, le PL net
# n'est pas un probleme") : net total 2179.62% -> 1717.44% -- l'essentiel vient
# de 4 trades qui percaient 0.25% apres un pic >=1.1 puis repartaient (+1.06 a
# +4.33) ; AUCUN signal ne les separe des 8 vraies pertes (teste : vol_ratio,
# volume adverse, DMI, age du bear, vitesse -- chevauchement total sur 13 cas).
# FRONTIERE mesuree (logs/sweep_security_micro.csv) : arm 1.15 -> 1769% mais ne
# couvre pas le cas 1.13 du user ; arm 1.05 -> 1640 ; lock 0.30 -> 1679.
# RISQUE RESIDUEL documente : une bougie 1-min de ~1% peut traverser toute la
# bande (1 cas / 131 trades UP : pic 1.02 -> cloture +0.06 brut = -0.05 net).
# Zero violation DOWN >=1.1 sur la fenetre ; DOWN pic 1.0-1.1 reste non garanti
# (1 cas a -1.56), l'etendre a 1.0 couterait ~100 pts de plus.
UP_RATCHET_STEPS_PCT = ((0.80, 0.05),)
# RETOUR DOWN AU COMMIT ecb04f8 (decision user 2026-08-14) : le verrou de
# securite DOWN (1.10 -> 0.25) est SUPPRIME -- lecture litterale "le down
# exactement comme le commit" confirmee, y compris la perte de la garantie
# "pic >= 1.10% ne finit jamais negatif". Etait ((1.10, 0.25),).
DOWN_RATCHET_STEPS_PCT = ()

# PALIER PROPORTIONNEL (arm_pct, frac) ou None. Generalise les paliers plats :
# des que le pic atteint arm_pct, le plancher ne redescend plus sous frac x pic
# -> on garantit de CONSERVER UNE FRACTION du meilleur gain atteint. C'est la
# forme qui correspond au vrai defaut du trailing classique : `pic - gap`
# autorise un giveback CONSTANT (1.20% cote DOWN, 1.56% en regime bear) quelle
# que soit la taille du gain, donc un pic de 1.5% peut rendre 100% de son gain.
# frac=0.0 -> breakeven garanti au-dela de arm_pct.
# SWEEP FIN 2026-08-11 (5670 configs autour de l'optimum, 2150 valides) : le point
# retenu est le CENTRE d'un plateau large, pas un pic. Voisins tous valides et
# proches -- arm 0.85/0.90/1.10 -> net moyen 156-158% (vs 161.6% a 1.00), frac
# 0.25/0.35 -> 156.0/157.0% (vs 158.6% a 0.30). Une erreur d'un cran coute ~3 pts.
UP_RATCHET_PROP_PCT = (1.00, 0.30)   # pic >= 1.00% -> plancher >= 30% du pic
DOWN_RATCHET_PROP_PCT = None         # DOWN : cliquet DESACTIVE (prouve destructeur)

# Interaction avec l'elargissement sur profit (widen-on-profit) : quand une
# position est elargie (TP/gap/hold multiplies), les paliers suivent-ils ?
#   False -> paliers ABSOLUS, inchanges par l'elargissement (le gain deja
#            verrouille reste verrouille au meme niveau, la position elargie
#            court plus loin mais avec le meme filet).
#   True  -> paliers multiplies par le meme facteur que le TP (*_WIDEN_TP_MULT).
RATCHET_SCALES_WITH_WIDEN = False

# derives (ne pas editer) -- decimales consommees par le moteur
def _build_ratchet(steps_pct):
    return tuple(sorted(((p / 100.0, f / 100.0) for p, f in steps_pct),
                        key=lambda s: s[0]))
UP_RATCHET_STEPS = _build_ratchet(UP_RATCHET_STEPS_PCT) if RATCHET_ENABLED else ()
DOWN_RATCHET_STEPS = _build_ratchet(DOWN_RATCHET_STEPS_PCT) if RATCHET_ENABLED else ()

def _build_prop(prop_pct):
    return None if not prop_pct else (prop_pct[0] / 100.0, prop_pct[1])
UP_RATCHET_PROP = _build_prop(UP_RATCHET_PROP_PCT) if RATCHET_ENABLED else None
DOWN_RATCHET_PROP = _build_prop(DOWN_RATCHET_PROP_PCT) if RATCHET_ENABLED else None

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
UP_MAX_HOLD_MIN = 240         # Valeur en place : 360. (Le re-sweep 2026-07-03 UP-standalone avait
                              # retenu 240 -- etait 90 puis 150 ; la copie live/ est restee a 240,
                              # divergence backtest/live REELLE.)
                              # Teste jusqu'a 1440 (24h) : au-dela de 240 le rendement DEGRADE
                              # (contrairement a DOWN) -- l'edge UP ne beneficie PAS d'un trend-ride long.
UP_TRAIL_ACTIVATE_PCT = 0.30   # RE-SWEEPE 2026-08-11 (campagne SECURISATION, conjoint avec le
                              # TP et le cliquet ; etait 0.25). Parametre PEU SENSIBLE : sur les
                              # 2150 configs valides du sweep fin, 0.25/0.30/0.35 donnent des nets
                              # moyens de 155.3/156.5/156.8% -- le choix ne joue quasiment pas.
UP_TRAIL_GAP_PCT = 1        # giveback de 1.00% une fois arme. RE-SWEEPE 2026-07-03 round ASYMETRIE
                              # (etait 2.5%) -- le trailing LARGE etait le principal moteur de perte
                              # (une fois arme a peine +0.4%, un giveback de 2.5% pouvait transformer
                              # un petit gain en grosse perte). Resserrer a 1.0% coupe la perte moyenne
                              # SANS toucher le gain moyen (le TP serre ci-dessous verrouille deja la
                              # plupart des gains avant meme le trailing).
UP_TAKE_PROFIT_PCT = 1.55     # RE-SWEEPE 2026-08-11 (campagne SECURISATION, etait 0.60 = "TP serre").
                              # CHANGEMENT DE PHILOSOPHIE cote UP : le TP serre verrouillait vite
                              # (WR 63.9%) mais coupait tous les mouvements ; couple au cliquet, laisser
                              # respirer paie beaucoup plus. WR 63.9% -> 59.5% mais gain moyen bien plus
                              # eleve : UP net +124.89% -> +181.99%, EV 0.6546 -> 0.8651.
                              # PLATEAU LARGE (sweep fin, net moyen des configs valides) : 1.50 -> 153.5%,
                              # 1.55 -> 162.7%, 1.60 -> 161.7%, 1.65 -> 160.8%, 1.70 -> 161.3% ; ca ne
                              # s'effondre qu'a 1.80 (150.5%). 1.55 = centre du plateau, pas un pic.
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
UP_SL_START_PCT = 1.5    # SL a l'entree : -1.50% prix (large). RE-SWEEPE 2026-07-03 UP-standalone round
                              # final (etait 1.0) -- laisse la position respirer avant que trailing/TP
                              # prennent le relais.
UP_SL_END_PCT   = 0.3       # SL a max_hold : -0.30% prix. RE-SWEEPE 2026-07-03 round ASYMETRIE (etait 0.2)
UP_SL_FLOOR_PCT = 0.7       # jamais plus serre que -0.70% prix. NB : un ancien commentaire disait "0.1"
                              # mais la VALEUR en place etait 0.7 -- la valeur fait foi, conservee.
                              # ⚠️ live/config_main.py a lui 0.1 (=0.10% prix) : divergence backtest/live REELLE.
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
# ⚠️ CAMPAGNE ADAPTATIVITE 2026-08-11 (optimize/RAPPORT_adaptativite_down_2026-08-11.md) :
# le gap FIXE 1.2% a ete confronte a ~270 variantes adaptatives (resserrage
# conditionne aux signaux 1-min vol/volume, stall-exit, gap normalise ATR entree
# et intra-trade, resserrage conditionne au regime neutral/bull) -- AUCUNE ne bat
# la baseline sur les 2 moities. L'echelle de retracement des trend-rides 1h est
# ~constante (~1.2%), PAS proportionnelle a la vol courte ; le profit restitue est
# la prime d'option de la queue des TP. NE PAS re-tenter un resserrage DOWN sans
# un signal d'une autre nature (carnet, funding, liquidations).
# ============================================================
#  Extension conditionnelle du TP DOWN en bear (campagne ADAPTATIVITE 2026-08-11)
# ============================================================
# LE resultat positif de la campagne adaptativite (apres l'echec de tout
# RESSERRAGE, cf bloc precedent) : le gisement etait de l'autre cote. Diagnostic
# (optimize/diag_down_posttp.csv) : sur les 39 sorties take_profit a +1.60%, le
# pic post-TP moyen est +2.98% ; en regime BEAR (28/39) il est +3.27% (p75
# +4.41%). Le systeme encaissait 1.6 et laissait la queue.
#
# MECANISME (implemente dans main._down_managed_exit) : quand un short atteint
# le TP (1.60%) avec (1) regime bear a la barre de decision et (2) un chemin
# PEU efficient depuis l'entree (ER = pp/path <= DOWN_TPEXT_MAX_ER -- l'autopsie
# montre que les GRINDERS continuent, les spikes rapides whipsawent), on ne sort
# PAS : on laisse courir avec DEUX sorties adaptatives :
#   - cassure de STRUCTURE : sortie des qu'un close 1-min repasse au-dessus du
#     dernier pivot haut confirme (DOWN_TPEXT_SWING_P barres de chaque cote)
#     forme PENDANT l'extension -- le "gap" suit la structure du marche : large
#     dans une jambe propre, immediat sur cassure ;
#   - plancher dur : jamais sous TP - DOWN_TPEXT_HARD_GUARD_PCT (gain verrouille).
# Le trailing classique (pic - gap) reste actif en parallele (max des planchers).
# Widen-on-profit prime : s'il se declenche, l'extension s'efface (TP x4 du widen).
#
# RESULTAT (fenetre 365j, config UP retenue) : DOWN +502.01% -> +694.89%
# (EV 1.786 -> 2.07, WR/worst/dd INCHANGES), total +1626.48% -> +2179.62%.
# PLATEAU verifie : p 12-25 et hard_guard 0.6-1.2 plats, ER 0.065-0.090 plat.
# ⚠️ LIMITE ASSUMEE (decision user 2026-08-11) : gain concentre dans la moitie
# bear de la fenetre (h1 +277->+423) ; la moitie haussiere h2 rend 8 pts
# (+59.7->+51.9) via 3 extensions perdantes en bear superficiel -- AUCUN gate
# principiel ne les separe (DMI, age du bear, vitesse testes). Le mecanisme est
# regime-dependant par construction : il paie quand il y a des bears a rider.
# Echantillon : 14 extensions sur la fenetre -- re-sweeper quand il y en a plus.
# RETOUR DOWN AU COMMIT ecb04f8 (decision user 2026-08-14) : extension COUPEE
# (le commit n'avait aucune extension de TP). Etait True.
DOWN_TPEXT_ENABLED = False
DOWN_TPEXT_REGIMES = ("bear",)     # regimes (barre de decision) ou l'extension s'arme
DOWN_TPEXT_MAX_ER = 0.075          # gate : ER du chemin entree->TP <= 0.075 (grinders only)
DOWN_TPEXT_SWING_P = 20            # pivot haut confirme : p barres de chaque cote (1-min)
DOWN_TPEXT_HARD_GUARD_PCT = 0.8    # plancher dur : jamais sous TP - 0.80% (= +0.80% prix)
# derives (ne pas editer)
DOWN_TPEXT_HARD_GUARD_DEC = DOWN_TPEXT_HARD_GUARD_PCT / 100.0

DOWN_MGMT_ENABLED = True
DOWN_MAX_HOLD_MIN = 720       # RE-SWEEPE 2026-07-04 (campagne STABILITE, TIER 2 "moyen securise") :
                              # 720 (etait 870). Raccourcit le trend-ride pour reduire l'exposition
                              # temporelle aux retournements haussiers violents. Combine a
                              # DOWN_CONF_DELTA=0.03 (plus haut) et DOWN_SL_END=0.004 (plus bas) : le
                              # user a explicitement PREFERE moins de rendement pour la stabilite
                              # (drawdown equipe -46.08%->-23.44%, pire mois -$85k->-$4k, net
                              # +2480%->+1051%). Valide 2 moities (validate_3_tiers.py) : EV DOWN
                              # en 2e moitie (regime + haussier) -0.26% -> +0.61% (etait NEGATIF).
DOWN_TRAIL_ACTIVATE_PCT = 0.15 # RETOUR DOWN AU COMMIT ecb04f8 (2026-08-14) : le commit armait le
                              # trailing a +0.15% (DOWN_TRAIL_ACTIVATE = 0.0015). Etait 0.0 ici.
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

# ============================================================
#  MODE DEFENSIF CHOP — couche trendiness (campagne TOUT-TERRAIN 2026-08-11)
# ============================================================
# PROBLEME corrige : le systeme etait un pur suiveur de tendance (TP larges
# 1.55/1.60 + trend-ride), calibre sur l'annee 2 (2025-07 -> 2026-07, bear
# tendanciel). Sur l'annee 1 (2024-07 -> 2025-07, bull/chop, corrections en V),
# il perdait -46.87% net alors que les modeles predisent toujours bien la
# bougie 1h (label_acc 58-65%) : l'annee 1 etait POSITIVE EN BRUT (+57.7 pts),
# ce sont les 80 sorties stop_loss (-404 pts) et les frais (89.6 pts) qui la
# tuaient — signature classique du trend-following en marche mean-reverting.
# Diagnostic complet : optimize/diag_year1.py + RAPPORT_tout_terrain_2026-08-11.md.
#
# MECANISME (implemente dans main.simulate_trades + les 2 sorties gerees, maps
# construits par filters.load_trend_context, AUCUN lookahead — closes <= barre
# de decision uniquement) : DEUX indicateurs de structure longue :
#   - ER  = Efficiency Ratio de Kaufman sur CHOP_ER_DAYS jours (pas quotidiens,
#     rafraichi chaque barre 1h) : |close_t - close_t-n| / somme des |pas 1j|.
#     ER bas = chemin inefficient = chop.
#   - age_bear = heures consecutives passees sous la SMA CHOP_SMA_DAYS jours.
#     Separe les corrections en V (annee 1 : age max 27j) des bears installes
#     (annee 2 : 60% des barres > 30j, jusqu'a 242j) — fosse structurel.
# MODE SCALP arme quand (ER < CHOP_ER_MAX) ET (age_bear <= CHOP_BEAR_AGE_MAX_DAYS) :
# les SORTIES deviennent mean-reversion (TP courts, gap serre, hold reduit,
# tpext coupe, override regime bear ignore). AUCUN trade supprime (l'entree ne
# change pas), AUCUN changement du SL temporel sauf horizon (= hold scalp).
# Hors chop : comportement STRICTEMENT inchange. En prime, TP UP serre aussi en
# macro-bull (close > SMA200) hors chop — ameliore LES DEUX annees (le TP 1.55
# etait une calibration annee-2/bear ; en bull les longs 1h ne courent pas si loin).
#
# RESULTAT (fenetre 720j, optimize/exp_er*.py, logs/exp_er*.csv) :
#   baseline  : annee 1 -46.87% / annee 2 +1652.97% / total  +831.35%
#   avec mode : annee 1 +47.46% / annee 2 +1032.03% / total +1569.24%
#   TOUTES les fenetres 365j glissantes (pas 30j) sont POSITIVES.
# PLATEAUX verifies : ER_MAX 0.20-0.30 et age 15-60j tous positifs annee 1 ;
# frontiere d'age dans un fosse (27j vs 100-242j). Arbitrage assume (goal
# user) : l'annee 2 rend ~620 pts pour rendre l'annee 1 positive.
# GATES TESTES ET REJETES : direction macro seule (SMA/momentum — les
# corrections 'mbear' de l'annee 1 chopent comme les vrais bears trendent),
# skip DOWN en macro-bull (tue les shorts gagnants de l'annee 2), skip DOWN
# en bull profond >= +10% SMA200 (retire plus de gagnants que de perdants).
CHOP_SCALP_ENABLED = True
# RETOUR DOWN AU COMMIT ecb04f8 (decision user 2026-08-14) : le mode chop ne
# s'applique plus qu'a l'UP. False = TOUTES les adaptations DOWN au contexte
# sont coupees : sorties scalp DOWN (CHOP_DOWN_TP/HOLD/TRAIL_*), bump d'entree
# CHOP_DOWN_CONF_DELTA_BUMP, et palier de cliquet chop/mbull
# (CHOP_DOWN_RATCHET_*). Les knobs CHOP_DOWN_* ci-dessous restent documentes
# mais INACTIFS tant que ce flag est False. L'UP garde tout (TP/SL chop, bump
# de confiance, mbull).
CHOP_DOWN_ENABLED = False
CHOP_ER_DAYS = 30             # lookback de l'ER (jours, pas quotidiens)
CHOP_ER_MAX = 0.25            # ER < 0.25 = chemin inefficient (chop)
CHOP_SMA_DAYS = 200           # SMA de reference du regime macro (jours)
CHOP_BEAR_AGE_MAX_DAYS = 30   # au-dela : bear installe -> on trend-ride, jamais scalp
CHOP_UP_TP_PCT = 0.60         # TP UP en mode scalp (retour au "TP serre" pre-2026-08-11)
CHOP_DOWN_TP_PCT = 1.10       # TP DOWN en mode scalp (vs 1.60 trend). RE-SWEEPE campagne
                              # DOWN-ANNEE-1 2026-08-11 (etait 0.90, choisi sous l'ancien
                              # trailing act 0.0) : avec act 0.4 + le bump de delta ci-dessous,
                              # 1.10 domine (a2 +1470 vs +1386 seul ; croise avec le bump :
                              # net +2656 vs +2519). 0.8 degrade, 1.0 intermediaire.
CHOP_DOWN_HOLD_MIN = 600      # hold max DOWN en mode scalp (vs 720 trend x2 bear)
# --- CAMPAGNE DOWN-ANNEE-1 (2026-08-11, 3e goal du jour, optimize/exp_dnfix.py) ---
# Diagnostic : le poste de perte residuel de l'annee 1 etait 34 stop-loss DOWN
# en chop (-153 pts) — des shorts MARGINAUX (P(down) juste au seuil) pris en
# marche indecis. Correctif d'ENTREE (jamais teste avant) : en chop, le seuil
# DOWN devient sidecar + DOWN_CONF_DELTA + CE BUMP — un short en chop doit etre
# plus sur qu'un short en tendance. Applique dans filters.effective_down_delta
# quand trend_ctx est fourni (main.run_backtest le passe ; le harnais optimize/
# ne le passe pas -> sweeps inchanges). RE-TESTE ET REJETE au passage : tout
# resserrage du SL DOWN chop, meme sous le nouveau trailing (6 variantes, toutes
# pires — le SL DOWN 1.0/0.4/0.15 est definitif).
# BANDE VIABLE mesuree (grille fine) : 0.0075-0.015 ; 0.005 et 0.02 s'effondrent
# des DEUX cotes (falaises) -> 0.01 = centre de bande, et la paire (0.01, 0.0125)
# x TP 1.10 est un plateau (net +2656 / +2717, a1 +60.0 / +59.9).
# RESULTAT (avec TP 1.10) : a1 +53.49 -> +60.02% (DOWN a1 +17.77 -> +29.98%),
# a2 +1386.21 -> +1622.41%, total +2181.15 -> +2656.17%, dd -27.21 -> -28.67%.
CHOP_DOWN_CONF_DELTA_BUMP = 0.01

# --- CAMPAGNE UP-PERTES (2026-08-11, 5e goal du jour, optimize/exp_upwr*.py) ---
# Constat user : cote UP annee 1, une perte vaut ~1.6x un gain (AvgLoss 3.90 vs
# AvgWin 2.46). Anatomie : 21 stop-loss UP en chop = -98.5 pts (avg -4.69) face
# a des TP a +2.89 — ce sont les longs MARGINAUX (P(up) juste au-dessus de 0.59).
# TESTES ET REJETES avant d'arriver la : (a) palier de cliquet UP hors bear
# installe — IMPOSSIBLE structurellement : le TP chop 0.60 sort avant tout
# armement <0.6, et armer plus bas ampute les gains (meme lecon que la campagne
# cliquet du matin : "le cliquet ne paie qu'une fois le TP releve") ; (b) relever
# le TP chop (0.7-1.0) — en chop les mouvements ne s'etendent pas, tout releve
# degrade ; (c) SL serre etendu au macro-bull — 3 trades concernes, net negatif.
# CORRECTIF retenu (miroir exact du bump DOWN) : en chop, une entree UP exige
# P(up) >= UP_CONF_MIN + CE BUMP (0.59 -> 0.625). Applique dans
# filters.effective_up_conf_min quand trend_ctx est fourni (harnais : non passe
# -> sweeps inchanges). BANDE VIABLE : 0.03-0.04 ; 0.025 flechit, 0.05
# S'EFFONDRE (falaise, comme le bump DOWN) -> 0.035 = centre de bande.
# RESULTAT (720j) : AvgLoss UP a1 3.90 -> 3.35 (ratio perte/gain 1.6 -> 1.3),
# UP a1 +27.22 -> +43.74%, a1 +94.59 -> +112.57%, total +2789.58 -> +2946.67%,
# dd -23.70 -> -24.23% (quasi stable), a2 UP +186.97 -> +176.89% (cout leger).
CHOP_UP_CONF_MIN_BUMP = 0.035
# ⚠️ CAMPAGNE PAYOFF-UP 2026-08-12 (optimize/exp_uppay.py, resultat NEGATIF) :
# egaliser AvgLoss/AvgWin UP en chop (ratio 1.31 -> ~1.0) a ete teste sur 3
# familles (SL re-serre, hold court, trailing serre) — le trailing serre y
# arrive (ratio 0.92) mais fait tomber le WR de 65 a 51% et l'EV a ~0 (UP a1
# +43.74 -> +8.98%, dd -24 -> -44). Le ratio 1.31 est le PRIX de l'edge chop
# (WR haut x TP serre vs SL respirant) : la cible est l'EV, pas le ratio.
# NE PAS re-tenter sans mecanisme d'une autre nature (cf RAPPORT).

# --- CAMPAGNE WR-DOWN (2026-08-11, 4e goal du jour, optimize/exp_dnwr.py) ---
# Demande user : remonter le WR DOWN de l'annee 1 (42.7%) sans sacrifier les
# gros gains. Palier de cliquet DOWN supplementaire actif HORS BEAR INSTALLE
# (barres en chop scalpable OU macro-bull ; en macro-bear age>30j : rien — les
# trend-rides de l'annee 2 gardent tout leur espace) : un short monte a
# +0.50% ne peut plus finir sous +0.10% (= verrou NET-POSITIF minimal : 0.10
# brut - 0.07 frais A/R > 0 ; un lock a 0.05 ferait PLUS de net total mais ne
# remonterait PAS le WR, la cloture serait nette-negative). S'ajoute au palier
# de securite (1.10 -> 0.25), qui reste actif partout.
# ZONE VIABLE (grille fine exp_dnwr2.csv) : arm 0.45-0.55 x lock 0.05-0.10 ;
# lock 0.15 = falaise (a1 s'effondre), arm 0.40 degrade a2 et le DD.
# RESULTAT (720j) : WR DOWN a1 42.70 -> 48.89% (AvgWin 5.37 -> 4.37, les rides
# restent), DOWN a1 +29.98 -> +52.95%, a1 +60.02 -> +94.59%, dd -28.67 ->
# -23.70% (meilleur de la journee), total +2656.17 -> +2789.58%. COUT ASSUME :
# a2 +1622.41 -> +1384.96% (ses phases chop/mbull sont clippees ; equilibre
# inter-annees AMELIORE, ce qui est la mission).
CHOP_DOWN_RATCHET_ARM_PCT = 0.50    # pic >= 0.50% (prix) -> verrou
CHOP_DOWN_RATCHET_LOCK_PCT = 0.10   # plancher >= +0.10% (net-positif apres frais)
# derives (ne pas editer)
CHOP_DOWN_RATCHET_STEP = (CHOP_DOWN_RATCHET_ARM_PCT / 100.0,
                          CHOP_DOWN_RATCHET_LOCK_PCT / 100.0)
MBULL_UP_TP_ENABLED = True    # TP UP serre AUSSI en macro-bull hors chop
MBULL_UP_TP_PCT = 0.60        # (ameliore a1 ET a2 : +1653 -> +1711 sur a2 seule)
# --- CAMPAGNE POINTS-FAIBLES (2026-08-11, meme jour, optimize/exp_upsl.py /
# exp_dntrail.py / exp_combo.py) : 3 corrections chirurgicales en mode chop.
# Diagnostic : les 14 pires trades de l'annee 1 etaient TOUS des stop_loss UP
# (SL 1.5% pour un TP scalp de 0.6% = risque 2.5x le gain vise), et le trailing
# DOWN arme des l'entree (act 0.0) whipsawait 81 sorties = -134 pts.
#   - SL UP resserre en chop : START 1.5 -> 1.2, FLOOR 0.7 -> 0.5 (plus serre
#     degrade : 1.0/0.5 et 0.8/0.4 perdent — le long doit encore respirer).
#   - trailing DOWN en chop : arme a +0.40% de profit prouve (etait 0.0) et gap
#     0.70 (etait 0.60). Optimum INTERIEUR sur les 2 axes (act 0.2-0.5 x gap
#     0.7-0.9 balayes ; gap 0.8/0.9 degradent l'annee 1, act 0.5 aussi).
# Combo verifie (720j) : a1 +47.46 -> +63.19%, a2 +1032.03 -> +1386.21%,
# total +1569.24 -> +2325.33%, dd -57.90 -> -42.02%. (Le "gouverneur d'equity"
# qui accompagnait cette campagne a ete SUPPRIME du code le 2026-08-14 sur
# decision user — historique dans optimize/RAPPORT_tout_terrain_2026-08-11.md.)
CHOP_DOWN_TRAIL_GAP_PCT = 0.70    # gap trailing DOWN en mode scalp (vs 1.20 trend)
CHOP_DOWN_TRAIL_ACTIVATE_PCT = 0.40  # trailing scalp arme a +0.40% (trend : 0.0)
CHOP_UP_SL_START_PCT = 1.2        # SL temporel UP a l'entree en chop (trend : 1.5)
CHOP_UP_SL_FLOOR_PCT = 0.5        # plancher du SL UP en chop (trend : 0.7) ; END inchange
# derives (ne pas editer)
CHOP_UP_TP = CHOP_UP_TP_PCT / 100.0
CHOP_DOWN_TP = CHOP_DOWN_TP_PCT / 100.0
CHOP_DOWN_MAX_STEPS = CHOP_DOWN_HOLD_MIN // DOWN_CHECK_MIN
CHOP_DOWN_TRAIL_GAP = CHOP_DOWN_TRAIL_GAP_PCT / 100.0
CHOP_DOWN_TRAIL_ACTIVATE = CHOP_DOWN_TRAIL_ACTIVATE_PCT / 100.0
CHOP_UP_SL_START_DEC = -(CHOP_UP_SL_START_PCT / 100.0)
CHOP_UP_SL_FLOOR_DEC = -(CHOP_UP_SL_FLOOR_PCT / 100.0)
MBULL_UP_TP = MBULL_UP_TP_PCT / 100.0
CHOP_BEAR_AGE_MAX_H = CHOP_BEAR_AGE_MAX_DAYS * 24

# NB : le GOUVERNEUR D'EQUITY (livre-fantome, campagne POINTS-FAIBLES
# 2026-08-11) a ete entierement SUPPRIME du code le 2026-08-14 (decision user,
# parite avec la prod qui n'en a jamais eu). Historique et chiffres dans
# optimize/RAPPORT_tout_terrain_2026-08-11.md ; sa cause premiere de
# declenchement etait le saignement DOWN annee 1, et il penalisait l'UP.