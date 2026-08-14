"""config_regime.py -- configuration centralisee du wrapper de detection de regime.

Tous les reglages editables du package `regime` vivent ici (meme convention que
modele_team/config_main.py). Le wrapper ne fait que DETECTER un regime (bear /
neutral / bull) sur les X derniers jours et ROUTER vers des overrides des knobs
d'entree DEJA existants dans modele_team/config_main.py -- il ne touche jamais
EXPOSURE / POSITION_FRACTION / LEVERAGE (contrainte actee, cf config_main.py).

Decisions actees (brainstorm 2026-07-05) :
  - modulation de 1h/modele_team (filtres + biais UP/DOWN), jamais le sizing.
  - indicateurs OHLCV uniquement (pas de funding/CVD/liquidations/long-short en v1).
  - un trade deja ouvert n'est jamais interrompu par un changement de regime.
  - les garde-fous de stabilite (DD/pire-mois) du pivot DOWN restent inchanges.
"""

# ============================================================
#  Interrupteur maitre
# ============================================================
# True depuis le ROUND 3 (cf REGIME_EXIT_OVERRIDES plus bas) : la gestion de sortie
# DOWN est regime-conditionnelle et validee (h1/h2 + plateau lookback). Le routage
# d'ENTREE (REGIME_OVERRIDES) reste noop -- round 1+2 n'ont rien trouve la-dessus.
REGIME_WRAPPER_ENABLED = True

# ============================================================
#  Fenetre(s) de detection -- "X derniers jours"
# ============================================================
LOOKBACK_DAYS_GRID = (14, 30, 60, 90)   # balaye par le harnais comparatif (compare_regime_variants.py)
LOOKBACK_DAYS_DEFAULT = 30              # utilise quand REGIME_WRAPPER_ENABLED=True (valeur unique, pas balayee)
MOMENTUM_FRACTION = 1.0 / 3.0           # fenetre momentum = LOOKBACK_DAYS * cette fraction (10j si X=30)

# ============================================================
#  Score composite -- poids des 3 axes (tendance / momentum / force DMI)
# ============================================================
WEIGHT_TREND = 1.0
WEIGHT_MOMENTUM = 1.0
WEIGHT_DMI = 1.0

NORM_WINDOW_DAYS = 365 * 2   # fenetre roulante (jours) pour normaliser chaque axe en z-score borne (adaptatif
                             # dans le temps plutot qu'un seuil de magnitude fixe qui perimerait sur 13 ans)
NORM_CLIP = 3.0              # z-score clippe a +/-NORM_CLIP puis divise par NORM_CLIP -> axe dans [-1, +1]

# DMI reutilise pour l'axe "force de tendance" -- instance SEPAREE du DMI par-trade
# de modele_team/config_main.py (DMI_TIMEFRAME/DMI_LENGTH) : memes formules Wilder,
# reglages propres au regime (peuvent diverger sans impacter le filtre par-trade).
REGIME_DMI_TIMEFRAME = "4h"
REGIME_DMI_LENGTH = 12

# ============================================================
#  Calibration des seuils : "manual" (fixe) vs "calibrated" (percentiles historiques)
#  -- 2 variantes comparees empiriquement par le harnais, cf decision utilisateur.
# ============================================================
CALIBRATION_METHOD = "calibrated"   # "manual" | "calibrated" -- utilise quand REGIME_WRAPPER_ENABLED=True

# Asymetrie voulue (des 2 cotes) : BEAR se declenche PLUS TOT (seuil moins extreme)
# ET persiste PLUS LONGTEMPS (bande hysteresis plus large) que BULL.
#   - manuel : seuils fixes sur le score compose [-1, +1]
MANUAL_BEAR_ENTER = -0.15    # score <= ceci (depuis neutre/bull) -> candidat bear
MANUAL_BEAR_EXIT = 0.00      # une fois bear confirme, score doit remonter AU-DESSUS de ceci pour sortir
MANUAL_BULL_ENTER = 0.30     # score >= ceci (depuis neutre/bear) -> candidat bull (barre haute = conservateur)
MANUAL_BULL_EXIT = 0.20      # une fois bull confirme, score doit redescendre SOUS ceci pour sortir

#   - calibre : percentiles de la distribution historique du score compose (meme asymetrie)
CALIB_BEAR_ENTER_PCTL = 30   # bear se declenche des le 30e percentile (facile)
CALIB_BEAR_EXIT_PCTL = 50    # ne sort du bear qu'au 50e percentile (bande large = maintien long)
CALIB_BULL_ENTER_PCTL = 75   # bull ne se declenche qu'au 75e percentile (dur, conservateur)
CALIB_BULL_EXIT_PCTL = 65    # bande plus etroite -> sort du bull plus facilement que le bear n'en sortirait

# ============================================================
#  Anti-whipsaw : confirmation + cooldown (en bougies 1h)
# ============================================================
CONFIRM_BARS = 24    # ~1 jour de persistance du signal brut avant qu'un nouveau regime soit confirme
COOLDOWN_BARS = 48   # ~2 jours apres un changement avant qu'un nouveau soit accepte, meme si confirme

# ============================================================
#  Routage regime -> knobs modele_team (JAMAIS EXPOSURE/POSITION_FRACTION/LEVERAGE)
# ============================================================
# None = garde la valeur actuelle de config_main.py inchangee (pas d'override).
#
# CONCLUSION CAMPAGNE 2026-07-05 (voir modele_team/compare_regime_variants.py) :
# ~77 combinaisons testees (profils de routage x methode de calibration x grille
# de lookback), validees sur les 2 moities du test window (comme le reste du
# repo, ex: validate_2_moities.py). AUCUNE ne bat la baseline de facon robuste :
#   - Loosen DOWN_CONF_DELTA (bear OU bull) reproduit exactement le probleme que
#     le delta=0.03 corrige (WR/DD s'effondrent -- ce delta EST deja la
#     robustesse-par-regime, cf sa note dans config_main.py) : DD jusqu'a -70%.
#   - Tighten UP_CONF_MIN en bear (0.59->0.65) coute plus de bons trades UP qu'il
#     n'en filtre de mauvais (net -45% a -55% vs baseline).
#   - Le seul point qui semblait gagner en agrege ("bull_up_loosen" calibrated
#     lb=30 : +1075% vs +1051%) ne tenait que sur 2 trades ajoutes en h2 (h1
#     strictement identique) -- pas un edge.
#   - Le seul point qui battait la baseline sur LES DEUX moities ("bull_down_loosen"
#     calibrated lb=90) est un PIC ISOLE : les lookbacks voisins (14/30/60) du
#     MEME profil donnent des drawdowns de -45% a -57% -- signature classique de
#     suroptimisation (pas un plateau), rejete sur ce 3e critere.
# Le detecteur de regime lui-meme (scorers/calibration/state_machine/api) est
# fonctionnel et teste (cf leur docstring) ; c'est le ROUTAGE sur ces 2 knobs
# precis qui ne bat pas une config deja tres regime-robuste. Garde "noop" tant
# qu'un nouveau signal (nouvelle feature, nouveau modele) n'a pas ete re-teste
# via compare_regime_variants.py.
#
# ROUND 2 (meme jour, sur demande explicite d'aller plus loin) -- axes supplementaires
# testes en filtre UP/DOWN additionnel (pas seulement en routage bear/neutral/bull) :
#   - dmi_norm seul (lb-independant, calcule sur REGIME_DMI_TIMEFRAME/LENGTH) : ecart
#     isole de 15pt WR sur UP, mais REDONDANT avec le filtre DMI deja applique dans
#     main.py (meme axe sous-jacent) -- teste comme filtre supplementaire sur le VRAI
#     pipeline (team_bets apres coherence+DMI+conf) : degrade net ET h1 ET h2
#     simultanement a tous les seuils testes (20e-50e percentile).
#   - funding rate (Binance BTCUSDT perp, seul historique assez profond -- OKX
#     n'expose que ~3 mois via l'endpoint public funding-rate-history, inutilisable) :
#     ecart isole modeste (6-9pt), AUCUNE config robuste sur 22 seuils testes (2 cotes
#     x 11 percentiles) ; net_ret moyen par exit_reason (TP/SL/trailing/timestop) est
#     PLAT par tercile de funding des 2 cotes -- ne predit meme pas la qualite de sortie.
#   - AND-gate dmi_norm ET funding (les 2 doivent s'accorder) : le point le plus proche
#     (DOWN, 25e percentile) bat net et h1 mais PAS h2 (+148.60% vs baseline +148.85%,
#     ecart de 0.25pt sur 11 trades retires seulement) -- bruit statistique, pas un edge.
# Conclusion round 1+2 : ~100+ configurations testees sur 6 familles d'hypotheses
# distinctes (regime macro, DMI seul, funding seul, AND-gate, heures DOWN, correlation
# de sortie), aucune ne survit a la validation h1/h2 -- EN MODULANT LES FILTRES
# D'ENTREE (down_conf_delta / up_conf_min). Voir ROUND 3 ci-dessous : moduler la
# GESTION DE SORTIE plutot que l'entree a fini par marcher.
_NOOP_SIDE = {"down_conf_delta": None, "up_conf_min": None}
REGIME_OVERRIDES = {
    "bear": dict(_NOOP_SIDE),
    "neutral": dict(_NOOP_SIDE),
    "bull": dict(_NOOP_SIDE),
}

# ============================================================
#  ROUND 3 (meme jour) -- ROUTAGE SUR LA GESTION DE SORTIE (PAS l'entree) : CA MARCHE
# ============================================================
# Hypothese : les 2 knobs d'entree (down_conf_delta/up_conf_min) sont deja optimises
# pour la robustesse CROSS-regime (c'est litteralement leur fonction documentee) --
# aucune marge pour un routage regime dessus (round 1+2). Mais la GESTION DE SORTIE
# (combien de temps laisser courir un DOWN gagnant) n'avait jamais ete testee comme
# axe regime-conditionnel. Elle ne viole pas la decision de brainstorm "un trade
# ouvert n'est jamais touche" : le choix de parametres se fait UNE FOIS a l'entree
# (selon le regime a CE moment-la), jamais revu en cours de route.
#
# Teste : DOWN_MAX_HOLD_MIN et DOWN_TRAIL_GAP multiplies UNIQUEMENT pour les DOWN
# entres en regime bear confirme (laisser courir plus loin un trend-ride confirme).
# CONTROLE CRITIQUE (indispensable) : appliquer le meme hold etendu a TOUS les DOWN
# (sans discrimination de regime) donne un resultat quasi neutre en net (+1100% vs
# +1051%) mais DEGRADE WR (58.9%->55.2%) ET drawdown (-23.4%->-34.0%) -- prouve que
# le regime fait le travail (il isole QUAND laisser courir aide vraiment), pas juste
# "tenir plus longtemps est toujours bon".
# Sweep multiplicateur de hold (x1.0 a x4.0) : optimum INTERNE net a x2.0-x2.5 (pas
# un effet de bord monotone -- signature d'un vrai optimum, pas d'un artefact) ; le
# pire mois se degrade nettement au-dela de x2.0 (-3.3k$ a x2.0 vs -8.0k$ a x2.5) ->
# x2.0 retenu (le plus prudent des 2 sur l'axe stabilite, prioritaire pour cet
# utilisateur). Robuste sur 6/8 combinaisons lookback x methode (seul lb=90 echoue,
# des 2 methodes) -- un vrai plateau, valide DIFFEREMMENT du round 1 (pas un pic isole).
# Meilleur point (calibrated, lb=30, hold x2.0, trail x1.3) vs baseline (365j) :
#   net  +1050.80% -> +1635.24%   (+55.6% relatif)
#   WR     58.92%  ->   58.25%    (quasi identique)
#   DD    -23.44%  ->  -17.59%    (MEILLEUR)
#   pire mois -4,008$ -> -3,313$  (MEILLEUR)
#   valide h1 (+362.46%->+511.01%) ET h2 (+148.85%->+184.00%)
#
# UP TESTE ET REJETE (meme jour, sur demande utilisateur) : meme idee (etendre le
# hold UP specifiquement en regime bull confirme) donne l'illusion de marcher SI on
# la compare a la mauvaise reference (baseline confondue, qui incluait deja l'ajout
# DOWN) -- piege methodologique classique. Isole proprement (UP seul, DOWN inchange,
# vs la VRAIE baseline sans rien) : UP seul = +1009.44% (PIRE que +1050.80% sans
# rien, h2 se degrade +148.85%->+135.02%) ; DOWN+UP combines = +1572.86% (PIRE que
# DOWN seul +1635.24%). Confirme ce que config_main.py documentait deja pour UP
# ("l'edge UP ne beneficie PAS d'un trend-ride long") -- vrai meme avec ciblage
# regime, pas seulement en extension globale. Ne pas ajouter de cle "up_*" ici.
REGIME_EXIT_OVERRIDES = {
    "bear": {"down_hold_mult": 2.0, "down_trail_gap_mult": 1.3},   # laisse courir le DOWN 2x plus longtemps
    "neutral": {"down_hold_mult": 1.0, "down_trail_gap_mult": 1.0},  # inchange (defaut config_main.py)
    "bull": {"down_hold_mult": 1.0, "down_trail_gap_mult": 1.0},     # inchange (defaut config_main.py)
}

# ============================================================
#  Harnais comparatif -- variantes testees empiriquement avant de choisir
# ============================================================
# "baseline" = wrapper desactive (comportement actuel de main.py, sans regime).
# Le classifieur ML (approche 2, evoquee en brainstorm) est garde en reserve --
# non teste tant que les variantes a regles n'ont pas montre un plafond reel.
COMPARISON_VARIANTS = ("baseline", "manual", "calibrated")

# ============================================================
#  Garde-fous de stabilite (pivot DOWN 2026-07-04, INCHANGES par ce projet)
# ============================================================
# Une variante ne "gagne" le comparatif que si net% > baseline SANS depasser ces
# plafonds -- actes explicitement : la stabilite prime, quel que soit le regime.
STABILITY_MAX_DRAWDOWN_PCT = -40.0     # plafond dur (deja respecte par la config TIER 2 actuelle, ~-23%)
STABILITY_WORST_MONTH_USDC = -46_000.0  # plafond dur pire mois (deja respecte, ~-4k$ actuellement)
STABILITY_MIN_NET_PCT = 750.0           # plancher rendement net (deja largement depasse, ~+1051%)
