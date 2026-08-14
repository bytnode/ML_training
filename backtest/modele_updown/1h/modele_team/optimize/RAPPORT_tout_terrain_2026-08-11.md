# RAPPORT — Campagne TOUT-TERRAIN (2026-08-11)

**Goal** : rendre le système rentable sur tout type de marché. Cible immédiate :
année 1 (2024-07-13 → 2025-07-13) positive ; sacrifice de l'année 2 autorisé.
Cible finale : positif sur n'importe quelle période.

**Résultat** : année 1 **−46.87% → +47.46%**, année 2 +1652.97% → **+1032.03%**,
total 720j +831.35% → **+1569.24%**. **Toutes les fenêtres 365j glissantes
positives** (min +19.23%, départ 2024-10-12).

## Phase 1 — Diagnostic (`diag_year1.py`, `logs/diag_year1_trades.csv`)

1. **L'année 1 était positive EN BRUT** (+57.7 pts) — frais 89.6 pts → net −31.9 pts.
   Edge brut/trade 0.23 pt vs 0.35 pt de frais ; année 2 : 1.69 pt/trade.
2. **Les stop-loss dominaient la perte** : 80 sorties SL = −404 pts (DOWN 56×
   −265.75, UP 24× −138.13). Taux de SL DOWN 40% (année 1) vs 32% (année 2) ;
   taux de TP UP 6% vs 12%. Signature du trend-following en marché mean-reverting.
3. **Le régime local 30j s'INVERSE entre les années** : année 1 bear-local DOWN
   WR 38.5% (−43 pts) / bull-local DOWN +49 pts ; année 2 exactement l'opposé
   (bear DOWN +171 pts, WR 61.3%). Le détecteur 30j ne voit pas la structure macro.
4. **Les modèles prédisent toujours bien** (label_acc 1h année 1 : UP 65.0%,
   DOWN 57.6%) — le problème était la couche de gestion, confirmant le diagnostic user.

## Phase 2 — Chemin des tests (`exp_macro.py` → `exp_er*.py`, `logs/exp_*.csv`)

- **Direction macro (SMA120/200, momentum 90/120j)** : sépare mal — les
  corrections « mbear » de l'année 1 chopent (les deux sens y perdent) tandis que
  le mbear de l'année 2 trende. Gate DOWN-en-mbull : tue l'année 2 (ses phases
  mbull contiennent des shorts gagnants, lag de la SMA). REJETÉ comme gate seul.
  Trouvaille annexe conservée : **TP UP 0.60 en macro-bull** améliore LES DEUX
  années (a2 +1653 → +1711 seule).
- **Trendiness (ER de Kaufman, pas quotidiens)** : la bonne variable. Scalper les
  *sorties* en chop (ER30<0.25) SANS supprimer d'entrée : a1 +15.31 (première
  config année-1-positive). Coût a2 (chop-labellisé mais trendant, lag ER).
- **Séparateur année 1 / année 2 en chop : l'ÂGE DU BEAR** (heures sous SMA200).
  Année 1 : max 27.2j, 0% des barres >30j. Année 2 : 60% des barres >30j (jusqu'à
  242j). Frontière à 30j = fossé structurel, pas un fil du rasoir. Condition
  finale : `ER30 < 0.25 ET âge ≤ 30j`.
- **Réglage scalp** : le hold veut rester LONG (600 min ; 240 → a1 +15, 600 →
  a1 +39) — c'est le TP court + gap serré qui paie, pas la sortie précoce.
  TP DOWN 0.90 > 0.80 ; gap 0.50 boosterait a1 (+57) mais coûte ~190 pts à a2 —
  0.60 retenu. upTP 0.50 ≈ 0.60 → 0.60 retenu (= MBULL_UP_TP, un seul niveau).
- **Skip DOWN en bull profond (close ≥ +6..20% SMA200)** : REJETÉ — retire plus
  de gagnants que de perdants (a1 retombe à +2..+25, a2 aussi touchée).

## Config retenue (implémentée : `config_main.py` bloc CHOP_*/MBULL_*, `filters.load_trend_context`, `main.simulate_trades`)

| knob | valeur |
|---|---|
| CHOP_ER_DAYS / CHOP_ER_MAX | 30j / 0.25 |
| CHOP_SMA_DAYS / CHOP_BEAR_AGE_MAX_DAYS | 200j / 30j |
| CHOP_UP_TP_PCT / CHOP_DOWN_TP_PCT | 0.60 / 0.90 |
| CHOP_DOWN_HOLD_MIN / CHOP_DOWN_TRAIL_GAP_PCT | 600 / 0.60 |
| MBULL_UP_TP_PCT (hors chop, macro-bull) | 0.60 |

En chop : tpext coupé, override régime bear coupé, SL temporel intact (horizon =
hold scalp), entrées/filtres/concurrence/cliquets inchangés. Hors chop : rien ne
change sauf le TP UP en macro-bull.

## Vérifications

- Run `python main.py` (720j) reproduit EXACTEMENT le harnais : a1 +47.46 /
  a2 +1032.03 / total +1569.24 / 516 trades / dd −57.90.
- Plateaux : ER 0.20/0.25/0.30 × âge 15-60j → a1 +2.5..+27 tous positifs sauf
  coin ER0.20 (a1 +2.5..+6.2) ; le point retenu n'est pas un pic isolé.
- Fenêtres glissantes (run final) : 365j toutes positives (min +19.23%) ;
  180j min −44.41% (déc-2024→juin-2025) ; 90j min −34.27%.

## Campagne POINTS-FAIBLES (même jour, 2e goal : "améliore drastiquement")

Diagnostic sur le champion (+47.46/+1032.03, dd −57.90) : **(a)** tout le DD est
UN segment (2024-11-27 → 2025-06-17) ; **(b)** les 14 pires trades de l'année 1
sont TOUS des stop_loss UP (−138 pts ; SL 1.5% pour un TP scalp 0.6% = risque
2.5× le gain) ; **(c)** trailing DOWN armé dès l'entrée en chop → 81 sorties
whipsaw = −134 pts. Trois sweeps parallèles (`exp_upsl.py`, `exp_dntrail.py`,
`exp_governor.py`) puis grille croisée (`exp_combo.py`) et gouverneur exact
(`exp_governor2.py`) :

1. **SL UP chop 1.2/0.5** (start/floor ; END 0.3 inchangé). Grille 1.1-1.3 ×
   0.5-0.7 : optimum à 1.2/0.5, plus serré (1.0/0.5, 0.8/0.4) DÉGRADE.
2. **Trailing DOWN chop : activate 0.40 / gap 0.70** (était 0.0/0.60). Grille
   act 0.2-0.5 × gap 0.7-0.9 : optima intérieurs des deux axes.
   SL DOWN resserré en chop : testé, rejeté (toutes variantes pires).
3. **Gouverneur d'equity (livre-fantôme)** : K=40 résultats clos, skip < −15 pts,
   resume ≥ −15. ⚠️ la v1 (`exp_governor.py`) comptait des trades encore ouverts
   (micro-lookahead, a1 +94.87 illusoire) ; la version exit-aware
   (`exp_governor2.py`) fait foi. 35 trades sautés, année 2 jamais touchée.

Combo sans gouverneur : a1 +63.19 / a2 +1386.21 / total +2325.33 / dd −42.02.
Avec gouverneur (retenu, stabilité ≻ net — priorité constante du user) :

| | avant campagne | après |
|---|---|---|
| année 1 | +47.46% | **+53.49%** (UP +30.33 / DOWN +17.77) |
| année 2 | +1032.03% | **+1386.21%** |
| total | +1569.24% | **+2181.15%** |
| max DD | −57.90% | **−27.21%** |
| pire fenêtre 365j | +19.23% | **+53.49%** |
| pire fenêtre 180j | −44.41% | **−16.35%** |
| pire fenêtre 90j | −34.27% | **−14.59%** |

Trimestres : seuls 2024Q3 (−0.79%) et 2025Q2 (−24.70%) négatifs. Implémenté :
`CHOP_UP_SL_*`/`CHOP_DOWN_TRAIL_ACTIVATE_PCT`/gap 0.70 + bloc `GOVERNOR_*` dans
`config_main.py`, `filters.apply_governor`, params `sl_override`/`trail_activate`
des sorties gérées, colonne `exit_ts` dans trades.csv. Run `python main.py`
reproduit exactement les chiffres du harnais.

## Campagne DOWN-ANNEE-1 (même jour, 3e goal : "fais mieux côté down")

Constat user : année 1 UP +30.33% vs DOWN +17.77%, alors que DOWN est le côté
structurellement fort. Anatomie (`exp_dnfix.py`) : le poste de perte résiduel
était **34 stop-loss DOWN en chop = −153 pts** (le trailing chop, réparé par la
campagne précédente, ne coûtait plus que −12 pts). Trois axes testés :

1. **SL DOWN chop re-serré** (6 variantes, re-testées sous le NOUVEAU trailing
   act 0.4 — le rejet précédent datait de l'ancien trailing) : toutes pires.
   Le SL DOWN 1.0/0.4/0.15 est définitif.
2. **TP scalp re-swept** : 1.10 > 0.90 (choisi sous l'ancien trailing) — surtout
   via l'année 2 (+1470 vs +1386 seul).
3. **Sélectivité d'entrée DOWN en chop** (`CHOP_DOWN_CONF_DELTA_BUMP`, premier
   correctif d'ENTRÉE de la journée) : les SL chop étaient des shorts marginaux ;
   +0.01 sur le delta en chop les élimine à la source. Bande viable 0.0075-0.015,
   falaises des deux côtés (0.005 → a1 +17, 0.02 → a2 +973) ; 0.01 = centre,
   paire plateau (0.01, 0.0125)×TP1.10 (net +2656/+2717).

Retenu : bump 0.01 + TP 1.10. État final vérifié (`python main.py`, 413 trades) :

| | début de journée | final |
|---|---|---|
| année 1 | −46.87% | **+60.02%** (UP +23.11 / DOWN +29.98) |
| année 2 | +1652.97% | **+1622.41%** (DOWN +500.20) |
| total | +831.35% | **+2656.17%** |
| max DD | −69.94% | **−28.67%** |
| EV/trade | +0.5882% | **+0.9246%** |
| pire fenêtre 365j | — | **+60.02%** (12/12 positives) |
| pire fenêtre 180j / 90j | — | −17.22% / −20.75% |

Seul trimestre négatif restant : 2025Q2 (−27.71%).

## Campagne WR-DOWN (même jour, 4e goal : "gros gains + bon WR")

Demande user : WR DOWN année 1 à 42.7% = le frein principal. Levier retenu
(`exp_dnwr.py` + grille fine `exp_dnwr2.csv`) : **palier de cliquet DOWN
(0.50 → 0.10) actif hors bear installé** (chop OU macro-bull). Un short monté
à +0.50% ne peut plus finir sous +0.10% (verrou net-positif minimal après les
0.07 de frais — lock 0.05 ferait plus de net mais pas de WR). Le bear installé
(âge > 30j) n'est jamais touché : les trend-rides de l'année 2 gardent leur
espace. Zone viable arm 0.45-0.55 × lock 0.05-0.10 ; lock 0.15 = falaise.

État final vérifié (`python main.py`, 418 trades, 30 sautés par le gouverneur) :

| | début de journée | final |
|---|---|---|
| année 1 | −46.87% | **+94.59%** (WR net 57.89% ; UP +27.22 / DOWN +52.95) |
| WR DOWN année 1 | 42.45% | **48.89%** (AvgWin 4.37 — rides conservés) |
| année 2 | +1652.97% | **+1384.96%** (DOWN +417.46) |
| total | +831.35% | **+2789.58%** (139 479 $ sur 5 000) |
| max DD | −69.94% | **−23.70%** |
| pire fenêtre 365j | — | **+71.61%** (12/12 positives) |
| pire fenêtre 180j / 90j | — | **−6.30%** / −10.41% |
| trimestres négatifs | 4 | **1** (2025Q2 −17.79%) |

## Campagne UP-PERTES (même jour, 5e goal : "une perte UP vaut ~2x un gain")

Anatomie (`exp_upwr.py`) : 21 stop-loss UP en chop = −98.5 pts (avg −4.69) face
à des TP à +2.89 — les longs marginaux. Trois familles testées :
palier de cliquet UP → **impossible structurellement** (le TP chop 0.60 sort
avant tout armement <0.6 ; armer plus bas détruit — arm 0.3 : a1 +10.6) ;
TP chop relevé 0.7-1.0 → tout dégrade (le chop ne s'étend pas) ; SL mbull →
négatif. **Retenu** : sélectivité d'entrée UP en chop (`CHOP_UP_CONF_MIN_BUMP`
= 0.035, soit P(up) ≥ 0.625 en chop), miroir du bump DOWN. Bande viable
0.03-0.04, falaise à 0.05.

État final (`python main.py`, 387 trades, 14 sautés par le gouverneur) :

| | début de journée | final |
|---|---|---|
| année 1 | −46.87% | **+112.57%** (UP +43.74 / DOWN +47.89) |
| ratio perte/gain UP a1 | 1.60 (3.90/2.46) | **1.31** (3.35/2.57) |
| année 2 | +1652.97% | **+1333.25%** |
| total | +831.35% | **+2946.67%** (147 334 $ sur 5 000) |
| EV/trade | +0.5882% | **+1.0046%** |
| max DD | −69.94% | **−24.23%** |
| pire fenêtre 365j | — | **+66.63%** (12/12 positives) |
| trimestres négatifs | 4 | **1** (2025Q2 −19.52%) |

## Campagne PAYOFF-UP (2026-08-12) — RÉSULTAT NÉGATIF documenté

Demande user : AvgLoss UP ≈ AvgWin (année 1 : 3.35 vs 2.57, ratio 1.31).
19 configs, 3 familles, toutes chop-only sous le stack final (`exp_uppay.py`) :
SL re-serré 0.7-1.0 × 0.3-0.5 (le meilleur, 1.0/0.5 : ratio 1.26 mais tout
baisse — a1 +107, net 2590) ; hold court 120-180 (AGGRAVE le ratio, 1.75 à
120 min — coupe les gains avant les pertes) ; trailing serré act 0.2-0.3 ×
gap 0.3-0.4 (SEULE famille qui égalise : ratio 0.92 à trail 0.3/0.4 — mais
WR 65→51%, UP a1 +43.74 → +8.98%, net 2947 → 1877, dd −24 → −44).

**Conclusion structurelle : le ratio 1.31 est le prix de l'edge chop** (WR 65%
× TP serré 0.6 contre SL respirant ~1.2). L'égaliser fait mécaniquement tomber
le WR et l'EV (0.52%/trade → ~0). La cible est l'EV, pas le ratio — ne pas
re-tenter sans un mécanisme d'une autre nature (sortie partielle = interdite
par décision user, ou signal exogène). Aucun changement de config.

## Limites connues / suites possibles

- **Horizon < 1 an non garanti** (résiduel après la campagne POINTS-FAIBLES) :
  pire fenêtre 180j −16.35%, pire 90j −14.59%, 2025Q2 −24.70%. Le gouverneur
  coupe les hémorragies mais un trimestre perdant reste possible.
- **Le gouverneur est calibré sur UN épisode de DD** (nov-2024→juin-2025) : le
  bloc K=40 est un plateau (dd −27 à −38 sur 12 combos), mais la validation
  multi-épisodes n'existera qu'avec plus d'historique 1-min.
- L'année 1 est IN-SAMPLE pour les modèles (contamination assumée, décision
  user du test 720j) — le WR année 1 est optimiste ; la correction apportée
  porte sur la GESTION, pas sur la prédiction.
- `optimize/_harness.py` ne passe pas `trend_ctx` : les sweeps existants restent
  comparables à l'historique (comportement pré-campagne). Passer
  `trend_ctx=filters.load_trend_context()` à `simulate_trades` pour sweeper la
  config production.
- **Pas porté à `live/`** (copies indépendantes) : porter = copier
  `config_main.py` + `filters.py` + ajouter le calcul ER/SMA/âge dans le daemon
  (il n'a que 400 bougies 1h en mémoire — il faudra étendre le fetch pour la
  SMA200j) et brancher `take_profit` dans `live/position.py::evaluate_exit`.

## Épilogue — retour du DOWN au commit ecb04f8 (décision user, 2026-08-14)

Sur demande explicite du user, le côté DOWN est revenu au comportement exact du
commit `ecb04f8` (2026-07-04, avant toutes les campagnes ci-dessus ET avant le
wrapper régime / widen / caps de juillet) ; le côté UP conserve tout le stack.
Coupé côté DOWN : surcharges régime-bear, tpext, verrou de sécurité 1.10→0.25
(perte de garantie confirmée par le user), chop DOWN complet (sorties scalp +
bump d'entrée + palier chop/mbull), widen DOWN ; trailing réarmé à +0.15%
(valeur commit). Conservés sur les deux sens (choix user) : caps de concurrence
+ gouverneur. Implémentation : flags `CHOP_DOWN_ENABLED` / `DOWN_WIDEN_ENABLED`
(nouveaux), `DOWN_TPEXT_ENABLED=False`, `DOWN_RATCHET_STEPS_PCT=()`,
`REGIME_WRAPPER_ENABLED=False`.

Résultat 720j (run de vérification 2026-08-14) : année 1 **−17.12%** (DOWN a1
−33.39%, WR 37.19%), année 2 **+489.42%**, total **+388.53%**, DD −48.30%,
EV +0.4761%/trade — contre +112.57 / +1333.25 / +2946.67 / −24.23 avec le stack
complet. Les conclusions des campagnes restent valides comme historique ; ce
rapport documente désormais un état qui n'est plus la production.

### Suite (même jour) — alignement sur la prod déployée, remplace le retour commit

Découverte en comparant avec le code réellement déployé (`Desktop/modele_team`) :
la prod n'est PAS le commit ecb04f8 mais un état ~mi-juillet = commit + caps de
concurrence + widen DOWN x4/x2/x4 + wrapper régime (bear : hold x2 / gap x1.3).
Diagnostic `optimize/exp_dncommit.py` : le +372.01% DOWN du commit se décompose
en +167.94% (115 trades séquentiels) x +76.16% (25 trades empilés, jusqu'à 4
shorts simultanés — infinançable et impossible sur OKX HEDGE) ; reproduction au
centième vérifiée. Décision user : backtest DOWN = prod DOWN. Réactivés :
`DOWN_WIDEN_ENABLED=True`, `REGIME_WRAPPER_ENABLED=True` ; gouverneur restreint
à l'UP (`GOVERNOR_SIDES=("up",)`, livre toutes-directions). Résultat 720j :
année 1 −19.17% (DOWN −27.19%), année 2 +1554.87% (DOWN +520.99% ≈ baseline
pré-campagnes +502, sanity check), total +1237.64%, DD −60.01%. La dépendance
au régime (année 1 négative, DD année 1 −60%) est le comportement prod assumé.

### Suppression du gouverneur (même jour, dernière décision)

Diagnostic `optimize/exp_upgov.py` (obsolète depuis, gardé comme trace) : avec
le DOWN aligné prod, l'UP seul ne déclenche JAMAIS le disjoncteur (0 suspension
sur 720j en livre UP-seul) — 100% des suspensions venaient du saignement DOWN
année 1, et `GOVERNOR_SIDES=("up",)` les faisait payer à l'UP seul (a1 UP
+32.98% intrinsèque → +11.01% gouverné ; dans le stack complet le même outil
AJOUTAIT ~11 pts car il élaguait les vraies mauvaises périodes). Décision user :
**suppression complète du code** (config, filters.apply_governor, main.py,
rapport) — parité prod totale. Sans gouverneur, 720j : UP a1 +32.98% (83
trades, WR 63.9%), UP a2 +176.89%, DOWN inchangé.
