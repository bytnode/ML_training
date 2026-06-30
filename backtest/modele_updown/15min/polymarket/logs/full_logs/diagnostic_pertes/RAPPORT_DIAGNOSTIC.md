# Rapport de diagnostic — pertes de la team de modèles `15min`

**Périmètre** : `modele_up` + `modele_down`, fenêtre de backtest de 365 jours
hors-échantillon (2025-05-09 → 2026-05-09).
**Données** : `all_bets.csv` / `lost_bets.csv` / `won_bets.csv` — 4123 paris pris.
**Date du diagnostic** : 2026-05-21.

---

## Résumé exécutif

La team prend 4123 paris sur l'année, avec un **win rate de 59.71 %**
(2462 gagnés / 1661 perdus). La mission était d'identifier les patterns
récurrents qui génèrent les **40.3 % de pertes**.

**Verdict principal — il n'existe aucun pattern de perte filtrable.**
Cinq méthodes d'analyse indépendantes (balayage univarié, calibration,
minage de règles par arbre, méta-modèle, analyse temporelle) **convergent
toutes** vers la même conclusion : les paris perdants sont statistiquement
**indiscernables** des paris gagnants à partir de toute l'information connue
au moment de la décision.

- Le pire sous-groupe identifiable perd dans **48 % des cas** — jamais « la
  majorité des cas », encore moins « à chaque occurrence ».
- Un méta-modèle XGBoost en validation croisée temporelle atteint une
  **AUC de 0.52** — à peine au-dessus du pur hasard (0.50).
- Les pertes ne sont **ni regroupées dans le temps, ni concentrées sur une
  heure / un régime / une feature**.

**Cause racine** : tous les paris pris vivent dans le **plateau saturé** de la
courbe de calibration du modèle. La probabilité brute du modèle a un vrai
signal (AUC 0.60) mais ce signal ne sert qu'à séparer « marché baissier » de
« marché haussier ». Le seuil de décision est posé au début de la zone
haussière ; au-delà, le taux de réussite réel est **plat à ~60 %**, quelle
que soit la confiance affichée. Le système transforme le marché en un pari
binaire à ~60/40 et **n'a aucune résolution plus fine que cela**.

**Conséquence directe** : le filtrage des paris existants est
**prouvé plafonné à +1.3 point de WR**. Pour dépasser ~61 % de WR, il faut de
**nouvelles sources d'information**, pas un meilleur filtre.

**Trois leviers réels mais modestes** ont quand même été isolés et chiffrés
(section 4) — le principal : couper les paris pris à la limite du seuil les
**jours ouvrés** (477 paris à 52.0 % de WR).

---

## 1. Périmètre et méthode

### 1.1 Données

| Fichier | Contenu | Lignes |
|---|---|---|
| `all_bets.csv` | tous les paris pris | 4123 |
| `lost_bets.csv` | paris perdants | 1661 |
| `won_bets.csv` | paris gagnants | 2462 |

Intégrité vérifiée (script `00_exploration.py`) : `lost ∪ won = all`, aucun
doublon de timestamp, série strictement croissante, 364 jours de couverture.
Sur 144 colonnes, **68 sont constantes** (hyperparamètres, config, sidecar) —
elles décrivent le modèle, pas le pari, et ne peuvent rien discriminer.
Il reste **76 colonnes variables**, dont les vraies variables d'analyse.

Baseline : perte 40.29 %, WR 59.71 %. Par sens : **UP** 2323 paris à 59.45 %,
**DOWN** 1800 paris à 60.06 %. Les deux modèles ont une compétence réelle
(la couleur de base est ~51 % verte ; chaque modèle sélectionne des bougies
à ~60 % favorables — un edge de +8 à +11 points). Le problème n'est pas un
modèle inutile, c'est **où cet edge s'évapore**.

### 1.2 Garde-fous anti-surajustement

Avec 144 colonnes et 4123 lignes, un minage agressif **fabrique toujours** de
faux patterns. Trois garde-fous appliqués systématiquement :

1. **Correction multi-tests** (Benjamini-Hochberg) sur tout balayage univarié.
2. **Robustesse temporelle** : un pattern doit tenir sur les deux moitiés du
   backtest (H1 = 1er semestre, H2 = 2e).
3. **Validation hors-échantillon stricte** : règles apprises sur une moitié et
   testées sur l'autre ; méta-modèle en validation croisée temporelle (jamais
   de fuite du futur), comparé à une bande de bruit par permutation des labels.

Aucun prédicteur n'utilise d'information postérieure à la décision (`next_*`,
`result` exclus). Les prix bruts sont exclus (le niveau du BTC, 63 k→125 k sur
l'année, n'est qu'un proxy temporel).

---

## 2. Diagnostic principal — les pertes ne sont pas structurées

Cinq angles d'attaque indépendants. Tous donnent le même résultat.

### 2.1 Balayage univarié — aucune feature ne prédit la perte

Script `01_univarie.py`. Pour chacun des 56 prédicteurs : taux de perte par
catégorie / décile, test binomial, correction FDR, robustesse H1/H2.

> **Résultat : 0 bin robuste ET significatif après correction FDR**, dans les
> trois segments (global, UP, DOWN).

- Corrélations de Spearman feature↔perte **toutes** dans `|ρ| < 0.07`.
- Information mutuelle feature↔perte **quasi nulle** (max 0.034).
- Le pire décile de la meilleure feature atteint **~48-50 %** de perte — soit
  un *lift* de 1.2 seulement, qui ne survit pas à la correction multi-tests.
- Les « heures faibles » apparentes (h4, h3 ~ 49-52 % de perte) ne survivent
  pas : ce sont des artefacts de tests multiples — l'« heure forte » h12
  (29 % de perte) est tout aussi « significative » en sens inverse, signature
  classique du bruit.

**Vérification directe — une donnée ressort-elle côté pertes ?**
Script `07_signature_pertes.py`. Pour chaque feature, comparaison directe de sa
distribution sur les paris perdus vs gagnés : moyenne, médiane, taille d'effet
de Cohen, Mann-Whitney, AUC mono-feature, robustesse H1/H2, queues extrêmes,
dispersion.

> **0 feature ne ressort** (FDR-significatif ET même sens sur les deux
> semestres), dans les trois segments. L'effet le plus fort de tout le
> balayage — `volatility_state` sur les paris DOWN — atteint un Cohen d de
> **0.13** (négligeable) et ne survit pas à la correction FDR.

Cas de l'**ATR** (intuition testée explicitement) : `atr_pct` est *légèrement*
plus élevé sur les pertes — médiane **0.2215 %** vs **0.2143 %** sur les gains,
Cohen d = +0.066. L'intuition est donc *directionnellement* correcte, mais
l'écart est trop faible pour discriminer quoi que ce soit (p = 0.04 brut, non
significatif après FDR ; AUC mono-feature 0.519). Idem `volume` (d = +0.05,
p = 0.79). Ni les queues extrêmes (top/bottom 2.5 %, n ≈ 103) ni la dispersion
(test de Levene : 0/56 feature significative) ne révèlent de coupable.

### 2.2 Calibration — le modèle ne sait pas quand il a raison

Script `02_calibration.py`. C'est le résultat structurel central.

| Score | Segment | AUC win/loss | IC95 |
|---|---|---|---|
| `bet_confidence` | GLOBAL | **0.503** | [0.485, 0.521] |
| `bet_confidence` | UP | 0.504 | [0.480, 0.528] |
| `bet_confidence` | DOWN | 0.506 | [0.480, 0.534] |
| `bet_edge` | GLOBAL | 0.521 | [0.503, 0.539] |
| `bet_edge` | UP | 0.529 | [0.505, 0.553] |

> **La confiance du modèle (`bet_confidence`) ne distingue pas un pari gagnant
> d'un pari perdant : AUC = 0.50.** Spearman(confiance, gain) = +0.007.
> La courbe de fiabilité est non monotone — un décile de *haute* confiance
> (0.585-0.591) affiche un des *plus bas* WR (52.6 %).

L'edge (`bet_edge` = confiance − seuil) garde un **murmure** de pouvoir
discriminant (AUC 0.52, côté UP surtout). C'est le seul signal exploitable
détecté dans toute l'étude — il est faible (voir section 4).

### 2.3 Minage de règles multivariées — tout s'effondre hors-échantillon

Script `03_interactions_regles.py`. Un arbre de décision (profondeur 4) est
appris sur une moitié temporelle ; ses feuilles à forte perte sont extraites,
puis **appliquées telles quelles sur l'autre moitié**.

> Sur **22 feuilles** à forte perte détectées en apprentissage (perte
> **52 % à 74 %**), **8 « tiennent »** — mais toutes s'effondrent à
> **42-49 % de perte** hors-échantillon. **Aucune règle, dans aucun segment,
> ne perd dans la majorité des cas en validation hors-échantillon.**

La structure trouvée en apprentissage est du surajustement pur.

### 2.4 Méta-modèle — les pertes sont-elles prédictibles ? Non.

Script `04_meta_modele.py`. Un méta-modèle est entraîné à prédire `loss` à
partir des 57 prédicteurs (features + **les sorties des deux modèles** :
`p_up`, `p_down`, edges…). Validation croisée temporelle stricte.

| Modèle | AUC (CV temporelle) |
|---|---|
| XGBoost | **0.5205** |
| LightGBM | 0.5058 |
| Régression logistique | 0.5052 |
| *Bande de bruit (labels permutés)* | *IC95 [0.4815, 0.5199]* |
| **Walk-forward, 2723 paris OOS** | **0.5174** |

> Le méta-modèle le plus puissant atteint **AUC 0.52** — à un cheveu du
> plafond de la bande de bruit (0.520). En walk-forward honnête : **0.517**.

Le **méta-filtre** simulé (sauter les X % de paris jugés les plus à risque) :

| % sautés | WR des gardés (OOS) | Gain |
|---|---|---|
| 5 % | 60.25 % | +0.09 pt |
| 15 % | 60.11 % | **−0.04 pt** |
| 25 % | 60.43 % | +0.28 pt |
| 40 % | 61.44 % | +1.29 pt |

> Sauter les 15 % de paris que le méta-modèle juge « les plus risqués »
> **dégrade** le WR. Même en sacrifiant 40 % du volume, le gain plafonne à
> **+1.3 point**. Les paris qu'il signale comme « les pires » gagnent encore
> 58-59 %.

### 2.5 Analyse temporelle — pas de dérive, pas de regroupement

Script `05_temporel.py`.

- **WR mensuel** : aucun mois significativement mauvais (le pire, 2025-07 à
  57.1 %, donne p = 0.41). Corrélation(temps, perte) = −0.016 → **pas de
  dérive** du modèle.
- **Test des suites de Wald-Wolfowitz** : z = **+0.075** → la séquence de
  pertes est **indiscernable de tirages indépendants**. Autocorrélations
  lags 1-10 toutes ≈ 0.
- **Perte conditionnelle au passé** : P(perte | pari précédent perdu) =
  **40.2 %** = P(perte | pari précédent gagné) = 40.4 %. **Zéro persistance.**
  Un « cooldown après une série de pertes » ne servirait à rien (après 3
  pertes d'affilée, le pari suivant gagne même légèrement *plus*).
- **Plus longue série perdante** : 8 — *inférieure* à l'attente sous
  indépendance (8.8 ; p95 = 11).

### 2.6 Tableau de convergence

| Méthode | Question | Réponse |
|---|---|---|
| Univarié + FDR | Une feature prédit-elle la perte ? | Non (ρ ≈ 0) |
| Calibration | La confiance classe-t-elle les paris ? | Non (AUC 0.50) |
| Arbre + OOS | Une règle perd-elle en majorité, OOS ? | Non (s'effondre à ~45 %) |
| Méta-modèle | Les pertes sont-elles prédictibles ? | Non (AUC 0.52) |
| Temporel | Les pertes se regroupent-elles ? | Non (z = 0.08) |

**Cinq méthodes, une seule conclusion : les 40 % de pertes sont du bruit
irréductible à partir de l'information disponible.**

---

## 3. Cause racine — le plateau de calibration

Pourquoi aucun filtre ne fonctionne ? La réponse est dans la calibration brute
de la probabilité du modèle (script `02`, section 4), mesurée sur les 4123
paris :

| `p_up` (probabilité de bougie verte) | Taux de vert réel |
|---|---|
| 0.446 – 0.474 (zone basse) | **38 – 40 %** |
| 0.474 – 0.552 (zone de bascule) | ~51 % |
| **0.552 – 0.616 (zone haute)** | **57 – 62 % — PLAT** |

`AUC(p_up → bougie verte) = 0.60` : la probabilité du modèle **a un vrai
signal**. Mais ce signal sert uniquement à séparer la zone baissière
(`p_up < 0.48` → 38 % de vert) de la zone haussière. **Au-dessus de 0.55, la
courbe est plate** : que le modèle annonce 0.55 ou 0.61, le marché monte
réellement ~60 % du temps.

Or le seuil de décision d'un pari UP est ~0.55-0.59. **Tous les paris pris
vivent donc dans le plateau plat.** D'où, mécaniquement :

- `bet_confidence` a une AUC de 0.50 sur les paris pris (il ne reste aucun
  gradient dans la zone prise) ;
- les 40 % de pertes sont le simple complément de ce plateau à ~60 % — ils
  ne sont **concentrés nulle part**.

> **La faille du système, formulée précisément : le modèle réduit le marché à
> un appel binaire « plutôt haussier / plutôt baissier » juste ~60 % du temps,
> et il n'a aucune résolution plus fine.** Chaque pari pris est une
> proposition à ~60/40. Les paris qui perdent ne sont pas de « mauvais
> paris » — ce sont le côté perdant de paris qui étaient tous à 60/40.

On ne peut pas filtrer une pièce 60/40 pour en faire une meilleure pièce.

---

## 4. Patterns réels et leviers exploitables

Le diagnostic est largement négatif, mais l'analyse a tout de même isolé des
leviers **réels, robustes et chiffrés**. Ils sont **modestes** — cohérent avec
la section 3 — mais exploitables immédiatement par configuration.

### 4.1 La zone d'edge marginal — concentrée sur les jours ouvrés

Le seul axe faiblement informatif est `bet_edge` (la marge confiance − seuil).
Les paris pris **à la limite du seuil** sont plus faibles :

| Sous-groupe | n | WR | p-value | Robuste H1/H2 |
|---|---|---|---|---|
| `bet_edge < 0.003` (global) | 847 | 55.0 % | 0.006 | oui |
| └ dont **jours ouvrés** | **477** | **52.0 %** | **0.0006** | oui (H1 49.8 %, H2 53.9 %) |
| └ dont week-end | 370 | 58.9 % | 0.75 | non — *pas un problème* |

> **Pattern le plus net de toute l'étude** : un pari pris à moins de 0.003
> au-dessus du seuil, **un jour ouvré**, gagne 52.0 % du temps (n = 477,
> p = 0.0006). Au 1er semestre, ces paris étaient même à **49.8 % de WR** —
> perdants nets. Et le problème est **entièrement** côté jours ouvrés : le
> même filtre côté week-end ne donne rien.

### 4.2 Le segment jours ouvrés est le maillon faible

| Segment | n | WR | Edge moyen |
|---|---|---|---|
| Jours ouvrés (`is_weekend=0`) | 1501 | **58.2 %** | 0.0078 |
| Week-end (`is_weekend=1`) | 2622 | 60.6 % | 0.0150 |

Les jours ouvrés sont pris avec un edge **deux fois plus faible** et un WR
inférieur de 2.3 points, stable sur les deux moitiés (H1 57.8 %, H2 58.6 %).
La calibration asymétrique semaine/week-end actuelle ne compense pas
totalement la difficulté intrinsèque des jours ouvrés.

### 4.3 `hour_boost` — dilutif, mais pas coupable

Le rabais de seuil de −0.02 sur 20h-22h admet 921 paris « boost-only »
(ils n'auraient pas passé le seuil sans le rabais). Leur WR : **59.2 %** —
sous la moyenne mais **non significatif** (p = 0.74) et non robuste. Le
`hour_boost` achète du volume à un WR légèrement sous la moyenne ; il n'est
pas une source de pertes nette. Levier de faible priorité.

### 4.4 Tableau des leviers (plancher d'edge global)

| Plancher d'edge | Volume gardé | WR après | Gain | Robuste |
|---|---|---|---|---|
| aucun (actuel) | 100 % | 59.71 % | — | — |
| `≥ 0.003` | 79.5 % | 60.93 % | +1.21 pt | oui |
| `≥ 0.010` | 47.5 % | 61.34 % | +1.62 pt | oui |
| `≥ 0.016` | 28.1 % | 60.33 % | +0.61 pt | oui |

> **Le WR plafonne à ~61.3 %** même en coupant la moitié du volume. C'est la
> preuve, par un autre chemin, qu'il n'existe pas de réservoir de pertes
> concentré. À noter : ce plancher d'edge, **une seule ligne de code**, fait
> aussi bien que le méta-modèle XGBoost (section 2.4) — inutile de construire
> une couche ML de filtrage.

---

## 5. Ce qui n'est PAS en cause

Résultat tout aussi utile : éviter au projet de chasser des fausses pistes.
Toutes les hypothèses suivantes ont été testées et **rejetées** :

- ❌ **Heures faibles** — aucune heure n'est mauvaise après correction FDR.
- ❌ **Régime de volatilité** — `volatility_state` ne discrimine pas (l'état
  extrême est déjà exclu par les `skip_cells` du sidecar).
- ❌ **Mean-reversion / chasse au momentum** — le modèle ne parie jamais UP
  au-dessus de la bande de Bollinger haute ni DOWN sous la bande basse ;
  il évite déjà structurellement ces zones.
- ❌ **Désaccord des deux modèles** — `p_up + p_down` (cohérence) : sans effet.
- ❌ **Modèle opposé « chaud »** — quand le modèle opposé a failli tirer :
  sans effet (la règle de cohérence d'équipe annule déjà ces cas).
- ❌ **Dérive temporelle** — aucune dégradation sur l'année.
- ❌ **Regroupement / séries de pertes** — séquence indépendante ; un
  « cooldown » serait inutile.
- ❌ **Pertes = petites bougies / pile ou face mécanique** — les bougies des
  paris perdus ne sont **pas** plus petites que celles des gagnants
  (Mann-Whitney p = 0.21) ; ce sont de vrais mouvements adverses.
- ❌ **Un méta-filtre ML** — plafonné à +1.3 pt, prouvé.

---

## 6. Recommandations priorisées

### Priorité 1 — Acter le plafond de signal *(stratégique)*

Le résultat le plus important : **le filtrage est prouvé plafonné à +1.3 pt
de WR**. Les features actuelles sont épuisées — le modèle a extrait tout le
signal binaire qu'elles contiennent. Pour dépasser ~61 % de WR, il faut
**créer de la résolution à l'intérieur de la zone haussière/baissière**, donc
de **nouvelles sources d'information** :
order-flow / carnet d'ordres, données cross-asset (l'arbre `30m` exploite déjà
ETH, funding rate, Coinbase), features de régime à plus long contexte.
**Ne pas investir d'effort dans un méta-filtre** des paris existants.

### Priorité 2 — Plancher d'edge sur les jours ouvrés *(gain immédiat)*

Couper le pattern le plus net (section 4.1) : remonter le seuil des **jours
ouvrés** d'environ +0.003 (`up` 0.587→0.590, `down` 0.555→0.558), ou ajouter
une grille `bet_edge ≥ 0.003` limitée à `is_weekend=0`. Effet : retirer 477
paris à 52.0 % de WR → WR global 59.71 % → **60.7 %** (−11.6 % de volume).
Robuste, significatif (p = 0.0006), risque quasi nul.

### Priorité 3 — Revoir la calibration des jours ouvrés *(structurel léger)*

Le segment jours ouvrés (WR 58.2 %) est le maillon faible. Revisiter
`asym_vol_frac_weekday` / la stratégie de seuil semaine pour resserrer la
sélection — quitte à accepter moins de volume en semaine.

### Priorité 4 — Réexaminer `hour_boost` *(faible priorité)*

Le rabais −0.02 sur 20h-22h est dilutif (WR boost-only 59.2 %) sans être
nuisible. Si le volume n'est pas l'objectif premier, réduire `hour_boost_delta`
ou le supprimer ; sinon le laisser.

### À ne PAS faire

Chercher des « mauvaises heures », un « cooldown après pertes », un
méta-modèle de filtrage, ou un régime de volatilité coupable — tout cela est
prouvé sans effet (section 5).

---

## 7. Limites de l'analyse

- **Sous-ensemble de features** : les logs contiennent ~40 features par pari ;
  les modèles de base peuvent en utiliser davantage. Un prédicteur de perte
  caché *uniquement* dans une feature non journalisée resterait invisible.
  Atténuation : les logs incluent **les sorties des deux modèles** (`p_up`,
  `p_down`, edges) — qui résument tout ce que les modèles ont conclu ; même
  ainsi, la perte reste imprédictible.
- **WR, pas P&L** : l'analyse porte sur le taux de réussite. Sur Polymarket le
  prix d'entrée varie ; un pari à 52 % de WR peut rester rentable ou non selon
  le prix. Les recommandations supposent que le WR est le proxy de rentabilité.
- **Une seule année** OOS (4123 paris). Statistiquement solide pour les effets
  forts ; un effet très faible (lift < 1.1) resterait sous le bruit.
- **2 dojis** (`next_open = next_close`) dans les données — impact négligeable.

---

## 8. Annexe — reproduire l'analyse

Tout le code est dans `diagnostic_pertes/`. Depuis ce dossier :

```bash
python 00_exploration.py         # intégrité des données
python 01_univarie.py            # balayage univarié + FDR
python 02_calibration.py         # calibration, AUC, fiabilité
python 03_interactions_regles.py # interactions + règles d'arbre OOS
python 04_meta_modele.py         # méta-modèle + walk-forward + SHAP
python 05_temporel.py            # dérive, clustering, séries
python 06_mecanique_leviers.py   # mécanique des pertes + leviers
python 07_signature_pertes.py    # distribution win vs loss de chaque feature
```

Tables détaillées dans `resultats/*.csv`. Figures :

- `resultats/02_calibration.png` — courbes de fiabilité + top-K.
- `resultats/04_meta_modele.png` — méta-filtre + importance SHAP.
- `resultats/05_temporel.png` — WR mensuel, WR glissant, drawdown.
- `resultats/06_leviers.png` — compromis WR/volume du plancher d'edge.
- `resultats/07_signature.png` — écart standardisé (Cohen d) par feature +
  distribution de l'ATR gagnés vs perdus.

---

*Diagnostic produit le 2026-05-21 — analyse de 4123 paris sur 365 jours
hors-échantillon. Méthodes : balayage univarié corrigé FDR, calibration,
minage de règles par arbre validé hors-échantillon, méta-modèle (XGBoost /
LightGBM / régression logistique) en validation croisée temporelle, analyse
temporelle (test des suites, autocorrélation).*
