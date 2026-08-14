# Rapport — Campagne « exit management adaptatif », côté DOWN (2026-08-11)

> **ADDENDUM (même jour, itérations 4-7) — la campagne a finalement ABOUTI.**
> Le verdict négatif ci-dessous reste valable pour tout **resserrage** : il est
> confirmé par le diagnostic de continuation post-sortie (les sorties actuelles
> ne sont PAS prématurées : 11/15 trailing et 22/38 SL auraient perdu davantage).
> Mais le gisement était de l'autre côté — **au-delà du take-profit** : sur les
> 39 sorties TP à +1,60 %, le pic post-TP moyen était +2,98 % (+3,27 % en bear).
> Mécanisme retenu (`DOWN_TPEXT_*`, porté dans `main.py`) :
> **extension conditionnelle du TP en bear** — au franchissement de 1,60 %, si
> régime bear (barre de décision) ET chemin peu efficient (ER ≤ 0,075, les
> *grinders* continuent, les spikes whipsawent — gate découvert par autopsie,
> mes premiers gates étaient inversés), on laisse courir avec sortie sur
> **cassure de structure** (pivot haut 1-min p=20 confirmé, formé pendant
> l'extension) + plancher dur à TP−0,80 %. Trailing classique et widen intacts.
> Résultat (365 j, `python main.py` vérifié bit-à-bit vs harnais) :
> **DOWN +502,01 % → +694,89 %** (EV 1,786 → 2,073, WR/worst/DD inchangés),
> **total +1626,48 % → +2179,62 %**. Plateaux : p 12–25, hard-guard 0,6–1,2,
> ER 0,065–0,090 — tous plats, centre retenu partout.
> **Limite assumée (décision user)** : gain concentré dans la moitié bear
> (h1 +277→+423) ; la moitié haussière rend ~8 pts (h2 +59,7→+51,9) via 3
> extensions en bear superficiel qu'aucun gate principiel ne sépare (DMI, âge
> du bear, vitesse : testés, autopsies dans ce dossier). 14 extensions au
> total sur la fenêtre — re-sweeper quand l'échantillon grossit. Chandelier,
> SAR et ATR-normalisation ont aussi été testés comme sortie d'extension :
> la cassure de structure les bat tous (ils neutralisent l'extension).
> Détail des runs : `logs/sweep_down_tpext*.csv`, `logs/diag_down_post*.csv`.

**Question posée** : le trailing DOWN à gap fixe (1,20 %) est-il de l'amateurisme,
et un resserrage adaptatif (détection de retournement → grille resserrée)
améliore-t-il le système ?

**Réponse, après ~270 configurations adaptatives sur 5 familles d'hypothèses
(plus les 3 150 de la campagne cliquet)** : **non — le laisser-courir à échelle
fixe est l'optimum empirique de ce système**, et la raison est maintenant
chiffrée, pas supposée. Aucune variante ne bat la baseline sur les deux moitiés
de la fenêtre.

Baseline DOWN (config UP retenue, fenêtre 2025-07-03 → 2026-07-03) :
**116 trades, net +502,01 %, EV +1,786 %, h1 +276,87 / h2 +59,74**.

---

## Phase 1 — Diagnostic (diag_down_adaptive.py → logs/diag_down_mfe.csv, diag_down_events.csv)

- **MFE capture ratio** (trades MFE ≥ 0,6 %) : médiane **1,00** (le trade médian
  capture tout — les 39 sorties TP), moyenne 0,49, tirée par 15 sorties
  `trailing` (pic moyen 1,10 % → final −0,34 %, capture −0,39) et 24 `timestop`
  (capture 0,28).
- **Profit restitué total** : 31,3 pts de prix (×5 levier = 157 pts d'equity).
  C'est le gisement théorique maximal, pas un gain accessible : personne ne sort
  au pic exact.
- **Event study** (60 événements « pic ≥ 0,6 % puis retrace ≥ 40 % du pic ») :
  **45 repartent vers un nouveau pic** (final moyen **+1,25 %**) contre
  **15 qui meurent** (final moyen −0,26 %). Deux signaux 1-min séparent les
  médianes : `vol_ratio` 15 min/60 min-pré-entrée (morts 1,29 vs survivants 0,87)
  et `adv_volshare` (0,59 vs 0,47). Vitesse de retrace et temps-depuis-pic ne
  séparent rien.

**Le déficit réel n'est pas celui que la prémisse suppose.** Le ratio 45/15 est
le cœur du problème : tout resserrage au moment du retrace mise contre un
événement à 75 % favorable, et l'enjeu est asymétrique (couper un « recovered »
coûte ~1,25 % en moyenne, sauver un « died » rapporte ~0,5 pt).

## Phase 2/3 — Hypothèses testées et verdicts (sweep_down_adaptive.py + runs annexes)

| Famille | Grille | Meilleur net DOWN | Verdict |
|---|---|---|---|
| Cliquet inconditionnel (campagne précédente, plats + proportionnels) | 3 150+ | 502 → s'effondre dès qu'il mord (+4 % à arm 0,2) | destructeur monotone |
| H2a/b/c — resserrage conditionné aux signaux 1-min (vol_ratio, adv_volshare, OR/AND) | 108 | **+307 %** (AND le plus strict) | le signal est informatif (+307 > +248 inconditionnel comparable) mais très loin de +502 |
| H3 — stall-exit (aucun nouveau pic depuis M min) | 9 | +239 % | échec |
| H1 — gap normalisé ATR à l'entrée (mult 14–40, correctement calibré) | 5 | +364 % | échec — même mult 26 (≡ 1,2 % médian) ne fait que +349 % : c'est la **dispersion** autour de 1,2 % qui détruit, dans les deux sens |
| H1b — gap re-normalisé ATR chaque minute | 5 | +176 % | échec net |
| Hreg — resserrage conditionné au régime 1h (neutral/bull serrés, bear intact) | 9 | +499 % (neutral 0,9) | le plus proche ; améliore h2 (+65,4 vs +59,7) mais détruit h1 (−68 pts) — jamais les deux moitiés |

Garde-fous appliqués partout : neutralité bit-à-bit du walk réécrit vérifiée par
assert avant chaque sweep ; validation exigée sur les deux moitiés (net ET EV) ;
n total ≥ 240 ; pire trade ≥ −15 % ; DD ≥ −20 % ; zéro lookahead (fenêtres
rolling closes à la minute de décision, régime lu à la barre de décision).

## Pourquoi le fixe gagne — l'explication structurelle

1. **L'échelle de retracement des trend-rides BTC 1h est ~constante (~1,2 %)**,
   non proportionnelle à la vol 1-min à l'entrée — d'où l'échec de la
   normalisation ATR, pourtant le standard « pro » : elle ajoute de la variance
   autour du bon niveau, et cette variance coûte des deux côtés.
2. **Le trou de 31 pts est une prime d'option, pas une fuite.** Les 39 TP à
   +1,9 % de moyenne (et la queue jusqu'à +5 %) ne s'achètent qu'en acceptant
   que 15 trades sur 60 retracements meurent. Chaque mécanisme qui réduit la
   prime réduit davantage le payoff.
3. **L'adaptativité utile est déjà dans le système** : SL temporel (adaptatif
   dans le temps), wrapper de régime (bear → hold ×2, gap ×1,3, choisi à
   l'entrée), widen-on-profit (adaptatif au flux de signaux). La pièce
   manquante supposée — détection de retournement intra-trade → resserrage —
   est précisément celle que les données rejettent.

## Recommandation finale

- **Ne pas déployer de resserrage adaptatif côté DOWN.** Conserver gap fixe
  1,20 %, TP 1,60 %, SL temporel et wrapper de régime inchangés.
- La seule piste restée entrouverte : le resserrage en régime **neutral** seul
  améliore h2 de façon répétée (~+5 pts) — insuffisant aujourd'hui (coût h1),
  mais à re-tester si la fenêtre glisse vers un marché durablement haussier.
- **Limites** : fenêtre unique de 365 j bornée par les test-slices des modèles
  (pas de vrai out-of-sample) ; le split h1/h2 est le seul proxy de robustesse ;
  15 « died » est un petit échantillon — un signal de retournement radicalement
  meilleur (carnet d'ordres, funding, liquidations) pourrait rouvrir la question,
  la donnée 1-min OHLCV ne suffit pas.

Scripts : `optimize/diag_down_adaptive.py`, `optimize/sweep_down_adaptive.py`.
CSV : `logs/diag_down_mfe.csv`, `logs/diag_down_events.csv`,
`logs/sweep_down_adaptive.csv`, `logs/sweep_down_atr.csv`,
`logs/sweep_down_regimetight.csv`.
