# Diagnostic des pertes — boîte à outils

Session d'analyse de données sur les logs de backtest de la team de modèles
`15min` (`modele_up` + `modele_down`). Objectif : identifier les patterns
récurrents responsables des paris perdants.

## Données analysées

Trois fichiers à la racine de `full_logs/` (un cycle complet, 365 j hors-échantillon) :

| fichier | contenu | lignes |
|---|---|---|
| `all_bets.csv` | tous les paris pris | 4123 |
| `lost_bets.csv` | paris perdants | 1661 |
| `won_bets.csv` | paris gagnants | 2462 |

144 colonnes par pari : décision d'équipe, p/seuil/edge des deux modèles,
OHLC + microstructure, vecteurs de features `up_`/`dn_`, hyperparams/config/sidecar.

## Scripts (lancer dans l'ordre, depuis ce dossier)

```bash
python 00_exploration.py        # intégrité des données, colonnes constantes
python lib.py                   # bibliothèque partagée (chargement + stats)
python 01_univarie.py           # balayage univarié : 1 feature à la fois
python 02_calibration.py        # la confiance du modèle classe-t-elle les paris ?
python 03_interactions_regles.py# interactions + minage de règles (arbre, OOS)
python 04_meta_modele.py        # méta-modèle : les pertes sont-elles prédictibles ?
python 05_temporel.py           # dérive, clustering, séries perdantes
python 06_mecanique_leviers.py  # mécanique des pertes + leviers de config
python 07_signature_pertes.py   # distribution win vs loss de chaque feature
```

Dépendances : `pandas numpy scipy scikit-learn xgboost lightgbm matplotlib`
(toutes déjà présentes dans l'environnement du projet).

## Méthodologie anti-surajustement

Avec 144 colonnes et 4123 lignes, un minage agressif *trouve toujours* des
faux patterns. Trois garde-fous appliqués partout :

1. **Correction multi-tests** (Benjamini-Hochberg) sur tout balayage univarié.
2. **Robustesse temporelle** : un pattern doit tenir sur les deux moitiés du
   backtest (H1 = 1er semestre, H2 = 2e).
3. **Validation hors-échantillon stricte** : les règles d'arbre sont apprises
   sur une moitié temporelle et testées sur l'autre ; le méta-modèle est évalué
   en validation croisée temporelle (jamais de fuite du futur) et comparé à une
   bande de bruit obtenue par permutation des labels.

Aucun prédicteur n'utilise d'information postérieure à la décision
(`next_*`, `result` exclus). Les prix bruts sont exclus (proxy temporel).

## Sorties

- `resultats/*.csv` — toutes les tables intermédiaires.
- `resultats/*.png` — figures (calibration, méta-filtre, temporel, leviers).
- `RAPPORT_DIAGNOSTIC.md` — le rapport de diagnostic final.
