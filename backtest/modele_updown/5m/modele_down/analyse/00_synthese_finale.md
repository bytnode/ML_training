# Synthese finale : pourquoi modele_down sous-estime ses probabilites

## Le probleme observe

**modele_down** : p_down moyen sur paris pris = 53-55%, WR observee = 60-61%. Ecart de 5-7pp.
**modele_up** : p_up moyen sur paris pris = 57.25%, WR observee = 60.41%. Ecart de 3pp.

Le modele DOWN sort des probabilites systematiquement trop basses par rapport a la WR reelle.

## Diagnostic complet (4 audits)

### Audit 1 — Features parquet (01_features_diff)

| | DOWN | UP |
|---|---|---|
| Total cols | 29 | 24 |
| Features used by train.py | 25 | 19 |
| Label P(class=1) | 0.4921 | 0.5079 |
| Common features values | identiques (16/16 checked) | identiques |

- **6 features DOWN-only** : `body_ratio_close_30m`, `dist_to_high_50`, `hour_sin`, `low_ret`, `obv_pressure_5m`, `volume_ratio_30m`
- **1 feature UP-only** : `volume_rel_30d`
- **Label inverse 100%** : pour chaque timestamp commun, label_down + label_up = 1 (verifie)
- **DOWN inclut `is_weekend` comme feature** (UP l'exclut, l'utilise seulement pour bucketing)

### Audit 2 — Variantes (02_train_variants)

Memes hyperparams sur 6 variantes :

| Variant | n_feat | mean | std | **max** |
|---|---|---|---|---|
| A_UP_full | 19 | 50.29% | 3.38pp | **59.39%** |
| B_UP_no_vol30d | 18 | 50.20% | 3.21pp | **59.14%** |
| C_DOWN_full (current) | 25 | 49.81% | 3.15pp | **56.38%** |
| D_DOWN_intersect_excl_isweekend | 18 | 49.80% | 3.20pp | **56.60%** |
| F_DOWN_intersect_with_isweekend | 19 | 49.77% | 3.43pp | **57.82%** |

**Comparaisons cles** :
- **B vs D (memes 18 features, label different)** : UP_max=59.14, DOWN_max=56.60 → **2.54pp gap pur du label**
- **D vs F (ajout de is_weekend)** : DOWN_max passe 56.60 → 57.82 → +1.22pp grace a is_weekend
- **A vs F (UP avec vol_rel_30d, DOWN avec intersect+is_weekend)** : 59.39 vs 57.82 → **1.57pp gap residuel intrinseque**

### Audit 3 — Ablation features DOWN-only (03_feature_ablation)

Sur baseline F, ajout d'UNE feature DOWN-only a la fois :

| Feature ajoutee | Δmax |
|---|---|
| body_ratio_close_30m | -1.15pp |
| **dist_to_high_50** | **-1.83pp** (pire) |
| hour_sin | -0.59pp |
| low_ret | -1.37pp |
| obv_pressure_5m | -1.24pp |
| volume_ratio_30m | -1.08pp |

**Conclusion** : aucune des 6 features DOWN-only n'aide. Toutes compriment la spread.

### Audit 4 — Hyperparam sweep (04_hyperparam_sweep)

Sur baseline F, variation de min_child_weight, max_depth, gamma, learning_rate :

| Tweak | max |
|---|---|
| baseline (depth=5) | 57.82% |
| **max_depth=6** | **58.60%** (+0.78pp) |
| max_depth=8 | 58.34% |
| max_depth=7 | 57.93% |
| min_child_weight tweaks | <= baseline |
| gamma tweaks | = baseline (aucun effet) |
| learning_rate tweaks | <= baseline |
| ensemble 5 seeds | 57.12% (pire !) |

**Conclusion** : seul `max_depth=6` ameliore.

### Audit 5 — Combo final (05_final_combo)

Application de la strategie complete V2 (quantile + hour_boost) :

| Config | n_taken | WR | p_mean_taken | p_max |
|---|---|---|---|---|
| **CURRENT (full, d=5)** | 1485 | **61.28%** | 54.73% | 57.15% |
| clean (19 feats), d=5 | 1423 | 59.94% | 55.14% | 57.82% |
| **clean (19 feats), d=6** | 1391 | 59.09% | **55.54%** | **58.60%** |
| full features, d=6 | 1460 | 60.68% | 54.63% | 57.42% |

## Causes identifiees

### Cause #1 (FONDAMENTALE) — Asymetrie du marche BTC

Les bougies BTC ont un leger biais UP (50.79% vertes, 49.21% rouges sur 14 ans). Le modele apprend les patterns "next is up" plus facilement parce que :
- Plus d'exemples positifs (label=1) dans la classe UP
- Les signaux predictifs pour UP sont plus distinctifs (peut-etre lies au biais haussier structurel du BTC)

Resultat : **avec EXACTEMENT les memes features et hyperparams**, le modele DOWN sort une distribution p_down moins etalee que p_up. **Gap minimum incompressible : ~1.5pp sur p_max.**

### Cause #2 — Features DOWN-only nuisibles

Le parquet DOWN contient 6 features absentes de UP : `body_ratio_close_30m`, `dist_to_high_50`, `hour_sin`, `low_ret`, `obv_pressure_5m`, `volume_ratio_30m`.

Chacune individuellement compresse la distribution p_down. Toutes ensemble : -1.25pp sur p_max.

Hypothese : ces features apportent peu de signal predictif additionnel (XGBoost leur donne 3-3.4% d'importance chacune) mais ajoutent du bruit. Le val logloss plateau plus tot → early stopping declenche plus tot → moins de trees → distribution moins etalee.

**Effet sur p_max : -1.25 a -1.8pp.**

### Cause #3 — max_depth sous-optimal

Le default `max_depth=5` (herite de modele_15m) donne max=57.82% sur DOWN. `max_depth=6` donne 58.60% (+0.78pp). Plus profond ne sert pas (depth=7,8 plateau).

**Effet : ~0.8pp.**

## Synthese quantitative

Decomposition du gap entre UP (max=59.39%) et CURRENT DOWN (max=57.15%) :

| Source | Contribution |
|---|---|
| Cause #1 (label asymetrie BTC) | ~1.5pp |
| Cause #2 (features DOWN-only nuisibles) | ~1.25pp |
| Cause #3 (max_depth=5 sous-optimal) | (potentiel +0.8pp gain) |
| **Gap observe (max UP - max DOWN_current)** | **2.24pp** |

## Trade-off final

L'audit revele un **trade-off entre confiance affichee et WR test** :

- **Config CURRENT** : 1485 paris @ **61.28% WR**, p_mean affichee 54.73% (incoherent mais WR excellent)
- **Config "clean+d=6"** : 1391 paris @ 59.09% WR, p_mean 55.54% (plus coherent mais WR plus bas)

Pour les CAUSES :
- Cause #1 est intrinseque, irreductible sans changer l'architecture du modele
- Cause #2 est correctible (retirer les 6 features DOWN-only) mais SACRIFIE la WR test
- Cause #3 est correctible (max_depth=6) avec leger impact sur la WR

## Recommandation

Le user doit choisir entre 2 mondes :

**Monde A — privilegier la WR** : garder CURRENT. WR=61.28%, mais p_mean=54.73% reste affichee comme "trop basse". C'est cosmetique.

**Monde B — privilegier la coherence proba/WR** : passer en clean features + max_depth=6. WR=59.09%, p_mean=55.54%. Plus interpretable mais WR baisse de 2pp.

**Ma recommandation** : **rester en Monde A** (CURRENT). La p_mean basse n'est pas un bug, c'est une caracteristique du modele liee a l'asymetrie BTC. Le ranking est correct (WR=61.28% le prouve). La cosmetique des probabilites n'a aucun impact en live - seul le ranking compte pour la decision take/skip.

Si tu veux quand meme avoir des probas plus lisibles dans les logs sans toucher au modele, on peut appliquer une **recalibration post-hoc** (isotonic regression) qui mappe raw p_down vers calibrated_p_down. Ne change pas le ranking, donc WR identique. Juste les logs montrent des probas plus en ligne avec la realite.
