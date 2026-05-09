# Feature analysis — modele_updown

_Genere le 2026-05-09 15:55 UTC sur 28 features._

## Compteurs

- **KEEP_STRONG**   : 9
- **KEEP_MEDIUM**   : 10
- **WEAK_REVIEW**   : 9
- **REDUNDANT**     : 0
- **DROP_CONSTANT** : 0

## Top 15 features (par score composite)

| rank | feature | verdict | score | gain | dlogloss | |AUC-0.5| | |d| | max_corr |
|---:|:---|:---|---:|---:|---:|---:|---:|---:|
| 1 | `dist_ema_20` | KEEP_STRONG | 0.966 | 6261.64 | 0.0008 | 0.0324 | 0.090 | 0.79 |
| 2 | `return_3` | KEEP_STRONG | 0.650 | 4126.51 | 0.0005 | 0.0333 | 0.100 | 0.81 |
| 3 | `bb_position` | KEEP_STRONG | 0.514 | 2626.02 | 0.0003 | 0.0338 | 0.121 | 0.78 |
| 4 | `close_ret` | KEEP_STRONG | 0.403 | 1848.21 | 0.0002 | 0.0303 | 0.097 | 0.68 |
| 5 | `close_ret_30m` | KEEP_STRONG | 0.363 | 2624.95 | 0.0001 | 0.0244 | 0.066 | 0.81 |
| 6 | `close_ret_1h` | KEEP_STRONG | 0.318 | 1346.33 | 0.0001 | 0.0174 | 0.052 | 0.69 |
| 7 | `close_ret_5m` | KEEP_STRONG | 0.300 | 2380.10 | 0.0001 | 0.0168 | 0.061 | 0.69 |
| 8 | `obv_pressure_5m` | KEEP_STRONG | 0.284 | 814.68 | 0.0000 | 0.0249 | 0.087 | 0.55 |
| 9 | `high_ret_5m` | KEEP_STRONG | 0.247 | 1214.89 | 0.0000 | 0.0103 | 0.042 | 0.69 |
| 10 | `low_ret` | KEEP_MEDIUM | 0.244 | 1228.04 | 0.0000 | 0.0223 | 0.067 | 0.68 |
| 11 | `macd` | KEEP_MEDIUM | 0.229 | 1262.02 | -0.0000 | 0.0149 | 0.038 | 0.79 |
| 12 | `macd_hist` | KEEP_MEDIUM | 0.220 | 1320.63 | -0.0000 | 0.0190 | 0.062 | 0.73 |
| 13 | `high_ret` | KEEP_MEDIUM | 0.206 | 1065.50 | 0.0000 | 0.0178 | 0.058 | 0.68 |
| 14 | `low_ret_5m` | KEEP_MEDIUM | 0.178 | 700.74 | 0.0000 | 0.0131 | 0.048 | 0.68 |
| 15 | `low_ret_30m` | KEEP_MEDIUM | 0.177 | 878.10 | 0.0000 | 0.0175 | 0.042 | 0.69 |

## Candidats au drop

| feature | verdict | recommandation |
|:---|:---|:---|
| `volume_ratio_30m` | WEAK_REVIEW | Faible sur tous les axes (|d|=0.008) — candidat au drop. |
| `upper_wick` | WEAK_REVIEW | Faible sur tous les axes (AUC=0.501 ; |d|=0.010) — candidat au drop. |
| `body_ratio_avg_1d` | WEAK_REVIEW | Faible sur tous les axes (|d|=0.017) — candidat au drop. |
| `hour_sin` | WEAK_REVIEW | Faible sur tous les axes (|d|=0.014) — candidat au drop. |
| `body_ratio_close_30m` | WEAK_REVIEW | Faible sur tous les axes (AUC=0.502 ; |d|=0.005) — candidat au drop. |
| `close_ret_4h` | WEAK_REVIEW | Faible sur tous les axes (|d|=0.011) — candidat au drop. |
| `lower_wick` | WEAK_REVIEW | Faible sur tous les axes (AUC=0.501 ; |d|=0.009) — candidat au drop. |
| `is_weekend` | WEAK_REVIEW | Faible sur tous les axes (|d|=0.012) — candidat au drop. |
| `volatility_state` | WEAK_REVIEW | Faible sur tous les axes (AUC=0.500 ; |d|=0.002) — candidat au drop. |

## Comment dropper une feature

Édite `data.py` :
- **MTF (suffixée `_30m`/`_1h`/`_4h`/`_1d`/`_5m`)** -> retire la colonne de `MTF_KEPT_COLS[<tf>]`. Le calcul + l'écriture sont skippés automatiquement.
- **Tier 1 dérivée** (`dist_ema_20`, `return_3`, `return_96`, `hour_sin/cos`, `upper_wick/lower_wick`) -> retire le bloc correspondant dans `_add_tier1_derived`.
- **Tier 2** (`macd*`, `bb_*`, `mfi_norm`, `rv_*`) -> retire le bloc correspondant dans `_add_tier2`.
- **Base 15m** (`open_ret`, `close_ret`, etc.) -> ces colonnes sont produites par `compute_normalized_features`. Ne pas les supprimer sans réfléchir : elles servent à plusieurs autres features (Tier 1, Tier 2, MTF).

Puis relance `python data.py && python train.py` et reviens ici pour vérifier le nouveau verdict.