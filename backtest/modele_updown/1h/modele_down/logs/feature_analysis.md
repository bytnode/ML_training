# Feature analysis — modele_updown

_Genere le 2026-05-10 16:08 UTC sur 25 features._

## Compteurs

- **KEEP_STRONG**   : 8
- **KEEP_MEDIUM**   : 9
- **WEAK_REVIEW**   : 8
- **REDUNDANT**     : 0
- **DROP_CONSTANT** : 0

## Top 15 features (par score composite)

| rank | feature | verdict | score | gain | dlogloss | |AUC-0.5| | |d| | max_corr |
|---:|:---|:---|---:|---:|---:|---:|---:|---:|
| 1 | `dist_ema_20` | KEEP_STRONG | 0.893 | 5280.24 | 0.0005 | 0.0324 | 0.090 | 0.79 |
| 2 | `return_3` | KEEP_STRONG | 0.802 | 6952.87 | 0.0004 | 0.0333 | 0.100 | 0.81 |
| 3 | `bb_position` | KEEP_STRONG | 0.501 | 2421.31 | 0.0002 | 0.0338 | 0.121 | 0.78 |
| 4 | `close_ret` | KEEP_STRONG | 0.470 | 2096.56 | 0.0002 | 0.0303 | 0.097 | 0.68 |
| 5 | `close_ret_30m` | KEEP_STRONG | 0.376 | 2688.12 | 0.0001 | 0.0244 | 0.066 | 0.81 |
| 6 | `close_ret_5m` | KEEP_STRONG | 0.340 | 2029.49 | 0.0002 | 0.0168 | 0.061 | 0.69 |
| 7 | `close_ret_1h` | KEEP_STRONG | 0.301 | 1581.48 | 0.0001 | 0.0174 | 0.052 | 0.69 |
| 8 | `dist_to_high_50` | KEEP_STRONG | 0.296 | 1469.81 | -0.0000 | 0.0208 | 0.049 | 0.73 |
| 9 | `obv_pressure_5m` | KEEP_MEDIUM | 0.272 | 640.31 | 0.0000 | 0.0249 | 0.087 | 0.55 |
| 10 | `dist_to_low_50` | KEEP_MEDIUM | 0.245 | 1188.61 | 0.0000 | 0.0230 | 0.056 | 0.61 |
| 11 | `low_ret` | KEEP_MEDIUM | 0.229 | 871.00 | 0.0000 | 0.0223 | 0.067 | 0.68 |
| 12 | `macd_hist` | KEEP_MEDIUM | 0.228 | 1441.88 | 0.0000 | 0.0190 | 0.062 | 0.73 |
| 13 | `macd` | KEEP_MEDIUM | 0.214 | 976.49 | 0.0000 | 0.0149 | 0.038 | 0.79 |
| 14 | `high_ret_5m` | KEEP_MEDIUM | 0.196 | 1284.91 | 0.0000 | 0.0103 | 0.042 | 0.69 |
| 15 | `mfi_norm` | KEEP_MEDIUM | 0.176 | 551.20 | 0.0000 | 0.0178 | 0.062 | 0.71 |

## Candidats au drop

| feature | verdict | recommandation |
|:---|:---|:---|
| `low_ret_30m` | WEAK_REVIEW | Faible sur tous les axes (|d|=0.042) — candidat au drop. |
| `volume_ratio_30m` | WEAK_REVIEW | Faible sur tous les axes (|d|=0.008) — candidat au drop. |
| `acc_4` | WEAK_REVIEW | Faible sur tous les axes (|d|=0.033) — candidat au drop. |
| `low_ret_1h` | WEAK_REVIEW | Faible sur tous les axes (|d|=0.029) — candidat au drop. |
| `autocorr_24` | WEAK_REVIEW | Faible sur tous les axes (AUC=0.501 ; |d|=0.001) — candidat au drop. |
| `hour_sin` | WEAK_REVIEW | Faible sur tous les axes (|d|=0.014) — candidat au drop. |
| `body_ratio_close_30m` | WEAK_REVIEW | Faible sur tous les axes (AUC=0.502 ; |d|=0.005) — candidat au drop. |
| `is_weekend` | WEAK_REVIEW | Faible sur tous les axes (|d|=0.012) — candidat au drop. |

## Comment dropper une feature

Édite `data.py` :
- **MTF (suffixée `_30m`/`_1h`/`_4h`/`_1d`/`_5m`)** -> retire la colonne de `MTF_KEPT_COLS[<tf>]`. Le calcul + l'écriture sont skippés automatiquement.
- **Tier 1 dérivée** (`dist_ema_20`, `return_3`, `return_96`, `hour_sin/cos`, `upper_wick/lower_wick`) -> retire le bloc correspondant dans `_add_tier1_derived`.
- **Tier 2** (`macd*`, `bb_*`, `mfi_norm`, `rv_*`) -> retire le bloc correspondant dans `_add_tier2`.
- **Base 15m** (`open_ret`, `close_ret`, etc.) -> ces colonnes sont produites par `compute_normalized_features`. Ne pas les supprimer sans réfléchir : elles servent à plusieurs autres features (Tier 1, Tier 2, MTF).

Puis relance `python data.py && python train.py` et reviens ici pour vérifier le nouveau verdict.