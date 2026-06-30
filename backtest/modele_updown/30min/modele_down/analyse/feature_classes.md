# Classement des features (modele DOWN) — ablation AUC

- Features analysees : **27**
- Baseline AUC test (seed prod) : **0.53449**
- Sigma bruit (4 seeds) : 0.00047 — bande = **0.00093** AUC
- Split : 119,546 train / 51,233 val / 17,437 test

`delta_auc = AUC(sans la feature) - AUC(baseline)`. delta<0 = la feature aide ; delta>0 = elle nuit.

| Feature | dAUC | dLogloss | gain | Classe |
|---|---:|---:|---:|---|
| ef_ret8 | -0.00132 | +0.00001 | 11.3 | MOYENNEMENT UTILE |
| ef_streak | -0.00097 | +0.00024 | 11.5 | MOYENNEMENT UTILE |
| acc_4 | -0.00096 | -0.00001 | 10.0 | MOYENNEMENT UTILE |
| dist_to_high_50 | -0.00082 | +0.00025 | 10.6 | INUTILE |
| volume_rel_30d | -0.00075 | -0.00049 | 10.3 | INUTILE |
| dist_to_low_50 | -0.00075 | -0.00019 | 10.6 | INUTILE |
| macd_hist | -0.00069 | +0.00028 | 11.2 | INUTILE |
| low_ret_30m | -0.00065 | -0.00027 | 10.6 | INUTILE |
| macd | -0.00065 | +0.00017 | 10.1 | INUTILE |
| low_ret | -0.00058 | -0.00000 | 9.3 | INUTILE |
| close_ret_30m | -0.00058 | -0.00008 | 31.6 | INUTILE |
| high_ret_5m | -0.00050 | -0.00023 | 11.9 | INUTILE |
| high_ret | -0.00050 | -0.00025 | 9.3 | INUTILE |
| mfi_norm | -0.00046 | -0.00020 | 8.7 | INUTILE |
| ef_macd | -0.00045 | -0.00013 | 9.3 | INUTILE |
| dist_ema_20 | -0.00043 | -0.00033 | 11.9 | INUTILE |
| autocorr_24 | -0.00041 | -0.00017 | 9.4 | INUTILE |
| close_ret | -0.00029 | -0.00027 | 42.8 | INUTILE |
| ef_ret2 | -0.00021 | -0.00014 | 11.7 | INUTILE |
| low_ret_5m | -0.00019 | -0.00023 | 10.4 | INUTILE |
| close_ret_1h | -0.00012 | -0.00041 | 14.5 | INUTILE |
| low_ret_1h | -0.00010 | -0.00036 | 9.7 | INUTILE |
| ef_body | -0.00008 | -0.00015 | 13.3 | INUTILE |
| obv_pressure_5m | -0.00004 | -0.00020 | 10.0 | INUTILE |
| bb_position | +0.00036 | -0.00022 | 18.6 | INUTILE |
| return_3 | +0.00053 | -0.00016 | 31.6 | INUTILE |
| close_ret_5m | +0.00057 | -0.00026 | 14.5 | INUTILE |

## Repartition

- **TRES UTILE** (0) : -
- **MOYENNEMENT UTILE** (3) : ef_ret8, ef_streak, acc_4
- **INUTILE** (24) : dist_to_high_50, volume_rel_30d, dist_to_low_50, macd_hist, low_ret_30m, macd, low_ret, close_ret_30m, high_ret_5m, high_ret, mfi_norm, ef_macd, dist_ema_20, autocorr_24, close_ret, ef_ret2, low_ret_5m, close_ret_1h, low_ret_1h, ef_body, obv_pressure_5m, bb_position, return_3, close_ret_5m
- **IMPACTANTE NEGATIVEMENT** (0) : -