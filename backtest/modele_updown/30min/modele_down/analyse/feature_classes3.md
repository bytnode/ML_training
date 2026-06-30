# Classement d'impact des features (3 classes, modele deterministe)

- Features analysees : **11**
- AUC modele complet : **0.54259** (sigma 2 seeds = 0.000000, deterministe)
- Config de mesure : `{'subsample': 1.0, 'colsample_bytree': 1.0}` | zone morte DEAD = 0.0003
- Split : 119,546 train / 51,233 val / 17,437 test

## Resume

- **IMPACT POSITIF (utiles)** : 11
- **IMPACT NUL** : 0 (dont 0 redondantes-avec-signal, 0 inertes)
- **IMPACT NEGATIF (nuisibles)** : 0

- **Utiles** (positif + redondant-avec-signal) : **11**
- **Inutiles** (inertes) : **0**
- **Mauvaises** (nuisibles) : **0**

| feature | delta_ablation | standalone_auc | power | classe |
|---|---:|---:|---:|---|
| body_pos | -0.00655 | 0.5302 | +0.0302 | POSITIF |
| ef_ret1 | -0.00255 | 0.5231 | +0.0231 | POSITIF |
| ef_ret4 | -0.00243 | 0.5271 | +0.0271 | POSITIF |
| rsi_6 | -0.00239 | 0.5316 | +0.0316 | POSITIF |
| stoch_k_14 | -0.00223 | 0.5296 | +0.0296 | POSITIF |
| vol_z_20 | -0.00157 | 0.4954 | -0.0046 | POSITIF |
| autocorr_12 | -0.00124 | 0.5008 | +0.0008 | POSITIF |
| hour_cos | -0.00081 | 0.4999 | -0.0001 | POSITIF |
| ret_1d_1 | -0.00060 | 0.5007 | +0.0007 | POSITIF |
| ef_upwick | -0.00054 | 0.4975 | -0.0025 | POSITIF |
| body_ratio_avg | -0.00039 | 0.5022 | +0.0022 | POSITIF |