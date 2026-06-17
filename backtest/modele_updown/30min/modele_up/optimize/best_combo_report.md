# best_combo — meilleure combinaison de features (modele_up)

- Pool analyse : **68** features POSITIF
- Selection sur : **AUC_VAL** | mesure deterministe {'subsample': 1.0, 'colsample_bytree': 1.0}
- Budget : 1108 entrainements de recherche + 6 de confirmation | duree 1398s
- Split : 119,546 train / 51,233 val / 17,437 test

## Verdict

- **Meilleur set (polish)** : **23 features**
- AUC stable : val **0.56196** / test **0.53840**
- AUC config PROD (moy 3 seeds) : val **0.56006** / test **0.53812**
- Set deploye (11) en PROD : test **0.53948** -> **delta = -0.00136** (PAS MIEUX)

### Features retenues

```python
FEATURE_WHITELIST = (
    "autocorr_12", "body_pos", "body_ratio_avg", "body_ratio_avg_1d", "body_signed_3",
    "close_ret_1h", "dist_ema_20", "dist_to_low_50", "ef_ret4", "efficiency_ratio_20",
    "eom_14", "high_ret_1d", "hour_cos", "hour_sin", "ret_1d_1",
    "ret_1h_6", "ret_24", "rsi_14_1h", "rsi_6", "stoch_k_14",
    "streak", "trend_slope_24", "volume_ratio_1d",
)
```

## Candidats par strategie

| strategie | n | AUC val (stable) | AUC test (stable) |
|---|---:|---:|---:|
| forward | 7 | 0.55803 | 0.53775 |
| genetic | 22 | 0.56156 | 0.53899 |
| polish **(retenu)** | 23 | 0.56196 | 0.53840 |

| pool complet | 68 | 0.55833 | 0.53642 |
| set deploye | 11 | 0.55677 | 0.54213 |
