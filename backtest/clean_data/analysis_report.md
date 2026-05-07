# Rapport d'analyse - training_data.csv

## 1. Vue d'ensemble

- **Lignes** : 290,438
- **Colonnes** : 24
- **Plage temporelle** : 2012-08-28 20:15:00 → 2026-05-05 17:15:00 (4997.9 jours)
- **Memoire** : 53.18 MB

## 2. Valeurs manquantes / infinies

- NaN totaux : **0**
- Inf totaux : **0**

## 3. Doublons / conflits

- Doublons exacts (features+label identiques) : **0**
- Doublons sur features (label peut differer) : **0**
- **Conflits** (memes features → labels differents) : **0 groupes**, **0 lignes**
- Doublons sur grain (timestamp+seq_len) : **0**

## 4. Distribution du label

- WR baseline global : **53.48%**
- Balance : `{1: 0.5347991653984672, 0: 0.46520083460153283}`

**WR par seq_len**:
| seq_len | n | WR |
|---|---|---|
| 1 | 164,776 | 52.663% |
| 2 | 76,276 | 54.201% |
| 3 | 34,172 | 55.133% |
| 4 | 15,214 | 55.002% |

**WR par seq_dir** (1=up, 0=down):
| seq_dir | n | WR |
|---|---|---|
| 0 | 137,014 | 54.754% |
| 1 | 153,424 | 52.342% |

## 5. Coherence metier

- `body_ratio_last_below_min` : 0
- `body_ratio_last_min_seen` : 0.6
- `candle_anatomy_violations_gt_1pct` : 0
- `candle_anatomy_max_abs_dev` : 0.0
- `seq_len_out_of_range` : 0
- `seq_len_unique` : [1, 2, 3, 4]
- `seq_dir_out_of_range` : 0
- `is_weekend_out_of_range` : 0
- `obv_vs_ema_out_of_range` : 0
- `volatility_state_out_of_range` : 0
- `hour_utc_out_of_range` : 0
- `day_of_week_out_of_range` : 0
- `is_weekend_inconsistent` : 0

## 6. Outliers (par feature continue)

| feature | %hors IQR k=1.5 | %hors IQR k=3 | %hors [p0.1, p99.9] | max |z| | p0.1 | p99.9 |
|---|---|---|---|---|---|---|
| volume_ratio_last | 8.647% | 4.153% | 0.2% | 154.49 | 0.005463 | 30.528313 |
| volume_ratio_avg | 7.765% | 3.405% | 0.2% | 190.13 | 0.009025 | 24.920496 |
| move_pct | 8.125% | 2.591% | 0.2% | 75.8 | -7.86076 | 7.027994 |
| move_atr | 5.881% | 1.7% | 0.2% | 12.82 | 0.00981 | 6.78564 |
| rsi_change_5 | 1.376% | 0.025% | 0.2% | 6.34 | -39.741199 | 38.166581 |
| obv_slope_10 | 23.605% | 15.489% | 0.2% | 203.22 | -0.216178 | 0.221563 |
| atr_ratio | 3.333% | 0.505% | 0.2% | 18.89 | 0.352864 | 2.619093 |
| dist_ema20 | 12.265% | 4.575% | 0.2% | 43.48 | -9.113616 | 7.218436 |
| dist_ema50 | 11.858% | 4.291% | 0.2% | 33.31 | -12.598765 | 9.67224 |
| dist_ema200 | 9.811% | 3.131% | 0.2% | 21.23 | -22.067838 | 17.632924 |
| rsi | 1.946% | 0.0% | 0.199% | 4.02 | 11.492621 | 89.90481 |
| body_ratio_avg | 0.06% | 0.0% | 0.1% | 3.27 | 0.269075 | 1.0 |
| upper_wick | 0.0% | 0.0% | 0.1% | 2.82 | 0.0 | 0.398917 |
| lower_wick | 0.0% | 0.0% | 0.1% | 2.63 | 0.0 | 0.399894 |
| body_ratio_last | 0.0% | 0.0% | 0.0% | 1.58 | 0.6 | 1.0 |

## 7. Features quasi-constantes / faible variance

- `seq_len` : 4 valeurs uniques, top=1.0 (56.734%)
- `seq_dir` : 2 valeurs uniques, top=1.0 (52.825%)
- `obv_vs_ema` : 2 valeurs uniques, top=1.0 (51.028%)
- `volatility_state` : 3 valeurs uniques, top=1.0 (83.369%)
- `is_weekend` : 2 valeurs uniques, top=0.0 (70.821%)

## 8. Correlations

**Paires |r| ≥ 0.85** (2 paires) :
- `volume_ratio_last` ↔ `volume_ratio_avg` : |r|=0.9231
- `dist_ema20` ↔ `dist_ema50` : |r|=0.9117

**Top features Spearman vs label** :
- `move_atr` : 0.0413
- `seq_dir` : -0.0241
- `body_ratio_avg` : -0.0225
- `move_pct` : -0.0214
- `body_ratio_last` : -0.021
- `volume_ratio_last` : 0.0209
- `volume_ratio_avg` : 0.0209
- `lower_wick` : 0.0203
- `upper_wick` : 0.0196
- `seq_len` : 0.0195
- `rsi_change_5` : -0.0182
- `rsi` : -0.0156
- `dist_ema20` : -0.0144
- `dist_ema50` : -0.0122
- `dist_ema200` : -0.0119

## 9. Mutual information avec label (top 15)

- `atr_ratio` : 0.03974
- `dist_ema20` : 0.03855
- `dist_ema200` : 0.03791
- `obv_slope_10` : 0.03718
- `rsi_change_5` : 0.03705
- `volume_ratio_last` : 0.03612
- `rsi` : 0.0356
- `dist_ema50` : 0.03343
- `body_ratio_last` : 0.02633
- `lower_wick` : 0.02242
- `upper_wick` : 0.01813
- `seq_len` : 0.00717
- `move_pct` : 0.0051
- `obv_vs_ema` : 0.00443
- `seq_dir` : 0.00398

## 10. Concept drift (1ere moitie vs 2e moitie)

- WR 1ere moitie : 52.023%
- WR 2e moitie : 54.936%

**Top 10 features qui drift le plus (KS test)** :
| feature | KS stat | p-value | mean 1ere moitie | mean 2e moitie |
|---|---|---|---|---|
| obv_slope_10 | 0.2528 | 0.0 | 0.0001 | 0.0 |
| upper_wick | 0.1776 | 0.0 | 0.081 | 0.108 |
| body_ratio_last | 0.1746 | 0.0 | 0.8262 | 0.7784 |
| lower_wick | 0.1637 | 0.0 | 0.0927 | 0.1136 |
| body_ratio_avg | 0.1217 | 0.0 | 0.7418 | 0.705 |
| dist_ema200 | 0.0733 | 0.0 | 0.3336 | 0.0908 |
| move_pct | 0.0699 | 0.0 | 0.0286 | 0.0167 |
| move_atr | 0.0682 | 0.0 | 1.0779 | 1.1781 |
| dist_ema50 | 0.0603 | 0.0 | 0.0899 | 0.0331 |
| dist_ema20 | 0.0504 | 0.0 | 0.0423 | 0.0208 |

## 11. Densite temporelle

- Median gap entre lignes seq_len=1 : 30.0 min
- Max gap : 6495.0 min
- Nb gaps > 1h : 28878

**Lignes par annee** :
- 2012 : 1,100
- 2013 : 22,542
- 2014 : 22,060
- 2015 : 21,103
- 2016 : 21,978
- 2017 : 23,481
- 2018 : 18,200
- 2019 : 16,983
- 2020 : 17,912
- 2021 : 20,143
- 2022 : 21,442
- 2023 : 24,465
- 2024 : 24,879
- 2025 : 26,181
- 2026 : 7,969
