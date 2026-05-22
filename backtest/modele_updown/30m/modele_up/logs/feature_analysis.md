# Feature analysis - modele_up (30m)

**Date :** 2026-05-14T22:57:05+00:00  
**Multi-seed :** 1  
**Ablation :** enabled  
**Calibration :** `asym_wknd_excl_hour_boost`  
**Duration :** 64.0s  
**Features evaluated :** 51 / 51

## Baseline (avec toutes les features)

- best_iter    : 32
- val_logloss  : 0.689828
- test_logloss : 0.690062
- taken        : 752
- WR           : 63.43%
- EV           : +101.00

## Tableau recapitulatif (trie par |delta_EV| decroissant)

| Rang | Feature | dEV | dTaken | dWR | dVal_ll | dTest_ll | perm_global | perm_taken | gain_rank | max_corr | cohens_d | MI | NaN% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `return_3` | +28.00 | +112 | +2.02pt | +0.000362 | -0.000388 | +0.000780 | +0.006887 | 1/51 | 0.755 | -0.126 | 0.00366 | 0.00% |
| 2 | `obv_pressure` | +27.00 | +208 | -0.17pt | +0.000048 | -0.000146 | -0.000007 | +0.000783 | 31/51 | 0.662 | -0.096 | 0.00118 | 0.00% |
| 3 | `high_ret` | +24.00 | +186 | -0.17pt | +0.000065 | -0.000090 | +0.000045 | +0.000133 | 15/51 | 0.670 | -0.055 | 0.00093 | 0.00% |
| 4 | `roc_5` | +24.00 | +128 | +1.09pt | +0.000015 | -0.000185 | +0.000183 | +0.004262 | 3/51 | 0.779 | -0.122 | 0.00318 | 0.00% |
| 5 | `close_ret_30m` | +23.00 | +94 | +1.58pt | +0.000073 | -0.000267 | +0.001395 | +0.004280 | 2/51 | 0.666 | -0.119 | 0.00360 | 0.00% |
| 6 | `dist_ema_100` | +21.00 | +116 | +0.85pt | -0.000141 | -0.000033 | +0.000068 | +0.000316 | 17/51 | 0.944 | -0.026 | 0.00014 | 0.00% |
| 7 | `obv_pressure_1d` | +19.50 | +183 | -0.89pt | -0.000062 | -0.000323 | -0.000077 | +0.000347 | 13/51 | 0.471 | +0.033 | 0.00022 | 0.00% |
| 8 | `roc_20` | +18.50 | +123 | +0.31pt | +0.000095 | -0.000357 | -0.000009 | -0.000221 | 26/51 | 0.868 | -0.038 | 0.00053 | 0.00% |
| 9 | `body_ratio_avg_30m` | +17.50 | +43 | +1.65pt | -0.000092 | -0.000040 | +0.000028 | +0.000295 | 46/51 | 0.403 | -0.002 | 0.00007 | 0.33% |
| 10 | `return_96` | +16.00 | +88 | +0.63pt | -0.000053 | -0.000054 | +0.000008 | +0.000262 | 16/51 | 0.857 | -0.000 | 0.00011 | 0.00% |
| 11 | `stoch_k_14` | +15.00 | -80 | +3.09pt | -0.000026 | -0.000284 | +0.001042 | +0.007152 | 4/51 | 0.875 | -0.164 | 0.00369 | 0.00% |
| 12 | `body_ratio_close` | +15.00 | +72 | +0.78pt | +0.000088 | +0.000176 | -0.000036 | -0.000122 | 35/51 | 0.628 | +0.007 | 0.00009 | 0.00% |
| 13 | `dist_high_20` | +14.50 | +131 | -0.50pt | +0.000058 | -0.000360 | +0.000313 | +0.001475 | 5/51 | 0.805 | -0.069 | 0.00215 | 0.00% |
| 14 | `roc_10` | +14.00 | +24 | +1.48pt | -0.000217 | +0.000010 | +0.000069 | +0.000440 | 7/51 | 0.884 | -0.087 | 0.00181 | 0.00% |
| 15 | `low_ret_30m` | +13.50 | +143 | -0.94pt | -0.000067 | +0.000043 | -0.000004 | -0.000036 | 28/51 | 0.713 | -0.084 | 0.00175 | 0.00% |
| 16 | `dist_low_20` | +13.50 | +135 | -0.75pt | +0.000154 | -0.000248 | +0.000332 | +0.000679 | 6/51 | 0.763 | -0.056 | 0.00210 | 0.00% |
| 17 | `hour_cos` | +13.00 | +86 | +0.22pt | -0.000249 | -0.000087 | +0.000119 | +0.000582 | 10/51 | 0.225 | -0.010 | 0.00030 | 0.00% |
| 18 | `macd_hist` | +12.00 | +78 | +0.23pt | +0.000029 | +0.000120 | +0.000025 | +0.000253 | 19/51 | 0.841 | -0.076 | 0.00088 | 0.00% |
| 19 | `volume_ratio_30m` | +11.50 | +41 | +0.84pt | -0.000152 | -0.000012 | +0.000076 | +0.000019 | 30/51 | 0.724 | +0.013 | 0.00014 | 0.00% |
| 20 | `dist_high_50` | +11.00 | +60 | +0.43pt | +0.000045 | -0.000019 | -0.000040 | +0.001114 | 14/51 | 0.805 | -0.043 | 0.00108 | 0.00% |
| 21 | `rv_100` | +10.50 | +117 | -0.82pt | +0.000009 | -0.000063 | +0.000005 | +0.000205 | 34/51 | 0.945 | -0.005 | 0.00010 | 0.00% |
| 22 | `body_ratio_avg_1d` | +10.50 | +45 | +0.63pt | -0.000038 | -0.000002 | -0.000012 | +0.000163 | 33/51 | 0.272 | +0.009 | 0.00010 | 0.00% |
| 23 | `open_ret_1d` | +9.50 | +77 | -0.12pt | -0.000033 | -0.000036 | -0.000004 | -0.000013 | 47/51 | 0.045 | -0.006 | 0.00012 | 0.00% |
| 24 | `atr_pct_1d` | +9.50 | +45 | +0.49pt | -0.000075 | +0.000001 | +0.000088 | +0.000040 | 9/51 | 0.843 | -0.016 | 0.00016 | 0.00% |
| 25 | `high_ret_30m` | +9.00 | -20 | +1.51pt | -0.000050 | +0.000004 | -0.000009 | -0.000053 | 50/51 | 0.670 | -0.057 | 0.00110 | 0.00% |
| 26 | `mfi_norm` | +8.00 | +2 | +1.03pt | +0.000011 | -0.000017 | -0.000009 | -0.000263 | 44/51 | 0.752 | -0.061 | 0.00060 | 0.00% |
| 27 | `open_ret_30m` | +8.00 | +42 | +0.33pt | -0.000001 | -0.000131 | +0.000034 | -0.000031 | 49/51 | 0.112 | -0.003 | 0.00009 | 0.00% |
| 28 | `upper_wick` | +8.00 | +4 | +1.00pt | -0.000139 | -0.000011 | +0.000068 | +0.000221 | 32/51 | 0.553 | +0.031 | 0.00029 | 0.00% |
| 29 | `volume_ratio_4h` | +7.50 | +101 | -0.93pt | +0.000104 | +0.000018 | +0.000000 | +0.000040 | 22/51 | 0.443 | +0.013 | 0.00009 | 0.00% |
| 30 | `volume_ratio_1d` | +7.50 | +59 | -0.06pt | +0.000063 | -0.000034 | +0.000012 | +0.000169 | 38/51 | 0.458 | +0.004 | 0.00007 | 0.00% |
| 31 | `rsi_14` | +7.50 | +69 | -0.26pt | -0.000019 | +0.000089 | +0.000005 | -0.000072 | 18/51 | 0.878 | -0.101 | 0.00161 | 0.00% |
| 32 | `body_ratio_avg` | +7.00 | +34 | +0.34pt | -0.000049 | +0.000014 | -0.000017 | +0.000050 | 41/51 | 0.403 | +0.007 | 0.00006 | 0.09% |
| 33 | `close_ret_4h` | +7.00 | -34 | +1.47pt | -0.000007 | +0.000095 | +0.000306 | +0.000239 | 8/51 | 0.700 | -0.072 | 0.00147 | 0.00% |
| 34 | `rv_ratio` | +6.50 | +47 | +0.03pt | +0.000048 | -0.000009 | +0.000022 | -0.000053 | 48/51 | 0.443 | +0.011 | 0.00007 | 0.00% |
| 35 | `hour_sin` | +6.00 | -46 | +1.53pt | +0.000030 | -0.000124 | +0.000009 | +0.000001 | 21/51 | 0.246 | -0.005 | 0.00025 | 0.00% |
| 36 | `low_ret` | +6.00 | +26 | +0.35pt | -0.000046 | +0.000042 | +0.000026 | +0.000025 | 12/51 | 0.713 | -0.074 | 0.00131 | 0.00% |
| 37 | `atr_pct_30m` | +6.00 | -4 | +0.86pt | +0.000033 | -0.000039 | -0.000056 | +0.000007 | 36/51 | 0.882 | +0.013 | 0.00008 | 0.00% |
| 38 | `body_ratio_close_1d` | +5.50 | +3 | +0.68pt | +0.000026 | -0.000052 | +0.000003 | +0.000180 | 43/51 | 0.272 | +0.002 | 0.00008 | 0.00% |
| 39 | `high_ret_1d` | +5.00 | -72 | +1.78pt | -0.000075 | +0.000017 | -0.000011 | -0.000008 | 37/51 | 0.663 | +0.005 | 0.00006 | 0.00% |
| 40 | `dist_low_50` | +5.00 | -2 | +0.70pt | -0.000106 | -0.000091 | +0.000034 | -0.000009 | 20/51 | 0.763 | -0.036 | 0.00099 | 0.00% |
| 41 | `low_ret_1d` | +5.00 | -16 | +0.93pt | -0.000152 | -0.000107 | +0.000011 | +0.000111 | 23/51 | 0.683 | +0.005 | 0.00009 | 0.00% |
| 42 | `atr_pct_4h` | +4.50 | +31 | +0.05pt | -0.000050 | -0.000093 | -0.000023 | +0.000078 | 39/51 | 0.945 | -0.003 | 0.00008 | 0.00% |
| 43 | `obv_pressure_4h` | +4.00 | +58 | -0.55pt | -0.000080 | +0.000006 | -0.000003 | +0.000066 | 45/51 | 0.642 | -0.012 | 0.00010 | 0.00% |
| 44 | `body_ratio_avg_4h` | +3.50 | +41 | -0.28pt | -0.000046 | -0.000011 | +0.000020 | -0.000001 | 51/51 | 0.233 | +0.013 | 0.00008 | 0.06% |
| 45 | `bb_position` | +3.50 | -15 | +0.72pt | -0.000034 | -0.000023 | +0.000051 | +0.000040 | 24/51 | 0.938 | -0.138 | 0.00255 | 0.00% |
| 46 | `dist_ema_50` | +3.50 | +57 | -0.60pt | +0.000009 | +0.000004 | +0.000043 | +0.000092 | 29/51 | 0.944 | -0.053 | 0.00059 | 0.00% |
| 47 | `price_in_range_20` | +3.50 | +47 | -0.40pt | -0.000011 | -0.000100 | +0.000042 | -0.000001 | 40/51 | 0.938 | -0.134 | 0.00277 | 0.00% |
| 48 | `price_in_range_50` | +3.00 | -30 | +0.90pt | -0.000043 | -0.000035 | +0.000002 | +0.000042 | 25/51 | 0.878 | -0.079 | 0.00144 | 0.00% |
| 49 | `dist_ema_20` | +2.50 | -31 | +0.85pt | +0.000036 | -0.000076 | +0.000041 | +0.000307 | 11/51 | 0.905 | -0.095 | 0.00202 | 0.00% |
| 50 | `volume_ratio` | +2.00 | -10 | +0.44pt | -0.000089 | +0.000030 | +0.000026 | -0.000261 | 27/51 | 0.724 | +0.019 | 0.00012 | 0.00% |
| 51 | `lower_wick` | +1.50 | -59 | +1.16pt | -0.000018 | +0.000038 | -0.000005 | -0.000096 | 42/51 | 0.628 | -0.037 | 0.00026 | 0.00% |

## Definitions des metriques

- **dEV** = `baseline_EV - ablated_EV`. Mesure en points combien la feature contribue a l'EV reel quand on retrain sans elle.
- **dTaken** = `baseline_taken - ablated_taken`. Variation du nombre de paris pris.
- **dWR** = `baseline_WR - ablated_WR`. Variation de la WR sur les paris pris.
- **dVal_ll / dTest_ll** = variation du logloss val/test entre baseline et ablated. Negatif = la feature aide.
- **perm_global** = delta logloss en permutant la feature sur tout le test set. Mesure d'impact global.
- **perm_taken** = delta logloss en permutant uniquement sur le subset des paris pris. Mesure l'impact dans la queue de distribution (ou se trouvent les bets).
- **gain_rank** = rang de l'importance native XGBoost (gain) parmi toutes les features.
- **max_corr** = correlation absolue maximale avec une autre feature. Au-dela de 0.85 indique une forte redondance.
- **cohens_d** = separation des distributions class 0 vs class 1 (effect size). Convention : >0.2 small, >0.5 medium, >0.8 large.
- **MI** = mutual information binnee entre la feature et le label.
- **NaN%** = proportion de valeurs manquantes (warmup + edge cases).

## Detail par feature

### 1. `return_3`

**Distribution**
- mean = 0.00029, std = 0.01302, skew = -0.193
- range = [-0.2913, +0.2031], NaN = 0.00%

**Relation avec le label (univariate)**
- Cohen's d   = -0.1264  (small)
- Mutual info = 0.00366

**Top 3 correlations avec autres features**
- `roc_5` : r = +0.755
- `dist_ema_20` : r = +0.664
- `bb_position` : r = +0.568

**Importance dans le baseline**
- gain XGBoost = 0.12712  (rang 1/51)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = +0.000780
- delta logloss sur le subset des paris pris = +0.006887

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.689465   (baseline = 0.689828, delta = +0.000362)
- test_logloss  : 0.690449   (baseline = 0.690062, delta = -0.000388)
- taken         : 640  (baseline = 752, delta = +112)
- WR            : 61.41%  (baseline = 63.43%, delta = +2.02pt)
- EV            : +73.00  (baseline = +101.00, **delta = +28.00**)

---

### 2. `obv_pressure`

**Distribution**
- mean = 0.01110, std = 0.36668, skew = +0.002
- range = [-1.0000, +1.0000], NaN = 0.00%

**Relation avec le label (univariate)**
- Cohen's d   = -0.0962  (small)
- Mutual info = 0.00118

**Top 3 correlations avec autres features**
- `bb_position` : r = +0.662
- `rsi_14` : r = +0.631
- `roc_10` : r = +0.629

**Importance dans le baseline**
- gain XGBoost = 0.01543  (rang 31/51)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = -0.000007
- delta logloss sur le subset des paris pris = +0.000783

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.689779   (baseline = 0.689828, delta = +0.000048)
- test_logloss  : 0.690208   (baseline = 0.690062, delta = -0.000146)
- taken         : 544  (baseline = 752, delta = +208)
- WR            : 63.60%  (baseline = 63.43%, delta = -0.17pt)
- EV            : +74.00  (baseline = +101.00, **delta = +27.00**)

---

### 3. `high_ret`

**Distribution**
- mean = 0.00461, std = 0.00638, skew = +6.293
- range = [-0.0051, +0.3197], NaN = 0.00%

**Relation avec le label (univariate)**
- Cohen's d   = -0.0549  (small)
- Mutual info = 0.00093

**Top 3 correlations avec autres features**
- `high_ret_30m` : r = +0.670
- `atr_pct_30m` : r = +0.510
- `dist_low_20` : r = +0.445

**Importance dans le baseline**
- gain XGBoost = 0.01802  (rang 15/51)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = +0.000045
- delta logloss sur le subset des paris pris = +0.000133

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.689763   (baseline = 0.689828, delta = +0.000065)
- test_logloss  : 0.690151   (baseline = 0.690062, delta = -0.000090)
- taken         : 566  (baseline = 752, delta = +186)
- WR            : 63.60%  (baseline = 63.43%, delta = -0.17pt)
- EV            : +77.00  (baseline = +101.00, **delta = +24.00**)

---

### 4. `roc_5`

**Distribution**
- mean = 0.00048, std = 0.01657, skew = -0.048
- range = [-0.2805, +0.2759], NaN = 0.00%

**Relation avec le label (univariate)**
- Cohen's d   = -0.1216  (small)
- Mutual info = 0.00318

**Top 3 correlations avec autres features**
- `dist_ema_20` : r = +0.779
- `return_3` : r = +0.755
- `close_ret_4h` : r = +0.700

**Importance dans le baseline**
- gain XGBoost = 0.02955  (rang 3/51)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = +0.000183
- delta logloss sur le subset des paris pris = +0.004262

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.689813   (baseline = 0.689828, delta = +0.000015)
- test_logloss  : 0.690247   (baseline = 0.690062, delta = -0.000185)
- taken         : 624  (baseline = 752, delta = +128)
- WR            : 62.34%  (baseline = 63.43%, delta = +1.09pt)
- EV            : +77.00  (baseline = +101.00, **delta = +24.00**)

---

### 5. `close_ret_30m`

**Distribution**
- mean = 0.00006, std = 0.00553, skew = +0.211
- range = [-0.1201, +0.1359], NaN = 0.00%

**Relation avec le label (univariate)**
- Cohen's d   = -0.1190  (small)
- Mutual info = 0.00360

**Top 3 correlations avec autres features**
- `high_ret_30m` : r = +0.666
- `low_ret_30m` : r = +0.633
- `return_3` : r = +0.382

**Importance dans le baseline**
- gain XGBoost = 0.05416  (rang 2/51)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = +0.001395
- delta logloss sur le subset des paris pris = +0.004280

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.689755   (baseline = 0.689828, delta = +0.000073)
- test_logloss  : 0.690329   (baseline = 0.690062, delta = -0.000267)
- taken         : 658  (baseline = 752, delta = +94)
- WR            : 61.85%  (baseline = 63.43%, delta = +1.58pt)
- EV            : +78.00  (baseline = +101.00, **delta = +23.00**)

---

### 6. `dist_ema_100`

**Distribution**
- mean = 0.00207, std = 0.03716, skew = -1.173
- range = [-0.7926, +0.2425], NaN = 0.00%

**Relation avec le label (univariate)**
- Cohen's d   = -0.0258  (small)
- Mutual info = 0.00014

**Top 3 correlations avec autres features**
- `dist_ema_50` : r = +0.944 (>0.85 = forte redondance)
- `return_96` : r = +0.857 (>0.85 = forte redondance)
- `dist_ema_20` : r = +0.749

**Importance dans le baseline**
- gain XGBoost = 0.01741  (rang 17/51)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = +0.000068
- delta logloss sur le subset des paris pris = +0.000316

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.689968   (baseline = 0.689828, delta = -0.000141)
- test_logloss  : 0.690095   (baseline = 0.690062, delta = -0.000033)
- taken         : 636  (baseline = 752, delta = +116)
- WR            : 62.58%  (baseline = 63.43%, delta = +0.85pt)
- EV            : +80.00  (baseline = +101.00, **delta = +21.00**)

---

### 7. `obv_pressure_1d`

**Distribution**
- mean = 0.04337, std = 0.34222, skew = +0.074
- range = [-0.9368, +1.0000], NaN = 0.00%

**Relation avec le label (univariate)**
- Cohen's d   = +0.0328  (small)
- Mutual info = 0.00022

**Top 3 correlations avec autres features**
- `return_96` : r = +0.471
- `dist_ema_100` : r = +0.436
- `dist_ema_50` : r = +0.277

**Importance dans le baseline**
- gain XGBoost = 0.01827  (rang 13/51)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = -0.000077
- delta logloss sur le subset des paris pris = +0.000347

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.689890   (baseline = 0.689828, delta = -0.000062)
- test_logloss  : 0.690385   (baseline = 0.690062, delta = -0.000323)
- taken         : 569  (baseline = 752, delta = +183)
- WR            : 64.32%  (baseline = 63.43%, delta = -0.89pt)
- EV            : +81.50  (baseline = +101.00, **delta = +19.50**)

---

### 8. `roc_20`

**Distribution**
- mean = 0.00193, std = 0.03307, skew = -0.012
- range = [-0.4626, +0.3573], NaN = 0.00%

**Relation avec le label (univariate)**
- Cohen's d   = -0.0380  (small)
- Mutual info = 0.00053

**Top 3 correlations avec autres features**
- `dist_ema_20` : r = +0.868 (>0.85 = forte redondance)
- `dist_ema_50` : r = +0.866 (>0.85 = forte redondance)
- `rsi_14` : r = +0.771

**Importance dans le baseline**
- gain XGBoost = 0.01616  (rang 26/51)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = -0.000009
- delta logloss sur le subset des paris pris = -0.000221

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.689733   (baseline = 0.689828, delta = +0.000095)
- test_logloss  : 0.690419   (baseline = 0.690062, delta = -0.000357)
- taken         : 629  (baseline = 752, delta = +123)
- WR            : 63.12%  (baseline = 63.43%, delta = +0.31pt)
- EV            : +82.50  (baseline = +101.00, **delta = +18.50**)

---

### 9. `body_ratio_avg_30m`

**Distribution**
- mean = 0.45146, std = 0.07789, skew = +0.112
- range = [+0.1409, +0.8141], NaN = 0.33%

**Relation avec le label (univariate)**
- Cohen's d   = -0.0015  (small)
- Mutual info = 0.00007

**Top 3 correlations avec autres features**
- `body_ratio_avg` : r = +0.403
- `body_ratio_close` : r = +0.186
- `body_ratio_avg_4h` : r = +0.177

**Importance dans le baseline**
- gain XGBoost = 0.01417  (rang 46/51)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = +0.000028
- delta logloss sur le subset des paris pris = +0.000295

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.689920   (baseline = 0.689828, delta = -0.000092)
- test_logloss  : 0.690101   (baseline = 0.690062, delta = -0.000040)
- taken         : 709  (baseline = 752, delta = +43)
- WR            : 61.78%  (baseline = 63.43%, delta = +1.65pt)
- EV            : +83.50  (baseline = +101.00, **delta = +17.50**)

---

### 10. `return_96`

**Distribution**
- mean = 0.00929, std = 0.07223, skew = +0.233
- range = [-0.4921, +0.5407], NaN = 0.00%

**Relation avec le label (univariate)**
- Cohen's d   = -0.0005  (small)
- Mutual info = 0.00011

**Top 3 correlations avec autres features**
- `dist_ema_100` : r = +0.857 (>0.85 = forte redondance)
- `dist_ema_50` : r = +0.697
- `dist_high_50` : r = +0.491

**Importance dans le baseline**
- gain XGBoost = 0.01781  (rang 16/51)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = +0.000008
- delta logloss sur le subset des paris pris = +0.000262

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.689881   (baseline = 0.689828, delta = -0.000053)
- test_logloss  : 0.690115   (baseline = 0.690062, delta = -0.000054)
- taken         : 664  (baseline = 752, delta = +88)
- WR            : 62.80%  (baseline = 63.43%, delta = +0.63pt)
- EV            : +85.00  (baseline = +101.00, **delta = +16.00**)

---

### 11. `stoch_k_14`

**Distribution**
- mean = 0.08556, std = 0.52570, skew = -0.130
- range = [-1.0000, +1.0000], NaN = 0.00%

**Relation avec le label (univariate)**
- Cohen's d   = -0.1643  (small)
- Mutual info = 0.00369

**Top 3 correlations avec autres features**
- `bb_position` : r = +0.875 (>0.85 = forte redondance)
- `price_in_range_20` : r = +0.874 (>0.85 = forte redondance)
- `rsi_14` : r = +0.766

**Importance dans le baseline**
- gain XGBoost = 0.02326  (rang 4/51)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = +0.001042
- delta logloss sur le subset des paris pris = +0.007152

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.689854   (baseline = 0.689828, delta = -0.000026)
- test_logloss  : 0.690346   (baseline = 0.690062, delta = -0.000284)
- taken         : 832  (baseline = 752, delta = -80)
- WR            : 60.34%  (baseline = 63.43%, delta = +3.09pt)
- EV            : +86.00  (baseline = +101.00, **delta = +15.00**)

---

### 12. `body_ratio_close`

**Distribution**
- mean = 0.43482, std = 0.25870, skew = +0.200
- range = [+0.0001, +1.0000], NaN = 0.00%

**Relation avec le label (univariate)**
- Cohen's d   = +0.0070  (small)
- Mutual info = 0.00009

**Top 3 correlations avec autres features**
- `lower_wick` : r = -0.628
- `upper_wick` : r = -0.553
- `body_ratio_avg` : r = +0.289

**Importance dans le baseline**
- gain XGBoost = 0.01523  (rang 35/51)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = -0.000036
- delta logloss sur le subset des paris pris = -0.000122

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.689740   (baseline = 0.689828, delta = +0.000088)
- test_logloss  : 0.689886   (baseline = 0.690062, delta = +0.000176)
- taken         : 680  (baseline = 752, delta = +72)
- WR            : 62.65%  (baseline = 63.43%, delta = +0.78pt)
- EV            : +86.00  (baseline = +101.00, **delta = +15.00**)

---

### 13. `dist_high_20`

**Distribution**
- mean = -0.01573, std = 0.02121, skew = -3.233
- range = [-0.4514, +0.0000], NaN = 0.00%

**Relation avec le label (univariate)**
- Cohen's d   = -0.0692  (small)
- Mutual info = 0.00215

**Top 3 correlations avec autres features**
- `dist_high_50` : r = +0.805
- `dist_ema_20` : r = +0.771
- `dist_ema_50` : r = +0.722

**Importance dans le baseline**
- gain XGBoost = 0.02179  (rang 5/51)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = +0.000313
- delta logloss sur le subset des paris pris = +0.001475

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.689770   (baseline = 0.689828, delta = +0.000058)
- test_logloss  : 0.690422   (baseline = 0.690062, delta = -0.000360)
- taken         : 621  (baseline = 752, delta = +131)
- WR            : 63.93%  (baseline = 63.43%, delta = -0.50pt)
- EV            : +86.50  (baseline = +101.00, **delta = +14.50**)

---

### 14. `roc_10`

**Distribution**
- mean = 0.00096, std = 0.02306, skew = -0.040
- range = [-0.3287, +0.3265], NaN = 0.00%

**Relation avec le label (univariate)**
- Cohen's d   = -0.0870  (small)
- Mutual info = 0.00181

**Top 3 correlations avec autres features**
- `dist_ema_20` : r = +0.884 (>0.85 = forte redondance)
- `macd_hist` : r = +0.841
- `dist_ema_50` : r = +0.728

**Importance dans le baseline**
- gain XGBoost = 0.01952  (rang 7/51)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = +0.000069
- delta logloss sur le subset des paris pris = +0.000440

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.690045   (baseline = 0.689828, delta = -0.000217)
- test_logloss  : 0.690052   (baseline = 0.690062, delta = +0.000010)
- taken         : 728  (baseline = 752, delta = +24)
- WR            : 61.95%  (baseline = 63.43%, delta = +1.48pt)
- EV            : +87.00  (baseline = +101.00, **delta = +14.00**)

---

### 15. `low_ret_30m`

**Distribution**
- mean = -0.00338, std = 0.00499, skew = -6.109
- range = [-0.1713, +0.0090], NaN = 0.00%

**Relation avec le label (univariate)**
- Cohen's d   = -0.0844  (small)
- Mutual info = 0.00175

**Top 3 correlations avec autres features**
- `low_ret` : r = +0.713
- `close_ret_30m` : r = +0.633
- `atr_pct_30m` : r = -0.507

**Importance dans le baseline**
- gain XGBoost = 0.01597  (rang 28/51)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = -0.000004
- delta logloss sur le subset des paris pris = -0.000036

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.689895   (baseline = 0.689828, delta = -0.000067)
- test_logloss  : 0.690019   (baseline = 0.690062, delta = +0.000043)
- taken         : 609  (baseline = 752, delta = +143)
- WR            : 64.37%  (baseline = 63.43%, delta = -0.94pt)
- EV            : +87.50  (baseline = +101.00, **delta = +13.50**)

---

### 16. `dist_low_20`

**Distribution**
- mean = 0.01887, std = 0.02365, skew = +3.176
- range = [+0.0000, +0.4210], NaN = 0.00%

**Relation avec le label (univariate)**
- Cohen's d   = -0.0561  (small)
- Mutual info = 0.00210

**Top 3 correlations avec autres features**
- `dist_low_50` : r = +0.763
- `roc_20` : r = +0.629
- `dist_ema_20` : r = +0.627

**Importance dans le baseline**
- gain XGBoost = 0.02013  (rang 6/51)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = +0.000332
- delta logloss sur le subset des paris pris = +0.000679

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.689674   (baseline = 0.689828, delta = +0.000154)
- test_logloss  : 0.690309   (baseline = 0.690062, delta = -0.000248)
- taken         : 617  (baseline = 752, delta = +135)
- WR            : 64.18%  (baseline = 63.43%, delta = -0.75pt)
- EV            : +87.50  (baseline = +101.00, **delta = +13.50**)

---

### 17. `hour_cos`

**Distribution**
- mean = -0.00012, std = 0.70713, skew = +0.000
- range = [-1.0000, +1.0000], NaN = 0.00%

**Relation avec le label (univariate)**
- Cohen's d   = -0.0098  (small)
- Mutual info = 0.00030

**Top 3 correlations avec autres features**
- `volume_ratio_30m` : r = -0.225
- `volume_ratio` : r = -0.201
- `rv_ratio` : r = +0.036

**Importance dans le baseline**
- gain XGBoost = 0.01853  (rang 10/51)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = +0.000119
- delta logloss sur le subset des paris pris = +0.000582

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.690077   (baseline = 0.689828, delta = -0.000249)
- test_logloss  : 0.690149   (baseline = 0.690062, delta = -0.000087)
- taken         : 666  (baseline = 752, delta = +86)
- WR            : 63.21%  (baseline = 63.43%, delta = +0.22pt)
- EV            : +88.00  (baseline = +101.00, **delta = +13.00**)

---

### 18. `macd_hist`

**Distribution**
- mean = -0.00000, std = 0.00251, skew = -0.184
- range = [-0.0606, +0.0320], NaN = 0.00%

**Relation avec le label (univariate)**
- Cohen's d   = -0.0756  (small)
- Mutual info = 0.00088

**Top 3 correlations avec autres features**
- `roc_10` : r = +0.841
- `dist_ema_20` : r = +0.726
- `roc_5` : r = +0.688

**Importance dans le baseline**
- gain XGBoost = 0.01707  (rang 19/51)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = +0.000025
- delta logloss sur le subset des paris pris = +0.000253

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.689799   (baseline = 0.689828, delta = +0.000029)
- test_logloss  : 0.689941   (baseline = 0.690062, delta = +0.000120)
- taken         : 674  (baseline = 752, delta = +78)
- WR            : 63.20%  (baseline = 63.43%, delta = +0.23pt)
- EV            : +89.00  (baseline = +101.00, **delta = +12.00**)

---

### 19. `volume_ratio_30m`

**Distribution**
- mean = 1.03087, std = 1.07063, skew = +3.263
- range = [+0.0000, +16.3493], NaN = 0.00%

**Relation avec le label (univariate)**
- Cohen's d   = +0.0127  (small)
- Mutual info = 0.00014

**Top 3 correlations avec autres features**
- `volume_ratio` : r = +0.724
- `low_ret_30m` : r = -0.295
- `low_ret` : r = -0.256

**Importance dans le baseline**
- gain XGBoost = 0.01560  (rang 30/51)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = +0.000076
- delta logloss sur le subset des paris pris = +0.000019

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.689979   (baseline = 0.689828, delta = -0.000152)
- test_logloss  : 0.690074   (baseline = 0.690062, delta = -0.000012)
- taken         : 711  (baseline = 752, delta = +41)
- WR            : 62.59%  (baseline = 63.43%, delta = +0.84pt)
- EV            : +89.50  (baseline = +101.00, **delta = +11.50**)

---

### 20. `dist_high_50`

**Distribution**
- mean = -0.02689, std = 0.03249, skew = -2.676
- range = [-0.4833, +0.0000], NaN = 0.00%

**Relation avec le label (univariate)**
- Cohen's d   = -0.0430  (small)
- Mutual info = 0.00108

**Top 3 correlations avec autres features**
- `dist_high_20` : r = +0.805
- `dist_ema_50` : r = +0.769
- `dist_ema_100` : r = +0.744

**Importance dans le baseline**
- gain XGBoost = 0.01825  (rang 14/51)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = -0.000040
- delta logloss sur le subset des paris pris = +0.001114

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.689783   (baseline = 0.689828, delta = +0.000045)
- test_logloss  : 0.690080   (baseline = 0.690062, delta = -0.000019)
- taken         : 692  (baseline = 752, delta = +60)
- WR            : 63.01%  (baseline = 63.43%, delta = +0.43pt)
- EV            : +90.00  (baseline = +101.00, **delta = +11.00**)

---

### 21. `rv_100`

**Distribution**
- mean = 0.00660, std = 0.00408, skew = +2.336
- range = [+0.0011, +0.0464], NaN = 0.00%

**Relation avec le label (univariate)**
- Cohen's d   = -0.0055  (small)
- Mutual info = 0.00010

**Top 3 correlations avec autres features**
- `atr_pct_4h` : r = +0.945 (>0.85 = forte redondance)
- `atr_pct_30m` : r = +0.799
- `atr_pct_1d` : r = +0.790

**Importance dans le baseline**
- gain XGBoost = 0.01528  (rang 34/51)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = +0.000005
- delta logloss sur le subset des paris pris = +0.000205

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.689819   (baseline = 0.689828, delta = +0.000009)
- test_logloss  : 0.690125   (baseline = 0.690062, delta = -0.000063)
- taken         : 635  (baseline = 752, delta = +117)
- WR            : 64.25%  (baseline = 63.43%, delta = -0.82pt)
- EV            : +90.50  (baseline = +101.00, **delta = +10.50**)

---

### 22. `body_ratio_avg_1d`

**Distribution**
- mean = 0.43003, std = 0.06218, skew = -0.021
- range = [+0.2177, +0.6416], NaN = 0.00%

**Relation avec le label (univariate)**
- Cohen's d   = +0.0094  (small)
- Mutual info = 0.00010

**Top 3 correlations avec autres features**
- `body_ratio_close_1d` : r = +0.272
- `rv_100` : r = +0.219
- `atr_pct_4h` : r = +0.214

**Importance dans le baseline**
- gain XGBoost = 0.01534  (rang 33/51)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = -0.000012
- delta logloss sur le subset des paris pris = +0.000163

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.689866   (baseline = 0.689828, delta = -0.000038)
- test_logloss  : 0.690064   (baseline = 0.690062, delta = -0.000002)
- taken         : 707  (baseline = 752, delta = +45)
- WR            : 62.80%  (baseline = 63.43%, delta = +0.63pt)
- EV            : +90.50  (baseline = +101.00, **delta = +10.50**)

---

### 23. `open_ret_1d`

**Distribution**
- mean = -0.00002, std = 0.00077, skew = -0.654
- range = [-0.0080, +0.0062], NaN = 0.00%

**Relation avec le label (univariate)**
- Cohen's d   = -0.0060  (small)
- Mutual info = 0.00012

**Top 3 correlations avec autres features**
- `return_96` : r = +0.045
- `obv_pressure_4h` : r = +0.044
- `dist_high_50` : r = +0.044

**Importance dans le baseline**
- gain XGBoost = 0.01411  (rang 47/51)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = -0.000004
- delta logloss sur le subset des paris pris = -0.000013

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.689861   (baseline = 0.689828, delta = -0.000033)
- test_logloss  : 0.690098   (baseline = 0.690062, delta = -0.000036)
- taken         : 675  (baseline = 752, delta = +77)
- WR            : 63.56%  (baseline = 63.43%, delta = -0.12pt)
- EV            : +91.50  (baseline = +101.00, **delta = +9.50**)

---

### 24. `atr_pct_1d`

**Distribution**
- mean = 0.05001, std = 0.02390, skew = +1.664
- range = [+0.0117, +0.2082], NaN = 0.00%

**Relation avec le label (univariate)**
- Cohen's d   = -0.0161  (small)
- Mutual info = 0.00016

**Top 3 correlations avec autres features**
- `atr_pct_4h` : r = +0.843
- `rv_100` : r = +0.790
- `atr_pct_30m` : r = +0.671

**Importance dans le baseline**
- gain XGBoost = 0.01900  (rang 9/51)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = +0.000088
- delta logloss sur le subset des paris pris = +0.000040

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.689903   (baseline = 0.689828, delta = -0.000075)
- test_logloss  : 0.690061   (baseline = 0.690062, delta = +0.000001)
- taken         : 707  (baseline = 752, delta = +45)
- WR            : 62.94%  (baseline = 63.43%, delta = +0.49pt)
- EV            : +91.50  (baseline = +101.00, **delta = +9.50**)

---

### 25. `high_ret_30m`

**Distribution**
- mean = 0.00314, std = 0.00455, skew = +6.796
- range = [-0.0051, +0.2207], NaN = 0.00%

**Relation avec le label (univariate)**
- Cohen's d   = -0.0568  (small)
- Mutual info = 0.00110

**Top 3 correlations avec autres features**
- `high_ret` : r = +0.670
- `close_ret_30m` : r = +0.666
- `atr_pct_30m` : r = +0.497

**Importance dans le baseline**
- gain XGBoost = 0.01397  (rang 50/51)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = -0.000009
- delta logloss sur le subset des paris pris = -0.000053

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.689878   (baseline = 0.689828, delta = -0.000050)
- test_logloss  : 0.690058   (baseline = 0.690062, delta = +0.000004)
- taken         : 772  (baseline = 752, delta = -20)
- WR            : 61.92%  (baseline = 63.43%, delta = +1.51pt)
- EV            : +92.00  (baseline = +101.00, **delta = +9.00**)

---

### 26. `mfi_norm`

**Distribution**
- mean = 0.01211, std = 0.37950, skew = -0.004
- range = [-1.0000, +1.0000], NaN = 0.00%

**Relation avec le label (univariate)**
- Cohen's d   = -0.0609  (small)
- Mutual info = 0.00060

**Top 3 correlations avec autres features**
- `rsi_14` : r = +0.752
- `price_in_range_20` : r = +0.713
- `bb_position` : r = +0.709

**Importance dans le baseline**
- gain XGBoost = 0.01440  (rang 44/51)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = -0.000009
- delta logloss sur le subset des paris pris = -0.000263

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.689817   (baseline = 0.689828, delta = +0.000011)
- test_logloss  : 0.690079   (baseline = 0.690062, delta = -0.000017)
- taken         : 750  (baseline = 752, delta = +2)
- WR            : 62.40%  (baseline = 63.43%, delta = +1.03pt)
- EV            : +93.00  (baseline = +101.00, **delta = +8.00**)

---

### 27. `open_ret_30m`

**Distribution**
- mean = -0.00000, std = 0.00071, skew = +0.290
- range = [-0.0123, +0.0129], NaN = 0.00%

**Relation avec le label (univariate)**
- Cohen's d   = -0.0028  (small)
- Mutual info = 0.00009

**Top 3 correlations avec autres features**
- `high_ret_30m` : r = +0.112
- `low_ret_30m` : r = +0.092
- `close_ret_30m` : r = +0.083

**Importance dans le baseline**
- gain XGBoost = 0.01404  (rang 49/51)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = +0.000034
- delta logloss sur le subset des paris pris = -0.000031

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.689829   (baseline = 0.689828, delta = -0.000001)
- test_logloss  : 0.690193   (baseline = 0.690062, delta = -0.000131)
- taken         : 710  (baseline = 752, delta = +42)
- WR            : 63.10%  (baseline = 63.43%, delta = +0.33pt)
- EV            : +93.00  (baseline = +101.00, **delta = +8.00**)

---

### 28. `upper_wick`

**Distribution**
- mean = 0.26158, std = 0.21115, skew = +0.727
- range = [+0.0000, +0.9988], NaN = 0.00%

**Relation avec le label (univariate)**
- Cohen's d   = +0.0307  (small)
- Mutual info = 0.00029

**Top 3 correlations avec autres features**
- `body_ratio_close` : r = -0.553
- `lower_wick` : r = -0.301
- `low_ret` : r = +0.257

**Importance dans le baseline**
- gain XGBoost = 0.01538  (rang 32/51)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = +0.000068
- delta logloss sur le subset des paris pris = +0.000221

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.689967   (baseline = 0.689828, delta = -0.000139)
- test_logloss  : 0.690073   (baseline = 0.690062, delta = -0.000011)
- taken         : 748  (baseline = 752, delta = +4)
- WR            : 62.43%  (baseline = 63.43%, delta = +1.00pt)
- EV            : +93.00  (baseline = +101.00, **delta = +8.00**)

---

### 29. `volume_ratio_4h`

**Distribution**
- mean = 1.03043, std = 0.79819, skew = +2.548
- range = [+0.0000, +10.9038], NaN = 0.00%

**Relation avec le label (univariate)**
- Cohen's d   = +0.0126  (small)
- Mutual info = 0.00009

**Top 3 correlations avec autres features**
- `rv_ratio` : r = +0.443
- `hour_sin` : r = -0.246
- `dist_high_20` : r = -0.245

**Importance dans le baseline**
- gain XGBoost = 0.01670  (rang 22/51)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = +0.000000
- delta logloss sur le subset des paris pris = +0.000040

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.689724   (baseline = 0.689828, delta = +0.000104)
- test_logloss  : 0.690043   (baseline = 0.690062, delta = +0.000018)
- taken         : 651  (baseline = 752, delta = +101)
- WR            : 64.36%  (baseline = 63.43%, delta = -0.93pt)
- EV            : +93.50  (baseline = +101.00, **delta = +7.50**)

---

### 30. `volume_ratio_1d`

**Distribution**
- mean = 1.01595, std = 0.62586, skew = +2.443
- range = [+0.1127, +6.7123], NaN = 0.00%

**Relation avec le label (univariate)**
- Cohen's d   = +0.0041  (small)
- Mutual info = 0.00007

**Top 3 correlations avec autres features**
- `low_ret_1d` : r = -0.458
- `atr_pct_30m` : r = +0.406
- `dist_high_50` : r = -0.367

**Importance dans le baseline**
- gain XGBoost = 0.01475  (rang 38/51)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = +0.000012
- delta logloss sur le subset des paris pris = +0.000169

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.689765   (baseline = 0.689828, delta = +0.000063)
- test_logloss  : 0.690096   (baseline = 0.690062, delta = -0.000034)
- taken         : 693  (baseline = 752, delta = +59)
- WR            : 63.49%  (baseline = 63.43%, delta = -0.06pt)
- EV            : +93.50  (baseline = +101.00, **delta = +7.50**)

---

### 31. `rsi_14`

**Distribution**
- mean = 0.02409, std = 0.24342, skew = +0.004
- range = [-0.9121, +0.9488], NaN = 0.00%

**Relation avec le label (univariate)**
- Cohen's d   = -0.1013  (small)
- Mutual info = 0.00161

**Top 3 correlations avec autres features**
- `price_in_range_50` : r = +0.878 (>0.85 = forte redondance)
- `bb_position` : r = +0.864 (>0.85 = forte redondance)
- `price_in_range_20` : r = +0.857 (>0.85 = forte redondance)

**Importance dans le baseline**
- gain XGBoost = 0.01736  (rang 18/51)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = +0.000005
- delta logloss sur le subset des paris pris = -0.000072

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.689847   (baseline = 0.689828, delta = -0.000019)
- test_logloss  : 0.689972   (baseline = 0.690062, delta = +0.000089)
- taken         : 683  (baseline = 752, delta = +69)
- WR            : 63.69%  (baseline = 63.43%, delta = -0.26pt)
- EV            : +93.50  (baseline = +101.00, **delta = +7.50**)

---

### 32. `body_ratio_avg`

**Distribution**
- mean = 0.43351, std = 0.07397, skew = +0.100
- range = [+0.1264, +0.7543], NaN = 0.09%

**Relation avec le label (univariate)**
- Cohen's d   = +0.0075  (small)
- Mutual info = 0.00006

**Top 3 correlations avec autres features**
- `body_ratio_avg_30m` : r = +0.403
- `body_ratio_close` : r = +0.289
- `body_ratio_avg_4h` : r = +0.233

**Importance dans le baseline**
- gain XGBoost = 0.01457  (rang 41/51)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = -0.000017
- delta logloss sur le subset des paris pris = +0.000050

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.689877   (baseline = 0.689828, delta = -0.000049)
- test_logloss  : 0.690048   (baseline = 0.690062, delta = +0.000014)
- taken         : 718  (baseline = 752, delta = +34)
- WR            : 63.09%  (baseline = 63.43%, delta = +0.34pt)
- EV            : +94.00  (baseline = +101.00, **delta = +7.00**)

---

### 33. `close_ret_4h`

**Distribution**
- mean = 0.00039, std = 0.01484, skew = -0.047
- range = [-0.1975, +0.2244], NaN = 0.00%

**Relation avec le label (univariate)**
- Cohen's d   = -0.0716  (small)
- Mutual info = 0.00147

**Top 3 correlations avec autres features**
- `roc_5` : r = +0.700
- `macd_hist` : r = +0.638
- `dist_ema_20` : r = +0.616

**Importance dans le baseline**
- gain XGBoost = 0.01925  (rang 8/51)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = +0.000306
- delta logloss sur le subset des paris pris = +0.000239

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.689834   (baseline = 0.689828, delta = -0.000007)
- test_logloss  : 0.689967   (baseline = 0.690062, delta = +0.000095)
- taken         : 786  (baseline = 752, delta = -34)
- WR            : 61.96%  (baseline = 63.43%, delta = +1.47pt)
- EV            : +94.00  (baseline = +101.00, **delta = +7.00**)

---

### 34. `rv_ratio`

**Distribution**
- mean = 0.92522, std = 0.34644, skew = +0.614
- range = [+0.1347, +2.1707], NaN = 0.00%

**Relation avec le label (univariate)**
- Cohen's d   = +0.0111  (small)
- Mutual info = 0.00007

**Top 3 correlations avec autres features**
- `volume_ratio_4h` : r = +0.443
- `dist_high_20` : r = -0.355
- `atr_pct_30m` : r = +0.345

**Importance dans le baseline**
- gain XGBoost = 0.01408  (rang 48/51)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = +0.000022
- delta logloss sur le subset des paris pris = -0.000053

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.689780   (baseline = 0.689828, delta = +0.000048)
- test_logloss  : 0.690071   (baseline = 0.690062, delta = -0.000009)
- taken         : 705  (baseline = 752, delta = +47)
- WR            : 63.40%  (baseline = 63.43%, delta = +0.03pt)
- EV            : +94.50  (baseline = +101.00, **delta = +6.50**)

---

### 35. `hour_sin`

**Distribution**
- mean = -0.00036, std = 0.70709, skew = +0.001
- range = [-1.0000, +1.0000], NaN = 0.00%

**Relation avec le label (univariate)**
- Cohen's d   = -0.0051  (small)
- Mutual info = 0.00025

**Top 3 correlations avec autres features**
- `volume_ratio_4h` : r = -0.246
- `volume_ratio` : r = -0.165
- `body_ratio_avg_30m` : r = +0.087

**Importance dans le baseline**
- gain XGBoost = 0.01678  (rang 21/51)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = +0.000009
- delta logloss sur le subset des paris pris = +0.000001

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.689798   (baseline = 0.689828, delta = +0.000030)
- test_logloss  : 0.690186   (baseline = 0.690062, delta = -0.000124)
- taken         : 798  (baseline = 752, delta = -46)
- WR            : 61.90%  (baseline = 63.43%, delta = +1.53pt)
- EV            : +95.00  (baseline = +101.00, **delta = +6.00**)

---

### 36. `low_ret`

**Distribution**
- mean = -0.00498, std = 0.00717, skew = -5.682
- range = [-0.2412, +0.0152], NaN = 0.00%

**Relation avec le label (univariate)**
- Cohen's d   = -0.0745  (small)
- Mutual info = 0.00131

**Top 3 correlations avec autres features**
- `low_ret_30m` : r = +0.713
- `atr_pct_30m` : r = -0.543
- `dist_high_20` : r = +0.535

**Importance dans le baseline**
- gain XGBoost = 0.01831  (rang 12/51)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = +0.000026
- delta logloss sur le subset des paris pris = +0.000025

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.689874   (baseline = 0.689828, delta = -0.000046)
- test_logloss  : 0.690020   (baseline = 0.690062, delta = +0.000042)
- taken         : 726  (baseline = 752, delta = +26)
- WR            : 63.09%  (baseline = 63.43%, delta = +0.35pt)
- EV            : +95.00  (baseline = +101.00, **delta = +6.00**)

---

### 37. `atr_pct_30m`

**Distribution**
- mean = 0.00668, std = 0.00467, skew = +2.863
- range = [+0.0004, +0.0808], NaN = 0.00%

**Relation avec le label (univariate)**
- Cohen's d   = +0.0130  (small)
- Mutual info = 0.00008

**Top 3 correlations avec autres features**
- `atr_pct_4h` : r = +0.882 (>0.85 = forte redondance)
- `rv_100` : r = +0.799
- `dist_high_50` : r = -0.677

**Importance dans le baseline**
- gain XGBoost = 0.01510  (rang 36/51)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = -0.000056
- delta logloss sur le subset des paris pris = +0.000007

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.689795   (baseline = 0.689828, delta = +0.000033)
- test_logloss  : 0.690101   (baseline = 0.690062, delta = -0.000039)
- taken         : 756  (baseline = 752, delta = -4)
- WR            : 62.57%  (baseline = 63.43%, delta = +0.86pt)
- EV            : +95.00  (baseline = +101.00, **delta = +6.00**)

---

### 38. `body_ratio_close_1d`

**Distribution**
- mean = 0.43031, std = 0.25294, skew = +0.128
- range = [+0.0000, +0.9964], NaN = 0.00%

**Relation avec le label (univariate)**
- Cohen's d   = +0.0019  (small)
- Mutual info = 0.00008

**Top 3 correlations avec autres features**
- `body_ratio_avg_1d` : r = +0.272
- `high_ret_1d` : r = +0.258
- `rv_ratio` : r = +0.228

**Importance dans le baseline**
- gain XGBoost = 0.01451  (rang 43/51)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = +0.000003
- delta logloss sur le subset des paris pris = +0.000180

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.689801   (baseline = 0.689828, delta = +0.000026)
- test_logloss  : 0.690114   (baseline = 0.690062, delta = -0.000052)
- taken         : 749  (baseline = 752, delta = +3)
- WR            : 62.75%  (baseline = 63.43%, delta = +0.68pt)
- EV            : +95.50  (baseline = +101.00, **delta = +5.50**)

---

### 39. `high_ret_1d`

**Distribution**
- mean = 0.02493, std = 0.02856, skew = +2.708
- range = [-0.0037, +0.2970], NaN = 0.00%

**Relation avec le label (univariate)**
- Cohen's d   = +0.0047  (small)
- Mutual info = 0.00006

**Top 3 correlations avec autres features**
- `dist_low_50` : r = +0.663
- `rv_100` : r = +0.447
- `atr_pct_4h` : r = +0.427

**Importance dans le baseline**
- gain XGBoost = 0.01505  (rang 37/51)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = -0.000011
- delta logloss sur le subset des paris pris = -0.000008

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.689903   (baseline = 0.689828, delta = -0.000075)
- test_logloss  : 0.690045   (baseline = 0.690062, delta = +0.000017)
- taken         : 824  (baseline = 752, delta = -72)
- WR            : 61.65%  (baseline = 63.43%, delta = +1.78pt)
- EV            : +96.00  (baseline = +101.00, **delta = +5.00**)

---

### 40. `dist_low_50`

**Distribution**
- mean = 0.03497, std = 0.03923, skew = +2.559
- range = [+0.0000, +0.4657], NaN = 0.00%

**Relation avec le label (univariate)**
- Cohen's d   = -0.0360  (small)
- Mutual info = 0.00099

**Top 3 correlations avec autres features**
- `dist_low_20` : r = +0.763
- `high_ret_1d` : r = +0.663
- `dist_ema_50` : r = +0.600

**Importance dans le baseline**
- gain XGBoost = 0.01697  (rang 20/51)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = +0.000034
- delta logloss sur le subset des paris pris = -0.000009

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.689934   (baseline = 0.689828, delta = -0.000106)
- test_logloss  : 0.690153   (baseline = 0.690062, delta = -0.000091)
- taken         : 754  (baseline = 752, delta = -2)
- WR            : 62.73%  (baseline = 63.43%, delta = +0.70pt)
- EV            : +96.00  (baseline = +101.00, **delta = +5.00**)

---

### 41. `low_ret_1d`

**Distribution**
- mean = -0.02567, std = 0.03077, skew = -3.190
- range = [-0.3980, +0.0010], NaN = 0.00%

**Relation avec le label (univariate)**
- Cohen's d   = +0.0048  (small)
- Mutual info = 0.00009

**Top 3 correlations avec autres features**
- `dist_high_50` : r = +0.683
- `atr_pct_4h` : r = -0.619
- `atr_pct_30m` : r = -0.604

**Importance dans le baseline**
- gain XGBoost = 0.01648  (rang 23/51)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = +0.000011
- delta logloss sur le subset des paris pris = +0.000111

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.689980   (baseline = 0.689828, delta = -0.000152)
- test_logloss  : 0.690169   (baseline = 0.690062, delta = -0.000107)
- taken         : 768  (baseline = 752, delta = -16)
- WR            : 62.50%  (baseline = 63.43%, delta = +0.93pt)
- EV            : +96.00  (baseline = +101.00, **delta = +5.00**)

---

### 42. `atr_pct_4h`

**Distribution**
- mean = 0.01968, std = 0.01175, skew = +2.067
- range = [+0.0037, +0.1056], NaN = 0.00%

**Relation avec le label (univariate)**
- Cohen's d   = -0.0029  (small)
- Mutual info = 0.00008

**Top 3 correlations avec autres features**
- `rv_100` : r = +0.945 (>0.85 = forte redondance)
- `atr_pct_30m` : r = +0.882 (>0.85 = forte redondance)
- `atr_pct_1d` : r = +0.843

**Importance dans le baseline**
- gain XGBoost = 0.01474  (rang 39/51)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = -0.000023
- delta logloss sur le subset des paris pris = +0.000078

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.689878   (baseline = 0.689828, delta = -0.000050)
- test_logloss  : 0.690155   (baseline = 0.690062, delta = -0.000093)
- taken         : 721  (baseline = 752, delta = +31)
- WR            : 63.38%  (baseline = 63.43%, delta = +0.05pt)
- EV            : +96.50  (baseline = +101.00, **delta = +4.50**)

---

### 43. `obv_pressure_4h`

**Distribution**
- mean = 0.02517, std = 0.36807, skew = -0.037
- range = [-1.0000, +1.0000], NaN = 0.00%

**Relation avec le label (univariate)**
- Cohen's d   = -0.0116  (small)
- Mutual info = 0.00010

**Top 3 correlations avec autres features**
- `price_in_range_50` : r = +0.642
- `dist_ema_50` : r = +0.578
- `dist_ema_100` : r = +0.572

**Importance dans le baseline**
- gain XGBoost = 0.01430  (rang 45/51)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = -0.000003
- delta logloss sur le subset des paris pris = +0.000066

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.689907   (baseline = 0.689828, delta = -0.000080)
- test_logloss  : 0.690056   (baseline = 0.690062, delta = +0.000006)
- taken         : 694  (baseline = 752, delta = +58)
- WR            : 63.98%  (baseline = 63.43%, delta = -0.55pt)
- EV            : +97.00  (baseline = +101.00, **delta = +4.00**)

---

### 44. `body_ratio_avg_4h`

**Distribution**
- mean = 0.41407, std = 0.06847, skew = +0.053
- range = [+0.1558, +0.6600], NaN = 0.06%

**Relation avec le label (univariate)**
- Cohen's d   = +0.0135  (small)
- Mutual info = 0.00008

**Top 3 correlations avec autres features**
- `body_ratio_avg` : r = +0.233
- `dist_low_50` : r = +0.230
- `dist_low_20` : r = +0.197

**Importance dans le baseline**
- gain XGBoost = 0.01366  (rang 51/51)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = +0.000020
- delta logloss sur le subset des paris pris = -0.000001

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.689874   (baseline = 0.689828, delta = -0.000046)
- test_logloss  : 0.690072   (baseline = 0.690062, delta = -0.000011)
- taken         : 711  (baseline = 752, delta = +41)
- WR            : 63.71%  (baseline = 63.43%, delta = -0.28pt)
- EV            : +97.50  (baseline = +101.00, **delta = +3.50**)

---

### 45. `bb_position`

**Distribution**
- mean = 0.52058, std = 0.33359, skew = -0.097
- range = [-0.5757, +1.5806], NaN = 0.00%

**Relation avec le label (univariate)**
- Cohen's d   = -0.1384  (small)
- Mutual info = 0.00255

**Top 3 correlations avec autres features**
- `price_in_range_20` : r = +0.938 (>0.85 = forte redondance)
- `stoch_k_14` : r = +0.875 (>0.85 = forte redondance)
- `rsi_14` : r = +0.864 (>0.85 = forte redondance)

**Importance dans le baseline**
- gain XGBoost = 0.01645  (rang 24/51)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = +0.000051
- delta logloss sur le subset des paris pris = +0.000040

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.689862   (baseline = 0.689828, delta = -0.000034)
- test_logloss  : 0.690084   (baseline = 0.690062, delta = -0.000023)
- taken         : 767  (baseline = 752, delta = -15)
- WR            : 62.71%  (baseline = 63.43%, delta = +0.72pt)
- EV            : +97.50  (baseline = +101.00, **delta = +3.50**)

---

### 46. `dist_ema_50`

**Distribution**
- mean = 0.00103, std = 0.02592, skew = -1.281
- range = [-0.6488, +0.2081], NaN = 0.00%

**Relation avec le label (univariate)**
- Cohen's d   = -0.0530  (small)
- Mutual info = 0.00059

**Top 3 correlations avec autres features**
- `dist_ema_100` : r = +0.944 (>0.85 = forte redondance)
- `dist_ema_20` : r = +0.905 (>0.85 = forte redondance)
- `roc_20` : r = +0.866 (>0.85 = forte redondance)

**Importance dans le baseline**
- gain XGBoost = 0.01591  (rang 29/51)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = +0.000043
- delta logloss sur le subset des paris pris = +0.000092

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.689818   (baseline = 0.689828, delta = +0.000009)
- test_logloss  : 0.690058   (baseline = 0.690062, delta = +0.000004)
- taken         : 695  (baseline = 752, delta = +57)
- WR            : 64.03%  (baseline = 63.43%, delta = -0.60pt)
- EV            : +97.50  (baseline = +101.00, **delta = +3.50**)

---

### 47. `price_in_range_20`

**Distribution**
- mean = 0.53272, std = 0.34314, skew = -0.132
- range = [+0.0000, +1.0000], NaN = 0.00%

**Relation avec le label (univariate)**
- Cohen's d   = -0.1336  (small)
- Mutual info = 0.00277

**Top 3 correlations avec autres features**
- `bb_position` : r = +0.938 (>0.85 = forte redondance)
- `stoch_k_14` : r = +0.874 (>0.85 = forte redondance)
- `rsi_14` : r = +0.857 (>0.85 = forte redondance)

**Importance dans le baseline**
- gain XGBoost = 0.01468  (rang 40/51)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = +0.000042
- delta logloss sur le subset des paris pris = -0.000001

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.689839   (baseline = 0.689828, delta = -0.000011)
- test_logloss  : 0.690161   (baseline = 0.690062, delta = -0.000100)
- taken         : 705  (baseline = 752, delta = +47)
- WR            : 63.83%  (baseline = 63.43%, delta = -0.40pt)
- EV            : +97.50  (baseline = +101.00, **delta = +3.50**)

---

### 48. `price_in_range_50`

**Distribution**
- mean = 0.54216, std = 0.32634, skew = -0.164
- range = [+0.0000, +1.0000], NaN = 0.00%

**Relation avec le label (univariate)**
- Cohen's d   = -0.0789  (small)
- Mutual info = 0.00144

**Top 3 correlations avec autres features**
- `rsi_14` : r = +0.878 (>0.85 = forte redondance)
- `dist_ema_50` : r = +0.749
- `price_in_range_20` : r = +0.742

**Importance dans le baseline**
- gain XGBoost = 0.01632  (rang 25/51)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = +0.000002
- delta logloss sur le subset des paris pris = +0.000042

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.689871   (baseline = 0.689828, delta = -0.000043)
- test_logloss  : 0.690096   (baseline = 0.690062, delta = -0.000035)
- taken         : 782  (baseline = 752, delta = -30)
- WR            : 62.53%  (baseline = 63.43%, delta = +0.90pt)
- EV            : +98.00  (baseline = +101.00, **delta = +3.00**)

---

### 49. `dist_ema_20`

**Distribution**
- mean = 0.00039, std = 0.01587, skew = -1.328
- range = [-0.4417, +0.1667], NaN = 0.00%

**Relation avec le label (univariate)**
- Cohen's d   = -0.0948  (small)
- Mutual info = 0.00202

**Top 3 correlations avec autres features**
- `dist_ema_50` : r = +0.905 (>0.85 = forte redondance)
- `roc_10` : r = +0.884 (>0.85 = forte redondance)
- `roc_20` : r = +0.868 (>0.85 = forte redondance)

**Importance dans le baseline**
- gain XGBoost = 0.01851  (rang 11/51)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = +0.000041
- delta logloss sur le subset des paris pris = +0.000307

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.689792   (baseline = 0.689828, delta = +0.000036)
- test_logloss  : 0.690138   (baseline = 0.690062, delta = -0.000076)
- taken         : 783  (baseline = 752, delta = -31)
- WR            : 62.58%  (baseline = 63.43%, delta = +0.85pt)
- EV            : +98.50  (baseline = +101.00, **delta = +2.50**)

---

### 50. `volume_ratio`

**Distribution**
- mean = 1.03933, std = 0.97370, skew = +3.081
- range = [+0.0001, +14.7904], NaN = 0.00%

**Relation avec le label (univariate)**
- Cohen's d   = +0.0194  (small)
- Mutual info = 0.00012

**Top 3 correlations avec autres features**
- `volume_ratio_30m` : r = +0.724
- `low_ret` : r = -0.327
- `low_ret_30m` : r = -0.265

**Importance dans le baseline**
- gain XGBoost = 0.01602  (rang 27/51)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = +0.000026
- delta logloss sur le subset des paris pris = -0.000261

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.689916   (baseline = 0.689828, delta = -0.000089)
- test_logloss  : 0.690031   (baseline = 0.690062, delta = +0.000030)
- taken         : 762  (baseline = 752, delta = -10)
- WR            : 62.99%  (baseline = 63.43%, delta = +0.44pt)
- EV            : +99.00  (baseline = +101.00, **delta = +2.00**)

---

### 51. `lower_wick`

**Distribution**
- mean = 0.30361, std = 0.22604, skew = +0.585
- range = [+0.0000, +0.9992], NaN = 0.00%

**Relation avec le label (univariate)**
- Cohen's d   = -0.0367  (small)
- Mutual info = 0.00026

**Top 3 correlations avec autres features**
- `body_ratio_close` : r = -0.628
- `upper_wick` : r = -0.301
- `high_ret` : r = -0.289

**Importance dans le baseline**
- gain XGBoost = 0.01453  (rang 42/51)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = -0.000005
- delta logloss sur le subset des paris pris = -0.000096

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.689845   (baseline = 0.689828, delta = -0.000018)
- test_logloss  : 0.690023   (baseline = 0.690062, delta = +0.000038)
- taken         : 811  (baseline = 752, delta = -59)
- WR            : 62.27%  (baseline = 63.43%, delta = +1.16pt)
- EV            : +99.50  (baseline = +101.00, **delta = +1.50**)

---
