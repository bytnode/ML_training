# Feature analysis - modele_up (30m)

**Date :** 2026-05-14T19:51:14+00:00  
**Multi-seed :** 1  
**Ablation :** enabled  
**Calibration :** `asym_wknd_excl_hour_boost`  
**Duration :** 89.8s  
**Features evaluated :** 50 / 50

## Baseline (avec toutes les features)

- best_iter    : 27
- val_logloss  : 0.690764
- test_logloss : 0.691871
- taken        : 1126
- WR           : 58.53%
- EV           : +96.00

## Tableau recapitulatif (trie par |delta_EV| decroissant)

| Rang | Feature | dEV | dTaken | dWR | dVal_ll | dTest_ll | perm_global | perm_taken | gain_rank | max_corr | cohens_d | MI | NaN% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `body_ratio_close_1h` | +28.50 | -13 | +2.60pt | -0.000063 | -0.000169 | -0.000069 | +0.000243 | 7/50 | 0.182 | +0.004 | 0.00003 | 0.00% |
| 2 | `body_ratio_avg_4h` | +18.50 | -35 | +1.85pt | -0.000043 | -0.000166 | -0.000003 | +0.000038 | 43/50 | 0.207 | +0.005 | 0.00004 | 0.06% |
| 3 | `return_3` | +18.00 | -4 | +1.62pt | -0.000283 | -0.000067 | +0.000299 | +0.001605 | 1/50 | 0.807 | -0.115 | 0.00290 | 0.00% |
| 4 | `rv_ratio` | +18.00 | +32 | +1.40pt | -0.000299 | +0.000088 | +0.000019 | +0.000081 | 29/50 | 0.553 | +0.001 | 0.00004 | 0.00% |
| 5 | `close_ret_4h` | +17.50 | -9 | +1.61pt | -0.000217 | -0.000184 | +0.000045 | +0.000077 | 16/50 | 0.619 | -0.039 | 0.00040 | 0.00% |
| 6 | `return_96` | +16.50 | -65 | +1.85pt | -0.000231 | +0.000057 | +0.000002 | -0.000003 | 19/50 | 0.651 | -0.006 | 0.00009 | 0.00% |
| 7 | `volume_ratio_4h` | +16.50 | -19 | +1.58pt | -0.000053 | -0.000073 | +0.000014 | -0.000006 | 25/50 | 0.553 | +0.004 | 0.00002 | 0.00% |
| 8 | `atr_pct_15m` | +15.50 | +27 | +1.20pt | -0.000118 | -0.000152 | +0.000016 | +0.000078 | 18/50 | 0.910 | +0.008 | 0.00006 | 0.00% |
| 9 | `obv_pressure_4h` | +15.00 | -34 | +1.54pt | -0.000062 | -0.000118 | -0.000009 | -0.000002 | 45/50 | 0.632 | +0.005 | 0.00005 | 0.00% |
| 10 | `macd` | +15.00 | +52 | +0.98pt | -0.000214 | -0.000071 | -0.000024 | -0.000061 | 35/50 | 0.955 | -0.027 | 0.00020 | 0.00% |
| 11 | `open_ret_1d` | +14.50 | -13 | +1.37pt | -0.000047 | -0.000158 | +0.000006 | +0.000008 | 33/50 | 0.055 | -0.008 | 0.00008 | 0.00% |
| 12 | `body_ratio_avg` | +14.50 | +49 | +0.96pt | -0.000068 | +0.000033 | -0.000019 | -0.000109 | 50/50 | 0.408 | -0.003 | 0.00004 | 0.30% |
| 13 | `high_ret_1d` | +14.50 | -39 | +1.53pt | -0.000063 | -0.000102 | -0.000010 | -0.000000 | 49/50 | 0.500 | +0.002 | 0.00006 | 0.00% |
| 14 | `volume_ratio_1d` | +14.00 | -14 | +1.33pt | +0.000001 | -0.000062 | -0.000019 | -0.000137 | 36/50 | 0.466 | +0.005 | 0.00005 | 0.00% |
| 15 | `volume_ratio_1h` | +13.00 | +10 | +1.09pt | -0.000085 | -0.000255 | -0.000008 | -0.000091 | 20/50 | 0.495 | +0.013 | 0.00008 | 0.00% |
| 16 | `hour_cos` | +12.50 | -71 | +1.55pt | -0.000239 | +0.000088 | +0.000015 | +0.000012 | 17/50 | 0.215 | -0.008 | 0.00012 | 0.00% |
| 17 | `body_ratio_close` | +11.50 | -11 | +1.09pt | -0.000093 | +0.000002 | +0.000063 | +0.000121 | 11/50 | 0.628 | -0.002 | 0.00005 | 0.00% |
| 18 | `dist_ema_20` | +11.50 | -47 | +1.32pt | -0.000117 | -0.000049 | +0.000084 | +0.000522 | 6/50 | 0.782 | -0.099 | 0.00213 | 0.00% |
| 19 | `obv_pressure` | +11.50 | +15 | +0.92pt | -0.000125 | +0.000029 | -0.000026 | +0.000087 | 15/50 | 0.651 | -0.096 | 0.00119 | 0.00% |
| 20 | `low_ret_1d` | +11.00 | -82 | +1.49pt | -0.000082 | -0.000009 | +0.000008 | +0.000074 | 32/50 | 0.629 | +0.011 | 0.00004 | 0.00% |
| 21 | `volume_ratio` | +10.50 | +5 | +0.90pt | -0.000098 | -0.000039 | +0.000054 | +0.000039 | 24/50 | 0.709 | +0.017 | 0.00009 | 0.00% |
| 22 | `volume_ratio_15m` | +10.50 | -3 | +0.95pt | -0.000248 | +0.000050 | -0.000001 | +0.000062 | 22/50 | 0.709 | +0.021 | 0.00011 | 0.00% |
| 23 | `body_ratio_avg_1d` | +9.50 | -47 | +1.15pt | -0.000126 | -0.000001 | -0.000017 | -0.000101 | 39/50 | 0.272 | +0.004 | 0.00005 | 0.00% |
| 24 | `high_ret_15m` | +9.50 | +33 | +0.61pt | -0.000079 | +0.000011 | -0.000036 | -0.000054 | 47/50 | 0.650 | -0.029 | 0.00038 | 0.00% |
| 25 | `bb_position` | +9.50 | +37 | +0.58pt | -0.000340 | +0.000001 | +0.000125 | +0.000836 | 3/50 | 0.729 | -0.140 | 0.00259 | 0.00% |
| 26 | `close_ret` | -9.00 | -90 | -0.11pt | -0.000380 | +0.000273 | +0.000595 | +0.001660 | 2/50 | 0.668 | -0.104 | 0.00276 | 0.00% |
| 27 | `open_ret_1h` | +9.00 | +26 | +0.62pt | -0.000069 | -0.000129 | -0.000006 | +0.000027 | 37/50 | 0.076 | -0.012 | 0.00008 | 0.00% |
| 28 | `close_ret_1h` | +9.00 | +32 | +0.57pt | -0.000255 | +0.000134 | -0.000190 | +0.000264 | 4/50 | 0.807 | -0.092 | 0.00201 | 0.00% |
| 29 | `body_ratio_avg_15m` | +8.50 | +17 | +0.64pt | -0.000122 | +0.000029 | +0.000022 | +0.000029 | 48/50 | 0.408 | +0.005 | 0.00006 | 1.41% |
| 30 | `low_ret_1h` | +8.50 | +43 | +0.45pt | -0.000150 | -0.000092 | -0.000001 | +0.000010 | 23/50 | 0.634 | -0.050 | 0.00061 | 0.00% |
| 31 | `macd_signal` | -7.50 | -13 | -0.56pt | -0.000222 | +0.000090 | +0.000013 | +0.000047 | 26/50 | 0.955 | -0.002 | 0.00005 | 0.00% |
| 32 | `rv_20` | +7.00 | +38 | +0.35pt | -0.000338 | -0.000004 | +0.000001 | -0.000151 | 27/50 | 0.910 | -0.002 | 0.00005 | 0.00% |
| 33 | `low_ret_15m` | +6.50 | +51 | +0.20pt | -0.000068 | +0.000012 | +0.000041 | +0.000038 | 31/50 | 0.700 | -0.055 | 0.00072 | 0.00% |
| 34 | `high_ret` | +6.50 | -7 | +0.63pt | -0.000153 | +0.000167 | +0.000002 | -0.000088 | 41/50 | 0.651 | -0.050 | 0.00095 | 0.00% |
| 35 | `hour_sin` | +6.00 | -24 | +0.70pt | -0.000227 | +0.000011 | -0.000019 | -0.000094 | 42/50 | 0.241 | -0.006 | 0.00010 | 0.00% |
| 36 | `body_ratio_close_1d` | +6.00 | -32 | +0.75pt | -0.000044 | +0.000032 | +0.000008 | -0.000011 | 46/50 | 0.272 | +0.001 | 0.00006 | 0.00% |
| 37 | `open_ret_15m` | +5.50 | +57 | +0.06pt | -0.000096 | +0.000021 | -0.000002 | +0.000003 | 34/50 | 0.158 | +0.000 | 0.00006 | 0.00% |
| 38 | `low_ret` | +5.50 | +19 | +0.35pt | -0.000161 | +0.000015 | +0.000014 | +0.000021 | 14/50 | 0.700 | -0.070 | 0.00138 | 0.00% |
| 39 | `upper_wick` | +5.00 | -6 | +0.49pt | -0.000188 | -0.000146 | -0.000013 | +0.000019 | 44/50 | 0.557 | +0.009 | 0.00008 | 0.00% |
| 40 | `macd_hist` | +4.50 | +17 | +0.28pt | -0.000099 | -0.000044 | -0.000044 | -0.000082 | 8/50 | 0.725 | -0.086 | 0.00112 | 0.00% |
| 41 | `atr_pct_1d` | +3.50 | -53 | +0.68pt | -0.000162 | +0.000118 | +0.000003 | +0.000003 | 13/50 | 0.841 | -0.015 | 0.00007 | 0.00% |
| 42 | `body_ratio_close_15m` | +3.50 | +11 | +0.23pt | -0.000154 | +0.000035 | +0.000009 | +0.000038 | 28/50 | 0.299 | +0.002 | 0.00005 | 0.12% |
| 43 | `bb_width` | +3.00 | +0 | +0.27pt | -0.000318 | +0.000029 | -0.000013 | -0.000054 | 38/50 | 0.860 | -0.000 | 0.00004 | 0.00% |
| 44 | `lower_wick` | -2.50 | -51 | +0.16pt | -0.000251 | +0.000105 | +0.000054 | +0.000074 | 12/50 | 0.628 | -0.006 | 0.00004 | 0.00% |
| 45 | `close_ret_15m` | -2.00 | +20 | -0.34pt | -0.000158 | -0.000044 | +0.000333 | +0.000649 | 5/50 | 0.668 | -0.083 | 0.00174 | 0.00% |
| 46 | `obv_pressure_1d` | -1.50 | -81 | +0.45pt | -0.000057 | +0.000020 | -0.000095 | -0.000108 | 9/50 | 0.306 | +0.026 | 0.00011 | 0.00% |
| 47 | `obv_pressure_15m` | +1.50 | +1 | +0.13pt | -0.000188 | +0.000030 | -0.000078 | -0.000171 | 10/50 | 0.561 | -0.111 | 0.00160 | 0.00% |
| 48 | `atr_pct_4h` | -0.50 | -41 | +0.26pt | -0.000091 | +0.000037 | +0.000002 | -0.000036 | 30/50 | 0.930 | -0.007 | 0.00004 | 0.00% |
| 49 | `rv_100` | +0.50 | -3 | +0.07pt | -0.000303 | +0.000011 | +0.000002 | -0.000011 | 21/50 | 0.930 | -0.005 | 0.00004 | 0.00% |
| 50 | `mfi_norm` | -0.00 | -8 | +0.06pt | -0.000338 | +0.000029 | -0.000011 | -0.000017 | 40/50 | 0.691 | -0.069 | 0.00066 | 0.00% |

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

### 1. `body_ratio_close_1h`

**Distribution**
- mean = 0.43358, std = 0.25914, skew = +0.198
- range = [+0.0000, +1.0000], NaN = 0.00%

**Relation avec le label (univariate)**
- Cohen's d   = +0.0040  (small)
- Mutual info = 0.00003

**Top 3 correlations avec autres features**
- `body_ratio_avg` : r = +0.182
- `body_ratio_avg_15m` : r = +0.135
- `volume_ratio` : r = +0.132

**Importance dans le baseline**
- gain XGBoost = 0.02120  (rang 7/50)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = -0.000069
- delta logloss sur le subset des paris pris = +0.000243

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.690827   (baseline = 0.690764, delta = -0.000063)
- test_logloss  : 0.692040   (baseline = 0.691871, delta = -0.000169)
- taken         : 1139  (baseline = 1126, delta = -13)
- WR            : 55.93%  (baseline = 58.53%, delta = +2.60pt)
- EV            : +67.50  (baseline = +96.00, **delta = +28.50**)

---

### 2. `body_ratio_avg_4h`

**Distribution**
- mean = 0.41418, std = 0.06846, skew = +0.053
- range = [+0.1558, +0.6600], NaN = 0.06%

**Relation avec le label (univariate)**
- Cohen's d   = +0.0053  (small)
- Mutual info = 0.00004

**Top 3 correlations avec autres features**
- `bb_width` : r = +0.207
- `rv_20` : r = +0.190
- `rv_100` : r = +0.188

**Importance dans le baseline**
- gain XGBoost = 0.01491  (rang 43/50)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = -0.000003
- delta logloss sur le subset des paris pris = +0.000038

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.690808   (baseline = 0.690764, delta = -0.000043)
- test_logloss  : 0.692037   (baseline = 0.691871, delta = -0.000166)
- taken         : 1161  (baseline = 1126, delta = -35)
- WR            : 56.68%  (baseline = 58.53%, delta = +1.85pt)
- EV            : +77.50  (baseline = +96.00, **delta = +18.50**)

---

### 3. `return_3`

**Distribution**
- mean = 0.00015, std = 0.00948, skew = +0.115
- range = [-0.1731, +0.3498], NaN = 0.00%

**Relation avec le label (univariate)**
- Cohen's d   = -0.1152  (small)
- Mutual info = 0.00290

**Top 3 correlations avec autres features**
- `close_ret_1h` : r = +0.807
- `dist_ema_20` : r = +0.665
- `close_ret` : r = +0.565

**Importance dans le baseline**
- gain XGBoost = 0.08008  (rang 1/50)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = +0.000299
- delta logloss sur le subset des paris pris = +0.001605

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.691047   (baseline = 0.690764, delta = -0.000283)
- test_logloss  : 0.691938   (baseline = 0.691871, delta = -0.000067)
- taken         : 1130  (baseline = 1126, delta = -4)
- WR            : 56.90%  (baseline = 58.53%, delta = +1.62pt)
- EV            : +78.00  (baseline = +96.00, **delta = +18.00**)

---

### 4. `rv_ratio`

**Distribution**
- mean = 0.93022, std = 0.34448, skew = +0.679
- range = [+0.0790, +2.1894], NaN = 0.00%

**Relation avec le label (univariate)**
- Cohen's d   = +0.0014  (small)
- Mutual info = 0.00004

**Top 3 correlations avec autres features**
- `volume_ratio_4h` : r = +0.553
- `bb_width` : r = +0.472
- `rv_20` : r = +0.471

**Importance dans le baseline**
- gain XGBoost = 0.01605  (rang 29/50)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = +0.000019
- delta logloss sur le subset des paris pris = +0.000081

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.691063   (baseline = 0.690764, delta = -0.000299)
- test_logloss  : 0.691783   (baseline = 0.691871, delta = +0.000088)
- taken         : 1094  (baseline = 1126, delta = +32)
- WR            : 57.13%  (baseline = 58.53%, delta = +1.40pt)
- EV            : +78.00  (baseline = +96.00, **delta = +18.00**)

---

### 5. `close_ret_4h`

**Distribution**
- mean = 0.00039, std = 0.01487, skew = -0.047
- range = [-0.1975, +0.2244], NaN = 0.00%

**Relation avec le label (univariate)**
- Cohen's d   = -0.0386  (small)
- Mutual info = 0.00040

**Top 3 correlations avec autres features**
- `macd` : r = +0.619
- `dist_ema_20` : r = +0.599
- `macd_hist` : r = +0.573

**Importance dans le baseline**
- gain XGBoost = 0.01736  (rang 16/50)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = +0.000045
- delta logloss sur le subset des paris pris = +0.000077

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.690981   (baseline = 0.690764, delta = -0.000217)
- test_logloss  : 0.692055   (baseline = 0.691871, delta = -0.000184)
- taken         : 1135  (baseline = 1126, delta = -9)
- WR            : 56.92%  (baseline = 58.53%, delta = +1.61pt)
- EV            : +78.50  (baseline = +96.00, **delta = +17.50**)

---

### 6. `return_96`

**Distribution**
- mean = 0.00465, std = 0.05097, skew = +0.149
- range = [-0.4811, +0.4657], NaN = 0.00%

**Relation avec le label (univariate)**
- Cohen's d   = -0.0062  (small)
- Mutual info = 0.00009

**Top 3 correlations avec autres features**
- `macd_signal` : r = +0.651
- `obv_pressure_4h` : r = +0.632
- `macd` : r = +0.616

**Importance dans le baseline**
- gain XGBoost = 0.01702  (rang 19/50)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = +0.000002
- delta logloss sur le subset des paris pris = -0.000003

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.690996   (baseline = 0.690764, delta = -0.000231)
- test_logloss  : 0.691814   (baseline = 0.691871, delta = +0.000057)
- taken         : 1191  (baseline = 1126, delta = -65)
- WR            : 56.68%  (baseline = 58.53%, delta = +1.85pt)
- EV            : +79.50  (baseline = +96.00, **delta = +16.50**)

---

### 7. `volume_ratio_4h`

**Distribution**
- mean = 1.03121, std = 0.79878, skew = +2.553
- range = [+0.0000, +10.9038], NaN = 0.00%

**Relation avec le label (univariate)**
- Cohen's d   = +0.0038  (small)
- Mutual info = 0.00002

**Top 3 correlations avec autres features**
- `rv_ratio` : r = +0.553
- `bb_width` : r = +0.443
- `rv_20` : r = +0.313

**Importance dans le baseline**
- gain XGBoost = 0.01624  (rang 25/50)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = +0.000014
- delta logloss sur le subset des paris pris = -0.000006

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.690818   (baseline = 0.690764, delta = -0.000053)
- test_logloss  : 0.691944   (baseline = 0.691871, delta = -0.000073)
- taken         : 1145  (baseline = 1126, delta = -19)
- WR            : 56.94%  (baseline = 58.53%, delta = +1.58pt)
- EV            : +79.50  (baseline = +96.00, **delta = +16.50**)

---

### 8. `atr_pct_15m`

**Distribution**
- mean = 0.00469, std = 0.00339, skew = +3.059
- range = [+0.0001, +0.0694], NaN = 0.00%

**Relation avec le label (univariate)**
- Cohen's d   = +0.0077  (small)
- Mutual info = 0.00006

**Top 3 correlations avec autres features**
- `rv_20` : r = +0.910 (>0.85 = forte redondance)
- `bb_width` : r = +0.830
- `atr_pct_4h` : r = +0.828

**Importance dans le baseline**
- gain XGBoost = 0.01720  (rang 18/50)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = +0.000016
- delta logloss sur le subset des paris pris = +0.000078

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.690882   (baseline = 0.690764, delta = -0.000118)
- test_logloss  : 0.692023   (baseline = 0.691871, delta = -0.000152)
- taken         : 1099  (baseline = 1126, delta = +27)
- WR            : 57.32%  (baseline = 58.53%, delta = +1.20pt)
- EV            : +80.50  (baseline = +96.00, **delta = +15.50**)

---

### 9. `obv_pressure_4h`

**Distribution**
- mean = 0.02511, std = 0.36803, skew = -0.037
- range = [-1.0000, +1.0000], NaN = 0.00%

**Relation avec le label (univariate)**
- Cohen's d   = +0.0046  (small)
- Mutual info = 0.00005

**Top 3 correlations avec autres features**
- `return_96` : r = +0.632
- `macd_signal` : r = +0.490
- `macd` : r = +0.443

**Importance dans le baseline**
- gain XGBoost = 0.01463  (rang 45/50)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = -0.000009
- delta logloss sur le subset des paris pris = -0.000002

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.690827   (baseline = 0.690764, delta = -0.000062)
- test_logloss  : 0.691989   (baseline = 0.691871, delta = -0.000118)
- taken         : 1160  (baseline = 1126, delta = -34)
- WR            : 56.98%  (baseline = 58.53%, delta = +1.54pt)
- EV            : +81.00  (baseline = +96.00, **delta = +15.00**)

---

### 10. `macd`

**Distribution**
- mean = 0.00015, std = 0.00597, skew = -1.025
- range = [-0.1288, +0.0465], NaN = 0.00%

**Relation avec le label (univariate)**
- Cohen's d   = -0.0268  (small)
- Mutual info = 0.00020

**Top 3 correlations avec autres features**
- `macd_signal` : r = +0.955 (>0.85 = forte redondance)
- `dist_ema_20` : r = +0.782
- `close_ret_4h` : r = +0.619

**Importance dans le baseline**
- gain XGBoost = 0.01547  (rang 35/50)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = -0.000024
- delta logloss sur le subset des paris pris = -0.000061

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.690979   (baseline = 0.690764, delta = -0.000214)
- test_logloss  : 0.691942   (baseline = 0.691871, delta = -0.000071)
- taken         : 1074  (baseline = 1126, delta = +52)
- WR            : 57.54%  (baseline = 58.53%, delta = +0.98pt)
- EV            : +81.00  (baseline = +96.00, **delta = +15.00**)

---

### 11. `open_ret_1d`

**Distribution**
- mean = -0.00002, std = 0.00077, skew = -0.666
- range = [-0.0080, +0.0062], NaN = 0.00%

**Relation avec le label (univariate)**
- Cohen's d   = -0.0080  (small)
- Mutual info = 0.00008

**Top 3 correlations avec autres features**
- `return_96` : r = +0.055
- `obv_pressure_4h` : r = +0.042
- `obv_pressure_1d` : r = +0.026

**Importance dans le baseline**
- gain XGBoost = 0.01561  (rang 33/50)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = +0.000006
- delta logloss sur le subset des paris pris = +0.000008

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.690812   (baseline = 0.690764, delta = -0.000047)
- test_logloss  : 0.692029   (baseline = 0.691871, delta = -0.000158)
- taken         : 1139  (baseline = 1126, delta = -13)
- WR            : 57.16%  (baseline = 58.53%, delta = +1.37pt)
- EV            : +81.50  (baseline = +96.00, **delta = +14.50**)

---

### 12. `body_ratio_avg`

**Distribution**
- mean = 0.45159, std = 0.07776, skew = +0.108
- range = [+0.1409, +0.8141], NaN = 0.30%

**Relation avec le label (univariate)**
- Cohen's d   = -0.0032  (small)
- Mutual info = 0.00004

**Top 3 correlations avec autres features**
- `body_ratio_avg_15m` : r = +0.408
- `body_ratio_close` : r = +0.291
- `lower_wick` : r = -0.189

**Importance dans le baseline**
- gain XGBoost = 0.01402  (rang 50/50)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = -0.000019
- delta logloss sur le subset des paris pris = -0.000109

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.690832   (baseline = 0.690764, delta = -0.000068)
- test_logloss  : 0.691838   (baseline = 0.691871, delta = +0.000033)
- taken         : 1077  (baseline = 1126, delta = +49)
- WR            : 57.57%  (baseline = 58.53%, delta = +0.96pt)
- EV            : +81.50  (baseline = +96.00, **delta = +14.50**)

---

### 13. `high_ret_1d`

**Distribution**
- mean = 0.02499, std = 0.02861, skew = +2.711
- range = [-0.0037, +0.2970], NaN = 0.00%

**Relation avec le label (univariate)**
- Cohen's d   = +0.0025  (small)
- Mutual info = 0.00006

**Top 3 correlations avec autres features**
- `return_96` : r = +0.500
- `rv_100` : r = +0.478
- `atr_pct_4h` : r = +0.424

**Importance dans le baseline**
- gain XGBoost = 0.01406  (rang 49/50)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = -0.000010
- delta logloss sur le subset des paris pris = -0.000000

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.690827   (baseline = 0.690764, delta = -0.000063)
- test_logloss  : 0.691973   (baseline = 0.691871, delta = -0.000102)
- taken         : 1165  (baseline = 1126, delta = -39)
- WR            : 57.00%  (baseline = 58.53%, delta = +1.53pt)
- EV            : +81.50  (baseline = +96.00, **delta = +14.50**)

---

### 14. `volume_ratio_1d`

**Distribution**
- mean = 1.01653, std = 0.62632, skew = +2.443
- range = [+0.1127, +6.7123], NaN = 0.00%

**Relation avec le label (univariate)**
- Cohen's d   = +0.0054  (small)
- Mutual info = 0.00005

**Top 3 correlations avec autres features**
- `rv_100` : r = +0.466
- `low_ret_1d` : r = -0.459
- `atr_pct_15m` : r = +0.359

**Importance dans le baseline**
- gain XGBoost = 0.01542  (rang 36/50)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = -0.000019
- delta logloss sur le subset des paris pris = -0.000137

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.690764   (baseline = 0.690764, delta = +0.000001)
- test_logloss  : 0.691933   (baseline = 0.691871, delta = -0.000062)
- taken         : 1140  (baseline = 1126, delta = -14)
- WR            : 57.19%  (baseline = 58.53%, delta = +1.33pt)
- EV            : +82.00  (baseline = +96.00, **delta = +14.00**)

---

### 15. `volume_ratio_1h`

**Distribution**
- mean = 1.04044, std = 0.97405, skew = +3.077
- range = [+0.0000, +14.7904], NaN = 0.00%

**Relation avec le label (univariate)**
- Cohen's d   = +0.0126  (small)
- Mutual info = 0.00008

**Top 3 correlations avec autres features**
- `volume_ratio` : r = +0.495
- `low_ret_1h` : r = -0.326
- `volume_ratio_15m` : r = +0.220

**Importance dans le baseline**
- gain XGBoost = 0.01698  (rang 20/50)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = -0.000008
- delta logloss sur le subset des paris pris = -0.000091

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.690849   (baseline = 0.690764, delta = -0.000085)
- test_logloss  : 0.692126   (baseline = 0.691871, delta = -0.000255)
- taken         : 1116  (baseline = 1126, delta = +10)
- WR            : 57.44%  (baseline = 58.53%, delta = +1.09pt)
- EV            : +83.00  (baseline = +96.00, **delta = +13.00**)

---

### 16. `hour_cos`

**Distribution**
- mean = -0.00060, std = 0.70721, skew = +0.002
- range = [-1.0000, +1.0000], NaN = 0.00%

**Relation avec le label (univariate)**
- Cohen's d   = -0.0079  (small)
- Mutual info = 0.00012

**Top 3 correlations avec autres features**
- `volume_ratio` : r = -0.215
- `volume_ratio_1h` : r = -0.163
- `rv_ratio` : r = +0.134

**Importance dans le baseline**
- gain XGBoost = 0.01730  (rang 17/50)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = +0.000015
- delta logloss sur le subset des paris pris = +0.000012

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.691003   (baseline = 0.690764, delta = -0.000239)
- test_logloss  : 0.691783   (baseline = 0.691871, delta = +0.000088)
- taken         : 1197  (baseline = 1126, delta = -71)
- WR            : 56.98%  (baseline = 58.53%, delta = +1.55pt)
- EV            : +83.50  (baseline = +96.00, **delta = +12.50**)

---

### 17. `body_ratio_close`

**Distribution**
- mean = 0.45396, std = 0.26786, skew = +0.164
- range = [+0.0000, +1.0000], NaN = 0.00%

**Relation avec le label (univariate)**
- Cohen's d   = -0.0025  (small)
- Mutual info = 0.00005

**Top 3 correlations avec autres features**
- `lower_wick` : r = -0.628
- `upper_wick` : r = -0.557
- `body_ratio_avg` : r = +0.291

**Importance dans le baseline**
- gain XGBoost = 0.01853  (rang 11/50)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = +0.000063
- delta logloss sur le subset des paris pris = +0.000121

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.690858   (baseline = 0.690764, delta = -0.000093)
- test_logloss  : 0.691869   (baseline = 0.691871, delta = +0.000002)
- taken         : 1137  (baseline = 1126, delta = -11)
- WR            : 57.43%  (baseline = 58.53%, delta = +1.09pt)
- EV            : +84.50  (baseline = +96.00, **delta = +11.50**)

---

### 18. `dist_ema_20`

**Distribution**
- mean = 0.00020, std = 0.01124, skew = -1.247
- range = [-0.3160, +0.1381], NaN = 0.00%

**Relation avec le label (univariate)**
- Cohen's d   = -0.0990  (small)
- Mutual info = 0.00213

**Top 3 correlations avec autres features**
- `macd` : r = +0.782
- `bb_position` : r = +0.729
- `macd_hist` : r = +0.725

**Importance dans le baseline**
- gain XGBoost = 0.02611  (rang 6/50)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = +0.000084
- delta logloss sur le subset des paris pris = +0.000522

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.690881   (baseline = 0.690764, delta = -0.000117)
- test_logloss  : 0.691920   (baseline = 0.691871, delta = -0.000049)
- taken         : 1173  (baseline = 1126, delta = -47)
- WR            : 57.20%  (baseline = 58.53%, delta = +1.32pt)
- EV            : +84.50  (baseline = +96.00, **delta = +11.50**)

---

### 19. `obv_pressure`

**Distribution**
- mean = 0.00786, std = 0.38310, skew = +0.000
- range = [-1.0000, +1.0000], NaN = 0.00%

**Relation avec le label (univariate)**
- Cohen's d   = -0.0955  (small)
- Mutual info = 0.00119

**Top 3 correlations avec autres features**
- `bb_position` : r = +0.651
- `mfi_norm` : r = +0.608
- `dist_ema_20` : r = +0.525

**Importance dans le baseline**
- gain XGBoost = 0.01748  (rang 15/50)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = -0.000026
- delta logloss sur le subset des paris pris = +0.000087

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.690889   (baseline = 0.690764, delta = -0.000125)
- test_logloss  : 0.691842   (baseline = 0.691871, delta = +0.000029)
- taken         : 1111  (baseline = 1126, delta = +15)
- WR            : 57.61%  (baseline = 58.53%, delta = +0.92pt)
- EV            : +84.50  (baseline = +96.00, **delta = +11.50**)

---

### 20. `low_ret_1d`

**Distribution**
- mean = -0.02572, std = 0.03081, skew = -3.184
- range = [-0.3980, +0.0010], NaN = 0.00%

**Relation avec le label (univariate)**
- Cohen's d   = +0.0106  (small)
- Mutual info = 0.00004

**Top 3 correlations avec autres features**
- `rv_100` : r = -0.629
- `atr_pct_4h` : r = -0.618
- `atr_pct_15m` : r = -0.551

**Importance dans le baseline**
- gain XGBoost = 0.01576  (rang 32/50)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = +0.000008
- delta logloss sur le subset des paris pris = +0.000074

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.690846   (baseline = 0.690764, delta = -0.000082)
- test_logloss  : 0.691880   (baseline = 0.691871, delta = -0.000009)
- taken         : 1208  (baseline = 1126, delta = -82)
- WR            : 57.04%  (baseline = 58.53%, delta = +1.49pt)
- EV            : +85.00  (baseline = +96.00, **delta = +11.00**)

---

### 21. `volume_ratio`

**Distribution**
- mean = 1.05182, std = 1.07416, skew = +3.203
- range = [+0.0000, +16.3493], NaN = 0.00%

**Relation avec le label (univariate)**
- Cohen's d   = +0.0173  (small)
- Mutual info = 0.00009

**Top 3 correlations avec autres features**
- `volume_ratio_15m` : r = +0.709
- `volume_ratio_1h` : r = +0.495
- `low_ret` : r = -0.295

**Importance dans le baseline**
- gain XGBoost = 0.01648  (rang 24/50)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = +0.000054
- delta logloss sur le subset des paris pris = +0.000039

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.690863   (baseline = 0.690764, delta = -0.000098)
- test_logloss  : 0.691910   (baseline = 0.691871, delta = -0.000039)
- taken         : 1121  (baseline = 1126, delta = +5)
- WR            : 57.63%  (baseline = 58.53%, delta = +0.90pt)
- EV            : +85.50  (baseline = +96.00, **delta = +10.50**)

---

### 22. `volume_ratio_15m`

**Distribution**
- mean = 1.01930, std = 1.14782, skew = +3.461
- range = [+0.0000, +20.0000], NaN = 0.00%

**Relation avec le label (univariate)**
- Cohen's d   = +0.0206  (small)
- Mutual info = 0.00011

**Top 3 correlations avec autres features**
- `volume_ratio` : r = +0.709
- `low_ret_15m` : r = -0.271
- `low_ret` : r = -0.232

**Importance dans le baseline**
- gain XGBoost = 0.01678  (rang 22/50)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = -0.000001
- delta logloss sur le subset des paris pris = +0.000062

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.691012   (baseline = 0.690764, delta = -0.000248)
- test_logloss  : 0.691821   (baseline = 0.691871, delta = +0.000050)
- taken         : 1129  (baseline = 1126, delta = -3)
- WR            : 57.57%  (baseline = 58.53%, delta = +0.95pt)
- EV            : +85.50  (baseline = +96.00, **delta = +10.50**)

---

### 23. `body_ratio_avg_1d`

**Distribution**
- mean = 0.43008, std = 0.06217, skew = -0.021
- range = [+0.2177, +0.6416], NaN = 0.00%

**Relation avec le label (univariate)**
- Cohen's d   = +0.0045  (small)
- Mutual info = 0.00005

**Top 3 correlations avec autres features**
- `body_ratio_close_1d` : r = +0.272
- `rv_100` : r = +0.217
- `atr_pct_4h` : r = +0.213

**Importance dans le baseline**
- gain XGBoost = 0.01525  (rang 39/50)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = -0.000017
- delta logloss sur le subset des paris pris = -0.000101

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.690890   (baseline = 0.690764, delta = -0.000126)
- test_logloss  : 0.691872   (baseline = 0.691871, delta = -0.000001)
- taken         : 1173  (baseline = 1126, delta = -47)
- WR            : 57.37%  (baseline = 58.53%, delta = +1.15pt)
- EV            : +86.50  (baseline = +96.00, **delta = +9.50**)

---

### 24. `high_ret_15m`

**Distribution**
- mean = 0.00219, std = 0.00322, skew = +6.574
- range = [-0.0090, +0.1402], NaN = 0.00%

**Relation avec le label (univariate)**
- Cohen's d   = -0.0291  (small)
- Mutual info = 0.00038

**Top 3 correlations avec autres features**
- `close_ret_15m` : r = +0.650
- `high_ret` : r = +0.640
- `atr_pct_15m` : r = +0.513

**Importance dans le baseline**
- gain XGBoost = 0.01429  (rang 47/50)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = -0.000036
- delta logloss sur le subset des paris pris = -0.000054

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.690843   (baseline = 0.690764, delta = -0.000079)
- test_logloss  : 0.691860   (baseline = 0.691871, delta = +0.000011)
- taken         : 1093  (baseline = 1126, delta = +33)
- WR            : 57.91%  (baseline = 58.53%, delta = +0.61pt)
- EV            : +86.50  (baseline = +96.00, **delta = +9.50**)

---

### 25. `bb_position`

**Distribution**
- mean = 0.51656, std = 0.32966, skew = -0.084
- range = [-0.5831, +1.5774], NaN = 0.00%

**Relation avec le label (univariate)**
- Cohen's d   = -0.1401  (small)
- Mutual info = 0.00259

**Top 3 correlations avec autres features**
- `dist_ema_20` : r = +0.729
- `mfi_norm` : r = +0.691
- `obv_pressure` : r = +0.651

**Importance dans le baseline**
- gain XGBoost = 0.04391  (rang 3/50)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = +0.000125
- delta logloss sur le subset des paris pris = +0.000836

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.691105   (baseline = 0.690764, delta = -0.000340)
- test_logloss  : 0.691870   (baseline = 0.691871, delta = +0.000001)
- taken         : 1089  (baseline = 1126, delta = +37)
- WR            : 57.94%  (baseline = 58.53%, delta = +0.58pt)
- EV            : +86.50  (baseline = +96.00, **delta = +9.50**)

---

### 26. `close_ret`

**Distribution**
- mean = 0.00005, std = 0.00560, skew = -0.297
- range = [-0.1619, +0.1359], NaN = 0.00%

**Relation avec le label (univariate)**
- Cohen's d   = -0.1035  (small)
- Mutual info = 0.00276

**Top 3 correlations avec autres features**
- `close_ret_15m` : r = +0.668
- `high_ret` : r = +0.651
- `low_ret` : r = +0.641

**Importance dans le baseline**
- gain XGBoost = 0.07049  (rang 2/50)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = +0.000595
- delta logloss sur le subset des paris pris = +0.001660

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.691144   (baseline = 0.690764, delta = -0.000380)
- test_logloss  : 0.691598   (baseline = 0.691871, delta = +0.000273)
- taken         : 1216  (baseline = 1126, delta = -90)
- WR            : 58.63%  (baseline = 58.53%, delta = -0.11pt)
- EV            : +105.00  (baseline = +96.00, **delta = -9.00**)

---

### 27. `open_ret_1h`

**Distribution**
- mean = -0.00001, std = 0.00071, skew = +0.157
- range = [-0.0087, +0.0152], NaN = 0.00%

**Relation avec le label (univariate)**
- Cohen's d   = -0.0116  (small)
- Mutual info = 0.00008

**Top 3 correlations avec autres features**
- `low_ret_1h` : r = +0.076
- `close_ret_1h` : r = +0.059
- `bb_position` : r = +0.056

**Importance dans le baseline**
- gain XGBoost = 0.01536  (rang 37/50)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = -0.000006
- delta logloss sur le subset des paris pris = +0.000027

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.690834   (baseline = 0.690764, delta = -0.000069)
- test_logloss  : 0.692000   (baseline = 0.691871, delta = -0.000129)
- taken         : 1100  (baseline = 1126, delta = +26)
- WR            : 57.91%  (baseline = 58.53%, delta = +0.62pt)
- EV            : +87.00  (baseline = +96.00, **delta = +9.00**)

---

### 28. `close_ret_1h`

**Distribution**
- mean = 0.00010, std = 0.00781, skew = +0.007
- range = [-0.1655, +0.2281], NaN = 0.00%

**Relation avec le label (univariate)**
- Cohen's d   = -0.0918  (small)
- Mutual info = 0.00201

**Top 3 correlations avec autres features**
- `return_3` : r = +0.807
- `low_ret_1h` : r = +0.634
- `dist_ema_20` : r = +0.538

**Importance dans le baseline**
- gain XGBoost = 0.03441  (rang 4/50)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = -0.000190
- delta logloss sur le subset des paris pris = +0.000264

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.691019   (baseline = 0.690764, delta = -0.000255)
- test_logloss  : 0.691737   (baseline = 0.691871, delta = +0.000134)
- taken         : 1094  (baseline = 1126, delta = +32)
- WR            : 57.95%  (baseline = 58.53%, delta = +0.57pt)
- EV            : +87.00  (baseline = +96.00, **delta = +9.00**)

---

### 29. `body_ratio_avg_15m`

**Distribution**
- mean = 0.47247, std = 0.08323, skew = +0.155
- range = [+0.0958, +0.9488], NaN = 1.41%

**Relation avec le label (univariate)**
- Cohen's d   = +0.0049  (small)
- Mutual info = 0.00006

**Top 3 correlations avec autres features**
- `body_ratio_avg` : r = +0.408
- `body_ratio_close_15m` : r = +0.299
- `body_ratio_close` : r = +0.183

**Importance dans le baseline**
- gain XGBoost = 0.01426  (rang 48/50)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = +0.000022
- delta logloss sur le subset des paris pris = +0.000029

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.690886   (baseline = 0.690764, delta = -0.000122)
- test_logloss  : 0.691842   (baseline = 0.691871, delta = +0.000029)
- taken         : 1109  (baseline = 1126, delta = +17)
- WR            : 57.89%  (baseline = 58.53%, delta = +0.64pt)
- EV            : +87.50  (baseline = +96.00, **delta = +8.50**)

---

### 30. `low_ret_1h`

**Distribution**
- mean = -0.00499, std = 0.00718, skew = -5.677
- range = [-0.2412, +0.0152], NaN = 0.00%

**Relation avec le label (univariate)**
- Cohen's d   = -0.0498  (small)
- Mutual info = 0.00061

**Top 3 correlations avec autres features**
- `close_ret_1h` : r = +0.634
- `atr_pct_15m` : r = -0.590
- `return_3` : r = +0.533

**Importance dans le baseline**
- gain XGBoost = 0.01671  (rang 23/50)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = -0.000001
- delta logloss sur le subset des paris pris = +0.000010

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.690914   (baseline = 0.690764, delta = -0.000150)
- test_logloss  : 0.691963   (baseline = 0.691871, delta = -0.000092)
- taken         : 1083  (baseline = 1126, delta = +43)
- WR            : 58.08%  (baseline = 58.53%, delta = +0.45pt)
- EV            : +87.50  (baseline = +96.00, **delta = +8.50**)

---

### 31. `macd_signal`

**Distribution**
- mean = 0.00015, std = 0.00563, skew = -0.960
- range = [-0.0950, +0.0416], NaN = 0.00%

**Relation avec le label (univariate)**
- Cohen's d   = -0.0015  (small)
- Mutual info = 0.00005

**Top 3 correlations avec autres features**
- `macd` : r = +0.955 (>0.85 = forte redondance)
- `return_96` : r = +0.651
- `dist_ema_20` : r = +0.602

**Importance dans le baseline**
- gain XGBoost = 0.01618  (rang 26/50)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = +0.000013
- delta logloss sur le subset des paris pris = +0.000047

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.690987   (baseline = 0.690764, delta = -0.000222)
- test_logloss  : 0.691781   (baseline = 0.691871, delta = +0.000090)
- taken         : 1139  (baseline = 1126, delta = -13)
- WR            : 59.09%  (baseline = 58.53%, delta = -0.56pt)
- EV            : +103.50  (baseline = +96.00, **delta = -7.50**)

---

### 32. `rv_20`

**Distribution**
- mean = 0.00429, std = 0.00339, skew = +3.148
- range = [+0.0002, +0.0624], NaN = 0.00%

**Relation avec le label (univariate)**
- Cohen's d   = -0.0020  (small)
- Mutual info = 0.00005

**Top 3 correlations avec autres features**
- `atr_pct_15m` : r = +0.910 (>0.85 = forte redondance)
- `bb_width` : r = +0.860 (>0.85 = forte redondance)
- `rv_100` : r = +0.796

**Importance dans le baseline**
- gain XGBoost = 0.01616  (rang 27/50)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = +0.000001
- delta logloss sur le subset des paris pris = -0.000151

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.691103   (baseline = 0.690764, delta = -0.000338)
- test_logloss  : 0.691875   (baseline = 0.691871, delta = -0.000004)
- taken         : 1088  (baseline = 1126, delta = +38)
- WR            : 58.18%  (baseline = 58.53%, delta = +0.35pt)
- EV            : +89.00  (baseline = +96.00, **delta = +7.00**)

---

### 33. `low_ret_15m`

**Distribution**
- mean = -0.00235, std = 0.00350, skew = -6.491
- range = [-0.1439, +0.0060], NaN = 0.00%

**Relation avec le label (univariate)**
- Cohen's d   = -0.0553  (small)
- Mutual info = 0.00072

**Top 3 correlations avec autres features**
- `low_ret` : r = +0.700
- `close_ret_15m` : r = +0.637
- `atr_pct_15m` : r = -0.502

**Importance dans le baseline**
- gain XGBoost = 0.01590  (rang 31/50)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = +0.000041
- delta logloss sur le subset des paris pris = +0.000038

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.690832   (baseline = 0.690764, delta = -0.000068)
- test_logloss  : 0.691859   (baseline = 0.691871, delta = +0.000012)
- taken         : 1075  (baseline = 1126, delta = +51)
- WR            : 58.33%  (baseline = 58.53%, delta = +0.20pt)
- EV            : +89.50  (baseline = +96.00, **delta = +6.50**)

---

### 34. `high_ret`

**Distribution**
- mean = 0.00322, std = 0.00458, skew = +6.029
- range = [-0.0051, +0.2207], NaN = 0.00%

**Relation avec le label (univariate)**
- Cohen's d   = -0.0501  (small)
- Mutual info = 0.00095

**Top 3 correlations avec autres features**
- `close_ret` : r = +0.651
- `high_ret_15m` : r = +0.640
- `atr_pct_15m` : r = +0.518

**Importance dans le baseline**
- gain XGBoost = 0.01499  (rang 41/50)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = +0.000002
- delta logloss sur le subset des paris pris = -0.000088

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.690918   (baseline = 0.690764, delta = -0.000153)
- test_logloss  : 0.691704   (baseline = 0.691871, delta = +0.000167)
- taken         : 1133  (baseline = 1126, delta = -7)
- WR            : 57.90%  (baseline = 58.53%, delta = +0.63pt)
- EV            : +89.50  (baseline = +96.00, **delta = +6.50**)

---

### 35. `hour_sin`

**Distribution**
- mean = -0.00089, std = 0.70701, skew = +0.002
- range = [-1.0000, +1.0000], NaN = 0.00%

**Relation avec le label (univariate)**
- Cohen's d   = -0.0058  (small)
- Mutual info = 0.00010

**Top 3 correlations avec autres features**
- `volume_ratio_4h` : r = -0.241
- `volume_ratio_1h` : r = -0.198
- `body_ratio_avg_15m` : r = +0.109

**Importance dans le baseline**
- gain XGBoost = 0.01492  (rang 42/50)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = -0.000019
- delta logloss sur le subset des paris pris = -0.000094

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.690992   (baseline = 0.690764, delta = -0.000227)
- test_logloss  : 0.691860   (baseline = 0.691871, delta = +0.000011)
- taken         : 1150  (baseline = 1126, delta = -24)
- WR            : 57.83%  (baseline = 58.53%, delta = +0.70pt)
- EV            : +90.00  (baseline = +96.00, **delta = +6.00**)

---

### 36. `body_ratio_close_1d`

**Distribution**
- mean = 0.43034, std = 0.25292, skew = +0.128
- range = [+0.0000, +0.9964], NaN = 0.00%

**Relation avec le label (univariate)**
- Cohen's d   = +0.0013  (small)
- Mutual info = 0.00006

**Top 3 correlations avec autres features**
- `body_ratio_avg_1d` : r = +0.272
- `high_ret_1d` : r = +0.258
- `volume_ratio_1d` : r = +0.189

**Importance dans le baseline**
- gain XGBoost = 0.01435  (rang 46/50)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = +0.000008
- delta logloss sur le subset des paris pris = -0.000011

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.690809   (baseline = 0.690764, delta = -0.000044)
- test_logloss  : 0.691839   (baseline = 0.691871, delta = +0.000032)
- taken         : 1158  (baseline = 1126, delta = -32)
- WR            : 57.77%  (baseline = 58.53%, delta = +0.75pt)
- EV            : +90.00  (baseline = +96.00, **delta = +6.00**)

---

### 37. `open_ret_15m`

**Distribution**
- mean = 0.00001, std = 0.00070, skew = +0.200
- range = [-0.0104, +0.0154], NaN = 0.00%

**Relation avec le label (univariate)**
- Cohen's d   = +0.0001  (small)
- Mutual info = 0.00006

**Top 3 correlations avec autres features**
- `high_ret_15m` : r = +0.158
- `low_ret_15m` : r = +0.141
- `close_ret_15m` : r = +0.128

**Importance dans le baseline**
- gain XGBoost = 0.01549  (rang 34/50)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = -0.000002
- delta logloss sur le subset des paris pris = +0.000003

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.690860   (baseline = 0.690764, delta = -0.000096)
- test_logloss  : 0.691850   (baseline = 0.691871, delta = +0.000021)
- taken         : 1069  (baseline = 1126, delta = +57)
- WR            : 58.47%  (baseline = 58.53%, delta = +0.06pt)
- EV            : +90.50  (baseline = +96.00, **delta = +5.50**)

---

### 38. `low_ret`

**Distribution**
- mean = -0.00348, std = 0.00513, skew = -6.584
- range = [-0.2412, +0.0152], NaN = 0.00%

**Relation avec le label (univariate)**
- Cohen's d   = -0.0702  (small)
- Mutual info = 0.00138

**Top 3 correlations avec autres features**
- `low_ret_15m` : r = +0.700
- `close_ret` : r = +0.641
- `atr_pct_15m` : r = -0.538

**Importance dans le baseline**
- gain XGBoost = 0.01757  (rang 14/50)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = +0.000014
- delta logloss sur le subset des paris pris = +0.000021

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.690925   (baseline = 0.690764, delta = -0.000161)
- test_logloss  : 0.691856   (baseline = 0.691871, delta = +0.000015)
- taken         : 1107  (baseline = 1126, delta = +19)
- WR            : 58.18%  (baseline = 58.53%, delta = +0.35pt)
- EV            : +90.50  (baseline = +96.00, **delta = +5.50**)

---

### 39. `upper_wick`

**Distribution**
- mean = 0.25296, std = 0.21818, skew = +0.790
- range = [+0.0000, +0.9999], NaN = 0.00%

**Relation avec le label (univariate)**
- Cohen's d   = +0.0093  (small)
- Mutual info = 0.00008

**Top 3 correlations avec autres features**
- `body_ratio_close` : r = -0.557
- `lower_wick` : r = -0.297
- `low_ret` : r = +0.236

**Importance dans le baseline**
- gain XGBoost = 0.01487  (rang 44/50)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = -0.000013
- delta logloss sur le subset des paris pris = +0.000019

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.690953   (baseline = 0.690764, delta = -0.000188)
- test_logloss  : 0.692017   (baseline = 0.691871, delta = -0.000146)
- taken         : 1132  (baseline = 1126, delta = -6)
- WR            : 58.04%  (baseline = 58.53%, delta = +0.49pt)
- EV            : +91.00  (baseline = +96.00, **delta = +5.00**)

---

### 40. `macd_hist`

**Distribution**
- mean = -0.00000, std = 0.00176, skew = -0.171
- range = [-0.0414, +0.0245], NaN = 0.00%

**Relation avec le label (univariate)**
- Cohen's d   = -0.0856  (small)
- Mutual info = 0.00112

**Top 3 correlations avec autres features**
- `dist_ema_20` : r = +0.725
- `bb_position` : r = +0.602
- `close_ret_4h` : r = +0.573

**Importance dans le baseline**
- gain XGBoost = 0.01924  (rang 8/50)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = -0.000044
- delta logloss sur le subset des paris pris = -0.000082

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.690863   (baseline = 0.690764, delta = -0.000099)
- test_logloss  : 0.691915   (baseline = 0.691871, delta = -0.000044)
- taken         : 1109  (baseline = 1126, delta = +17)
- WR            : 58.25%  (baseline = 58.53%, delta = +0.28pt)
- EV            : +91.50  (baseline = +96.00, **delta = +4.50**)

---

### 41. `atr_pct_1d`

**Distribution**
- mean = 0.05011, std = 0.02391, skew = +1.665
- range = [+0.0117, +0.2082], NaN = 0.00%

**Relation avec le label (univariate)**
- Cohen's d   = -0.0151  (small)
- Mutual info = 0.00007

**Top 3 correlations avec autres features**
- `atr_pct_4h` : r = +0.841
- `rv_100` : r = +0.716
- `atr_pct_15m` : r = +0.631

**Importance dans le baseline**
- gain XGBoost = 0.01762  (rang 13/50)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = +0.000003
- delta logloss sur le subset des paris pris = +0.000003

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.690927   (baseline = 0.690764, delta = -0.000162)
- test_logloss  : 0.691753   (baseline = 0.691871, delta = +0.000118)
- taken         : 1179  (baseline = 1126, delta = -53)
- WR            : 57.85%  (baseline = 58.53%, delta = +0.68pt)
- EV            : +92.50  (baseline = +96.00, **delta = +3.50**)

---

### 42. `body_ratio_close_15m`

**Distribution**
- mean = 0.47227, std = 0.28222, skew = +0.131
- range = [+0.0000, +1.0000], NaN = 0.12%

**Relation avec le label (univariate)**
- Cohen's d   = +0.0025  (small)
- Mutual info = 0.00005

**Top 3 correlations avec autres features**
- `body_ratio_avg_15m` : r = +0.299
- `body_ratio_close` : r = +0.261
- `lower_wick` : r = -0.165

**Importance dans le baseline**
- gain XGBoost = 0.01616  (rang 28/50)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = +0.000009
- delta logloss sur le subset des paris pris = +0.000038

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.690918   (baseline = 0.690764, delta = -0.000154)
- test_logloss  : 0.691836   (baseline = 0.691871, delta = +0.000035)
- taken         : 1115  (baseline = 1126, delta = +11)
- WR            : 58.30%  (baseline = 58.53%, delta = +0.23pt)
- EV            : +92.50  (baseline = +96.00, **delta = +3.50**)

---

### 43. `bb_width`

**Distribution**
- mean = 0.02779, std = 0.02667, skew = +2.999
- range = [+0.0009, +0.4714], NaN = 0.00%

**Relation avec le label (univariate)**
- Cohen's d   = -0.0004  (small)
- Mutual info = 0.00004

**Top 3 correlations avec autres features**
- `rv_20` : r = +0.860 (>0.85 = forte redondance)
- `atr_pct_15m` : r = +0.830
- `rv_100` : r = +0.651

**Importance dans le baseline**
- gain XGBoost = 0.01534  (rang 38/50)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = -0.000013
- delta logloss sur le subset des paris pris = -0.000054

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.691082   (baseline = 0.690764, delta = -0.000318)
- test_logloss  : 0.691842   (baseline = 0.691871, delta = +0.000029)
- taken         : 1126  (baseline = 1126, delta = +0)
- WR            : 58.26%  (baseline = 58.53%, delta = +0.27pt)
- EV            : +93.00  (baseline = +96.00, **delta = +3.00**)

---

### 44. `lower_wick`

**Distribution**
- mean = 0.29309, std = 0.23349, skew = +0.649
- range = [+0.0000, +0.9994], NaN = 0.00%

**Relation avec le label (univariate)**
- Cohen's d   = -0.0059  (small)
- Mutual info = 0.00004

**Top 3 correlations avec autres features**
- `body_ratio_close` : r = -0.628
- `upper_wick` : r = -0.297
- `high_ret` : r = -0.266

**Importance dans le baseline**
- gain XGBoost = 0.01840  (rang 12/50)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = +0.000054
- delta logloss sur le subset des paris pris = +0.000074

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.691015   (baseline = 0.690764, delta = -0.000251)
- test_logloss  : 0.691766   (baseline = 0.691871, delta = +0.000105)
- taken         : 1177  (baseline = 1126, delta = -51)
- WR            : 58.37%  (baseline = 58.53%, delta = +0.16pt)
- EV            : +98.50  (baseline = +96.00, **delta = -2.50**)

---

### 45. `close_ret_15m`

**Distribution**
- mean = 0.00003, std = 0.00392, skew = +0.231
- range = [-0.1061, +0.1391], NaN = 0.00%

**Relation avec le label (univariate)**
- Cohen's d   = -0.0825  (small)
- Mutual info = 0.00174

**Top 3 correlations avec autres features**
- `close_ret` : r = +0.668
- `high_ret_15m` : r = +0.650
- `low_ret_15m` : r = +0.637

**Importance dans le baseline**
- gain XGBoost = 0.02724  (rang 5/50)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = +0.000333
- delta logloss sur le subset des paris pris = +0.000649

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.690922   (baseline = 0.690764, delta = -0.000158)
- test_logloss  : 0.691915   (baseline = 0.691871, delta = -0.000044)
- taken         : 1106  (baseline = 1126, delta = +20)
- WR            : 58.86%  (baseline = 58.53%, delta = -0.34pt)
- EV            : +98.00  (baseline = +96.00, **delta = -2.00**)

---

### 46. `obv_pressure_1d`

**Distribution**
- mean = 0.04323, std = 0.34235, skew = +0.075
- range = [-0.9368, +1.0000], NaN = 0.00%

**Relation avec le label (univariate)**
- Cohen's d   = +0.0261  (small)
- Mutual info = 0.00011

**Top 3 correlations avec autres features**
- `return_96` : r = +0.306
- `atr_pct_1d` : r = -0.259
- `low_ret_1d` : r = +0.242

**Importance dans le baseline**
- gain XGBoost = 0.01924  (rang 9/50)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = -0.000095
- delta logloss sur le subset des paris pris = -0.000108

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.690822   (baseline = 0.690764, delta = -0.000057)
- test_logloss  : 0.691851   (baseline = 0.691871, delta = +0.000020)
- taken         : 1207  (baseline = 1126, delta = -81)
- WR            : 58.08%  (baseline = 58.53%, delta = +0.45pt)
- EV            : +97.50  (baseline = +96.00, **delta = -1.50**)

---

### 47. `obv_pressure_15m`

**Distribution**
- mean = 0.00419, std = 0.40683, skew = -0.001
- range = [-1.0000, +1.0000], NaN = 0.00%

**Relation avec le label (univariate)**
- Cohen's d   = -0.1111  (small)
- Mutual info = 0.00160

**Top 3 correlations avec autres features**
- `bb_position` : r = +0.561
- `obv_pressure` : r = +0.458
- `dist_ema_20` : r = +0.429

**Importance dans le baseline**
- gain XGBoost = 0.01912  (rang 10/50)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = -0.000078
- delta logloss sur le subset des paris pris = -0.000171

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.690952   (baseline = 0.690764, delta = -0.000188)
- test_logloss  : 0.691841   (baseline = 0.691871, delta = +0.000030)
- taken         : 1125  (baseline = 1126, delta = +1)
- WR            : 58.40%  (baseline = 58.53%, delta = +0.13pt)
- EV            : +94.50  (baseline = +96.00, **delta = +1.50**)

---

### 48. `atr_pct_4h`

**Distribution**
- mean = 0.01973, std = 0.01176, skew = +2.068
- range = [+0.0037, +0.1056], NaN = 0.00%

**Relation avec le label (univariate)**
- Cohen's d   = -0.0066  (small)
- Mutual info = 0.00004

**Top 3 correlations avec autres features**
- `rv_100` : r = +0.930 (>0.85 = forte redondance)
- `atr_pct_1d` : r = +0.841
- `atr_pct_15m` : r = +0.828

**Importance dans le baseline**
- gain XGBoost = 0.01590  (rang 30/50)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = +0.000002
- delta logloss sur le subset des paris pris = -0.000036

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.690855   (baseline = 0.690764, delta = -0.000091)
- test_logloss  : 0.691834   (baseline = 0.691871, delta = +0.000037)
- taken         : 1167  (baseline = 1126, delta = -41)
- WR            : 58.27%  (baseline = 58.53%, delta = +0.26pt)
- EV            : +96.50  (baseline = +96.00, **delta = -0.50**)

---

### 49. `rv_100`

**Distribution**
- mean = 0.00465, std = 0.00305, skew = +2.450
- range = [+0.0005, +0.0353], NaN = 0.00%

**Relation avec le label (univariate)**
- Cohen's d   = -0.0045  (small)
- Mutual info = 0.00004

**Top 3 correlations avec autres features**
- `atr_pct_4h` : r = +0.930 (>0.85 = forte redondance)
- `atr_pct_15m` : r = +0.819
- `rv_20` : r = +0.796

**Importance dans le baseline**
- gain XGBoost = 0.01690  (rang 21/50)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = +0.000002
- delta logloss sur le subset des paris pris = -0.000011

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.691067   (baseline = 0.690764, delta = -0.000303)
- test_logloss  : 0.691859   (baseline = 0.691871, delta = +0.000011)
- taken         : 1129  (baseline = 1126, delta = -3)
- WR            : 58.46%  (baseline = 58.53%, delta = +0.07pt)
- EV            : +95.50  (baseline = +96.00, **delta = +0.50**)

---

### 50. `mfi_norm`

**Distribution**
- mean = 0.00516, std = 0.38600, skew = +0.013
- range = [-1.0000, +1.0000], NaN = 0.00%

**Relation avec le label (univariate)**
- Cohen's d   = -0.0692  (small)
- Mutual info = 0.00066

**Top 3 correlations avec autres features**
- `bb_position` : r = +0.691
- `obv_pressure` : r = +0.608
- `dist_ema_20` : r = +0.595

**Importance dans le baseline**
- gain XGBoost = 0.01502  (rang 40/50)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = -0.000011
- delta logloss sur le subset des paris pris = -0.000017

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.691102   (baseline = 0.690764, delta = -0.000338)
- test_logloss  : 0.691842   (baseline = 0.691871, delta = +0.000029)
- taken         : 1134  (baseline = 1126, delta = -8)
- WR            : 58.47%  (baseline = 58.53%, delta = +0.06pt)
- EV            : +96.00  (baseline = +96.00, **delta = -0.00**)

---
