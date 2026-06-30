# Feature analysis - modele_up

**Date :** 2026-05-09T18:55:18+00:00  
**Multi-seed :** 1  
**Ablation :** enabled  
**Duration :** 118.1s  
**Features evaluated :** 25 / 25

## Baseline (avec toutes les features)

- best_iter    : 23
- val_logloss  : 0.691221
- test_logloss : 0.690918
- taken        : 1253
- WR           : 55.79%
- EV           : +72.50

## Tableau recapitulatif (trie par |delta_EV| decroissant)

| Rang | Feature | dEV | dTaken | dWR | dVal_ll | dTest_ll | perm_global | perm_taken | gain_rank | max_corr | cohens_d | MI | NaN% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `volume_ratio_30m` | -75.50 | -1,633 | +0.66pt | +0.000019 | -0.000022 | +0.000022 | +0.000022 | 18/25 | 0.298 | +0.013 | 0.00006 | 0.00% |
| 2 | `close_ret_1h` | +20.50 | +151 | +1.07pt | -0.000153 | -0.000015 | +0.000117 | -0.000200 | 7/25 | 0.645 | -0.061 | 0.00087 | 0.00% |
| 3 | `low_ret` | -12.50 | -345 | +0.47pt | -0.000022 | +0.000159 | +0.000012 | +0.000044 | 13/25 | 0.645 | -0.048 | 0.00041 | 0.00% |
| 4 | `bb_position` | +11.50 | +149 | +0.26pt | +0.000029 | +0.000008 | +0.000211 | +0.000214 | 4/25 | 0.707 | -0.126 | 0.00204 | 0.00% |
| 5 | `body_ratio_close_30m` | -10.00 | -268 | +0.36pt | +0.000032 | +0.000026 | +0.000034 | -0.000022 | 17/25 | 0.098 | +0.002 | 0.00003 | 0.01% |
| 6 | `dist_ema_20` | -9.50 | -151 | -0.05pt | -0.000104 | +0.000105 | +0.000488 | +0.000893 | 2/25 | 0.772 | -0.097 | 0.00182 | 0.00% |
| 7 | `close_ret_30m` | -9.00 | -82 | -0.32pt | +0.000064 | +0.000219 | +0.000176 | +0.000217 | 3/25 | 0.803 | -0.082 | 0.00150 | 0.00% |
| 8 | `dist_to_high_50` | -8.50 | -179 | +0.13pt | +0.000080 | +0.000072 | -0.000022 | -0.000137 | 11/25 | 0.666 | -0.052 | 0.00069 | 0.00% |
| 9 | `obv_pressure_5m` | -7.50 | -193 | +0.25pt | -0.000014 | +0.000100 | -0.000003 | -0.000253 | 22/25 | 0.482 | -0.087 | 0.00096 | 0.00% |
| 10 | `dist_to_low_50` | +6.50 | -253 | +1.40pt | +0.000058 | +0.000033 | +0.000067 | +0.000103 | 8/25 | 0.477 | -0.045 | 0.00075 | 0.00% |
| 11 | `close_ret` | +6.50 | +37 | +0.36pt | -0.000039 | +0.000037 | +0.000236 | -0.000118 | 6/25 | 0.645 | -0.072 | 0.00098 | 0.00% |
| 12 | `low_ret_30m` | +6.00 | +18 | +0.40pt | +0.000002 | +0.000059 | -0.000010 | +0.000076 | 23/25 | 0.641 | -0.054 | 0.00071 | 0.00% |
| 13 | `hour_sin` | -5.00 | -130 | +0.18pt | -0.000038 | +0.000010 | +0.000026 | -0.000174 | 15/25 | 0.055 | -0.014 | 0.00005 | 0.00% |
| 14 | `high_ret_5m` | -5.00 | -40 | -0.21pt | +0.000004 | +0.000173 | +0.000013 | +0.000092 | 14/25 | 0.641 | -0.037 | 0.00029 | 0.00% |
| 15 | `macd` | -5.00 | -188 | +0.41pt | +0.000017 | +0.000082 | +0.000002 | +0.000034 | 19/25 | 0.772 | -0.043 | 0.00038 | 0.00% |
| 16 | `high_ret` | +5.00 | -270 | +1.35pt | +0.000011 | +0.000134 | +0.000007 | +0.000143 | 21/25 | 0.641 | -0.033 | 0.00027 | 0.00% |
| 17 | `macd_hist` | -3.50 | -55 | -0.02pt | -0.000008 | +0.000013 | +0.000025 | -0.000238 | 9/25 | 0.739 | -0.077 | 0.00089 | 0.00% |
| 18 | `close_ret_5m` | +3.50 | -311 | +1.37pt | -0.000016 | -0.000046 | +0.000160 | +0.000644 | 5/25 | 0.646 | -0.070 | 0.00078 | 0.00% |
| 19 | `return_3` | -2.50 | +29 | -0.34pt | -0.000006 | -0.000091 | +0.000400 | +0.000875 | 1/25 | 0.803 | -0.098 | 0.00193 | 0.00% |
| 20 | `mfi_norm` | -2.00 | -12 | -0.10pt | +0.000043 | +0.000041 | +0.000001 | -0.000014 | 25/25 | 0.666 | -0.071 | 0.00065 | 0.00% |
| 21 | `low_ret_5m` | +2.00 | -14 | +0.22pt | +0.000063 | +0.000120 | +0.000005 | +0.000055 | 20/25 | 0.646 | -0.053 | 0.00041 | 0.00% |
| 22 | `acc_4` | -1.50 | -193 | +0.67pt | +0.000077 | +0.000040 | +0.000014 | +0.000089 | 12/25 | 0.611 | -0.048 | 0.00037 | 0.00% |
| 23 | `low_ret_1h` | -1.50 | -113 | +0.37pt | +0.000002 | +0.000003 | +0.000008 | -0.000003 | 24/25 | 0.634 | -0.034 | 0.00024 | 0.00% |
| 24 | `is_weekend` | -1.00 | -166 | +0.61pt | +0.000018 | -0.000020 | +0.000000 | +0.000000 | 10/25 | 0.099 | -0.008 | 0.00000 | 0.00% |
| 25 | `autocorr_24` | +0.50 | -207 | +0.85pt | -0.000002 | -0.000016 | +0.000017 | -0.000014 | 16/25 | 0.150 | -0.003 | 0.00001 | 0.00% |

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

### 1. `volume_ratio_30m`

**Distribution**
- mean = 1.05323, std = 1.09373, skew = +3.281
- range = [+0.0000, +20.0000], NaN = 0.00%

**Relation avec le label (univariate)**
- Cohen's d   = +0.0129  (small)
- Mutual info = 0.00006

**Top 3 correlations avec autres features**
- `low_ret_30m` : r = -0.298
- `low_ret` : r = -0.176
- `high_ret` : r = +0.172

**Importance dans le baseline**
- gain XGBoost = 0.02622  (rang 18/25)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = +0.000022
- delta logloss sur le subset des paris pris = +0.000022

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.691202   (baseline = 0.691221, delta = +0.000019)
- test_logloss  : 0.690940   (baseline = 0.690918, delta = -0.000022)
- taken         : 2886  (baseline = 1253, delta = -1,633)
- WR            : 55.13%  (baseline = 55.79%, delta = +0.66pt)
- EV            : +148.00  (baseline = +72.50, **delta = -75.50**)

---

### 2. `close_ret_1h`

**Distribution**
- mean = 0.00008, std = 0.00782, skew = -0.013
- range = [-0.1655, +0.2281], NaN = 0.00%

**Relation avec le label (univariate)**
- Cohen's d   = -0.0614  (small)
- Mutual info = 0.00087

**Top 3 correlations avec autres features**
- `macd_hist` : r = +0.645
- `low_ret_1h` : r = +0.634
- `dist_ema_20` : r = +0.618

**Importance dans le baseline**
- gain XGBoost = 0.03287  (rang 7/25)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = +0.000117
- delta logloss sur le subset des paris pris = -0.000200

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.691375   (baseline = 0.691221, delta = -0.000153)
- test_logloss  : 0.690933   (baseline = 0.690918, delta = -0.000015)
- taken         : 1102  (baseline = 1253, delta = +151)
- WR            : 54.72%  (baseline = 55.79%, delta = +1.07pt)
- EV            : +52.00  (baseline = +72.50, **delta = +20.50**)

---

### 3. `low_ret`

**Distribution**
- mean = -0.00248, std = 0.00372, skew = -7.413
- range = [-0.2412, +0.0152], NaN = 0.00%

**Relation avec le label (univariate)**
- Cohen's d   = -0.0477  (small)
- Mutual info = 0.00041

**Top 3 correlations avec autres features**
- `close_ret` : r = +0.645
- `low_ret_5m` : r = +0.567
- `low_ret_30m` : r = +0.511

**Importance dans le baseline**
- gain XGBoost = 0.02677  (rang 13/25)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = +0.000012
- delta logloss sur le subset des paris pris = +0.000044

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.691243   (baseline = 0.691221, delta = -0.000022)
- test_logloss  : 0.690760   (baseline = 0.690918, delta = +0.000159)
- taken         : 1598  (baseline = 1253, delta = -345)
- WR            : 55.32%  (baseline = 55.79%, delta = +0.47pt)
- EV            : +85.00  (baseline = +72.50, **delta = -12.50**)

---

### 4. `bb_position`

**Distribution**
- mean = 0.51139, std = 0.32674, skew = -0.076
- range = [-0.5897, +1.5897], NaN = 0.00%

**Relation avec le label (univariate)**
- Cohen's d   = -0.1256  (small)
- Mutual info = 0.00204

**Top 3 correlations avec autres features**
- `dist_ema_20` : r = +0.707
- `mfi_norm` : r = +0.666
- `macd_hist` : r = +0.589

**Importance dans le baseline**
- gain XGBoost = 0.06144  (rang 4/25)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = +0.000211
- delta logloss sur le subset des paris pris = +0.000214

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.691192   (baseline = 0.691221, delta = +0.000029)
- test_logloss  : 0.690910   (baseline = 0.690918, delta = +0.000008)
- taken         : 1104  (baseline = 1253, delta = +149)
- WR            : 55.53%  (baseline = 55.79%, delta = +0.26pt)
- EV            : +61.00  (baseline = +72.50, **delta = +11.50**)

---

### 5. `body_ratio_close_30m`

**Distribution**
- mean = 0.45028, std = 0.27069, skew = +0.175
- range = [+0.0000, +1.0000], NaN = 0.01%

**Relation avec le label (univariate)**
- Cohen's d   = +0.0020  (small)
- Mutual info = 0.00003

**Top 3 correlations avec autres features**
- `autocorr_24` : r = +0.098
- `volume_ratio_30m` : r = +0.079
- `high_ret` : r = +0.049

**Importance dans le baseline**
- gain XGBoost = 0.02653  (rang 17/25)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = +0.000034
- delta logloss sur le subset des paris pris = -0.000022

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.691189   (baseline = 0.691221, delta = +0.000032)
- test_logloss  : 0.690893   (baseline = 0.690918, delta = +0.000026)
- taken         : 1521  (baseline = 1253, delta = -268)
- WR            : 55.42%  (baseline = 55.79%, delta = +0.36pt)
- EV            : +82.50  (baseline = +72.50, **delta = -10.00**)

---

### 6. `dist_ema_20`

**Distribution**
- mean = 0.00006, std = 0.00806, skew = -1.550
- range = [-0.3060, +0.1267], NaN = 0.00%

**Relation avec le label (univariate)**
- Cohen's d   = -0.0969  (small)
- Mutual info = 0.00182

**Top 3 correlations avec autres features**
- `macd` : r = +0.772
- `macd_hist` : r = +0.739
- `bb_position` : r = +0.707

**Importance dans le baseline**
- gain XGBoost = 0.14052  (rang 2/25)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = +0.000488
- delta logloss sur le subset des paris pris = +0.000893

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.691326   (baseline = 0.691221, delta = -0.000104)
- test_logloss  : 0.690813   (baseline = 0.690918, delta = +0.000105)
- taken         : 1404  (baseline = 1253, delta = -151)
- WR            : 55.84%  (baseline = 55.79%, delta = -0.05pt)
- EV            : +82.00  (baseline = +72.50, **delta = -9.50**)

---

### 7. `close_ret_30m`

**Distribution**
- mean = 0.00004, std = 0.00563, skew = -0.334
- range = [-0.1619, +0.1359], NaN = 0.00%

**Relation avec le label (univariate)**
- Cohen's d   = -0.0816  (small)
- Mutual info = 0.00150

**Top 3 correlations avec autres features**
- `return_3` : r = +0.803
- `low_ret_30m` : r = +0.641
- `dist_ema_20` : r = +0.538

**Importance dans le baseline**
- gain XGBoost = 0.06376  (rang 3/25)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = +0.000176
- delta logloss sur le subset des paris pris = +0.000217

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.691158   (baseline = 0.691221, delta = +0.000064)
- test_logloss  : 0.690699   (baseline = 0.690918, delta = +0.000219)
- taken         : 1335  (baseline = 1253, delta = -82)
- WR            : 56.10%  (baseline = 55.79%, delta = -0.32pt)
- EV            : +81.50  (baseline = +72.50, **delta = -9.00**)

---

### 8. `dist_to_high_50`

**Distribution**
- mean = -0.01683, std = 0.02015, skew = -3.786
- range = [-0.5995, +0.0000], NaN = 0.00%

**Relation avec le label (univariate)**
- Cohen's d   = -0.0515  (small)
- Mutual info = 0.00069

**Top 3 correlations avec autres features**
- `macd` : r = +0.666
- `dist_ema_20` : r = +0.585
- `low_ret_1h` : r = +0.575

**Importance dans le baseline**
- gain XGBoost = 0.02864  (rang 11/25)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = -0.000022
- delta logloss sur le subset des paris pris = -0.000137

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.691141   (baseline = 0.691221, delta = +0.000080)
- test_logloss  : 0.690846   (baseline = 0.690918, delta = +0.000072)
- taken         : 1432  (baseline = 1253, delta = -179)
- WR            : 55.66%  (baseline = 55.79%, delta = +0.13pt)
- EV            : +81.00  (baseline = +72.50, **delta = -8.50**)

---

### 9. `obv_pressure_5m`

**Distribution**
- mean = -0.00030, std = 0.46192, skew = -0.002
- range = [-1.0000, +1.0000], NaN = 0.00%

**Relation avec le label (univariate)**
- Cohen's d   = -0.0870  (small)
- Mutual info = 0.00096

**Top 3 correlations avec autres features**
- `bb_position` : r = +0.482
- `return_3` : r = +0.456
- `close_ret_30m` : r = +0.370

**Importance dans le baseline**
- gain XGBoost = 0.02451  (rang 22/25)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = -0.000003
- delta logloss sur le subset des paris pris = -0.000253

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.691235   (baseline = 0.691221, delta = -0.000014)
- test_logloss  : 0.690818   (baseline = 0.690918, delta = +0.000100)
- taken         : 1446  (baseline = 1253, delta = -193)
- WR            : 55.53%  (baseline = 55.79%, delta = +0.25pt)
- EV            : +80.00  (baseline = +72.50, **delta = -7.50**)

---

### 10. `dist_to_low_50`

**Distribution**
- mean = 0.01915, std = 0.01955, skew = +3.063
- range = [+0.0000, +0.3411], NaN = 0.00%

**Relation avec le label (univariate)**
- Cohen's d   = -0.0446  (small)
- Mutual info = 0.00075

**Top 3 correlations avec autres features**
- `macd` : r = +0.477
- `dist_ema_20` : r = +0.465
- `high_ret` : r = +0.396

**Importance dans le baseline**
- gain XGBoost = 0.03052  (rang 8/25)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = +0.000067
- delta logloss sur le subset des paris pris = +0.000103

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.691164   (baseline = 0.691221, delta = +0.000058)
- test_logloss  : 0.690885   (baseline = 0.690918, delta = +0.000033)
- taken         : 1506  (baseline = 1253, delta = -253)
- WR            : 54.38%  (baseline = 55.79%, delta = +1.40pt)
- EV            : +66.00  (baseline = +72.50, **delta = +6.50**)

---

### 11. `close_ret`

**Distribution**
- mean = 0.00002, std = 0.00411, skew = +0.062
- range = [-0.1447, +0.1968], NaN = 0.00%

**Relation avec le label (univariate)**
- Cohen's d   = -0.0723  (small)
- Mutual info = 0.00098

**Top 3 correlations avec autres features**
- `low_ret` : r = +0.645
- `high_ret` : r = +0.641
- `return_3` : r = +0.562

**Importance dans le baseline**
- gain XGBoost = 0.03493  (rang 6/25)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = +0.000236
- delta logloss sur le subset des paris pris = -0.000118

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.691260   (baseline = 0.691221, delta = -0.000039)
- test_logloss  : 0.690881   (baseline = 0.690918, delta = +0.000037)
- taken         : 1216  (baseline = 1253, delta = +37)
- WR            : 55.43%  (baseline = 55.79%, delta = +0.36pt)
- EV            : +66.00  (baseline = +72.50, **delta = +6.50**)

---

### 12. `low_ret_30m`

**Distribution**
- mean = -0.00355, std = 0.00520, skew = -6.763
- range = [-0.2412, +0.0152], NaN = 0.00%

**Relation avec le label (univariate)**
- Cohen's d   = -0.0542  (small)
- Mutual info = 0.00071

**Top 3 correlations avec autres features**
- `close_ret_30m` : r = +0.641
- `return_3` : r = +0.530
- `low_ret_1h` : r = +0.524

**Importance dans le baseline**
- gain XGBoost = 0.02403  (rang 23/25)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = -0.000010
- delta logloss sur le subset des paris pris = +0.000076

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.691219   (baseline = 0.691221, delta = +0.000002)
- test_logloss  : 0.690859   (baseline = 0.690918, delta = +0.000059)
- taken         : 1235  (baseline = 1253, delta = +18)
- WR            : 55.38%  (baseline = 55.79%, delta = +0.40pt)
- EV            : +66.50  (baseline = +72.50, **delta = +6.00**)

---

### 13. `hour_sin`

**Distribution**
- mean = -0.00133, std = 0.70695, skew = +0.003
- range = [-1.0000, +1.0000], NaN = 0.00%

**Relation avec le label (univariate)**
- Cohen's d   = -0.0138  (small)
- Mutual info = 0.00005

**Top 3 correlations avec autres features**
- `low_ret_1h` : r = +0.055
- `low_ret_5m` : r = +0.053
- `low_ret_30m` : r = +0.053

**Importance dans le baseline**
- gain XGBoost = 0.02667  (rang 15/25)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = +0.000026
- delta logloss sur le subset des paris pris = -0.000174

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.691259   (baseline = 0.691221, delta = -0.000038)
- test_logloss  : 0.690908   (baseline = 0.690918, delta = +0.000010)
- taken         : 1383  (baseline = 1253, delta = -130)
- WR            : 55.60%  (baseline = 55.79%, delta = +0.18pt)
- EV            : +77.50  (baseline = +72.50, **delta = -5.00**)

---

### 14. `high_ret_5m`

**Distribution**
- mean = 0.00121, std = 0.00203, skew = +8.484
- range = [-0.0081, +0.1713], NaN = 0.00%

**Relation avec le label (univariate)**
- Cohen's d   = -0.0374  (small)
- Mutual info = 0.00029

**Top 3 correlations avec autres features**
- `close_ret_5m` : r = +0.641
- `high_ret` : r = +0.509
- `dist_to_low_50` : r = +0.315

**Importance dans le baseline**
- gain XGBoost = 0.02671  (rang 14/25)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = +0.000013
- delta logloss sur le subset des paris pris = +0.000092

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.691217   (baseline = 0.691221, delta = +0.000004)
- test_logloss  : 0.690746   (baseline = 0.690918, delta = +0.000173)
- taken         : 1293  (baseline = 1253, delta = -40)
- WR            : 55.99%  (baseline = 55.79%, delta = -0.21pt)
- EV            : +77.50  (baseline = +72.50, **delta = -5.00**)

---

### 15. `macd`

**Distribution**
- mean = 0.00005, std = 0.00420, skew = -1.025
- range = [-0.1129, +0.0391], NaN = 0.00%

**Relation avec le label (univariate)**
- Cohen's d   = -0.0433  (small)
- Mutual info = 0.00038

**Top 3 correlations avec autres features**
- `dist_ema_20` : r = +0.772
- `dist_to_high_50` : r = +0.666
- `mfi_norm` : r = +0.505

**Importance dans le baseline**
- gain XGBoost = 0.02567  (rang 19/25)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = +0.000002
- delta logloss sur le subset des paris pris = +0.000034

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.691204   (baseline = 0.691221, delta = +0.000017)
- test_logloss  : 0.690837   (baseline = 0.690918, delta = +0.000082)
- taken         : 1441  (baseline = 1253, delta = -188)
- WR            : 55.38%  (baseline = 55.79%, delta = +0.41pt)
- EV            : +77.50  (baseline = +72.50, **delta = -5.00**)

---

### 16. `high_ret`

**Distribution**
- mean = 0.00227, std = 0.00337, skew = +7.117
- range = [-0.0090, +0.2033], NaN = 0.00%

**Relation avec le label (univariate)**
- Cohen's d   = -0.0330  (small)
- Mutual info = 0.00027

**Top 3 correlations avec autres features**
- `close_ret` : r = +0.641
- `high_ret_5m` : r = +0.509
- `dist_to_low_50` : r = +0.396

**Importance dans le baseline**
- gain XGBoost = 0.02456  (rang 21/25)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = +0.000007
- delta logloss sur le subset des paris pris = +0.000143

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.691210   (baseline = 0.691221, delta = +0.000011)
- test_logloss  : 0.690784   (baseline = 0.690918, delta = +0.000134)
- taken         : 1523  (baseline = 1253, delta = -270)
- WR            : 54.43%  (baseline = 55.79%, delta = +1.35pt)
- EV            : +67.50  (baseline = +72.50, **delta = +5.00**)

---

### 17. `macd_hist`

**Distribution**
- mean = -0.00000, std = 0.00129, skew = -0.251
- range = [-0.0347, +0.0322], NaN = 0.00%

**Relation avec le label (univariate)**
- Cohen's d   = -0.0773  (small)
- Mutual info = 0.00089

**Top 3 correlations avec autres features**
- `dist_ema_20` : r = +0.739
- `close_ret_1h` : r = +0.645
- `bb_position` : r = +0.589

**Importance dans le baseline**
- gain XGBoost = 0.02914  (rang 9/25)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = +0.000025
- delta logloss sur le subset des paris pris = -0.000238

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.691229   (baseline = 0.691221, delta = -0.000008)
- test_logloss  : 0.690906   (baseline = 0.690918, delta = +0.000013)
- taken         : 1308  (baseline = 1253, delta = -55)
- WR            : 55.81%  (baseline = 55.79%, delta = -0.02pt)
- EV            : +76.00  (baseline = +72.50, **delta = -3.50**)

---

### 18. `close_ret_5m`

**Distribution**
- mean = 0.00001, std = 0.00242, skew = -0.554
- range = [-0.1214, +0.0852], NaN = 0.00%

**Relation avec le label (univariate)**
- Cohen's d   = -0.0701  (small)
- Mutual info = 0.00078

**Top 3 correlations avec autres features**
- `low_ret_5m` : r = +0.646
- `high_ret_5m` : r = +0.641
- `close_ret` : r = +0.527

**Importance dans le baseline**
- gain XGBoost = 0.03818  (rang 5/25)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = +0.000160
- delta logloss sur le subset des paris pris = +0.000644

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.691237   (baseline = 0.691221, delta = -0.000016)
- test_logloss  : 0.690964   (baseline = 0.690918, delta = -0.000046)
- taken         : 1564  (baseline = 1253, delta = -311)
- WR            : 54.41%  (baseline = 55.79%, delta = +1.37pt)
- EV            : +69.00  (baseline = +72.50, **delta = +3.50**)

---

### 19. `return_3`

**Distribution**
- mean = 0.00006, std = 0.00689, skew = +0.014
- range = [-0.2424, +0.2939], NaN = 0.00%

**Relation avec le label (univariate)**
- Cohen's d   = -0.0983  (small)
- Mutual info = 0.00193

**Top 3 correlations avec autres features**
- `close_ret_30m` : r = +0.803
- `dist_ema_20` : r = +0.667
- `acc_4` : r = +0.611

**Importance dans le baseline**
- gain XGBoost = 0.15642  (rang 1/25)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = +0.000400
- delta logloss sur le subset des paris pris = +0.000875

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.691227   (baseline = 0.691221, delta = -0.000006)
- test_logloss  : 0.691010   (baseline = 0.690918, delta = -0.000091)
- taken         : 1224  (baseline = 1253, delta = +29)
- WR            : 56.13%  (baseline = 55.79%, delta = -0.34pt)
- EV            : +75.00  (baseline = +72.50, **delta = -2.50**)

---

### 20. `mfi_norm`

**Distribution**
- mean = -0.00600, std = 0.40819, skew = +0.026
- range = [-1.0000, +1.0000], NaN = 0.00%

**Relation avec le label (univariate)**
- Cohen's d   = -0.0714  (small)
- Mutual info = 0.00065

**Top 3 correlations avec autres features**
- `bb_position` : r = +0.666
- `dist_ema_20` : r = +0.554
- `macd` : r = +0.505

**Importance dans le baseline**
- gain XGBoost = 0.02110  (rang 25/25)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = +0.000001
- delta logloss sur le subset des paris pris = -0.000014

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.691178   (baseline = 0.691221, delta = +0.000043)
- test_logloss  : 0.690878   (baseline = 0.690918, delta = +0.000041)
- taken         : 1265  (baseline = 1253, delta = -12)
- WR            : 55.89%  (baseline = 55.79%, delta = -0.10pt)
- EV            : +74.50  (baseline = +72.50, **delta = -2.00**)

---

### 21. `low_ret_5m`

**Distribution**
- mean = -0.00131, std = 0.00217, skew = -6.796
- range = [-0.1475, +0.0152], NaN = 0.00%

**Relation avec le label (univariate)**
- Cohen's d   = -0.0532  (small)
- Mutual info = 0.00041

**Top 3 correlations avec autres features**
- `close_ret_5m` : r = +0.646
- `low_ret` : r = +0.567
- `low_ret_30m` : r = +0.395

**Importance dans le baseline**
- gain XGBoost = 0.02457  (rang 20/25)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = +0.000005
- delta logloss sur le subset des paris pris = +0.000055

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.691158   (baseline = 0.691221, delta = +0.000063)
- test_logloss  : 0.690799   (baseline = 0.690918, delta = +0.000120)
- taken         : 1267  (baseline = 1253, delta = -14)
- WR            : 55.56%  (baseline = 55.79%, delta = +0.22pt)
- EV            : +70.50  (baseline = +72.50, **delta = +2.00**)

---

### 22. `acc_4`

**Distribution**
- mean = -0.00001, std = 0.01136, skew = +0.448
- range = [-0.3320, +0.4322], NaN = 0.00%

**Relation avec le label (univariate)**
- Cohen's d   = -0.0477  (small)
- Mutual info = 0.00037

**Top 3 correlations avec autres features**
- `return_3` : r = +0.611
- `close_ret_30m` : r = +0.494
- `close_ret` : r = +0.346

**Importance dans le baseline**
- gain XGBoost = 0.02695  (rang 12/25)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = +0.000014
- delta logloss sur le subset des paris pris = +0.000089

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.691144   (baseline = 0.691221, delta = +0.000077)
- test_logloss  : 0.690878   (baseline = 0.690918, delta = +0.000040)
- taken         : 1446  (baseline = 1253, delta = -193)
- WR            : 55.12%  (baseline = 55.79%, delta = +0.67pt)
- EV            : +74.00  (baseline = +72.50, **delta = -1.50**)

---

### 23. `low_ret_1h`

**Distribution**
- mean = -0.00508, std = 0.00723, skew = -5.904
- range = [-0.2648, +0.0152], NaN = 0.00%

**Relation avec le label (univariate)**
- Cohen's d   = -0.0336  (small)
- Mutual info = 0.00024

**Top 3 correlations avec autres features**
- `close_ret_1h` : r = +0.634
- `dist_to_high_50` : r = +0.575
- `low_ret_30m` : r = +0.524

**Importance dans le baseline**
- gain XGBoost = 0.02385  (rang 24/25)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = +0.000008
- delta logloss sur le subset des paris pris = -0.000003

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.691219   (baseline = 0.691221, delta = +0.000002)
- test_logloss  : 0.690916   (baseline = 0.690918, delta = +0.000003)
- taken         : 1366  (baseline = 1253, delta = -113)
- WR            : 55.42%  (baseline = 55.79%, delta = +0.37pt)
- EV            : +74.00  (baseline = +72.50, **delta = -1.50**)

---

### 24. `is_weekend`

**Distribution**
- mean = 0.28425, std = 0.45106, skew = +0.957
- range = [+0.0000, +1.0000], NaN = 0.00%

**Relation avec le label (univariate)**
- Cohen's d   = -0.0083  (small)
- Mutual info = 0.00000

**Top 3 correlations avec autres features**
- `dist_to_low_50` : r = -0.099
- `dist_to_high_50` : r = +0.076
- `low_ret_1h` : r = +0.061

**Importance dans le baseline**
- gain XGBoost = 0.02882  (rang 10/25)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = +0.000000
- delta logloss sur le subset des paris pris = +0.000000

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.691203   (baseline = 0.691221, delta = +0.000018)
- test_logloss  : 0.690939   (baseline = 0.690918, delta = -0.000020)
- taken         : 1419  (baseline = 1253, delta = -166)
- WR            : 55.18%  (baseline = 55.79%, delta = +0.61pt)
- EV            : +73.50  (baseline = +72.50, **delta = -1.00**)

---

### 25. `autocorr_24`

**Distribution**
- mean = -0.10779, std = 0.20730, skew = +0.058
- range = [-0.9001, +0.8695], NaN = 0.00%

**Relation avec le label (univariate)**
- Cohen's d   = -0.0026  (small)
- Mutual info = 0.00001

**Top 3 correlations avec autres features**
- `dist_to_low_50` : r = +0.150
- `dist_to_high_50` : r = -0.138
- `low_ret_1h` : r = -0.111

**Importance dans le baseline**
- gain XGBoost = 0.02660  (rang 16/25)

**Permutation (modele baseline, feature melangee aleatoirement)**
- delta logloss sur tout le test = +0.000017
- delta logloss sur le subset des paris pris = -0.000014

**Ablation : retrain SANS cette feature**
- val_logloss   : 0.691223   (baseline = 0.691221, delta = -0.000002)
- test_logloss  : 0.690935   (baseline = 0.690918, delta = -0.000016)
- taken         : 1460  (baseline = 1253, delta = -207)
- WR            : 54.93%  (baseline = 55.79%, delta = +0.85pt)
- EV            : +72.00  (baseline = +72.50, **delta = +0.50**)

---
