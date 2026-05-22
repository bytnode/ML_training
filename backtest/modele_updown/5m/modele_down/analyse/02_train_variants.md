# Audit des variantes UP/DOWN

## Hyperparams (identiques pour toutes les variantes)
```
{
  "n_estimators": 1000,
  "max_depth": 5,
  "learning_rate": 0.05,
  "min_child_weight": 20,
  "subsample": 0.8,
  "colsample_bytree": 0.8,
  "objective": "binary:logistic",
  "eval_metric": "logloss",
  "early_stopping_rounds": 50,
  "random_state": 42,
  "n_jobs": -1,
  "gamma": 1.5
}
```

## Splits
- DOWN: train=339,475 val=37,719 test=34,830
- UP  : train=339,475 val=37,719 test=34,830

## Distributions p_positive sur test

| variant | n_feat | best_iter | mean | std | max | q90 | q99 | q99.9 |
|---|---|---|---|---|---|---|---|---|
| **A_UP_full** | 19 | 34 | 50.29% | 3.38pp | **59.39%** | 54.82% | 57.17% | 58.08% |
| **B_UP_no_vol30d** | 18 | 30 | 50.20% | 3.21pp | **59.14%** | 54.44% | 56.65% | 57.86% |
| **C_DOWN_full** | 25 | 28 | 49.81% | 3.15pp | **56.38%** | 54.20% | 55.42% | 55.88% |
| **D_DOWN_intersect_excl_isweekend** | 18 | 30 | 49.80% | 3.20pp | **56.60%** | 54.23% | 55.50% | 56.14% |
| **E_DOWN_no_isweekend** | 24 | 27 | 49.85% | 3.10pp | **56.42%** | 54.12% | 55.40% | 55.70% |
| **F_DOWN_intersect_with_isweekend** | 19 | 34 | 49.77% | 3.43pp | **57.82%** | 54.52% | 55.99% | 56.53% |

## Reliability per variant (WR observee par bin de p_positive)

### A_UP_full

| p_bin | n | WR |
|---|---|---|
| [0.400, 0.425) | 0 | 0.00% |
| [0.425, 0.450) | 2074 | 41.61% |
| [0.450, 0.475) | 6198 | 46.92% |
| [0.475, 0.500) | 8224 | 49.49% |
| [0.500, 0.525) | 8193 | 50.26% |
| [0.525, 0.550) | 7003 | 54.39% |
| [0.550, 0.575) | 2973 | 55.47% |
| [0.575, 0.600) | 165 | 55.76% |
| [0.600, 0.625) | 0 | 0.00% |
| [0.625, 0.650) | 0 | 0.00% |

### B_UP_no_vol30d

| p_bin | n | WR |
|---|---|---|
| [0.400, 0.425) | 0 | 0.00% |
| [0.425, 0.450) | 1371 | 42.52% |
| [0.450, 0.475) | 7262 | 46.02% |
| [0.475, 0.500) | 7892 | 49.54% |
| [0.500, 0.525) | 8719 | 50.73% |
| [0.525, 0.550) | 7247 | 54.22% |
| [0.550, 0.575) | 2277 | 56.35% |
| [0.575, 0.600) | 62 | 62.90% |
| [0.600, 0.625) | 0 | 0.00% |
| [0.625, 0.650) | 0 | 0.00% |

### C_DOWN_full

| p_bin | n | WR |
|---|---|---|
| [0.400, 0.425) | 13 | 23.08% |
| [0.425, 0.450) | 2179 | 43.55% |
| [0.450, 0.475) | 7157 | 45.72% |
| [0.475, 0.500) | 8834 | 49.17% |
| [0.500, 0.525) | 8313 | 51.00% |
| [0.525, 0.550) | 7145 | 53.76% |
| [0.550, 0.575) | 1189 | 56.52% |
| [0.575, 0.600) | 0 | 0.00% |
| [0.600, 0.625) | 0 | 0.00% |
| [0.625, 0.650) | 0 | 0.00% |

### D_DOWN_intersect_excl_isweekend

| p_bin | n | WR |
|---|---|---|
| [0.400, 0.425) | 58 | 36.21% |
| [0.425, 0.450) | 2251 | 44.16% |
| [0.450, 0.475) | 7322 | 45.55% |
| [0.475, 0.500) | 8679 | 49.42% |
| [0.500, 0.525) | 7893 | 50.41% |
| [0.525, 0.550) | 7265 | 54.00% |
| [0.550, 0.575) | 1362 | 57.27% |
| [0.575, 0.600) | 0 | 0.00% |
| [0.600, 0.625) | 0 | 0.00% |
| [0.625, 0.650) | 0 | 0.00% |

### E_DOWN_no_isweekend

| p_bin | n | WR |
|---|---|---|
| [0.400, 0.425) | 67 | 37.31% |
| [0.425, 0.450) | 1915 | 44.33% |
| [0.450, 0.475) | 7129 | 44.94% |
| [0.475, 0.500) | 8943 | 49.28% |
| [0.500, 0.525) | 8473 | 50.78% |
| [0.525, 0.550) | 7157 | 54.32% |
| [0.550, 0.575) | 1146 | 56.28% |
| [0.575, 0.600) | 0 | 0.00% |
| [0.600, 0.625) | 0 | 0.00% |
| [0.625, 0.650) | 0 | 0.00% |

### F_DOWN_intersect_with_isweekend

| p_bin | n | WR |
|---|---|---|
| [0.400, 0.425) | 185 | 43.78% |
| [0.425, 0.450) | 2945 | 44.45% |
| [0.450, 0.475) | 6912 | 45.66% |
| [0.475, 0.500) | 8169 | 49.97% |
| [0.500, 0.525) | 7892 | 50.03% |
| [0.525, 0.550) | 6474 | 53.31% |
| [0.550, 0.575) | 2252 | 57.42% |
| [0.575, 0.600) | 1 | 100.00% |
| [0.600, 0.625) | 0 | 0.00% |
| [0.625, 0.650) | 0 | 0.00% |


## Features par variant

### A_UP_full (19)
```
  acc_4
  autocorr_24
  bb_position
  close_ret
  close_ret_1h
  close_ret_30m
  close_ret_5m
  dist_ema_20
  dist_to_low_50
  high_ret
  high_ret_5m
  low_ret_1h
  low_ret_30m
  low_ret_5m
  macd
  macd_hist
  mfi_norm
  return_3
  volume_rel_30d
```

### B_UP_no_vol30d (18)
```
  acc_4
  autocorr_24
  bb_position
  close_ret
  close_ret_1h
  close_ret_30m
  close_ret_5m
  dist_ema_20
  dist_to_low_50
  high_ret
  high_ret_5m
  low_ret_1h
  low_ret_30m
  low_ret_5m
  macd
  macd_hist
  mfi_norm
  return_3
```

### C_DOWN_full (25)
```
  acc_4
  autocorr_24
  bb_position
  body_ratio_close_30m
  close_ret
  close_ret_1h
  close_ret_30m
  close_ret_5m
  dist_ema_20
  dist_to_high_50
  dist_to_low_50
  high_ret
  high_ret_5m
  hour_sin
  is_weekend
  low_ret
  low_ret_1h
  low_ret_30m
  low_ret_5m
  macd
  macd_hist
  mfi_norm
  obv_pressure_5m
  return_3
  volume_ratio_30m
```

### D_DOWN_intersect_excl_isweekend (18)
```
  acc_4
  autocorr_24
  bb_position
  close_ret
  close_ret_1h
  close_ret_30m
  close_ret_5m
  dist_ema_20
  dist_to_low_50
  high_ret
  high_ret_5m
  low_ret_1h
  low_ret_30m
  low_ret_5m
  macd
  macd_hist
  mfi_norm
  return_3
```

### E_DOWN_no_isweekend (24)
```
  acc_4
  autocorr_24
  bb_position
  body_ratio_close_30m
  close_ret
  close_ret_1h
  close_ret_30m
  close_ret_5m
  dist_ema_20
  dist_to_high_50
  dist_to_low_50
  high_ret
  high_ret_5m
  hour_sin
  low_ret
  low_ret_1h
  low_ret_30m
  low_ret_5m
  macd
  macd_hist
  mfi_norm
  obv_pressure_5m
  return_3
  volume_ratio_30m
```

### F_DOWN_intersect_with_isweekend (19)
```
  acc_4
  autocorr_24
  bb_position
  close_ret
  close_ret_1h
  close_ret_30m
  close_ret_5m
  dist_ema_20
  dist_to_low_50
  high_ret
  high_ret_5m
  low_ret_1h
  low_ret_30m
  low_ret_5m
  macd
  macd_hist
  mfi_norm
  return_3
  is_weekend
```

