# Pourquoi DOWN compresse vs UP (memes features + memes HP)

## Setup

- 19 features identiques pour les 2 modeles : `['acc_4', 'autocorr_24', 'bb_position', 'close_ret', 'close_ret_1h', 'close_ret_30m', 'close_ret_5m', 'dist_ema_20', 'dist_to_low_50', 'high_ret', 'high_ret_5m', 'low_ret_1h', 'low_ret_30m', 'low_ret_5m', 'macd', 'macd_hist', 'mfi_norm', 'return_3', 'is_weekend']`
- HP identiques : max_depth=5, n_estim=1000, lr=0.05, min_child_weight=20

## Comparaison directe

| metric | UP | DOWN | gap |
|---|---|---|---|
| **val_logloss** | 0.691133 | 0.691136 | **+0.00m** |
| best_iteration | 34 | 34 | +0 |
| n_trees (= best+1) | 35 | 35 | |
| raw_log-odds max | 0.3973 | 0.3155 | -0.0818 |
| raw_log-odds std | 0.1375 | 0.1375 | -0.0001 |
| p max | 0.5980 | 0.5782 | -1.98pp |
| p mean | 0.5022 | 0.4977 | -0.45pp |
| p std | 0.0343 | 0.0343 | -0.00pp |
| label base rate | 0.5066 | 0.4934 | |

## Interpretation

## Reliability comparison (WR observee a chaque bin de p)

| p_bin | UP n | UP WR | DOWN n | DOWN WR | Δ |
|---|---|---|---|---|---|
| [0.4, 0.425) | 1 | 0.00% | 185 | 43.78% | N/A |
| [0.425, 0.45) | 2260 | 42.57% | 2945 | 44.45% | +1.88pp |
| [0.45, 0.475) | 6465 | 46.68% | 6912 | 45.66% | -1.02pp |
| [0.475, 0.5) | 7893 | 49.98% | 8169 | 49.97% | -0.01pp |
| [0.5, 0.525) | 8169 | 50.04% | 7892 | 50.03% | -0.02pp |
| [0.525, 0.55) | 6912 | 54.34% | 6474 | 53.31% | -1.03pp |
| [0.55, 0.575) | 2945 | 55.55% | 2252 | 57.42% | +1.86pp |
| [0.575, 0.6) | 185 | 56.22% | 1 | 100.00% | N/A |
| [0.6, 0.625) | 0 | 0.00% | 0 | 0.00% | N/A |
| [0.625, 0.65) | 0 | 0.00% | 0 | 0.00% | N/A |

## Importance features (memes features, mais le modele les utilise differemment)

| feature | UP imp | DOWN imp | ratio (DOWN/UP) |
|---|---|---|---|
| acc_4 | 0.0363 | 0.0353 | 0.97 |
| autocorr_24 | 0.0357 | 0.0359 | 1.00 |
| bb_position | 0.0585 | 0.0567 | 0.97 |
| close_ret | 0.0485 | 0.0483 | 1.00 |
| close_ret_1h | 0.0448 | 0.0453 | 1.01 |
| close_ret_30m | 0.0799 | 0.0776 | 0.97 |
| close_ret_5m | 0.0488 | 0.0483 | 0.99 |
| dist_ema_20 | 0.1528 | 0.1484 | 0.97 |
| dist_to_low_50 | 0.0367 | 0.0363 | 0.99 |
| high_ret | 0.0351 | 0.0345 | 0.98 |
| high_ret_5m | 0.0362 | 0.0364 | 1.01 |
| low_ret_1h | 0.0292 | 0.0289 | 0.99 |
| low_ret_30m | 0.0340 | 0.0346 | 1.02 |
| low_ret_5m | 0.0327 | 0.0323 | 0.99 |
| macd | 0.0333 | 0.0328 | 0.99 |
| macd_hist | 0.0381 | 0.0391 | 1.02 |
| mfi_norm | 0.0290 | 0.0285 | 0.98 |
| return_3 | 0.1535 | 0.1604 | 1.04 |
| is_weekend | 0.0371 | 0.0404 | 1.09 |
