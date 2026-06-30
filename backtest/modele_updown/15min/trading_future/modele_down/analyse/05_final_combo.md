# Combo final : ablation features + max_depth

## Strategie
- vol_frac_weekday = 0.01, vol_frac_weekend = 0.1
- skip = [(2, 0), (2, 1)], hour_boost = [20, 21, 22] @ 0.02

## Resultats

| config | n_feat | best_iter | T_wd | T_we | n_taken | %vol | WR | p_mean_taken | p_max | p_wins/loss |
|---|---|---|---|---|---|---|---|---|---|---|
| CURRENT (full features, max_depth=5) | 25 | 31 | 0.5581 | 0.5408 | 1485 | 4.26% | **61.28%** | **54.73%** | 57.15% | 54.74/54.70% |
| clean features, max_depth=5 | 19 | 34 | 0.5605 | 0.5459 | 1423 | 4.09% | **59.94%** | **55.14%** | 57.82% | 55.17/55.11% |
| clean features, max_depth=6 | 19 | 34 | 0.5636 | 0.5485 | 1391 | 3.99% | **59.09%** | **55.54%** | 58.60% | 55.55/55.52% |
| clean features, max_depth=7 | 19 | 23 | 0.5548 | 0.5414 | 1512 | 4.34% | **59.46%** | **54.58%** | 57.93% | 54.60/54.56% |
| clean features, max_depth=8 | 19 | 18 | 0.5505 | 0.5372 | 1511 | 4.34% | **58.84%** | **54.14%** | 58.34% | 54.18/54.10% |
| full features, max_depth=6 | 25 | 26 | 0.5574 | 0.5408 | 1460 | 4.19% | **60.68%** | **54.63%** | 57.42% | 54.63/54.62% |
