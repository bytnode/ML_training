# Ablation des features DOWN-only

## Setup
- Baseline = 18 features intersect + `is_weekend` = 19 features.
- Pour chaque feature DOWN-only, on l'ajoute (= +1 feature) et on mesure la nouvelle distribution.

## Resultats

| config | n_feat | best_iter | mean | std | max | q99 | q99.9 | importance |
|---|---|---|---|---|---|---|---|---|
| **baseline (F)** | 19 | 34 | 49.77% | 3.43pp | 57.82% | 55.99% | 56.53% | - |
| +body_ratio_close_30m | 20 | 26 | 49.82% | 3.01pp | 56.67% **(-1.15pp ↓)** | 55.22% | 55.63% | 0.0333 |
| +dist_to_high_50 | 20 | 26 | 49.85% | 3.01pp | 55.99% **(-1.83pp ↓)** | 55.35% | 55.58% | 0.0342 |
| +hour_sin | 20 | 32 | 49.79% | 3.35pp | 57.23% **(-0.59pp ↓)** | 55.86% | 56.40% | 0.0320 |
| +low_ret | 20 | 26 | 49.81% | 3.01pp | 56.45% **(-1.37pp ↓)** | 55.19% | 55.59% | 0.0307 |
| +obv_pressure_5m | 20 | 26 | 49.83% | 3.00pp | 56.58% **(-1.24pp ↓)** | 55.23% | 55.60% | 0.0305 |
| +volume_ratio_30m | 20 | 27 | 49.80% | 3.05pp | 56.75% **(-1.08pp ↓)** | 55.35% | 55.90% | 0.0336 |
| **+ALL 6** | 25 | 31 | 49.82% | 3.31pp | 56.57% **(-1.25pp)** | 55.65% | 56.12% | - |
