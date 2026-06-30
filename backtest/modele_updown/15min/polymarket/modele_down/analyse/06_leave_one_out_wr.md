# Leave-One-Out feature impact sur WR (modele DOWN)

## Setup
- Baseline = config CURRENT (25 features, max_depth=5)
- Strategie : quantile (vw=0.01, we=0.1) + hour_boost
- Baseline: n=1485 WR=61.28% EV=+167.50  p_max=57.15%

## Resultats LOO (trie par Δ WR descendant = retirer LES PIRES en haut)

| feature retiree | importance | n_taken | Δn | WR | **Δ WR** | EV | Δ EV | p_max | verdict |
|---|---|---|---|---|---|---|---|---|---|
| `close_ret` | 0.0375 | 1492 | +7 | 61.13% | **-0.15pp** | +166.00 | -1.50 | 57.33% | neutral |
| `dist_ema_20` | 0.1318 | 1398 | -87 | 61.02% | **-0.26pp** | +154.00 | -13.50 | 57.51% | HELPS (retire = -) |
| `macd` | 0.0274 | 1426 | -59 | 60.87% | **-0.41pp** | +155.00 | -12.50 | 57.02% | HELPS (retire = -) |
| `dist_to_low_50` | 0.0300 | 1426 | -59 | 60.80% | **-0.48pp** | +154.00 | -13.50 | 56.98% | HELPS (retire = -) |
| `mfi_norm` | 0.0225 | 1446 | -39 | 60.65% | **-0.63pp** | +154.00 | -13.50 | 56.93% | HELPS (retire = -) |
| `low_ret_30m` | 0.0250 | 1502 | +17 | 60.52% | **-0.76pp** | +158.00 | -9.50 | 56.36% | HELPS (retire = -) |
| `low_ret_5m` | 0.0250 | 1494 | +9 | 60.31% | **-0.97pp** | +154.00 | -13.50 | 56.54% | HELPS (retire = -) |
| `obv_pressure_5m` | 0.0246 | 1549 | +64 | 60.30% | **-0.98pp** | +159.50 | -8.00 | 56.32% | HELPS (retire = -) |
| `dist_to_high_50` | 0.0308 | 1402 | -83 | 60.20% | **-1.08pp** | +143.00 | -24.50 | 57.32% | HELPS (retire = -) |
| `high_ret_5m` | 0.0280 | 1497 | +12 | 60.19% | **-1.09pp** | +152.50 | -15.00 | 57.04% | HELPS (retire = -) |
| `acc_4` | 0.0278 | 1501 | +16 | 60.09% | **-1.19pp** | +151.50 | -16.00 | 57.05% | HELPS (retire = -) |
| `is_weekend` | 0.0303 | 1477 | -8 | 60.05% | **-1.23pp** | +148.50 | -19.00 | 56.71% | HELPS (retire = -) |
| `bb_position` | 0.0583 | 1406 | -79 | 60.03% | **-1.25pp** | +141.00 | -26.50 | 57.41% | HELPS (retire = -) |
| `autocorr_24` | 0.0275 | 1480 | -5 | 60.00% | **-1.28pp** | +148.00 | -19.50 | 57.08% | HELPS (retire = -) |
| `hour_sin` | 0.0268 | 1488 | +3 | 59.95% | **-1.33pp** | +148.00 | -19.50 | 57.05% | HELPS (retire = -) |
| `volume_ratio_30m` | 0.0273 | 1452 | -33 | 59.92% | **-1.36pp** | +144.00 | -23.50 | 57.52% | HELPS (retire = -) |
| `close_ret_1h` | 0.0344 | 1642 | +157 | 59.81% | **-1.47pp** | +161.00 | -6.50 | 55.48% | HELPS (retire = -) |
| `macd_hist` | 0.0294 | 1588 | +103 | 59.57% | **-1.71pp** | +152.00 | -15.50 | 55.82% | HELPS (retire = -) |
| `close_ret_5m` | 0.0408 | 1624 | +139 | 59.54% | **-1.74pp** | +155.00 | -12.50 | 55.86% | HELPS (retire = -) |
| `low_ret` | 0.0276 | 1515 | +30 | 59.34% | **-1.94pp** | +141.50 | -26.00 | 56.24% | HELPS (retire = -) |
| `close_ret_30m` | 0.0626 | 1588 | +103 | 59.26% | **-2.02pp** | +147.00 | -20.50 | 56.03% | HELPS (retire = -) |
| `body_ratio_close_30m` | 0.0267 | 1560 | +75 | 59.23% | **-2.05pp** | +144.00 | -23.50 | 56.26% | HELPS (retire = -) |
| `low_ret_1h` | 0.0253 | 1596 | +111 | 59.09% | **-2.19pp** | +145.00 | -22.50 | 56.16% | HELPS (retire = -) |
| `return_3` | 0.1467 | 1494 | +9 | 58.84% | **-2.44pp** | +132.00 | -35.50 | 56.76% | HELPS (retire = -) |
| `high_ret` | 0.0260 | 1535 | +50 | 58.76% | **-2.52pp** | +134.50 | -33.00 | 56.31% | HELPS (retire = -) |

## Synthese
- **Features qui NUISENT** (retirer ameliore WR de plus de 0.2pp) : **0**
- **Features qui AIDENT** (retirer baisse WR de plus de 0.2pp) : **24**
  - `high_ret` : retirer → WR 61.28% → 58.76% (-2.52pp)
  - `low_ret` : retirer → WR 61.28% → 59.34% (-1.94pp)
  - `high_ret_5m` : retirer → WR 61.28% → 60.19% (-1.09pp)
  - `low_ret_5m` : retirer → WR 61.28% → 60.31% (-0.97pp)
  - `close_ret_5m` : retirer → WR 61.28% → 59.54% (-1.74pp)
  - `obv_pressure_5m` : retirer → WR 61.28% → 60.30% (-0.98pp)
  - `low_ret_30m` : retirer → WR 61.28% → 60.52% (-0.76pp)
  - `close_ret_30m` : retirer → WR 61.28% → 59.26% (-2.02pp)
  - `volume_ratio_30m` : retirer → WR 61.28% → 59.92% (-1.36pp)
  - `body_ratio_close_30m` : retirer → WR 61.28% → 59.23% (-2.05pp)
  - `low_ret_1h` : retirer → WR 61.28% → 59.09% (-2.19pp)
  - `close_ret_1h` : retirer → WR 61.28% → 59.81% (-1.47pp)
  - `dist_ema_20` : retirer → WR 61.28% → 61.02% (-0.26pp)
  - `return_3` : retirer → WR 61.28% → 58.84% (-2.44pp)
  - `hour_sin` : retirer → WR 61.28% → 59.95% (-1.33pp)
  - `macd` : retirer → WR 61.28% → 60.87% (-0.41pp)
  - `macd_hist` : retirer → WR 61.28% → 59.57% (-1.71pp)
  - `bb_position` : retirer → WR 61.28% → 60.03% (-1.25pp)
  - `mfi_norm` : retirer → WR 61.28% → 60.65% (-0.63pp)
  - `is_weekend` : retirer → WR 61.28% → 60.05% (-1.23pp)
  - `dist_to_high_50` : retirer → WR 61.28% → 60.20% (-1.08pp)
  - `dist_to_low_50` : retirer → WR 61.28% → 60.80% (-0.48pp)
  - `autocorr_24` : retirer → WR 61.28% → 60.00% (-1.28pp)
  - `acc_4` : retirer → WR 61.28% → 60.09% (-1.19pp)
