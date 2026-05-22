# Diff features UP vs DOWN parquet

## Shapes

- DOWN : 480,000 rows, 29 cols
- UP   : 480,000 rows, 24 cols

## Common features (23)

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
  is_weekend
  label
  low_ret_1h
  low_ret_30m
  low_ret_5m
  macd
  macd_hist
  mfi_norm
  return_3
  timestamp
  timestamp_ms
  volatility_state
```

## DOWN-only (6)

```
  body_ratio_close_30m
  dist_to_high_50
  hour_sin
  low_ret
  obv_pressure_5m
  volume_ratio_30m
```

## UP-only (1)

```
  volume_rel_30d
```

## Label distribution

### DOWN

- label=1 : 224738  (0.4921)
- label=0 : 231908  (0.4831)
- label NaN : 23354

### UP

- label=1 : 231908  (0.5079)
- label=0 : 224738  (0.4682)
- label NaN : 23354

## Label coherence check (on common timestamps)

- Common timestamps with both labels valid : 456,646
- label_down + label_up == 1 (inverse, attendu): 456,646 (100.00%)
- label_down + label_up == 0 ou 2 (incoherent): 0 (0.00%)

## Common features value coherence (échantillon 1000 rows)

| feature | DOWN std | UP std | DOWN mean | UP mean | match? |
|---|---|---|---|---|---|
| acc_4 | 0.0124 | 0.0124 | 0.0004 | 0.0004 | YES |
| autocorr_24 | 0.2154 | 0.2154 | -0.1255 | -0.1255 | YES |
| bb_position | 0.3212 | 0.3212 | 0.5116 | 0.5116 | YES |
| close_ret | 0.0043 | 0.0043 | 0.0001 | 0.0001 | YES |
| close_ret_1h | 0.0087 | 0.0087 | -0.0002 | -0.0002 | YES |
| close_ret_30m | 0.0054 | 0.0054 | -0.0002 | -0.0002 | YES |
| close_ret_5m | 0.0024 | 0.0024 | 0.0000 | 0.0000 | YES |
| dist_ema_20 | 0.0074 | 0.0074 | 0.0003 | 0.0003 | YES |
| dist_to_low_50 | 0.0218 | 0.0218 | 0.0195 | 0.0195 | YES |
| high_ret | 0.0039 | 0.0039 | 0.0021 | 0.0021 | YES |
| high_ret_5m | 0.0021 | 0.0021 | 0.0012 | 0.0012 | YES |
| is_weekend | 0.4627 | 0.4627 | 0.3100 | 0.3100 | YES |
| low_ret_1h | 0.0071 | 0.0071 | -0.0053 | -0.0053 | YES |
| low_ret_30m | 0.0054 | 0.0054 | -0.0036 | -0.0036 | YES |
| low_ret_5m | 0.0024 | 0.0024 | -0.0012 | -0.0012 | YES |
| macd | 0.0039 | 0.0039 | 0.0001 | 0.0001 | YES |
| macd_hist | 0.0013 | 0.0013 | 0.0000 | 0.0000 | YES |
| mfi_norm | 0.4224 | 0.4224 | 0.0058 | 0.0058 | YES |
| return_3 | 0.0073 | 0.0073 | 0.0002 | 0.0002 | YES |
| volatility_state | 0.4386 | 0.4386 | 0.9470 | 0.9470 | YES |
