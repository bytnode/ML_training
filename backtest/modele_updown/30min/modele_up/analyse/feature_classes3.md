# Classement d'impact des features (3 classes, modele deterministe)

- Features analysees : **109**
- AUC modele complet : **0.53713** (sigma 2 seeds = 0.000000, deterministe)
- Config de mesure : `{'subsample': 1.0, 'colsample_bytree': 1.0}` | zone morte DEAD = 0.0003
- Split : 119,546 train / 51,233 val / 17,437 test

## Resume

- **IMPACT POSITIF (utiles)** : 68
- **IMPACT NUL** : 39 (dont 28 redondantes-avec-signal, 11 inertes)
- **IMPACT NEGATIF (nuisibles)** : 2

- **Utiles** (positif + redondant-avec-signal) : **96**
- **Inutiles** (inertes) : **11**
- **Mauvaises** (nuisibles) : **2**

| feature | delta_ablation | standalone_auc | power | classe |
|---|---:|---:|---:|---|
| lower_wick | -0.00203 | 0.4948 | -0.0052 | POSITIF |
| eth_vol_ratio | -0.00161 | 0.4997 | -0.0003 | POSITIF |
| ef_ret16 | -0.00152 | 0.5162 | +0.0162 | POSITIF |
| return_96 | -0.00143 | 0.5052 | +0.0052 | POSITIF |
| rsi_14_4h | -0.00143 | 0.4934 | -0.0066 | POSITIF |
| cmf_20 | -0.00141 | 0.5111 | +0.0111 | POSITIF |
| eth_btc_ratio_mom_8 | -0.00132 | 0.5037 | +0.0037 | POSITIF |
| stoch_k_14 | -0.00129 | 0.5296 | +0.0296 | POSITIF |
| dist_to_low_50 | -0.00125 | 0.5239 | +0.0239 | POSITIF |
| rsi_6 | -0.00120 | 0.5317 | +0.0317 | POSITIF |
| atr_pct_1d | -0.00113 | 0.4960 | -0.0040 | POSITIF |
| dist_ema_20_1h | -0.00108 | 0.5186 | +0.0186 | POSITIF |
| volume_ratio_1d | -0.00107 | 0.5000 | +0.0000 | POSITIF |
| ef_vol_z | -0.00107 | 0.4979 | -0.0021 | POSITIF |
| macd | -0.00106 | 0.5122 | +0.0122 | POSITIF |
| body_ratio_avg | -0.00106 | 0.5022 | +0.0022 | POSITIF |
| ret_48 | -0.00106 | 0.5052 | +0.0052 | POSITIF |
| btc_eth_div | -0.00105 | 0.4982 | -0.0018 | POSITIF |
| range_pct | -0.00103 | 0.5009 | +0.0009 | POSITIF |
| ef_ret96 | -0.00102 | 0.5017 | +0.0017 | POSITIF |
| low_ret_1d | -0.00096 | 0.4992 | -0.0008 | POSITIF |
| ret_24 | -0.00096 | 0.5172 | +0.0172 | POSITIF |
| autocorr_12 | -0.00095 | 0.5008 | +0.0008 | POSITIF |
| ef_ret48 | -0.00091 | 0.5155 | +0.0155 | POSITIF |
| ef_macd_hist | -0.00091 | 0.5089 | +0.0089 | POSITIF |
| efficiency_ratio_20 | -0.00088 | 0.4998 | -0.0002 | POSITIF |
| streak | -0.00088 | 0.5252 | +0.0252 | POSITIF |
| body_pos | -0.00087 | 0.5303 | +0.0303 | POSITIF |
| volume_ratio | -0.00087 | 0.4980 | -0.0020 | POSITIF |
| ef_ret4 | -0.00087 | 0.5273 | +0.0273 | POSITIF |
| body_ratio_avg_1d | -0.00084 | 0.5035 | +0.0035 | POSITIF |
| high_ret_5m | -0.00082 | 0.5073 | +0.0073 | POSITIF |
| high_ret_1d | -0.00082 | 0.4961 | -0.0039 | POSITIF |
| zscore_close_48 | -0.00080 | 0.5238 | +0.0238 | POSITIF |
| ef_upwick | -0.00079 | 0.4977 | -0.0023 | POSITIF |
| acc_8 | -0.00078 | 0.4968 | -0.0032 | POSITIF |
| ef_streak | -0.00077 | 0.5238 | +0.0238 | POSITIF |
| acc_4 | -0.00076 | 0.5155 | +0.0155 | POSITIF |
| parkinson_20 | -0.00075 | 0.5001 | +0.0001 | POSITIF |
| close_ret_1h | -0.00073 | 0.5180 | +0.0180 | POSITIF |
| eth_rsi_14 | -0.00070 | 0.5300 | +0.0300 | POSITIF |
| dist_ema_20 | -0.00069 | 0.5279 | +0.0279 | POSITIF |
| range_rel_20 | -0.00067 | 0.5015 | +0.0015 | POSITIF |
| ef_lowick | -0.00066 | 0.4986 | -0.0014 | POSITIF |
| high_ret | -0.00058 | 0.5154 | +0.0154 | POSITIF |
| ret_1h_6 | -0.00057 | 0.5194 | +0.0194 | POSITIF |
| close_ret_4h | -0.00057 | 0.5094 | +0.0094 | POSITIF |
| ef_ret2 | -0.00055 | 0.5248 | +0.0248 | POSITIF |
| ret_4h_3 | -0.00054 | 0.5094 | +0.0094 | POSITIF |
| eom_14 | -0.00053 | 0.5150 | +0.0150 | POSITIF |
| rv_ratio | -0.00052 | 0.4950 | -0.0050 | POSITIF |
| ef_body | -0.00050 | 0.5262 | +0.0262 | POSITIF |
| ret_12 | -0.00049 | 0.5209 | +0.0209 | POSITIF |
| vol_price_corr_20 | -0.00049 | 0.5113 | +0.0113 | POSITIF |
| wick_asym | -0.00049 | 0.5037 | +0.0037 | POSITIF |
| hour_sin | -0.00049 | 0.5017 | +0.0017 | POSITIF |
| autocorr_24 | -0.00048 | 0.4986 | -0.0014 | POSITIF |
| macd_signal | -0.00046 | 0.5065 | +0.0065 | POSITIF |
| atr_pct_4h | -0.00045 | 0.5017 | +0.0017 | POSITIF |
| dist_vwap_48 | -0.00042 | 0.5218 | +0.0218 | POSITIF |
| cci_20 | -0.00040 | 0.5185 | +0.0185 | POSITIF |
| dist_to_high_50 | -0.00035 | 0.5185 | +0.0185 | POSITIF |
| trend_slope_24 | -0.00035 | 0.5111 | +0.0111 | POSITIF |
| rsi_14_1h | -0.00035 | 0.5174 | +0.0174 | POSITIF |
| body_signed_3 | -0.00033 | 0.5260 | +0.0260 | POSITIF |
| obv_pressure_1d | -0.00032 | 0.5023 | +0.0023 | POSITIF |
| rv_ratio_fast | -0.00032 | 0.5026 | +0.0026 | POSITIF |
| low_ret_5m | -0.00030 | 0.5110 | +0.0110 | POSITIF |
| eth_dist_ema_20 | -0.00030 | 0.5269 | +0.0269 | NUL |
| bb_width | -0.00028 | 0.5002 | +0.0002 | NUL |
| ema_slope_20 | -0.00028 | 0.5145 | +0.0145 | NUL |
| return_3 | -0.00027 | 0.5267 | +0.0267 | NUL |
| obv_pressure_4h | -0.00025 | 0.4951 | -0.0049 | NUL |
| body_ratio_avg_4h | -0.00022 | 0.4996 | -0.0004 | NUL |
| open_ret_1d | -0.00022 | 0.5002 | +0.0002 | NUL |
| ef_ret1 | -0.00021 | 0.5231 | +0.0231 | NUL |
| rv_100 | -0.00021 | 0.5054 | +0.0054 | NUL |
| upper_wick | -0.00019 | 0.5009 | +0.0009 | NUL |
| mfi_norm | -0.00016 | 0.5153 | +0.0153 | NUL |
| bb_position | -0.00013 | 0.5246 | +0.0246 | NUL |
| body_ratio_close_1d | -0.00012 | 0.5036 | +0.0036 | NUL |
| ret_6 | -0.00012 | 0.5233 | +0.0233 | NUL |
| ef_ret8 | -0.00010 | 0.5197 | +0.0197 | NUL |
| obv_pressure | -0.00005 | 0.5131 | +0.0131 | NUL |
| low_ret_1h | -0.00004 | 0.5122 | +0.0122 | NUL |
| hour_cos | -0.00003 | 0.5002 | +0.0002 | NUL |
| dist_ema_100 | -0.00003 | 0.5136 | +0.0136 | NUL |
| rv_20 | -0.00002 | 0.5015 | +0.0015 | NUL |
| macd_hist | -0.00002 | 0.5122 | +0.0122 | NUL |
| low_ret | +0.00000 | 0.5144 | +0.0144 | NUL |
| close_ret | +0.00000 | 0.5250 | +0.0250 | NUL |
| mtf_align | +0.00000 | 0.5062 | +0.0062 | NUL |
| rsi_14 | +0.00000 | 0.5262 | +0.0262 | NUL |
| dist_ema_50 | +0.00000 | 0.5186 | +0.0186 | NUL |
| low_ret_30m | +0.00000 | 0.5144 | +0.0144 | NUL |
| close_ret_30m | +0.00000 | 0.5250 | +0.0250 | NUL |
| eth_stoch_14 | +0.00000 | 0.5314 | +0.0314 | NUL |
| ef_macd | +0.00001 | 0.5155 | +0.0155 | NUL |
| volume_rel_30d | +0.00003 | 0.4983 | -0.0017 | NUL |
| volume_ratio_4h | +0.00004 | 0.4992 | -0.0008 | NUL |
| signed_vol_mom_6 | +0.00009 | 0.5161 | +0.0161 | NUL |
| dist_ema_10_1d | +0.00010 | 0.5033 | +0.0033 | NUL |
| body_ratio_close | +0.00012 | 0.5002 | +0.0002 | NUL |
| force_index_13 | +0.00013 | 0.5227 | +0.0227 | NUL |
| ret_1d_1 | +0.00014 | 0.5026 | +0.0026 | NUL |
| vol_z_20 | +0.00023 | 0.4954 | -0.0046 | NUL |
| close_ret_5m | +0.00027 | 0.5125 | +0.0125 | NUL |
| atr_pct_cur | +0.00036 | 0.4995 | -0.0005 | NEGATIF |
| obv_pressure_5m | +0.00042 | 0.5151 | +0.0151 | NEGATIF |