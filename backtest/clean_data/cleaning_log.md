# Cleaning log

- Input  : `C:\Users\fremo\Desktop\xgboost_training\training_data.csv`
- Output : `C:\Users\fremo\Desktop\xgboost_training\training_data_clean.csv`
- Rows   : 290,438 (inchange — aucune ligne supprimee)
- Cols avant : 24, Cols apres : 33

## Winsorization

Quantiles utilises : p0.5 / p99.5

| feature | borne basse | borne haute | clipped bas | clipped haut |
|---|---|---|---|---|
| volume_ratio_last | 0.021566 | 15.549313 | 1,453 | 1,453 |
| volume_ratio_avg | 0.029439 | 11.747137 | 1,453 | 1,453 |
| obv_slope_10 | -0.060168 | 0.072807 | 1,452 | 1,452 |
| move_pct | -3.771021 | 3.657851 | 1,453 | 1,453 |
| move_atr | 0.044542 | 5.025626 | 1,453 | 1,453 |
| rsi_change_5 | -30.527666 | 29.525521 | 1,452 | 1,453 |
| atr_ratio | 0.424414 | 2.094215 | 1,452 | 1,453 |
| dist_ema20 | -4.503354 | 4.145794 | 1,452 | 1,450 |
| dist_ema50 | -6.387706 | 5.883001 | 1,453 | 1,453 |
| dist_ema200 | -11.441774 | 11.613435 | 1,453 | 1,452 |

## Sample weights

- alpha = 0.15 (decroissance exponentielle par annee de recul)
- min / mean / max : 0.2880 / 1.0000 / 2.2431
- p10 / p90 : 0.3661 / 1.8903
- Une row de 2026 pese ~2.24 ; une row de 2012 pese ~0.29
- Pour utiliser : `model.fit(X, y, sample_weight=df['sample_weight'])`

## Features ajoutees

- `volume_ratio_last_log`
- `volume_ratio_avg_log`
- `wick_imbalance`
- `dist_ema_aligned`
- `rsi_extreme`
- `move_signed_atr`
- `dist_ema20_vs_ema200`

## Colonnes meta ajoutees (a exclure des features X dans train.py)

- `era_year` : annee de la row, utile comme feature OU comme groupe pour CV temporelle
- `sample_weight` : NE PAS inclure dans X — passer au .fit() via sample_weight=...

## Comment l'utiliser dans train.py

```python
# remplacer CSV_PATH par 'training_data_clean.csv' dans config.py
df = pd.read_csv(CSV_PATH)
# exclure timestamp, label, ET sample_weight des features
feature_cols = [c for c in df.columns
                if c not in ('timestamp', 'label', 'sample_weight')]
# era_year reste DANS X — c'est une feature volontaire
model.fit(X_train, y_train,
          sample_weight=train['sample_weight'],
          eval_set=[(X_val, y_val)],
          sample_weight_eval_set=[val['sample_weight']])
```