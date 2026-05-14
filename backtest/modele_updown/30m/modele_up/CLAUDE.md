# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

This file is package-scoped: see the repo-root `CLAUDE.md` (`../../../CLAUDE.md`) for repo-wide conventions and for `modele_15m`. The sister package `modele_down/` predicts the opposite direction with the same pipeline.

## What this package is

`modele_up` is an **UP-only** XGBoost binary classifier. On the close of each 15m BTC/USD bar it predicts:

```
label = 1  if the NEXT 15m candle is green (close[i+1] > open[i+1])
label = 0  if the NEXT 15m candle is red
```

The model output is `P(next is up)` and is treated as a single confidence axis. All bets the model emits are UP bets — there is no DOWN head and no two-sided arbitration. Up-bias is by construction, not a feature of the data.

Per-bucket adaptive threshold:
```
take UP bet  iff  P(up) >= ADAPTIVE_TARGET_WR + delta(volatility_state, is_weekend)
```
6 buckets total (vol ∈ {0,1,2} × wknd ∈ {0,1}), one delta per bucket, written to `btc_updown_model.thresholds.json`. Sidecar shape : `{"deltas": {"v_w": float, ...}, "metadata": {...}}` and `predict.decide()` / `api.evaluate_on_slice()` assume that shape.

## Working-directory contract — `cd backtest/modele_updown/modele_up/` first

`config.py` uses **relative paths**: `DB_PATH = "DB/btc_multi_tf_15m.parquet"` and `MODEL_OUTPUT = "btc_updown_model.json"`. The patch scripts also do `Path(__file__).resolve().parent / "DB" / ...`, so running from the package root is fine, but launching `python backtest/modele_updown/modele_up/train.py` from the repo root will read/write into the wrong place. Always `cd` first.

`logs/` is anchored by `Path(__file__)` (in `train.py`), so it lives under `backtest/modele_updown/modele_up/logs/` regardless. The artifacts (`runs.csv`, `test_predictions.csv`) are gitignored CSVs.

## Build pipeline

The master parquet `DB/btc_multi_tf_15m.parquet` is built by `data.py` :

```powershell
cd backtest\modele_updown\modele_up

# 1. Cold fetch + features. Drops + refetches ~5000 days OHLCV from Bitstamp
#    on 6 timeframes (5m, 15m, 30m, 1h, 4h, 1d), computes 9 normalized features
#    per TF, joins higher TFs no-lookahead, computes is_weekend +
#    volatility_state + label. Writes DB/btc_multi_tf.db + DB/btc_multi_tf_15m.parquet.
#    Takes ~3-5 min.
python data.py

# 2. (optional) Add patch features one by one.
python add_acceleration.py
python add_adx.py
python add_autocorr.py
python add_range_expansion.py
python add_structural_pressure.py
python add_upside_velocity.py
python add_up_down_ratio.py
python add_downside_velocity.py
# Each one is idempotent and reads close/high/low (or atr_pct) from the
# SQLite candles table to compute its feature, then merges into the parquet
# by timestamp_ms.

# 3. Train + evaluate.
python train.py [--label name] [--no-save] [--no-log]

# Inspection
python predict.py             # smoke-test on the last test_predictions row
python api.py                 # evaluate_on_slice on the full window
python check_new_features.py  # full-ablation feature impact analyzer
python compare_runs.py        # diff between recent runs.csv rows
python analyze_features.py    # 7-section feature audit
```

`features.py` is the runtime builder used by `train.py`. It reads the parquet, drops the warmup, drops rows with NaN label, and renames `timestamp_ms -> timestamp` for compatibility. **No on-the-fly label flip** : the parquet already stores the UP label (computed by `data.py` line ~772 `(cr_next > or_next)`).

## What `train.py` writes

- `btc_updown_model.json` — XGBoost native format.
- `btc_updown_model.thresholds.json` — sidecar `{"deltas": {"v_w": float, ...}, "metadata": {...}}`. Base of the threshold = `config.ADAPTIVE_TARGET_WR`.
- `logs/runs.csv` — one row per `train_and_evaluate(log_run=True)` call. Append-only with column-union reindex.
- `logs/test_predictions.csv` — written **only** by direct `python train.py` invocations. Schema : `{full feature row}, label, p_up, confidence` (where `confidence` is an alias for `p_up`). `api.py` reads this file to implement the standardized inference contract.

### `runs.csv` legacy columns — read carefully

This `runs.csv` was copied from the `modele_down` package, so older rows contain `baseline_wr_down` (= P(label=1=down) under the down model). New runs of `modele_up` write `baseline_wr_up` (= P(label=1=up)). The reindex logic preserves both columns. Ne pas mélanger les deux dans une analyse.

## The standardized `api.py` contract

```python
NAME : str = "modele_up"

def evaluate_on_slice(start_ms: int, end_ms: int) -> pd.DataFrame:
    """One row per setup the model would have *taken* in the window.
    Columns: timestamp (int ms), direction ('up' for this package),
             confidence (= P(up)), label (int 0|1, 1 = bet was correct)."""
```

`direction` is always `"up"` for this model.

## Naming traps

- **`config.DB_PATH`** points to the **parquet** `DB/btc_multi_tf_15m.parquet`, not the SQLite (`DB/btc_multi_tf.db`). The SQLite is only used by `data.py` and the `add_*.py` patches.
- **"up" in this package = "green candle next"**, defined as `close[i+1] > open[i+1]`. This is the *direction of the next candle*, not "price went up by some threshold". Do not redefine the label without retraining.
- **"confidence" in `test_predictions.csv` is an alias of `p_up`**, not a calibrated score. Threshold comparisons are always against `ADAPTIVE_TARGET_WR + delta(bucket)`, never against 0.5.
- **Le label inversé sur le parquet** : si tu copies un parquet de `modele_down`, n'oublie pas d'inverser la colonne `label` (`df["label"] = 1 - df["label"]`) sinon le modèle apprend la mauvaise direction. `data.py` calcule directement `(cr_next > or_next)` donc un cold rebuild via `python data.py` produit le bon label.

## Data conventions (same as the root CLAUDE.md)

- **No-lookahead joins**: every MTF lookup uses `searchsorted` to find the latest higher-TF bar provably closed at decision time `T = open_15m + 15m`.
- **Bitstamp seconds vs internal milliseconds**: only `data.fetch_klines_window` and timeframe windowing see seconds. Everything else is ms.
- **All features are scale-invariant** by design — relative quantities (returns, ratios, normalized OBV, ATR%).
- **Rows are not independent**: temporal split (TRAIN_DAYS window before the last TEST_DAYS, val = last VAL_FRACTION of train+val).
- **WARMUP** in config is calibrated for the longest indicator on 15m. Don't add features with longer lookbacks without bumping warmup.

## Hyperparameter conventions

`XGB_REG_LAMBDA`, `XGB_REG_ALPHA`, `XGB_GAMMA` use the **`is not False`** identity check in `train.build_hyperparams` — setting them to `False` (not `0`, not `None`) means *don't pass the kwarg to `XGBClassifier`* so the native XGBoost default applies. `0` is a valid value meaning "explicitly disable regularization".

## Repo-wide (recap)

- Python 3.10+. Manual install : `pip install requests pandas numpy xgboost scikit-learn scipy`.
- No tests, no linter. Verification = inspect `logs/runs.csv` across runs, watch `adaptive_total_wr` and `adaptive_total_taken`. Beating the baseline (`baseline_wr_up`, naturally ~0.51) is the bar.
- Windows / PowerShell is the primary shell.
- `.csv`, `__pycache__`, `*.parquet`, `*.db`, `btc_updown_model.json`, `btc_updown_model.thresholds.json` are gitignored.
