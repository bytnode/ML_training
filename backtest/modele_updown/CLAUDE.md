# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

This file is package-scoped: see the repo-root `CLAUDE.md` (`../../CLAUDE.md`) for repo-wide conventions and for `modele_15m`.

## What this package is

`modele_updown` is a **DOWN-only** XGBoost binary classifier. On the close of each 15m BTC/USD bar it predicts:

```
label = 1  if the NEXT 15m candle is red  (close[i+1] < open[i+1])
label = 0  if the NEXT 15m candle is green
```

The model output is `P(next is down)` and is treated as a single confidence axis. All bets the model emits are DOWN bets — there is no UP head and no two-sided arbitration. Down-bias is by construction, not a feature of the data.

Per-bucket adaptive threshold:
```
take DOWN bet  iff  P(down) >= ADAPTIVE_TARGET_WR + delta(volatility_state, is_weekend)
```
6 buckets total (vol ∈ {0,1,2} × wknd ∈ {0,1}), one delta per bucket, written to `btc_updown_model.thresholds.json`. **Do not reintroduce a separate UP delta** — the sidecar shape is `{"deltas": {"v_w": float, ...}, "metadata": {...}}` and `predict.decide()` / `api.evaluate_on_slice()` assume that shape.

## Working-directory contract — `cd backtest/modele_updown/` first

`config.py` uses **relative paths**: `DB_PATH = "db/btc_multi_tf_15m.csv"` and `MODEL_OUTPUT = "btc_updown_model.json"`. The patch scripts also do `Path(__file__).resolve().parent / "db" / ...`, so running from the package root is fine, but launching `python backtest/modele_updown/train.py` from the repo root will read/write into the wrong place. Always `cd` first.

`logs/` is anchored by `Path(__file__)` (in `train.py`), so it lives under `backtest/modele_updown/logs/` regardless. The artifacts (`runs.csv`, `test_predictions.csv`) are gitignored CSVs.

## Build pipeline — order matters

The CSV `db/btc_multi_tf_15m.csv` is built incrementally by a chain of scripts. Each one is idempotent (skips columns already present), but **the order matters** because later scripts read columns produced by earlier ones:

```powershell
cd backtest\modele_updown

# 1. Cold fetch — DROPs the SQLite tables and refetches ~5000 days of OHLCV from
#    Bitstamp on 5 timeframes (15m, 30m, 1h, 4h, 1d), computes 9 normalized
#    features per TF, writes db/btc_multi_tf.db + 5 CSVs (one per TF).
#    Takes ~2 min. Run when you start fresh.
python data.py            # --pair btcusd --days 5000 --db db/btc_multi_tf.db

# 2. (Legacy patch) Adds body_signed = close_ret - open_ret to the DB and all
#    CSVs without refetching. Useful only when the cold fetch was done before
#    body_signed existed in data.py. New cold fetches don't need this.
python patch_body_signed.py

# 3. Adds the 5m timeframe as additional context (9 _5m columns suffixed).
#    Fetches 5m candles from Bitstamp (~3-5 min if no cache) and joins them
#    no-lookahead onto the 15m CSV. Cached in db/btc_multi_tf_5m.csv.
python fetch_and_add_5m.py

# 4. Adds 51 Tier 1 features to the 15m CSV in place :
#      40 MTF (10 cols × {30m,1h,4h,1d}) joined no-lookahead via searchsorted
#      11 derived on the reconstructed relative close (RSI, EMA distances,
#         multi-bar returns, hour_sin/hour_cos, upper/lower wicks)
python add_tier1_features.py

# 5. Drops 20 features identified as redundant (5 pure dups r >= 0.99,
#    6 strong correlations 0.85-0.99, 9 WEAK on every axis). Reads decisions
#    from analyze_features.py output. Idempotent.
python drop_redundant_features.py

# 6. Train + evaluate. Writes btc_updown_model.json + sidecar + logs/runs.csv
#    + logs/test_predictions.csv. Direct CLI runs always log per-row test
#    predictions (log_predictions=True is hardcoded in main()).
python train.py [--label name] [--no-save] [--no-log]

# Inspection tools
python predict.py             # smoke-test on the last test_predictions row
python api.py                 # exercise evaluate_on_slice on the full window
python analyze_features.py    # 7-section feature audit (writes to stdout/log)
python asymmetry_diagnostic.py # cohens_d / asymmetry on train set per feature
```

`features.py` is the runtime feature builder used by `train.py` — it does **not** modify the CSV; it just reads it, drops the warmup, computes `volatility_state` + `is_weekend` + `label`, and skips flat candles (`open == close` on bar `i` or `i+1`). Anything that needs to be persisted across runs goes through one of the patch scripts above, not through `features.py`.

## What `train.py` writes

- `btc_updown_model.json` — XGBoost native format.
- `btc_updown_model.thresholds.json` — sidecar `{"deltas": {"v_w": float, ...}, "metadata": {...}}`. Base of the threshold = `config.ADAPTIVE_TARGET_WR` (single source of truth, not duplicated in the sidecar).
- `logs/runs.csv` — one row per `train_and_evaluate(log_run=True)` call. Append-only with column-union reindex (so adding a new metric doesn't corrupt earlier rows).
- `logs/test_predictions.csv` — written **only** by direct `python train.py` invocations. The schema is `{full feature row}, label, p_down, confidence` (where `confidence` is an alias for `p_down`). `api.py` reads this file to implement the standardized inference contract.

### `runs.csv` legacy columns — read carefully

`logs/runs.csv` was started under a previous two-axis (UP+DOWN) regime. Old rows contain columns like `baseline_wr_up`, `adaptive_up_taken`, `adaptive_up_wr`, `delta_up_v0_w0`, `delta_down_v0_w0`, etc. Current rows write the single-axis schema (`baseline_wr_down`, `adaptive_total_taken`, `adaptive_total_wr`, `delta_v0_w0`, ...). The reindex logic preserves all columns across runs; both shapes coexist in the file. **Do not "clean up" the legacy columns** without checking whether `analyze_runs.py` (in the sister `modele_15m` package) or any external script keys off them.

## The standardized `api.py` contract

Every model package exposes the same shape to the planned `backtest/main.py` team orchestrator:

```python
NAME : str = "modele_updown"

def evaluate_on_slice(start_ms: int, end_ms: int) -> pd.DataFrame:
    """One row per setup the model would have *taken* in the window.
    Columns: timestamp (int ms), direction ('down' for this package),
             confidence (= P(down)), label (int 0|1, 1 = bet was correct)."""
```

This package's implementation reads `logs/test_predictions.csv` (so it covers exactly the test slice produced by the most recent `python train.py`) and applies the per-bucket adaptive threshold to filter taken bets. **Keep this contract intact** — `direction` is always `"down"` for this model and the orchestrator only sees that standardized DataFrame.

## Naming traps

- **`config.DB_PATH` is misnamed**: it points to the **CSV** `db/btc_multi_tf_15m.csv`, not the SQLite DB (`db/btc_multi_tf.db`). The SQLite is only used by `data.py` and `patch_body_signed.py` directly via hard-coded paths. Don't be fooled into pointing `DB_PATH` at `.db`.
- **"down" in this package = "red candle next"**, defined as `close[i+1] < open[i+1]`. This is the *direction of the next candle*, not "price went down by some threshold". Do not redefine the label without retraining everything downstream.
- **"confidence" in `test_predictions.csv` is an alias of `p_down`**, not a calibrated score. Threshold comparisons are always against `ADAPTIVE_TARGET_WR + delta(bucket)`, never against 0.5.

## Data conventions (same as the root CLAUDE.md)

- **No-lookahead joins**: every MTF lookup uses `searchsorted` to find the latest higher-TF bar provably closed at decision time `T = open_15m + 15m`. See `add_tier1_features.lookup_mtf_indices` and `fetch_and_add_5m.add_5m_columns_to_15m_csv`. Don't bypass it.
- **Bitstamp seconds vs internal milliseconds**: only `data.fetch_klines_window` and the windowing in `data.fetch_timeframe` / `fetch_and_add_5m.fetch_5m_raw` see seconds. Everything else is ms.
- **All CSV features are scale-invariant** by design — `data.compute_normalized_features` produces only relative quantities (returns, ratios, normalized OBV, ATR%). The reconstructed `close_rel` in `add_tier1_features` exists *only* to produce more scale-invariant features (RSI, EMA distance, returns); never persist it.
- **Rows are not independent**: same as `modele_15m`, the temporal split (`TRAIN_DAYS` window before the last `TEST_DAYS`, val = last `VAL_FRACTION` of train+val) is correct. Don't switch to shuffled splits.
- **`WARMUP = 200`** is calibrated for the longest indicator on 15m. If you add a feature with a longer lookback (e.g. EMA-200 already uses `Path` reconstructions but stays inside warmup), bump `WARMUP` or the early rows get poisoned with NaNs that `features.py` would silently skip.

## Hyperparameter conventions

`XGB_REG_LAMBDA`, `XGB_REG_ALPHA`, `XGB_GAMMA` use the **`is not False`** identity check in `train.build_hyperparams` — setting them to `False` (not `0`, not `None`) means *don't pass the kwarg to `XGBClassifier`* so the native XGBoost default applies. `0` is a valid value meaning "explicitly disable regularization". Preserve this convention if you add similar opt-in hyperparameters.

## Repo-wide (recap from `../../CLAUDE.md`)

- Python 3.10+ (tested on 3.14). Manual install: `pip install requests pandas numpy xgboost scikit-learn scipy`. No `requirements.txt`.
- No tests, no linter, no build system. Verification = inspect `logs/runs.csv` across runs, watch `adaptive_total_wr` and `adaptive_total_taken`. Beating the baseline (`baseline_wr_down`, naturally ~0.50) is the bar — beating 50% is the wrong frame because half the candles are red regardless.
- Windows / PowerShell is the primary shell. Use PowerShell syntax in any commands run on this machine.
- `.csv`, `__pycache__`, `btc_updown_model.json`, `btc_updown_model.thresholds.json` are gitignored — regeneratable, don't commit.
