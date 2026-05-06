# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Pipeline

Three Python files. The first two are a sequential pipeline; the third is a library:

1. `python data.py` — fetches 15m OHLC candles from Bitstamp's public OHLC endpoint (`bitstamp.net/api/v2/ohlc/{pair}/`), normalises them to the Binance kline shape `[open_time_ms, o, h, l, c, v]`, computes indicators, extracts feature rows, writes `training_data.csv`. The pair is set via `SYMBOL` in `config.py` using Bitstamp's lowercase code (e.g. `btcusd`).
2. `python train.py` — loads the CSV, trains an XGBoost binary classifier, prints evaluation, saves `btc_reversal_model.json`.
3. `predict.py` — **library module**, not a pipeline step. Exposes `predict_proba(features)` and `should_take_bet(features, threshold=…)` for use in backtest / live scripts. The model is loaded lazily on first call and cached in a module-level singleton (`_model`). The expected feature columns are read off the booster via `model.get_booster().feature_names`, so any extra columns in the input (e.g. `timestamp`, `label`) are ignored automatically. `predict_proba` accepts a `dict`, `list[dict]`, or `pd.DataFrame`. Running it directly (`python predict.py`) just smoke-tests on the last row of the CSV.

There is no `requirements.txt`. The runtime needs: `requests`, `pandas`, `numpy`, `xgboost`, `scikit-learn`.

## What the model predicts

The label is **"will the next candle reverse a streak of N same-direction candles?"** — not raw price movement. The dataset is built by sliding over every position in the kline series and emitting one row per `(i, seq_len)` pair where:

- `seq_len ∈ {1, 2, 3, 4}` consecutive candles all have the same direction (all up or all down),
- the last candle of the streak has `body_ratio ≥ BODY_RATIO_MIN` (currently 0.6) — this is a hardcoded prefilter in `data.py`, not a learned feature, and it defines which setups even enter the dataset,
- `flat` candles (open == close) are skipped entirely, both inside the streak window and as the next candle.

`label = 1` iff candle `i + seq_len` reverses the streak direction. The same starting position can produce up to `MAX_SEQ_LEN` rows (one per `seq_len`), so rows are not independent — keep this in mind if changing the split or adding cross-validation.

## CSV shape

`training_data.csv` has 24 columns: `timestamp` + 22 features + `label`. Feature list (in the order written by `data.py`):

`seq_len, seq_dir, body_ratio_last, body_ratio_avg, upper_wick, lower_wick, volume_ratio_last, volume_ratio_avg, move_pct, move_atr, rsi, rsi_change_5, obv_slope_10, obv_vs_ema, atr_ratio, volatility_state, dist_ema20, dist_ema50, dist_ema200, hour_utc, day_of_week, is_weekend`.

`seq_dir` is `1=up, 0=down`. `volatility_state` is a categorical encoded as `{0=low, 1=normal, 2=high}` based on `atr_ratio` vs `VOLATILITY_LOW`/`VOLATILITY_HIGH`. The pipeline is currently single-timeframe (15m only) — there is no multi-timeframe data fetched or merged.

## Why the split is temporal, not random

`train.py` splits by sorted timestamp with **no shuffling**, and the test slice is defined by **duration**, not by row ratio:

- `test`  = the last `TEST_DAYS` days of the CSV (config: currently 365).
- `val`   = the last `VAL_FRACTION` (= 15%) of everything *before* the test cutoff.
- `train` = the rest.

The cutoff is computed via `df["timestamp"].searchsorted(max_ts - TEST_DAYS * 86_400_000)`, so changing `DAYS_BACK` (history length) or `TEST_DAYS` (held-out window) shifts only the boundary, not the logic. Random shuffling would leak future information via temporally adjacent rows that share market regime — don't switch to `train_test_split(shuffle=True)` or `KFold`; if you need CV, use `TimeSeriesSplit` or walk-forward.

`early_stopping_rounds=XGB_EARLY_STOPPING_ROUNDS` (default 50) uses the validation slice, so the test slice stays untouched until the threshold sweep at the end.

## Evaluation shape

`train.py` reports win rate at every probability threshold in `THRESHOLDS` (config), how many bets survive each threshold, and what % of total volume that represents. The interpretation is **trading-style**: higher threshold = fewer but higher-confidence bets. The "baseline" line is the WR you'd get by taking every setup that passed the body_ratio prefilter (= `y_test.mean()`) — beating that is the bar, not 50%.

The per-`seq_len` breakdown is run at `BREAKDOWN_THRESHOLD` (config: currently 0.58). It also prints a **TOTAL** line above the per-streak detail with the aggregate count of bets taken and the global WR at that threshold — that's the headline number to watch. Longer streaks are rarer and behave differently, so check the per-`seq_len` lines before assuming an aggregate WR generalizes. After the breakdown, it prints the top-10 feature importance.

## Configuration

Every tunable knob lives in `config.py`, split into two clearly-labelled sections:

- **SECTION 1 — DATA** (consumed by `data.py`): `SYMBOL`, `INTERVAL`, `DAYS_BACK`, fetch parameters, indicator periods, feature-window sizes, `BODY_RATIO_MIN`, `WARMUP`, `MAX_SEQ_LEN`, volatility thresholds.
- **SECTION 2 — TRAINING** (consumed by `train.py` and `predict.py`): file paths, `TEST_DAYS`, `VAL_FRACTION`, `THRESHOLDS`, `BREAKDOWN_THRESHOLD`, all `XGB_*` hyperparameters.

Don't reintroduce hardcoded values in `data.py` / `train.py` — import from `config.py` instead.

### Disabling regularization hyperparameters

`XGB_REG_LAMBDA`, `XGB_REG_ALPHA`, `XGB_GAMMA` follow a special convention: setting them to `False` (not `0`, not `None`) means **"don't pass this argument to `XGBClassifier` at all"**, so XGBoost falls back to its built-in default (1.0, 0.0, 0.0 respectively). Setting any of them to a number activates that regularization. The toggle is implemented in `train.py` via a conditional `**optional_kwargs` spread (`if XGB_REG_LAMBDA is not False: ...`) — the `is not False` check (not truthiness) matters because `0` is a valid value meaning "disable regularization explicitly". Keep this contract if you add similar opt-in hyperparameters.

## Indicators are hand-rolled

`compute_obv`, `compute_ema`, `compute_rsi`, `compute_atr`, `compute_volume_avg` are implemented in pure Python in `data.py` (no `pandas-ta` / `talib`). The `WARMUP = 200` constant in `extract_features` exists because EMA-200 needs that many bars to stabilize on the 15m series — if you add an indicator with a longer lookback, raise the warmup or you'll poison early rows. ATR uses Wilder smoothing seeded with the simple mean over the first `period` bars.

## Data fetching specifics

- `DAYS_BACK` in `config.py` controls history length; the script paginates automatically (`KLINES_LIMIT=1000` candles per Bitstamp call, 8 parallel workers via `FETCH_WORKERS`).
- Bitstamp uses **seconds** for `start`/`end` and `step`, while the rest of the pipeline works in **milliseconds**. `fetch_chunk` divides ms by 1000 going out and multiplies the candle timestamp by 1000 coming back, so downstream code never sees seconds. If you add a new interval, update `INTERVAL_MS` in `config.py`.
- `fetch_all_klines` deduplicates by timestamp because adjacent Bitstamp windows can overlap at the boundary (`start`/`end` are inclusive on at least one side).
- Retries: exponential backoff (`2 ** attempt`), max 3 attempts, **only on `Timeout` / `ConnectionError`** — HTTP errors raise immediately.
- Bitstamp's public OHLC endpoint requires no API key.
