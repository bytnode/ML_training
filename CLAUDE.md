# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Pipeline

Three-stage script pipeline; first two run in order, third is reusable:

1. `python data.py` — fetches 15m OHLC candles from Bitstamp's public OHLC endpoint (`bitstamp.net/api/v2/ohlc/{pair}/`), normalises them to the Binance kline shape `[open_time_ms, o, h, l, c, v]`, computes indicators, extracts feature rows, writes `training_data.csv`. The pair is set via `SYMBOL` in `config.py` using Bitstamp's lowercase code (e.g. `btcusd`).
2. `python train.py` — loads the CSV, trains an XGBoost binary classifier, prints evaluation, saves `btc_reversal_model.json`.
3. `predict.py` — **library module**, not a pipeline step. Exposes `predict_proba(features)` and `should_take_bet(features, threshold=…)` for use in backtest / live scripts. The model is loaded lazily on first call and cached in a module-level singleton. The expected feature columns are read off the booster (`model.get_booster().feature_names`), so any extra columns in the input (e.g. `timestamp`, `label`) are ignored automatically. Running it directly (`python predict.py`) just smoke-tests on the last row of the CSV.

There is no `requirements.txt`. The runtime needs: `requests`, `pandas`, `numpy`, `xgboost`, `scikit-learn`.

## What the model predicts

The label is **"will the next candle reverse a streak of N same-direction candles?"** — not raw price movement. The dataset is built by sliding over every position in the kline series and emitting one row per `(i, seq_len)` pair where:

- `seq_len ∈ {1, 2, 3, 4}` consecutive candles all have the same direction (all up or all down),
- the last candle of the streak has `body_ratio ≥ 0.6` (this is a hardcoded prefilter in `data.py`, not a learned feature — it defines which setups even enter the dataset),
- `flat` candles are skipped entirely.

`label = 1` iff candle `i + seq_len` reverses the streak direction. The same starting position can produce up to 4 rows (one per `seq_len`), so rows are not independent — keep this in mind if changing the split or adding cross-validation.

## Why the split is temporal, not random

`train.py` does a 70/15/15 split by sorted timestamp with **no shuffling**. This is intentional: random shuffling leaks future information into the train set via temporally adjacent rows that share market regime. Don't switch to `train_test_split` with `shuffle=True` or `KFold`; if you need CV, use `TimeSeriesSplit` or walk-forward.

The `early_stopping_rounds=50` uses the validation slice (the middle 15%), so the test slice (final 15%) stays untouched until the threshold sweep at the end.

## Evaluation shape

`train.py` reports win rate at several probability thresholds (`THRESHOLDS = [0.50, 0.52, ...0.65]`) and shows how many bets survive each threshold. The interpretation is **trading-style**: higher threshold = fewer but higher-confidence bets. The "baseline" line is the WR you'd get by taking every setup that passed the body_ratio prefilter — beating that is the bar, not 50%.

The per-`seq_len` breakdown at threshold 0.55 is there because longer streaks are rarer and behave differently; check it before assuming an aggregate WR generalizes.

## Indicators are hand-rolled

`compute_obv`, `compute_ema`, `compute_rsi`, `compute_atr`, `compute_volume_avg` are implemented in pure Python in `data.py` (no `pandas-ta` / `talib`). The `warmup = 200` constant in `extract_features` exists because EMA-200 needs that many bars to stabilize — if you add an indicator with a longer lookback, raise the warmup or you'll poison early rows.

## Data fetching specifics

- `DAYS_BACK` in `config.py` controls history length; the script paginates automatically (`KLINES_LIMIT=1000` candles per Bitstamp call, 8 parallel workers via `FETCH_WORKERS`).
- Bitstamp uses **seconds** for `start`/`end` and `step`, while the rest of the pipeline works in **milliseconds**. `fetch_chunk` divides ms by 1000 going out and multiplies the candle timestamp by 1000 coming back, so downstream code never sees seconds.
- `fetch_all_klines` deduplicates by timestamp because adjacent Bitstamp windows can overlap at the boundary (`start`/`end` are inclusive on at least one side).
- Retries: exponential backoff, max 3 attempts, only on Timeout/ConnectionError — HTTP errors raise immediately.
- Bitstamp's public OHLC endpoint requires no API key.
