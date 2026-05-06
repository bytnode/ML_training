# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# install (no requirements.txt — Python 3.10+, tested on 3.14)
pip install requests pandas numpy xgboost scikit-learn scipy

# end-to-end pipeline
python data.py        # fetch + features  -> training_data.csv
python train.py       # train + eval      -> btc_reversal_model.json + appends logs/runs.csv
python predict.py     # smoke-test predict on last CSV row

# train.py CLI flags
python train.py --label myrun     # custom run_label in logs/runs.csv (default "manual")
python train.py --no-save         # don't overwrite btc_reversal_model.json
python train.py --no-log          # don't append to logs/runs.csv

# hyperparameter sweep (does NOT overwrite the prod model)
python sweep.py                                # all sweeps, 5 values per HP
python sweep.py --quick                        # 3 values per HP
python sweep.py --only max_depth,learning_rate # restrict to listed HPs

# analyze sweep results
python analyze_runs.py
python analyze_runs.py --metric breakdown_total_wr --top 20
python analyze_runs.py --only max_depth,learning_rate

# optional cleaning pipeline
python clean_data/analyze.py     # audits training_data.csv -> analysis_report.{md,json}
python clean_data/clean.py       # writes training_data_clean.csv + cleaning_log.md
```

There is **no test suite, no linter, no build system** — verification happens by inspecting `logs/runs.csv` across runs and watching the `breakdown_total_wr` / `wr_at_0.58` metrics. Don't add `pytest`, `ruff`, or a `Makefile` without asking.

## Pipeline

Three core Python files at the root + an optional `clean_data/` module + sweep tooling.

1. `python data.py` — fetches 15m OHLC candles from Bitstamp's public OHLC endpoint (`bitstamp.net/api/v2/ohlc/{pair}/`), normalises them to the Binance kline shape `[open_time_ms, o, h, l, c, v]`, **also fetches the higher timeframes listed in `MTF_INTERVALS`** (currently 30m/1h/4h/1d), computes indicators, extracts feature rows, writes `training_data.csv`. The pair is set via `SYMBOL` in `config.py` using Bitstamp's lowercase code (e.g. `btcusd`).
2. `python train.py` — loads the CSV, trains an XGBoost binary classifier, prints evaluation, saves `btc_reversal_model.json`, and **appends one row to `logs/runs.csv`**. CLI flags: `--label <name>` (run label written to runs.csv, default `"manual"`), `--no-save` (don't overwrite the model on disk), `--no-log` (don't append to runs.csv). Also exposes `train_and_evaluate(overrides, save_model, verbose, log_run, run_label) -> dict` as a library API — that's what `sweep.py` calls.
3. `predict.py` — **library module**, not a pipeline step. Exposes `predict_proba(features)` and `should_take_bet(features, threshold=…)` for use in backtest / live scripts. The model is loaded lazily on first call and cached in a module-level singleton (`_model`). The expected feature columns are read off the booster via `model.get_booster().feature_names`, so any extra columns in the input (e.g. `timestamp`, `label`) are ignored automatically. `predict_proba` accepts a `dict`, `list[dict]`, or `pd.DataFrame`. Running it directly (`python predict.py`) just smoke-tests on the last row of the CSV.

Optional tooling:
- `python sweep.py` — one-at-a-time hyperparameter sweep around the current `config.py` values. Calls `train_and_evaluate()` for each value and logs each run to `logs/runs.csv` with `run_label="sweep:<hp>=<value>"`. **Does not overwrite `btc_reversal_model.json`** (`save_model=False`). Flags: `--only max_depth,learning_rate` to restrict, `--quick` for 3 values per HP instead of 5. Sweepable HPs are defined in `SWEEPS_FULL` / `SWEEPS_QUICK` dicts at the top of the file.
- `python analyze_runs.py` — reads `logs/runs.csv`, parses `sweep:<hp>=<value>` labels, prints per-HP impact tables and a top-N global leaderboard. Flags: `--metric breakdown_total_wr` to change the ranking metric (default `wr_at_0.58`), `--only` to restrict, `--top N` for leaderboard size. The "reference" line uses the most recent `run_label="manual"` row.
- `clean_data/` — see "Data cleaning module" below.

There is no `requirements.txt`. The runtime needs: `requests`, `pandas`, `numpy`, `xgboost`, `scikit-learn`. The cleaning module additionally needs `scipy`.

## What the model predicts

The label is **"will the next candle reverse a streak of N same-direction candles?"** — not raw price movement. The dataset is built by sliding over every position in the kline series and emitting one row per `(i, seq_len)` pair where:

- `seq_len ∈ {1, 2, 3, 4}` consecutive candles all have the same direction (all up or all down),
- the last candle of the streak has `body_ratio ≥ BODY_RATIO_MIN` (currently 0.6) — this is a hardcoded prefilter in `data.py`, not a learned feature, and it defines which setups even enter the dataset,
- `flat` candles (open == close) are skipped entirely, both inside the streak window and as the next candle.

`label = 1` iff candle `i + seq_len` reverses the streak direction. The same starting position can produce up to `MAX_SEQ_LEN` rows (one per `seq_len`), so rows are not independent — keep this in mind if changing the split or adding cross-validation.

## CSV shape

`training_data.csv` has **44 columns**: `timestamp` + 22 15m features + 20 MTF OHLCV features + `label`. Feature list (in the order written by `data.py`):

- **15m features (22)**: `seq_len, seq_dir, body_ratio_last, body_ratio_avg, upper_wick, lower_wick, volume_ratio_last, volume_ratio_avg, move_pct, move_atr, rsi, rsi_change_5, obv_slope_10, obv_vs_ema, atr_ratio, volatility_state, dist_ema20, dist_ema50, dist_ema200, hour_utc, day_of_week, is_weekend`.
- **MTF features (20)**: for each interval in `MTF_INTERVALS = ["30m", "1h", "4h", "1d"]`, five raw OHLCV columns: `open_<iv>, high_<iv>, low_<iv>, close_<iv>, vol_<iv>`.

`seq_dir` is `1=up, 0=down`. `volatility_state` is a categorical encoded as `{0=low, 1=normal, 2=high}` based on `atr_ratio` vs `VOLATILITY_LOW`/`VOLATILITY_HIGH`.

### MTF features are no-lookahead

The decision timestamp is `T = times[last_idx] + 15m` (i.e. the moment the 15m streak's last candle has just closed). For each MTF, `last_closed_index(open_times_sorted, T, INTERVAL_MS[iv])` returns the index of the latest bar whose `open_time + interval_ms <= T` — i.e. the most recent bar that is provably closed at decision time. If no MTF bar has yet closed (early history), the row is skipped (`skipped_mtf` counter). **Don't bypass `last_closed_index`** when adding new MTF features — using `closes[i]` directly on a higher TF leaks future information into the row.

## Why the split is temporal, not random

`train.py` splits by sorted timestamp with **no shuffling**. All three slices are defined by **duration**, not by row ratio:

- `test`        = the last `TEST_DAYS` days of the CSV (config: currently 365).
- `train+val`   = the `TRAIN_DAYS` days immediately before the test cutoff (config: currently 3600). If `TRAIN_DAYS = None`, everything before the test cutoff is used. Older rows are dropped (printed as `ignore (donnees plus vieilles que train)` in the run output).
- `val`         = the last `VAL_FRACTION` (= 15%) of the train+val block.
- `train`       = what remains of the train+val block.

Cutoffs are computed via `df["timestamp"].searchsorted(...)`. `TRAIN_DAYS` and `TEST_DAYS` are independent knobs: shrinking `TRAIN_DAYS` discards old history (useful if you suspect concept drift); shrinking `TEST_DAYS` shifts the train/test boundary forward. Random shuffling would leak future information via temporally adjacent rows that share market regime — don't switch to `train_test_split(shuffle=True)` or `KFold`; if you need CV, use `TimeSeriesSplit` or walk-forward.

`early_stopping_rounds=XGB_EARLY_STOPPING_ROUNDS` (default 50) uses the validation slice, so the test slice stays untouched until the threshold sweep at the end.

## Evaluation shape

`train.py` reports win rate at every probability threshold in `THRESHOLDS` (config), how many bets survive each threshold, and what % of total volume that represents. The interpretation is **trading-style**: higher threshold = fewer but higher-confidence bets. The "baseline" line is the WR you'd get by taking every setup that passed the body_ratio prefilter (= `y_test.mean()`) — beating that is the bar, not 50%.

The per-`seq_len` breakdown is run at `BREAKDOWN_THRESHOLD` (config: currently 0.58). It also prints a **TOTAL** line above the per-streak detail with the aggregate count of bets taken and the global WR at that threshold — that's the headline number to watch. Longer streaks are rarer and behave differently, so check the per-`seq_len` lines before assuming an aggregate WR generalizes. After the breakdown, it prints the top-10 feature importance.

### `logs/runs.csv`

Every call to `train_and_evaluate(log_run=True)` (which is the default for `python train.py` and for `sweep.py`) appends one row. Schema is **flat by design**: `timestamp_iso`, `run_label`, `duration_sec`, split sizes, all `hp_<name>` hyperparameters, `wr_at_<th>` / `n_at_<th>` / `pct_at_<th>` for the fixed `LOG_THRESHOLDS = [0.50, 0.55, 0.58, 0.60, 0.62]` (independent of `THRESHOLDS` in config — keeping the schema stable across runs), `breakdown_total_*`, and `wr_seq<n>` / `n_seq<n>` per streak length. If a new run introduces columns the existing CSV doesn't have, `append_to_runs_csv` does a column-union reindex and rewrites the whole file (cheap because the file stays small — designed for < ~1000 runs). Keep `LOG_THRESHOLDS` stable so `analyze_runs.py` keeps working across runs.

## Configuration

Every tunable knob lives in `config.py`, split into two clearly-labelled sections:

- **SECTION 1 — DATA** (consumed by `data.py`): `SYMBOL`, `INTERVAL`, `DAYS_BACK`, fetch parameters, `INTERVAL_MS`, `MTF_INTERVALS`, indicator periods, feature-window sizes, `BODY_RATIO_MIN`, `WARMUP`, `MAX_SEQ_LEN`, volatility thresholds.
- **SECTION 2 — TRAINING** (consumed by `train.py` and `predict.py`): file paths, `TRAIN_DAYS`, `TEST_DAYS`, `VAL_FRACTION`, `SEQ_TEST`, `THRESHOLDS`, `BREAKDOWN_THRESHOLD`, all `XGB_*` hyperparameters.

Don't reintroduce hardcoded values in `data.py` / `train.py` — import from `config.py` instead.

### Adaptive per-regime threshold (`ADAPTIVE_THRESHOLD`)

`BREAKDOWN_THRESHOLD = 0.58` stays as a hardcoded **base**, but the actual decision threshold floats per market regime: `effective_th = BREAKDOWN_THRESHOLD + delta(volatility_state, is_weekend)`. The deltas are calibrated on val and persisted next to the model in a JSON sidecar — same architecture as the model file.

**Buckets**: `(volatility_state ∈ {0,1,2}, is_weekend ∈ {0,1})` = 6 fixed buckets. No new features required — both already exist in the CSV.

**Calibration recipe** (run inside `train.py` after `model.fit`):
1. Slice val to the last `ADAPTIVE_CALIB_FRACTION × TEST_DAYS` days by timestamp (default `0.15 × 365 ≈ 54` days). The window is expressed as a fraction of `TEST_DAYS` so the calibration scale follows the evaluation horizon — if you shrink the test window, the calib window shrinks proportionally. The resolved day count is computed once at module load (`train.ADAPTIVE_CALIB_DAYS`) and logged to `runs.csv` as `adaptive_calib_days` (alongside the source `adaptive_calib_fraction` for traceability).
2. For each bucket, sweep candidate thresholds in `[0.45, 0.75]` step 0.005, keeping only candidates with ≥ `ADAPTIVE_MIN_N_AT_THRESHOLD` bets (default 30).
3. Pick the smallest threshold whose realized WR ≥ `ADAPTIVE_TARGET_WR`. If unreachable, fall back to the threshold with the highest WR in that bucket.
4. `delta = chosen_th - BREAKDOWN_THRESHOLD`, clamped to `ADAPTIVE_DELTA_CLAMP` (default `(-0.05, +0.08)`).
5. Per-bucket fallback cascade (sparse buckets like high-vol-weekend are common in BTC data):
   - `n_calib ≥ ADAPTIVE_MIN_N_VAL` → use calib window (`source = "calib_window"`, **high confidence**).
   - else `n_full ≥ ADAPTIVE_MIN_N_VAL` → use full val (`fallback_full_val`, high confidence, recency sacrificed).
   - else `n_full ≥ ADAPTIVE_MIN_N_VAL_SOFT` (default 30) → use full val (`low_conf_full_val`, **low confidence**). The clamp + `MIN_N_AT_THRESHOLD` floor + max-WR fallback inside `_compute_bucket_threshold` are what keep a noisy delta from being dangerous in this tier — don't loosen them without thought.
   - else → `delta = 0`, `source = "insufficient_data"`.

**Sidecar JSON** (`<MODEL_OUTPUT>.thresholds.json`, e.g. `btc_reversal_model.thresholds.json`):
```json
{
  "base_threshold": 0.58,
  "deltas": {"0_0": -0.025, "0_1": 0.0, "1_0": 0.0, "1_1": 0.015, "2_0": 0.04, "2_1": 0.06},
  "metadata": {"trained_at": "...", "calib_days": 90, "target_wr": 0.58, ...}
}
```
Keys are `"<vol>_<wknd>"`. Saved only when `train.py` runs with `save_model=True` (so sweeps don't clobber the deployed sidecar — same convention as `MODEL_OUTPUT` itself).

**`predict.py` contract**:
- `should_take_bet(features)` (no `threshold` arg) → loads the sidecar lazily and applies the per-bucket threshold automatically. If the sidecar is missing (legacy model, or sidecar deleted), falls back to `BREAKDOWN_THRESHOLD`.
- `should_take_bet(features, threshold=0.58)` → ignores the sidecar entirely, useful for A/B vs the fixed threshold or when the caller doesn't have `volatility_state`/`is_weekend` available.
- The sidecar is cached in a module-level singleton (`_thresholds`) just like `_model`.

**`runs.csv` schema additions**:
- `adaptive_threshold_enabled`, `adaptive_calib_days`, `adaptive_target_wr` — config snapshot.
- `adaptive_total_taken`, `adaptive_total_wr` — head-to-head vs the fixed `breakdown_total_*` on the same test slice.
- `delta_v{0,1,2}_w{0,1}` — the 6 calibrated deltas (`None` when bucket fell back to insufficient_data).

**Disabling**: set `ADAPTIVE_THRESHOLD = False` in `config.py`. `train.py` skips calibration, doesn't write the sidecar, and the metrics dict has `adaptive_total_*` set to `None`/`0`. `predict.py` continues to work — if the sidecar exists from a prior run, it's still used unless deleted; if missing, falls back to `BREAKDOWN_THRESHOLD`.

**Important**: with very small `VAL_FRACTION` (e.g. 0.01), the calib window may have fewer rows than `ADAPTIVE_MIN_N_VAL` per bucket and many buckets will silently fall back to `delta = 0`. Watch the `source_flag` column in the printed calibration table — if most buckets read `insufficient_data`, lower `ADAPTIVE_MIN_N_VAL` or raise `VAL_FRACTION`.

### Restricting the test slice to specific streak lengths (`SEQ_TEST`)

`SEQ_TEST` filters which `seq_len` values are kept in the **test slice only** — train and val are never filtered, so the model still sees every streak length during fitting. The filter is applied right after the temporal split, before the threshold sweep / breakdown / `runs.csv` logging, which means baseline WR, `wr_at_<th>`, and `breakdown_total_*` are all computed against the filtered test set.

- `SEQ_TEST = None` (or `[]`) — keep all rows (default, current behaviour).
- `SEQ_TEST = [2, 3]` — evaluate only on `seq_len ∈ {2, 3}` rows.

The chosen filter is logged as a `seq_test` column in `logs/runs.csv` (value `"all"` or e.g. `"2,3"`), so `analyze_runs.py` results across runs with different `SEQ_TEST` are distinguishable. The per-`seq_len` breakdown print already suppresses empty buckets, so excluded streaks just disappear from the output rather than printing as 0.

### Disabling regularization hyperparameters

`XGB_REG_LAMBDA`, `XGB_REG_ALPHA`, `XGB_GAMMA` follow a special convention: setting them to `False` (not `0`, not `None`) means **"don't pass this argument to `XGBClassifier` at all"**, so XGBoost falls back to its built-in default (1.0, 0.0, 0.0 respectively). Setting any of them to a number activates that regularization. The toggle is implemented in `train.py` via a conditional `**optional_kwargs` spread (`if XGB_REG_LAMBDA is not False: ...`) — the `is not False` check (not truthiness) matters because `0` is a valid value meaning "disable regularization explicitly". Keep this contract if you add similar opt-in hyperparameters.

## Indicators are hand-rolled

`compute_obv`, `compute_ema`, `compute_rsi`, `compute_atr`, `compute_volume_avg` are implemented in pure Python in `data.py` (no `pandas-ta` / `talib`). The `WARMUP = 200` constant in `extract_features` exists because EMA-200 needs that many bars to stabilize on the 15m series — if you add an indicator with a longer lookback (e.g. raising `ATR_LONG_PERIOD` past 200), raise `WARMUP` or you'll poison early rows. ATR uses Wilder smoothing seeded with the simple mean over the first `period` bars.

## Data fetching specifics

- `DAYS_BACK` in `config.py` controls history length; the script paginates automatically (`KLINES_LIMIT=1000` candles per Bitstamp call, 8 parallel workers via `FETCH_WORKERS`).
- Bitstamp uses **seconds** for `start`/`end` and `step`, while the rest of the pipeline works in **milliseconds**. `fetch_chunk` divides ms by 1000 going out and multiplies the candle timestamp by 1000 coming back, so downstream code never sees seconds. If you add a new interval, update `INTERVAL_MS` in `config.py`.
- `fetch_all_klines` deduplicates by timestamp because adjacent Bitstamp windows can overlap at the boundary (`start`/`end` are inclusive on at least one side).
- Retries: exponential backoff (`2 ** attempt`), max 3 attempts, **only on `Timeout` / `ConnectionError`** — HTTP errors raise immediately.
- Bitstamp's public OHLC endpoint requires no API key.
- MTF fetches happen sequentially across timeframes (`fetch_all_timeframes`) but each `fetch_all_klines` parallelises across windows internally.

## Data cleaning module (`clean_data/`)

Optional offline pipeline that produces a cleaned variant of the dataset. **Does not run automatically** — invoke explicitly:

- `python clean_data/analyze.py` — audits `training_data.csv` (missing values, duplicates, label distribution, outliers, multicollinearity, KS-test concept drift, mutual information vs label, business-rule sanity like `body_ratio_last >= 0.6`). Writes `clean_data/analysis_report.md` (human) + `analysis_report.json` (machine). Modifies nothing.
- `python clean_data/clean.py` — applies non-destructive corrections (no rows dropped) and writes `training_data_clean.csv` to the repo root + `clean_data/cleaning_log.md`. Adds: winsorization at p0.5 / p99.5 on 10 heavy-tailed columns, an `era_year` column (year as a learnable feature), a `sample_weight` column with exponential decay back in time (`α = 0.15` per year, normalised to `mean=1`), and 7 derived features (`volume_ratio_*_log`, `wick_imbalance`, `dist_ema_aligned`, `rsi_extreme`, `move_signed_atr`, `dist_ema20_vs_ema200`).

To use the cleaned CSV in `train.py`: change `CSV_PATH` in `config.py` to `"training_data_clean.csv"`. **Important**: the current `train.py` builds `feature_cols` as "everything except `timestamp` and `label`", which would include `sample_weight` as a feature — that's wrong. When wiring the clean CSV in, also exclude `sample_weight` from `feature_cols` and pass it via `model.fit(..., sample_weight=train["sample_weight"], sample_weight_eval_set=[val["sample_weight"]])`. Keep `era_year` *in* `feature_cols` — it's intentional. See `cleaning_log.md` for the full snippet.
