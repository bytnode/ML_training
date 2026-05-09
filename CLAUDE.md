# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repo state — transitional, read before editing

The repo is mid-refactor. `git status` shows the previous `backtest/*.py` and `live/*.py` files as `D` (deleted in working tree). The new layout in the working tree is:

- `backtest/modele_15m/` — self-contained model package (the active code). Untracked at the moment but it is what runs.
- `backtest/logs/team_runs.csv` — single output row from a planned **team backtest orchestrator** (`backtest/main.py`) that is referenced by `backtest/modele_15m/api.py` but does not yet exist on disk.
- `live/` — empty directory in the working tree. The previous live orchestrator (Aure daemon, Telegram notifs, 15min cycle) was deleted. `infra/Dockerfile` and `infra/docker-compose.yml` still reference `live/orchestrator.py` and the root model artifact `aure.json`, so Docker build will fail until either `live/` is restored or the infra files are updated. **Don't run `docker compose up` blind.**
- `aure.json` (root) — the deployed Aure model artifact. The companion sidecar `aure.thresholds.json` is gitignored and produced by training (`<MODEL_OUTPUT>.thresholds.json` from `backtest/modele_15m/train.py`).
- `docs/superpowers/specs/` and `docs/superpowers/plans/` — design docs from the live orchestrator and the multi-pair dataset experiment. Useful for *intent*, but treat them as historical context, not the current source of truth — the code they describe was deleted in commit `a422d18` (multi-pair revert) and the most recent `aed01d1 add live mode` commit landed the dockerised live orchestrator that is now staged for rework.

When editing, first reconcile against the current working tree (`backtest/modele_15m/` is real and runnable; everything else is in flux). Don't restore deleted files unless the user asks — they may be intentionally being rewritten.

## The active model — `backtest/modele_15m/`

A single XGBoost binary classifier that, on the close of each 15m BTC/USD bar, predicts the probability that **the next candle reverses a streak of N same-direction candles** (N ∈ {1,2,3,4}, last candle must satisfy `body_ratio ≥ BODY_RATIO_MIN = 0.6`). Output of `train.py` is `btc_reversal_model.json` plus an adaptive-threshold sidecar `btc_reversal_model.thresholds.json` (per-bucket deltas keyed on `volatility_state × is_weekend`).

The package's own `README.md` (12k) and `DOCUMENTATION.md` (24k) at `backtest/modele_15m/` cover the pipeline, feature set, label definition and adaptive-threshold contract in depth — read them before touching `data.py`, `train.py`, or `predict.py`. The points below are the working conventions that aren't obvious from those docs.

### Working-directory contract

`backtest/modele_15m/config.py` uses **relative paths** for `CSV_PATH`, `MODEL_OUTPUT`, `OUTPUT_CSV`. All scripts in this package must run with `backtest/modele_15m/` as cwd:

```powershell
cd backtest\modele_15m
python data.py        # fetch + features  -> training_data.csv (~290k rows, 44 cols)
python train.py       # train + eval      -> btc_reversal_model.json + sidecar + logs/runs.csv + logs/test_predictions.csv
python predict.py     # smoke-test the cached model on the last CSV row
python sweep.py       # hyperparam sweep, does NOT overwrite the prod model (save_model=False)
python analyze_runs.py # leaderboard from logs/runs.csv
```

`logs/` lives under `backtest/modele_15m/logs/` (anchored by `Path(__file__)`), but `CSV_PATH` and `MODEL_OUTPUT` aren't anchored — running `python backtest/modele_15m/train.py` from the repo root will read/write into the *root* directory. Always `cd` first.

### Standard inference contract — `api.py`

`backtest/modele_15m/api.py` defines the contract every model package will expose to the planned `backtest/main.py` team orchestrator:

```python
NAME : str = "modele_15m"

def evaluate_on_slice(start_ms: int, end_ms: int) -> pd.DataFrame:
    """Returns one row per setup the model would have *taken* in the window.
    Columns: timestamp (int ms), direction ('up'|'down'), confidence (float), label (int 0|1)."""
```

The `modele_15m` implementation reads `logs/test_predictions.csv` (regenerated every `python train.py`) and applies the adaptive-threshold sidecar to filter taken bets. Future model packages (e.g. `modele_onchain/`) implement `evaluate_on_slice` however they want (runtime fetch + predict, etc.). **Keep this contract intact** — `main.py` only sees the standardized DataFrame.

### What `train.py` writes

- `btc_reversal_model.json` — XGBoost native format.
- `btc_reversal_model.thresholds.json` — sidecar with `{"deltas": {"<vol>_<wknd>": float, ...}, "metadata": {...}}`. Base of the threshold = `config.ADAPTIVE_TARGET_WR` (single source of truth, not duplicated in the sidecar).
- `logs/runs.csv` — one row per `train_and_evaluate(log_run=True)` call. Schema is flat by design and `LOG_THRESHOLDS = [0.50, 0.55, 0.58, 0.60, 0.62]` is held stable independently of `THRESHOLDS` in config so `analyze_runs.py` keeps working across runs. New columns trigger a column-union reindex of the whole file (cheap because the file stays small — designed for < ~1000 runs).
- `logs/test_predictions.csv` — written **only** by direct `python train.py` runs (CLI sets `log_predictions=True`). `sweep.py` deliberately does *not* write it (overwriting it 30+ times per sweep with only the last config's predictions would be misleading).

### Hyperparameter conventions

`XGB_REG_LAMBDA`, `XGB_REG_ALPHA`, `XGB_GAMMA` follow a special convention: setting them to **`False`** (not `0`, not `None`) means *don't pass this argument to `XGBClassifier` at all*, so XGBoost uses its native default. The check is `is not False` (identity, not truthiness) because `0` is a valid value meaning "disable regularization explicitly". Preserve this if you add similar opt-in hyperparameters.

`SEQ_TEST` filters the **test slice only**; `train` and `val` always see all `seq_len` values. The current default `[2, 3, 4]` excludes single-candle setups from evaluation (noisier).

### Data conventions to preserve

- **Bitstamp seconds vs internal milliseconds** — `data.fetch_chunk` is the only place that sees seconds (`step`, `start`, `end`). Everything downstream is ms (Binance kline shape `[open_time_ms, o, h, l, c, v]`). Don't propagate seconds upward.
- **MTF features are no-lookahead** — `last_closed_index(open_times_sorted, T, INTERVAL_MS[iv])` returns the latest higher-TF bar provably closed at decision time `T = times[last_idx] + 15m`. Don't bypass it; using `closes[i]` directly on a higher TF leaks the future.
- **Indicators are hand-rolled** (OBV, EMA, RSI, ATR, vol avg) in pure Python — no `talib` / `pandas-ta`. `WARMUP = 200` is calibrated for EMA-200 on 15m. Adding an indicator with a longer lookback requires raising `WARMUP` or the early rows get poisoned.
- **Rows are not independent** — the same starting position can produce up to `MAX_SEQ_LEN = 4` rows. Random shuffling/`KFold` leaks. Use `TimeSeriesSplit` or walk-forward if you need CV. The current temporal split (`TRAIN_DAYS` window before the last `TEST_DAYS`, val = last `VAL_FRACTION` of train+val) is correct and shouldn't switch to `train_test_split(shuffle=True)`.
- **`BODY_RATIO_MIN` is a business filter, not a feature** — it defines who *enters* the dataset. Changing it changes the baseline WR mechanically and makes runs incomparable.

## Optional cleaning pipeline — `backtest/modele_15m/clean_data/`

Non-destructive (no rows dropped):

```powershell
cd backtest\modele_15m
python clean_data\analyze.py   # audits training_data.csv -> analysis_report.{md,json}
python clean_data\clean.py     # writes training_data_clean.csv + adds era_year, sample_weight, derived features
```

To use the cleaned CSV in training: point `config.CSV_PATH` to `"training_data_clean.csv"`. **Important**: `train.py` builds `feature_cols` as "everything except `timestamp` and `label`", which would include `sample_weight` as a feature — that's wrong. When wiring the clean CSV in, also exclude `sample_weight` from `feature_cols` and pass it via `model.fit(..., sample_weight=train["sample_weight"], sample_weight_eval_set=[val["sample_weight"]])`. Keep `era_year` *in* `feature_cols` — it's intentional. See `clean_data/cleaning_log.md` for the full snippet.

## Repo-wide conventions

- **Python 3.10+** (tested on 3.14). **No `requirements.txt`**; install deps manually: `pip install requests pandas numpy xgboost scikit-learn scipy`.
- **No tests, no linter, no build system.** Don't add `pytest` / `ruff` / `Makefile` without asking. Verification happens by inspecting `backtest/modele_15m/logs/runs.csv` across runs and watching `breakdown_total_wr` / `wr_at_0.58`. The "baseline" line in `train.py` output is `y_test.mean()` (= the WR you'd get taking every body_ratio-filtered setup). Battre 50% n'a aucun sens — the bar is the baseline.
- **Windows / PowerShell** is the primary shell. Use PowerShell syntax in any commands (`$env:VAR`, backtick line continuation, `$null` not `/dev/null`). Bash is available via the Bash tool when needed.
- **Specs live in `docs/superpowers/specs/` and `backtest/docs/superpowers/specs/`**. Read them for intent on recent decisions (live orchestrator design, multi-pair dataset, test-predictions log) — but cross-check against the working tree because some described code has since been deleted.
- **`.gitignore` ignores `*.csv`, `__pycache__`, `btc_reversal_model.json`, `aure.thresholds.json`, `live/logs/*.csv`, `.env`** — these are regeneratable artifacts or secrets. Don't commit them.
