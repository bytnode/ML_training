# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

This file is package-scoped. See the repo-root `CLAUDE.md` (`../../../../CLAUDE.md`) for repo-wide conventions, the `15min/` import trick, the team orchestrator (`main.py`), the live daemon, and the standardized `api.py` contract shared with the sister `modele_down` package.

## What this package is

`modele_up` is an **UP-only** XGBoost binary classifier. On the close of each 15m BTC/USD bar it predicts:

```
label = 1  if the NEXT 15m candle is green  (close[i+1] > open[i+1])
label = 0  if the NEXT 15m candle is red
```

The model output is `P(next is up)` and is treated as a **single confidence axis** (`confidence`). All bets the model emits are UP bets — there is no DOWN head and no two-sided arbitration inside this package. Up-bias is by construction.

Decision rule:
```
take UP bet  iff  P(up) >= effective_threshold(row)
```
where the threshold comes from the per-axis quantile calibration described below (Sidecar section). `NAME = "modele_up"` in `api.py`.

## Working-directory contract — `cd 15min/modele_up/` first

`config.py` uses **relative paths** (`DB_PATH = "db/btc_multi_tf_15m.parquet"`, `MODEL_OUTPUT = "btc_updown_model.json"`). Running `python 15min/modele_up/train.py` from the repo root reads/writes into the wrong place. Always:

```powershell
cd backtest\modele_updown\15min\modele_up
```

`logs/` is anchored by `Path(__file__)` in `train.py`, so it always lands at `15min/modele_up/logs/` regardless of cwd. The on-disk folder is `DB/` (uppercase) while `config.DB_PATH` says `db/` — this only works because Windows is case-insensitive. Keep them consistent if you touch either.

## Build pipeline — pre-built parquet, no `data.py` here

`modele_up` ships **no `data.py` and no `add_*.py`** — `git log` confirms they were never tracked here. The only Python files are `api.py`, `config.py`, `features.py`, `predict.py`, `train.py`, and two read-only scripts under `analyse/`.

The master dataset `DB/btc_multi_tf_15m.parquet` (gitignored, ~480k rows / ~13.7 years, 24 columns) is a **pre-built artifact** — treat it as a frozen input. The Bitstamp fetch + feature-engineering pipeline lives only in the sister package's `modele_down/data.py`. It is **not** interchangeable: the `modele_up` parquet carries its own **19-feature set** (smaller and different from `modele_down`'s 25), so running `modele_down/data.py` will not reproduce it. The `DB/btc_multi_tf.db` SQLite file and the per-TF `*.csv` files are leftover debug artifacts — no script in this package reads them.

Dev loop (the parquet is already present):

```powershell
cd backtest\modele_updown\15min\modele_up
python train.py     # train + sidecar + logs/runs.csv + logs/test_predictions.csv
python predict.py   # smoke-test on the last test_predictions row
python api.py       # exercise evaluate_on_slice on the full parquet
```

`features.py` is the runtime loader used by `train.py`: it reads the master (parquet or CSV — auto-detected by suffix), drops the first `WARMUP` rows, drops rows with no valid `label`, drops any pre-existing `timestamp` column, and renames `timestamp_ms` -> `timestamp`. It does **not** fetch data or compute `label` / `volatility_state` / `is_weekend` — those must already be in the parquet. If it raises "Colonnes pre-calculees manquantes", the parquet is missing a precomputed column.

## What `train.py` writes

- `btc_updown_model.json` — XGBoost native format.
- `btc_updown_model.thresholds.json` — the sidecar (shape below).
- `logs/runs.csv` — one row per `train_and_evaluate(log_run=True)`. Append-only with column-union reindex.
- `logs/test_predictions.csv` — written **only** by direct `python train.py` runs (`log_predictions=True` is hardcoded in `main()`). Schema: full test row + `p_up` + `confidence` (`confidence` is an alias of `p_up`). Read by `predict.py`'s smoke test — **not** by `api.py`.

`train.py` CLI flags: `--label NAME` (run label in `runs.csv`), `--no-save`, `--no-log`. Feature columns = every parquet column except `timestamp`, `label`, `volatility_state`, and `is_weekend`; the last two stay in the parquet only as bucket keys for the adaptive threshold — they are **not** fed to the model. The current parquet yields **19 features**: `high_ret`/`close_ret` (15m base), `high_ret_5m`/`low_ret_5m`/`close_ret_5m`, `low_ret_30m`/`close_ret_30m`, `low_ret_1h`/`close_ret_1h` (MTF returns), and `dist_ema_20`, `return_3`, `macd`, `macd_hist`, `bb_position`, `mfi_norm`, `dist_to_low_50`, `autocorr_24`, `acc_4`, `volume_rel_30d`.

### `runs.csv` legacy columns — read carefully

`logs/runs.csv` was started under an older delta-based / target-WR calibration. Old rows carry columns like `adaptive_target_wr`, `adaptive_calib_fraction`, `adaptive_calib_days`, the `delta_*` family (`delta_v0_w0` … `delta_v2_w1`, `delta_w0/w1`, `weekend_delta_boost`), and `asym_target_wd/we`. Current rows write the quantile schema (`adaptive_strategy`, `adaptive_total_taken`, `adaptive_total_wr`, `asym_T_weekday`, `asym_T_weekend`, `calib_method`, `vol_frac_weekday/weekend`, …). The reindex logic keeps every column ever written, so both shapes coexist in the file. **Do not "clean up" the legacy columns.**

## Sidecar shape — V1/V2 quantile calibration

The sidecar is **not** a `{"deltas": {...}}` map, and there is no `ADAPTIVE_TARGET_WR` config constant. `ADAPTIVE_STRATEGY` in `config.py` selects one of two shapes:

- **V1 — `asym_wknd_excl`** : `{"strategy", "thresholds_per_wknd": {"0": float, "1": float}, "skip_cells": [[vol, wknd], ...], "metadata": {...}}`. One confidence threshold per weekday/weekend axis. Cells in `skip_cells` (combinations of `volatility_state × is_weekend`) are forced to threshold `1.0` (never bet).
- **V2 — `asym_wknd_excl_hour_boost`** (current default) : V1 + `"hour_boost": {"good_hours": [int], "delta": float}`. When the bar's UTC hour is in `good_hours`, the effective threshold drops by `delta`. Skip cells are never boosted.

Thresholds are calibrated by **quantile**, not by a target-WR search: `T_axis = quantile(P(up) on non-skip val of that axis, 1 - ASYM_VOL_FRAC_axis)`. This keeps bet volume stable cross-window — see the `compute_thresholds_asym_wknd_excl` docstring in `train.py`.

The threshold lookup is centralized in `_bucket_threshold` — the **same logic appears in `api.py`, `predict.py`, and `train.py`** (`apply_thresholds_asym_*`). If you change one, change all three. **Single confidence axis**: always compare `P(up)` against the bucket threshold; never against `0.5`, never build a separate "P(down)" axis.

## The standardized `api.py` contract

```python
NAME : str = "modele_up"

def evaluate_on_slice(start_ms: int, end_ms: int) -> pd.DataFrame:
    """One row per setup the model would have *taken* in [start_ms, end_ms].
    Columns: timestamp (int ms), direction ('up'),
             confidence (= P(up)), label (int 0|1, 1 = bet was correct)."""
```

`api.py` reads the **parquet directly and predicts on the fly** — it does **not** read `logs/test_predictions.csv`. So `evaluate_on_slice` can run on any window inside the parquet's range, but bars seen during training return an optimistic WR — the caller (`15min/main.py`) is responsible for picking an out-of-sample window. `direction` is always `"up"`. Keep this contract intact: the team orchestrator only ever sees this 4-column DataFrame.

`api.py` also exposes **`evaluate_full_on_slice(start_ms, end_ms)`** — the same window but **one row per bar** (taken or not): `timestamp, confidence, threshold, take (0|1), label, volatility_state, is_weekend, hour`. `evaluate_on_slice` is exactly this filtered to `take == 1`; both share the private `_predict_all_bars` helper, so don't change one prediction path without the other. `15min/main.py` uses the full view to build the detailed per-bar logs (`team_all_bars.csv` / `team_skipped.csv` / `team_taken.csv`).

## Naming traps

- **`config.DB_PATH` points to the parquet** (`db/btc_multi_tf_15m.parquet`), not the SQLite `DB/btc_multi_tf.db`. The `.db` and per-TF `*.csv` files in `DB/` are unused leftovers in this package.
- **"up" = "next candle is green"**, defined as `close[i+1] > open[i+1]` — the *direction of the next candle*, not "price rose by a threshold". Don't redefine the label without retraining everything downstream.
- **"confidence" is an alias of `p_up`**, a raw model probability, not a calibrated score. Threshold comparisons are always against `effective_threshold(row)`, never `0.5`.
- **The `modele_up` parquet is independent**, not a label-flipped copy of `modele_down`'s. The two packages have diverged feature sets (19 vs 25), so `df["label"] = 1 - df["label"]` on a `modele_down` parquet does **not** produce a valid `modele_up` dataset.

## `config.py` knobs

- `WARMUP = 500` — bars dropped at the start (longest rolling indicator). Bump it if you add a feature with a longer lookback, or early rows get NaN-poisoned.
- `TRAIN_DAYS = 3600`, `TEST_DAYS = 366`, `VAL_FRACTION = 0.1` — temporal split (train window before the last `TEST_DAYS`, val = last `VAL_FRACTION` of train+val). Rows are not independent: do not switch to shuffled K-fold.
- `ADAPTIVE_STRATEGY`, `ASYM_VOL_FRAC_WEEKDAY/WEEKEND`, `ASYM_SKIP_CELLS`, `ASYM_HOUR_BOOST_GOOD_HOURS/DELTA` — drive the sidecar (above).
- `ATR_LONG_PERIOD`, `VOLATILITY_HIGH/LOW` — define the 3-way `volatility_state` bucket.
- `XGB_SCALE_POS_WEIGHT` — `False` (don't pass the kwarg), `"auto"` (= `n_down / n_up` on the train fold), or a fixed float (current: `1.03`). Resolved in `_resolve_scale_pos_weight`.

## Hyperparameter convention

`XGB_REG_LAMBDA`, `XGB_REG_ALPHA`, `XGB_GAMMA` use an **`is not False`** identity check in `train.build_hyperparams` — setting them to `False` (not `0`, not `None`) means *don't pass the kwarg to `XGBClassifier`* so XGBoost's native default applies. `0` is a valid value that explicitly disables regularization. Preserve this convention for any similar opt-in hyperparameter.

## `analyse/` — diagnostic scripts

`analyse/` holds two standalone, read-only scripts:

- `check_new_features.py` — feature-ablation impact tool. Trains a baseline (all features) vs an ablated model (minus the target feature) and compares `val_logloss` / `adaptive_total_wr` / EV. Modes: full ablation (default), `--quick` (permutation importance), `--multi-seed N`, `--list`.
- `analyze_features.py` — per-feature audit across four dimensions (univariate, correlation, permutation, full ablation). Writes `logs/feature_analysis.md`.

They are investigations only — they do not feed the build pipeline. Run them from the package root.

## Repo-wide (recap from the root `CLAUDE.md`)

- Python 3.10+ (tested on 3.14). Manual install: `pip install requests pandas numpy xgboost scikit-learn scipy rich`. No `requirements.txt`.
- No tests, no linter, no build system. Verification = inspect `logs/runs.csv` across runs (`adaptive_total_wr`, `adaptive_total_taken`). The bar is beating `baseline_wr_up` (~0.50), not 50% — roughly half the candles are green regardless.
- No-lookahead MTF joins: every higher-TF feature is the latest bar provably closed at decision time `T = open_15m + 15m`. Don't bypass it if you ever rebuild the parquet.
- All persisted features are scale-invariant (returns, ratios, normalized indicators) — no raw prices in the parquet. Persisted timestamps are milliseconds.
- Windows / PowerShell is the primary shell.
- `*.csv`, `*.parquet`, `*.db`, `__pycache__/`, `btc_updown_model.json`, `btc_updown_model.thresholds.json` are gitignored — regeneratable, don't commit.
