# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

This file is package-scoped. See the repo-root `CLAUDE.md` (`../../../../CLAUDE.md`) for repo-wide conventions, the `5m/` import trick, the team orchestrator (`main.py`), and the standardized `api.py` contract shared with the sister `modele_down` package.

## What this package is

`modele_up` is an **UP-only** XGBoost binary classifier. On the close of each 5m BTC/USD bar it predicts:

```
label = 1  if the NEXT 5m candle is green  (close[i+1] > open[i+1])
label = 0  if the NEXT 5m candle is red
```

The model output is `P(next is up)` and is treated as a **single confidence axis** (`confidence`). All bets the model emits are UP bets — there is no DOWN head and no two-sided arbitration inside this package. Up-bias is by construction.

Decision rule:
```
take UP bet  iff  P(up) >= effective_threshold(row)
```
where the threshold comes from the per-axis quantile calibration described below (Sidecar section). `NAME = "modele_up"` in `api.py`.

## Working-directory contract — `cd 5m/modele_up/` first

`config.py` uses **relative paths** (`DB_PATH = "db/btc_multi_tf_5m.parquet"`, `MODEL_OUTPUT = "btc_updown_model.json"`). Running `python 5m/modele_up/train.py` from the repo root reads/writes into the wrong place. Always:

```powershell
cd backtest\modele_updown\5m\modele_up
```

`logs/` is anchored by `Path(__file__)` in `train.py`, so it always lands at `5m/modele_up/logs/` regardless of cwd. The on-disk folder is `DB/` (uppercase) while `config.DB_PATH` says `db/` — this only works because Windows is case-insensitive.

## Build pipeline — parquet derived from `modele_down`, no `data.py` here

`modele_up` ships **no `data.py`**. Its parquet is **derived from the sister `modele_down` package**: copy `modele_down`'s 5m parquet and flip the label (`df["label"] = 1 - df["label"]`). UP and DOWN are exact complements of the next-5m-candle direction, so the two packages share one identical feature set.

```powershell
# 1. Build the parquet via modele_down
cd backtest\modele_updown\5m\modele_down
python data.py            # produces DB/btc_multi_tf_5m.parquet

# 2. Copy it into modele_up/DB/ and flip the label
cd ..\modele_up
New-Item -ItemType Directory -Force -Path DB | Out-Null
Copy-Item ..\modele_down\DB\btc_multi_tf_5m.parquet .\DB\
python -c "import pandas as pd; p='DB/btc_multi_tf_5m.parquet'; df=pd.read_parquet(p); df['label']=1-df['label']; df.to_parquet(p, index=False)"

# 3. Train + evaluate
python train.py           # train + sidecar + logs/runs.csv + logs/test_predictions.csv
python predict.py         # smoke-test on the last test_predictions row
python api.py             # exercise evaluate_on_slice on the full parquet
```

This is the same mechanism `30m/modele_up` uses. It is **different from `15min/modele_up`**, whose parquet is an independent frozen artifact with its own feature set — for the `5m/` tree, `modele_up`'s parquet is always a label-flipped copy of `modele_down`'s.

`features.py` is the runtime loader used by `train.py`: it reads the master (parquet or CSV — auto-detected by suffix), drops the first `WARMUP` rows, drops rows with no valid `label`, drops any pre-existing `timestamp` column, and renames `timestamp_ms` -> `timestamp`. It does **not** fetch data or compute `label` / `volatility_state` / `is_weekend` — those must already be in the parquet.

## What `train.py` writes

- `btc_updown_model.json` — XGBoost native format.
- `btc_updown_model.thresholds.json` — the sidecar (shape below).
- `logs/runs.csv` — one row per `train_and_evaluate(log_run=True)`. Append-only with column-union reindex.
- `logs/test_predictions.csv` — written **only** by direct `python train.py` runs (`log_predictions=True` is hardcoded in `main()`). Schema: full test row + `p_up` + `confidence` (`confidence` is an alias of `p_up`). Read by `predict.py`'s smoke test — **not** by `api.py`.

`train.py` CLI flags: `--label NAME` (run label in `runs.csv`), `--no-save`, `--no-log`. Feature columns = every parquet column except `timestamp`, `label`, `volatility_state`, and `is_weekend`; the last two stay in the parquet only as bucket keys for the adaptive threshold — they are **not** fed to the model. The exact feature set is whatever `modele_down`'s `data.py` produced (base 5m + 1m fine context + MTF `[15m,30m,1h,4h,1d]` + Tier1 + Tier2).

## Sidecar shape — V1/V2 quantile calibration

The sidecar is **not** a `{"deltas": {...}}` map, and there is no `ADAPTIVE_TARGET_WR` config constant. `ADAPTIVE_STRATEGY` in `config.py` selects one of two shapes:

- **V1 — `asym_wknd_excl`** : `{"strategy", "thresholds_per_wknd": {"0": float, "1": float}, "skip_cells": [[vol, wknd], ...], "metadata": {...}}`. One confidence threshold per weekday/weekend axis. Cells in `skip_cells` (combinations of `volatility_state × is_weekend`) are forced to threshold `1.0` (never bet).
- **V2 — `asym_wknd_excl_hour_boost`** (current default) : V1 + `"hour_boost": {"good_hours": [int], "delta": float}`. When the bar's UTC hour is in `good_hours`, the effective threshold drops by `delta`. Skip cells are never boosted.

Thresholds are calibrated by **quantile**: `T_axis = quantile(P(up) on non-skip val of that axis, 1 - ASYM_VOL_FRAC_axis)`. The threshold lookup is centralized in `_bucket_threshold` — the **same logic appears in `api.py`, `predict.py`, and `train.py`**. If you change one, change all three. **Single confidence axis**: always compare `P(up)` against the bucket threshold; never against `0.5`, never build a separate "P(down)" axis.

## The standardized `api.py` contract

```python
NAME : str = "modele_up"

def evaluate_on_slice(start_ms: int, end_ms: int) -> pd.DataFrame:
    """One row per setup the model would have *taken* in [start_ms, end_ms].
    Columns: timestamp (int ms), direction ('up'),
             confidence (= P(up)), label (int 0|1, 1 = bet was correct)."""
```

`api.py` reads the **parquet directly and predicts on the fly** — it does **not** read `logs/test_predictions.csv`. So `evaluate_on_slice` can run on any window inside the parquet's range, but bars seen during training return an optimistic WR — the caller (`5m/main.py`) is responsible for picking an out-of-sample window. `direction` is always `"up"`.

`api.py` also exposes **`evaluate_full_on_slice(start_ms, end_ms)`** — the same window but **one row per bar** (taken or not): `timestamp, confidence, threshold, take (0|1), label, volatility_state, is_weekend, hour`. `evaluate_on_slice` is exactly this filtered to `take == 1`; both share the private `_predict_all_bars` helper. `5m/main.py` uses the full view to build the detailed per-bar logs.

## Naming traps

- **`config.DB_PATH` points to the parquet** (`db/btc_multi_tf_5m.parquet`), not the SQLite `DB/btc_multi_tf.db`.
- **"up" = "next candle is green"**, defined as `close[i+1] > open[i+1]` — the *direction of the next candle*, not "price rose by a threshold". Don't redefine the label without retraining everything downstream.
- **"confidence" is an alias of `p_up`**, a raw model probability, not a calibrated score. Threshold comparisons are always against `effective_threshold(row)`, never `0.5`.
- **The `5m/modele_up` parquet is a label-flipped copy of `5m/modele_down`'s** — rebuild it (step 2 of the build pipeline) whenever `modele_down`'s parquet changes. This is the **opposite** of `15min/modele_up`, whose parquet is independent.
- **The hour formula is 5m-native**: `((ts_ms // 5min) % 288) // 12` — in `_hour_from_ts_ms` (`api.py` / `predict.py`) and `hour_of_day_from_ms` (`train.py`).

## `config.py` knobs

- `WARMUP = 500` — bars dropped at the start (~1.7 days on 5m).
- `TRAIN_DAYS`, `TEST_DAYS`, `VAL_FRACTION` — temporal split. Rows are not independent: do not switch to shuffled K-fold.
- `ADAPTIVE_STRATEGY`, `ASYM_VOL_FRAC_WEEKDAY/WEEKEND`, `ASYM_SKIP_CELLS`, `ASYM_HOUR_BOOST_GOOD_HOURS/DELTA` — drive the sidecar.
- `ATR_LONG_PERIOD`, `VOLATILITY_HIGH/LOW` — define the 3-way `volatility_state` bucket.
- `XGB_SCALE_POS_WEIGHT` — `False` / `"auto"` / fixed float. Resolved in `_resolve_scale_pos_weight`.

All hyperparameters and calibration constants are the **`15min/` baseline**, carried over unchanged — this tree is a structural conversion to a 5m base TF. 5m-specific tuning is a separate pass.

## Hyperparameter convention

`XGB_REG_LAMBDA`, `XGB_REG_ALPHA`, `XGB_GAMMA` use an **`is not False`** identity check in `train.build_hyperparams` — setting them to `False` (not `0`, not `None`) means *don't pass the kwarg to `XGBClassifier`* so XGBoost's native default applies. `0` is a valid value that explicitly disables regularization.

## `analyse/` — diagnostic scripts

`analyse/` holds two standalone, read-only scripts — `check_new_features.py` (feature-ablation impact) and `analyze_features.py` (per-feature audit). They were copied verbatim from `15min/modele_up`; their hardcoded parquet paths are 15m-era — re-point them before running on 5m. They do not feed the build pipeline. Run from the package root.

## Repo-wide (recap from `../../../../CLAUDE.md`)

- Python 3.10+ (tested on 3.14). Manual install: `pip install requests pandas numpy xgboost scikit-learn scipy rich`. No `requirements.txt`.
- No tests, no linter, no build system. Verification = inspect `logs/runs.csv` across runs (`adaptive_total_wr`, `adaptive_total_taken`). The bar is beating `baseline_wr_up` (~0.50), not 50%. 5m candles are noisier than 15m, so expect a lower signal ceiling.
- No-lookahead MTF joins: every higher-TF feature is the latest bar provably closed at decision time `T = open_5m + 5m`.
- All persisted features are scale-invariant (returns, ratios, normalized indicators) — no raw prices in the parquet. Persisted timestamps are milliseconds.
- Windows / PowerShell is the primary shell.
- `*.csv`, `*.parquet`, `*.db`, `__pycache__/`, `btc_updown_model.json`, `btc_updown_model.thresholds.json` are gitignored — regeneratable, don't commit.
