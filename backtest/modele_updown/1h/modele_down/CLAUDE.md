# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

This file is package-scoped. See the repo-root `CLAUDE.md` (`../../../CLAUDE.md`) for repo-wide conventions, the `15min/` import trick, the team orchestrator (`main.py`), the live daemon, and the standardized `api.py` contract shared with the sister `modele_up` package.

## What this package is

`modele_down` is a **DOWN-only** XGBoost binary classifier. On the close of each 15m BTC/USD bar it predicts:

```
label = 1  if the NEXT 15m candle is red    (close[i+1] < open[i+1])
label = 0  if the NEXT 15m candle is green
```

The model output is `P(next is down)` and is treated as a **single confidence axis** (`confidence`). All bets the model emits are DOWN bets — there is no UP head and no two-sided arbitration inside this package. Down-bias is by construction.

Decision rule:
```
take DOWN bet  iff  P(down) >= effective_threshold(row)
```
where the threshold comes from the per-axis quantile calibration described below (Sidecar section). `NAME = "modele_updown"` (the package-level constant in `api.py`; the directory is `modele_down`).

## Working-directory contract — `cd 15min/modele_down/` first

`config.py` uses **relative paths** (`DB_PATH = "db/btc_multi_tf_15m.parquet"`, `MODEL_OUTPUT = "btc_updown_model.json"`). Running `python 15min/modele_down/train.py` from the repo root reads/writes into the wrong place. Always:

```powershell
cd backtest\modele_updown\15min\modele_down
```

`logs/` is anchored by `Path(__file__)` in `train.py`, so it always lands at `15min/modele_down/logs/` regardless of cwd. The on-disk folder is `DB/` (uppercase) while `config.DB_PATH` says `db/` — this only works because Windows is case-insensitive. Keep them consistent if you touch either.

## Build pipeline — `data.py` + the ETH patch

The master dataset is built by a **single self-contained script**, `data.py`, then patched once by **`add_eth_features.py`** (it appends the 14 ETH/USD `ef_*` features — see *Exogenous ETH features* below). `data.py` itself runs 5 phases:

```powershell
cd backtest\modele_updown\15min\modele_down

# Phase 1  fetch raw OHLCV from Bitstamp on 5 TFs (5m,15m,30m,1h,4h,1d) -> SQLite candles_raw
# Phase 2  compute normalized scale-invariant features -> SQLite candles
# Phase 3  DROP candles_raw + VACUUM, export one debug CSV per TF
# Phase 4  build_master_15m: no-lookahead MTF join + Tier1/Tier2 derived features,
#          precompute label / volatility_state / is_weekend -> DB/btc_multi_tf_15m.parquet
# Phase 5  recap
python data.py            # defaults: --pair btcusd --days 5000 --db db/btc_multi_tf.db
python add_eth_features.py # patch: append the 14 ETH ef_* features (idempotent, keeps .bak)

python train.py           # train + sidecar + logs/runs.csv + logs/test_predictions.csv
python predict.py         # smoke-test on the last test_predictions row
python api.py             # exercise evaluate_on_slice on the full parquet
```

`data.py` keeps raw `close`/`volume` in SQLite only — the parquet stores **only scale-invariant features**. After `add_eth_features.py` it is 43 cols = `timestamp_ms` + `timestamp` + 39 features + `label` + `volatility_state`. The per-TF `*.csv` files and `btc_multi_tf.db` are debug/intermediate artifacts; `train.py` and `api.py` read the **parquet**.

## Exogenous ETH features

`add_eth_features.py` appends 14 ETH/USD 15m features (`ef_ret1/2/4/8/16/48/96`, `ef_streak`, `ef_macd`, `ef_macd_hist`, `ef_vol_z`, `ef_upwick`, `ef_lowick`, `ef_body`), exact-aligned no-lookahead onto each BTC 15m bar (the concurrent ETH bar is closed at the decision time `T = open_15m + 15m`). Validated by a 6-window walk-forward: ETH lifts honest pooled WR by ~+1.4pt — every window improves. Raw ETH OHLCV lives in `eth_data/eth_15m.parquet` (2017-08 → present); BTC bars before ETH history get NaN (XGBoost routes NaN natively). The patch keeps a one-off `<parquet>.bak`; re-running `data.py` rebuilds the parquet **without** the `ef_*` columns, so re-run `add_eth_features.py` afterwards.

**Live:** the production model needs the 14 `ef_*` features at predict time, and `live/modele_15m/live_helpers/features_down.py` builds them — `_compute_eth_features` there is a byte-identical port of `compute_eth_features` in this script, fed by a live `ethusd` 15m fetch (`live.py` pulls `WARMUP_ETH_15M = 1000` bars). The concurrent ETH bar is joined by exact timestamp; if it is missing, the `ef_*` go NaN (graceful, and logged as a warning). The sister gold feature was **removed** from `modele_up` on 2026-05-22 — the gold CSV ends 2026-01-30 with no live feed, so it could only ever be `0.0` live; see `add_gold_features.py`.

`features.py` is the runtime loader used by `train.py`: it reads the master (parquet or CSV — auto-detected by suffix), drops the first `WARMUP` rows, drops rows with no valid `label`, and renames `timestamp_ms` -> `timestamp`. It does **not** compute `label`, `volatility_state`, or `is_weekend` — those are precomputed by `data.py` in Phase 4. If `features.py` raises "Colonnes pre-calculees manquantes", re-run `python data.py`.

## What `train.py` writes

- `btc_updown_model.json` — XGBoost native format.
- `btc_updown_model.thresholds.json` — the sidecar (shape below).
- `logs/runs.csv` — one row per `train_and_evaluate(log_run=True)`. Append-only with column-union reindex.
- `logs/test_predictions.csv` — written **only** by direct `python train.py` runs (`log_predictions=True` is hardcoded in `main()`). Schema: full feature row + `label` + `p_down` + `confidence` (`confidence` is an alias of `p_down`). Read by `predict.py`'s smoke test, **not** by `api.py`.

`train.py` CLI flags: `--label NAME` (run label in `runs.csv`), `--no-save`, `--no-log`. Feature columns = every parquet column except `timestamp`, `label`, and `volatility_state` (the last is kept in the parquet only as a bucket key — it is not fed to the model). Current config gives 39 features (25 base + 14 ETH `ef_*`).

### `runs.csv` legacy columns — read carefully

`logs/runs.csv` was started under a previous two-axis (UP+DOWN) regime. Old rows carry columns like `baseline_wr_up`, `adaptive_up_taken`, `delta_up_v0_w0`, `delta_down_v0_w0`, etc. Current rows write the single-axis schema (`baseline_wr_down`, `adaptive_total_taken`, `adaptive_total_wr`, `asym_T_weekday`, `asym_T_weekend`, ...). The reindex logic preserves all columns across runs, so both shapes coexist in the file. **Do not "clean up" the legacy columns** — `analyse/compare_runs.py` and the sister packages may key off them.

## Sidecar shape — V1/V2 quantile calibration

The sidecar is **not** a `{"deltas": {...}}` map. `ADAPTIVE_STRATEGY` in `config.py` selects one of two shapes:

- **V1 — `asym_wknd_excl`** : `{"strategy", "thresholds_per_wknd": {"0": float, "1": float}, "skip_cells": [[vol, wknd], ...], "metadata": {...}}`. One confidence threshold per weekday/weekend axis. Cells in `skip_cells` (combinations of `volatility_state × is_weekend`) are forced to threshold `1.0` (never bet).
- **V2 — `asym_wknd_excl_hour_boost`** (current default) : V1 + `"hour_boost": {"good_hours": [int], "delta": float}`. When the bar's UTC hour is in `good_hours`, the effective threshold drops by `delta`. Skip cells are never boosted.

Thresholds are calibrated by **quantile**, not by a target-WR search: `T_axis = quantile(p_down on non-skip val of that axis, 1 - ASYM_VOL_FRAC_axis)`. This keeps bet volume stable cross-window — see the `compute_thresholds_asym_wknd_excl` docstring in `train.py`. The base threshold lives entirely in the sidecar; there is no `ADAPTIVE_TARGET_WR` config constant.

The threshold lookup is centralized in `_bucket_threshold` — the **same logic appears in `api.py`, `predict.py`, and `train.py`** (`apply_thresholds_asym_*`). If you change one, change all three. **Single confidence axis**: always compare `P(down)` against the bucket threshold; never against `0.5`, never build a separate "P(up)" axis.

## The standardized `api.py` contract

```python
NAME : str = "modele_updown"

def evaluate_on_slice(start_ms: int, end_ms: int) -> pd.DataFrame:
    """One row per setup the model would have *taken* in [start_ms, end_ms].
    Columns: timestamp (int ms), direction ('down'),
             confidence (= P(down)), label (int 0|1, 1 = bet was correct)."""
```

`api.py` reads the **parquet directly and predicts on the fly** — it does **not** read `logs/test_predictions.csv`. So `evaluate_on_slice` can run on any window inside the parquet's range, but bars seen during training return an optimistic WR — the caller (`15min/main.py`) is responsible for picking an out-of-sample window. `direction` is always `"down"`. Keep this contract intact: the team orchestrator only ever sees this 4-column DataFrame.

`api.py` also exposes **`evaluate_full_on_slice(start_ms, end_ms)`** — the same window but **one row per bar** (taken or not): `timestamp, confidence, threshold, take (0|1), label, volatility_state, is_weekend, hour`. `evaluate_on_slice` is exactly this filtered to `take == 1`; both share the private `_predict_all_bars` helper, so don't change one prediction path without the other. `15min/main.py` uses the full view to build the detailed per-bar logs (`team_all_bars.csv` / `team_skipped.csv` / `team_taken.csv`).

## Naming traps

- **`config.DB_PATH` points to the parquet** (`db/btc_multi_tf_15m.parquet`), not the SQLite DB (`db/btc_multi_tf.db`). The SQLite file is used only inside `data.py`. Don't repoint `DB_PATH` at `.db`.
- **"down" = "next candle is red"**, defined as `close[i+1] < open[i+1]` — the *direction of the next candle*, not "price dropped by a threshold". Don't redefine the label without retraining everything downstream.
- **"confidence" is an alias of `p_down`**, a raw model probability, not a calibrated score. Threshold comparisons are always against `effective_threshold(row)`, never `0.5`.

## `config.py` knobs

- `WARMUP = 500` — bars dropped at the start (longest rolling indicator). Bump it if you add a feature with a longer lookback, or early rows get NaN-poisoned.
- `TRAIN_DAYS=4000`, `TEST_DAYS=365`, `VAL_FRACTION=0.1` — temporal split (train window before the last `TEST_DAYS`, val = last `VAL_FRACTION` of train+val). Rows are not independent: do not switch to shuffled K-fold.
- `ADAPTIVE_STRATEGY`, `ASYM_VOL_FRAC_WEEKDAY/WEEKEND`, `ASYM_SKIP_CELLS`, `ASYM_HOUR_BOOST_GOOD_HOURS/DELTA` — drive the sidecar (above).
- `ATR_LONG_PERIOD`, `VOLATILITY_HIGH/LOW` — define the 3-way `volatility_state` bucket.
- `XGB_SCALE_POS_WEIGHT` — `False` (don't pass the kwarg), `"auto"` (= `n_up / n_down` on the train fold), or a fixed float. Resolved in `_resolve_scale_pos_weight`.

## Hyperparameter convention

`XGB_REG_LAMBDA`, `XGB_REG_ALPHA`, `XGB_GAMMA` use an **`is not False`** identity check in `train.build_hyperparams` — setting them to `False` (not `0`, not `None`) means *don't pass the kwarg to `XGBClassifier`* so XGBoost's native default applies. `0` is a valid value that explicitly disables regularization. Preserve this convention for any similar opt-in hyperparameter.

## Repo-wide (recap from `../../../CLAUDE.md`)

- Python 3.10+ (tested on 3.14). Manual install: `pip install requests pandas numpy xgboost scikit-learn scipy rich`. No `requirements.txt`.
- No tests, no linter, no build system. Verification = inspect `logs/runs.csv` across runs (`adaptive_total_wr`, `adaptive_total_taken`). The bar is beating `baseline_wr_down` (~0.50), not 50% — half the candles are red regardless.
- No-lookahead MTF joins via `searchsorted` for the latest higher-TF bar provably closed at decision time `T = open_15m + 15m`. Don't bypass it.
- Bitstamp fetch layer (`data.py`) sees seconds; everything persisted (`timestamp_ms`, sidecar, features) is milliseconds.
- Windows / PowerShell is the primary shell.
- `*.csv`, `*.parquet`, `*.db`, `__pycache__/`, `btc_updown_model.json`, `btc_updown_model.thresholds.json` are gitignored — regeneratable, don't commit.
