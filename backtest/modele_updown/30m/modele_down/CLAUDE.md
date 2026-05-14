# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

This file is package-scoped: see the repo-root `CLAUDE.md` (`../../../CLAUDE.md`) for repo-wide conventions and for `modele_15m`.

## What this package is

`modele_updown` is a **DOWN-only** XGBoost binary classifier. On the close of each 30m BTC/USD bar it predicts:

```
label = 1  if the NEXT 30m candle is red  (close[i+1] < open[i+1])
label = 0  if the NEXT 30m candle is green
```

The model output is `P(next is down)` and is treated as a single confidence axis. All bets the model emits are DOWN bets — there is no UP head and no two-sided arbitration. Down-bias is by construction, not a feature of the data.

Per-bucket adaptive threshold:
```
take DOWN bet  iff  P(down) >= ADAPTIVE_TARGET_WR + delta(volatility_state, is_weekend)
```
6 buckets total (vol ∈ {0,1,2} × wknd ∈ {0,1}), one delta per bucket, written to `btc_updown_model.thresholds.json`. **Do not reintroduce a separate UP delta** — the sidecar shape is `{"deltas": {"v_w": float, ...}, "metadata": {...}}` and `predict.decide()` / `api.evaluate_on_slice()` assume that shape.

## Working-directory contract — `cd backtest/modele_updown/30m/modele_down/` first

`config.py` uses **relative paths**: `DB_PATH = "db/btc_multi_tf_30m.parquet"` and `MODEL_OUTPUT = "btc_updown_model.json"`. Running from the package root is fine, but launching `python backtest/modele_updown/30m/modele_down/train.py` from the repo root will read/write into the wrong place. Always `cd` first.

`logs/` is anchored by `Path(__file__)` (in `train.py`), so it lives under `backtest/modele_updown/30m/modele_down/logs/` regardless. The artifacts (`runs.csv`, `test_predictions.csv`) are gitignored CSVs.

## Build pipeline

Le pipeline data est consolidé dans un seul script :

```powershell
cd backtest\modele_updown\30m\modele_down
python data.py            # cold fetch + features Tier 0+1+2 en un seul script
python train.py           # entraîne, écrit btc_updown_model.json + sidecar + logs/
python predict.py         # smoke-test
python api.py             # exercice evaluate_on_slice
```

## What `train.py` writes

- `btc_updown_model.json` — XGBoost native format.
- `btc_updown_model.thresholds.json` — sidecar `{"deltas": {"v_w": float, ...}, "metadata": {...}}`. Base of the threshold = `config.ADAPTIVE_TARGET_WR` (single source of truth, not duplicated in the sidecar).
- `logs/runs.csv` — one row per `train_and_evaluate(log_run=True)` call. Append-only with column-union reindex (so adding a new metric doesn't corrupt earlier rows).
- `logs/test_predictions.csv` — written **only** by direct `python train.py` invocations. The schema is `{full feature row}, label, p_down, confidence` (where `confidence` is an alias for `p_down`). `api.py` reads this file to implement the standardized inference contract.

### `runs.csv` legacy columns — read carefully

`logs/runs.csv` was started under a previous two-axis (UP+DOWN) regime. Old rows contain columns like `baseline_wr_up`, `adaptive_up_taken`, `adaptive_up_wr`, `delta_up_v0_w0`, `delta_down_v0_w0`, etc. Current rows write the single-axis schema (`baseline_wr_down`, `adaptive_total_taken`, `adaptive_total_wr`, `delta_v0_w0`, ...). The reindex logic preserves all columns across runs; both shapes coexist in the file. **Do not "clean up" the legacy columns** without checking whether `analyze_runs.py` (in the sister `modele_15m` package) or any external script keys off them.

## The standardized `api.py` contract

Every model package exposes the same shape to the `backtest/modele_updown/30m/main.py` team orchestrator:

```python
NAME : str = "modele_updown"

def evaluate_on_slice(start_ms: int, end_ms: int) -> pd.DataFrame:
    """One row per setup the model would have *taken* in the window.
    Columns: timestamp (int ms), direction ('down' for this package),
             confidence (= P(down)), label (int 0|1, 1 = bet was correct)."""
```

This package's implementation reads `logs/test_predictions.csv` (so it covers exactly the test slice produced by the most recent `python train.py`) and applies the per-bucket adaptive threshold to filter taken bets. **Keep this contract intact** — `direction` is always `"down"` for this model and the orchestrator only sees that standardized DataFrame.

## Naming traps

- **`config.DB_PATH` points to the parquet**: `db/btc_multi_tf_30m.parquet`. Do not confuse with the SQLite DB (`db/btc_multi_tf.db`). The SQLite is only used by `data.py` directly via hard-coded paths.
- **"down" in this package = "red candle next"**, defined as `close[i+1] < open[i+1]`. This is the *direction of the next candle*, not "price went down by some threshold". Do not redefine the label without retraining everything downstream.
- **"confidence" in `test_predictions.csv` is an alias of `p_down`**, not a calibrated score. Threshold comparisons are always against `ADAPTIVE_TARGET_WR + delta(bucket)`, never against 0.5.

## Data conventions (same as the root CLAUDE.md)

- **No-lookahead joins**: every MTF lookup uses `searchsorted` to find the latest higher-TF bar provably closed at decision time `T = open_30m + 30m`. The 15m fine-context lookup also uses this: `T - 15m` is the open of the last 15m bar closed at `T`. Don't bypass it.
- **Bitstamp seconds vs internal milliseconds**: only `data.fetch_klines_window` and the windowing in `data.fetch_timeframe` see seconds. Everything else is ms.
- **All parquet features are scale-invariant** by design — `data.compute_normalized_features` produces only relative quantities (returns, ratios, normalized OBV, ATR%). Raw close/volume are only kept in SQLite for MACD/BB/MFI computation, then dropped before the parquet write.
- **Rows are not independent**: same as `modele_15m`, the temporal split (`TRAIN_DAYS` window before the last `TEST_DAYS`, val = last `VAL_FRACTION` of train+val) is correct. Don't switch to shuffled splits.
- **`WARMUP = 500`** is calibrated for the longest indicator on 30m. If you add a feature with a longer lookback, bump `WARMUP` or the early rows get poisoned with NaNs that `features.py` would silently skip.

## Hyperparameter conventions

`XGB_REG_LAMBDA`, `XGB_REG_ALPHA`, `XGB_GAMMA` use the **`is not False`** identity check in `train.build_hyperparams` — setting them to `False` (not `0`, not `None`) means *don't pass the kwarg to `XGBClassifier`* so the native XGBoost default applies. `0` is a valid value meaning "explicitly disable regularization". Preserve this convention if you add similar opt-in hyperparameters.

## Repo-wide (recap from `../../CLAUDE.md`)

- Python 3.10+ (tested on 3.14). Manual install: `pip install requests pandas numpy xgboost scikit-learn scipy`. No `requirements.txt`.
- No tests, no linter, no build system. Verification = inspect `logs/runs.csv` across runs, watch `adaptive_total_wr` and `adaptive_total_taken`. Beating the baseline (`baseline_wr_down`, naturally ~0.50) is the bar — beating 50% is the wrong frame because half the candles are red regardless.
- Windows / PowerShell is the primary shell. Use PowerShell syntax in any commands run on this machine.
- `.csv`, `__pycache__`, `btc_updown_model.json`, `btc_updown_model.thresholds.json` are gitignored — regeneratable, don't commit.
