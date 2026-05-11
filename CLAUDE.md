# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repo layout — three independent code surfaces

The repo has three code surfaces, each with its own runtime contract. Don't assume edits in one propagate to the others.

- `backtest/modele_15m/` — the original XGBoost model. Reversal-after-streak classifier. Tracked, runnable, and **the model currently consumed by `live/`** (see live section below). Produces `btc_reversal_model.json` + sidecar.
- `backtest/modele_updown/modele_up/` and `backtest/modele_updown/modele_down/` — two sister XGBoost packages (untracked at the moment). Each predicts whether the next 15m candle is green / red. Each has its own package-scoped `CLAUDE.md` — read it before editing. They produce `btc_updown_model.json` + sidecar inside their own directory.
- `live/` — dockerised daemon. Wakes at every xx:00/15/30/45 UTC, detects setups, calls the `modele_15m` artifact, places bets on Polymarket, logs to `live/logs/`. `infra/` builds and runs it.

`aure.json` at the repo root is the **deployed** copy of `btc_reversal_model.json` (older snapshot kept at root for historical reasons; `live/` reads from `backtest/modele_15m/` directly, not from `aure.json`). The companion `aure.thresholds.json` is gitignored.

The previous `backtest/modele_updown/*.py` flat layout (`api.py`, `train.py`, `data.py`, …) is being **deleted** — `git status` shows those files as `D`. The new content lives in the `modele_up/` and `modele_down/` subdirs. Don't restore the deleted flat files.

`docs/superpowers/specs/` and `backtest/docs/superpowers/specs/` carry design intent (live orchestrator design, multi-pair dataset, test-predictions log). Useful for *why*, but cross-check against the working tree — some of the described code has been deleted or moved (e.g. the multi-pair experiment was reverted in `a422d18`).

## `backtest/modele_15m/` — reversal classifier, currently in production

A single XGBoost binary classifier that, on the close of each 15m BTC/USD bar, predicts the probability that **the next candle reverses a streak of N same-direction candles** (N ∈ {1,2,3,4}, last candle must satisfy `body_ratio ≥ BODY_RATIO_MIN = 0.6`). Output of `train.py` is `btc_reversal_model.json` plus the adaptive-threshold sidecar `btc_reversal_model.thresholds.json` (per-bucket deltas keyed on `volatility_state × is_weekend`).

The package's own `README.md` and `DOCUMENTATION.md` at `backtest/modele_15m/` cover the pipeline, feature set, label definition and adaptive-threshold contract in depth — read them before touching `data.py`, `train.py`, or `predict.py`. The points below are the working conventions that aren't obvious from those docs.

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

`backtest/modele_15m/api.py` defines the contract every model package will expose to the planned `backtest/main.py` team orchestrator (`main.py` does **not** yet exist on disk — it's referenced by `api.py` but no team backtest runner has been written):

```python
NAME : str = "modele_15m"

def evaluate_on_slice(start_ms: int, end_ms: int) -> pd.DataFrame:
    """Returns one row per setup the model would have *taken* in the window.
    Columns: timestamp (int ms), direction ('up'|'down'), confidence (float), label (int 0|1)."""
```

The `modele_15m` implementation reads `logs/test_predictions.csv` (regenerated every `python train.py`) and applies the adaptive-threshold sidecar to filter taken bets. Future packages implement `evaluate_on_slice` however they want (runtime fetch + predict, etc.). **Keep this contract intact** — when `main.py` is written it will only see the standardized DataFrame.

### What `train.py` writes

- `btc_reversal_model.json` — XGBoost native format.
- `btc_reversal_model.thresholds.json` — sidecar with `{"deltas": {"<vol>_<wknd>": float, ...}, "metadata": {...}}`. Base of the threshold = `config.ADAPTIVE_TARGET_WR` (single source of truth, not duplicated in the sidecar).
- `logs/runs.csv` — one row per `train_and_evaluate(log_run=True)` call. Schema is flat by design and `LOG_THRESHOLDS = [0.50, 0.55, 0.58, 0.60, 0.62]` is held stable independently of `THRESHOLDS` in config so `analyze_runs.py` keeps working across runs. New columns trigger a column-union reindex of the whole file (cheap because the file stays small — designed for < ~1000 runs).
- `logs/test_predictions.csv` — written **only** by direct `python train.py` runs (CLI sets `log_predictions=True`). `sweep.py` deliberately does *not* write it (overwriting it 30+ times per sweep with only the last config's predictions would be misleading).

### Hyperparameter conventions (shared across all XGBoost packages)

`XGB_REG_LAMBDA`, `XGB_REG_ALPHA`, `XGB_GAMMA` follow a special convention: setting them to **`False`** (not `0`, not `None`) means *don't pass this argument to `XGBClassifier` at all*, so XGBoost uses its native default. The check is `is not False` (identity, not truthiness) because `0` is a valid value meaning "disable regularization explicitly". Preserve this if you add similar opt-in hyperparameters.

`SEQ_TEST` filters the **test slice only**; `train` and `val` always see all `seq_len` values. The current default `[2, 3, 4]` excludes single-candle setups from evaluation (noisier).

### Data conventions to preserve (apply to all model packages)

- **Bitstamp seconds vs internal milliseconds** — `data.fetch_chunk` is the only place that sees seconds (`step`, `start`, `end`). Everything downstream is ms (kline shape `[open_time_ms, o, h, l, c, v]`). Don't propagate seconds upward.
- **MTF features are no-lookahead** — `last_closed_index(open_times_sorted, T, INTERVAL_MS[iv])` returns the latest higher-TF bar provably closed at decision time `T = times[last_idx] + 15m`. Don't bypass it; using `closes[i]` directly on a higher TF leaks the future.
- **Indicators are hand-rolled** (OBV, EMA, RSI, ATR, vol avg) in pure Python — no `talib` / `pandas-ta`. `WARMUP = 200` is calibrated for EMA-200 on 15m. Adding an indicator with a longer lookback requires raising `WARMUP` or the early rows get poisoned.
- **Rows are not independent** — in `modele_15m`, the same starting position can produce up to `MAX_SEQ_LEN = 4` rows. Random shuffling / `KFold` leaks. Use `TimeSeriesSplit` or walk-forward if you need CV. The current temporal split (`TRAIN_DAYS` window before the last `TEST_DAYS`, val = last `VAL_FRACTION` of train+val) is correct.
- **`BODY_RATIO_MIN` is a business filter, not a feature** (modele_15m only) — it defines who *enters* the dataset. Changing it changes the baseline WR mechanically and makes runs incomparable.

## `backtest/modele_updown/{modele_up, modele_down}` — sister direction classifiers

Two independent **next-candle direction** classifiers, each producing one side of a virtual UP/DOWN team. The label flips between the two — same pipeline, opposite target:

- `modele_up/` — `label = 1` iff next candle is green (`close[i+1] > open[i+1]`). Output `p_up`.
- `modele_down/` — `label = 1` iff next candle is red (`close[i+1] < open[i+1]`). Output `p_down`.

Each package owns its own `CLAUDE.md`, `data.py` / build chain, `train.py`, `predict.py`, `api.py`, `DB/`, `logs/`. **They are not symlinked** — copy-paste changes manually if a fix applies to both, and re-read the per-package CLAUDE.md before editing because the build chains diverge:

- `modele_up/` uses a **parquet** master (`DB/btc_multi_tf_15m.parquet`) built by a single `data.py` that fetches all 6 TFs (5m, 15m, 30m, 1h, 4h, 1d), plus optional `add_*.py` patch scripts for extra features.
- `modele_down/` uses a **CSV** master (`db/btc_multi_tf_15m.csv`) built incrementally by a chain (`data.py` → `patch_body_signed.py` → `fetch_and_add_5m.py` → `add_tier1_features.py` → `drop_redundant_features.py`). **Order matters** in this chain — later scripts read columns produced by earlier ones.

Both packages share with `modele_15m`:
- The same `api.py` contract (`evaluate_on_slice(start_ms, end_ms) -> DataFrame`), but `direction` is hardcoded `"up"` or `"down"` per package.
- The same hyperparameter convention (`False` means "don't pass the kwarg").
- The same adaptive sidecar shape `{"deltas": {"<vol>_<wknd>": float, ...}, "metadata": {...}}` with base = `ADAPTIVE_TARGET_WR`. **Single confidence axis** — there is no separate UP delta inside the DOWN package or vice versa.
- The same working-directory contract: `cd` into the package root before running anything.

Trap to remember: if you copy a parquet from `modele_down` into `modele_up` (or vice versa), invert the `label` column — otherwise the model learns the opposite direction.

## `live/` — dockerised Polymarket trader (consumes `modele_15m`)

`live/` is a daemon that wakes at every UTC 15m boundary, detects qualifying reversal setups, asks the `modele_15m` model whether to take a bet (using the adaptive sidecar), places real orders on Polymarket via the CLOB, and logs to `live/logs/{decisions.csv, bets.csv, wallet_history.csv, orchestrator.log}`.

Key wiring:

- `live/settings.py` injects `backtest/modele_15m/` into `sys.path` so `orchestrator.py` can `import predict`, `import data`, `from config import ...` directly from the modele_15m package. `MODEL_PATH` and `SIDECAR_PATH` are **absolute** (`REPO_ROOT / "backtest" / "modele_15m" / ...`) — the orchestrator pre-loads the model + sidecar via `inject_model_and_sidecar()` so `predict.py`'s lazy relative-path load never fires.
- `live/.env` (gitignored) holds Polygon keys + Polymarket addresses + Telegram tokens. Repo-root `.env` is loaded as fallback. `settings.assert_env_ready()` raises at startup if mandatory vars (`POLYGON_PRIVATE_KEY`, `POLYGON_EOA_ADDRESS`, `POLYMARKET_PROXY_ADDRESS`) are missing.
- `SEQ_LIVE = [2, 3, 4]` filters which `seq_len` setups the daemon fires on (default skips noisier `seq_len=1`).
- Bet sizing: `BET_FRACTION = 0.10` of free USDC per window, floor at `BET_MIN_USDC = 1.00`, hard cap at `BET_MAX_FRACTION = 0.95`.

Commands:

```powershell
# From repo root (Windows / PowerShell)
cd live
python orchestrator.py            # run forever (UTC 15m cycle)
python orchestrator.py --dry-run  # one cycle, log everything, no real order placed
```

### Docker — `infra/` builds + runs the daemon

```powershell
cd infra
docker compose up -d --build      # build context = repo root, then start in background
docker compose logs -f aure-live  # follow logs
docker compose restart aure-live  # restart after a retrain
docker compose down               # stop + remove
```

`infra/docker-compose.yml` mounts the model artifacts read-only at `/app/backtest/modele_15m/{btc_reversal_model.json,btc_reversal_model.thresholds.json}` (matching `MODEL_PATH` and `SIDECAR_PATH`) and persists `live/logs/` on the host. Reads secrets from `live/.env`. **No port published** — the daemon only does outbound calls.

If the live daemon is currently swapped to a different model (e.g. an `updown` head), update `settings.MODEL_PATH` / `SIDECAR_PATH` *and* the volume mounts in `infra/docker-compose.yml` together. `live/docker-compose.yml` (untracked) is a sibling override you may also need to keep in sync.

## Repo-wide conventions

- **Python 3.10+** (tested on 3.14). **No `requirements.txt`**; install deps manually: `pip install requests pandas numpy xgboost scikit-learn scipy python-dotenv` (the last one is required by `live/settings.py`). Live mode also needs `py-clob-client` and `web3` for Polymarket — see `infra/Dockerfile`.
- **No tests, no linter, no build system.** Don't add `pytest` / `ruff` / `Makefile` without asking. Verification happens by inspecting `logs/runs.csv` across runs and watching the per-model headline metric (`breakdown_total_wr` / `wr_at_0.58` for `modele_15m`; `adaptive_total_wr` / `adaptive_total_taken` for `modele_up` and `modele_down`). The "baseline" line in `train.py` output is `y_test.mean()` — beating that, not 50%, is the bar.
- **Windows / PowerShell** is the primary shell. Use PowerShell syntax in any commands (`$env:VAR`, backtick line continuation, `$null` not `/dev/null`). Bash is available via the Bash tool when needed.
- **Specs live in `docs/superpowers/specs/` and `backtest/docs/superpowers/specs/`**. Read them for intent on past decisions, but cross-check against the working tree — some described code has been deleted or moved.
- **`.gitignore` ignores `*.csv`, `*.parquet`, `*.db`, `__pycache__`, `btc_reversal_model.json`, `btc_updown_model.json`, `*.thresholds.json`, `live/logs/*.csv`, `.env`** — these are regeneratable artifacts or secrets. Don't commit them.
