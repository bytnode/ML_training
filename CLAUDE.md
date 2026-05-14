# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What's here

The repo is a single code surface: `backtest/modele_updown/`. Everything else at the repo root (`aure.json`, `docs/superpowers/specs/`, `backtest/docs/superpowers/specs/`) is either a gitignored artifact or historical design notes that describe surfaces (`modele_15m`, the old `live/` + `infra/` Polymarket daemon) that have since been deleted. Don't restore those — cross-check against the working tree, not the specs.

There are three layers, in dependency order:

1. **`backtest/modele_updown/15min/modele_up/`** and **`backtest/modele_updown/15min/modele_down/`** — two sister XGBoost binary classifiers. Each predicts the direction of the next 15m BTC/USD candle from its own side.
   - `modele_up` : `label = 1` iff next candle is green (`close[i+1] > open[i+1]`). Output `p_up`.
   - `modele_down` : `label = 1` iff next candle is red. Output `p_down`.
   - Each package owns its own `CLAUDE.md` — **read those first** before editing inside a package. They document the per-package build chain (parquet vs CSV master, patch scripts, feature set), the sidecar shape, and the working-directory contract.
2. **`backtest/modele_updown/15min/main.py`** — team backtest orchestrator. Calls `api.evaluate_on_slice(start_ms, end_ms)` on both sister packages, applies the team coherence rule, prints a Rich report, and appends one row to `logs/team_runs.csv` (= `backtest/modele_updown/15min/logs/team_runs.csv`).
3. **`backtest/modele_updown/15min/live/`** — live trading daemon, **two modes** behind the `LIVE_MODE` env var:
   - `LIVE_MODE=false` (default / absent) — **paper trading**, no orders placed, no Polymarket calls, no `py-clob-client` / `web3` / `eth-account` deps touched at runtime (they're lazy-imported only when needed).
   - `LIVE_MODE=true` — **real money**. Same model + decision flow, but every `UP` / `DOWN` team decision triggers a FAK BUY on the matching `btc-updown-15m-<unix_s>` Polymarket market. Winning bets are auto-redeemed on-chain (`CTF.redeemPositions`) immediately after resolution. Pending bets are reconciled at startup and at the top of every cycle.

   Files in `live/`:
   - `live.py` — the loop. Reads `LIVE_MODE`; the only CLI flag is `--dry-run` (real mode but skip the actual `POST /order`).
   - `polymarket.py` — **the single file** that touches Polymarket / Polygon / `py-clob-client` / `web3`. `PolymarketClient.from_env()` + `place_fak_order` / `read_settlement` / `redeem` / balance reads. Empirically validated contract addresses pinned at the top.
   - `live_helpers/` — `fetch.py` (Bitstamp public OHLC), `features_up.py`, `features_down.py`, `notify.py` (Telegram, with `send_bet_placed` / `send_bet_resolved_real` / `send_stuck` / `send_low_matic` events on top of the existing paper-mode `send_resolution`).
   - `logs/live_predictions.csv` (one row per cycle, paper-equivalent forever) + `logs/bets.csv` (one row per real order attempt, only written when `LIVE_MODE=true`). 23-column schema for `bets.csv` is defined in `live.BETS_COLUMNS`.
   - `Dockerfile` + `docker-compose.yml` — `live-team` service. **Build context is the parent directory (`backtest/modele_updown/`)** so the image can `COPY` model artifacts from `15min/`. Compose declares this with `build: { context: .., dockerfile: live/Dockerfile }`. Same image works in both modes — flip `LIVE_MODE` in `.env` and `docker compose restart live-team`.
   - `Dockerfile.dockerignore` — BuildKit per-Dockerfile ignore. Without it the context balloons from ~MB to ~500 MB (strips `DB/`, `logs/`, `*.csv`, `*.parquet`, training-only modules).
   - `.env` (gitignored) + `.env.example` (template) — Telegram + (when `LIVE_MODE=true`) `POLYGON_PRIVATE_KEY`, `POLYGON_EOA_ADDRESS`, `POLYMARKET_PROXY_ADDRESS`, `POLYGON_RPC_URL`, `CLOB_HOST`, `GAMMA_HOST`, plus tunables `SLIPPAGE_BPS` / `BET_FRACTION` / `BET_MIN_USDC`. Optional `TELEGRAM_ERROR_BOT_TOKEN` / `TELEGRAM_ERROR_CHAT_ID` route a **separate** Telegram bot to receive every `logger.error(...)` / `logger.exception(...)` from the daemon — wired via a `logging.Handler` in `notify.py`, so any new error site picks it up automatically.

   **France (and most of Western EU) is geo-blocked by Polymarket** — `LIVE_MODE=true` only works from a non-blocked-region VPS. Paper mode runs anywhere. Design notes at `docs/superpowers/specs/2026-05-14-polymarket-live-betting-design.md`.

## The `15min/` import trick

`15min/` starts with a digit, so `from 15min.modele_up import api` is a syntax error. Both `main.py` and the Dockerfile rely on putting `15min/` itself on `sys.path` and importing the packages as top-level (`from modele_up import api as api_up`). If you add a new entry point that needs both packages, copy the `sys.path.insert` block from `main.py` — don't rename `15min/` to make this go away (the live container's `COPY 15min/...` paths and the per-package working-directory contract both depend on the name).

## Standardized `api.py` contract — the team coupling point

Both sister packages expose the same shape, and `main.py` only sees this DataFrame:

```python
NAME : str = "modele_up" | "modele_updown"   # not enforced as identical

def evaluate_on_slice(start_ms: int, end_ms: int) -> pd.DataFrame:
    """One row per setup the model would have *taken* in the window.
    Columns: timestamp (int ms), direction ('up' or 'down'),
             confidence (= P(next-is-this-direction)), label (int 0|1, 1 = bet was correct)."""
```

Important behavioural detail not visible from the signature: **the current `api.py` reads the package's parquet (`DB/btc_multi_tf_15m.parquet`) directly and predicts on the fly**. It does not depend on `logs/test_predictions.csv`. This means `evaluate_on_slice` can run on any window inside the parquet's ~13.7-year range, but the caller is responsible for picking an out-of-sample window — bars seen during training will return an optimistic WR. `main.py` solves this by anchoring `end` to the most recent timestamp present in **both** packages' `logs/test_predictions.csv` and going back `TEAM_TEST_DAYS` from there.

## Team coherence rule (used by both `main.py` and `live.py`)

```
up bets UP   + down abstains  -> KEEP up bet         (direction = UP)
down bets DN + up abstains    -> KEEP down bet       (direction = DOWN)
both bet                       -> CANCEL              (one says UP, other DOWN)
neither bet                    -> PASS
```

Empirically, `CANCEL` is rare (the 365-day team run logged in commit `578a85b` shows 0 cancellations on ~4k bets). Don't optimise for it without checking that against the latest `logs/team_runs.csv`.

## Sidecar shape (single confidence axis)

Both packages write `btc_updown_model.thresholds.json` with one of two strategies:

- **V1 — `asym_wknd_excl`** : `{"strategy": "asym_wknd_excl", "skip_cells": [[vol, wknd], ...], "thresholds_per_wknd": {"0": float, "1": float}, ...}`. One confidence threshold per weekday/weekend axis. Cells listed in `skip_cells` (combinations of `volatility_state × is_weekend`) are forced to threshold 1.0 (= never take).
- **V2 — `asym_wknd_excl_hour_boost`** : V1 plus a `hour_boost: {"good_hours": [int], "delta": float}`. When the bar's hour is in `good_hours`, the effective threshold drops by `delta`.

The threshold lookup is centralised in `_bucket_threshold` — same logic appears in `api.py` (for backtest) and `live.py` (for live). If you change one, change the other. **Single confidence axis** : compare `P(direction)` against the bucket threshold; never compare against `0.5` and never construct a separate "other direction" axis inside a package.

## Working-directory contract

Every script in this repo reads/writes paths relative to its own package. Always `cd` into the package first:

```powershell
# Training a sister model
cd backtest\modele_updown\15min\modele_up
python data.py        # cold-fetch + build parquet (slow, ~3-5 min)
python train.py       # train + sidecar + logs/runs.csv + logs/test_predictions.csv
python predict.py     # smoke-test
python api.py         # exercise evaluate_on_slice end-to-end

# Team backtest (after both packages have been trained at least once)
cd backtest\modele_updown\15min
python main.py        # team coherence run, writes logs/team_bets.csv + logs/team_runs.csv (inside 15min/)

# Live paper trading (host)
cd backtest\modele_updown\15min\live
python live.py        # runs forever, prints Rich panels per cycle, logs to live/logs/live_predictions.csv (inside 15min/live/)
```

Each package's own `CLAUDE.md` documents its full build chain (including any `add_*.py` / patch scripts that must run in order for `modele_down`'s CSV chain or `modele_up`'s parquet patches). **Don't run `train.py` from the repo root** — relative paths will resolve to the wrong place.

## Live mode — paper trading only

`live.py` is a paper-trading loop. It:

1. Sleeps until the next `xx:00 / xx:15 / xx:30 / xx:45 UTC` boundary.
2. Calls `freshness_retry` to make sure Bitstamp has actually published the just-closed bar (not just *a* later bar) before reading.
3. Resolves the previous cycle's prediction against the now-closed target bar (sets `resolved_correct`).
4. Fetches fresh OHLCV on 4 timeframes (`15m`, `5m`, `30m`, `1h`), builds features for both sides via `live_helpers/features_up.py` and `live_helpers/features_down.py`, predicts, applies the team coherence rule, appends one row to `logs/live_predictions.csv`, renders Rich panels.
5. On a resolved UP/DOWN bet (not PASS / CANCEL), sends a Telegram notification via `live_helpers/notify.py`.

There is **no order placement**, no wallet, no Polymarket CLOB. The earlier `live/` daemon that did all that has been removed.

Telegram credentials come from env vars `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID`. They live in `backtest/modele_updown/15min/live/.env` (gitignored), with `.env.example` checked in as a template alongside. If either is missing, the loop still runs and `notify` becomes a silent no-op.

## Docker — `live-team` service

The Dockerfile and compose file sit inside `backtest/modele_updown/15min/live/`. Compose itself runs from there, but the build context is the **parent** directory (`backtest/modele_updown/15min/`) so the Dockerfile can `COPY` model files from the sibling `modele_up/` and `modele_down/`:

```powershell
cd backtest\modele_updown\15min\live
docker compose build
docker compose up -d
docker compose logs -f live-team   # Rich panels render here thanks to tty: true
docker compose restart live-team   # restart after a retrain (model files are baked in)
docker compose down
```

`docker-compose.yml` mounts `./logs:/app/15min/live/logs` (host path = `backtest/modele_updown/15min/live/logs/`) so `live_predictions.csv` + `bets.csv` survive rebuilds, and reads `./.env` for Telegram + (when `LIVE_MODE=true`) the Polymarket creds. The image only `COPY`s the runtime essentials (`live/live.py`, `live/polymarket.py`, `live/live_helpers/`, each package's `config.py` + `data.py` (needed by `features_down`) + `btc_updown_model.json` + `btc_updown_model.thresholds.json`), nested under `/app/15min/` to mirror the host layout. The training pipeline (`train.py`, `data.py` for modele_up, patch scripts, parquet/SQLite) is **not** in the image — train on the host, rebuild the image, restart the service.

No port is published; the container only does outbound calls (Bitstamp public OHLC + Telegram).

## Hyperparameter convention shared by both packages

`XGB_REG_LAMBDA`, `XGB_REG_ALPHA`, `XGB_GAMMA` use an **`is not False`** identity check in each package's `train.build_hyperparams`. Setting them to `False` (not `0`, not `None`) means *don't pass the kwarg to `XGBClassifier`* — XGBoost's native default applies. `0` is a valid value that explicitly disables regularization. Preserve this convention if you add similar opt-in hyperparameters.

## Data conventions (apply to both packages)

- **No-lookahead MTF joins.** Higher-TF features are joined via `searchsorted` for the latest bar provably closed at decision time `T = open_15m + 15m`. Don't bypass it (using `closes[i]` on a higher TF directly leaks the future).
- **Bitstamp seconds vs internal milliseconds.** Only the fetch layer (`data.py`, `live_helpers/fetch.py`) sees seconds. Everything else — kline `open_time`, features, parquet `timestamp_ms`, sidecar `ts_ms` — is milliseconds. Don't propagate seconds upward.
- **Scale-invariant features.** All persisted features are relative (returns, ratios, normalized OBV, ATR%). Don't store raw prices in the parquet.
- **Rows are not independent.** Temporal split (`TRAIN_DAYS` window before the last `TEST_DAYS`, val = last `VAL_FRACTION` of train+val). Don't switch to shuffled K-fold.
- **DB columns are the feature set.** When the user says "train on the DB" they mean the literal columns in the parquet/CSV — don't quietly add derived indicators on top.

## Repo-wide conventions

- **Python 3.10+** (tested on 3.14). **No `requirements.txt`**. Host install: `pip install requests pandas numpy xgboost scikit-learn scipy rich`. Live container pins exact versions in the `Dockerfile`.
- **No tests, no linter, no build system.** Verification = inspect `logs/runs.csv` per package, watch `adaptive_total_wr` and `adaptive_total_taken`; for team runs, watch `team_wr` in `logs/team_runs.csv`. The "baseline" line is `y_test.mean()` (~0.51 for modele_up, ~0.50 for modele_down) — beating *that*, not 50%, is the bar. Don't add `pytest` / `ruff` / `Makefile` without asking.
- **Windows / PowerShell** is the primary shell. Use PowerShell syntax in any commands run on this machine (`$env:VAR`, backtick line continuation, `$null` not `/dev/null`). Bash via the Bash tool is available when needed.
- **Gitignored** : `*.csv`, `*.parquet`, `*.db`, `*.sqlite`, `*.log`, `__pycache__/`, `btc_updown_model.json`, `btc_updown_model.thresholds.json`, `aure.json`, `aure.thresholds.json`, `.env`, `.env.*` (except `.env.example`). Don't commit any of these — they are regeneratable artifacts or secrets.
