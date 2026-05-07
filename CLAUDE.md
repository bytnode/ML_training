# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repo layout

This is a monorepo. Code lives in subprojects, **not at the root**.

- `backtest/` — XGBoost binary classifier for "next-candle reversal after a body_ratio-filtered streak" on BTC/USD 15m bars (Bitstamp). The full pipeline (`data.py` → `train.py` → `predict.py`), hyperparameter sweep tooling, and a data-cleaning module. **Read `backtest/CLAUDE.md` before touching anything in there**.
- `live/` — Production orchestrator that runs the trained Aure model (`aure.json` at the repo root) in real conditions: fetches Bitstamp every 15min, predicts, archives to CSVs, sends Telegram notifications. **Read `live/CLAUDE.md` before touching anything in there**.

The root contains repo-wide config (`.gitignore`, this file) and the model artifact (`aure.json` + `aure.thresholds.json` sidecar — created by `live/orchestrator.py` at first run).

## Working-directory contract

`backtest/config.py` uses **relative paths** (`CSV_PATH = "training_data.csv"`, `MODEL_OUTPUT = "btc_reversal_model.json"`), so all scripts in `backtest/` must be run with `backtest/` as the cwd:

```powershell
cd backtest
python data.py
python train.py
```

Running `python backtest/train.py` from the repo root will read/write CSVs and model files in the *root* directory, which is wrong. (`LOGS_DIR` is anchored to `__file__` so `logs/` always lands inside `backtest/` — but `CSV_PATH` and `MODEL_OUTPUT` aren't, so don't rely on that pattern for the others.)

Same convention for `live/`: scripts must run with `live/` as cwd. `live/settings.py` uses `Path(__file__)` for its own paths (so `live/logs/` always lands in `live/`), but it imports `backtest/data.py` via a sys.path hack — to keep that working without subtle bugs, run from `live/`:

```powershell
cd live
python orchestrator.py
```

Required env vars for Telegram notifs (sinon le daemon tourne mais ne notifie pas) :

```powershell
$env:AURE_TELEGRAM_BOT_TOKEN = "123:abc..."
$env:AURE_TELEGRAM_CHAT_ID = "-100..."
```

## Repo-wide conventions

- **Python 3.10+** (tested on 3.14). No `requirements.txt`; install deps manually as listed in `backtest/CLAUDE.md`.
- **No tests, no linter, no build system.** Don't add `pytest` / `ruff` / `Makefile` without asking. Verification happens by inspecting `backtest/logs/runs.csv` across runs and watching the `breakdown_total_wr` / `wr_at_0.58` metrics.
- **Windows / PowerShell** is the primary shell. Use PowerShell syntax in any commands you generate (`$env:VAR`, backtick line continuation, `$null` not `/dev/null`).
- **Specs live in `backtest/docs/superpowers/specs/`** (e.g. the 2026-05-06 spec for `logs/test_predictions.csv`). Check there for recent design decisions before assuming the code is the authoritative source on intent.
