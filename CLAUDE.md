# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What's here

Two top-level code areas:

- **`backtest/modele_updown/`** — all the research/training code, split into **three parallel timeframe trees**:
  - **`15min/`** — the mature tree. Three layers, documented in the rest of this file.
  - **`5m/`** — a sibling variant predicting the next **5-minute** candle. See *The `5m/` sibling tree* below.
  - **`30m/`** — a sibling variant and the current active optimization target. See *The `30m/` sibling tree* below. (The folder name is stale — its content has been converted to a **1h** base timeframe; the rename is pending.)
- **`live/`** — the **self-contained live trading daemon** (layer 3 below). It carries its own copies of the 15min model artifacts and is deployable on its own, independent of `backtest/`. (Was `live/modele_15m/` until it was flattened up one level to `live/`.)

Other repo-root content (`aure.json`, most of `docs/superpowers/specs/`) is either a gitignored artifact or historical design notes for surfaces (an old `modele_15m`, and an **old** top-level `live/` + `infra/` Polymarket daemon) that have since been **deleted** — don't restore those. ⚠️ Don't confuse that *deleted* old `live/` with the **current** repo-root `live/` daemon described above (it was renamed up from `live/modele_15m/`); the current `live/` is active, its design note is `docs/superpowers/specs/2026-05-14-polymarket-live-betting-design.md`. The exception is the `30m` design notes under `backtest/docs/superpowers/`, which describe live in-progress work. Always cross-check specs against the working tree.

The **`15min/`** tree has three layers, in dependency order:

1. **`backtest/modele_updown/15min/modele_up/`** and **`backtest/modele_updown/15min/modele_down/`** — two sister XGBoost binary classifiers. Each predicts the direction of the next 15m BTC/USD candle from its own side.
   - `modele_up` : `label = 1` iff next candle is green (`close[i+1] > open[i+1]`). Output `p_up`.
   - `modele_down` : `label = 1` iff next candle is red. Output `p_down`.
   - Each package owns its own `CLAUDE.md` — **read those first** before editing inside a package. They document the per-package build chain (`modele_down` has a 5-phase `data.py`; `modele_up` has **no `data.py`** — its parquet is a pre-built artifact), the feature set, the sidecar shape, and the working-directory contract.
2. **`backtest/modele_updown/15min/main.py`** — team backtest orchestrator. Calls `api.evaluate_on_slice(start_ms, end_ms)` on both sister packages, applies the team coherence rule, prints a Rich report, and appends one row to `logs/team_runs.csv` (= `backtest/modele_updown/15min/logs/team_runs.csv`).
3. **`live/`** — live trading daemon. **A self-contained folder at the repo root** (NOT under `backtest/`): it carries its own copies of the model artifacts (`modele_up/` + `modele_down/`) and needs nothing from `backtest/` to run. After a retrain, copy the fresh `btc_updown_model.json` + `btc_updown_model.thresholds.json` from `backtest/modele_updown/15min/modele_*/` into `live/modele_*/`. Two modes behind the `LIVE_MODE` env var:
   - `LIVE_MODE=false` (default / absent) — **paper trading**, no orders placed, no Polymarket calls, no `py-clob-client-v2` / `web3` / `eth-account` deps touched at runtime (they're lazy-imported only when needed).
   - `LIVE_MODE=true` — **real money**. Same model + decision flow, but every `UP` / `DOWN` team decision triggers a FAK BUY on the matching `btc-updown-15m-<unix_s>` Polymarket market. Pending bets are reconciled at startup and at the top of every cycle. **Winning bets are auto-redeemed via Polymarket's relayer** — CLOB V2 deposit-wallet redemption is operator-gated (`DepositWallet.execute` is `OnlyFactory`; `DepositWalletFactory.proxy` is `OnlyOperator`), so `redeem()` builds + EIP-712-signs a `redeemPositions` batch and submits it to the relayer (`POST {RELAYER_HOST}/submit`, `type:"WALLET"`). Requires a self-service `RELAYER_API_KEY` (polymarket.com → Settings → API Keys); without it `redeem()` returns `MANUAL_REDEEM_REQUIRED` and the daemon Telegram-alerts you to claim by hand.

   Files in `live/` (flat, self-contained — `live.py` + `modele_up/` + `modele_down/` + `live_helpers/` are all siblings):
   - `live.py` — the loop. Reads `LIVE_MODE`; the only CLI flag is `--dry-run` (real mode but skip the actual `POST /order`).
   - `polymarket.py` — **the single file** that touches Polymarket / Polygon / `py-clob-client-v2` / `web3`. `PolymarketClient.from_env()` + `place_fak_order` / `read_settlement` / `redeem` / balance reads. Uses CLOB **V2** (`py-clob-client-v2`; V1 is dead since the 2026-04-28 migration) with `signature_type=3` (deposit-wallet flow) and market FAK orders. Empirically validated contract addresses pinned at the top.
   - `live_helpers/` — `fetch.py` (Bitstamp public OHLC), `features_up.py`, `features_down.py`, `notify.py` (Telegram, with `send_bet_placed` / `send_bet_resolved_real` / `send_stuck` / `send_low_matic` events on top of the existing paper-mode `send_resolution`).
   - Two log files, both written in **both** modes:
     - `logs/live_predictions.csv` — one **comprehensive row per cycle** (every PASS/CANCEL/UP/DOWN): both models' `p`/threshold/take, the `volatility_state`/`is_weekend`/`hour` bucket, the just-closed bar OHLC, the previous cycle's resolution, and cumulative running stats.
     - `logs/bets.csv` — one row per **bet** (every UP/DOWN decision). Paper bets are written **complete** (decision + outcome) at resolution time; real bets follow the `pending → win/loss/stuck` lifecycle (`place_real_bet` + `resolve_pending_bets`). Schema (`live.BETS_COLUMNS`): the original 23 real-order columns + 13 appended context/resolution columns (`mode`, `threshold`, `edge`, `p_up`, `p_dn`, bucket, `resolved_*`). `order_status="paper"` marks a paper row.
   - `modele_up/` + `modele_down/` — **copies** of the runtime files the daemon needs: `config.py`, the model `.json` + `.thresholds.json`, and (modele_down only) `data.py` (its feature helpers are imported by `features_down.py`). These are copies of the training packages in `backtest/` — kept here so the folder is independent.
   - `Dockerfile` + `docker-compose.yml` — `live-team` service. **Build context is this folder itself** (`build: .`) — everything is inside it, no parent-directory hack. Same image works in both modes — flip `LIVE_MODE` in `.env` and `docker compose restart live-team`.
   - `Dockerfile.dockerignore` — BuildKit per-Dockerfile ignore (`__pycache__`, `logs/`, `.env`, stray data files).
   - `.env` (gitignored) — Telegram + (when `LIVE_MODE=true`) `POLYGON_PRIVATE_KEY`, `POLYGON_EOA_ADDRESS`, `POLYMARKET_PROXY_ADDRESS`, `POLYGON_RPC_URL`, `CLOB_HOST`, `GAMMA_HOST`, plus tunables `SLIPPAGE_BPS` / `BET_FRACTION` / `BET_MIN_USDC`. (There is **no** `.env.example` in `live/` anymore — it was removed in the flatten; copy keys from this env-var list. The `5m/live/` daemon still keeps its own `.env.example`.) Optional `TELEGRAM_ERROR_BOT_TOKEN` / `TELEGRAM_ERROR_CHAT_ID` route a **separate** Telegram bot to receive every `logger.error(...)` / `logger.exception(...)` from the daemon — wired via a `logging.Handler` in `notify.py`, so any new error site picks it up automatically.

   **France (and most of Western EU) is geo-blocked by Polymarket** — `LIVE_MODE=true` only works from a non-blocked-region VPS. Paper mode runs anywhere. Design notes at `docs/superpowers/specs/2026-05-14-polymarket-live-betting-design.md`.

## The `15min/` import trick

`15min/` starts with a digit, so `from 15min.modele_up import api` is a syntax error. Both `main.py` and the Dockerfile rely on putting `15min/` itself on `sys.path` and importing the packages as top-level (`from modele_up import api as api_up`). If you add a new entry point that needs both packages, copy the `sys.path.insert` block from `main.py` — don't rename `15min/` to make this go away (the per-package working-directory contract depends on the name). Note: the **live container no longer copies from `15min/`** — `live/` is self-contained and its Dockerfile `COPY`s its own flat files (`live.py`, `modele_up/`, `modele_down/`, `live_helpers/`); only `main.py`'s sys.path trick still relies on the `15min/` name.

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

Each `api.py` also exposes **`evaluate_full_on_slice(start_ms, end_ms)`** — the same window but **one row per bar** (every bar with a valid label and complete features, taken or not): `timestamp, confidence, threshold, take (0|1), label, volatility_state, is_weekend, hour`. `evaluate_on_slice` is exactly this filtered to `take == 1`; both share the private `_predict_all_bars` helper so the taken-only and full views can never drift. `main.py` uses `evaluate_full_on_slice` to build the detailed per-bar logs (below).

## Team coherence rule (used by `main.py`, `live.py`, and `detailed_log.py`)

```
up bets UP   + down abstains  -> KEEP up bet         (direction = UP)
down bets DN + up abstains    -> KEEP down bet       (direction = DOWN)
both bet                       -> CANCEL              (one says UP, other DOWN)
neither bet                    -> PASS
```

Empirically, `CANCEL` is rare (the 365-day team run logged in commit `578a85b` shows 0 cancellations on ~4k bets). Don't optimise for it without checking that against the latest `logs/team_runs.csv`.

## Detailed backtest logs

Every `python main.py` run writes three CSVs into `15min/logs/`, all sharing one 21-column per-bar schema (`detailed_log.LOG_COLUMNS`) so `skipped + taken` always partition `all`:

- `team_all_bars.csv` — one row per analyzed 15m bar in the window.
- `team_skipped.csv` — bars where the team placed no bet (`team_decision` is `PASS` or `CANCEL`).
- `team_taken.csv` — bars where the team placed a bet (`team_decision` is `UP` or `DOWN`); `result` is filled with `win` / `loss`.

Each row carries both models' probability + effective threshold + take flag, the team decision, the analyzed bar's OHLC, the next (target) bar's open/close, the next candle's colour, and the outcome. `detailed_log.py` owns the OHLC reconstruction and the CSV writing; `main.py` owns the per-bar team-decision assembly. OHLC is rebuilt from the SQLite `candles` table (`modele_down/DB/btc_multi_tf.db`) — the parquet stores no raw prices; if that DB is absent the price columns are left blank and the run still completes. These three files **overwrite** each run; `team_runs.csv` stays the append-only summary. (`team_taken.csv` replaced the old thin `team_bets.csv`.)

`main.py` also writes a **wide per-bet variant** into `15min/logs/full_logs/` via `full_logs.py` — `lost_bets.csv` / `won_bets.csv` / `all_bets.csv`, one row per **taken** bet (so `lost + won` partition `all`, the same population as `team_taken.csv`). Each row carries *everything*: the analyzed bar's OHLC + `volume`/`atr_pct`/microstructure (from the SQLite `candles` table), the target bar, **both models' full feature vectors** (prefixed `up_` / `dn_`), per-model `p`/threshold/take/edge, and the constant hyperparameters + config + sidecar of both packages (prefixed `up_hp_`/`up_cfg_`/`up_sc_`, and the `dn_` set) — ~144 columns. `full_logs.py` reuses the parquet/model/sidecar already cached by each `api.py` (no extra prediction pass) and is the analysis-oriented counterpart to the thin `detailed_log.py` view. Also overwrites each run.

## Sidecar shape (single confidence axis)

Both packages write `btc_updown_model.thresholds.json` with one of two strategies:

- **V1 — `asym_wknd_excl`** : `{"strategy": "asym_wknd_excl", "skip_cells": [[vol, wknd], ...], "thresholds_per_wknd": {"0": float, "1": float}, ...}`. One confidence threshold per weekday/weekend axis. Cells listed in `skip_cells` (combinations of `volatility_state × is_weekend`) are forced to threshold 1.0 (= never take).
- **V2 — `asym_wknd_excl_hour_boost`** : V1 plus a `hour_boost: {"good_hours": [int], "delta": float}`. When the bar's hour is in `good_hours`, the effective threshold drops by `delta`.

The threshold lookup is centralised in `_bucket_threshold` — same logic appears in `api.py` (for backtest) and `live.py` (for live). If you change one, change the other. **Single confidence axis** : compare `P(direction)` against the bucket threshold; never compare against `0.5` and never construct a separate "other direction" axis inside a package.

This V1/V2 quantile sidecar is the **`15min/`** shape. The `30m/` tree still uses an older delta-based sidecar — see *The `30m/` sibling tree* below.

## Working-directory contract

Every script in this repo reads/writes paths relative to its own package. Always `cd` into the package first:

```powershell
# Training a sister model (modele_down shown — it owns the data build;
# modele_up has no data.py, its parquet is a pre-built frozen artifact)
cd backtest\modele_updown\15min\modele_down
python data.py        # cold-fetch + build parquet (slow, ~3-5 min) — modele_down only
python train.py       # train + sidecar + logs/runs.csv + logs/test_predictions.csv
python predict.py     # smoke-test
python api.py         # exercise evaluate_on_slice end-to-end

# Team backtest (after both packages have been trained at least once)
cd backtest\modele_updown\15min
python main.py        # team coherence run, writes the 3 detailed logs + logs/team_runs.csv (inside 15min/)

# Live trading daemon (host) — self-contained folder at the repo root
cd live
python live.py        # runs forever, prints Rich panels per cycle, logs to logs/live_predictions.csv + logs/bets.csv
```

Each package's own `CLAUDE.md` documents its full build chain — `modele_down`'s self-contained 5-phase `data.py`, and the fact that `modele_up` has **no `data.py`** (its parquet is a frozen, pre-built artifact). Neither `15min/` package has an `add_*.py` patch chain — that exists only in `30m/modele_down`. **Don't run `train.py` from the repo root** — relative paths will resolve to the wrong place.

**Read-only experiment directories** (not part of any build pipeline; both are documented in the owning package's `CLAUDE.md`):

- **`analyse/`** — standalone numbered diagnostic scripts (`00_…` … `08_…`, each a `.py` + `.md` writeup) plus `analyze_features.py` / `check_new_features.py` / `compare_runs.py`. Present under both `15min/` packages and `30m/modele_down`. Run from the package root.
- **`optimize/`** — `sweep_*.py` / numbered exploration scripts (hyperparameter, calibration, feature-filter searches). Present under both `30m/` packages **and** `15min/modele_down` (the latter currently untracked, alongside its `eth_best_cols.txt` artifact). Read-only; they don't feed the build pipeline.

## The `live.py` cycle

`live.py` runs forever, one cycle per `xx:00 / xx:15 / xx:30 / xx:45 UTC` boundary. The decision flow is identical in both modes — `LIVE_MODE=true` only **adds** the real-money steps (marked *real-mode only* below).

Each cycle:

1. Sleeps until the next 15m boundary.
2. Calls `freshness_retry` to confirm Bitstamp has published the just-closed bar (not merely *a* later bar) before reading.
3. *Real-mode only* — `resolve_pending_bets`: reconciles every `pending` row in `bets.csv` (Polymarket settlement first, Bitstamp fallback after 1 cycle, `stuck` after 6); winning bets are redeemed on-chain.
4. Resolves the previous cycle's paper prediction against the now-closed target bar (sets `resolved_correct`).
5. Fetches fresh OHLCV on 4 timeframes (`15m`, `5m`, `30m`, `1h`), builds features for both sides via `live_helpers/features_up.py` / `features_down.py`, predicts, applies the team coherence rule, appends one row to `logs/live_predictions.csv`, renders Rich panels.
6. *Real-mode only* — on an `UP` / `DOWN` team decision, `place_real_bet` places the FAK BUY and appends one row to `bets.csv`.
7. Sends Telegram notifications via `live_helpers/notify.py` (paper resolution always; `send_bet_placed` / `send_bet_resolved_real` / `send_stuck` in real mode).

Paper mode (`LIVE_MODE=false`, the default) skips steps 3 and 6 — no wallet, no Polymarket CLOB, and `live_predictions.csv` is the only log written. Real-mode reconciliation (step 3) also runs once at startup to catch stale pending bets.

Telegram credentials come from env vars `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` in `live/.env` (gitignored; there is no `.env.example` in `live/` — copy the keys from the file-list above). If either is missing, the loop still runs and `notify` becomes a silent no-op.

## Docker — `live-team` service

The Dockerfile and compose file sit inside `live/`. The folder is self-contained, so the build context is **the folder itself** (`build: .`) — no parent-directory hack:

```powershell
cd live
docker compose build
docker compose up -d
docker compose logs -f live-team   # Rich panels render here thanks to tty: true
docker compose restart live-team   # restart after a retrain (model files are baked in)
docker compose down
```

`docker-compose.yml` mounts `./logs:/app/logs` (host path = `live/logs/`) so `live_predictions.csv` + `bets.csv` survive rebuilds, and reads `./.env` for Telegram + (when `LIVE_MODE=true`) the Polymarket + relayer creds. The build context is the folder itself, copied flat into `/app/` (`live.py`, `polymarket.py`, `live_helpers/`, `modele_up/`, `modele_down/`). The whole training pipeline (`train.py`, `features.py`/`predict.py`, the parquet + SQLite DB) is **not** in the folder at all — it lives in `backtest/`. Retrain on the host, copy the fresh model `.json` files into `live/modele_*/`, rebuild the image, restart the service.

No port is published; the container only does outbound calls — Bitstamp public OHLC + Telegram, plus the Polymarket CLOB / Polygon RPC when `LIVE_MODE=true`.

## The `30m/` sibling tree

`backtest/modele_updown/30m/` is a parallel timeframe variant of the `15min/` tree. It mirrors the same shape — `modele_up/` + `modele_down/` sister packages plus a `main.py` team orchestrator — but with several deliberate differences. There is **no `live/` daemon under `30m/`** — only the `15min/` daemon (now flattened to the repo-root `live/`) and the `5m/live/` daemon exist.

- **The folder name is stale.** Commit `5ff4ca3` converted this tree's content to a **1h base timeframe**; the rename `30m/` → `1h/` is still pending. Inside the tree, `main.py`, both package `CLAUDE.md`s, and `config.py` already say `1h`. Treat it as a **1h** model — the `30m/` directory name is the mismatch, not the code.
- **Older calibration regime.** The sidecar is a delta map — `{"deltas": {"v_w": float, ...}, "metadata": {...}}` — and the effective threshold is `config.ADAPTIVE_TARGET_WR + delta(volatility_state, is_weekend)` across 6 buckets. This is **not** the V1/V2 quantile sidecar used by `15min/`. Don't copy sidecar or `_bucket_threshold` logic between the two trees.
- **Different `api.py` source.** `30m/modele_down/api.py` reads `logs/test_predictions.csv` (so it covers exactly the last `train.py` test slice), unlike `15min/`'s on-the-fly parquet prediction. Read each package's `api.py` before assuming which one a tree uses.
- **Build chains differ per package.** `30m/modele_down` builds its parquet with `data.py` / `rebuild_master.py` plus a **patch chain** of `add_*.py` scripts (`add_5m_features.py`, `add_eth_features.py`, `add_coinbase_features.py`, `add_funding_rate.py`, …); several are still untracked, so cross-check the working tree, not just the package `CLAUDE.md`. `30m/modele_up` has no `data.py` — it copies `modele_down`'s parquet and flips the label (`df["label"] = 1 - df["label"]`).
- **`optimize/` sweep scripts.** Both `30m/` packages carry an `optimize/` directory of `sweep_*.py` scripts (hyperparameter, calibration, feature-filter searches). They are read-only experiments and don't feed the build pipeline. (No longer `30m`-exclusive — `15min/modele_down` has one too; see *Read-only experiment directories* above.)

This tree is the **current active optimization target** (pushing `modele_down`'s win rate). Each `30m/` package owns its own `CLAUDE.md` — read it first; it is the authority for that tree.

## The `5m/` sibling tree

`backtest/modele_updown/5m/` is a parallel timeframe variant of the `15min/` tree, predicting the direction of the next **5-minute** BTC/USD candle. It mirrors the `15min/` shape exactly — `modele_up/` + `modele_down/` sister packages, a `main.py` team orchestrator, `detailed_log.py` + `full_logs.py` — and keeps the **same V1/V2 quantile sidecar** and the same on-the-fly-parquet `api.py` contract. It also carries its **own self-contained `live/` daemon** (`backtest/modele_updown/5m/live/` — tracked, mirrors the repo-root `live/` layout: `live.py` + `polymarket.py` + `live_helpers/` + `modele_up/` + `modele_down/` + Dockerfile + its own `.env.example`). The primary deployed daemon is still the 15min one at the repo root; `5m/live/` is the 5m counterpart.

It is a structural clone of `15min/`, one base TF lower. The deliberate differences:

- **Base TF = 5m.** The label is the direction of the next 5m candle.
- **Timeframe ladder shifted down.** `15min/` uses base 15m + fine context 5m + MTF `[30m,1h,4h,1d]`. `5m/` uses **base 5m + fine context 1m + MTF `[15m,30m,1h,4h,1d]`**. `1m` is a new timeframe fetched from Bitstamp — `modele_down/data.py` fetches **7 TFs** (`1m,5m,15m,30m,1h,4h,1d`) and builds `DB/btc_multi_tf_5m.parquet`.
- **`modele_up` derives its parquet from `modele_down`** (copy + `df["label"] = 1 - df["label"]`), exactly like `30m/modele_up` — *not* the independent frozen parquet that `15min/modele_up` uses.
- **The hour helper is 5m-native** (`((ts // 5min) % 288) // 12`) in `api.py` / `predict.py` / `train.py` of both packages.
- **Hyperparameters, feature set and calibration are the `15min/` baseline**, carried over unchanged — this was a structural conversion; 5m-specific tuning is a later pass (`optimize/`). 5m candles are noisier, so expect a lower signal ceiling than the 15m/1h models.
- **The `sys.path` import trick applies the same way** — `5m/` also starts with a digit. Don't pre-import the `15min/` and `5m/` packages in one process.

To run the team backtest: build `modele_down`'s parquet + train it, derive `modele_up`'s parquet + train it, then `cd backtest\modele_updown\5m && python main.py`. Each `5m/` package owns its own `CLAUDE.md` — read it first.

## Hyperparameter convention shared by both packages

`XGB_REG_LAMBDA`, `XGB_REG_ALPHA`, `XGB_GAMMA` use an **`is not False`** identity check in each package's `train.build_hyperparams`. Setting them to `False` (not `0`, not `None`) means *don't pass the kwarg to `XGBClassifier`* — XGBoost's native default applies. `0` is a valid value that explicitly disables regularization. Preserve this convention if you add similar opt-in hyperparameters.

## Data conventions (apply to both packages)

- **No-lookahead MTF joins.** Higher-TF features are joined via `searchsorted` for the latest bar provably closed at decision time `T = open_15m + 15m`. Don't bypass it (using `closes[i]` on a higher TF directly leaks the future).
- **Bitstamp seconds vs internal milliseconds.** Only the fetch layer (`data.py`, `live_helpers/fetch.py`) sees seconds. Everything else — kline `open_time`, features, parquet `timestamp_ms`, sidecar `ts_ms` — is milliseconds. Don't propagate seconds upward.
- **Scale-invariant features.** All persisted features are relative (returns, ratios, normalized OBV, ATR%). Don't store raw prices in the parquet.
- **Rows are not independent.** Temporal split (`TRAIN_DAYS` window before the last `TEST_DAYS`, val = last `VAL_FRACTION` of train+val). Don't switch to shuffled K-fold.
- **DB columns are the feature set.** When the user says "train on the DB" they mean the literal columns in the parquet/CSV — don't quietly add derived indicators on top.

## Repo-wide conventions

- **Python 3.10+** (tested on 3.14). **No `requirements.txt`**. Host install: `pip install requests pandas numpy xgboost scikit-learn scipy rich` (plus `plotly` for the `15min/logs/analyse_logs/` charts). Live container pins exact versions in the `Dockerfile`.
- **No tests, no linter, no build system.** Verification = inspect `logs/runs.csv` per package, watch `adaptive_total_wr` and `adaptive_total_taken`; for team runs, watch `team_wr` in `logs/team_runs.csv`. The "baseline" line is `y_test.mean()` (~0.51 for modele_up, ~0.50 for modele_down) — beating *that*, not 50%, is the bar. Don't add `pytest` / `ruff` / `Makefile` without asking.
- **Windows / PowerShell** is the primary shell. Use PowerShell syntax in any commands run on this machine (`$env:VAR`, backtick line continuation, `$null` not `/dev/null`). Bash via the Bash tool is available when needed.
- **Gitignored** : `*.csv`, `*.parquet`, `*.db`, `*.sqlite`, `*.log`, `__pycache__/`, `btc_updown_model.json`, `btc_updown_model.thresholds.json`, `aure.json`, `aure.thresholds.json`, `.env`, `.env.*` (except `.env.example`). Don't commit any of these — they are regeneratable artifacts or secrets.
