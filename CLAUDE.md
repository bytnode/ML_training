# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Orientation — one tree, four packages

Everything live or under active work is in **`backtest/modele_updown/1h/`**: an XGBoost up/down
direction model on the **1h BTC/USD candle**, wrapped in a leveraged trading backtest, plus the
**live OKX daemon** that runs the same decision + execution code on real money.

| Package | Role |
|---|---|
| `1h/modele_up/` | XGBoost binary classifier, `label = 1` iff next 1h candle is **green**. Output `p_up`. |
| `1h/modele_down/` | Sister classifier, `label = 1` iff next 1h candle is **red**. Output `p_down`. Also owns the SQLite `candles` table every other package reads for raw OHLC. |
| `1h/regime/` | bear/neutral/bull detector. Modulates **DOWN exit management** only. Active in prod. |
| `1h/modele_team/` | The hub: `main.py` trading backtest, all tunable knobs (`config_main.py`), all decision logic (`filters.py`), log writers, and **`live/`** — the OKX daemon. |

**Each of these owns its own `CLAUDE.md`, and those are the authority for anything inside them.**
Read the relevant one before editing. Richest and most current: `modele_team/CLAUDE.md` (engine,
filters, campaign history) and `modele_team/live/CLAUDE.md` (daemon, backtest/live parity).

Dependency direction: `modele_up` / `modele_down` / `regime` know nothing about the team.
`modele_team` consumes them through `api.py`. `modele_team/live/` consumes `modele_team`.

### ⚠️ Working tree vs `HEAD`

On branch `algov2_test`, `HEAD` still carries large deleted-but-uncommitted surfaces: a `15min/`
tree, a `30min/` tree, a repo-root `live/modele_15m/` Polymarket daemon, `1h/live_team/` (the MEXC
predecessor of `modele_team/live/`), `ML_agent_poly`, `aure.json`. ~400 unstaged deletions.
**The working tree is the intent; don't restore anything from `HEAD` or from `docs/superpowers/`
without asking.** Likewise, Polymarket/MEXC/15m specs under `docs/superpowers/specs/` are history —
the only live-relevant one is `2026-07-09-okx-live-team-1h-design.md`.

## Working-directory contract — always `cd` into the package first

Every script resolves paths relative to its own package (`config.DB_PATH = "db/btc_multi_tf_1h.parquet"`).
Running `python backtest/modele_updown/1h/modele_up/train.py` from the repo root reads and writes in
the wrong place.

```powershell
# 1. Build + train a directional model (identical chain in modele_up and modele_down,
#    each builds its OWN parquet from Bitstamp — neither derives from the other)
cd backtest\modele_updown\1h\modele_down
python data.py                      # slow cold fetch: Bitstamp multi-TF -> SQLite -> DB/btc_multi_tf_1h.parquet
python add_engineered_features.py   # patch the parquet in place (idempotent, keeps a .bak)
python add_eth_features.py          # exogenous ETH ef_* features
python train.py                     # model .json + .thresholds.json sidecar + logs/runs.csv + logs/test_predictions.csv
python predict.py                   # smoke test
python api.py                       # exercise the team-facing contract

cd ..\modele_up                     # same, PLUS the alpha patch (3 of its 6 features come from it)
python data.py && python add_engineered_features.py && python add_eth_features.py && python add_alpha_features.py
python train.py

# 2. Team trading backtest (needs both models trained at least once)
cd ..\modele_team
python main.py                      # Rich report + logs/trades.csv + appends logs/trading_runs.csv
python g.py                         # optional Plotly dashboard over the run just written

# 3. Live OKX daemon (DRY_RUN=true by default — see live/CLAUDE.md before flipping it)
cd live
python live.py
```

`train.py` flags (both packages): `--label NAME`, `--no-save`, `--no-log`.
Each `add_*.py` patch is idempotent and rewrites columns in place, but **`data.py` rebuilds the
parquet without them** — re-run the whole patch chain after any `data.py`.

Docker (daemon only; the training pipeline is never containerized):

```powershell
cd backtest\modele_updown\1h\modele_team\live
docker compose build && docker compose up -d
docker compose logs -f live-okx-test
```

Build context is **`modele_team/`** (`context: ..`), not `live/` — the Dockerfile copies
`config_main.py`/`filters.py`/`okx_fees.py` from the parent before `COPY live/`. On this branch all
Compose names carry a `-test` suffix (project `live-okx-test`) so a `docker compose down` here can
never stop the production deployment on the same VPS; remove the suffixes if this branch is merged
to prod.

## The `1h/` import trick

`1h/` starts with a digit, so `from 1h.modele_up import api` is a syntax error. `modele_team/main.py`
injects its grandparent (`1h/`) onto `sys.path` and imports the siblings as top-level
(`from modele_up import api as api_up`). Copy that block from `main.py` for any new entry point —
don't rename `1h/` (the per-package working-directory contract depends on the layout).

⚠️ `filters.py` deliberately has **no `sys.path` bootstrap of its own**: it resolves `modele_down` /
`regime` to whatever the *caller* already placed on `sys.path` (`main.py` → the `1h/` siblings;
`live.py` → `live/`'s embedded copies). Python caches modules by name, so adding a self-sufficient
`sys.path.insert` to `filters.py` would silently shadow the live copies with the backtest ones. Also
never pre-import both a backtest package and its `live/` copy in one process.

## The `api.py` contract — the team coupling point

Both directional packages expose the same shape, and `modele_team` only ever sees these DataFrames:

```python
NAME: str   # "modele_up" / "modele_updown" — not enforced identical

def evaluate_on_slice(start_ms, end_ms) -> pd.DataFrame:
    """One row per setup the model would have TAKEN.
    Columns: timestamp (int ms), direction ('up'|'down'), confidence (= P(direction)), label (0|1)."""

def evaluate_full_on_slice(start_ms, end_ms) -> pd.DataFrame:
    """Same window, one row per BAR (taken or not):
    timestamp, confidence, threshold, take (0|1), label, volatility_state, is_weekend, hour."""
```

`evaluate_on_slice` is exactly the full view filtered to `take == 1`; both share the private
`_predict_all_bars`, so the taken-only and full views can never drift. `main.py` uses the full view.

Both read the **parquet directly and predict on the fly** — not `logs/test_predictions.csv`. So any
window inside the parquet's ~13.7-year range works, but **the caller must pick an out-of-sample
window**: bars seen during training return an optimistic WR. `main.py`'s `compute_window` solves
this by anchoring `end` to the latest timestamp common to both packages' `test_predictions.csv`,
then going back `TEAM_TEST_DAYS`.

## Team coherence rule

```
up bets UP   + down abstains  -> UP
down bets DN + up abstains    -> DOWN
both bet                      -> CANCEL
neither bet                   -> PASS
```

⚠️ The rule does **not** see raw model output — several entry filters run **before** it and have
already zeroed some `take` flags, and more filters plus a concurrency cap run after it. The exact
order is load-bearing and lives in *Entry filters* in `modele_team/CLAUDE.md`. Read it before
touching the pipeline.

## Backtest and live share the decision AND the engine

This is the single most important cross-cutting invariant:

- `live/` does **not** copy `config_main.py` / `filters.py` / `okx_fees.py` — it **imports them**
  from `modele_team/`. There is no `live/filters.py`. So deleting or renaming a name in either file
  can break the daemon **at import** (this has happened: 4 names removed → `ImportError` at startup).
- `live/position.py`'s exit engine is a line-by-line port of `main.py`'s `_up_managed_exit` /
  `_down_managed_exit`, walking the same 1-minute grid. Parity is verified and documented; **never
  re-implement a rule in `main.py` or in `live/`** — change it in `config_main.py` / `filters.py`.
- `live/modele_up/`, `live/modele_down/`, `live/regime/` **are** copies (code + model `.json` +
  `.thresholds.json` + `regime_seed.json`, never the heavy `DB/`). **After a retrain or any source
  edit, re-copy them by hand** — nothing automates this and a stale copy fails silently.
- The container pins `xgboost==3.2.*`. A version mismatch between the host that trained the model
  and the container that loads it silently breaks prediction parity (serialization format). If you
  retrain under a different xgboost, update the Dockerfile.

## Sidecar shape (single confidence axis)

`train.py` writes `btc_updown_model.thresholds.json` next to the model, in one of two quantile shapes
selected by `config.ADAPTIVE_STRATEGY`:

- **V1 `asym_wknd_excl`** — `{"strategy", "thresholds_per_wknd": {"0": T, "1": T}, "skip_cells": [[vol, wknd], ...], ...}`.
  Cells in `skip_cells` are forced to threshold `1.0` (never bet).
- **V2 `asym_wknd_excl_hour_boost`** (current default) — V1 plus `hour_boost: {good_hours, delta}`,
  plus adaptive bad-hour / bad-day lists that `train.py` computes and writes into the sidecar.

Thresholds are calibrated **by quantile** of the validation `p` distribution (volume-stable
cross-window), not by a target-WR search. The lookup is centralized in `_bucket_threshold`, which is
**duplicated in `api.py`, `predict.py` and `train.py`** of each package, and mirrored again in
`live/`. Change one, change all of them.

**Single confidence axis**: always compare `P(direction)` against the bucket threshold. Never
against `0.5`, and never build a second "other direction" axis inside a package.

## Data conventions (both directional packages)

- **No-lookahead MTF joins.** Higher-TF features are joined via `searchsorted` for the latest bar
  provably closed at decision time `T = open + 1h`. Using `closes[i]` of a higher TF directly leaks
  the future. The same discipline applies to every `add_*.py` patch.
- **Seconds only at the fetch layer.** Bitstamp returns seconds; everything persisted or passed
  around (`timestamp_ms`, sidecar, features, `traded_ts`) is **milliseconds**. Don't propagate seconds up.
- **Scale-invariant features only in the parquet** — returns, ratios, normalized OBV, ATR%. Raw
  `close`/`volume` stay in SQLite; the backtest reconstructs body returns as
  `(1+close_ret)/(1+open_ret) - 1` rather than storing prices.
- **Rows are not independent.** Temporal split (`TRAIN_DAYS` window before the last `TEST_DAYS`,
  val = last `VAL_FRACTION`, optional `BRIDGE_DAYS` embargo). Never switch to shuffled K-fold.
- **The feature set is a whitelist, not the parquet.** `config.FEATURE_WHITELIST` pins the exact
  columns fed to XGBoost (currently 6 per model, chosen by forward selection); the parquet carries
  100+ candidates that are simply ignored. A missing whitelisted column is a hard error. When the
  user says "train on the DB", they mean literal columns — don't quietly derive new indicators.
- Hour-of-day is derived natively from the 1h grid; `volatility_state` / `is_weekend` are precomputed
  by `data.py` as **sidecar bucket keys only** — they are not model features.

## Hyperparameter convention

`XGB_REG_LAMBDA`, `XGB_REG_ALPHA`, `XGB_GAMMA` use an **`is not False`** identity check in
`train.build_hyperparams`. `False` (not `0`, not `None`) means *don't pass the kwarg* so XGBoost's
native default applies; `0` explicitly disables regularization. `XGB_SCALE_POS_WEIGHT` works the same
way, and also accepts `"auto"`. Preserve this for any similar opt-in hyperparameter.

## Verification — no tests, no linter, no build system

Don't add `pytest` / `ruff` / a `Makefile` without asking. Verification is by inspecting logs:

- **Per model**: `modele_*/logs/runs.csv` — watch `adaptive_total_wr` / `adaptive_total_taken`. The
  bar to beat is the `baseline_wr_*` column (~0.51 up / ~0.50 down), not 50%.
- **Team**: `modele_team/logs/trading_runs.csv` (append-only, schema-tolerant) — net compounded
  return, WR, **EV/trade**, max drawdown. `logs/trades.csv` + the detail CSVs overwrite each run.
  `trades.csv` and the headline stats are the *real* executed book; `detailed_log` / `full_logs` are
  the wider per-bar / per-bet views.
- **Live**: `live/logs/cycles.csv` (1 row/hour incl. which filter blocked a signal), `positions.csv`,
  `positions_state.json`, `live.log`.
- `runs.csv` carries legacy columns from an older two-axis regime, preserved by column-union reindex.
  **Don't clean them up** — comparison scripts key off them.

A swept number is only trusted here at `n >= ~150` trades, validated on **both halves** of the
window, and sitting on a **plateau** rather than an isolated spike. `config_main.py`'s inline
comments carry the campaign name + date behind each default.

## Trusting the docs in this repo

The per-package `CLAUDE.md` files are detailed and carry real campaign history, but they drift.
Always cross-check against the working tree. Known, verified-as-of-2026-10-21 drift:

- `modele_up/CLAUDE.md` and `modele_down/CLAUDE.md` are **byte-identical copies of a 15min-era
  `modele_down` doc**. The 15m/`5m` paths, the 5-phase-only build chain, the 39-feature count and
  the "modele_up has no data.py" claim are all wrong for this tree. Treat them as background on
  the *conventions*, and the package's `config.py` + `data.py` as the fact.
- `modele_team/CLAUDE.md` says `fetch_1m.py` was removed; it is present. Its headline performance
  table is explicitly dated and its own text warns earlier numbers in the file were wrong.
- `regime/CLAUDE.md` points at `modele_team/compare_regime_variants.py`, which is not in the tree.
- Any specific number (WR, net %, a threshold) is a snapshot. `config_main.py` and `config*.py` are
  the only sources of truth for current values — read the code, not the prose.

## Repo-wide

- **Python 3.10+**, host tested on 3.14 (container pins 3.12). **No `requirements.txt`**:
  `pip install requests pandas numpy xgboost scikit-learn scipy rich` (plus `plotly` for `g.py`).
- **Windows / PowerShell** is the primary shell — use PowerShell syntax (`$env:VAR`, backtick
  continuation, `$null`) for commands run on this machine. Bash is available via the Bash tool.
- **Gitignored and never committed** (regeneratable artifacts or secrets): `*.csv`, `*.parquet`,
  `*.db`, `*.pkl`, `*.log`, `*.bak*`, `__pycache__/`, `.ruff_cache/`, `g_evolution.html`, `.env` and
  `.env.*` (except `.env.example`). Model `.json` / `.thresholds.json` **are** tracked in `1h/` —
  they are the deployable artifact.
- **Secrets**: `live/.env` holds real OKX keys (`OKX_API_KEY` / `OKX_API_SECRET` /
  `OKX_API_PASSPHRASE`), `DRY_RUN`, Telegram tokens, and the money-sensitive overrides
  `OKX_LEVERAGE` / `OKX_POSITION_FRACTION`. `live/.env.example` lists the keys.
  `okx_fees.py` separately reads `OKX_API_*` for a read-only fee lookup.
- **`DRY_RUN=true` is the default and the safe state.** Everything about flipping it to real money —
  the native catastrophe backstop, reconciliation, sizing, the audit history — is in
  `modele_team/live/CLAUDE.md`. Read it first.
