# BTC 1h Directional Trading System

An XGBoost up/down classifier pair on the **1-hour BTC/USD candle**, wrapped in a
leveraged trading backtest with full intrabar exit management, and deployed as a
live trading daemon on **OKX perpetual futures** (`BTC-USDT-SWAP`).

Two sister models vote on the direction of the next 1h candle. A coherence rule plus a
stack of entry filters decides whether to take the trade. A 1-minute exit engine then
manages the position (time-decay stop-loss, trailing, ratchet floors, take-profit,
timestop). The same `config` and `filters` modules drive both the backtest and the live
daemon — by direct import, not by copy — so a rule can never drift between them.

> ### ⚠️ Real money
> `LIVE` mode places real orders with leverage on OKX. The daemon ships with
> `DRY_RUN=true` and that is the safe default. Read
> [`live/CLAUDE.md`](backtest/modele_updown/1h/modele_team/live/CLAUDE.md) in full before
> changing it — it documents the catastrophe backstop, startup reconciliation, position
> sizing and the complete pre-production audit history.
>
> Past backtest results are not predictive. This repository is a personal research
> system, not financial advice.

---

## Architecture

| Component | Path | Role |
|---|---|---|
| **`modele_up`** | `backtest/modele_updown/1h/modele_up/` | XGBoost binary classifier. `label = 1` iff the next 1h candle closes **green**. Outputs `p_up`. |
| **`modele_down`** | `backtest/modele_updown/1h/modele_down/` | Sister classifier. `label = 1` iff the next candle closes **red**. Outputs `p_down`. Also owns the SQLite `candles` table that every other component reads for raw OHLC. |
| **`regime`** | `backtest/modele_updown/1h/regime/` | bear / neutral / bull detector (trend + momentum + Wilder DMI, z-scored, with hysteresis). Modulates **DOWN exit management** only. Active in production. |
| **`modele_team`** | `backtest/modele_updown/1h/modele_team/` | The hub: trading backtest (`main.py`), every tunable knob (`config_main.py`), all decision logic (`filters.py`), log writers. |
| **`live`** | `backtest/modele_updown/1h/modele_team/live/` | The OKX daemon. Imports the backtest's config and filters directly; carries its own copies of the three model packages so it deploys standalone. |

### Decision flow

```
Bitstamp OHLCV (1h / 4h / ETH 1h)
        │
        ├──> modele_up    ──> p_up    ──┐
        └──> modele_down  ──> p_down  ──┤
                                        │
          per-bucket sidecar threshold  │   (volatility × weekend × hour)
                                        ▼
            PRE-coherence filters:  UP drop-hours · DOWN selectivity · UP cross-model
                                        ▼
                     coherence rule:  UP · DOWN · PASS · CANCEL
                                        ▼
            POST-coherence filters:  4h DMI sign · UP confidence floor
                                        ▼
                   concurrency cap (1 slot per side, 2 total)
                                        ▼
         entry at next bar open (maker limit at P0, else missed)
                                        ▼
    1-minute exit walk:  take-profit · trailing · ratchet floor
                         time-decay stop-loss · timestop
                         (regime- and chop-conditional overrides)
                                        ▼
                      OKX market exit  ──>  logs
```

### Team coherence rule

```
up bets UP   + down abstains  -> UP
down bets DN + up abstains    -> DOWN
both bet                      -> CANCEL
neither bet                   -> PASS
```

The rule never sees raw model output: the pre-coherence filters have already cleared
some `take` flags, which changes which bars can reach `CANCEL` or `PASS`. Filter order
is load-bearing and documented in
[`modele_team/CLAUDE.md`](backtest/modele_updown/1h/modele_team/CLAUDE.md).

---

## Requirements

- **Python 3.10+** (developed on 3.14)
- No `requirements.txt`. Install directly:

```bash
pip install requests pandas numpy xgboost scikit-learn scipy rich
pip install plotly   # only for the g.py dashboard
```

- The live container pins `python:3.12-slim` and **`xgboost==3.2.*`**. A version mismatch
  between the host that trained a model and the runtime that loads it silently breaks
  prediction parity — keep them aligned.

---

## Working-directory contract

**Every script resolves its paths relative to its own package.** Running
`python backtest/modele_updown/1h/modele_up/train.py` from the repository root reads and
writes in the wrong place. Always `cd` into the package first.

### 1. Build a dataset and train a model

Both directional packages build their **own** parquet from Bitstamp — neither derives
from the other. The chain is identical except that `modele_up` adds an alpha patch.

```bash
cd backtest/modele_updown/1h/modele_down
python data.py                      # cold multi-timeframe fetch -> SQLite -> db/btc_multi_tf_1h.parquet (slow)
python add_engineered_features.py   # patch the parquet in place (idempotent, keeps a .bak)
python add_eth_features.py          # exogenous ETH ef_* features
python train.py                     # model .json + .thresholds.json sidecar + logs/runs.csv
python predict.py                   # smoke test
python api.py                       # exercise the team-facing contract

cd ../modele_up
python data.py
python add_engineered_features.py
python add_eth_features.py
python add_alpha_features.py        # required: 3 of its 6 whitelisted features come from here
python train.py
```

Each `add_*.py` patch is idempotent, but **`data.py` rebuilds the parquet without the
patched columns** — re-run the whole chain after any `data.py`.

### 2. Run the team trading backtest

```bash
cd backtest/modele_updown/1h/modele_team
python main.py    # Rich report + logs/trades.csv + appends logs/trading_runs.csv
python g.py       # optional Plotly equity dashboard over the run just written
```

### 3. Run the live daemon

```bash
cd backtest/modele_updown/1h/modele_team/live
cp .env.example .env   # fill in OKX keys + Telegram; DRY_RUN=true by default
python live.py
```

Docker (build context is `modele_team/`, not `live/`):

```bash
cd backtest/modele_updown/1h/modele_team/live
docker compose build
docker compose up -d
docker compose logs -f live-okx-test
```

**After every retrain**, copy the fresh `btc_updown_model.json` and
`btc_updown_model.thresholds.json` into `live/modele_up/` and `live/modele_down/`, then
rebuild the image. Nothing automates this, and a stale copy fails silently.

---

## Where things live

```
.
├── README.md
├── CLAUDE.md                        # conventions and traps, for AI assistants and humans
├── backtest/modele_updown/1h/
│   ├── modele_up/                   # classifier (green), own dataset + patch chain
│   ├── modele_down/                 # classifier (red), owns the SQLite candles table
│   ├── regime/                      # bear/neutral/bull detector
│   └── modele_team/
│       ├── main.py                  # the trading backtest
│       ├── config_main.py           # ALL tunable knobs — single source of truth
│       ├── filters.py               # ALL decision logic, shared with live/
│       ├── okx_fees.py              # best-effort real maker/taker fee lookup
│       ├── detailed_log.py          # thin per-bar CSV writer
│       ├── full_logs.py             # wide per-bet CSV writer
│       ├── g.py                     # Plotly equity dashboard
│       ├── db_1m/                   # 1-minute intrabar source for the exit walk
│       ├── optimize/                # read-only research scripts + campaign reports
│       ├── logs/                    # all backtest output
│       └── live/                    # the OKX daemon (see its own CLAUDE.md)
└── docs/superpowers/specs/          # design notes (most are historical)
```

Each of the five packages carries its **own `CLAUDE.md`**, and those are the authority
for anything inside them. The richest are `modele_team/CLAUDE.md` (engine, filter order,
campaign history) and `modele_team/live/CLAUDE.md` (daemon internals, backtest/live
parity proofs, audit findings).

---

## Conventions that matter

- **No lookahead.** Higher-timeframe features are joined via `searchsorted` to the latest
  bar provably closed at decision time `T = open + 1h`. Indexing a higher timeframe
  directly leaks the future. The same discipline applies to every `add_*.py` patch and to
  the regime detector.
- **Milliseconds everywhere.** Only the Bitstamp fetch layer sees seconds. Everything
  persisted or passed around — `timestamp_ms`, sidecar entries, `traded_ts` — is in
  milliseconds.
- **Scale-invariant features only.** The parquet stores returns, ratios, normalized OBV
  and ATR% — never raw prices. Raw `close`/`volume` stay in SQLite, and the backtest
  reconstructs body returns as `(1 + close_ret) / (1 + open_ret) - 1`.
- **Rows are not independent.** Strictly temporal split: a `TRAIN_DAYS` window before the
  last `TEST_DAYS`, validation as the tail of train+val, with an optional `BRIDGE_DAYS`
  embargo. Never shuffled K-fold.
- **The feature set is a whitelist.** `config.FEATURE_WHITELIST` pins the exact columns
  fed to XGBoost (currently 6 per model, chosen by deterministic forward selection). The
  parquet carries 100+ candidates that are simply ignored; a missing whitelisted column
  is a hard error.
- **Single confidence axis.** Always compare `P(direction)` against the per-bucket sidecar
  threshold. Never against `0.5`, and never build a second opposite-direction axis inside
  a package.
- **Backtest and live share the decision *and* the engine.** `live/` imports
  `config_main.py` and `filters.py` from `modele_team/` — there is no `live/filters.py`.
  Renaming or deleting a name in either file can break the daemon at import. The live
  exit engine is a line-by-line port of the backtest's, walking the same 1-minute grid,
  with documented parity runs.

---

## Verifying a change

There are no tests, no linter and no build system, and none should be added without
discussion. Verification is empirical:

| Scope | What to read | What to watch |
|---|---|---|
| A model | `modele_*/logs/runs.csv` | `adaptive_total_wr`, `adaptive_total_taken`. The bar is beating the `baseline_wr_*` column (~0.51 up, ~0.50 down), not 50% |
| The team | `modele_team/logs/trading_runs.csv` (append-only) | net compounded return, net WR, EV per trade, max drawdown |
| Per trade | `modele_team/logs/trades.csv` | the real executed book (the detail logs are wider but include non-executed candidates) |
| Live | `live/logs/cycles.csv`, `positions.csv`, `live.log` | one row per hourly cycle, including which filter blocked a signal |

A swept parameter is only trusted here when it holds on **n ≥ ~150 trades**, validates on
**both halves** of the window, and sits on a **plateau** rather than an isolated spike.
`config_main.py`'s inline comments record the campaign name and date behind each default.

### Reference run

`python main.py` on a 700-day window, maker entries, `POSITION_FRACTION = 0.1` ×
`LEVERAGE = 10` (effective exposure 1.0):

| | Value |
|---|---|
| Signals / trades taken | 499 / 391 |
| Net compounded P/L | **+97.93%** |
| Net win rate | 57.55% |
| EV per trade | +0.1794% |
| Maker fill rate | 90.4% |

Two caveats, both load-bearing: the window **ends 2026-07-03** because the parquet and
SQLite stop there, so the backtest says nothing about the period since; and results scale
directly with `EXPOSURE`, so any figure quoted elsewhere in the docs at a different
`POSITION_FRACTION` is not comparable to this one.

---

## Configuration and secrets

All strategy knobs live in **`modele_team/config_main.py`** — `main.py` holds none of its
own. The live daemon imports that same file, so the two can never disagree on a value.
`live/config_live.py` holds only live-specific constants (instrument id, tick cadence, log
paths) and never redefines a strategy threshold.

`live/.env` (gitignored, see `.env.example`):

| Variable | Purpose |
|---|---|
| `OKX_API_KEY` / `OKX_API_SECRET` / `OKX_API_PASSPHRASE` | OKX v5 REST credentials. Scope to Trade only, and IP-whitelist them |
| `DRY_RUN` | `true` (default) stubs every write call. `false` is real money |
| `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` | operational notifications; a separate optional error bot receives every `logger.error` |
| `OKX_LEVERAGE` / `OKX_POSITION_FRACTION` | optional overrides. Left empty, they fall back to `config_main.py` — fill them only to deliberately diverge from what the backtest validated |

Gitignored and never committed: `*.csv`, `*.parquet`, `*.db`, `*.pkl`, `*.log`, `*.bak*`,
`__pycache__/`, `.env`. The model `.json` and `.thresholds.json` files **are** tracked —
they are the deployable artifact.
