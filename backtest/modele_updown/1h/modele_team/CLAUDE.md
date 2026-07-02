# CLAUDE.md — modele_team

The **third package** of the `1h/` tree, sibling to `modele_up/` and `modele_down/`.
It holds the **entire team system**: the trading-backtest orchestrator, its log
writers, and the 1-minute data fetcher. The two directional models stay separate
packages; `modele_team` only consumes them through their standardized `api.py`
(`evaluate_full_on_slice` → one row per bar, taken or not).

> This package is the **1h** tree's team layer. It is the trading-P/L successor to
> the `15min/` tree's WR-only `main.py`: instead of counting win/loss it **simulates
> the real trade** (entry, intrabar exit management, leverage, fees) and reports
> compounded equity. Read the repo-root `CLAUDE.md` for how the three timeframe
> trees relate, and each sister package's own `CLAUDE.md` before editing inside it.

## Contents

- `main.py` — the **TRADING backtest** orchestrator (real P/L net of fees, not just WR).
  Pipeline: predict both models → team coherence rule → **DMI directional filter** →
  **UP-confidence filter** → simulate each bet → Rich report (incl. an **EV-per-trade**
  table + headline) + CSVs.
  Writes `logs/trades.csv` + `logs/trading_runs.csv` and delegates the detailed/full logs.
- `config_main.py` — **all** editable knobs of `main.py`. `main.py` imports it and
  holds no tunables of its own (only `__file__`-derived paths). This is where you
  change anything: window, fees, capital/leverage/exposure, MAKER vs TAKER, the
  **DMI filter** (`DMI_FILTER_ENABLED` / `DMI_LENGTH` / `DMI_TIMEFRAME`), the **UP
  confidence filter** (`UP_CONF_FILTER_ENABLED` / `UP_CONF_MIN`), and the per-side
  UP/DOWN exit management. Read its inline comments — they record the sweep
  results behind each default.
- `detailed_log.py` — the thin per-bar CSVs into `logs/` (shared `LOG_COLUMNS`).
- `full_logs.py` — the wide per-bet CSVs into `logs/full_logs/`.
- `fetch_1m.py` — fetches raw 1-min Bitstamp OHLCV into `DB_1m/btc_1m.parquet`, the
  intrabar source for SL / TP / trailing / maker-fill simulation. Standalone — does
  **not** touch modele_down's SQLite/parquet; writes raw (non-normalized) OHLCV.
- `DB_1m/` — 1-minute data (gitignored). `logs/` — all team outputs.

## The trading engine (the part that needs `main.py` + `config_main.py` together)

This is the non-obvious core. Per team bet at decision bar `i`:

1. **Entry** at the `open` of bar `i+1` (`traded_ts = ts + 1h`). UP → LONG, DOWN → SHORT.
2. **Body return** of bar `i+1` comes from modele_down's SQLite `candles` table,
   `timeframe='1h'`, as `body = (1+close_ret)/(1+open_ret) − 1` — scale-invariant, no
   raw prices. (`_load_body_returns`'s docstring still says "30m"; the **query says
   `1h`** and is correct — this is a 1h model. The DB lives under modele_down's `db/`
   or `DB/btc_multi_tf.db`; absent → `FileNotFoundError`, run its `data.py`.)
3. **Exit management** (per side, `_up_managed_exit` / `_down_managed_exit`) walks the
   1-min path from `DB_1m` and exits at the first of: `take_profit`, `trailing`
   (arms once peak ≥ `*_TRAIL_ACTIVATE`, exits on `*_TRAIL_GAP` giveback), `stop_loss`,
   or `timestop` (`*_MAX_HOLD_MIN`, runs **past** the 1h close — UP 90 min, DOWN 150 min).
   Falls back to `close` (plain 1h-close exit) when `DB_1m` is missing. SHORT profit is
   `pp = 1 − close/entry` (mirror of LONG `close/entry − 1`).
4. **P/L**: `net = gross − (entry_fee + exit_fee)`. The **entry** fee follows MAKER/TAKER
   (below); the **exit** fee is **always taker** (`FEE_PER_SIDE`) — an exit is triggered by
   a condition (`close`/`take_profit`/`trailing`/`stop_loss`) and needs guaranteed
   execution, so it can never be simulated as a passive limit fill. Both `net` and the fees are multiplied by
   **`EXPOSURE = POSITION_FRACTION · LEVERAGE`** (notional = EXPOSURE · equity), then the
   equity is compounded (`np.cumprod`). So leverage amplifies fees too — that's the
   whole point of the trading backtest vs the old WR view.

### Two knobs that trip people up

- **SL is in % of *real capital*, leverage-independent.** `UP_STOP_LOSS` / `DOWN_STOP_LOSS`
  (and the time-decay `*_SL_START/END/FLOOR`) are "% of equity". The engine divides by
  `EXPOSURE` to get the internal **price** stop (`*_STOP_LOSS_DEC`). TP and trailing,
  by contrast, are decimals on the **price**. Don't mix the two axes.
- **Time-decay SL** (`*_SL_TIME_DECAY=True`): the stop tightens linearly from `*_SL_START`
  (wide, at entry) toward `*_SL_END` (tight, at max_hold), floored at `*_SL_FLOOR`
  (`_sl_threshold`). Toggle off → constant `*_STOP_LOSS`.

### MAKER mode (currently ON) — ENTRY only

`MAKER` only ever governs the **entry** leg; the **exit** leg is always taker (see
*The trading engine* above). `MAKER=True` posts a limit at `P0` (the `i+1` open) and
**only fills if price comes back to touch it** within `MAKER_FILL_WINDOW_MIN` (LONG: a
1-min low < P0; SHORT: a 1-min high > P0), using the 1-min highs/lows from `DB_1m`.
Unfilled = a *missed opportunity* (counted, surfaced as a fill-rate in the report) —
this isolates the cost of maker queueing. A filled maker trade has the identical P0 and
exit path as taker; only the entry fee differs (`MAKER_FEE_PER_SIDE` vs `FEE_PER_SIDE`).
`MAKER=False` → entry is also taker at market on the open (both legs then pay
`FEE_PER_SIDE`). `simulate_trades`'s `maker_info` dict carries `entry_fee` / `exit_fee`
separately (was a single `eff_fee`); the report and `trading_runs.csv` show
`entry_fee_pct` / `exit_fee_pct` (was one blended `fee_per_side_pct`).

### The DMI directional filter (post-coherence, `DMI_FILTER_ENABLED`)

After the coherence rule produces `team_bets`, `_apply_dmi_filter` **cancels** any bet
whose side contradicts the Wilder DMI of the **decision** bar (never bar `i+1` → no
lookahead): a DOWN bet is dropped when `+DI > −DI`, an UP bet when `−DI > +DI`. Only the
**sign** of `+DI − −DI` matters — there is **no ADX/trend-strength threshold**. Equality
(`+DI == −DI`), warmup NaNs, or an unknown bar all **keep** the bet.

⚠️ **Naming drift:** the timeframe is `DMI_TIMEFRAME` in `config_main.py` and is currently
**`"4h"`**, but the helpers (`_load_dmi_1h`) and several docstrings still say "1h". Trust
the config value, not the names. The DMI OHLC is rebuilt from modele_down's SQLite
`candles` table (`close` absolute + `*_ret`, so `close_prev = close/(1+close_ret)`) over the
full history; `DMI_LENGTH` (default 7) sets both the DI window and the Wilder smoothing.
The report shows how many UP / DOWN bets the filter removed.

### The UP-confidence filter (post-DMI, `UP_CONF_FILTER_ENABLED`) — the fee-robustness lever

The **second** entry filter, applied by `_apply_conf_filter` right after the DMI stage.
It keeps an **UP** bet only when modele_up's confidence (`confidence` = `P(up)`) is
`≥ UP_CONF_MIN` (default **0.65**). **DOWN bets are never touched** — DOWN's edge is
robust across both window-halves, UP's is not. **Fee campaign 2026-07-01** (fees inside
the sim loop, 0.02%/side): UP's per-trade EV in the *first* window-half is **negative at
every threshold 0.59→0.64** — its whole edge sits in h2. Only at `≥0.65` does UP-h1 turn
positive (≈93 UP trades/yr, the surest setups). Raising the threshold to 0.65 lifts the
**team** from EV `+0.200`/trade (h1 `+0.086`) to `+0.287` (h1 `+0.221`), worst trade
`−7.9%→−5.0%`, net `+331%→+296%` (527 vs 841 trades). `0.62` keeps more volume (675
trades, net +340%) but h1 stays fragile (+0.092) — a volume/robustness trade-off. Set
`UP_CONF_FILTER_ENABLED=False` for the bare behaviour (`n=504`, **not** fee-robust). The
report shows how many UP bets it removed; `trading_runs.csv` carries `conf_filter` /
`conf_min` / `conf_removed_up`.

## Window selection

`compute_window` anchors `end` to the most recent timestamp present in **both**
packages' `logs/test_predictions.csv`, then goes back `TEAM_TEST_DAYS` (in
`config_main.py`). So both sister models must have been trained (`train.py`) first, and
the window never extends past their common test coverage.

## The import trick (siblings live ONE level up)

`main.py` lives in `1h/modele_team/`, but `modele_up` / `modele_down` live in `1h/`
(one level **up**). `main.py` injects that **grandparent** dir
(`Path(__file__).resolve().parent.parent` = `1h/`) onto `sys.path` so the sister
packages import as top-level (`from modele_up import api`). `1h/` starts with a
digit, so it can't be a dotted package — hence the sys.path insert. Every other
path (DB_1m, logs, modele_down's SQLite candles DB, each package's
`logs/test_predictions.csv`) is resolved relative to `__file__` or the imported
module, so the move into `modele_team/` needed only that single sys.path line.

## Logs

All written into `logs/`. Per-bar/per-bet detail files **overwrite** each run;
`trading_runs.csv` is the **append-only** history (one summary row per run, schema-tolerant
— new columns are reindexed in).

- `trades.csv` — one row per simulated trade (`gross_ret`/`net_ret`/`win_net`/`exit_reason`).
- `trading_runs.csv` — append-only run summary (P/L, WR, drawdown, fill-rate, sizing, …).
- `team_all_bars.csv` / `team_skipped.csv` / `team_taken.csv` — thin per-bar logs
  (`detailed_log.LOG_COLUMNS`); `skipped + taken` partition `all`.
- `full_logs/{all,won,lost}_bets.csv` — wide per-bet logs.
- `team_run.log` — run log. (`trailing_trades.csv` is a stale artifact from a removed
  `trailing_backtest.py`; nothing in the current package writes it.)

### The Rich report's EV view

The console report ends with an **"EV par trade"** table (one row per side + a "Tous /
headline" row) and an **EV** figure on the headline panel. `EV = WR·AvgWin − LR·AvgLoss`
where `LR = 1 − WR`, `AvgWin`/`AvgLoss` are **net-of-fees magnitudes ×leverage**. This is
mathematically identical to the **average net % per trade** — it's surfaced decomposed
because the user reasons in WR/AvgWin/AvgLoss terms. `ev_per_trade_pct` (headline) and the
per-side `ev_pct` are persisted into `trading_runs.csv`. EV/trade > 0 is the real
profitability bar (a high WR with a fat AvgLoss can still be EV-negative).

## Run

```powershell
cd backtest/modele_updown/1h/modele_team
python fetch_1m.py    # (once) build DB_1m/btc_1m.parquet — needed for SL/TP/trailing/maker
python main.py        # team backtest -> overwrites logs/ detail + appends trading_runs.csv + Rich report
```

## Team coherence rule

```
up bets UP   + down abstains  -> UP
down bets DN + up abstains    -> DOWN
both bet                       -> CANCEL
neither bet                    -> PASS
```

The surviving `UP` / `DOWN` bets then pass through **two entry filters** — the **DMI
directional filter** then the **UP-confidence filter** (both above) — before simulation,
either of which can cancel still more.
