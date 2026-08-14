# OKX live daemon for `1h/modele_team` — design

Date: 2026-07-09
Status: approved (brainstorming), pending implementation plan

## Context

`backtest/modele_updown/1h/modele_team/` runs a **trading-P/L backtest** (`main.py` +
`config_main.py`): two sister XGBoost models (`modele_up`, `modele_down`) predict,
a team-coherence rule combines them, then a stack of entry filters (DMI, UP
confidence/hour/cross-model, concurrency cap) and a regime-conditional exit wrapper
(`regime/`) decide what survives, before `simulate_trades` walks 1-minute Bitstamp
data to work out the real P/L of each simulated trade.

A live counterpart for this exact 1h team system existed once (`1h/live_team/`,
MEXC futures, designed 2026-06-18 — see `2026-06-18-mexc-live-team-1h-design.md`)
but is no longer present in the working tree, and — more importantly — predates
several features the backtest has since grown (DMI filter, the 3-filter UP stack,
the concurrency cap, the regime wrapper, real-fee-rate lookup). It is not a valid
base to copy from as-is.

This spec defines a **new** live daemon, in a `live/` subfolder **inside**
`modele_team/` (not a sibling package), trading on **OKX** instead of MEXC, that
stays in lockstep with the current (and future) backtest logic by construction
rather than by manual sync.

## Goals

- Real-money (or dry-run) execution on OKX of exactly what `main.py` would have
  simulated for the same bar — no behavioural drift between backtest and live.
- A single source of truth for every tunable (`config_main.py`) and every
  filter/threshold decision (new `filters.py`, extracted from `main.py`) — both
  `main.py` and `live/live.py` import them; nothing is duplicated or hand-recopied.
- Fee assumptions used by the backtest reflect the exchange actually traded on
  (OKX), not a different one (MEXC).

## Non-goals

- `live/` is **not** self-contained/independently deployable the way the
  repo-root `live/` or the old `live_team/` were. It depends on `modele_team/`
  and, transitively, on `1h/modele_up`, `1h/modele_down`, `1h/regime` via
  imports — this is an explicit trade-off in exchange for guaranteed symmetry
  (user decision).
- No support for exchanges other than OKX in this pass.
- No UI/dashboard for the live daemon (the existing `g.py` dashboard reads
  backtest `logs/trades.csv` only; a live equivalent is out of scope here).

## Architecture

### Shared-logic extraction (`modele_team/filters.py`, new)

Move out of `main.py` into an importable module, unchanged in behaviour:
`_wilder_dmi`, `_load_dmi_1h` (or an OKX/Bitstamp-source-agnostic variant),
`_apply_dmi_filter`, `_apply_conf_filter`, the UP cross-model filter logic,
`_effective_down_delta`, `_effective_up_conf_min`, `_effective_down_exit_params`,
`_load_regime_map`, `_sl_threshold`, and the sidecar/threshold-bucket lookup used
by both models' `api.py` if it needs to be called live (confirm at implementation
time whether live re-uses `modele_up.api`/`modele_down.api` directly for
prediction, which is the default assumption — see *Prediction* below).

`main.py` imports these from `filters.py` instead of defining them inline;
`live/live.py` imports the exact same functions. This is the mechanism that
makes "backtest and live are two copies of the same logic" true by construction.

### `okx_fees.py` (replaces `mexc_fees.py`)

Same contract as today's `mexc_fees.py`: one signed, read-only GET against OKX's
fee-rate endpoint, keyed off `OKX_API_KEY`/`OKX_API_SECRET`/`OKX_API_PASSPHRASE`
in the process environment. Returns `{"maker": float, "taker": float}` or `None`
on any failure (absent keys, network error, bad response) — never raises.
`main.py`'s fallback-and-print behavior is unchanged: if `None`, fall back to
`config_main.MAKER_FEE_PER_SIDE`/`FEE_PER_SIDE` and print which source was used,
never silently.

### `modele_team/live/` contents

```
live/
  live.py            1-minute tick loop: bootstrap/reconcile, per-minute position
                      monitoring, top-of-hour prediction+entry decision, logging
  api_okx.py          SOLE file touching the OKX API: signed REST (orders,
                      positions, klines, balance, fee rate). SWAP instrument,
                      HEDGE position mode, ISOLATED margin, leverage set at boot.
  calcule_feature.py  Bitstamp OHLC fetch (mirrors modele_team's data conventions)
                      + feature computation for both models' inputs
  position.py         PositionManager: reads live PnL% per open position directly
                      from OKX's position endpoint, compares against SL/TP/
                      trailing/timestop thresholds sourced from config_main.py
                      (price-based thresholds already expressed compatibly with
                      OKX's leveraged PnL% reporting), executes exits
  notify.py           Telegram alerts (open/close/error/stuck) — same pattern as
                      other daemons in this repo (live_helpers/notify.py)
  config_live.py      Live-only constants: OKX_INST_ID, OKX_LEVERAGE (must match
                      config_main.LEVERAGE), TICK_INTERVAL_S=60, log paths, and
                      the expected env var names. Does NOT redefine anything
                      already in config_main.py.
  .env                Real file (gitignored), not a .env.example — filled in with
                      actual OKX_API_KEY / OKX_API_SECRET / OKX_API_PASSPHRASE,
                      DRY_RUN, Telegram bot token/chat id.
  Dockerfile
  docker-compose.yml   build context = modele_team/ (needs main.py's siblings:
                       filters.py, config_main.py, modele_up/, modele_down/,
                       regime/ — all one level up from modele_team, two from live/)
  logs/
    cycles.csv          one row per minute/hour tick
    positions.csv        one row per position (open->close lifecycle)
    positions_state.json persisted open-position state, read at boot to
                          reconcile after a restart/crash
```

## Live cycle

**Every minute** (`live.py` tick):
1. `PositionManager.check_positions()` — for each open position, `api_okx.py`
   reads live PnL% from OKX's `/positions` endpoint. Compare against, in order:
   `take_profit`, `trailing` (armed once peak PnL% crosses `*_TRAIL_ACTIVATE`,
   exits on `*_TRAIL_GAP` giveback), `stop_loss` (constant or time-decay per
   side, per `config_main.py`), `timestop` (`*_MAX_HOLD_MIN` elapsed since
   entry). First condition hit → reduce-only market exit (always taker, same
   convention as the backtest).

**At the top of every hour (`:00`)**:
2. Fetch fresh Bitstamp OHLCV on the same timeframe ladder the models were
   trained on; `calcule_feature.py` builds both models' feature vectors.
3. Predict with `modele_up`/`modele_down` (reusing their existing `api.py` /
   model-loading code — exact same artifacts as the backtest).
4. Team coherence rule → `filters.py` (DMI, UP confidence, UP hour-of-day, UP
   cross-model, regime-conditional DOWN exit params) → concurrency cap check
   (`MAX_CONCURRENT_POSITIONS=2`, tracked from `positions_state.json`'s
   currently-open count, both sides combined).
5. If a bet survives and a slot is free: `api_okx.py` posts the entry. If
   `MAKER=True` (config_main.py): post-only limit at the bar's open price,
   re-price/cancel if unfilled within `MAKER_FILL_WINDOW_MIN`, otherwise take
   at market. Entry fee follows maker/taker actually achieved; exit fee is
   always taker.

**At boot**: reconcile — load `positions_state.json`, cross-check against
OKX's live `/positions` to catch anything opened/closed while the daemon was
down, resume monitoring.

**`DRY_RUN`** (env var, `.env`): `true` → the full cycle runs (fetch, predict,
filter, decide, log) but `api_okx.py` makes no write calls (no order posted, no
leverage/margin-mode changes); `false` → real trading.

## Data sources

- **Prediction/features**: Bitstamp — identical to training, zero train/live skew.
- **Execution & position monitoring**: OKX — real fill prices, real leveraged
  PnL%, real fees. This is the authoritative source for anything that determines
  actual money (entry/exit decisions on open positions), matching the old
  MEXC live_team's split and avoiding a skew between simulated-exit price and
  actual OKX execution price.
- **Fees**: OKX, via `okx_fees.py` (shared with the backtest — see above).

## Config & credentials

- `live/.env` (gitignored, real file): `OKX_API_KEY`, `OKX_API_SECRET`,
  `OKX_API_PASSPHRASE`, `DRY_RUN` (`true`/`false`), `TELEGRAM_BOT_TOKEN`,
  `TELEGRAM_CHAT_ID`, optionally `TELEGRAM_ERROR_BOT_TOKEN`/`_CHAT_ID` (repo
  convention for a separate error-alert bot).
- `live/config_live.py`: OKX instrument id, leverage (must equal
  `config_main.LEVERAGE`), tick interval, log paths. No trading thresholds —
  those live exclusively in `config_main.py` and are imported, not copied.

## Error handling & notifications

Telegram (`live/notify.py`, same pattern as existing daemons): position
opened, position closed (win/loss/stuck), and every `logger.error`/
`logger.exception` routed to a distinct error bot/chat if configured. A
position that can't be reconciled or exited automatically triggers a "stuck"
alert requiring manual intervention (mirrors `send_stuck` in the repo-root
`live/`).

## Packaging

Dockerfile + `docker-compose.yml` inside `live/`, build context = `modele_team/`
(the parent folder) so the image can `COPY` `main.py`, `filters.py`,
`config_main.py`, and reach `modele_up/`, `modele_down/`, `regime/` one level up
again. `docker compose up -d`, logs mounted to `./logs`, `.env` read for
credentials — same operational shape as the repo's other live daemons, minus
the self-containment.

## Open questions for the implementation plan

- Exact shape of the PnL%-to-price-threshold conversion when reading OKX's
  reported position PnL% (which already includes leverage) versus
  `config_main.py`'s price-based `*_TAKE_PROFIT`/`*_TRAIL_ACTIVATE`/
  `*_TRAIL_GAP` decimals — needs `× EXPOSURE` consistently, confirm against
  OKX's exact PnL% definition (margin-based vs notional-based) before wiring.
- Whether `modele_up`/`modele_down`'s existing `api.py` prediction path can be
  called bar-by-bar live as-is, or needs a thin live-prediction wrapper
  (single-row inference vs its current batch/slice-oriented interface).
- Confirm OKX's SWAP fee-rate endpoint path/response shape for `okx_fees.py`
  (mirrors `mexc_fees.py`'s `tiered_fee_rate/v2`, but OKX's equivalent needs to
  be looked up against current API docs at implementation time).
