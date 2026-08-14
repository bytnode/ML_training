# CLAUDE.md — live (OKX daemon, 1h team)

Live trading daemon for `modele_team`'s 1h up+down team system, on **OKX**
(`BTC-USDT-SWAP`, HEDGE mode, ISOLATED margin). Physically sits **inside**
`modele_team/` (not a sibling package) but is **fully self-contained as of
2026-08-03**: it imports nothing from `modele_team/` or `1h/` — every module it
needs (`config_main.py`, `filters.py`, `modele_up/`, `modele_down/`, `regime/`)
is an embedded copy inside `live/` itself. Copy this folder alone to a VPS and
it runs. Design spec: `docs/superpowers/specs/2026-07-09-okx-live-team-1h-design.md`.

**Full self-containment (2026-08-03, 3rd campaign)**: `live/` now carries its
**own embedded copies of every dependency**, code only — `modele_up/`,
`modele_down/`, `regime/` (`api.py`, `config.py`, `data.py`, `add_*.py`, the
model `.json`/`.thresholds.json`, `regime_seed.json`, since 2026-07-10), **plus
`config_main.py` and `filters.py`** (since 2026-08-03, previously imported live
from `modele_team/` for automatic backtest/live parity). Never the heavy
`DB/`/`logs/`/training-only scripts. **Tradeoff, explicitly accepted (same one
already lived with for `modele_up|down/regime/`, now extended to
`config_main.py`/`filters.py` too)**: after a retrain, or any edit to
`modele_up`/`modele_down`/`regime`/`config_main.py`/`filters.py`'s *backtest*
source, you must manually re-copy the changed files into `live/` — nothing does
this automatically, and a stale copy fails silently (same tradeoff the
repo-root `15min/live/` already lives with). **This is a real behavior change**:
before 2026-08-03, editing `modele_team/config_main.py`/`filters.py` changed
backtest AND live together automatically (single shared file) — now it only
changes the backtest until you copy the file into `live/` by hand.
`live/filters.py` (this copy) still has **no sys.path bootstrap of its own** —
it resolves `modele_down`/`regime`/`config_main` to whatever the *caller*
already put on `sys.path` (`live.py`/`market_data.py`/`position.py` all push
`live/` itself before `import filters`). Do not add a self-sufficient
`sys.path.insert` to `filters.py` — Python caches resolved package names in
`sys.modules`, so whichever copy resolves *first* wins for the rest of that
process.

**Data autonomy (2026-07-10, 1st campaign)**: `live/` never touches modele_down's
SQLite — not at startup, not ever. All market data comes from Bitstamp, fetched
once at launch and maintained incrementally (`market_data.py`); the regime
detector bootstraps from `regime/regime_seed.json` (now `live/regime/regime_seed.json`,
the embedded copy) and advances one bar at a time (`regime_live.py`),
persisting its own state inside `live/logs/`. See *Files* below.

**Bug-fix campaign (2026-07-15)**: a full audit (5 parallel independent reviews
across every file) after DRY_RUN was observed never tracking or closing
positions in prod. Root cause: `check_positions`/`try_widen` called the
ALWAYS-real `get_positions()` to validate DRY_RUN paper state — which OKX can
never show (writes are stubbed) — causing an immediate false "desync" that
wiped tracked state within a minute of every entry. Fixed by making
`position.py` DRY_RUN-aware: paper positions are never checked against
`get_positions()`, price comes from `get_ticker()` instead. The same audit
found and fixed 9 more issues:
- `config_live.is_dry_run()` didn't `.strip()` the env var — a stray trailing
  space/CRLF in `.env` silently flipped paper mode to **real trading**, no
  error, no log. Highest-severity finding of the campaign.
- `main()`'s `md.bootstrap()` was unguarded — a Bitstamp/regime-seed failure at
  startup killed the process with **zero Telegram alert** (crash before any
  logger call runs), crash-looping silently under Docker and leaving an
  already-open position unmonitored. Now wrapped: alerts via
  `notify.send_cycle_error`, then re-raises (still exits — Docker's restart
  policy is the retry mechanism — but now visibly).
- `buf_btc_4h` (feeds the per-trade DMI filter) had **zero gap protection** —
  unlike the 1h/eth buffers, a Bitstamp-omitted candle here silently corrupts
  `filters._wilder_dmi` with no crash and no alert. New
  `calcule_feature.heal_gaps()` scans a buffer for any internal gap (not just
  at one queried timestamp, cf. `backfill_gap_if_needed`) and backfills it —
  applied to all 3 buffers.
- `get_order()` in DRY_RUN always answered `"live"`, never `"filled"` — every
  paper maker entry silently fell back to taker after the fill window
  regardless of price (100% `maker_fallback_taker` in the observed logs).
  `advance_pending_entry` now simulates a realistic fill in DRY_RUN by
  comparing the current ticker to the posted price.
- Taker entries (direct and maker-fallback) recorded the pre-trade ticker
  snapshot as `avg_px`, never the real fill price — real-mode slippage was
  invisible to every downstream SL/TP/trailing check. Now polls `get_order()`
  once after a taker fill (best-effort, falls back to the snapshot).
- The regime's hourly `bootstrap_and_advance` re-fetched ~726 1h-bars + ~142
  4h-bars from Bitstamp **every single hour**, even for the routine
  single-bar advance, instead of reusing `market_data`'s buffers already in
  RAM — contradicted this file's own "never a full re-fetch" principle. Now
  serves from the in-memory buffer when it has enough history, only falls
  back to a real fetch after a long outage.
- `api_okx.OkxClient._request()` called `raise_for_status()` before reading the
  response body — OKX's own `sCode`/`sMsg` diagnostic (e.g. distinguishing a
  bad key from an unwhitelisted IP) was discarded, leaving only a bare
  `401 Client Error`. Now logs the body before re-raising.
- `live.py`'s "widened" `positions.csv` row was missing the `peak_pp`/
  `worst_pp`/`pnl_usdt` columns added by an earlier session — a real
  regression that silently misaligned that row under the fixed CSV header.
- `compute_size_contracts` floored to `min_sz` even when `notional_usd <= 0`
  (zero/negative balance, misconfigured `POSITION_FRACTION`) — would place a
  real minimum-size order instead of skipping the trade; now raises instead.
  Its `sz` formatting (`:g`) could also emit scientific notation for a very
  small `contracts` value — switched to fixed-decimal.
- `entry.get("notional_usd", 0.0)` / `entry.get("sz", "0")` only substitute the
  default when the key is *absent*, not when explicitly `null` — a
  hand-edited/corrupted state file could `TypeError` mid-tick. Switched to
  `entry.get(x) or default`.

**Known, accepted limitations** (not fixable without a bigger redesign):
DRY_RUN's `mark_px` comes from `get_ticker()`'s last-trade price, while REAL
mode uses OKX's `markPx` (index-derived, smoother) — the two legitimately
diverge during volatility, so DRY_RUN P&L/exit timing is a close but not
bit-identical preview of REAL mode. Separately, `run_hourly_cycle` runs
synchronously in the same single thread as `check_positions`/
`advance_pending_entry` — a slow Bitstamp fetch (`freshness_retry` can block up
to ~60s per pair) delays position monitoring for that span. Both are
structural to the current single-threaded design, not bugs to patch in place.

**Symmetry principle**: every decision (coherence rule, DMI filter, the 3
UP-only filters, sidecar threshold lookup, regime wrapper) runs through
`filters.py` and `modele_up.api`/`modele_down.api` — the same logic `main.py`
uses for the backtest, kept identical by **copying**, not by importing a
shared file (since the 2026-08-03 autonomy campaign, see *Full
self-containment* above). If you need to change a threshold or a rule, change
it in `modele_team/config_main.py`/`filters.py` (the backtest originals) **and
then manually copy the changed file(s) into `live/`** — editing only one side
no longer updates the other.

## Files

- `live.py` — the loop. Every minute: `PositionManager.check_positions` (SL/TP/
  trailing/timestop) + `advance_pending_entry` (non-blocking maker-entry state
  machine). At `:00`: fetch Bitstamp → `calcule_feature` → predict → team rule
  → `filters.py` → `PositionManager.start_entry`.
- `api_okx.py` — **the only file that touches the OKX API**. Signed REST
  (HMAC-SHA256, OKX v5 headers). `DRY_RUN=true` stubs every write call (order/
  cancel/amend/leverage/position-mode) and logs what it *would* have done;
  reads (`get_positions`/`get_balance`/`get_ticker`/`get_order`/`get_instrument`)
  always hit the real API, even in dry-run.
- `calcule_feature.py` — Bitstamp OHLC fetch + the 6 features of each model,
  reusing the training packages' own helpers (`add_engineered_features.py`,
  `add_alpha_features.py`, `add_eth_features.py`, `data.py`) so there is zero
  train/serve skew. **`modele_up/` and `modele_down/` both have same-named
  modules** (`data.py`, `config.py`, `add_engineered_features.py`,
  `add_eth_features.py`) — `_import_isolated` purges `sys.modules` before/after
  each package's import so one package can never silently borrow the other's
  code. Also computes `volatility_state`/`is_weekend` (the sidecar threshold
  bucket keys — not model features) with the same ATR/rolling formula as
  `data.py`. ETH input is **native 1h** (matches the current `add_eth_features.py`
  — no 15m→1h resample; the old, now-deleted `live_team` package expected a
  resample step that no longer exists).
- `position.py` — `PositionManager`. Persists open-position state (peak/floor
  for trailing, pending maker-entry) to `logs/positions_state.json` — the
  backtest's `_up_managed_exit`/`_down_managed_exit` see the whole future in one
  loop; live only ever has the current tick, so trailing state has to survive
  between calls (and a restart). `evaluate_exit` computes `pp` from OKX's
  `avgPx`/`markPx` directly (**not** `uplRatio`, whose exact sign/definition
  isn't worth trusting) — same formula as the backtest (`cl/entry - 1` LONG,
  `1 - cl/entry` SHORT), compared against the exact same `config_main.py`
  thresholds via `filters.sl_threshold`.
  **Exchange constraint**: OKX HEDGE mode shows one merged position per
  `(instId, posSide)` — a 2nd same-side entry would blend `avgPx` and corrupt
  the older entry's SL/TP/trailing tracking. So this caps at **1 slot per side
  (long, short) = 2 total**. As of 2026-07-10 the backtest enforces the exact
  same cap (`MAX_CONCURRENT_POSITIONS_PER_SIDE=1` in `config_main.py`) — this
  is no longer a live-only deviation, both systems agree.
  **`try_widen(pos_side)`** (added 2026-07-10, symmetry with `main.py`'s
  widen-on-profit): called from `live.py` right before a same-side signal
  would otherwise be discarded as "slot occupied" — if the already-open
  position of that side is currently profitable (`pp > 0`, same
  `avgPx`/`markPx` formula) and hasn't been widened yet, sets `entry["widened"]
  = True` **once** (never cumulative). `evaluate_exit` then multiplies
  take-profit/trailing-gap/max-hold by `UP_WIDEN_*`/`DOWN_WIDEN_*`
  (`config_main.py`) — the SL horizon is **never** widened, same as the
  backtest. ⚠️ Those multipliers were swept on a sample of only 7 UP / 3 DOWN
  widened *backtest* positions — provisional, not a validated final answer.
- `market_data.py` — `LiveMarketData`: owns the 3 in-memory Bitstamp buffers
  (BTC 1h — 800 bars, covers both model features and the regime lookback;
  BTC 4h — 300 bars, shared by the per-trade DMI filter and the regime's own
  DMI axis; ETH 1h — 400 bars). `.bootstrap()` fetches all three once at
  startup; `.refresh(target_ts)` merges just the newest few bars each hourly
  cycle (never a full re-fetch), confirms freshness (same `freshness_retry`
  guarantee as before), then advances the regime by one bar. `.dmi_blocks()`
  computes the per-trade DMI filter from the local 4h buffer via
  `filters._wilder_dmi`/`filters.dmi_blocks_direction` **unchanged** — only the
  data source moved (Bitstamp buffer instead of SQLite), never the formula.
  `.current_regime_map()` returns a tiny duck-typed `.get(ts, default)` object
  that always answers the current regime regardless of `ts` — regime changes at
  most once/hour (`confirm_bars`/`cooldown_bars`), so a literal single-key dict
  would break the lookup the moment `advance_pending_entry` (called every
  minute) asks for a slightly different timestamp than the current cycle's.
- `regime_live.py` — incremental regime computation, **reconstructs** what the
  old `live_team/regime_live.py` did (that file was never committed and is
  gone — lost to a working-tree cleanup mid-session, not recoverable via git;
  this is a fresh implementation built from the same documented contract).
  Reuses `regime/state_machine.py::apply_state_machine` **directly, unmodified**
  (it already natively supports resuming from `initial_state=`) and duplicates
  `regime/scorers.py`'s formulas (trend/momentum/DMI raw values, rolling
  z-score) — same convention as `filters._wilder_dmi` elsewhere in this repo.
  `load_or_init_state` resumes from `logs/regime_state.json` if present, else
  bootstraps from `regime/regime_seed.json` (the rolling 730-day tail of raw
  axis values + the hysteresis state machine's terminal state — regenerate via
  `cd 1h/regime && python export_seed.py`, needed every 3-6 months since the
  seed's calibrated thresholds barely drift). `bootstrap_and_advance` replays
  any gap between the loaded state and now, one hour at a time, from
  `market_data`'s buffers — same function handles the very first launch (large
  gap) and a routine hourly tick (gap of exactly one bar) uniformly.
  **Verified 2026-07-10**: both the raw-value computation and the incremental
  normalization+state-machine advancement were cross-checked against
  `regime/scorers.py`'s vectorized computation on identical historical data —
  0 mismatches over 500 bars.
- `notify.py` — Telegram (operational bot + optional separate error bot via
  `TelegramErrorHandler`, attached to the root logger — any `logger.error`/
  `logger.exception` anywhere in `live/` reaches it automatically).
- `config_live.py` — live-only constants (OKX instrument id, tick cadence, log
  paths). Never redefines a threshold that already lives in `config_main.py`.
- `.env` (gitignored, real file — not `.env.example`) — `OKX_API_KEY`/
  `OKX_API_SECRET`/`OKX_API_PASSPHRASE`, `DRY_RUN` (`true`/`false`), Telegram
  tokens. `DRY_RUN=true` by default. **`OKX_LEVERAGE`/`OKX_POSITION_FRACTION`**
  (2026-07-15, explicit operator request — `.env` doubles as the config file
  for money-sensitive knobs, not just credentials): real overridable values in
  `config_live.py` (`_env_int`/`_env_float`, never raise on malformed input,
  log + fall back instead). **Deliberately breaks the symmetry principle**
  above when set — empty/absent (the default) still falls back to exactly
  `config_main.LEVERAGE`/`POSITION_FRACTION`, so live only diverges from what
  the backtest validated if the operator explicitly fills these in.
- `Dockerfile` + `docker-compose.yml` — build context is **`live/` itself**
  (`context: .`, changed 2026-08-03, 3rd campaign — see *Full self-containment*
  above; was `modele_team/` via `context: ..` before). A single `COPY . /app/`
  brings in everything — `config_main.py`, `filters.py`, `modele_up/`,
  `modele_down/`, `regime/`, no `COPY` reaching outside `live/` at all anymore.
  `live/.dockerignore` (own file now, replaces the old `modele_team/.dockerignore`
  which only applied when the context was `modele_team/`) keeps `__pycache__/`/
  `logs/`/`.env` out of the image. **No SQLite volume mount** (removed
  2026-07-10 — nothing in `live/` reads it anymore, see *Data autonomy* above).
  `okx_fees.py` was never actually imported by anything in `live/` (grep-confirmed
  2026-08-03) — not embedded, it's backtest-only (`main.py`'s fee lookup).
- `logs/` — `cycles.csv` (1 row/hour: predictions, thresholds, decision, which
  filter blocked it if any), `positions.csv` (1 row/position event),
  `positions_state.json` (persisted open-position + pending-entry state),
  `regime_state.json` (persisted regime rolling-window + hysteresis state,
  cf `regime_live.py`), `live.log`.

## Run

```powershell
cd backtest/modele_updown/1h/modele_team/live
# fill live/.env first (OKX keys, Telegram); DRY_RUN=true by default
python live.py
```

Docker:

```powershell
cd backtest/modele_updown/1h/modele_team/live
docker compose build
docker compose up -d
docker compose logs -f live-okx
```
