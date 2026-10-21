# CLAUDE.md — regime

Sibling package to `modele_up/`, `modele_down/`, `modele_team/` under `1h/`. A
market-regime detector (bear / neutral / bull) that modulates `modele_team`'s
**DOWN exit management** by regime. **Active in production** (`REGIME_WRAPPER_ENABLED
= True`) — see *Status* below before touching it.

## Status (2026-07-05): ACTIVE — regime-conditional DOWN exit management, validated

Two independent routing tables, two very different outcomes:

- **`REGIME_OVERRIDES`** (entry-filter routing, `down_conf_delta`/`up_conf_min`) —
  **no-op**. ~100+ configurations tested (regime macro at 4 lookbacks, DMI-alone,
  funding rate [Binance, 6.8y history], AND-gates, DOWN hour-of-day, exit-outcome
  correlation) — none beat the baseline robustly on both halves of the test window.
  Root cause: `DOWN_CONF_DELTA`/`UP_CONF_MIN` are already tuned specifically for
  cross-regime robustness (that's their documented purpose in `config_main.py`) —
  no margin left for a coarser regime split to improve on them. Full evidence:
  `modele_team/logs/regime_backtest_compare.csv` + the long comment above
  `REGIME_OVERRIDES` in `config_regime.py`.

- **`REGIME_EXIT_OVERRIDES`** (exit-parameter routing, DOWN only) — **this is what
  actually works**. Doubling `DOWN_MAX_HOLD_MIN` and widening `DOWN_TRAIL_GAP` by
  1.3x, but **only for DOWN trades entered while regime=bear**, chosen once at
  entry (never revises an open trade — respects the brainstorm decision) beats the
  baseline on every axis simultaneously: net +1050.80%→+1635.24%, drawdown
  -23.44%→-17.59%, worst month -$4,008→-$3,313, validated on both halves of the
  window (h1 +362→+511%, h2 +149→+184%) and robust across 6 of 8 lookback x
  calibration-method combinations (a real plateau, not a spike — see
  `config_regime.py`'s ROUND 3 comment for the full sweep). **Critical control
  check**: applying the same extended hold to ALL DOWN trades (no regime
  targeting) is net-neutral but actively *worsens* WR (58.9%→55.2%) and drawdown
  (-23.4%→-34.0%) — proof the regime detector is doing real, necessary work here,
  not just "holding longer is always better."

Key lessons if you come back to tune this further:
- Entry-side confidence thresholds are already regime-robust by design; don't
  expect a coarse 3-state regime split to beat them again.
- Exit-management parameters (how long to let a winner run) are a much better
  match for what "regime" naturally means than entry selectivity is.
- Always run the "apply to everyone, no regime targeting" control — if it does
  nearly as well as the regime-targeted version, the regime detector isn't adding
  anything and you're just looking at a global mistuned constant.
- Validate h1 AND h2 separately (never trust the aggregate), and check neighboring
  lookback/method values (a real finding is a plateau, a fluke is a spike
  surrounded by disasters — this exact pattern killed 2 earlier false leads,
  see `config_regime.py`'s ROUND 1/2 comments).

## What's here

- `config_regime.py` — every tunable (lookback grid, indicator weights, calibration
  method, hysteresis/confirm/cooldown, `REGIME_OVERRIDES` routing table, stability
  guardrails). Nothing hardcoded elsewhere.
- `scorers.py` — OHLCV-only composite score: trend (cumulative log-return) +
  momentum (shorter sub-window) + trend-strength (Wilder DMI, duplicated from
  `modele_team/main.py::_wilder_dmi` — small helper, not worth sharing across
  packages, same convention as `_bucket_threshold` in the `15min/` tree). Each
  axis normalized via rolling z-score (clipped, adaptive over the ~13.7y history)
  before being combined.
- `calibration.py` — two threshold-derivation methods compared empirically:
  `manual` (fixed) and `calibrated` (percentiles of the full historical score
  distribution). Same asymmetry both ways: bear triggers earlier and has a wider
  hysteresis band (persists longer) than bull.
- `state_machine.py` — hysteresis + confirmation (`CONFIRM_BARS`) + cooldown
  (`COOLDOWN_BARS`) -> confirmed regime per bar. NaN/warmup keeps the current
  regime (defaults to neutral at the very start of history).
- `api.py` — standardized contract (same shape as `modele_up`/`modele_down`):
  `evaluate_regime_full(lookback_days, calibration_method) -> DataFrame[timestamp,
  trend_norm, momentum_norm, dmi_norm, score, regime]`, plus
  `evaluate_regime_on_slice(start_ms, end_ms, ...)` for a future live_team-style
  consumer that only needs a recent window (score/hysteresis still computed on
  full history first, then truncated — never the other way around).

## Data source and validation window

OHLCV only (no funding/CVD/liquidations/long-short ratio in this version — v1
scope decision, see brainstorm). Reads modele_down's SQLite `candles` table
directly (`1h` for the score axes, `REGIME_DMI_TIMEFRAME` for the DMI axis),
covering the full ~13.7-year history (2012-10-24 onward) — **independent of**
`modele_team`'s `TEAM_TEST_DAYS` (365d), deliberately wider so the regime
detector and any future comparison genuinely span multiple real bear/neutral/bull
cycles instead of whatever a single production test window happens to contain.

## The import trick

Same as the rest of `1h/`: `1h/` starts with a digit, so `regime` is imported as
a top-level package once `1h/` is on `sys.path` (done once by `modele_team/main.py`
or `compare_regime_variants.py`). Internal cross-module imports use the
`try: from . import X / except ImportError: import X` pattern (copied from
`modele_down/api.py`) so each file also works standalone (`cd regime && python -c
"import api; print(api.evaluate_regime_full())"`).

## modele_team integration points

All in `modele_team/main.py`, all additive and gated on `cfg_regime.REGIME_WRAPPER_ENABLED`
(currently `True`) with `regime_map=None` as the universal escape hatch (reproduces
the exact pre-regime behavior everywhere it's threaded through — verified byte-identical):

- `_load_regime_map()` — `timestamp -> regime` dict, only called if
  `cfg_regime.REGIME_WRAPPER_ENABLED`.
- `_effective_down_delta` / `_effective_up_conf_min` — per-bar effective
  `DOWN_CONF_DELTA` / `UP_CONF_MIN` (entry filters, currently no-op — reads
  `REGIME_OVERRIDES`).
- `_effective_down_exit_params(ts, regime_map)` — per-trade `(max_steps, trail_gap)`
  for `_down_managed_exit` (exit management, **the mechanism that actually works** —
  reads `REGIME_EXIT_OVERRIDES`). Chosen once from the regime at the *decision* bar
  `ts`, never revisited — an already-open trade always runs to completion under
  the parameters it started with.
- `_prepare_predictions()` / `run_backtest(prepared, regime_map=None)` — the
  refactor that made this possible: predictions (expensive, regime-independent)
  split from filters+simulation (cheap, regime-conditional). `main()` is now a
  thin wrapper: `_prepare_predictions()` -> optional `_load_regime_map()` ->
  `run_backtest()` -> file writes + report, byte-identical to the pre-refactor
  `main()` when `regime_map=None`.

## Re-testing after a change (new features, retrained models, etc.)

```powershell
cd backtest/modele_updown/1h/modele_team
python compare_regime_variants.py   # re-runs the ENTRY-filter profile x method x lookback grid, h1/h2-validated
```

`compare_regime_variants.py` only covers `REGIME_OVERRIDES` (entry filters) — it
predates the `REGIME_EXIT_OVERRIDES` discovery and was not extended to sweep exit
parameters (that sweep was done ad hoc, see `config_regime.py`'s ROUND 3 comment
for the exact grid: hold multiplier x1.0-x4.0, trail-gap multiplier x1.0-x2.0,
crossed with regime lookback/method). If you retune `REGIME_EXIT_OVERRIDES`, redo
that sweep (not just `compare_regime_variants.py`) and re-run the two control
checks: (1) apply to ALL trades regardless of regime — if that does nearly as
well, the regime split isn't adding anything; (2) check neighboring lookback/method
values — a real finding is a plateau, not an isolated spike.

Add new entry-filter hypotheses to `OVERRIDE_PROFILES` in that file rather than
guessing directly in `config_regime.py`'s `REGIME_OVERRIDES` — only promote a
profile to `REGIME_OVERRIDES` once it clears all
three bars: beats baseline net% within the stability envelope, on **both**
halves of the window, **and** isn't a lone spike vs its neighboring lookbacks.
