"""Team trading daemon on the real Bitstamp feed.

Two modes, switched via the LIVE_MODE env var in .env :

  LIVE_MODE=false (or absent)  -> paper trading. No Polymarket calls. No deps on
                                  py-clob-client / web3 / eth-account. This is
                                  the default and was the original behavior.

  LIVE_MODE=true               -> real-money mode. The same model/decision flow
                                  runs, AND every UP/DOWN team decision triggers
                                  a real FAK BUY on the matching Polymarket
                                  15min market. Winning bets are auto-redeemed
                                  on-chain immediately after resolution. Pending
                                  bets are reconciled at startup and at the top
                                  of every cycle. Requires Polymarket env vars.

Every 15min UTC boundary:
  0. (LIVE_MODE=true only) Reconcile any pending real bets in bets.csv
     (Polymarket first, Bitstamp fallback after 1 cycle, stuck after 6).
  1. Resolve the paper prediction made last cycle.
  2. Fetch fresh OHLCV (15m + 5m + 30m + 1h).
  3. Build features for the bar that just closed.
  4. Predict next bar's direction with both models.
  5. Apply team coherence rule (UP / DOWN / PASS / CANCEL).
  6. Log + display Rich panel.
  7. (LIVE_MODE=true only) If team is UP or DOWN, place a real FAK BUY on
     Polymarket and append one row to bets.csv.

CLI:
  python live.py                # mode from LIVE_MODE in .env
  python live.py --dry-run      # if LIVE_MODE=true: skip the POST /order step

This folder (live/modele_15m/) is self-contained: it carries its own copies of
the model artifacts (modele_up/ + modele_down/) and needs nothing from
backtest/. After a retrain, copy the fresh btc_updown_model.json +
btc_updown_model.thresholds.json from backtest/modele_updown/15min/modele_*/
into the modele_*/ folders here.

cd live/modele_15m && python live.py
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, Future
from datetime import datetime, timezone
from pathlib import Path

# Force UTF-8 stdout so Rich's box-drawing renders on Windows cp1252 terminals.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

import numpy as np
import pandas as pd
import xgboost as xgb
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from live_helpers import fetch
from live_helpers import features_up as feat_up_mod
from live_helpers import features_down as feat_dn_mod
from live_helpers import notify

ROOT = Path(__file__).resolve().parent       # live/modele_15m/ - self-contained
LOGS_DIR = ROOT / "logs"
LIVE_CSV = LOGS_DIR / "live_predictions.csv"
BETS_CSV = LOGS_DIR / "bets.csv"

# Serializes every read-modify-write or append to bets.csv so the main thread
# (place_real_bet -> _append_bet_row) and the background resolve thread
# (resolve_pending_bets -> _save_bets) never clobber each other.
_BETS_CSV_LOCK = threading.Lock()

# Model artifacts live in this folder's own modele_up/ + modele_down/ subdirs
# (copies, kept independent of backtest/).
UP_MODEL_PATH = ROOT / "modele_up" / "btc_updown_model.json"
UP_SIDECAR    = ROOT / "modele_up" / "btc_updown_model.thresholds.json"
DN_MODEL_PATH = ROOT / "modele_down" / "btc_updown_model.json"
DN_SIDECAR    = ROOT / "modele_down" / "btc_updown_model.thresholds.json"

PAIR = "btcusd"
ETH_PAIR = "ethusd"   # exogenous ETH/USD feed for modele_down's 14 ef_* features
WARMUP_15M = 3200   # >= 2880 (volume_rel_30d) + margin
WARMUP_5M  = 200
WARMUP_30M = 200
WARMUP_1H  = 200
WARMUP_ETH_15M = 1000  # ef_ret96 + rolling-96 (ef_vol_z) + EMA convergence margin

FRESHNESS_RETRY_MAX = 12
FRESHNESS_RETRY_DELAY_S = 1
INTERVAL_15M_MS = 15 * 60 * 1000

# Per-cycle incremental refresh size : only the last N bars per TF are fetched
# each cycle ; merged into the cached buffers. Enough to cover the gap
# (typically just the 1-2 newest bars) + safety margin against any out-of-order
# delivery from Bitstamp.
INCREMENTAL_REFRESH_N_BARS = 20

# Module-level cached buffers, shared between cycles. Cold start fills the full
# warmup ; subsequent cycles only refresh the tail in parallel. Keyed by
# (pair, tf). See refresh_buffers().
_CACHED_BUFFERS: dict[tuple[str, str], list[list]] = {}

# Background thread pool for I/O fire-and-forget (Telegram notifs, pre-fetch
# Polymarket calls). One small pool shared across the daemon's lifetime.
_BG_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="live-bg")

# Defaults (env vars override; see polymarket.py.from_env).
DEFAULT_BET_FRACTION = 0.10
DEFAULT_BET_MIN_USDC = 2.50
DEFAULT_BET_MAX_FRACTION = 0.95
SETTLEMENT_STUCK_AFTER_CYCLES = 6   # 6 * 15min = 90min past close before we mark "stuck"
MATIC_LOW_THRESHOLD = 0.5

# Schema for bets.csv - one row per bet (every UP/DOWN team decision), in BOTH
# paper and real mode. Centralized so the writer + the catch-up logic agree.
# The first 23 columns are the original real-order schema (kept in order so the
# tested real-money path is untouched); the last 11 are appended context +
# resolution detail so the file is self-explanatory when you open it.
BETS_COLUMNS = [
    # --- original 23 (real-order lifecycle) ---
    "bet_id", "cycle_at",
    "target_open_ms", "target_close_ms",
    "direction", "confidence",
    "polymarket_slug", "polymarket_condition_id", "polymarket_token_id",
    "bet_size_usdc", "polymarket_order_id", "order_status",
    "fill_price_avg", "shares_filled", "usdc_spent", "fee_usdc",
    "outcome", "outcome_source", "outcome_resolved_at_iso",
    "realized_pnl_usdc", "redeem_tx_hash",
    "wallet_usdc_pre", "wallet_usdc_post",
    # --- appended 2026-05-20: comprehensive per-bet context ---
    "mode",                    # paper | real | dry_run
    "target_window_utc",       # human-readable target window, e.g. "13:30-13:45 UTC"
    "threshold",               # the bucket threshold the confidence had to beat
    "edge",                    # confidence - threshold
    "p_up", "p_dn",            # both models' probabilities this cycle
    "volatility_state", "is_weekend", "hour",   # the threshold bucket
    "resolved_actual",         # up | down | flat
    "resolved_open_price", "resolved_close_price", "resolved_pct_move",
]


def _parse_live_mode(raw: str | None) -> bool:
    """Tolerant parse: 'true' / 'True' / '1' / 'yes' -> True; everything else -> False.
    Absent or unparseable defaults to False (paper mode = safer default)."""
    if raw is None:
        return False
    return raw.strip().lower() in ("true", "1", "yes", "y", "on")


# ============================================================
#  Model + sidecar load
# ============================================================
def _load_model(path: Path) -> xgb.XGBClassifier:
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    m = xgb.XGBClassifier()
    m.load_model(str(path))
    return m


def _load_sidecar(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Sidecar not found: {path}")
    with open(path) as f:
        return json.load(f)


def _bucket_threshold(volatility_state: int, is_weekend: int, sidecar: dict, ts_ms: int) -> float:
    """Reproduces predict.decide()'s threshold lookup (V1 + V2 hour_boost
    + optional V2 bad_hours)."""
    strategy = sidecar.get("strategy")
    if strategy not in ("asym_wknd_excl", "asym_wknd_excl_hour_boost"):
        raise ValueError(f"Unknown sidecar strategy: {strategy!r}")
    skip = {(int(v), int(w)) for v, w in sidecar.get("skip_cells", [])}
    if (int(volatility_state), int(is_weekend)) in skip:
        return 1.0
    hour = int(((ts_ms // (15 * 60 * 1000)) % 96) // 4)
    # Bad-hours exclusion (V2 extension) : force T=1.0 = never bet.
    bad_hours = set(int(h) for h in sidecar.get("bad_hours", []))
    if bad_hours and hour in bad_hours:
        return 1.0
    T = float(sidecar.get("thresholds_per_wknd", {}).get(str(int(is_weekend)), 1.0))
    if strategy == "asym_wknd_excl_hour_boost":
        hb = sidecar.get("hour_boost", {})
        good = set(int(h) for h in hb.get("good_hours", []))
        delta = float(hb.get("delta", 0.0))
        if hour in good:
            T -= delta
    return T


def _predict_one(model: xgb.XGBClassifier, sidecar: dict, features: dict) -> tuple[float, bool, float]:
    """Returns (confidence, take, threshold)."""
    feature_cols = model.get_booster().feature_names
    df = pd.DataFrame([features])
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing model features at predict time: {missing}")
    conf = float(model.predict_proba(df[feature_cols])[0, 1])
    th = _bucket_threshold(int(features["volatility_state"]), int(features["is_weekend"]),
                            sidecar, int(features["timestamp"]))
    return conf, conf >= th, th


# ============================================================
#  Cycle helpers
# ============================================================
def next_15m_boundary_s(now_s: float | None = None) -> int:
    """Returns the next xx:00 / xx:15 / xx:30 / xx:45 UTC as unix seconds."""
    if now_s is None:
        now_s = time.time()
    step = 15 * 60
    return int((int(now_s) // step + 1) * step)


def wait_until(ts_s: int) -> None:
    while True:
        delta = ts_s - time.time()
        if delta <= 0:
            return
        time.sleep(min(delta, 10))


def refresh_buffers() -> dict[str, list[list]]:
    """Multi-TF Bitstamp fetch with incremental cache.

    First call (cold): fetches the FULL warmup for all 5 TFs in parallel
    (~3-5s wall time, dominated by the 15m pagination of 3200 bars).

    Subsequent calls (warm): fetches only the last INCREMENTAL_REFRESH_N_BARS
    bars per TF in parallel (~0.3-0.5s wall time, single chunk per TF) and
    merges into the cached buffers. Old bars are trimmed to keep cache bounded.

    Returns a dict keyed by friendly names matching the previous main-loop
    locals : {"15m", "5m", "30m", "1h", "eth_15m"}.
    """
    global _CACHED_BUFFERS
    target_sizes = {
        (PAIR, "15m"): WARMUP_15M,
        (PAIR, "5m"):  WARMUP_5M,
        (PAIR, "30m"): WARMUP_30M,
        (PAIR, "1h"):  WARMUP_1H,
        (ETH_PAIR, "15m"): WARMUP_ETH_15M,
    }
    if not _CACHED_BUFFERS:
        # Cold start : parallel fetch of every warmup. Each TF runs in its own
        # thread ; total wall time = max(per-TF), not sum.
        specs = [(pair, tf, n) for (pair, tf), n in target_sizes.items()]
        out = fetch.fetch_many_parallel(specs)
        _CACHED_BUFFERS = dict(out)
    else:
        # Warm : only refresh the tail of each TF in parallel. The new bars
        # are merged into the cached buffer (existing rows are updated, new
        # ones appended), and the buffer is trimmed to the original warmup size.
        specs = [(pair, tf, INCREMENTAL_REFRESH_N_BARS)
                 for (pair, tf), _ in target_sizes.items()]
        fresh = fetch.fetch_many_parallel(specs)
        for key, new_bars in fresh.items():
            merged = fetch.merge_klines(_CACHED_BUFFERS.get(key, []), new_bars)
            _CACHED_BUFFERS[key] = merged[-target_sizes[key]:]

    return {
        "15m":     _CACHED_BUFFERS[(PAIR, "15m")],
        "5m":      _CACHED_BUFFERS[(PAIR, "5m")],
        "30m":     _CACHED_BUFFERS[(PAIR, "30m")],
        "1h":      _CACHED_BUFFERS[(PAIR, "1h")],
        "eth_15m": _CACHED_BUFFERS[(ETH_PAIR, "15m")],
    }


def freshness_retry(target_open_ms: int) -> list[list]:
    """Refetch last 15m bars until BOTH the just-closed bar (at target_open_ms)
    AND the currently-opening bar (at target_open_ms + 15m) are in Bitstamp's
    response.

    Why both? build_features picks `last_idx = len(df15) - 2`, assuming the
    last entry is the currently-open bar so the second-to-last is the just-
    closed target. If Bitstamp returns only `[..., bar(B-30m), bar(B-15m)]`
    (just-closed at [-1], new bar not published yet), then last_idx=-2 lands
    on bar(B-30m) -- features get computed on the bar from TWO cycles ago.
    Requiring both bars eliminates this off-by-one race.

    Previously this passed as soon as kl[-1][0] >= target_open_ms, which is
    true the moment Bitstamp publishes the currently-forming bar - but the
    just-closed target bar might still be missing for another second or two.
    The strict check fixes the 'Target bar not in buffer; skipping resolution'
    warnings.
    """
    open_bar_ms = target_open_ms + INTERVAL_15M_MS
    for _ in range(FRESHNESS_RETRY_MAX):
        kl = fetch.fetch_klines(PAIR, "15m", n_bars=10)
        ts_set = {int(k[0]) for k in kl}
        if target_open_ms in ts_set and open_bar_ms in ts_set:
            return kl
        time.sleep(FRESHNESS_RETRY_DELAY_S)
    raise TimeoutError(
        f"Bitstamp did not publish both the closed bar {target_open_ms} and "
        f"the opening bar {open_bar_ms} after {FRESHNESS_RETRY_MAX} attempts."
    )


def find_bar(buffer_15m: list[list], target_open_ms: int) -> list | None:
    """Find a 15m bar by its open_time."""
    for k in reversed(buffer_15m):
        if int(k[0]) == target_open_ms:
            return k
    return None


def team_decision(take_up: bool, take_dn: bool) -> str:
    """up_only -> UP ; down_only -> DOWN ; both -> CANCEL ; none -> PASS."""
    if take_up and take_dn:
        return "CANCEL"
    if take_up:
        return "UP"
    if take_dn:
        return "DOWN"
    return "PASS"


# ============================================================
#  CSV log
# ============================================================
def append_live_log(row: dict) -> None:
    """Append one row to live_predictions.csv. Direct mode='a' append (no concat)
    so we don't trigger the pandas FutureWarning on all-NA columns when
    resolved_* fields are None.
    """
    LOGS_DIR.mkdir(exist_ok=True)
    df_row = pd.DataFrame([row])
    write_header = not LIVE_CSV.exists()
    df_row.to_csv(LIVE_CSV, mode="a", header=write_header, index=False)


# ============================================================
#  Rich rendering
# ============================================================
console = Console()


def _wr_style(wr: float | None) -> str:
    if wr is None or pd.isna(wr):
        return "dim"
    if wr >= 0.60:
        return "bold bright_green"
    if wr >= 0.55:
        return "green"
    if wr >= 0.50:
        return "yellow"
    return "red"


def _dir_text(direction: str | None) -> Text:
    if direction == "UP":
        return Text("▲ UP", style="bright_blue")
    if direction == "DOWN":
        return Text("▼ DOWN", style="bright_red")
    if direction == "CANCEL":
        return Text("✗ CANCEL", style="yellow")
    if direction == "PASS":
        return Text("- PASS", style="dim")
    if direction == "up":
        return Text("▲ up", style="bright_blue")
    if direction == "down":
        return Text("▼ down", style="bright_red")
    if direction == "flat":
        return Text("= flat", style="dim")
    return Text(str(direction), style="dim")


def render_cycle(cycle_n: int, just_closed_ts: int, resolution: dict | None,
                 prediction: dict, stats: dict) -> None:
    # Header
    h = Text.assemble(
        ("LIVE TEAM ", "bold cyan"),
        ("modele_updown", "bold magenta"),
        (f"  -  cycle #{cycle_n}", "dim"),
        ("\n", ""),
        ("bar just closed at ", "dim"),
        (pd.Timestamp(just_closed_ts + INTERVAL_15M_MS, unit="ms", tz="UTC")
            .strftime("%Y-%m-%d %H:%M UTC"), "white"),
    )
    console.print(Panel(h, border_style="cyan", padding=(0, 2)))

    # Resolution of previous cycle
    if resolution:
        tab = Table(title="Resolution of previous cycle",
                    title_style="bold cyan", border_style="cyan",
                    header_style="bold", expand=False)
        tab.add_column("Field", style="bold")
        tab.add_column("Value")
        tab.add_row("target bar open",
                    pd.Timestamp(resolution["target_open"], unit="ms", tz="UTC")
                        .strftime("%Y-%m-%d %H:%M UTC"))
        tab.add_row("predicted", _dir_text(resolution["predicted"]))
        tab.add_row("actual",    _dir_text(resolution["actual"]))
        tab.add_row("open / close",
                    f"{resolution['open']:.2f}  ->  {resolution['close']:.2f}")
        if resolution["correct"] is True:
            tab.add_row("verdict", Text("CORRECT", style="bold green"))
        elif resolution["correct"] is False:
            tab.add_row("verdict", Text("WRONG", style="bold red"))
        else:
            tab.add_row("verdict", Text("n/a (no bet placed)", style="dim"))
        console.print(tab)

    # New prediction
    pt = Table(title="New prediction for the bar that just opened",
                title_style="bold cyan", border_style="cyan",
                header_style="bold", expand=False)
    pt.add_column("Model", style="bold")
    pt.add_column("Confidence", justify="right")
    pt.add_column("Threshold", justify="right")
    pt.add_column("Take?", justify="right")
    for name, p, th, take, color in [
        ("modele_up",   prediction["p_up"], prediction["th_up"], prediction["take_up"], "bright_blue"),
        ("modele_down", prediction["p_dn"], prediction["th_dn"], prediction["take_dn"], "bright_red"),
    ]:
        pt.add_row(
            Text(name, style=color),
            f"{p*100:6.2f}%",
            f"{th*100:6.2f}%",
            Text("yes", style="bold green") if take else Text("no", style="dim"),
        )
    console.print(pt)

    headline = Text.assemble(
        ("TEAM decision : ", "bold"),
        (_dir_text(prediction["team"])),
    )
    console.print(Panel(headline, border_style=_dir_text(prediction["team"]).style or "white",
                        padding=(0, 2)))

    # Running stats
    wr = stats["correct"] / stats["bets"] if stats["bets"] else None
    bet_line = Text.assemble(
        ("running  ", "dim"),
        (f"{stats['cycles']:>3} cycles  |  ", "white"),
        (f"{stats['bets']:>3} bets, {stats['correct']:>3} correct", "white"),
        (f"  ->  WR=", "dim"),
        (f"{wr*100:5.2f}%" if wr is not None else "n/a", _wr_style(wr)),
        (f"   ({stats['pass']} pass, {stats['cancel']} cancel)", "dim"),
    )
    console.print(bet_line)
    console.print()


# ============================================================
#  bets.csv helpers (only used when LIVE_MODE=true)
# ============================================================
def _append_bet_row(row: dict) -> None:
    """Append one row to bets.csv with the canonical column order."""
    LOGS_DIR.mkdir(exist_ok=True)
    full_row = {k: row.get(k) for k in BETS_COLUMNS}
    df_row = pd.DataFrame([full_row], columns=BETS_COLUMNS)
    with _BETS_CSV_LOCK:
        write_header = not BETS_CSV.exists()
        df_row.to_csv(BETS_CSV, mode="a", header=write_header, index=False)


def _window_utc(open_ms: int) -> str:
    """Human-readable 15m window label, e.g. '2026-05-20 13:30-13:45 UTC'."""
    o = pd.Timestamp(int(open_ms), unit="ms", tz="UTC")
    c = pd.Timestamp(int(open_ms) + INTERVAL_15M_MS, unit="ms", tz="UTC")
    return f"{o.strftime('%Y-%m-%d %H:%M')}-{c.strftime('%H:%M')} UTC"


def _hour_of(ts_ms: int) -> int:
    """UTC hour-of-day (0-23) of a millisecond timestamp."""
    return int(((int(ts_ms) // (15 * 60 * 1000)) % 96) // 4)


def _append_paper_bet(pending: dict, resolution: dict) -> None:
    """Write ONE complete paper-mode bet row to bets.csv (decision + outcome on
    the same row). Called at resolution time, so a paper bet is logged once,
    fully resolved - paper mode has no money at risk and no restart-recovery
    need, so there is no pending lifecycle. Real-mode bets instead go through
    place_real_bet (pending) + resolve_pending_bets (update in place).
    """
    team = pending["team"]                       # "UP" | "DOWN"
    conf = pending["p_up"] if team == "UP" else pending["p_dn"]
    th = pending["th_up"] if team == "UP" else pending["th_dn"]
    target_open_ms = int(pending["target_open"])
    o = float(resolution["open"])
    c = float(resolution["close"])
    pct = (c - o) / o * 100.0 if o else 0.0
    row = {
        "bet_id": str(uuid.uuid4()),
        "cycle_at": pending["cycle_at"],
        "target_open_ms": target_open_ms,
        "target_close_ms": target_open_ms + INTERVAL_15M_MS,
        "direction": team,
        "confidence": round(float(conf), 6),
        "polymarket_slug": f"btc-updown-15m-{target_open_ms // 1000}",
        "order_status": "paper",
        "shares_filled": 0.0,
        "usdc_spent": 0.0,
        "outcome": "win" if resolution["correct"] else "loss",
        "outcome_source": "bitstamp",
        "outcome_resolved_at_iso": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "mode": "paper",
        "target_window_utc": _window_utc(target_open_ms),
        "threshold": round(float(th), 6),
        "edge": round(float(conf) - float(th), 6),
        "p_up": round(float(pending["p_up"]), 6),
        "p_dn": round(float(pending["p_dn"]), 6),
        "volatility_state": pending["volatility_state"],
        "is_weekend": pending["is_weekend"],
        "hour": pending["hour"],
        "resolved_actual": resolution["actual"],
        "resolved_open_price": o,
        "resolved_close_price": c,
        "resolved_pct_move": round(pct, 4),
    }
    _append_bet_row(row)


# Columns that hold strings but are often all-empty when bets.csv is first read
# (pandas then infers float64). resolve_pending_bets writes strings into them
# later; without forcing object dtype that triggers a pandas FutureWarning and,
# in a future pandas, a hard error. _load_bets coerces them up front.
_BETS_STR_COLUMNS = [
    "cycle_at", "direction", "polymarket_slug", "polymarket_condition_id",
    "polymarket_token_id", "polymarket_order_id", "order_status",
    "outcome", "outcome_source", "outcome_resolved_at_iso", "redeem_tx_hash",
    "mode", "target_window_utc", "resolved_actual",
]


def _load_bets() -> pd.DataFrame:
    """Read bets.csv. Returns empty DataFrame with the right schema if missing."""
    if not BETS_CSV.exists():
        return pd.DataFrame(columns=BETS_COLUMNS)
    df = pd.read_csv(BETS_CSV)
    for col in _BETS_STR_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype("object")
    return df


def _save_bets(df: pd.DataFrame) -> None:
    """Overwrite bets.csv with the given frame (after in-place outcome updates)."""
    LOGS_DIR.mkdir(exist_ok=True)
    # Keep column order canonical even if updates added/reordered things.
    df = df.reindex(columns=BETS_COLUMNS)
    df.to_csv(BETS_CSV, index=False)


def _running_pnl_stats(df: pd.DataFrame) -> tuple[int, float]:
    """Count + PnL sum of all resolved (non-pending, non-stuck) real bets."""
    resolved_mask = df["outcome"].isin(["win", "loss", "bitstamp_win", "bitstamp_loss"])
    sub = df[resolved_mask]
    if sub.empty:
        return 0, 0.0
    return int(len(sub)), float(sub["realized_pnl_usdc"].sum())


# ============================================================
#  Pending-bet reconciliation
# ============================================================
def resolve_pending_bets(client, buf_15m: list[list], stats: dict) -> list[dict]:
    """For every row in bets.csv with outcome='pending' whose window has closed,
    try to determine the outcome (Polymarket -> Bitstamp fallback -> stuck).

    Returns a list of "newly resolved" dicts (for Telegram + Rich), one per row
    updated this cycle. Does NOT call Telegram itself - the caller does, so the
    main loop keeps a single notify call site.

    Side effects:
        - bets.csv is rewritten in place if any row changed.
        - Winning bets trigger client.redeem(condition_id) immediately and
          redeem_tx_hash is recorded on the row.
    """
    if client is None:
        return []
    df = _load_bets()
    if df.empty:
        return []
    if "outcome" not in df.columns:
        return []

    now_ms = int(time.time() * 1000)
    pending_mask = (df["outcome"] == "pending") & (df["target_close_ms"].astype("int64") <= now_ms)
    if not pending_mask.any():
        return []

    newly_resolved: list[dict] = []
    # Track which bet_ids we mutated on the snapshot so the final save (under
    # lock) can re-apply by bet_id without holding the lock across the slow
    # Polymarket / relayer network calls below.
    changed_bet_ids: list[str] = []

    for idx in df.index[pending_mask]:
        row = df.loc[idx]
        # Paper-mode rows are resolved by the main loop's own Bitstamp resolution
        # (see _append_paper_bet); skip them here so we never query Polymarket
        # for a paper bet if LIVE_MODE was flipped between runs.
        if str(row.get("order_status")) == "paper":
            continue
        bet_id = str(row["bet_id"])
        slug = str(row["polymarket_slug"])
        target_close_ms = int(row["target_close_ms"])
        cycles_since_close = (now_ms - target_close_ms) // INTERVAL_15M_MS
        direction = str(row["direction"])
        condition_id = str(row["polymarket_condition_id"])

        # 1. Try Polymarket first.
        settlement = client.read_settlement(slug)
        if settlement is not None and settlement.get("resolved"):
            winner = settlement["winning_outcome"]
            outcome = "win" if winner == direction else "loss"
            source = "polymarket"
            redeem_tx = client.redeem(condition_id) if outcome == "win" else None
            _write_outcome(df, idx, outcome, source, redeem_tx,
                           resolved_actual=str(winner).lower())
            newly_resolved.append(_resolution_event(df.loc[idx], stats, df))
            changed_bet_ids.append(bet_id)
            continue

        # 2. Polymarket not resolved yet. Fall back to Bitstamp if at least one
        #    full cycle has passed since close.
        if cycles_since_close >= 1:
            target_open_ms = int(row["target_open_ms"])
            target_bar = find_bar(buf_15m, target_open_ms)
            if target_bar is not None:
                o, c = float(target_bar[1]), float(target_bar[4])
                if c > o:
                    actual = "UP"
                elif c < o:
                    actual = "DOWN"
                else:
                    # Polymarket convention: '>=' rounds to "Up" on a flat tick.
                    actual = "UP"
                outcome = "bitstamp_win" if actual == direction else "bitstamp_loss"
                source = "bitstamp_fallback"
                # We still try redeem in case Polymarket later confirms the same.
                redeem_tx = client.redeem(condition_id) if outcome == "bitstamp_win" else None
                _write_outcome(df, idx, outcome, source, redeem_tx,
                               resolved_actual=actual.lower(),
                               resolved_open=o, resolved_close=c)
                newly_resolved.append(_resolution_event(df.loc[idx], stats, df))
                changed_bet_ids.append(bet_id)
                continue

        # 3. Still nothing. If too many cycles have passed, mark stuck.
        if cycles_since_close > SETTLEMENT_STUCK_AFTER_CYCLES:
            df.at[idx, "outcome"] = "stuck"
            df.at[idx, "outcome_resolved_at_iso"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            changed_bet_ids.append(bet_id)
            newly_resolved.append({
                "_stuck": True,
                "slug": slug,
                "target_close_iso": pd.Timestamp(target_close_ms, unit="ms", tz="UTC").strftime("%Y-%m-%d %H:%M UTC"),
                "cycles_pending": int(cycles_since_close),
            })

    # Re-apply the snapshot mutations to the live file under the lock so any
    # row that place_real_bet appended while we were doing network calls is
    # preserved. We match by bet_id (place_real_bet always writes a fresh uuid,
    # so no collision with rows we resolved here).
    if changed_bet_ids:
        with _BETS_CSV_LOCK:
            live = _load_bets()
            if "bet_id" in live.columns:
                for bet_id in changed_bet_ids:
                    src_rows = df[df["bet_id"] == bet_id]
                    if src_rows.empty:
                        continue
                    src = src_rows.iloc[0]
                    dst_mask = live["bet_id"] == bet_id
                    if not dst_mask.any():
                        continue
                    for col in df.columns:
                        if col in live.columns:
                            live.loc[dst_mask, col] = src[col]
                _save_bets(live)
    return newly_resolved


def _write_outcome(df: pd.DataFrame, idx, outcome: str, source: str,
                    redeem_tx_hash: str | None,
                    resolved_actual: str | None = None,
                    resolved_open: float | None = None,
                    resolved_close: float | None = None) -> None:
    """Update one row of bets.csv in place: outcome + source + resolved_at + PnL
    + redeem_tx, plus the resolved-detail columns (actual direction, prices, %
    move) when the caller has them."""
    row = df.loc[idx]
    is_win = outcome.endswith("win")
    shares = float(row.get("shares_filled") or 0.0)
    spent = float(row.get("usdc_spent") or 0.0)
    if is_win:
        # On a binary, each winning share pays $1; pnl = shares - what we spent.
        pnl = shares - spent
    else:
        pnl = -spent
    df.at[idx, "outcome"] = outcome
    df.at[idx, "outcome_source"] = source
    df.at[idx, "outcome_resolved_at_iso"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    df.at[idx, "realized_pnl_usdc"] = pnl
    if redeem_tx_hash is not None:
        df.at[idx, "redeem_tx_hash"] = redeem_tx_hash
    if resolved_actual is not None:
        df.at[idx, "resolved_actual"] = resolved_actual
    if resolved_open is not None and resolved_close is not None:
        df.at[idx, "resolved_open_price"] = resolved_open
        df.at[idx, "resolved_close_price"] = resolved_close
        df.at[idx, "resolved_pct_move"] = round(
            (resolved_close - resolved_open) / resolved_open * 100.0, 4
        ) if resolved_open else 0.0


def _resolution_event(row: pd.Series, stats: dict, df_after_write: pd.DataFrame) -> dict:
    """Build the dict the main loop hands to notify.send_bet_resolved_real."""
    running_bets, running_pnl = _running_pnl_stats(df_after_write)
    redeem_tx = row.get("redeem_tx_hash")
    redeem_tx = str(redeem_tx) if pd.notna(redeem_tx) else None
    return {
        "_stuck": False,
        "direction": str(row["direction"]),
        "outcome": str(row["outcome"]),
        "source": str(row["outcome_source"]),
        "realized_pnl_usdc": float(row["realized_pnl_usdc"]) if pd.notna(row["realized_pnl_usdc"]) else None,
        "fill_price": float(row["fill_price_avg"]) if pd.notna(row["fill_price_avg"]) else None,
        "target_close_iso": pd.Timestamp(int(row["target_close_ms"]), unit="ms", tz="UTC").strftime("%Y-%m-%d %H:%M UTC"),
        "running_bets": running_bets,
        "running_pnl": running_pnl,
        "redeem_tx_hash": redeem_tx,
    }


# ============================================================
#  Place a real bet (LIVE_MODE=true)
# ============================================================
def place_real_bet(client, team: str, p_up: float, p_dn: float,
                    target_open_ms: int, dry_run: bool, context: dict,
                    prefetched_balance: float | None = None,
                    prefetched_market: dict | None = None) -> dict:
    """Place a real Polymarket FAK BUY for this cycle's UP/DOWN signal.
    Appends one row to bets.csv. Returns the row that was written (for the
    cycle's Telegram message).

    `client` must be non-None and ready. Caller already checked LIVE_MODE.
    `context` carries the decision context for the log row:
        {th_up, th_dn, volatility_state, is_weekend, hour}.
    `prefetched_balance` / `prefetched_market` : optional values fetched during
    the prior wait_until() so the boundary-time hot path only has to do the
    POST /order. Pass None to fetch fresh inline (slow path).
    """
    cycle_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    target_close_ms = target_open_ms + INTERVAL_15M_MS
    window_start_unix_s = target_open_ms // 1000
    confidence = p_up if team == "UP" else p_dn
    threshold = context["th_up"] if team == "UP" else context["th_dn"]

    bet_fraction = float(os.environ.get("BET_FRACTION", DEFAULT_BET_FRACTION))
    bet_min = float(os.environ.get("BET_MIN_USDC", DEFAULT_BET_MIN_USDC))
    bet_max_frac = DEFAULT_BET_MAX_FRACTION

    # 1. Read balance + size. Use the pre-fetched value if the caller has one
    # (saves the ~500ms web3 RPC round-trip on the hot path).
    if prefetched_balance is not None:
        usdc_pre = float(prefetched_balance)
    else:
        try:
            usdc_pre = client.get_usdc_balance()
        except Exception as e:
            logging.exception("get_usdc_balance failed: %s", e)
            usdc_pre = 0.0
    bet_usdc = round(usdc_pre * bet_fraction, 2)

    # Base row written regardless of outcome (so we can audit skips too).
    base = {
        "bet_id": str(uuid.uuid4()),
        "cycle_at": cycle_at,
        "target_open_ms": int(target_open_ms),
        "target_close_ms": int(target_close_ms),
        "direction": team,
        "confidence": round(float(confidence), 6),
        "polymarket_slug": f"btc-updown-15m-{window_start_unix_s}",
        "polymarket_condition_id": None,
        "polymarket_token_id": None,
        "bet_size_usdc": bet_usdc,
        "polymarket_order_id": None,
        "order_status": None,
        "fill_price_avg": None,
        "shares_filled": 0.0,
        "usdc_spent": 0.0,
        "fee_usdc": None,
        "outcome": "pending",
        "outcome_source": None,
        "outcome_resolved_at_iso": None,
        "realized_pnl_usdc": None,
        "redeem_tx_hash": None,
        "wallet_usdc_pre": usdc_pre,
        "wallet_usdc_post": usdc_pre,
        # comprehensive per-bet context
        "mode": "dry_run" if dry_run else "real",
        "target_window_utc": _window_utc(target_open_ms),
        "threshold": round(float(threshold), 6),
        "edge": round(float(confidence) - float(threshold), 6),
        "p_up": round(float(p_up), 6),
        "p_dn": round(float(p_dn), 6),
        "volatility_state": context["volatility_state"],
        "is_weekend": context["is_weekend"],
        "hour": context["hour"],
        "resolved_actual": None,
        "resolved_open_price": None,
        "resolved_close_price": None,
        "resolved_pct_move": None,
    }

    if bet_usdc < bet_min:
        base["order_status"] = "skipped_balance_low"
        # No bet placed -> not pending, and we don't need to wait for resolution.
        base["outcome"] = "skipped"
        _append_bet_row(base)
        return base
    if bet_usdc > usdc_pre * bet_max_frac:
        # Sanity guard - should never fire because bet_fraction is way below max_frac.
        base["order_status"] = "skipped_sizing_anomaly"
        base["outcome"] = "skipped"
        _append_bet_row(base)
        return base

    # 2. Look up the Polymarket market. Use the pre-fetched market dict if the
    # caller has one (saves the ~500ms Gamma API call on the hot path).
    if prefetched_market is not None:
        market = prefetched_market
    else:
        try:
            from polymarket import MarketNotFoundError
            market = client.find_btc_15m_market(window_start_unix_s)
        except MarketNotFoundError as e:
            logging.warning("Market not found for window_start=%s: %s", window_start_unix_s, e)
            base["order_status"] = "skipped_market_not_found"
            base["outcome"] = "skipped"
            _append_bet_row(base)
            return base
        except Exception as e:
            logging.exception("find_btc_15m_market failed: %s", e)
            base["order_status"] = "failed"
            base["outcome"] = "skipped"
            _append_bet_row(base)
            return base

    token_id = market["up_token_id"] if team == "UP" else market["down_token_id"]
    base["polymarket_condition_id"] = market["condition_id"]
    base["polymarket_token_id"] = token_id

    # 3. Place the order. dry_run is passed through to place_fak_order, which
    #    still reads the book + builds + signs the EIP-712 order, and only
    #    skips the final POST. This means --dry-run genuinely exercises the
    #    order-construction + signing path.
    try:
        result = client.place_fak_order(token_id, bet_usdc, dry_run=dry_run)
    except Exception as e:
        logging.exception("place_fak_order failed: %s", e)
        base["order_status"] = "failed"
        base["outcome"] = "skipped"
        _append_bet_row(base)
        return base

    base["order_status"] = result["status"]
    base["polymarket_order_id"] = result.get("order_id")
    base["fill_price_avg"] = result.get("fill_price_avg")
    base["shares_filled"] = float(result.get("shares_filled") or 0.0)
    base["usdc_spent"] = float(result.get("usdc_spent") or 0.0)
    base["fee_usdc"] = result.get("fee_usdc")

    # dry_run never gets reconciled later -> mark it skipped, not pending.
    if result["status"] == "dry_run":
        base["outcome"] = "skipped"
        base["error_msg"] = result.get("error_msg")
        _append_bet_row(base)
        return base

    if result["status"] == "filled":
        # Read balance again for audit.
        try:
            base["wallet_usdc_post"] = client.get_usdc_balance()
        except Exception:
            pass
        # outcome stays "pending" - we'll resolve at the next cycle.
    elif result["status"] == "rejected":
        # 0 shares filled -> no real bet placed. NOT a loss: nothing was spent
        # and no money was at risk. Use a distinct outcome so _running_pnl_stats
        # does NOT count this in WR / bet count (the canonical filter is
        # outcome in {win, loss, bitstamp_win, bitstamp_loss}).
        base["outcome"] = "rejected"
        base["outcome_source"] = "rejected"
        base["outcome_resolved_at_iso"] = cycle_at
        base["realized_pnl_usdc"] = 0.0
    elif result["status"] == "failed":
        base["outcome"] = "skipped"
    base["error_msg"] = result.get("error_msg")  # ignored by writer (not in BETS_COLUMNS)

    _append_bet_row(base)
    return base


def _emit_resolution_telegrams(events: list[dict]) -> None:
    """Send one Telegram message per newly-resolved (or newly-stuck) bet."""
    for ev in events:
        if ev.get("_stuck"):
            notify.send_stuck(
                slug=ev["slug"],
                target_close_iso=ev["target_close_iso"],
                cycles_pending=ev["cycles_pending"],
            )
        else:
            notify.send_bet_resolved_real(
                direction=ev["direction"],
                outcome=ev["outcome"],
                source=ev["source"],
                realized_pnl_usdc=ev["realized_pnl_usdc"],
                fill_price=ev["fill_price"],
                target_close_iso=ev["target_close_iso"],
                running_bets=ev["running_bets"],
                running_pnl=ev["running_pnl"],
                redeem_tx_hash=ev.get("redeem_tx_hash"),
            )


# ============================================================
#  Main loop
# ============================================================
def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Wire the dedicated error-bot BEFORE anything else can fail. Errors raised
    # while we're still bootstrapping (model load, env parse, polymarket creds)
    # will still flow to the error bot if it's configured.
    notify.install_error_handler()

    # --- CLI + env ---
    parser = argparse.ArgumentParser(description="LIVE TEAM modele_updown daemon")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only honored if LIVE_MODE=true. Builds + signs the order but "
                             "skips the POST /order. Writes a 'dry_run' row to bets.csv.")
    args = parser.parse_args()

    live_mode = _parse_live_mode(os.environ.get("LIVE_MODE"))
    dry_run = bool(args.dry_run)
    logging.info("LIVE_MODE=%s  dry_run=%s", live_mode, dry_run)

    # --- Bootstrap Polymarket client only when needed ---
    polymarket_client = None
    if live_mode:
        # Lazy import: paper-mode users don't need py-clob-client / web3 installed.
        from polymarket import PolymarketClient
        try:
            polymarket_client = PolymarketClient.from_env()
        except RuntimeError as e:
            # Missing env vars -> hard fail. We refuse to fall back to paper silently
            # because the user explicitly set LIVE_MODE=true.
            # logging.error first so the dedicated error bot fires, THEN print + exit.
            logging.error("LIVE_MODE=true but config is incomplete: %s", e)
            console.print(f"[bold red]LIVE_MODE=true but config is incomplete:[/bold red]\n{e}")
            sys.exit(2)
        except Exception as e:
            logging.exception("PolymarketClient.from_env() unexpectedly failed: %s", e)
            console.print(f"[bold red]Polymarket client init failed:[/bold red] {e}")
            sys.exit(3)
        usdc_balance_boot = polymarket_client.get_usdc_balance()
        matic_balance_boot = polymarket_client.get_matic_balance()
        console.print(f"[dim]Polymarket client OK. usdc={usdc_balance_boot:.2f} matic={matic_balance_boot:.4f}[/dim]")
        if matic_balance_boot < MATIC_LOW_THRESHOLD:
            notify.send_low_matic(matic_balance_boot, MATIC_LOW_THRESHOLD)
    else:
        usdc_balance_boot = None
        matic_balance_boot = None

    banner_text = (f"Starting LIVE TEAM (LIVE_MODE={'true' if live_mode else 'false'}"
                    f"{', DRY-RUN' if dry_run else ''})\n"
                    f"Press Ctrl+C to stop.")
    console.print(Panel(Text(banner_text, style="bold cyan"), border_style="cyan"))

    # --- Load models + sidecars ---
    model_up = _load_model(UP_MODEL_PATH)
    sidecar_up = _load_sidecar(UP_SIDECAR)
    model_dn = _load_model(DN_MODEL_PATH)
    sidecar_dn = _load_sidecar(DN_SIDECAR)
    console.print("[dim]Models + sidecars loaded.[/dim]")
    notify.send_startup(live_mode=live_mode,
                         usdc_balance=usdc_balance_boot,
                         matic_balance=matic_balance_boot)

    # --- Startup catch-up on stale pending bets ---
    # We don't have a 15m buffer yet, so the Bitstamp fallback won't fire here.
    # Polymarket settlement check still runs - typically catches anything older
    # than ~5 minutes. Anything still pending is reconciled on the next cycle.
    if polymarket_client is not None:
        boot_resolved = resolve_pending_bets(polymarket_client, buf_15m=[], stats={})
        if boot_resolved:
            logging.info("startup catch-up resolved %d pending bet(s)", len(boot_resolved))
            _emit_resolution_telegrams(boot_resolved)

    pending: dict | None = None
    stats = {"cycles": 0, "bets": 0, "correct": 0, "pass": 0, "cancel": 0}

    while True:
        # Sleep until next xx:00 / xx:15 / xx:30 / xx:45
        next_boundary = next_15m_boundary_s()
        next_boundary_ms = next_boundary * 1000

        # --- Pre-fetch Polymarket info during the wait, so the boundary-time
        # hot path only does the POST /order. Both calls are slow (~500ms each
        # web3 RPC + ~500ms Gamma API) but we have ~10-15min of idle wait
        # before the next boundary. The returned values may be slightly stale
        # by the time we use them, but balance changes only when a bet is
        # placed (= during this same cycle) and the market for the upcoming
        # window is listed on Polymarket several windows in advance.
        pre_market_fut: Future | None = None
        pre_balance_fut: Future | None = None
        if polymarket_client is not None:
            pre_market_fut = _BG_EXECUTOR.submit(
                polymarket_client.find_btc_15m_market, next_boundary,
            )
            pre_balance_fut = _BG_EXECUTOR.submit(
                polymarket_client.get_usdc_balance,
            )

        wait_until(next_boundary)

        try:
            t_cycle_start = time.perf_counter()
            freshness_retry(next_boundary_ms - INTERVAL_15M_MS)  # latest closed bar's open_time
            t_after_fresh = time.perf_counter()

            # Multi-TF parallel + incremental fetch. Cold start does the full
            # warmup (~3-5s once at boot) ; warm cycles only fetch the last 20
            # bars per TF in parallel (~0.3-0.5s).
            buffers = refresh_buffers()
            buf_15m     = buffers["15m"]
            buf_5m      = buffers["5m"]
            buf_30m     = buffers["30m"]
            buf_1h      = buffers["1h"]
            buf_eth_15m = buffers["eth_15m"]
            t_after_fetch = time.perf_counter()

            # The bar that JUST CLOSED has open_time = next_boundary - 15min
            just_closed_open = next_boundary_ms - INTERVAL_15M_MS

            # === 0. Resolve pending REAL bets (LIVE_MODE=true) -- IN BACKGROUND ===
            # The Polymarket Gamma + relayer calls take ~1-2s ; run them in a
            # thread parallel with feature build + predict (CPU-bound, ~400ms).
            # We await the result BEFORE place_real_bet to avoid a bets.csv
            # write race (both resolve and place_real_bet read+modify+write the
            # same CSV). Net saving on the critical path : ~0.5-1.5s.
            resolve_fut: Future | None = None
            if polymarket_client is not None:
                resolve_fut = _BG_EXECUTOR.submit(
                    resolve_pending_bets, polymarket_client, buf_15m, stats,
                )

            # === 1. Resolve previous cycle ===
            resolution = None
            if pending is not None:
                target_bar = find_bar(buf_15m, pending["target_open"])
                if target_bar is None:
                    logging.warning("Target bar %s not in buffer; skipping resolution",
                                    pending["target_open"])
                else:
                    o, c = float(target_bar[1]), float(target_bar[4])
                    if c > o:
                        actual = "up"
                    elif c < o:
                        actual = "down"
                    else:
                        actual = "flat"
                    pred_dir = pending["team"].lower() if pending["team"] in ("UP", "DOWN") else None
                    correct = (actual == pred_dir) if pred_dir is not None else None
                    resolution = {
                        "target_open": pending["target_open"],
                        "predicted": pending["team"],
                        "actual": actual,
                        "open": o, "close": c,
                        "correct": correct,
                    }
                    stats["cycles"] += 1
                    if pending["team"] in ("UP", "DOWN"):
                        stats["bets"] += 1
                        if correct:
                            stats["correct"] += 1
                        # Paper mode: log the now-resolved bet to bets.csv as one
                        # complete row. Real mode logs bets via place_real_bet +
                        # resolve_pending_bets instead.
                        if polymarket_client is None and "th_up" in pending:
                            _append_paper_bet(pending, resolution)
                    elif pending["team"] == "CANCEL":
                        stats["cancel"] += 1
                    else:
                        stats["pass"] += 1

            # === 2. Build features for the bar that just closed ===
            # last_idx = -2 inside build_features ; the last entry of buf_15m must be
            # the CURRENTLY OPEN bar. fetch returns bars up to and including the
            # bar at next_boundary_ms (which has open_time = next_boundary_ms).
            #
            # Both models are independent at feature time (2026-05-25 cleanup
            # removed the leaky `p_up_xmodel` feature from modele_down). The
            # cross-model link is now only the runtime gating below via the
            # `max_p_*_runtime` sidecar fields.
            feats_up = feat_up_mod.build_features(buf_15m, buf_5m, buf_30m, buf_1h)
            p_up, take_up, th_up = _predict_one(model_up, sidecar_up, feats_up)

            feats_dn = feat_dn_mod.build_features(buf_15m, buf_5m, buf_30m, buf_1h,
                                                  buf_eth_15m)
            if np.isnan(feats_dn["ef_ret2"]):
                logging.warning("No ETH 15m bar concurrent with the just-closed "
                                "BTC bar; modele_down ef_* features are NaN this cycle.")

            # === 3. Predict modele_down ===
            p_dn, take_dn, th_dn = _predict_one(model_dn, sidecar_dn, feats_dn)
            t_after_predict = time.perf_counter()

            # === 3b. Cross-model runtime filters (post-take gating) ===
            # resolve_fut keeps running in the background ; its result is no
            # longer awaited here (used to be step 3a) so place_real_bet below
            # can fire the POST /order without waiting for the slow Polymarket
            # settlement reads. The bets.csv read-modify-write race is now
            # handled by _BETS_CSV_LOCK ; we collect resolved_events after the
            # POST.
            # Each sidecar carries an optional `max_p_*_runtime` (1.0 = disabled).
            # modele_up's filter : reject UP bet if modele_down's p_down >= max_p_down_runtime.
            # modele_down's filter: reject DOWN bet if modele_up's p_up >= max_p_up_runtime.
            max_p_down_rt = float(sidecar_up.get("max_p_down_runtime", 1.0))
            if take_up and p_dn >= max_p_down_rt:
                logging.info(f"cross-model: UP take rejected (p_dn={p_dn:.4f} >= "
                             f"max_p_down_runtime={max_p_down_rt:.4f})")
                take_up = False
            max_p_up_rt = float(sidecar_dn.get("max_p_up_runtime", 1.0))
            if take_dn and p_up >= max_p_up_rt:
                logging.info(f"cross-model: DOWN take rejected (p_up={p_up:.4f} >= "
                             f"max_p_up_runtime={max_p_up_rt:.4f})")
                take_dn = False

            team = team_decision(take_up, take_dn)
            prediction = {
                "p_up": p_up, "take_up": take_up, "th_up": th_up,
                "p_dn": p_dn, "take_dn": take_dn, "th_dn": th_dn,
                "team": team,
            }

            # === 4. CSV log : one comprehensive row per cycle ===
            jc_bar = find_bar(buf_15m, just_closed_open)
            hour = _hour_of(feats_up["timestamp"])
            row = {
                "cycle_n": stats["cycles"] + 1,
                "cycle_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                "bar_closed_utc": pd.Timestamp(just_closed_open + INTERVAL_15M_MS, unit="ms",
                                               tz="UTC").strftime("%Y-%m-%d %H:%M UTC"),
                "bar_open":  round(float(jc_bar[1]), 2) if jc_bar else None,
                "bar_high":  round(float(jc_bar[2]), 2) if jc_bar else None,
                "bar_low":   round(float(jc_bar[3]), 2) if jc_bar else None,
                "bar_close": round(float(jc_bar[4]), 2) if jc_bar else None,
                "p_up": round(p_up, 6), "th_up": round(th_up, 6), "take_up": int(take_up),
                "p_dn": round(p_dn, 6), "th_dn": round(th_dn, 6), "take_dn": int(take_dn),
                "volatility_state": int(feats_up["volatility_state"]),
                "is_weekend": int(feats_up["is_weekend"]),
                "hour": hour,
                "team": team,
                "target_window_utc": _window_utc(next_boundary_ms),
                "just_closed_open_ms": just_closed_open,
                "target_open_ms": next_boundary_ms,
                # resolution of the PREVIOUS cycle (one row lag is expected here;
                # the per-bet view in bets.csv has decision + outcome together):
                "prev_predicted": resolution["predicted"] if resolution else None,
                "prev_actual":    resolution["actual"]    if resolution else None,
                "prev_open":      resolution["open"]      if resolution else None,
                "prev_close":     resolution["close"]     if resolution else None,
                "prev_pct_move":  (round((resolution["close"] - resolution["open"])
                                         / resolution["open"] * 100.0, 4)
                                   if resolution and resolution["open"] else None),
                "prev_correct":   (None if resolution is None
                                   else (1 if resolution["correct"]
                                         else (0 if resolution["correct"] is False else None))),
                # running stats (cumulative since daemon start):
                "run_cycles": stats["cycles"],
                "run_bets":   stats["bets"],
                "run_wins":   stats["correct"],
                "run_winrate": (round(stats["correct"] / stats["bets"], 4)
                                if stats["bets"] else None),
                "run_pass":   stats["pass"],
                "run_cancel": stats["cancel"],
            }
            append_live_log(row)

            # === 4.5. Place real bet (LIVE_MODE=true only) ===
            # After live_predictions.csv is written so the cycle log always has the
            # decision for this cycle, regardless of order success.
            bet_row = None
            if polymarket_client is not None and team in ("UP", "DOWN"):
                # Each model bucketizes independently in its own features dict.
                # Log the bucket that the WINNING model actually used so the
                # bets.csv row matches the sidecar lookup that picked the
                # threshold for this bet (avoids audit confusion when feats_up
                # and feats_dn disagree on volatility_state, e.g. after a fresh
                # restart with short warmup).
                feats_for_bucket = feats_up if team == "UP" else feats_dn
                # Resolve the pre-fetched market + balance (kicked off during
                # the prior wait_until). Block at most a couple of seconds in
                # case Polymarket was slow ; fall back to None which makes
                # place_real_bet do the slow fresh fetch.
                prefetched_market = None
                prefetched_balance = None
                try:
                    if pre_market_fut is not None:
                        prefetched_market = pre_market_fut.result(timeout=3)
                except Exception as e:
                    logging.warning("pre_market_fut failed (will refetch inline): %s", e)
                try:
                    if pre_balance_fut is not None:
                        prefetched_balance = pre_balance_fut.result(timeout=3)
                except Exception as e:
                    logging.warning("pre_balance_fut failed (will refetch inline): %s", e)

                bet_row = place_real_bet(
                    polymarket_client, team, p_up, p_dn,
                    target_open_ms=next_boundary_ms, dry_run=dry_run,
                    context={
                        "th_up": th_up, "th_dn": th_dn,
                        "volatility_state": int(feats_for_bucket["volatility_state"]),
                        "is_weekend": int(feats_for_bucket["is_weekend"]),
                        "hour": hour,
                    },
                    prefetched_balance=prefetched_balance,
                    prefetched_market=prefetched_market,
                )
                target_open_iso = pd.Timestamp(int(bet_row["target_open_ms"]), unit="ms", tz="UTC").strftime("%H:%M")
                target_close_iso = pd.Timestamp(int(bet_row["target_close_ms"]), unit="ms", tz="UTC").strftime("%H:%M UTC")
                # Fire telegram in the background -- the order has already been
                # placed, the user doesn't need us to block on the Telegram POST.
                _BG_EXECUTOR.submit(
                    notify.send_bet_placed,
                    direction=str(bet_row["direction"]),
                    usdc_amount=float(bet_row["bet_size_usdc"]),
                    fill_price=bet_row.get("fill_price_avg"),
                    confidence=float(bet_row["confidence"]),
                    target_open_iso=target_open_iso,
                    target_close_iso=target_close_iso,
                    order_id=bet_row.get("polymarket_order_id"),
                    status=str(bet_row["order_status"]) if bet_row.get("order_status") else "unknown",
                    shares_filled=float(bet_row["shares_filled"]) if bet_row.get("shares_filled") else None,
                    error_msg=str(bet_row.get("error_msg") or ""),
                )
            t_after_bet = time.perf_counter()

            # === 4.6. Await background resolve (moved here from before
            # team_decision so the POST /order above is never blocked by the
            # slow Polymarket settlement reads). The _BETS_CSV_LOCK in
            # _append_bet_row + resolve_pending_bets keeps the read-modify-write
            # race safe even though they may now run concurrently.
            if resolve_fut is not None:
                try:
                    resolved_events = resolve_fut.result(timeout=10)
                except Exception as e:
                    logging.exception("background resolve_pending_bets failed: %s", e)
                    resolved_events = []
                if resolved_events:
                    _BG_EXECUTOR.submit(_emit_resolution_telegrams, resolved_events)

            # === 5. Render ===
            cycle_n = stats["cycles"] + 1
            render_cycle(cycle_n, just_closed_open, resolution, prediction, stats)

            # === 6. Telegram notif (only when a real bet was just resolved) ===
            # Fire-and-forget : the resolution is already in stats / bets.csv ;
            # the Telegram POST can finish on its own time.
            if resolution is not None and pending is not None and pending["team"] in ("UP", "DOWN"):
                bet_conf = pending["p_up"] if pending["team"] == "UP" else pending["p_dn"]
                target_ms = pending["target_open"]
                close_ms = target_ms + INTERVAL_15M_MS
                _BG_EXECUTOR.submit(
                    notify.send_resolution,
                    cycle_n=stats["cycles"],  # # of the cycle that just RESOLVED
                    bar_close_ts_iso=pd.Timestamp(close_ms, unit="ms", tz="UTC")
                                       .strftime("%Y-%m-%d %H:%M UTC"),
                    direction=pending["team"],
                    confidence=bet_conf,
                    window_open_iso=pd.Timestamp(target_ms, unit="ms", tz="UTC").strftime("%H:%M"),
                    window_close_iso=pd.Timestamp(close_ms,   unit="ms", tz="UTC").strftime("%H:%M UTC"),
                    open_price=resolution["open"],
                    close_price=resolution["close"],
                    correct=bool(resolution["correct"]),
                    running_bets=stats["bets"],
                    running_wr=(stats["correct"] / stats["bets"]) if stats["bets"] else None,
                )

            # === 7. Save pending for next cycle ===
            # Carries the full decision context so the next cycle can write a
            # complete paper bets.csv row when this cycle resolves. The bucket
            # comes from the model that actually bet (matches what the sidecar
            # used to pick the threshold).
            pending_bucket_feats = feats_up if team == "UP" else feats_dn
            pending = {"target_open": next_boundary_ms, "team": team,
                       "p_up": p_up, "p_dn": p_dn,
                       "th_up": th_up, "th_dn": th_dn,
                       "cycle_at": row["cycle_at"],
                       "volatility_state": int(pending_bucket_feats["volatility_state"]),
                       "is_weekend": int(pending_bucket_feats["is_weekend"]),
                       "hour": hour}

            # === 8. Per-cycle timing breakdown (for performance monitoring) ===
            t_cycle_end = time.perf_counter()
            logging.info(
                "cycle timing : fresh=%.2fs fetch=%.2fs predict=%.2fs bet=%.2fs render=%.2fs TOTAL=%.2fs",
                t_after_fresh - t_cycle_start,
                t_after_fetch - t_after_fresh,
                t_after_predict - t_after_fetch,
                t_after_bet - t_after_predict,
                t_cycle_end - t_after_bet,
                t_cycle_end - t_cycle_start,
            )

        except KeyboardInterrupt:
            console.print("\n[bold red]Stopping.[/bold red]")
            break
        except Exception as e:
            logging.exception("Cycle failed: %s", e)
            time.sleep(5)


if __name__ == "__main__":
    main()
