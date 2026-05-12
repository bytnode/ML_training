"""Team paper-trading on real Bitstamp feed (no orders placed).

Every 15min UTC boundary:
  1. Resolve the prediction made last cycle (the bar that just closed).
  2. Fetch fresh OHLCV (15m + 5m + 30m + 1h).
  3. Build features for the bar that just closed (= modele_up + modele_down).
  4. Predict next bar's direction with both models.
  5. Apply team coherence rule (same as main.py):
        up fires + down abstains  -> bet UP
        down fires + up abstains  -> bet DOWN
        both fire                 -> CANCEL  (very rare per main.py analysis)
        both abstain              -> PASS
  6. Log + display Rich panel.

cd backtest/modele_updown && python live.py
"""
from __future__ import annotations

import json
import logging
import sys
import time
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

ROOT = Path(__file__).resolve().parent
LOGS_DIR = ROOT / "logs"
LIVE_CSV = LOGS_DIR / "live_predictions.csv"

UP_MODEL_PATH = ROOT / "modele_up" / "btc_updown_model.json"
UP_SIDECAR    = ROOT / "modele_up" / "btc_updown_model.thresholds.json"
DN_MODEL_PATH = ROOT / "modele_down" / "btc_updown_model.json"
DN_SIDECAR    = ROOT / "modele_down" / "btc_updown_model.thresholds.json"

PAIR = "btcusd"
WARMUP_15M = 3200   # >= 2880 (volume_rel_30d) + margin
WARMUP_5M  = 200
WARMUP_30M = 200
WARMUP_1H  = 200

FRESHNESS_RETRY_MAX = 12
FRESHNESS_RETRY_DELAY_S = 1
INTERVAL_15M_MS = 15 * 60 * 1000


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
    """Reproduces predict.decide()'s threshold lookup (V1 + V2 hour_boost)."""
    strategy = sidecar.get("strategy")
    if strategy not in ("asym_wknd_excl", "asym_wknd_excl_hour_boost"):
        raise ValueError(f"Unknown sidecar strategy: {strategy!r}")
    skip = {(int(v), int(w)) for v, w in sidecar.get("skip_cells", [])}
    if (int(volatility_state), int(is_weekend)) in skip:
        return 1.0
    T = float(sidecar.get("thresholds_per_wknd", {}).get(str(int(is_weekend)), 1.0))
    if strategy == "asym_wknd_excl_hour_boost":
        hb = sidecar.get("hour_boost", {})
        good = set(int(h) for h in hb.get("good_hours", []))
        delta = float(hb.get("delta", 0.0))
        hour = int(((ts_ms // (15 * 60 * 1000)) % 96) // 4)
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


def freshness_retry(target_open_ms: int) -> list[list]:
    """Refetch last 15m bars until the bar at open_time == target_open_ms is
    actually present in Bitstamp's response (not just a more-recent bar).

    Previously this passed as soon as kl[-1][0] >= target_open_ms, which is
    true the moment Bitstamp publishes the currently-forming bar - but the
    just-closed target bar might still be missing for another second or two.
    The strict check fixes the 'Target bar not in buffer; skipping resolution'
    warnings.
    """
    for _ in range(FRESHNESS_RETRY_MAX):
        kl = fetch.fetch_klines(PAIR, "15m", n_bars=10)
        if any(int(k[0]) == target_open_ms for k in kl):
            return kl
        time.sleep(FRESHNESS_RETRY_DELAY_S)
    raise TimeoutError(
        f"Target bar {target_open_ms} not published by Bitstamp after "
        f"{FRESHNESS_RETRY_MAX} attempts."
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
#  Main loop
# ============================================================
def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    console.print(Panel(Text("Starting LIVE TEAM (paper trading, no orders)\n"
                              "Press Ctrl+C to stop.",
                              style="bold cyan"),
                          border_style="cyan"))

    # Load models + sidecars once
    model_up = _load_model(UP_MODEL_PATH)
    sidecar_up = _load_sidecar(UP_SIDECAR)
    model_dn = _load_model(DN_MODEL_PATH)
    sidecar_dn = _load_sidecar(DN_SIDECAR)
    console.print("[dim]Models + sidecars loaded.[/dim]")

    pending: dict | None = None
    stats = {"cycles": 0, "bets": 0, "correct": 0, "pass": 0, "cancel": 0}

    while True:
        # Sleep until next xx:00 / xx:15 / xx:30 / xx:45
        next_boundary = next_15m_boundary_s()
        next_boundary_ms = next_boundary * 1000
        wait_until(next_boundary)

        try:
            freshness_retry(next_boundary_ms - INTERVAL_15M_MS)  # latest closed bar's open_time

            buf_15m = fetch.fetch_klines(PAIR, "15m", WARMUP_15M)
            buf_5m  = fetch.fetch_klines(PAIR, "5m",  WARMUP_5M)
            buf_30m = fetch.fetch_klines(PAIR, "30m", WARMUP_30M)
            buf_1h  = fetch.fetch_klines(PAIR, "1h",  WARMUP_1H)

            # The bar that JUST CLOSED has open_time = next_boundary - 15min
            just_closed_open = next_boundary_ms - INTERVAL_15M_MS

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
                    elif pending["team"] == "CANCEL":
                        stats["cancel"] += 1
                    else:
                        stats["pass"] += 1

            # === 2. Build features for the bar that just closed ===
            # last_idx = -2 inside build_features ; the last entry of buf_15m must be
            # the CURRENTLY OPEN bar. fetch returns bars up to and including the
            # bar at next_boundary_ms (which has open_time = next_boundary_ms).
            feats_up = feat_up_mod.build_features(buf_15m, buf_5m, buf_30m, buf_1h)
            feats_dn = feat_dn_mod.build_features(buf_15m, buf_5m, buf_30m, buf_1h)

            # === 3. Predict ===
            p_up, take_up, th_up = _predict_one(model_up, sidecar_up, feats_up)
            p_dn, take_dn, th_dn = _predict_one(model_dn, sidecar_dn, feats_dn)

            team = team_decision(take_up, take_dn)
            prediction = {
                "p_up": p_up, "take_up": take_up, "th_up": th_up,
                "p_dn": p_dn, "take_dn": take_dn, "th_dn": th_dn,
                "team": team,
            }

            # === 4. CSV log ===
            row = {
                "cycle_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                "just_closed_open_ms": just_closed_open,
                "p_up": round(p_up, 6), "th_up": round(th_up, 6), "take_up": int(take_up),
                "p_dn": round(p_dn, 6), "th_dn": round(th_dn, 6), "take_dn": int(take_dn),
                "team": team,
                "target_open_ms": next_boundary_ms,
                "resolved_target_open_ms": resolution["target_open"] if resolution else None,
                "resolved_predicted": resolution["predicted"] if resolution else None,
                "resolved_actual":    resolution["actual"]    if resolution else None,
                "resolved_open":      resolution["open"]      if resolution else None,
                "resolved_close":     resolution["close"]     if resolution else None,
                "resolved_correct":   (None if resolution is None
                                       else (1 if resolution["correct"]
                                             else (0 if resolution["correct"] is False else None))),
            }
            append_live_log(row)

            # === 5. Render ===
            cycle_n = stats["cycles"] + 1
            render_cycle(cycle_n, just_closed_open, resolution, prediction, stats)

            # === 6. Save pending for next cycle ===
            pending = {"target_open": next_boundary_ms, "team": team,
                       "p_up": p_up, "p_dn": p_dn}

        except KeyboardInterrupt:
            console.print("\n[bold red]Stopping.[/bold red]")
            break
        except Exception as e:
            logging.exception("Cycle failed: %s", e)
            time.sleep(5)


if __name__ == "__main__":
    main()
