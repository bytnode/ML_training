"""Team backtest : combine modele_up and modele_down decisions.

Each sister package (modele_up, modele_down) exposes the standard
`evaluate_on_slice(start_ms, end_ms) -> DataFrame` contract and returns one row
per setup where it would have taken a bet. We treat the two outputs as a virtual
team and keep only **coherent** bets :

    up bets UP   + down abstains  -> KEEP up bet
    down bets DN + up abstains    -> KEEP down bet
    both bet                      -> CANCEL (one says UP, the other DOWN)

Working-directory contract :
    cd backtest/modele_updown/15min
    python main.py
"""
from __future__ import annotations

import sys

# Force UTF-8 stdout on legacy Windows terminals (cp1252 can't encode box-drawing
# / arrows). Must happen before rich.Console is instantiated.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# main.py lives at backtest/modele_updown/15min/main.py — its parent dir IS the
# 15min/ folder. Inject parent on sys.path so the sibling packages modele_up /
# modele_down can be imported as top-level (their directory names start with
# letters; only their parent `15min/` starts with a digit and would block dotted
# imports if we kept them nested).
_15MIN_DIR = str(Path(__file__).resolve().parent)
if _15MIN_DIR not in sys.path:
    sys.path.insert(0, _15MIN_DIR)

from modele_up import api as api_up
from modele_down import api as api_down
import detailed_log
import full_logs

ROOT = Path(__file__).resolve().parent
LOGS_DIR = ROOT / "logs"
TEAM_RUNS_CSV = LOGS_DIR / "team_runs.csv"

TEAM_TEST_DAYS = 720
DAY_MS = 24 * 3600 * 1000


def _read_csv_timestamps(api_module) -> pd.Series:
    """Read the timestamp column from a package's test_predictions.csv."""
    csv_path = Path(api_module.__file__).resolve().parent / "logs" / "test_predictions.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} introuvable. Lance d'abord `python train.py` dans le package."
        )
    return pd.read_csv(csv_path, usecols=["timestamp"])["timestamp"]


def compute_window() -> tuple[int, int]:
    """Anchor end to the most recent timestamp covered by *both* packages."""
    up_ts = _read_csv_timestamps(api_up)
    down_ts = _read_csv_timestamps(api_down)
    common_end = int(min(up_ts.max(), down_ts.max()))
    start = common_end - TEAM_TEST_DAYS * DAY_MS
    return start, common_end


def _taken_bets(full: pd.DataFrame, direction: str) -> pd.DataFrame:
    """Derive the 4-column taken-bets frame (the evaluate_on_slice contract
    shape) from a full per-bar frame returned by evaluate_full_on_slice. Lets
    us predict once and reuse the result for both the team stats and the logs."""
    cols = ["timestamp", "direction", "confidence", "label"]
    if full.empty:
        return pd.DataFrame(columns=cols)
    taken = full[full["take"] == 1].copy()
    if taken.empty:
        return pd.DataFrame(columns=cols)
    taken["direction"] = direction
    return taken[cols].reset_index(drop=True)


def build_team_bets(up_bets: pd.DataFrame, down_bets: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Apply the coherence rule. Returns (team_bets, cancelled_count)."""
    up_ts = set(up_bets["timestamp"]) if not up_bets.empty else set()
    down_ts = set(down_bets["timestamp"]) if not down_bets.empty else set()

    cancelled = up_ts & down_ts
    coherent_up_ts = up_ts - down_ts
    coherent_down_ts = down_ts - up_ts

    kept_up = up_bets[up_bets["timestamp"].isin(coherent_up_ts)]
    kept_down = down_bets[down_bets["timestamp"].isin(coherent_down_ts)]

    team = pd.concat([kept_up, kept_down], ignore_index=True)
    if not team.empty:
        team = team.sort_values("timestamp").reset_index(drop=True)
    return team, len(cancelled)


def _wr(df: pd.DataFrame) -> float:
    return float(df["label"].mean()) if not df.empty else float("nan")


_WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday",
                  "Friday", "Saturday", "Sunday"]


def _weekday_take_rate(per_bar: pd.DataFrame) -> pd.DataFrame:
    """Per-weekday counts of analyzed bars vs bets taken (UP or DOWN). Returns a
    DataFrame indexed by weekday name (Mon..Sun) with columns
    [bars, taken, wins, pct_of_taken, wr] where pct_of_taken is the share of
    THE TOTAL taken bets that fall on this weekday (the 7 values sum to 100%)
    and wr is the win-rate of taken bets that day."""
    cols = ["bars", "taken", "wins", "pct_of_taken", "wr"]
    if per_bar.empty:
        return pd.DataFrame(columns=cols)
    weekday = pd.to_datetime(per_bar["timestamp"], unit="ms", utc=True).dt.day_name()
    bars = weekday.value_counts().reindex(_WEEKDAY_ORDER, fill_value=0)
    taken_mask = per_bar["team_decision"].isin(["UP", "DOWN"])
    taken = weekday[taken_mask].value_counts().reindex(_WEEKDAY_ORDER, fill_value=0)
    win_mask = taken_mask & (per_bar["result"] == "win")
    wins = weekday[win_mask].value_counts().reindex(_WEEKDAY_ORDER, fill_value=0)
    out = pd.DataFrame({"bars": bars, "taken": taken, "wins": wins})
    total_taken = int(taken.sum())
    if total_taken > 0:
        out["pct_of_taken"] = (out["taken"] / total_taken * 100).round(2)
    else:
        out["pct_of_taken"] = pd.NA
    # .where(taken > 0) returns floats (NaN for zero-taken rows), so the
    # division and .round() stay numeric. Using .replace(0, pd.NA) here would
    # coerce dtype to object and break .round() under pandas >= 2.2.
    denom = out["taken"].where(out["taken"] > 0)
    out["wr"] = (out["wins"].astype(float) / denom * 100).round(2)
    return out


def _fmt_ms(ts_ms: int) -> str:
    return pd.Timestamp(ts_ms, unit="ms", tz="UTC").strftime("%Y-%m-%d %H:%M UTC")


def append_run(stats: dict) -> None:
    """Append one row to logs/team_runs.csv. Append-only history of team runs."""
    new_row = pd.DataFrame([stats])
    if TEAM_RUNS_CSV.exists():
        existing = pd.read_csv(TEAM_RUNS_CSV)
        combined = pd.concat([existing, new_row], ignore_index=True)
    else:
        combined = new_row
    combined.to_csv(TEAM_RUNS_CSV, index=False)


def main() -> None:
    start_ms, end_ms = compute_window()

    # Full per-bar prediction frames (every bar in the window, taken or not).
    up_full = api_up.evaluate_full_on_slice(start_ms, end_ms)
    down_full = api_down.evaluate_full_on_slice(start_ms, end_ms)

    # Taken-bets frames (the old evaluate_on_slice 4-col shape) derived from the
    # full frames, so each model still predicts exactly once per run.
    up_bets = _taken_bets(up_full, "up")
    down_bets = _taken_bets(down_full, "down")

    team_bets, cancelled = build_team_bets(up_bets, down_bets)

    LOGS_DIR.mkdir(exist_ok=True)

    # Detailed per-bar logs : team_all_bars.csv / team_skipped.csv / team_taken.csv
    per_bar = detailed_log.build_per_bar_table(up_full, down_full)
    log_counts = detailed_log.write_detailed_logs(per_bar, LOGS_DIR)

    # Full per-bet logs : logs/full_logs/{lost,won,all}_bets.csv
    full_counts = full_logs.write_full_logs(per_bar, api_up, api_down, LOGS_DIR)

    up_wr = _wr(up_bets)
    down_wr = _wr(down_bets)
    team_wr = _wr(team_bets)
    kept_up = int((team_bets["direction"] == "up").sum()) if not team_bets.empty else 0
    kept_down = int((team_bets["direction"] == "down").sum()) if not team_bets.empty else 0

    stats = {
        "run_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "window_days": TEAM_TEST_DAYS,
        "window_start": _fmt_ms(start_ms),
        "window_end": _fmt_ms(end_ms),
        "up_solo_bets": len(up_bets),
        "up_solo_wr": round(up_wr, 4),
        "down_solo_bets": len(down_bets),
        "down_solo_wr": round(down_wr, 4),
        "cancelled": cancelled,
        "team_bets": len(team_bets),
        "team_kept_up": kept_up,
        "team_kept_down": kept_down,
        "team_wr": round(team_wr, 4),
    }
    append_run(stats)

    weekday_table = _weekday_take_rate(per_bar)

    _render_report(stats, team_wr, up_wr, down_wr,
                   {**log_counts, **full_counts}, weekday_table)


def _wr_color(wr: float) -> str:
    """Map WR to a semantic color band (Rich style string)."""
    if pd.isna(wr):
        return "dim"
    if wr >= 0.60:
        return "bold bright_green"
    if wr >= 0.55:
        return "green"
    if wr >= 0.50:
        return "yellow"
    return "red"


def _render_report(stats: dict, team_wr: float, up_wr: float, down_wr: float,
                   log_counts: dict, weekday_table: pd.DataFrame) -> None:
    console = Console()

    header = Text.assemble(
        ("TEAM ", "bold cyan"),
        ("modele_updown", "bold magenta"),
        ("  -  backtest report", "dim"),
    )
    window = Text.assemble(
        ("window  ", "dim"),
        (f"{stats['window_start']}  ->  {stats['window_end']}", "white"),
        (f"   ({stats['window_days']} days)", "dim"),
    )
    console.print(Panel(Text("\n").join([header, window]), border_style="cyan", padding=(0, 2)))

    # --- Solo performance ---
    solo = Table(
        title="Solo performance",
        title_style="bold cyan",
        border_style="cyan",
        header_style="bold",
        show_lines=False,
        expand=False,
    )
    solo.add_column("Model", style="bold")
    solo.add_column("Bets", justify="right")
    solo.add_column("Winrate", justify="right")
    solo.add_column("vs 50%", justify="right", style="dim")

    solo.add_row(
        Text("▲ modele_up", style="bright_blue"),
        f"{stats['up_solo_bets']:>5}",
        Text(f"{up_wr*100:6.2f}%", style=_wr_color(up_wr)),
        f"{(up_wr-0.5)*100:+5.2f} pp",
    )
    solo.add_row(
        Text("▼ modele_down", style="bright_red"),
        f"{stats['down_solo_bets']:>5}",
        Text(f"{down_wr*100:6.2f}%", style=_wr_color(down_wr)),
        f"{(down_wr-0.5)*100:+5.2f} pp",
    )
    console.print(solo)

    # --- Team coherence ---
    coh = Table(
        title="Team coherence",
        title_style="bold cyan",
        border_style="cyan",
        header_style="bold",
        show_lines=False,
        expand=False,
    )
    coh.add_column("Bucket", style="bold")
    coh.add_column("Count", justify="right")
    coh.add_column("Note", style="dim")

    cancelled_style = "red" if stats["cancelled"] > 0 else "dim"
    coh.add_row("Cancelled", Text(f"{stats['cancelled']:>5}", style=cancelled_style),
                "both models fired -> opposite directions")
    coh.add_row(Text("Kept UP", style="bright_blue"), f"{stats['team_kept_up']:>5}",
                "up fired, down abstained")
    coh.add_row(Text("Kept DOWN", style="bright_red"), f"{stats['team_kept_down']:>5}",
                "down fired, up abstained")
    coh.add_row(Text("Total team", style="bold"), Text(f"{stats['team_bets']:>5}", style="bold"),
                "bets actually taken by the team")
    console.print(coh)

    # --- Share of taken bets per weekday (the 7 % values sum to 100%) ---
    if not weekday_table.empty:
        wk = Table(
            title="Share of taken bets per weekday",
            title_style="bold cyan",
            border_style="cyan",
            header_style="bold",
            show_lines=False,
            expand=False,
        )
        wk.add_column("Day", style="bold")
        wk.add_column("Bars", justify="right")
        wk.add_column("Taken", justify="right")
        wk.add_column("% of taken", justify="right")
        wk.add_column("WR", justify="right")
        for day, r in weekday_table.iterrows():
            pct = r["pct_of_taken"]
            pct_str = "  n/a" if pd.isna(pct) else f"{pct:6.2f}%"
            wr = r["wr"]
            wr_str = "  n/a" if pd.isna(wr) else f"{wr:6.2f}%"
            wr_style = _wr_color(float(wr) / 100.0) if pd.notna(wr) else "dim"
            wk.add_row(
                day,
                f"{int(r['bars']):>6}",
                f"{int(r['taken']):>5}",
                pct_str,
                Text(wr_str, style=wr_style),
            )
        total_bars = int(weekday_table["bars"].sum())
        total_taken = int(weekday_table["taken"].sum())
        total_wins = int(weekday_table["wins"].sum())
        total_pct_str = "  n/a" if total_taken == 0 else "100.00%"
        total_wr = (total_wins / total_taken * 100) if total_taken else float("nan")
        total_wr_str = "  n/a" if pd.isna(total_wr) else f"{total_wr:6.2f}%"
        total_wr_style = _wr_color(total_wr / 100.0) if not pd.isna(total_wr) else "dim"
        wk.add_row(
            Text("Total", style="bold"),
            Text(f"{total_bars:>6}", style="bold"),
            Text(f"{total_taken:>5}", style="bold"),
            Text(total_pct_str, style="bold"),
            Text(total_wr_str, style=f"bold {total_wr_style}"),
        )
        console.print(wk)

    # --- Final headline ---
    delta_up = team_wr - up_wr
    delta_dn = team_wr - down_wr
    headline = Text.assemble(
        ("TEAM WINRATE  ", "bold"),
        (f"{team_wr*100:.2f}%", f"bold {_wr_color(team_wr)}"),
        ("   on ", "dim"),
        (f"{stats['team_bets']}", "bold"),
        (" bets", "dim"),
        ("\n", ""),
        ("vs solo       ", "dim"),
        (f"{delta_up*100:+.2f} pp", "green" if delta_up >= 0 else "red"),
        ("  vs up,   ", "dim"),
        (f"{delta_dn*100:+.2f} pp", "green" if delta_dn >= 0 else "red"),
        ("  vs down", "dim"),
    )
    console.print(Panel(headline, border_style=_wr_color(team_wr), padding=(0, 2)))

    # --- Files written ---
    for fname, n_rows in log_counts.items():
        console.print(
            Text.assemble(
                ("  written  ", "dim"),
                (f"logs/{fname}", "white"),
                (f"   ({n_rows} rows)", "dim"),
            )
        )
    console.print(
        Text.assemble(
            ("  logged   ", "dim"),
            (f"{TEAM_RUNS_CSV.relative_to(ROOT)}", "white"),
            ("   (+1 line)", "dim"),
        )
    )


if __name__ == "__main__":
    main()
