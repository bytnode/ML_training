"""Team backtest : combine modele_up and modele_down decisions.

Each sister package (modele_up, modele_down) exposes the standard
`evaluate_on_slice(start_ms, end_ms) -> DataFrame` contract and returns one row
per setup where it would have taken a bet. We treat the two outputs as a virtual
team and keep only **coherent** bets :

    up bets UP   + down abstains  -> KEEP up bet
    down bets DN + up abstains    -> KEEP down bet
    both bet                      -> CANCEL (one says UP, the other DOWN)

Working-directory contract :
    cd backtest/modele_updown
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

from modele_up import api as api_up
from modele_down import api as api_down

ROOT = Path(__file__).resolve().parent
LOGS_DIR = ROOT / "logs"
TEAM_BETS_CSV = LOGS_DIR / "team_bets.csv"
TEAM_RUNS_CSV = LOGS_DIR / "team_runs.csv"

TEAM_TEST_DAYS = 365
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

    up_bets = api_up.evaluate_on_slice(start_ms, end_ms)
    down_bets = api_down.evaluate_on_slice(start_ms, end_ms)

    team_bets, cancelled = build_team_bets(up_bets, down_bets)

    LOGS_DIR.mkdir(exist_ok=True)
    team_bets.to_csv(TEAM_BETS_CSV, index=False)

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

    _render_report(stats, team_wr, up_wr, down_wr)


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


def _render_report(stats: dict, team_wr: float, up_wr: float, down_wr: float) -> None:
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
    console.print(
        Text.assemble(
            ("  written  ", "dim"),
            (f"{TEAM_BETS_CSV.relative_to(ROOT)}", "white"),
            (f"   ({stats['team_bets']} rows)", "dim"),
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
