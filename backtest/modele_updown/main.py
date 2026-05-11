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

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

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

    print(f"=== TEAM modele_updown ===")
    print(f"window      : {_fmt_ms(start_ms)} -> {_fmt_ms(end_ms)} ({TEAM_TEST_DAYS} days)")
    print(f"up   solo   : {len(up_bets):>5} bets, WR={up_wr:.4f}")
    print(f"down solo   : {len(down_bets):>5} bets, WR={down_wr:.4f}")
    print(f"cancelled   : {cancelled:>5} timestamps where both fired (opposite directions)")
    print(f"team kept   : {kept_up} up + {kept_down} down = {len(team_bets)} bets, WR={team_wr:.4f}")
    if not team_bets.empty:
        print(f"team vs solo: {team_wr - up_wr:+.4f} vs up, {team_wr - down_wr:+.4f} vs down")
    print(f"written     : {TEAM_BETS_CSV.relative_to(ROOT)}")
    print(f"logged      : {TEAM_RUNS_CSV.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
