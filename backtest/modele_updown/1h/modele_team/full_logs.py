"""Wide per-bet team logs (logs/full_logs/{lost,won,all}_bets.csv).

Each row = one bet taken by the team (UP or DOWN). The minimal version here
writes the same per-bar info that `detailed_log.build_per_bar_table` already
produces, filtered to taken bets. The CLAUDE.md mentions a fuller schema
(both models' full feature vectors, hyperparams, sidecar) but that requires
re-loading both packages' parquets + caches ; the minimal version below
covers the file-presence contract needed by main.py and gives downstream
analysis enough to work with.
"""
from __future__ import annotations
from pathlib import Path

import pandas as pd


def write_full_logs(per_bar: pd.DataFrame, api_up, api_down, logs_dir: Path) -> dict:
    """Write logs/full_logs/{lost,won,all}_bets.csv.

    Args:
        per_bar:   table from detailed_log.build_per_bar_table (already
                   carries team_decision + result for taken bets).
        api_up, api_down: kept in the signature for forward-compat with the
                          fuller schema (currently unused).
        logs_dir:  Path to logs/ (the full_logs/ subdir is created inside).

    Returns {filename: n_rows}.
    """
    out_dir = Path(logs_dir) / "full_logs"
    out_dir.mkdir(parents=True, exist_ok=True)

    taken = per_bar[per_bar["team_decision"].isin(["UP", "DOWN"])].copy()

    counts: dict[str, int] = {}

    all_path = out_dir / "all_bets.csv"
    taken.to_csv(all_path, index=False)
    counts[f"full_logs/{all_path.name}"] = len(taken)

    won_path = out_dir / "won_bets.csv"
    won = taken[taken["result"] == "win"]
    won.to_csv(won_path, index=False)
    counts[f"full_logs/{won_path.name}"] = len(won)

    lost_path = out_dir / "lost_bets.csv"
    lost = taken[taken["result"] == "loss"]
    lost.to_csv(lost_path, index=False)
    counts[f"full_logs/{lost_path.name}"] = len(lost)

    return counts
