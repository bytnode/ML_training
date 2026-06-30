"""Per-bar team backtest logging.

Owns the merging of the two sister packages' `evaluate_full_on_slice` outputs
into one row-per-bar table, plus the writing of the three detailed CSVs used
by main.py :

    team_all_bars.csv  - one row per analyzed 15m bar in the window
    team_skipped.csv   - bars where the team did NOT bet (PASS or CANCEL)
    team_taken.csv     - bars where the team placed a bet (UP or DOWN)

Each row carries both models' p / threshold / take flag, the team decision,
the bar's volatility / weekend / hour bucket, and the outcome (when taken).
Files are overwritten on each run.
"""
from __future__ import annotations
from pathlib import Path

import pandas as pd

# 21 cols : the shape advertised in CLAUDE.md. Skipped/taken partition all.
LOG_COLUMNS = [
    "timestamp", "ts_utc",
    "volatility_state", "is_weekend", "hour",
    "up_p", "up_threshold", "up_take", "up_label",
    "dn_p", "dn_threshold", "dn_take", "dn_label",
    "team_decision",   # UP / DOWN / PASS / CANCEL
    "direction",       # up / down / "" if not taken
    "result",          # win / loss / "" if not taken
    "open", "high", "low", "close",   # current bar OHLC (left blank: no SQLite DB needed for the minimum)
    "next_color",       # green / red / "" if no next bar info available
]


def _decide(up_take: int, dn_take: int) -> str:
    if up_take == 1 and dn_take == 0:
        return "UP"
    if up_take == 0 and dn_take == 1:
        return "DOWN"
    if up_take == 1 and dn_take == 1:
        return "CANCEL"
    return "PASS"


def build_per_bar_table(up_full: pd.DataFrame, down_full: pd.DataFrame) -> pd.DataFrame:
    """Merge the two `evaluate_full_on_slice` outputs into one row per bar.

    The two frames share the bucketing columns (volatility_state, is_weekend,
    hour) by construction, so we keep them from `up_full` and only add the
    DOWN-side columns.
    """
    if up_full.empty and down_full.empty:
        return pd.DataFrame(columns=LOG_COLUMNS)

    up = up_full.rename(columns={
        "confidence": "up_p",
        "threshold":  "up_threshold",
        "take":       "up_take",
        "label":      "up_label",
    }).copy()
    dn = down_full.rename(columns={
        "confidence": "dn_p",
        "threshold":  "dn_threshold",
        "take":       "dn_take",
        "label":      "dn_label",
    })[["timestamp", "dn_p", "dn_threshold", "dn_take", "dn_label"]].copy()

    merged = up.merge(dn, on="timestamp", how="outer").sort_values("timestamp").reset_index(drop=True)

    merged["ts_utc"] = pd.to_datetime(merged["timestamp"], unit="ms", utc=True)\
                         .dt.strftime("%Y-%m-%d %H:%M:%S")

    # Team decision per bar.
    up_take_arr = merged["up_take"].fillna(0).astype(int)
    dn_take_arr = merged["dn_take"].fillna(0).astype(int)
    merged["team_decision"] = [_decide(int(u), int(d)) for u, d in zip(up_take_arr, dn_take_arr)]

    # Direction + result are filled only when the team took a bet.
    direction = []
    result = []
    next_color = []
    for _, r in merged.iterrows():
        td = r["team_decision"]
        if td == "UP":
            direction.append("up")
            # Win iff modele_up's label is 1 (next candle is green / up).
            lbl = r.get("up_label")
            result.append("win" if int(lbl) == 1 else "loss" if pd.notna(lbl) else "")
            next_color.append("green" if pd.notna(lbl) and int(lbl) == 1 else
                              "red"   if pd.notna(lbl) else "")
        elif td == "DOWN":
            direction.append("down")
            lbl = r.get("dn_label")
            result.append("win" if int(lbl) == 1 else "loss" if pd.notna(lbl) else "")
            next_color.append("red" if pd.notna(lbl) and int(lbl) == 1 else
                              "green" if pd.notna(lbl) else "")
        else:
            direction.append("")
            result.append("")
            # When neither model bet, infer next color from whichever label we have.
            lbl_up = r.get("up_label")
            lbl_dn = r.get("dn_label")
            if pd.notna(lbl_up):
                next_color.append("green" if int(lbl_up) == 1 else "red")
            elif pd.notna(lbl_dn):
                next_color.append("red" if int(lbl_dn) == 1 else "green")
            else:
                next_color.append("")

    merged["direction"] = direction
    merged["result"] = result
    merged["next_color"] = next_color

    # OHLC kept blank (no SQLite candles DB read needed for the minimum).
    for c in ("open", "high", "low", "close"):
        merged[c] = ""

    # Make sure every LOG_COLUMNS entry exists.
    for c in LOG_COLUMNS:
        if c not in merged.columns:
            merged[c] = ""

    return merged[LOG_COLUMNS].reset_index(drop=True)


def write_detailed_logs(per_bar: pd.DataFrame, logs_dir: Path) -> dict:
    """Write the three CSVs and return {filename: n_rows}.

    The three files partition the per_bar table : team_all_bars =
    team_skipped + team_taken.
    """
    logs_dir = Path(logs_dir)
    logs_dir.mkdir(exist_ok=True)

    counts: dict[str, int] = {}

    all_path = logs_dir / "team_all_bars.csv"
    per_bar.to_csv(all_path, index=False)
    counts[all_path.name] = len(per_bar)

    skipped_mask = per_bar["team_decision"].isin(["PASS", "CANCEL"])
    skipped = per_bar[skipped_mask]
    skipped_path = logs_dir / "team_skipped.csv"
    skipped.to_csv(skipped_path, index=False)
    counts[skipped_path.name] = len(skipped)

    taken_mask = per_bar["team_decision"].isin(["UP", "DOWN"])
    taken = per_bar[taken_mask]
    taken_path = logs_dir / "team_taken.csv"
    taken.to_csv(taken_path, index=False)
    counts[taken_path.name] = len(taken)

    return counts
