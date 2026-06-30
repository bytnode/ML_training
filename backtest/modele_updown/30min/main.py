"""Team TRADING backtest (marche reel, P/L net de frais).

Difference avec l'ancien backtest WR : on ne compte plus seulement les paris
gagnants/perdants, on SIMULE le trade reel et on calcule le benefice/perte :

    1. les 2 modeles predisent (equipe : regle de coherence UP/DOWN/PASS/CANCEL)
    2. sur chaque pari pris, on ENTRE une position au debut de la bougie suivante
       (entree = open de la bougie i+1 ~ cloture de la bougie de decision)
         - pari UP   -> LONG  (on profite si la bougie monte)
         - pari DOWN -> SHORT (on profite si la bougie baisse)
    3. on ATTEND l'evolution de la bougie i+1
    4. on VEND/CLOTURE a la cloture de la bougie i+1
    5. on calcule le P/L : rendement directionnel du corps de bougie
         body_ret = (close[i+1] - open[i+1]) / open[i+1]
         gross    = +body_ret (UP)  ou  -body_ret (DOWN)
       moins les FRAIS : 0.05 % a l'achat + 0.05 % a la vente (= 0.10 % A/R).

Le rendement du corps de bougie est reconstruit depuis la table SQLite `candles`
(timeframe 30m) du package modele_down : body = (1+close_ret)/(1+open_ret) - 1,
sans prix absolus (scale-invariant). Aucune dependance a un flux de prix brut.

Working-directory contract :
    cd backtest/modele_updown/30min
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

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# main.py lives at backtest/modele_updown/30min/main.py — its parent dir IS the
# 30min/ folder. Inject parent on sys.path so the sibling packages modele_up /
# modele_down import as top-level (their dir names start with letters; only the
# parent `30min/` starts with a digit and would block dotted imports).
_30MIN_DIR = str(Path(__file__).resolve().parent)
if _30MIN_DIR not in sys.path:
    sys.path.insert(0, _30MIN_DIR)

from modele_up import api as api_up
from modele_down import api as api_down
import detailed_log
import full_logs

ROOT = Path(__file__).resolve().parent
LOGS_DIR = ROOT / "logs"
TRADING_RUNS_CSV = LOGS_DIR / "trading_runs.csv"
TRADES_CSV = LOGS_DIR / "trades.csv"

TEAM_TEST_DAYS = 365
DAY_MS = 24 * 3600 * 1000
INTERVAL_30M_MS = 30 * 60 * 1000

# --- Parametres de trading ---
FEE_PER_SIDE = 0.0005       # 0.05 % par transaction (achat ET vente) -> 0.10 % A/R
INITIAL_CAPITAL = 1000.0    # capital de depart (unites arbitraires), full-invest par trade


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
    """Derive the 4-column taken-bets frame from a full per-bar frame."""
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


# ----------------------------------------------------------------------------
#  Rendements de bougie (corps) depuis la SQLite candles 30m
# ----------------------------------------------------------------------------
def _load_body_returns() -> dict[int, float]:
    """Map open-time (ms) -> body return (close-open)/open de la bougie 30m.

    body = (1+close_ret)/(1+open_ret) - 1  (scale-invariant, pas de prix brut).
    Source : table `candles` (timeframe='30m') du package modele_down.
    """
    down_dir = Path(api_down.__file__).resolve().parent
    db_path = next((p for p in (down_dir / "db" / "btc_multi_tf.db",
                                down_dir / "DB" / "btc_multi_tf.db") if p.exists()), None)
    if db_path is None:
        raise FileNotFoundError(
            f"SQLite candles introuvable sous {down_dir} (db/ ou DB/). Lance data.py."
        )
    con = sqlite3.connect(str(db_path))
    try:
        c = pd.read_sql_query(
            "SELECT timestamp_ms, open_ret, close_ret FROM candles WHERE timeframe='30m'", con)
    finally:
        con.close()
    c = c.dropna(subset=["open_ret", "close_ret"])
    body = (1.0 + c["close_ret"].to_numpy()) / (1.0 + c["open_ret"].to_numpy()) - 1.0
    return dict(zip(c["timestamp_ms"].astype("int64").to_numpy(), body))


def simulate_trades(team_bets: pd.DataFrame, body_map: dict[int, float]) -> pd.DataFrame:
    """Un trade par pari pris. Entree open(i+1), sortie close(i+1).
    gross = +body (UP/long) ou -body (DOWN/short) ; net = gross - 2*FEE_PER_SIDE."""
    cols = ["timestamp", "ts_utc", "traded_ts", "direction", "label",
            "body_ret", "gross_ret", "net_ret", "win_net"]
    if team_bets.empty:
        return pd.DataFrame(columns=cols)
    rows = []
    for _, b in team_bets.iterrows():
        ts = int(b["timestamp"])
        traded_ts = ts + INTERVAL_30M_MS          # bougie i+1 = celle qu'on trade
        body = body_map.get(traded_ts)
        if body is None:
            continue                               # pas de bougie suivante connue -> skip
        gross = body if b["direction"] == "up" else -body
        net = gross - 2.0 * FEE_PER_SIDE
        rows.append({
            "timestamp": ts,
            "ts_utc": pd.Timestamp(ts, unit="ms", tz="UTC").strftime("%Y-%m-%d %H:%M"),
            "traded_ts": traded_ts,
            "direction": b["direction"],
            "label": int(b["label"]) if pd.notna(b["label"]) else None,
            "body_ret": body,
            "gross_ret": gross,
            "net_ret": net,
            "win_net": bool(net > 0),
        })
    return pd.DataFrame(rows, columns=cols)


def trading_stats(trades: pd.DataFrame, initial: float = INITIAL_CAPITAL) -> dict:
    """Agrege le P/L : equity composee, ROI net, frais, WR, drawdown, par sens."""
    if trades.empty:
        return {"n_trades": 0}
    net = trades["net_ret"].to_numpy(dtype=float)
    gross = trades["gross_ret"].to_numpy(dtype=float)
    equity = initial * np.cumprod(1.0 + net)
    final = float(equity[-1])
    peak = np.maximum.accumulate(equity)
    max_dd = float(((equity - peak) / peak).min())

    # check d'alignement : le pari etait-il directionnellement correct ssi gross>0 ?
    lbl = trades["label"]
    chk = trades[lbl.notna() & (trades["body_ret"].abs() > 1e-12)]
    align = float(((chk["gross_ret"] > 0).to_numpy() == (chk["label"] == 1).to_numpy()).mean()) \
        if len(chk) else float("nan")

    def _dir(d):
        sub = trades[trades["direction"] == d]
        if sub.empty:
            return {"n": 0, "wr": float("nan"), "net_pct": 0.0}
        return {"n": len(sub),
                "wr": float((sub["net_ret"] > 0).mean()),
                "net_pct": float((np.prod(1.0 + sub["net_ret"].to_numpy()) - 1.0) * 100)}

    return {
        "n_trades": len(net),
        "gross_total_pct": float((np.prod(1.0 + gross) - 1.0) * 100),
        "net_total_pct": float((final / initial - 1.0) * 100),
        "final_equity": final,
        "pnl": final - initial,
        "fee_drag_pct": float(len(net) * 2.0 * FEE_PER_SIDE * 100),  # frais cumules (pts de capital, additif)
        "avg_net_per_trade_pct": float(net.mean() * 100),
        "avg_gross_per_trade_pct": float(gross.mean() * 100),
        "wr_net": float((net > 0).mean()),
        "wr_dir": float((trades["label"] == 1).mean()),
        "best_pct": float(net.max() * 100),
        "worst_pct": float(net.min() * 100),
        "max_drawdown_pct": max_dd * 100,
        "align_check": align,
        "up": _dir("up"),
        "down": _dir("down"),
        "equity_final": final,
    }


def _fmt_ms(ts_ms: int) -> str:
    return pd.Timestamp(ts_ms, unit="ms", tz="UTC").strftime("%Y-%m-%d %H:%M UTC")


def append_run(stats: dict) -> None:
    """Append one summary row to logs/trading_runs.csv (append-only history)."""
    flat = {k: v for k, v in stats.items() if not isinstance(v, (dict, np.ndarray))}
    new_row = pd.DataFrame([flat])
    if TRADING_RUNS_CSV.exists():
        existing = pd.read_csv(TRADING_RUNS_CSV)
        all_cols = list(existing.columns) + [c for c in new_row.columns if c not in existing.columns]
        existing = existing.reindex(columns=all_cols)
        new_row = new_row.reindex(columns=all_cols)
        combined = pd.concat([existing, new_row], ignore_index=True)
    else:
        combined = new_row
    combined.to_csv(TRADING_RUNS_CSV, index=False)


def main() -> None:
    start_ms, end_ms = compute_window()

    # Predictions per-bar des 2 modeles, puis paris d'equipe (regle de coherence).
    up_full = api_up.evaluate_full_on_slice(start_ms, end_ms)
    down_full = api_down.evaluate_full_on_slice(start_ms, end_ms)
    up_bets = _taken_bets(up_full, "up")
    down_bets = _taken_bets(down_full, "down")
    team_bets, cancelled = build_team_bets(up_bets, down_bets)

    LOGS_DIR.mkdir(exist_ok=True)

    # Logs detailles par bougie (inchanges : team_all_bars / team_skipped / team_taken).
    per_bar = detailed_log.build_per_bar_table(up_full, down_full)
    detailed_log.write_detailed_logs(per_bar, LOGS_DIR)
    full_logs.write_full_logs(per_bar, api_up, api_down, LOGS_DIR)

    # --- Simulation de trading ---
    body_map = _load_body_returns()
    trades = simulate_trades(team_bets, body_map)
    coverage = (len(trades) / len(team_bets)) if len(team_bets) else 0.0
    trades.to_csv(TRADES_CSV, index=False)

    stats = trading_stats(trades)
    summary = {
        "run_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "window_days": TEAM_TEST_DAYS,
        "window_start": _fmt_ms(start_ms),
        "window_end": _fmt_ms(end_ms),
        "fee_per_side_pct": FEE_PER_SIDE * 100,
        "initial_capital": INITIAL_CAPITAL,
        "team_bets": len(team_bets),
        "cancelled": cancelled,
        "coverage": round(coverage, 4),
        **{k: (round(v, 5) if isinstance(v, float) else v)
           for k, v in stats.items() if not isinstance(v, dict)},
    }
    append_run(summary)

    _render_report(summary, stats, coverage)


# ----------------------------------------------------------------------------
#  Rapport Rich
# ----------------------------------------------------------------------------
def _pnl_color(pct: float) -> str:
    if pd.isna(pct):
        return "dim"
    return "bold bright_green" if pct > 0 else "red" if pct < 0 else "yellow"


def _render_report(summary: dict, stats: dict, coverage: float) -> None:
    console = Console()

    header = Text.assemble(
        ("TEAM ", "bold cyan"),
        ("modele_updown", "bold magenta"),
        ("  -  TRADING backtest (P/L net de frais)", "dim"),
    )
    window = Text.assemble(
        ("window  ", "dim"),
        (f"{summary['window_start']}  ->  {summary['window_end']}", "white"),
        (f"   ({summary['window_days']} days)", "dim"),
    )
    fees = Text.assemble(
        ("frais   ", "dim"),
        (f"{summary['fee_per_side_pct']:.3f}% / cote", "white"),
        (f"  (= {2*summary['fee_per_side_pct']:.3f}% aller-retour)", "dim"),
        ("    capital initial ", "dim"),
        (f"{summary['initial_capital']:.0f}", "white"),
    )
    console.print(Panel(Text("\n").join([header, window, fees]),
                        border_style="cyan", padding=(0, 2)))

    if stats.get("n_trades", 0) == 0:
        console.print(Panel(Text("Aucun trade sur la fenetre.", style="red"),
                            border_style="red"))
        return

    # --- Bloc trades / WR ---
    t = Table(title="Trades", title_style="bold cyan", border_style="cyan",
              header_style="bold", expand=False)
    t.add_column("Metrique", style="bold")
    t.add_column("Valeur", justify="right")
    t.add_row("Trades pris", f"{stats['n_trades']}")
    t.add_row("Couverture bougie i+1", f"{coverage*100:.1f}%")
    t.add_row("WR directionnel", f"{stats['wr_dir']*100:.2f}%")
    t.add_row("WR net (apres frais)", f"{stats['wr_net']*100:.2f}%")
    t.add_row("Check alignement signe/label",
              "n/a" if pd.isna(stats["align_check"]) else f"{stats['align_check']*100:.1f}%")
    console.print(t)

    # --- Bloc P/L ---
    p = Table(title="Profit / Perte", title_style="bold cyan", border_style="cyan",
              header_style="bold", expand=False)
    p.add_column("Metrique", style="bold")
    p.add_column("Valeur", justify="right")
    p.add_row("Rendement BRUT (hors frais)",
              Text(f"{stats['gross_total_pct']:+.2f}%", style=_pnl_color(stats['gross_total_pct'])))
    p.add_row("Rendement NET (compose)",
              Text(f"{stats['net_total_pct']:+.2f}%", style=_pnl_color(stats['net_total_pct'])))
    p.add_row("Capital final", f"{stats['final_equity']:.2f}")
    p.add_row("P/L", Text(f"{stats['pnl']:+.2f}", style=_pnl_color(stats['pnl'])))
    p.add_row("Frais cumules (drag)", Text(f"-{stats['fee_drag_pct']:.2f} pts", style="red"))
    p.add_row("Moyenne / trade (net)",
              Text(f"{stats['avg_net_per_trade_pct']:+.4f}%",
                   style=_pnl_color(stats['avg_net_per_trade_pct'])))
    p.add_row("Moyenne / trade (brut)", f"{stats['avg_gross_per_trade_pct']:+.4f}%")
    p.add_row("Meilleur / pire trade",
              f"{stats['best_pct']:+.2f}% / {stats['worst_pct']:+.2f}%")
    p.add_row("Max drawdown", Text(f"{stats['max_drawdown_pct']:.2f}%", style="red"))
    console.print(p)

    # --- Par sens ---
    d = Table(title="Par sens", title_style="bold cyan", border_style="cyan",
              header_style="bold", expand=False)
    d.add_column("Sens", style="bold")
    d.add_column("Trades", justify="right")
    d.add_column("WR net", justify="right")
    d.add_column("Rendement net", justify="right")
    for name, key, style in (("LONG (up)", "up", "bright_blue"),
                             ("SHORT (down)", "down", "bright_red")):
        s = stats[key]
        wr = "n/a" if pd.isna(s["wr"]) else f"{s['wr']*100:.2f}%"
        d.add_row(Text(name, style=style), f"{s['n']}", wr,
                  Text(f"{s['net_pct']:+.2f}%", style=_pnl_color(s["net_pct"])))
    console.print(d)

    # --- Headline ---
    net = stats["net_total_pct"]
    headline = Text.assemble(
        ("P/L NET  ", "bold"),
        (f"{net:+.2f}%", f"bold {_pnl_color(net)}"),
        (f"   ({stats['pnl']:+.2f} sur {summary['initial_capital']:.0f})", "dim"),
        ("   sur ", "dim"),
        (f"{stats['n_trades']}", "bold"),
        (" trades, frais ", "dim"),
        (f"{2*summary['fee_per_side_pct']:.2f}% A/R", "dim"),
    )
    console.print(Panel(headline, border_style=_pnl_color(net), padding=(0, 2)))

    console.print(Text.assemble(("  written  ", "dim"), ("logs/trades.csv", "white"),
                                (f"   ({stats['n_trades']} trades)", "dim")))
    console.print(Text.assemble(("  logged   ", "dim"),
                                (f"{TRADING_RUNS_CSV.relative_to(ROOT)}", "white"),
                                ("   (+1 line)", "dim")))


if __name__ == "__main__":
    main()
