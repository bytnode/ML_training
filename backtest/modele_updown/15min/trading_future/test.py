"""Backtest TRADING (futures 1x) de l'équipe modele_up + modele_down.

Même logique de DÉCISION que le backtest polymarket voisin (mêmes 2 modèles,
mêmes seuils, même règle de cohérence d'équipe), mais payoff de TRADING CLASSIQUE
au lieu du binaire 100% / -100% de Polymarket :

    - On entre à l'OUVERTURE de la bougie cible (i+1) dans le sens prédit
      (UP = long, DOWN = short), on SORT à sa CLÔTURE (pas de TP/SL).
    - Gain/perte = mouvement RÉEL du corps de la bougie cible, lu dans la table
      `candles_raw` (15m) de la base SQLite du backtest polymarket.
    - Frais : 0,05% à l'achat + 0,05% à la vente = 0,10% aller-retour (sur le notionnel).
    - Capital : 100$ au départ, 50% du capital courant engagé à chaque trade,
      levier 1x, trades séquentiels (1 bougie = 1 position, jamais de chevauchement).

Contrat de répertoire :
    cd backtest/modele_updown/15min/trading_future
    python test.py
"""
from __future__ import annotations

import sys

# Force UTF-8 stdout (Windows cp1252 ne sait pas encoder les caractères Rich).
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# trading_future/ est le parent direct de ce fichier : on l'injecte sur sys.path
# pour importer les packages sœurs comme top-level (cf. "15min import trick").
_TF_DIR = str(Path(__file__).resolve().parent)
if _TF_DIR not in sys.path:
    sys.path.insert(0, _TF_DIR)

from modele_up import api as api_up
from modele_down import api as api_down

ROOT = Path(__file__).resolve().parent
LOGS_DIR = ROOT / "logs"
TRADING_RUNS_CSV = LOGS_DIR / "trading_runs.csv"

# ---------------------------------------------------------------------------
# Paramètres du backtest
# ---------------------------------------------------------------------------
TEAM_TEST_DAYS = 720           # même fenêtre que le backtest polymarket
DAY_MS = 24 * 3600 * 1000
STEP_MS = 15 * 60 * 1000       # une bougie 15m

INITIAL_CAPITAL = 100.0        # $
POSITION_FRACTION = 0.50       # 50% du capital courant par trade
LEVERAGE = 1.0                 # 1x (sans levier)
FEE_PER_SIDE = 0.0005          # 0,05% à l'achat ET à la vente
ROUND_TRIP_FEE = 2 * FEE_PER_SIDE   # 0,10% aller-retour

# OHLC brut 15m : on lit la table candles_raw. On prend la base locale si elle a
# été reconstruite (data.py), sinon celle — partagée — du backtest polymarket.
RAW_DB_CANDIDATES = [
    ROOT / "modele_down" / "DB" / "btc_multi_tf.db",
    ROOT.parent / "polymarket" / "modele_down" / "DB" / "btc_multi_tf.db",
]


# ---------------------------------------------------------------------------
# Fenêtre + signaux (identiques au backtest polymarket)
# ---------------------------------------------------------------------------
def _read_csv_timestamps(api_module) -> pd.Series:
    csv_path = Path(api_module.__file__).resolve().parent / "logs" / "test_predictions.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} introuvable. Lance d'abord `python train.py` dans le package."
        )
    return pd.read_csv(csv_path, usecols=["timestamp"])["timestamp"]


def compute_window() -> tuple[int, int]:
    up_ts = _read_csv_timestamps(api_up)
    down_ts = _read_csv_timestamps(api_down)
    common_end = int(min(up_ts.max(), down_ts.max()))
    start = common_end - TEAM_TEST_DAYS * DAY_MS
    return start, common_end


def _taken_bets(full: pd.DataFrame, direction: str) -> pd.DataFrame:
    cols = ["timestamp", "direction", "confidence", "label"]
    if full.empty:
        return pd.DataFrame(columns=cols)
    taken = full[full["take"] == 1].copy()
    if taken.empty:
        return pd.DataFrame(columns=cols)
    taken["direction"] = direction
    return taken[cols].reset_index(drop=True)


def build_team_bets(up_bets: pd.DataFrame, down_bets: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Règle de cohérence : on garde le pari seulement si UN SEUL modèle l'a pris."""
    up_ts = set(up_bets["timestamp"]) if not up_bets.empty else set()
    down_ts = set(down_bets["timestamp"]) if not down_bets.empty else set()

    cancelled = up_ts & down_ts
    kept_up = up_bets[up_bets["timestamp"].isin(up_ts - down_ts)]
    kept_down = down_bets[down_bets["timestamp"].isin(down_ts - up_ts)]

    team = pd.concat([kept_up, kept_down], ignore_index=True)
    if not team.empty:
        team = team.sort_values("timestamp").reset_index(drop=True)
    return team, len(cancelled)


# ---------------------------------------------------------------------------
# OHLC brut + simulation de trading
# ---------------------------------------------------------------------------
def load_raw_ohlc() -> dict[int, tuple[float, float]]:
    """Retourne {open_time_ms: (open, close)} pour toutes les bougies 15m."""
    db = next((p for p in RAW_DB_CANDIDATES if p.exists()), None)
    if db is None:
        raise FileNotFoundError(
            "Aucune base candles_raw trouvée. Cherché :\n  "
            + "\n  ".join(str(p) for p in RAW_DB_CANDIDATES)
        )
    con = sqlite3.connect(str(db))
    try:
        raw = pd.read_sql(
            "SELECT timestamp_ms, open, close FROM candles_raw WHERE timeframe='15m'",
            con,
        )
    finally:
        con.close()
    return {int(t): (float(o), float(c))
            for t, o, c in zip(raw["timestamp_ms"], raw["open"], raw["close"])}


def simulate(team_bets: pd.DataFrame, ohlc: dict[int, tuple[float, float]]) -> tuple[pd.DataFrame, int]:
    """Joue les trades en séquence et renvoie (trades_df, n_missing)."""
    rows = []
    equity = INITIAL_CAPITAL
    missing = 0

    for _, bet in team_bets.iterrows():
        ts = int(bet["timestamp"])           # open de la bougie i (= décision)
        target = ts + STEP_MS                # open de la bougie cible i+1
        if target not in ohlc:
            missing += 1
            continue
        o, c = ohlc[target]
        if o <= 0:
            missing += 1
            continue

        body = (c - o) / o                   # corps réel de la bougie cible
        direction = bet["direction"]
        gross = body if direction == "up" else -body   # long vs short
        net_ret = gross - ROUND_TRIP_FEE     # rendement net sur le notionnel

        stake = POSITION_FRACTION * equity * LEVERAGE
        fee_paid = stake * ROUND_TRIP_FEE
        pnl = stake * net_ret
        equity += pnl

        rows.append({
            "timestamp": ts,
            "ts_utc": pd.Timestamp(ts, unit="ms", tz="UTC").strftime("%Y-%m-%d %H:%M"),
            "direction": direction.upper(),
            "confidence": round(float(bet["confidence"]), 4),
            "target_open": round(o, 2),
            "target_close": round(c, 2),
            "body_pct": round(body * 100, 4),
            "gross_pct": round(gross * 100, 4),
            "net_pct": round(net_ret * 100, 4),
            "result": "win" if gross > 0 else "loss",
            "stake": round(stake, 4),
            "fee_paid": round(fee_paid, 6),
            "pnl": round(pnl, 6),
            "equity": round(equity, 4),
        })

    return pd.DataFrame(rows), missing


# ---------------------------------------------------------------------------
# Métriques + rapport
# ---------------------------------------------------------------------------
def _max_drawdown(equity: pd.Series) -> float:
    """Drawdown max relatif (en %) sur la courbe d'équité (incl. capital initial)."""
    curve = pd.concat([pd.Series([INITIAL_CAPITAL]), equity], ignore_index=True)
    peak = curve.cummax()
    dd = (curve - peak) / peak
    return float(dd.min() * 100)


def compute_metrics(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {"n_trades": 0}

    wins = trades[trades["result"] == "win"]
    losses = trades[trades["result"] == "loss"]
    final_eq = float(trades["equity"].iloc[-1])
    gross_profit = float(trades.loc[trades["pnl"] > 0, "pnl"].sum())
    gross_loss = float(trades.loc[trades["pnl"] < 0, "pnl"].sum())

    return {
        "n_trades": len(trades),
        "n_up": int((trades["direction"] == "UP").sum()),
        "n_down": int((trades["direction"] == "DOWN").sum()),
        "win_rate": round(len(wins) / len(trades) * 100, 2),
        "avg_win_gross_pct": round(float(wins["gross_pct"].mean()), 4) if not wins.empty else 0.0,
        "avg_loss_gross_pct": round(float(losses["gross_pct"].mean()), 4) if not losses.empty else 0.0,
        "avg_net_pct": round(float(trades["net_pct"].mean()), 4),
        "total_fees": round(float(trades["fee_paid"].sum()), 4),
        "initial_capital": INITIAL_CAPITAL,
        "final_equity": round(final_eq, 4),
        "total_return_pct": round((final_eq / INITIAL_CAPITAL - 1) * 100, 2),
        "max_drawdown_pct": round(_max_drawdown(trades["equity"]), 2),
        "profit_factor": round(gross_profit / abs(gross_loss), 3) if gross_loss != 0 else float("inf"),
        "best_trade": round(float(trades["pnl"].max()), 4),
        "worst_trade": round(float(trades["pnl"].min()), 4),
    }


def append_run(stats: dict) -> None:
    LOGS_DIR.mkdir(exist_ok=True)
    new_row = pd.DataFrame([stats])
    if TRADING_RUNS_CSV.exists():
        combined = pd.concat([pd.read_csv(TRADING_RUNS_CSV), new_row], ignore_index=True)
    else:
        combined = new_row
    combined.to_csv(TRADING_RUNS_CSV, index=False)


def _fmt_ms(ts_ms: int) -> str:
    return pd.Timestamp(ts_ms, unit="ms", tz="UTC").strftime("%Y-%m-%d %H:%M UTC")


def render(stats: dict, window: tuple[int, int], cancelled: int, missing: int,
           bh_pct: float | None) -> None:
    console = Console()

    console.print(Panel(
        Text.assemble(
            ("TRADING FUTURE ", "bold cyan"), ("modele_updown", "bold magenta"),
            ("  -  backtest 1x, sortie à la clôture\n", "dim"),
            ("window  ", "dim"),
            (f"{_fmt_ms(window[0])}  ->  {_fmt_ms(window[1])}", "white"),
            (f"   ({TEAM_TEST_DAYS} days)", "dim"),
        ),
        border_style="cyan", padding=(0, 2),
    ))

    if stats["n_trades"] == 0:
        console.print("[red]Aucun trade dans la fenêtre.[/red]")
        return

    setup = Table(title="Paramètres", title_style="bold cyan", border_style="cyan",
                  header_style="bold", expand=False)
    setup.add_column("Clé", style="bold"); setup.add_column("Valeur", justify="right")
    setup.add_row("Capital initial", f"{INITIAL_CAPITAL:.2f} $")
    setup.add_row("Fraction / trade", f"{POSITION_FRACTION*100:.0f} %")
    setup.add_row("Levier", f"{LEVERAGE:.0f}x")
    setup.add_row("Frais aller-retour", f"{ROUND_TRIP_FEE*100:.2f} %")
    setup.add_row("Trades (UP / DOWN)", f"{stats['n_trades']} ({stats['n_up']} / {stats['n_down']})")
    setup.add_row("Annulés (cohérence)", f"{cancelled}")
    if missing:
        setup.add_row("Bougies manquantes", f"{missing}")
    console.print(setup)

    perf = Table(title="Performance", title_style="bold cyan", border_style="cyan",
                 header_style="bold", expand=False)
    perf.add_column("Métrique", style="bold"); perf.add_column("Valeur", justify="right")
    wr = stats["win_rate"]
    wr_style = "bold bright_green" if wr >= 60 else "green" if wr >= 55 else "yellow" if wr >= 50 else "red"
    perf.add_row("Win rate (directionnel)", Text(f"{wr:.2f} %", style=wr_style))
    perf.add_row("Gain moyen brut", f"{stats['avg_win_gross_pct']:+.4f} %")
    perf.add_row("Perte moyenne brute", f"{stats['avg_loss_gross_pct']:+.4f} %")
    perf.add_row("Rendement net moyen / trade", f"{stats['avg_net_pct']:+.4f} %")
    perf.add_row("Profit factor", f"{stats['profit_factor']}")
    perf.add_row("Max drawdown", f"{stats['max_drawdown_pct']:.2f} %")
    perf.add_row("Frais totaux payés", f"{stats['total_fees']:.2f} $")
    console.print(perf)

    ret = stats["total_return_pct"]
    ret_style = "bold bright_green" if ret > 0 else "bold red"
    extra = ""
    if bh_pct is not None:
        extra = f"\nBuy & Hold BTC sur la fenêtre : {bh_pct:+.2f} %"
    console.print(Panel(
        Text.assemble(
            ("CAPITAL FINAL  ", "bold"),
            (f"{stats['final_equity']:.2f} $", ret_style),
            (f"   (départ {INITIAL_CAPITAL:.0f} $)\n", "dim"),
            ("RENDEMENT      ", "bold"),
            (f"{ret:+.2f} %", ret_style),
            (f"   sur {stats['n_trades']} trades", "dim"),
            (extra, "dim"),
        ),
        border_style=ret_style.split()[-1], padding=(0, 2),
    ))


def main() -> None:
    start_ms, end_ms = compute_window()

    up_full = api_up.evaluate_full_on_slice(start_ms, end_ms)
    down_full = api_down.evaluate_full_on_slice(start_ms, end_ms)
    up_bets = _taken_bets(up_full, "up")
    down_bets = _taken_bets(down_full, "down")
    team_bets, cancelled = build_team_bets(up_bets, down_bets)

    ohlc = load_raw_ohlc()
    trades, missing = simulate(team_bets, ohlc)

    LOGS_DIR.mkdir(exist_ok=True)
    trades.to_csv(LOGS_DIR / "trades.csv", index=False)

    # Buy & Hold de référence sur la fenêtre (close fin / close début).
    bh_pct = None
    win_keys = sorted(k for k in ohlc if start_ms <= k <= end_ms)
    if len(win_keys) >= 2:
        c0 = ohlc[win_keys[0]][1]
        c1 = ohlc[win_keys[-1]][1]
        if c0 > 0:
            bh_pct = (c1 / c0 - 1) * 100

    stats = compute_metrics(trades)
    run_row = {
        "run_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "window_days": TEAM_TEST_DAYS,
        "window_start": _fmt_ms(start_ms),
        "window_end": _fmt_ms(end_ms),
        "cancelled": cancelled,
        "missing_candles": missing,
        "buy_hold_pct": round(bh_pct, 2) if bh_pct is not None else None,
        **{k: v for k, v in stats.items()},
    }
    append_run(run_row)

    render(stats, (start_ms, end_ms), cancelled, missing, bh_pct)
    console = Console()
    console.print(Text.assemble(("  written  ", "dim"), ("logs/trades.csv", "white"),
                                (f"   ({len(trades)} trades)", "dim")))
    console.print(Text.assemble(("  logged   ", "dim"), ("logs/trading_runs.csv", "white"),
                                ("   (+1 line)", "dim")))


if __name__ == "__main__":
    main()
