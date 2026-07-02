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
    cd backtest/modele_updown/1h/modele_team
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

# main.py lives at backtest/modele_updown/1h/modele_team/main.py. The sibling
# packages modele_up / modele_down live ONE LEVEL UP (in 1h/). Inject that
# grandparent dir on sys.path so they import as top-level (1h/ starts with a
# digit and can't be a package; modele_team/ holds main.py + its helpers).
_PKG_PARENT = str(Path(__file__).resolve().parent.parent)
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from modele_up import api as api_up
from modele_down import api as api_down
import detailed_log
import full_logs

ROOT = Path(__file__).resolve().parent
LOGS_DIR = ROOT / "logs"
TRADING_RUNS_CSV = LOGS_DIR / "trading_runs.csv"
TRADES_CSV = LOGS_DIR / "trades.csv"

# Toute la config editable du backtest vit dans config_main.py.
from config_main import (
    TEAM_TEST_DAYS, DAY_MS, INTERVAL_1H_MS,
    FEE_PER_SIDE, INITIAL_CAPITAL, POSITION_FRACTION, LEVERAGE, EXPOSURE,
    STOP_LOSS_ENABLED, STOP_LOSS_PCT, STOP_LOSS_SIDES,
    STOP_LOSS_CHECK_MIN, STOP_LOSS_STEP_MS, STOP_LOSS_N_STEPS,
    UP_MGMT_ENABLED, UP_MAX_STEPS, UP_STEP_MS, UP_CHECK_MIN,
    UP_TRAIL_ACTIVATE, UP_TRAIL_GAP, UP_TAKE_PROFIT, UP_STOP_LOSS, UP_STOP_LOSS_DEC,
    UP_SL_TIME_DECAY, UP_SL_START_DEC, UP_SL_END_DEC, UP_SL_FLOOR_DEC,
    UP_SKIP_BUCKETS,
    DOWN_MGMT_ENABLED, DOWN_MAX_STEPS, DOWN_STEP_MS,
    DOWN_TRAIL_ACTIVATE, DOWN_TRAIL_GAP, DOWN_TAKE_PROFIT, DOWN_STOP_LOSS, DOWN_STOP_LOSS_DEC,
    DOWN_SL_TIME_DECAY, DOWN_SL_START_DEC, DOWN_SL_END_DEC, DOWN_SL_FLOOR_DEC,
    MAKER, MAKER_FEE_PER_SIDE, MAKER_FILL_WINDOW_MIN,
    DMI_FILTER_ENABLED, DMI_LENGTH, DMI_TIMEFRAME, DMI_SIDES,
    UP_CONF_FILTER_ENABLED, UP_CONF_MIN, UP_DROP_HOURS, DOWN_CONF_DELTA,
    UP_XMODEL_FILTER_ENABLED, UP_XMODEL_MAX_PDOWN,
)


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
#  Filtre DMI (Directional Movement Index) Wilder sur le TF 1h
# ----------------------------------------------------------------------------
def _wilder_dmi(high, low, close, n: int):
    """+DI / -DI de Wilder (longueur n), alignes bougie a bougie sur les entrees.

    +DI[i] / -DI[i] n'utilisent que les bougies <= i (high/low/close de i et i-1,
    puis lissage recursif) -> connus a la cloture de la bougie i. Renvoie 2 arrays
    (NaN sur les premieres bougies / le warmup). On ne compare que le SIGNE de
    (+DI - -DI), donc le seed exact du lissage est sans incidence apres rodage.
    """
    h = np.asarray(high, dtype=float)
    l = np.asarray(low, dtype=float)
    cl = np.asarray(close, dtype=float)
    m = len(h)
    plus = np.full(m, np.nan)
    minus = np.full(m, np.nan)
    if m < 2:
        return plus, minus
    up_move = h[1:] - h[:-1]
    down_move = l[:-1] - l[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0.0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0.0), down_move, 0.0)
    prev_close = cl[:-1]
    tr = np.maximum.reduce([h[1:] - l[1:],
                            np.abs(h[1:] - prev_close),
                            np.abs(l[1:] - prev_close)])

    def _rma(x):  # lissage de Wilder = EMA recursive alpha=1/n (adjust=False)
        return pd.Series(x).ewm(alpha=1.0 / n, adjust=False).mean().to_numpy()

    atr = _rma(tr)
    s_plus = _rma(plus_dm)
    s_minus = _rma(minus_dm)
    with np.errstate(divide="ignore", invalid="ignore"):
        plus[1:] = 100.0 * s_plus / atr
        minus[1:] = 100.0 * s_minus / atr
    return plus, minus


def _load_dmi_1h() -> dict[int, tuple[float, float]]:
    """Map open-time (ms) -> (+DI, -DI) du DMI Wilder(DMI_LENGTH) sur les bougies
    DMI_TIMEFRAME (1h). OHLC reconstruit depuis la table `candles` du package
    modele_down : close est absolu, et *_ret = (val - close_prev)/close_prev, donc
        close_prev = close / (1 + close_ret)
        high = close_prev*(1+high_ret) ; low = close_prev*(1+low_ret).
    Couvre tout l'historique (la fenetre de test est un sous-ensemble)."""
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
            "SELECT timestamp_ms, close, close_ret, high_ret, low_ret "
            "FROM candles WHERE timeframe=? ORDER BY timestamp_ms",
            con, params=(DMI_TIMEFRAME,))
    finally:
        con.close()
    c = c.dropna(subset=["close", "close_ret", "high_ret", "low_ret"]).reset_index(drop=True)
    close = c["close"].to_numpy(dtype=float)
    close_prev = close / (1.0 + c["close_ret"].to_numpy(dtype=float))
    high = close_prev * (1.0 + c["high_ret"].to_numpy(dtype=float))
    low = close_prev * (1.0 + c["low_ret"].to_numpy(dtype=float))
    plus_di, minus_di = _wilder_dmi(high, low, close, DMI_LENGTH)
    ts = c["timestamp_ms"].astype("int64").to_numpy()
    return {int(t): (float(p), float(mi)) for t, p, mi in zip(ts, plus_di, minus_di)}


def _apply_dmi_filter(team_bets: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    """Annule les paris dont le sens contredit le DMI 1h de la bougie de DECISION.
        - pari DOWN annule si +DI > -DI ; pari UP annule si -DI > +DI.
    +DI == -DI (egalite) ou warmup (NaN) ou bougie inconnue -> on GARDE le pari.
    Renvoie (team_bets_filtre, n_removed_up, n_removed_down)."""
    if not DMI_FILTER_ENABLED or team_bets.empty:
        return team_bets, 0, 0
    dmi_map = _load_dmi_1h()
    keep = np.ones(len(team_bets), dtype=bool)
    removed_up = removed_down = 0
    for idx, b in enumerate(team_bets.itertuples(index=False)):
        di = dmi_map.get(int(b.timestamp))
        if di is None:
            continue                                 # bougie inconnue -> pas d'info, on garde
        pdi, mdi = di
        if np.isnan(pdi) or np.isnan(mdi):
            continue                                 # warmup -> on garde
        if b.direction == "down" and "down" in DMI_SIDES and pdi > mdi:
            keep[idx] = False
            removed_down += 1
        elif b.direction == "up" and "up" in DMI_SIDES and mdi > pdi:
            keep[idx] = False
            removed_up += 1
    return team_bets[keep].reset_index(drop=True), removed_up, removed_down


def _apply_conf_filter(team_bets: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Annule les paris UP dont la confiance P(up) < UP_CONF_MIN. DOWN inchange.
    La colonne `confidence` (= P(direction)) est portee par chaque pari depuis
    l'api du modele. Renvoie (team_bets_filtre, n_removed_up)."""
    if not UP_CONF_FILTER_ENABLED or team_bets.empty:
        return team_bets, 0
    drop = (team_bets["direction"] == "up") & (team_bets["confidence"] < UP_CONF_MIN)
    removed = int(drop.sum())
    return team_bets[~drop].reset_index(drop=True), removed


# ----------------------------------------------------------------------------
#  Rendements de bougie (corps) depuis la SQLite candles 30m
# ----------------------------------------------------------------------------
def _load_body_returns() -> dict[int, float]:
    """Map open-time (ms) -> body return (close-open)/open de la bougie 1h.

    body = (1+close_ret)/(1+open_ret) - 1  (scale-invariant, pas de prix brut).
    Source : table `candles` (timeframe='1h') du package modele_down.
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
            "SELECT timestamp_ms, open_ret, close_ret FROM candles WHERE timeframe='1h'", con)
    finally:
        con.close()
    c = c.dropna(subset=["open_ret", "close_ret"])
    body = (1.0 + c["close_ret"].to_numpy()) / (1.0 + c["open_ret"].to_numpy()) - 1.0
    return dict(zip(c["timestamp_ms"].astype("int64").to_numpy(), body))


def _load_intrabar_prices():
    """Renvoie (open_by_ts, close_by_ts) des sous-bougies pour verifier le SL a
    chaque STOP_LOSS_CHECK_MIN minutes. Source : DB_<m>m/btc_<m>m.parquet (OHLCV
    brut, m = STOP_LOSS_CHECK_MIN). None si absent (SL ignore, sortie classique)."""
    m = STOP_LOSS_CHECK_MIN
    fp = ROOT / f"DB_{m}m" / f"btc_{m}m.parquet"
    if not fp.exists():
        return None
    df = pd.read_parquet(fp)
    ts = df["timestamp_ms"].astype("int64").to_numpy()
    open_by = dict(zip(ts, df["open"].astype(float).to_numpy()))
    close_by = dict(zip(ts, df["close"].astype(float).to_numpy()))
    return open_by, close_by


def _load_intrabar_extremes():
    """Renvoie (low_by, high_by) des sous-bougies 1-min pour le test de remplissage
    MAKER (une limite a P0 est remplie si le prix vient la toucher). Meme source que
    _load_intrabar_prices : DB_1m/btc_1m.parquet. None si absent."""
    m = STOP_LOSS_CHECK_MIN
    fp = ROOT / f"DB_{m}m" / f"btc_{m}m.parquet"
    if not fp.exists():
        return None
    df = pd.read_parquet(fp)
    ts = df["timestamp_ms"].astype("int64").to_numpy()
    low_by = dict(zip(ts, df["low"].astype(float).to_numpy()))
    high_by = dict(zip(ts, df["high"].astype(float).to_numpy()))
    return low_by, high_by


def _sl_threshold(j, max_steps, start_dec, end_dec, floor_dec):
    """SL TEMPOREL lineaire : seuil prix (decimal neg) de start_dec (large, j=0) vers
    end_dec (serre, j=max_steps), jamais plus serre (plus proche de 0) que floor_dec."""
    if max_steps <= 0:
        return start_dec
    s = start_dec + (end_dec - start_dec) * (j / max_steps)
    return min(s, floor_dec)


def _up_managed_exit(prices_intrabar, traded_ts):
    # Sortie geree UNIFORME d'un trade UP (long) : take-profit + trailing 'let-run'
    # (verrouillent le gain -> remontent le WR) + stop-loss dur large (cape la perte
    # sans whipsaw). Verifie chaque UP_CHECK_MIN min sur au plus UP_MAX_STEPS pas.
    # Renvoie (gross, reason) ; (None, "close") si pas de 1min.
    open_ib, close_ib = prices_intrabar
    entry = open_ib.get(traded_ts)
    if entry is None:
        return None, "close"
    peak = -9.0
    floor = None
    last = None
    for j in range(UP_MAX_STEPS):
        cl = close_ib.get(traded_ts + j * UP_STEP_MS)
        if cl is None:
            break
        pp = cl / entry - 1.0
        last = pp
        sl_th = (_sl_threshold(j, UP_MAX_STEPS, UP_SL_START_DEC, UP_SL_END_DEC, UP_SL_FLOOR_DEC)
                 if UP_SL_TIME_DECAY else UP_STOP_LOSS_DEC)
        if pp <= sl_th:
            return pp, "stop_loss"
        if pp >= UP_TAKE_PROFIT:
            return pp, "take_profit"
        if pp > peak:
            peak = pp
        if peak >= UP_TRAIL_ACTIVATE:
            floor = peak - UP_TRAIL_GAP
        if floor is not None and pp < floor:
            return pp, "trailing"
    if last is None:
        return None, "close"
    return last, "timestop"


def _down_managed_exit(prices_intrabar, traded_ts):
    # Sortie geree UNIFORME d'un trade DOWN (short), calque exact de _up_managed_exit
    # mais sur le profit SHORT pp = 1 - close/entry (un short gagne quand le prix baisse).
    # take-profit + trailing 'let-run' (verrouillent le gain) + stop-loss dur large
    # (cape la perte sans whipsaw). Verifie chaque DOWN_CHECK_MIN min sur <= DOWN_MAX_STEPS pas.
    # Renvoie (gross_short, reason) ; (None, "close") si pas de 1min.
    open_ib, close_ib = prices_intrabar
    entry = open_ib.get(traded_ts)
    if entry is None:
        return None, "close"
    peak = -9.0
    floor = None
    last = None
    for j in range(DOWN_MAX_STEPS):
        cl = close_ib.get(traded_ts + j * DOWN_STEP_MS)
        if cl is None:
            break
        pp = 1.0 - cl / entry            # profit directionnel d'un short
        last = pp
        sl_th = (_sl_threshold(j, DOWN_MAX_STEPS, DOWN_SL_START_DEC, DOWN_SL_END_DEC, DOWN_SL_FLOOR_DEC)
                 if DOWN_SL_TIME_DECAY else DOWN_STOP_LOSS_DEC)
        if pp <= sl_th:
            return pp, "stop_loss"
        if pp >= DOWN_TAKE_PROFIT:
            return pp, "take_profit"
        if pp > peak:
            peak = pp
        if peak >= DOWN_TRAIL_ACTIVATE:
            floor = peak - DOWN_TRAIL_GAP
        if floor is not None and pp < floor:
            return pp, "trailing"
    if last is None:
        return None, "close"
    return last, "timestop"


def simulate_trades(team_bets: pd.DataFrame, body_map: dict[int, float],
                    prices_intrabar: tuple | None = None,
                    up_meta: dict[int, tuple] | None = None,
                    extremes: tuple | None = None) -> tuple[pd.DataFrame, dict]:
    """Un trade par pari pris. Entree open(i+1), sortie close(i+1).
    gross = +body (UP/long) ou -body (DOWN/short) ; net = gross - (entry_fee + exit_fee).
    L'ENTREE suit MAKER/TAKER (limite a P0, simulee via le fill intrabar) ; la SORTIE
    (close/take_profit/trailing/stop_loss) est TOUJOURS TAKER : impossible de garantir
    un fill maker sur une sortie declenchee par une condition (execution immediate requise).

    Si prices_intrabar est fourni, un STOP-LOSS (STOP_LOSS_PCT) est verifie tous
    les STOP_LOSS_CHECK_MIN min, UNIQUEMENT pour les sens dans STOP_LOSS_SIDES :
    au 1er pas ou le profit <= STOP_LOSS_PCT on sort la (exit_reason='stop_loss'),
    sinon sortie a la cloture 1h (exit_reason='close')."""
    cols = ["timestamp", "ts_utc", "traded_ts", "direction", "label",
            "body_ret", "gross_ret", "net_ret", "win_net", "exit_reason"]
    entry_fee = MAKER_FEE_PER_SIDE if MAKER else FEE_PER_SIDE   # entree : maker si rempli, sinon taker
    exit_fee = FEE_PER_SIDE                                      # sortie : toujours taker
    low_by, high_by = extremes if (MAKER and extremes is not None) else (None, None)
    open_ib0 = prices_intrabar[0] if prices_intrabar is not None else None
    n_would_be = n_missed = missed_up = missed_down = 0
    if team_bets.empty:
        return (pd.DataFrame(columns=cols),
                {"maker": MAKER, "n_would_be": 0, "n_missed": 0, "missed_up": 0,
                 "missed_down": 0, "fill_rate": 1.0, "entry_fee": entry_fee, "exit_fee": exit_fee})
    rows = []
    for _, b in team_bets.iterrows():
        ts = int(b["timestamp"])
        traded_ts = ts + INTERVAL_1H_MS          # bougie i+1 = celle qu'on trade
        body = body_map.get(traded_ts)
        if body is None:
            continue                               # pas de bougie suivante connue -> skip
        up = b["direction"] == "up"
        vol, wknd = (up_meta or {}).get(ts, (-1, -1)) if up else (-1, -1)
        # --- Filtre d'entree UP (gestion des pertes) : on NE PREND PAS le pari dans
        #     les buckets pourris (volatility_state, is_weekend) de UP_SKIP_BUCKETS ---
        if up and UP_MGMT_ENABLED and (vol, wknd) in UP_SKIP_BUCKETS:
            continue
        n_would_be += 1
        # --- Mode MAKER : limite postee a P0 (open i+1) ; remplie SEULEMENT si le prix
        #     revient la toucher (long: un low 1-min < P0 ; short: un high 1-min > P0).
        #     Sinon = opportunite perdue (on ne prend pas le trade). ---
        if MAKER and low_by is not None and open_ib0 is not None:
            entry0 = open_ib0.get(traded_ts)
            lo = hi = None
            if entry0 is not None:
                for j in range(MAKER_FILL_WINDOW_MIN):
                    t = traded_ts + j * 60_000
                    lv = low_by.get(t)
                    hv = high_by.get(t)
                    if lv is not None:
                        lo = lv if lo is None else min(lo, lv)
                    if hv is not None:
                        hi = hv if hi is None else max(hi, hv)
            if entry0 is not None and lo is not None:
                filled = (lo < entry0) if up else (hi > entry0)
                if not filled:
                    n_missed += 1
                    if up:
                        missed_up += 1
                    else:
                        missed_down += 1
                    continue
        gross = body if up else -body
        exit_reason = "close"                      # defaut : sortie a la cloture 1h
        # --- Gestion de sortie geree par sens (let-run + TP + SL backstop) ; sinon SL simple ---
        if up and UP_MGMT_ENABLED and prices_intrabar is not None:
            g2, exit_reason = _up_managed_exit(prices_intrabar, traded_ts)
            if g2 is not None:
                gross = g2
        elif (not up) and DOWN_MGMT_ENABLED and prices_intrabar is not None:
            g2, exit_reason = _down_managed_exit(prices_intrabar, traded_ts)
            if g2 is not None:
                gross = g2
        elif (STOP_LOSS_ENABLED and b["direction"] in STOP_LOSS_SIDES
                and prices_intrabar is not None):
            open_ib, close_ib = prices_intrabar
            entry = open_ib.get(traded_ts)
            if entry is not None:
                for j in range(STOP_LOSS_N_STEPS):
                    cl = close_ib.get(traded_ts + j * STOP_LOSS_STEP_MS)
                    if cl is None:
                        continue
                    pp = (cl / entry - 1.0) if up else (1.0 - cl / entry)
                    if pp <= STOP_LOSS_PCT:
                        gross = pp
                        exit_reason = "stop_loss"
                        break
        net = gross - (entry_fee + exit_fee)
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
            "exit_reason": exit_reason,
        })
    fill_rate = (1.0 - n_missed / n_would_be) if n_would_be else 1.0
    maker_info = {"maker": MAKER, "n_would_be": n_would_be, "n_missed": n_missed,
                  "missed_up": missed_up, "missed_down": missed_down,
                  "fill_rate": fill_rate, "entry_fee": entry_fee, "exit_fee": exit_fee}
    return pd.DataFrame(rows, columns=cols), maker_info


def trading_stats(trades: pd.DataFrame, initial: float = INITIAL_CAPITAL) -> dict:
    """Agrege le P/L : equity composee, ROI net, frais, WR, drawdown, par sens."""
    if trades.empty:
        return {"n_trades": 0}
    # Levier : exposition au mouvement de prix = POSITION_FRACTION * LEVERAGE (= EXPOSURE).
    # P/L ET frais (deja inclus dans net_ret) portent sur le notionnel -> x EXPOSURE.
    net = trades["net_ret"].to_numpy(dtype=float) * EXPOSURE
    gross = trades["gross_ret"].to_numpy(dtype=float) * EXPOSURE
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
        net = sub["net_ret"].to_numpy() * EXPOSURE
        wins = net[net > 0]
        losses = net[net < 0]
        wr = float((net > 0).mean())
        lr = float((net < 0).mean())
        aw_mag = float(wins.mean() * 100) if len(wins) else 0.0      # AvgWin magnitude (%)
        al_mag = float(-losses.mean() * 100) if len(losses) else 0.0  # AvgLoss magnitude (%)
        return {"n": len(sub),
                "wr": wr,
                "net_pct": float((np.prod(1.0 + net) - 1.0) * 100),
                "worst_pct": float(net.min() * 100),
                "best_pct": float(net.max() * 100),
                "avg_win_pct": float(wins.mean() * 100) if len(wins) else float("nan"),
                "avg_loss_pct": float(losses.mean() * 100) if len(losses) else float("nan"),
                "ev_pct": wr * aw_mag - lr * al_mag}  # EV = WR*AvgWin - LR*AvgLoss (%/trade)

    # --- P/L mensuel (USDC) : decoupe l'equity composee par mois calendaire ---
    # Les P/L $ mensuels se telescopent (somme = P/L total), donc moyenne = total / nb mois.
    tr_ts = trades["traded_ts"].to_numpy(dtype="int64")
    months = pd.to_datetime(tr_ts, unit="ms", utc=True).strftime("%Y-%m")
    equity_prev = np.concatenate(([initial], equity[:-1]))
    m_start, m_end = {}, {}
    for m, e0, e1 in zip(months, equity_prev, equity):
        if m not in m_start:
            m_start[m] = e0
        m_end[m] = e1
    month_usdc = {m: m_end[m] - m_start[m] for m in m_start}
    month_pct = {m: (m_end[m] / m_start[m] - 1.0) * 100 for m in m_start}
    n_months = len(m_start)
    total_pnl = final - initial
    best_m = max(month_usdc, key=month_usdc.get)
    worst_m = min(month_usdc, key=month_usdc.get)
    monthly_stats = {
        "usdc_total": total_pnl,
        "usdc_avg_per_month": total_pnl / n_months if n_months else 0.0,
        "n_months": n_months,
        "avg_month_pct": float(np.mean(list(month_pct.values()))),
        "best_month": best_m,
        "best_month_usdc": month_usdc[best_m],
        "best_month_pct": month_pct[best_m],
        "worst_month": worst_m,
        "worst_month_usdc": month_usdc[worst_m],
        "worst_month_pct": month_pct[worst_m],
    }

    # --- EV (esperance) par trade : EV = WR*avg_win - LR*avg_loss (net de frais, x
    #     levier). Mathematiquement = moyenne nette par trade ; on l'expose
    #     decomposee (formule utilisateur) pour l'afficher sur le headline. ---
    wins_all = net[net > 0]
    losses_all = net[net < 0]
    n_all = len(net)
    wr_all = len(wins_all) / n_all if n_all else 0.0
    lr_all = len(losses_all) / n_all if n_all else 0.0
    avg_win_all = float(wins_all.mean()) if len(wins_all) else 0.0
    avg_loss_all = float(-losses_all.mean()) if len(losses_all) else 0.0  # magnitude > 0
    ev_per_trade_pct = (wr_all * avg_win_all - lr_all * avg_loss_all) * 100

    return {
        "n_trades": len(net),
        **monthly_stats,
        "gross_total_pct": float((np.prod(1.0 + gross) - 1.0) * 100),
        "net_total_pct": float((final / initial - 1.0) * 100),
        "final_equity": final,
        "pnl": final - initial,
        "fee_drag_pct": float((gross - net).sum() * 100),  # frais cumules reels (x levier, maker/taker) en pts
        "avg_net_per_trade_pct": float(net.mean() * 100),
        "ev_per_trade_pct": float(ev_per_trade_pct),
        "avg_win_pct_all": float(avg_win_all * 100),    # AvgWin magnitude (%) tous trades
        "avg_loss_pct_all": float(avg_loss_all * 100),  # AvgLoss magnitude (%) tous trades
        "avg_gross_per_trade_pct": float(gross.mean() * 100),
        "wr_net": float((net > 0).mean()),
        "wr_dir": float((trades["label"] == 1).mean()),
        "best_pct": float(net.max() * 100),
        "worst_pct": float(net.min() * 100),
        "max_drawdown_pct": max_dd * 100,
        "align_check": align,
        "n_stop_loss": int((trades["exit_reason"] == "stop_loss").sum()) if "exit_reason" in trades else 0,
        "n_take_profit": int((trades["exit_reason"] == "take_profit").sum()) if "exit_reason" in trades else 0,
        "n_trailing": int((trades["exit_reason"] == "trailing").sum()) if "exit_reason" in trades else 0,
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
    # --- UP : drop des heures fragiles (campagne 500%/cote) : pas de long a ces heures UTC ---
    if UP_DROP_HOURS:
        up_full = up_full.copy()
        up_full.loc[up_full["hour"].isin(UP_DROP_HOURS), "take"] = 0
    # --- DOWN : selectivite (campagne 500%/cote) : short ssi P(down) >= seuil + DOWN_CONF_DELTA ---
    if DOWN_CONF_DELTA:
        down_full = down_full.copy()
        down_full["take"] = (down_full["confidence"] >= down_full["threshold"] + DOWN_CONF_DELTA).astype(int)
    up_bets = _taken_bets(up_full, "up")
    # --- UP : filtre CROSS-MODEL (Strategie 2) : long seulement si P(down) < UP_XMODEL_MAX_PDOWN
    #     (le modele DOWN confirme l'absence de chute). down_full["confidence"] = P(down) par bar. ---
    if UP_XMODEL_FILTER_ENABLED and not up_bets.empty:
        _pdown = dict(zip(down_full["timestamp"].astype("int64"), down_full["confidence"]))
        _keep = up_bets["timestamp"].map(lambda t: _pdown.get(int(t), 0.0) < UP_XMODEL_MAX_PDOWN)
        up_bets = up_bets[_keep].reset_index(drop=True)
    down_bets = _taken_bets(down_full, "down")
    team_bets, cancelled = build_team_bets(up_bets, down_bets)

    # --- Filtre DMI 1h : annule les paris dont le sens contredit le DMI de la
    #     bougie de decision (+DI vs -DI ; pas de seuil ADX). ---
    team_bets, dmi_removed_up, dmi_removed_down = _apply_dmi_filter(team_bets)

    # --- Filtre de confiance UP : ne garde un pari UP que si P(up) >= UP_CONF_MIN. ---
    team_bets, conf_removed_up = _apply_conf_filter(team_bets)

    # Meta par bar UP (volatility_state, is_weekend) pour le filtre d'entree UP.
    _uf = up_full[up_full["take"] == 1]
    up_meta = {int(r["timestamp"]):
               (int(r["volatility_state"]) if pd.notna(r["volatility_state"]) else -1,
                int(r["is_weekend"]) if pd.notna(r["is_weekend"]) else -1)
               for _, r in _uf.iterrows()}

    LOGS_DIR.mkdir(exist_ok=True)

    # Logs detailles par bougie (inchanges : team_all_bars / team_skipped / team_taken).
    per_bar = detailed_log.build_per_bar_table(up_full, down_full)
    detailed_log.write_detailed_logs(per_bar, LOGS_DIR)
    full_logs.write_full_logs(per_bar, api_up, api_down, LOGS_DIR)

    # --- Simulation de trading ---
    body_map = _load_body_returns()
    prices_intrabar = _load_intrabar_prices()
    extremes = _load_intrabar_extremes() if MAKER else None
    trades, maker_info = simulate_trades(team_bets, body_map, prices_intrabar, up_meta, extremes)
    coverage = (maker_info["n_would_be"] / len(team_bets)) if len(team_bets) else 0.0
    trades.to_csv(TRADES_CSV, index=False)

    stats = trading_stats(trades)
    summary = {
        "run_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "window_days": TEAM_TEST_DAYS,
        "window_start": _fmt_ms(start_ms),
        "window_end": _fmt_ms(end_ms),
        "entry_fee_pct": maker_info["entry_fee"] * 100,
        "exit_fee_pct": maker_info["exit_fee"] * 100,
        "initial_capital": INITIAL_CAPITAL,
        "position_fraction": POSITION_FRACTION,
        "leverage": LEVERAGE,
        "exposure": EXPOSURE,
        "maker": maker_info["maker"],
        "fill_rate": round(maker_info["fill_rate"], 4),
        "n_would_be": maker_info["n_would_be"],
        "n_missed": maker_info["n_missed"],
        "missed_up": maker_info["missed_up"],
        "missed_down": maker_info["missed_down"],
        "team_bets": len(team_bets),
        "cancelled": cancelled,
        "dmi_filter": DMI_FILTER_ENABLED,
        "dmi_removed_up": dmi_removed_up,
        "dmi_removed_down": dmi_removed_down,
        "conf_filter": UP_CONF_FILTER_ENABLED,
        "conf_min": UP_CONF_MIN,
        "conf_removed_up": conf_removed_up,
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
    round_trip_fee_pct = summary['entry_fee_pct'] + summary['exit_fee_pct']
    fees = Text.assemble(
        ("frais   ", "dim"),
        (f"entree {summary['entry_fee_pct']:.3f}%", "white"),
        (" (maker)" if summary.get("maker") else " (taker)", "dim"),
        (f"  /  sortie {summary['exit_fee_pct']:.3f}% (taker)", "white"),
        (f"  (= {round_trip_fee_pct:.3f}% aller-retour)", "dim"),
        ("    capital initial ", "dim"),
        (f"{summary['initial_capital']:.0f}$", "white"),
    )
    sizing = Text.assemble(
        ("sizing  ", "dim"),
        (f"{summary['position_fraction']*100:.0f}% du capital", "white"),
        (f"  x levier {summary['leverage']:.0f}", "white"),
        (f"  = exposition {summary['exposure']:.1f}x", "bold yellow"),
        (" le mouvement de prix", "dim"),
    )
    console.print(Panel(Text("\n").join([header, window, fees, sizing]),
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
    t.add_row("Mode d'entree",
              Text("MAKER (limite @ P0)", style="bold magenta") if summary.get("maker")
              else "TAKER (marche @ open)")
    if summary.get("maker"):
        t.add_row("Opportunites perdues (non remplies)",
                  Text(f"{(1-summary['fill_rate'])*100:.1f}%   "
                       f"({summary['n_missed']}/{summary['n_would_be']} ; "
                       f"UP {summary['missed_up']} / DOWN {summary['missed_down']})",
                       style="bold red"))
    t.add_row("Check alignement signe/label",
              "n/a" if pd.isna(stats["align_check"]) else f"{stats['align_check']*100:.1f}%")
    if summary.get("dmi_filter"):
        t.add_row(f"Filtre DMI {DMI_TIMEFRAME} (len {DMI_LENGTH}) - paris annules",
                  Text(f"{summary.get('dmi_removed_up', 0)} UP / "
                       f"{summary.get('dmi_removed_down', 0)} DOWN", style="yellow"))
    if summary.get("conf_filter"):
        t.add_row(f"Filtre confiance UP (p_up >= {UP_CONF_MIN}) - paris annules",
                  Text(f"{summary.get('conf_removed_up', 0)} UP", style="yellow"))
    t.add_row(f"Gestion UP (TP {UP_TAKE_PROFIT*100:.2f}% / SL -{UP_STOP_LOSS:.2f}% prix / trail {UP_TRAIL_GAP*100:.2f}% / max {UP_MAX_STEPS}min)",
              Text(f"SL = -{UP_STOP_LOSS:.2f}% du prix (sur la position, levier hors-jeu)", style="yellow"))
    t.add_row(f"Gestion DOWN (TP {DOWN_TAKE_PROFIT*100:.2f}% / SL -{DOWN_STOP_LOSS:.2f}% prix / trail {DOWN_TRAIL_GAP*100:.2f}% / max {DOWN_MAX_STEPS}min)",
              Text(f"SL = -{DOWN_STOP_LOSS:.2f}% du prix (sur la position, levier hors-jeu)", style="yellow"))
    t.add_row("Sorties gerees (tous sens : TP / SL / trail)",
              f"{stats.get('n_take_profit', 0)} / {stats.get('n_stop_loss', 0)} / {stats.get('n_trailing', 0)}")
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

    # --- Bloc mensuel (USDC) ---
    mtab = Table(title="Par mois (USDC)", title_style="bold cyan", border_style="cyan",
                 header_style="bold", expand=False)
    mtab.add_column("Metrique", style="bold")
    mtab.add_column("Valeur", justify="right")
    mtab.add_row("USDC gagne au total",
                 Text(f"{stats['usdc_total']:+,.2f}$", style=_pnl_color(stats['usdc_total'])))
    mtab.add_row(f"Moyenne gagnee / mois ({stats['n_months']} mois)",
                 Text(f"{stats['usdc_avg_per_month']:+,.2f}$", style=_pnl_color(stats['usdc_avg_per_month'])))
    mtab.add_row(f"Meilleur mois ({stats['best_month']})",
                 Text(f"{stats['best_month_usdc']:+,.2f}$   ({stats['best_month_pct']:+.1f}%)",
                      style="bold bright_green"))
    mtab.add_row(f"Pire mois ({stats['worst_month']})",
                 Text(f"{stats['worst_month_usdc']:+,.2f}$   ({stats['worst_month_pct']:+.1f}%)",
                      style="red"))
    console.print(mtab)

    # --- Par sens ---
    d = Table(title="Par sens", title_style="bold cyan", border_style="cyan",
              header_style="bold", expand=False)
    d.add_column("Sens", style="bold")
    d.add_column("Trades", justify="right")
    d.add_column("WR net", justify="right")
    d.add_column("Rendement net", justify="right")
    d.add_column("Gain moyen", justify="right")
    d.add_column("Perte moyenne", justify="right")
    d.add_column("Pire trade", justify="right")
    for name, key, style in (("LONG (up)", "up", "bright_blue"),
                             ("SHORT (down)", "down", "bright_red")):
        s = stats[key]
        wr = "n/a" if pd.isna(s["wr"]) else f"{s['wr']*100:.2f}%"
        worst = "n/a" if not s.get("n") else f"{s['worst_pct']:+.2f}%"
        awin = "n/a" if not s.get("n") or pd.isna(s.get("avg_win_pct", float("nan"))) else f"{s['avg_win_pct']:+.3f}%"
        aloss = "n/a" if not s.get("n") or pd.isna(s.get("avg_loss_pct", float("nan"))) else f"{s['avg_loss_pct']:+.3f}%"
        d.add_row(Text(name, style=style), f"{s['n']}", wr,
                  Text(f"{s['net_pct']:+.2f}%", style=_pnl_color(s["net_pct"])),
                  Text(awin, style="bright_green"),
                  Text(aloss, style="red"),
                  Text(worst, style="red"))
    console.print(d)

    # --- Recap EV par trade (n / WR / AvgWin / AvgLoss / EV) ---
    ev_tab = Table(title="EV par trade", title_style="bold cyan", border_style="cyan",
                   header_style="bold", expand=False, show_lines=True)
    ev_tab.add_column("", style="bold")
    ev_tab.add_column("n", justify="right")
    ev_tab.add_column("WR", justify="right")
    ev_tab.add_column("AvgWin", justify="right")
    ev_tab.add_column("AvgLoss", justify="right")
    ev_tab.add_column("EV /trade", justify="right")
    for name, key, style in (("UP", "up", "bright_blue"), ("DOWN", "down", "bright_red")):
        s = stats[key]
        if not s.get("n"):
            continue
        ev = s.get("ev_pct", 0.0)
        ev_tab.add_row(
            Text(name, style=style), f"{s['n']}", f"{s['wr']*100:.2f}%",
            f"{abs(s.get('avg_win_pct', float('nan'))):.3f}%",
            f"{abs(s.get('avg_loss_pct', float('nan'))):.3f}%",
            Text(f"{ev:+.4f}%", style=_pnl_color(ev)))
    ev_all = stats["ev_per_trade_pct"]
    ev_tab.add_row(
        Text("Tous (headline)", style="bold"), f"{stats['n_trades']}",
        f"{stats['wr_net']*100:.2f}%",
        f"{stats.get('avg_win_pct_all', float('nan')):.3f}%",
        f"{stats.get('avg_loss_pct_all', float('nan')):.3f}%",
        Text(f"{ev_all:+.4f}%", style=_pnl_color(ev_all)))
    console.print(ev_tab)

    # --- Headline ---
    net = stats["net_total_pct"]
    headline = Text.assemble(
        ("P/L NET  ", "bold"),
        (f"{net:+.2f}%", f"bold {_pnl_color(net)}"),
        (f"   ({stats['pnl']:+.2f} sur {summary['initial_capital']:.0f})", "dim"),
        ("   sur ", "dim"),
        (f"{stats['n_trades']}", "bold"),
        (" trades, frais ", "dim"),
        (f"{(summary['entry_fee_pct'] + summary['exit_fee_pct']):.2f}% A/R", "dim"),
        ("   EV ", "dim"),
        (f"{stats['ev_per_trade_pct']:+.4f}%/trade", f"bold {_pnl_color(stats['ev_per_trade_pct'])}"),
    )
    console.print(Panel(headline, border_style=_pnl_color(net), padding=(0, 2)))

    console.print(Text.assemble(("  written  ", "dim"), ("logs/trades.csv", "white"),
                                (f"   ({stats['n_trades']} trades)", "dim")))
    console.print(Text.assemble(("  logged   ", "dim"),
                                (f"{TRADING_RUNS_CSV.relative_to(ROOT)}", "white"),
                                ("   (+1 line)", "dim")))


if __name__ == "__main__":
    main()
