"""scorers.py -- construction du score composite de regime (OHLCV uniquement).

Trois axes normalises (z-score roulant borne) puis combines en somme ponderee :
  1. tendance   -- rendement log cumule sur la fenetre LOOKBACK_DAYS
  2. momentum   -- rendement log cumule sur une sous-fenetre plus courte
  3. force DMI  -- (+DI - -DI) de Wilder sur REGIME_DMI_TIMEFRAME, aligne sans lookahead

Aucune fuite : la tendance/momentum ne regardent que des bougies <= i (shift trailing),
et l'axe DMI n'utilise que la derniere bougie REGIME_DMI_TIMEFRAME deja cloturee au
moment du bar 1h (searchsorted), jamais une bougie en cours ou future.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from . import config_regime as _cfg
except ImportError:
    import config_regime as _cfg


def _find_db() -> Path:
    """Locate modele_down's SQLite candles DB (sibling package, one level up)."""
    down_dir = Path(__file__).resolve().parent.parent / "modele_down"
    for p in (down_dir / "db" / "btc_multi_tf.db", down_dir / "DB" / "btc_multi_tf.db"):
        if p.exists():
            return p
    raise FileNotFoundError(
        f"SQLite candles introuvable sous {down_dir} (db/ ou DB/). Lance modele_down/data.py."
    )


def _load_ohlc(timeframe: str) -> pd.DataFrame:
    """OHLC absolu reconstruit depuis `candles` (close absolu + *_ret), tout l'historique.
    Meme reconstruction que modele_team/main.py::_load_dmi_1h (close_prev = close/(1+close_ret))."""
    con = sqlite3.connect(str(_find_db()))
    try:
        c = pd.read_sql_query(
            "SELECT timestamp_ms, close, open_ret, close_ret, high_ret, low_ret "
            "FROM candles WHERE timeframe=? ORDER BY timestamp_ms",
            con, params=(timeframe,))
    finally:
        con.close()
    c = c.dropna(subset=["close", "close_ret", "high_ret", "low_ret"]).reset_index(drop=True)
    close = c["close"].to_numpy(dtype=float)
    close_prev = close / (1.0 + c["close_ret"].to_numpy(dtype=float))
    c["high"] = close_prev * (1.0 + c["high_ret"].to_numpy(dtype=float))
    c["low"] = close_prev * (1.0 + c["low_ret"].to_numpy(dtype=float))
    return c


def _wilder_dmi(high, low, close, n: int):
    """+DI / -DI de Wilder (longueur n). Duplique volontairement de
    modele_team/main.py::_wilder_dmi (meme formule) -- petit helper autonome,
    convention deja suivie dans ce repo pour _bucket_threshold (api.py / live.py)."""
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

    def _rma(x):
        return pd.Series(x).ewm(alpha=1.0 / n, adjust=False).mean().to_numpy()

    atr = _rma(tr)
    s_plus = _rma(plus_dm)
    s_minus = _rma(minus_dm)
    with np.errstate(divide="ignore", invalid="ignore"):
        plus[1:] = 100.0 * s_plus / atr
        minus[1:] = 100.0 * s_minus / atr
    return plus, minus


def _rolling_norm(x: pd.Series, window: int, clip: float) -> pd.Series:
    """Z-score roulant (rolling mean/std), clippe a +/-clip puis divise -> [-1, +1].
    Adaptatif dans le temps (une "grosse" tendance en 2013 et en 2025 sont comparees
    a l'echelle de LEUR propre voisinage), plutot qu'un seuil de magnitude fixe."""
    mean = x.rolling(window, min_periods=max(2, window // 4)).mean()
    std = x.rolling(window, min_periods=max(2, window // 4)).std()
    z = (x - mean) / std.replace(0.0, np.nan)
    return (z.clip(-clip, clip) / clip)


def build_raw_axes(lookback_days: int) -> pd.DataFrame:
    """DataFrame[timestamp_ms, trend_norm, momentum_norm, dmi_norm] aligne sur la
    grille 1h."""
    h1 = _load_ohlc("1h")
    lookback_bars = lookback_days * 24
    momentum_bars = max(1, int(lookback_bars * _cfg.MOMENTUM_FRACTION))
    norm_window = _cfg.NORM_WINDOW_DAYS * 24

    close = h1["close"]
    log_close = np.log(close)
    trend_raw = log_close - log_close.shift(lookback_bars)
    momentum_raw = log_close - log_close.shift(momentum_bars)

    # --- Axe DMI : calcule sur REGIME_DMI_TIMEFRAME, aligne sur la grille 1h SANS
    #     lookahead (dernier bar DMI_TF deja clos <= t, via searchsorted). ---
    dmi_tf = _load_ohlc(_cfg.REGIME_DMI_TIMEFRAME)
    plus_di, minus_di = _wilder_dmi(dmi_tf["high"], dmi_tf["low"], dmi_tf["close"], _cfg.REGIME_DMI_LENGTH)
    dmi_diff = (plus_di - minus_di)
    dmi_ts = dmi_tf["timestamp_ms"].to_numpy()
    h1_ts = h1["timestamp_ms"].to_numpy()
    idx = np.searchsorted(dmi_ts, h1_ts, side="right") - 1
    dmi_aligned = np.where(idx >= 0, dmi_diff[np.clip(idx, 0, len(dmi_ts) - 1)], np.nan)

    out = pd.DataFrame({
        "timestamp_ms": h1_ts,
        # Colonnes BRUTES (pre-normalisation) -- exposees pour regime/export_seed.py
        # (campagne symetrie 2026-07-09 : live_team a besoin de la queue glissante de
        # ces valeurs pour normaliser ses propres nouveaux bars incrementalement,
        # cf live_team/regime_live.py). Non consommees par api.py/compute_score.
        "trend_raw": trend_raw.to_numpy(),
        "momentum_raw": momentum_raw.to_numpy(),
        "dmi_raw": dmi_aligned,
        "trend_norm": _rolling_norm(trend_raw, norm_window, _cfg.NORM_CLIP).to_numpy(),
        "momentum_norm": _rolling_norm(momentum_raw, norm_window, _cfg.NORM_CLIP).to_numpy(),
        "dmi_norm": _rolling_norm(pd.Series(dmi_aligned), norm_window, _cfg.NORM_CLIP).to_numpy(),
    })
    return out


def compute_score(lookback_days: int) -> pd.DataFrame:
    """DataFrame[timestamp_ms, trend_norm, momentum_norm, dmi_norm, score].
    score = moyenne ponderee des 3 axes (poids config_regime), clippee a [-1, +1].
    Un axe NaN (warmup) compte comme 0 (neutre) dans la moyenne -- fait pencher les
    bars de tout debut d'historique vers neutre plutot que d'extrapoler depuis des
    axes incomplets ; score NaN uniquement si les 3 axes sont NaN en meme temps."""
    axes = build_raw_axes(lookback_days)
    w_sum = _cfg.WEIGHT_TREND + _cfg.WEIGHT_MOMENTUM + _cfg.WEIGHT_DMI
    score = (_cfg.WEIGHT_TREND * axes["trend_norm"].fillna(0.0)
             + _cfg.WEIGHT_MOMENTUM * axes["momentum_norm"].fillna(0.0)
             + _cfg.WEIGHT_DMI * axes["dmi_norm"].fillna(0.0)) / w_sum
    all_nan = axes[["trend_norm", "momentum_norm", "dmi_norm"]].isna().all(axis=1)
    axes["score"] = score.where(~all_nan, np.nan).clip(-1.0, 1.0)
    return axes
