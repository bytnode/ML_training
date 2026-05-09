"""Setup detection + 42-feature build at the just-closed 15m bar.

Mirrors backtest/modele_15m/data.extract_features exactly, restricted to the last
closed bar (last_idx = n - 2: the candle that just closed at the cycle's boundary)
and to seq_len in SEQ_LIVE.

Logic parity with the backtest is the contract — this module reuses the same
helpers (compute_*, last_closed_index, body_ratio, ...) imported from data.py.
"""
import time

import settings  # triggers sys.path injection

# Helpers from backtest/modele_15m/data.py
from data import (  # noqa: E402
    body_ratio, upper_wick_ratio, lower_wick_ratio, get_direction,
    compute_obv, compute_ema, compute_rsi, compute_atr, compute_volume_avg,
    last_closed_index, _index_mtf,
)

# Constants from backtest/modele_15m/config.py
from config import (  # noqa: E402
    INTERVAL_MS, MTF_INTERVALS,
    BODY_RATIO_MIN,
    EMA_OBV_PERIOD, EMA_CLOSE_PERIODS, RSI_PERIOD,
    ATR_PERIOD, ATR_LONG_PERIOD, VOL_AVG_PERIOD,
    OBV_SLOPE_WINDOW, RSI_CHANGE_WINDOW,
    VOLATILITY_HIGH, VOLATILITY_LOW,
)


def detect_qualifying_setups(opens, highs, lows, closes, last_idx,
                              seq_live=None) -> list[dict]:
    """Returns list of {seq_len, seq_dir} candidates qualifying at last_idx.

    Mirrors data.extract_features but evaluated at one fixed last_idx and only
    for seq_len values in seq_live. seq_dir is 'up' or 'down' (string, like the
    backtest internal repr).
    """
    if seq_live is None:
        seq_live = settings.SEQ_LIVE

    directions = [get_direction(opens[k], closes[k]) for k in range(len(closes))]
    candidates = []

    for seq_len in seq_live:
        i = last_idx - seq_len + 1
        if i < 0:
            continue
        window = directions[i:i + seq_len]
        if "flat" in window or len(set(window)) != 1:
            continue
        seq_dir = window[0]

        br_last = body_ratio(opens[last_idx], highs[last_idx], lows[last_idx], closes[last_idx])
        if br_last < BODY_RATIO_MIN:
            continue

        candidates.append({"seq_len": seq_len, "seq_dir": seq_dir, "last_idx": last_idx, "i": i})

    return candidates


def build_features(opens, highs, lows, closes, volumes, times, klines_mtf,
                    last_idx: int, seq_len: int, seq_dir: str) -> dict | None:
    """Build the 42-feature dict for one setup. Returns None if MTF data insufficient.

    Direct port of the per-row block in data.extract_features. seq_dir is 'up'/'down'.
    """
    i = last_idx - seq_len + 1
    interval_15m_ms = INTERVAL_MS["15m"]

    # Indicators precomputed on the full 15m series (same as data.extract_features).
    ema_p_short, ema_p_mid, ema_p_long = EMA_CLOSE_PERIODS
    obv = compute_obv(closes, volumes)
    ema_obv_20 = compute_ema(obv, EMA_OBV_PERIOD)
    ema_close_20 = compute_ema(closes, ema_p_short)
    ema_close_50 = compute_ema(closes, ema_p_mid)
    ema_close_200 = compute_ema(closes, ema_p_long)
    rsi_14 = compute_rsi(closes, RSI_PERIOD)
    atr_14 = compute_atr(highs, lows, closes, ATR_PERIOD)
    atr_100_avg = compute_atr(highs, lows, closes, ATR_LONG_PERIOD)
    vol_avg_20 = compute_volume_avg(volumes, VOL_AVG_PERIOD)

    # MTF lookup (no-lookahead via last_closed_index).
    T = times[last_idx] + interval_15m_ms
    mtf = {iv: _index_mtf(kl) for iv, kl in klines_mtf.items()}
    mtf_features = {}
    for iv in MTF_INTERVALS:
        d = mtf[iv]
        idx = last_closed_index(d["times"], T, INTERVAL_MS[iv])
        if idx < 0:
            return None
        mtf_features[f"open_{iv}"]  = d["opens"][idx]
        mtf_features[f"high_{iv}"]  = d["highs"][idx]
        mtf_features[f"low_{iv}"]   = d["lows"][idx]
        mtf_features[f"close_{iv}"] = d["closes"][idx]
        mtf_features[f"vol_{iv}"]   = d["volumes"][idx]

    # 15m features (identical formulas to data.extract_features).
    f_seq_len = seq_len
    f_seq_dir = 1 if seq_dir == "up" else 0

    f_body_ratio_last = body_ratio(opens[last_idx], highs[last_idx], lows[last_idx], closes[last_idx])
    f_body_ratio_avg = sum(
        body_ratio(opens[j], highs[j], lows[j], closes[j])
        for j in range(i, i + seq_len)
    ) / seq_len

    f_upper_wick = upper_wick_ratio(opens[last_idx], highs[last_idx], lows[last_idx], closes[last_idx])
    f_lower_wick = lower_wick_ratio(opens[last_idx], highs[last_idx], lows[last_idx], closes[last_idx])

    f_volume_ratio_last = volumes[last_idx] / vol_avg_20[last_idx] if vol_avg_20[last_idx] > 0 else 1
    f_volume_ratio_avg = (sum(volumes[j] for j in range(i, i + seq_len)) / seq_len
                          / vol_avg_20[last_idx]) if vol_avg_20[last_idx] > 0 else 1

    f_move_pct = (closes[last_idx] - opens[i]) / opens[i] * 100
    f_move_atr = abs(closes[last_idx] - opens[i]) / atr_14[last_idx] if atr_14[last_idx] > 0 else 0

    f_rsi = rsi_14[last_idx]
    f_rsi_change_5 = (rsi_14[last_idx] - rsi_14[last_idx - RSI_CHANGE_WINDOW]
                      if last_idx >= RSI_CHANGE_WINDOW else 0)

    if last_idx >= OBV_SLOPE_WINDOW and abs(obv[last_idx - OBV_SLOPE_WINDOW]) > 1e-9:
        f_obv_slope_10 = (obv[last_idx] - obv[last_idx - OBV_SLOPE_WINDOW]) / abs(obv[last_idx - OBV_SLOPE_WINDOW])
    else:
        f_obv_slope_10 = 0
    f_obv_vs_ema = 1 if obv[last_idx] > ema_obv_20[last_idx] else 0

    f_atr_ratio = atr_14[last_idx] / atr_100_avg[last_idx] if atr_100_avg[last_idx] > 0 else 1
    if f_atr_ratio > VOLATILITY_HIGH:
        f_volatility_state = 2
    elif f_atr_ratio < VOLATILITY_LOW:
        f_volatility_state = 0
    else:
        f_volatility_state = 1

    price = closes[last_idx]
    f_dist_ema20  = (price - ema_close_20[last_idx])  / ema_close_20[last_idx]  * 100
    f_dist_ema50  = (price - ema_close_50[last_idx])  / ema_close_50[last_idx]  * 100
    f_dist_ema200 = (price - ema_close_200[last_idx]) / ema_close_200[last_idx] * 100

    ts_seconds = times[last_idx] // 1000
    tm = time.gmtime(ts_seconds)
    f_hour_utc = tm.tm_hour
    f_day_of_week = tm.tm_wday
    f_is_weekend = 1 if f_day_of_week >= 5 else 0

    row = {
        "timestamp": times[last_idx],
        "seq_len": f_seq_len,
        "seq_dir": f_seq_dir,
        "body_ratio_last": f_body_ratio_last,
        "body_ratio_avg": f_body_ratio_avg,
        "upper_wick": f_upper_wick,
        "lower_wick": f_lower_wick,
        "volume_ratio_last": f_volume_ratio_last,
        "volume_ratio_avg": f_volume_ratio_avg,
        "move_pct": f_move_pct,
        "move_atr": f_move_atr,
        "rsi": f_rsi,
        "rsi_change_5": f_rsi_change_5,
        "obv_slope_10": f_obv_slope_10,
        "obv_vs_ema": f_obv_vs_ema,
        "atr_ratio": f_atr_ratio,
        "volatility_state": f_volatility_state,
        "dist_ema20": f_dist_ema20,
        "dist_ema50": f_dist_ema50,
        "dist_ema200": f_dist_ema200,
        "hour_utc": f_hour_utc,
        "day_of_week": f_day_of_week,
        "is_weekend": f_is_weekend,
    }
    row.update(mtf_features)
    return row
