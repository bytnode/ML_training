"""Detection des setups qualifiants au dernier index 15m + extraction des 42
features attendues par Aure.

MIRROR de backtest/data.py:extract_features (lignes 211-371). Si tu ajoutes
une feature dans extract_features, ajoute-la ici aussi. Pas de label calcule
ici (la bougie cible n'est pas encore fermee au moment de la decision).
"""
import os
import sys
import time

_BACKTEST_DIR = os.path.join(os.path.dirname(__file__), "..", "backtest")
sys.path.insert(0, os.path.abspath(_BACKTEST_DIR))

from data import (  # noqa: E402
    compute_obv, compute_ema, compute_rsi, compute_atr, compute_volume_avg,
    body_ratio, upper_wick_ratio, lower_wick_ratio, get_direction,
    last_closed_index, _index_mtf,
)
from config import (  # noqa: E402
    INTERVAL_MS, MTF_INTERVALS, MAX_SEQ_LEN, BODY_RATIO_MIN, WARMUP,
    EMA_CLOSE_PERIODS, RSI_PERIOD, ATR_PERIOD, ATR_LONG_PERIOD,
    VOL_AVG_PERIOD, OBV_SLOPE_WINDOW, RSI_CHANGE_WINDOW, EMA_OBV_PERIOD,
    VOLATILITY_HIGH, VOLATILITY_LOW,
)


def find_setups_at_latest(klines_15m: list, klines_mtf: dict) -> list:
    """Trouve tous les setups qualifiants dont le streak se termine a la
    derniere bougie 15m fermee (klines_15m[-1]).

    Args:
        klines_15m : liste triee de [open_time_ms, o, h, l, c, v]. Doit
                     contenir au moins WARMUP + MAX_SEQ_LEN bougies pour
                     stabiliser les indicateurs.
        klines_mtf : dict {iv: liste klines} pour iv in MTF_INTERVALS.

    Returns:
        list de tuples (seq_len, features_dict). features_dict contient les
        42 features attendues par Aure (sans timestamp ni label). Liste vide
        si aucun setup ne qualifie au dernier index.
    """
    n = len(klines_15m)
    if n < WARMUP + MAX_SEQ_LEN:
        return []

    # Decompose klines en arrays (mirror de extract_features — cast float/int
    # car Bitstamp renvoie les OHLCV comme strings dans le JSON)
    times = [int(k[0]) for k in klines_15m]
    opens = [float(k[1]) for k in klines_15m]
    highs = [float(k[2]) for k in klines_15m]
    lows = [float(k[3]) for k in klines_15m]
    closes = [float(k[4]) for k in klines_15m]
    volumes = [float(k[5]) for k in klines_15m]
    directions = [get_direction(o, c) for o, c in zip(opens, closes)]

    ema_p_short, ema_p_mid, ema_p_long = EMA_CLOSE_PERIODS

    # Indicateurs (mirror de extract_features)
    obv = compute_obv(closes, volumes)
    ema_obv_20 = compute_ema(obv, EMA_OBV_PERIOD)
    ema_close_20 = compute_ema(closes, ema_p_short)
    ema_close_50 = compute_ema(closes, ema_p_mid)
    ema_close_200 = compute_ema(closes, ema_p_long)
    rsi_14 = compute_rsi(closes, RSI_PERIOD)
    atr_14 = compute_atr(highs, lows, closes, ATR_PERIOD)
    atr_100_avg = compute_atr(highs, lows, closes, ATR_LONG_PERIOD)
    vol_avg_20 = compute_volume_avg(volumes, VOL_AVG_PERIOD)

    mtf = {iv: _index_mtf(kl) for iv, kl in klines_mtf.items()}

    interval_15m_ms = INTERVAL_MS["15m"]
    last_idx = n - 1
    setups = []

    for seq_len in range(1, MAX_SEQ_LEN + 1):
        i = n - seq_len  # streak de i a last_idx
        if i < WARMUP:
            continue

        window = directions[i:i + seq_len]
        if "flat" in window:
            continue
        if len(set(window)) != 1:
            continue

        seq_dir = window[0]

        br_last = body_ratio(opens[last_idx], highs[last_idx], lows[last_idx], closes[last_idx])
        if br_last < BODY_RATIO_MIN:
            continue

        # Decision time T (mirror de extract_features)
        T = times[last_idx] + interval_15m_ms

        # MTF features (no-lookahead via last_closed_index)
        mtf_features = {}
        mtf_ok = True
        for iv in MTF_INTERVALS:
            d = mtf[iv]
            idx = last_closed_index(d["times"], T, INTERVAL_MS[iv])
            if idx < 0:
                mtf_ok = False
                break
            mtf_features[f"open_{iv}"] = d["opens"][idx]
            mtf_features[f"high_{iv}"] = d["highs"][idx]
            mtf_features[f"low_{iv}"] = d["lows"][idx]
            mtf_features[f"close_{iv}"] = d["closes"][idx]
            mtf_features[f"vol_{iv}"] = d["volumes"][idx]

        if not mtf_ok:
            continue

        # Features 15m (mirror exact de extract_features)
        f_seq_dir = 1 if seq_dir == "up" else 0
        f_body_ratio_last = br_last
        f_body_ratio_avg = sum(
            body_ratio(opens[j], highs[j], lows[j], closes[j])
            for j in range(i, i + seq_len)
        ) / seq_len
        f_upper_wick = upper_wick_ratio(opens[last_idx], highs[last_idx], lows[last_idx], closes[last_idx])
        f_lower_wick = lower_wick_ratio(opens[last_idx], highs[last_idx], lows[last_idx], closes[last_idx])
        f_volume_ratio_last = volumes[last_idx] / vol_avg_20[last_idx] if vol_avg_20[last_idx] > 0 else 1
        f_volume_ratio_avg = (
            sum(volumes[j] for j in range(i, i + seq_len)) / seq_len / vol_avg_20[last_idx]
            if vol_avg_20[last_idx] > 0 else 1
        )
        f_move_pct = (closes[last_idx] - opens[i]) / opens[i] * 100
        f_move_atr = abs(closes[last_idx] - opens[i]) / atr_14[last_idx] if atr_14[last_idx] > 0 else 0
        f_rsi = rsi_14[last_idx]
        f_rsi_change_5 = (
            rsi_14[last_idx] - rsi_14[last_idx - RSI_CHANGE_WINDOW]
            if last_idx >= RSI_CHANGE_WINDOW else 0
        )
        f_obv_slope_10 = (
            (obv[last_idx] - obv[last_idx - OBV_SLOPE_WINDOW]) / abs(obv[last_idx - OBV_SLOPE_WINDOW])
            if abs(obv[last_idx - OBV_SLOPE_WINDOW]) > 1e-9 else 0
        )
        f_obv_vs_ema = 1 if obv[last_idx] > ema_obv_20[last_idx] else 0
        f_atr_ratio = atr_14[last_idx] / atr_100_avg[last_idx] if atr_100_avg[last_idx] > 0 else 1
        if f_atr_ratio > VOLATILITY_HIGH:
            f_volatility_state = 2
        elif f_atr_ratio < VOLATILITY_LOW:
            f_volatility_state = 0
        else:
            f_volatility_state = 1

        price = closes[last_idx]
        f_dist_ema20 = (price - ema_close_20[last_idx]) / ema_close_20[last_idx] * 100
        f_dist_ema50 = (price - ema_close_50[last_idx]) / ema_close_50[last_idx] * 100
        f_dist_ema200 = (price - ema_close_200[last_idx]) / ema_close_200[last_idx] * 100

        ts_seconds = times[last_idx] // 1000
        tm = time.gmtime(ts_seconds)
        f_hour_utc = tm.tm_hour
        f_day_of_week = tm.tm_wday
        f_is_weekend = 1 if f_day_of_week >= 5 else 0

        features = {
            "seq_len": seq_len,
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
        features.update(mtf_features)
        # Note: timestamp = times[last_idx] (open_time_ms de la derniere bougie
        # du streak), coherent avec extract_features. target_open_time = T.
        features["timestamp"] = times[last_idx]
        features["target_open_time"] = T

        setups.append((seq_len, features))

    return setups


if __name__ == "__main__":
    # Smoke-test : fetche les 250 dernieres bougies 15m + 50 MTF, appelle
    # find_setups_at_latest, et imprime le nombre de setups trouves.
    from fetch import fetch_warmup_window
    print("Fetching warmup window (250 15m + 50 MTF)...")
    klines = fetch_warmup_window(250, 50)
    print(f"  fetched: {sum(len(v) for v in klines.values())} candles total")
    klines_15m = klines.pop("15m")
    setups = find_setups_at_latest(klines_15m, klines)
    print(f"Setups qualifiants au dernier index : {len(setups)}")
    for seq_len, feat in setups:
        print(f"  seq_len={seq_len}, seq_dir={feat['seq_dir']}, "
              f"body_ratio_last={feat['body_ratio_last']:.3f}, "
              f"timestamp={feat['timestamp']}, target_open_time={feat['target_open_time']}")
