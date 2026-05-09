import bisect
import csv
import time
from concurrent.futures import ThreadPoolExecutor

import requests

from config import (
    SYMBOL, INTERVAL, DAYS_BACK, OUTPUT_CSV,
    API_URL, KLINES_LIMIT, REQUEST_TIMEOUT, MAX_RETRIES, FETCH_WORKERS,
    INTERVAL_MS, MTF_INTERVALS,
    WARMUP, MAX_SEQ_LEN, BODY_RATIO_MIN,
    EMA_OBV_PERIOD, EMA_CLOSE_PERIODS, RSI_PERIOD,
    ATR_PERIOD, ATR_LONG_PERIOD, VOL_AVG_PERIOD,
    OBV_SLOPE_WINDOW, RSI_CHANGE_WINDOW,
    VOLATILITY_HIGH, VOLATILITY_LOW,
)

session = requests.Session()


# ============================================================
#  FETCH BITSTAMP (parametre par symbol et interval)
# ============================================================
def fetch_chunk(symbol, start_sec, end_sec, interval):
    """
    Appelle Bitstamp OHLC pour une paire + interval donnes et normalise au
    format kline Binance [open_time_ms, open, high, low, close, volume] pour
    garder le reste du pipeline inchange.
    """
    url = API_URL.format(pair=symbol)
    step_seconds = INTERVAL_MS[interval] // 1000
    params = {
        "step": step_seconds,
        "limit": KLINES_LIMIT,
        "start": start_sec,
        "end": end_sec,
    }
    for attempt in range(MAX_RETRIES):
        try:
            r = session.get(url, params=params, timeout=REQUEST_TIMEOUT)
            r.raise_for_status()
            ohlc = r.json().get("data", {}).get("ohlc", [])
            return [
                [
                    int(c["timestamp"]) * 1000,
                    c["open"], c["high"], c["low"], c["close"], c["volume"],
                ]
                for c in ohlc
            ]
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(2 ** attempt)
    return []


def fetch_all_klines(symbol, start_time, end_time, interval):
    step = INTERVAL_MS[interval] * KLINES_LIMIT
    windows = []
    current = start_time
    while current < end_time:
        w_end = min(current + step, end_time)
        windows.append((current // 1000, w_end // 1000))
        current += step

    all_klines = []
    with ThreadPoolExecutor(max_workers=FETCH_WORKERS) as executor:
        results = executor.map(lambda w: fetch_chunk(symbol, w[0], w[1], interval), windows)
        for chunk in results:
            all_klines.extend(chunk)

    # Dedup par timestamp (les fenetres Bitstamp peuvent se chevaucher aux bords)
    by_ts = {k[0]: k for k in all_klines}
    return sorted(by_ts.values(), key=lambda k: k[0])


def fetch_all_timeframes(symbol, start_time, end_time, intervals):
    """Sequentiel au niveau TF (chaque fetch_all_klines parallelise deja en interne)."""
    out = {}
    for iv in intervals:
        print(f"  Fetching {iv}...", flush=True)
        klines = fetch_all_klines(symbol, start_time, end_time, interval=iv)
        out[iv] = klines
        print(f"    {len(klines):,} bars.")
    return out


# ============================================================
#  HELPER NO-LOOKAHEAD
# ============================================================
def last_closed_index(open_times_sorted, t_decision_ms, interval_ms):
    """
    Retourne l'index de la derniere bougie *closed* avant t_decision_ms.
    Une bougie est closed ssi open_time + interval_ms <= t_decision_ms.
    Equivaut a : on cherche le plus grand i tel que open_times[i] <= t_decision_ms - interval_ms.
    Retourne -1 si aucune bougie n'est encore closed.
    """
    target = t_decision_ms - interval_ms
    return bisect.bisect_right(open_times_sorted, target) - 1


# ============================================================
#  INDICATEURS
# ============================================================
def compute_obv(closes, volumes):
    obv = [0.0]
    for i in range(1, len(closes)):
        if closes[i] > closes[i-1]:
            obv.append(obv[-1] + volumes[i])
        elif closes[i] < closes[i-1]:
            obv.append(obv[-1] - volumes[i])
        else:
            obv.append(obv[-1])
    return obv


def compute_ema(values, period):
    ema = [values[0]]
    alpha = 2 / (period + 1)
    for i in range(1, len(values)):
        ema.append(alpha * values[i] + (1 - alpha) * ema[-1])
    return ema


def compute_rsi(closes, period=14):
    rsi = [50.0] * len(closes)
    gains = [0.0]
    losses = [0.0]
    for i in range(1, len(closes)):
        change = closes[i] - closes[i-1]
        gains.append(max(change, 0))
        losses.append(max(-change, 0))
    if len(closes) < period + 1:
        return rsi
    avg_gain = sum(gains[1:period+1]) / period
    avg_loss = sum(losses[1:period+1]) / period
    for i in range(period, len(closes)):
        avg_gain = (avg_gain * (period-1) + gains[i]) / period
        avg_loss = (avg_loss * (period-1) + losses[i]) / period
        if avg_loss == 0:
            rsi[i] = 100
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100 - (100 / (1 + rs))
    return rsi


def compute_atr(highs, lows, closes, period=14):
    tr = [highs[0] - lows[0]]
    for i in range(1, len(closes)):
        tr.append(max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i-1]),
            abs(lows[i] - closes[i-1])
        ))
    atr = [tr[0]] * len(closes)
    if len(closes) >= period:
        atr_init = sum(tr[:period]) / period
        atr[period-1] = atr_init
        for i in range(period, len(closes)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    return atr


def compute_volume_avg(volumes, period=20):
    avg = [volumes[0]] * len(volumes)
    for i in range(period, len(volumes)):
        avg[i] = sum(volumes[i-period:i]) / period
    return avg


# ============================================================
#  EXTRACTION DES FEATURES
# ============================================================
def body_ratio(o, h, l, c):
    rng = h - l
    return abs(c - o) / rng if rng > 0 else 0


def upper_wick_ratio(o, h, l, c):
    rng = h - l
    body_top = max(o, c)
    return (h - body_top) / rng if rng > 0 else 0


def lower_wick_ratio(o, h, l, c):
    rng = h - l
    body_bot = min(o, c)
    return (body_bot - l) / rng if rng > 0 else 0


def get_direction(o, c):
    if c > o: return "up"
    if c < o: return "down"
    return "flat"


def _index_mtf(klines):
    """Decompose une serie de klines en arrays paralleles pour acces rapide."""
    return {
        "times":   [int(k[0]) for k in klines],
        "opens":   [float(k[1]) for k in klines],
        "highs":   [float(k[2]) for k in klines],
        "lows":    [float(k[3]) for k in klines],
        "closes":  [float(k[4]) for k in klines],
        "volumes": [float(k[5]) for k in klines],
    }


def extract_features(klines, klines_mtf, max_seq_len=MAX_SEQ_LEN):
    """
    Pour chaque sequence de N bougies de meme sens (N = 1 a max_seq_len),
    extrait les features 15m + OHLCV des MTF (no-lookahead) + le label.
    """
    n = len(klines)
    opens   = [float(k[1]) for k in klines]
    highs   = [float(k[2]) for k in klines]
    lows    = [float(k[3]) for k in klines]
    closes  = [float(k[4]) for k in klines]
    volumes = [float(k[5]) for k in klines]
    times   = [int(k[0]) for k in klines]

    directions = [get_direction(opens[i], closes[i]) for i in range(n)]

    # Indicateurs precalcules sur toute la serie 15m
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

    # Index des MTF pour acces rapide
    mtf = {iv: _index_mtf(kl) for iv, kl in klines_mtf.items()}

    warmup = WARMUP
    rows = []
    skipped_mtf = 0
    interval_15m_ms = INTERVAL_MS["15m"]

    for seq_len in range(1, max_seq_len + 1):
        for i in range(warmup, n - seq_len - 1):
            window = directions[i : i + seq_len]
            next_dir = directions[i + seq_len]

            if "flat" in window or next_dir == "flat":
                continue
            if len(set(window)) != 1:
                continue

            seq_dir = window[0]
            last_idx = i + seq_len - 1

            br_last = body_ratio(opens[last_idx], highs[last_idx], lows[last_idx], closes[last_idx])
            if br_last < BODY_RATIO_MIN:
                continue

            # ---------- DECISION TIMESTAMP T (no-lookahead) ----------
            T = times[last_idx] + interval_15m_ms

            # Lookup OHLCV de la derniere bougie closed avant T pour chaque MTF
            mtf_features = {}
            mtf_ok = True
            for iv in MTF_INTERVALS:
                d = mtf[iv]
                idx = last_closed_index(d["times"], T, INTERVAL_MS[iv])
                if idx < 0:
                    mtf_ok = False
                    break
                mtf_features[f"open_{iv}"]  = d["opens"][idx]
                mtf_features[f"high_{iv}"]  = d["highs"][idx]
                mtf_features[f"low_{iv}"]   = d["lows"][idx]
                mtf_features[f"close_{iv}"] = d["closes"][idx]
                mtf_features[f"vol_{iv}"]   = d["volumes"][idx]

            if not mtf_ok:
                skipped_mtf += 1
                continue

            # ---------- FEATURES 15m (inchangees) ----------
            f_seq_len = seq_len
            f_seq_dir = 1 if seq_dir == "up" else 0

            f_body_ratio_last = br_last
            f_body_ratio_avg = sum(
                body_ratio(opens[j], highs[j], lows[j], closes[j])
                for j in range(i, i + seq_len)
            ) / seq_len

            f_upper_wick = upper_wick_ratio(opens[last_idx], highs[last_idx], lows[last_idx], closes[last_idx])
            f_lower_wick = lower_wick_ratio(opens[last_idx], highs[last_idx], lows[last_idx], closes[last_idx])

            f_volume_ratio_last = volumes[last_idx] / vol_avg_20[last_idx] if vol_avg_20[last_idx] > 0 else 1
            f_volume_ratio_avg = sum(volumes[j] for j in range(i, i + seq_len)) / seq_len / vol_avg_20[last_idx] if vol_avg_20[last_idx] > 0 else 1

            move_pct = (closes[last_idx] - opens[i]) / opens[i] * 100
            f_move_pct = move_pct
            f_move_atr = abs(closes[last_idx] - opens[i]) / atr_14[last_idx] if atr_14[last_idx] > 0 else 0

            f_rsi = rsi_14[last_idx]
            f_rsi_change_5 = rsi_14[last_idx] - rsi_14[last_idx - RSI_CHANGE_WINDOW] if last_idx >= RSI_CHANGE_WINDOW else 0

            f_obv_slope_10 = (obv[last_idx] - obv[last_idx - OBV_SLOPE_WINDOW]) / abs(obv[last_idx - OBV_SLOPE_WINDOW]) if abs(obv[last_idx - OBV_SLOPE_WINDOW]) > 1e-9 else 0
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

            # ---------- LABEL ----------
            expected_reversal = "down" if seq_dir == "up" else "up"
            label = 1 if next_dir == expected_reversal else 0

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
            row["label"] = label
            rows.append(row)

    if skipped_mtf > 0:
        print(f"  (skipped {skipped_mtf:,} rows en debut d'historique : MTF data insuffisante)")

    return rows


# ============================================================
#  MAIN
# ============================================================
def main():
    end = int(time.time() * 1000)
    start = end - (DAYS_BACK * 24 * 60 * 60 * 1000)

    print(f"Fetching {DAYS_BACK} days of {INTERVAL} klines for {SYMBOL}...")
    klines_15m = fetch_all_klines(SYMBOL, start, end, interval=INTERVAL)
    print(f"  Fetched {len(klines_15m):,} 15m klines.\n")

    print(f"Fetching MTF klines : {MTF_INTERVALS}...")
    klines_mtf = fetch_all_timeframes(SYMBOL, start, end, MTF_INTERVALS)
    print()

    print("Extracting features and labels...")
    rows = extract_features(klines_15m, klines_mtf, max_seq_len=MAX_SEQ_LEN)
    print(f"Generated {len(rows):,} training examples.")

    by_seq = {1: [], 2: [], 3: [], 4: []}
    for r in rows:
        by_seq[r["seq_len"]].append(r["label"])
    for sl, labels in by_seq.items():
        if labels:
            wr = sum(labels) / len(labels) * 100
            print(f"  Seq {sl} bougies : {len(labels):>6,} exemples, WR baseline = {wr:.2f}%")

    if not rows:
        print("No rows generated. Exiting.")
        return

    fieldnames = list(rows[0].keys())
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nCSV exported : {OUTPUT_CSV}")
    print(f"Columns ({len(fieldnames)}) : {fieldnames}")


if __name__ == "__main__":
    main()
