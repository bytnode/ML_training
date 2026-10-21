"""
data.py — pipeline complet pour modele_updown.

Un seul fichier qui (VERSION 30m : bougie de base 30m, contexte fin 5m) :
  PARITE FEATURES 15min : ce pipeline produit EXACTEMENT le meme jeu de
  features que le modele 15min/polymarket/modele_down (porte verbatim du
  builder runtime live/modele_15m/live_helpers/features_down.py), juste avec
  la timeframe de base 15m -> 30m. Le contexte fin reste **5m** (la sous-bougie
  5m qui ferme a T+30m, open_time = T+25m), les MTF restent 30m/1h/4h/1d, et
  on ajoute la microstructure 15min (dist_to_high/low_50, autocorr_24, acc_4,
  volume_rel_30d). La parite finale au niveau MODELE est garantie par la
  whitelist config.FEATURE_WHITELIST appliquee dans train.py.

  1. Fetch OHLCV BTC/USD depuis Bitstamp sur 6 timeframes (5m, 15m, 30m, 1h, 4h, 1d)
  2. Calcule les 9 features normalisées par TF + persiste close & volume bruts
  3. Construit le master 30m (.parquet) avec UNIQUEMENT des features normalisées :
       - features de base 30m
       - 4  features 5m (contexte fin, lookup no-lookahead, 5m close-aligned a T+30m)
       - features MTF (30m/1h/4h/1d, no-lookahead)
       -  7 features Tier 1 dérivées : dist_ema_20, return_3, return_96,
            hour_sin, hour_cos, upper_wick, lower_wick
       -  9 features Tier 2          : macd, macd_signal, macd_hist,
            bb_position, bb_width, mfi_norm, rv_20, rv_100, rv_ratio
       -  5 features microstructure  : dist_to_high_50, dist_to_low_50,
            autocorr_24, acc_4, volume_rel_30d

  Les valeurs BRUTES close/volume sont persistées dans la SQLite (table candles)
  mais ne sont JAMAIS écrites dans les sorties (parquet ou CSV). Phase 4 les
  charge depuis SQLite, les utilise pour calculer MACD/BB/MFI/microstructure,
  puis les drop avant l'écriture du master 30m.

Ce fichier remplace l'ancienne chaîne :
  data.py + fetch_and_add_5m.py + add_tier1_features.py + drop_redundant_features.py
  (+ patch_body_signed.py qui n'est plus nécessaire car body_signed n'est jamais écrit).
Les anciens scripts sont laissés sur disque mais ne sont plus utiles.

Usage :
    cd backtest/modele_updown/modele_down
    python data.py                 # ~3-5 min : 6 TFs en parallèle + agrégation
    python train.py                # entraîne sur le master CSV produit

Schéma SQLite (table `candles`) :
    timeframe         TEXT
    timestamp_ms      INTEGER
    timestamp         TEXT
    close             REAL  -- BRUT (USD)
    volume            REAL  -- BRUT
    open_ret          REAL  -- (open  - close_prev) / close_prev
    high_ret          REAL
    low_ret           REAL
    close_ret         REAL
    volume_ratio      REAL  -- volume / SMA(volume, 20)
    body_ratio_close  REAL  -- |close-open|/(high-low) ∈ [0,1]
    body_ratio_avg    REAL  -- SMA(body_ratio_close, 14)
    obv_pressure      REAL  -- (obv - obv_{-10}) / Σvol_{-10..0} ∈ [-1,1]
    atr_pct           REAL  -- atr_Wilder(14) / close
    PRIMARY KEY (timeframe, timestamp_ms)

NB : on ne stocke pas `body_signed` (= close_ret − open_ret, doublon mesuré r ≥ 0.998).
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from config import ATR_LONG_PERIOD, VOLATILITY_HIGH, VOLATILITY_LOW


# ============================================================================
# Configuration
# ============================================================================

BITSTAMP_OHLC_URL = "https://www.bitstamp.net/api/v2/ohlc/{pair}/"

DEFAULT_PAIR = "btcusd"
DEFAULT_DAYS_BACK = 5000
DEFAULT_DB_PATH = "DB/btc_multi_tf.db"

# (step_ms, step_sec)
TIMEFRAMES: dict[str, tuple[int, int]] = {
    "5m":  ( 5 * 60 * 1000,           5 * 60),
    "15m": (15 * 60 * 1000,          15 * 60),
    "30m": (30 * 60 * 1000,          30 * 60),
    "1h":  (60 * 60 * 1000,          60 * 60),
    "4h":  ( 4 * 60 * 60 * 1000,  4 * 60 * 60),
    "1d":  (24 * 60 * 60 * 1000, 24 * 60 * 60),
}

BITSTAMP_MAX_LIMIT = 1000
WORKERS_PER_TF = 8
SLEEP_BETWEEN_CALLS_SEC = 0.1
MAX_RETRIES = 5
RETRY_BACKOFF_BASE_SEC = 2.0

# Périodes des features normalisées par TF
ATR_PERIOD            = 14
BODY_RATIO_AVG_WINDOW = 14
VOLUME_AVG_WINDOW     = 20
OBV_PRESSURE_WINDOW   = 10

# Périodes Tier 1 (15m) — RSI / EMA / Returns dropped sont ignorés
EMA_PERIOD_FAST  = 12   # MACD fast
EMA_PERIOD_SLOW  = 26   # MACD slow
EMA_PERIOD_SIG   = 9    # MACD signal
EMA_PERIOD_DIST  = 20   # dist_ema_20 (kept)
RETURN_PERIODS_KEPT = (3, 96)  # return_3 (45min), return_96 (24h) — return_12 dropped

# Périodes Tier 2 (15m)
BB_PERIOD     = 20
BB_NSIGMA     = 2
MFI_PERIOD    = 14
RV_SHORT_WINDOW = 20
RV_LONG_WINDOW  = 100

INTERVAL_5M_MS  = 5  * 60 * 1000
INTERVAL_15M_MS = 15 * 60 * 1000
INTERVAL_30M_MS = 30 * 60 * 1000
INTERVAL_1H_MS  = 60 * 60 * 1000   # bougie de base = 1h

DB_WRITE_LOCK = threading.Lock()


# ============================================================================
# MTF feature filter — ne pas calculer ce qui sera droppé
# ============================================================================
# Décisions issues de drop_redundant_features.py (Tier 1 audit).
#  - PURE: body_signed* (r ≥ 0.99 avec close_ret*)
#  - STRONG: rsi_14_norm, return_12, dist_ema_50, dist_ema_200,
#            atr_pct_30m (r=0.97 atr_pct), atr_pct_1h (r=0.90 atr_pct_4h)
#  - WEAK: open_ret_4h, obv_pressure_1h, high_ret_1h, obv_pressure_30m,
#          body_ratio_close_4h, low_ret_4h, high_ret_4h, close_ret_1d,
#          body_ratio_avg_1h
#
# Pour chaque MTF on liste explicitement les colonnes À CALCULER (post-drop).
# PARITE 15min : on porte le sous-ensemble exact de colonnes joint par le
# builder runtime live/modele_15m/live_helpers/features_down.py :
#   5m  -> high_ret, low_ret, close_ret, obv_pressure
#   30m -> low_ret, close_ret
#   1h  -> low_ret, close_ret
# (4h/1d sont conserves pour diagnostic mais seront exclus du modele par la
#  whitelist FEATURE_WHITELIST ; on garde les memes sous-ensembles que le 15min
#  d'origine pour ne pas casser les CSV per-TF.)
MTF_KEPT_COLS: dict[str, list[str]] = {
    # 5m : CONTEXTE FIN du modele 30m — la bougie 5m qui ferme a T+30m
    #      (open_time = T+25m). Memes 4 colonnes que le 15min (close-align 5m).
    "5m":  ["high_ret", "low_ret", "close_ret", "obv_pressure"],
    # 30m : MTF = la base elle-meme. close_ret_30m ~= close_ret, low_ret_30m ~=
    #       low_ret (voulu pour la parite de colonnes avec le 15min).
    "30m": ["low_ret", "close_ret"],
    # 1h : low_ret, close_ret (parite 15min : low_ret_1h, close_ret_1h)
    "1h":  ["low_ret", "close_ret"],
    # 4h : conserve pour diagnostic (exclu du modele par la whitelist)
    "4h":  ["close_ret", "volume_ratio",
            "body_ratio_avg", "obv_pressure", "atr_pct"],
    # 1d : conserve pour diagnostic (exclu du modele par la whitelist)
    "1d":  ["open_ret", "high_ret", "low_ret",
            "volume_ratio", "body_ratio_close", "body_ratio_avg",
            "obv_pressure", "atr_pct"],
}

# Toutes les colonnes normalisées calculées en phase 2 (avant filtrage MTF).
ALL_NORM_COLS = ["open_ret", "high_ret", "low_ret", "close_ret",
                 "volume_ratio", "body_ratio_close", "body_ratio_avg",
                 "obv_pressure", "atr_pct"]


# ============================================================================
# Phase 1 — Fetch Bitstamp avec retry
# ============================================================================

def fetch_klines_window(pair: str, step_sec: int, start_sec: int, end_sec: int) -> list[dict]:
    """Une fenêtre Bitstamp avec retry exponentiel."""
    url = BITSTAMP_OHLC_URL.format(pair=pair)
    params = {"step": step_sec, "limit": BITSTAMP_MAX_LIMIT,
              "start": start_sec, "end": end_sec}

    last_err: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, timeout=20)
            if resp.status_code == 429 or resp.status_code >= 500:
                raise requests.HTTPError(f"HTTP {resp.status_code}: {resp.text[:200]}")
            resp.raise_for_status()
            return resp.json().get("data", {}).get("ohlc", [])
        except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as e:
            last_err = e
            if attempt == MAX_RETRIES - 1:
                break
            wait = RETRY_BACKOFF_BASE_SEC * (2 ** attempt)
            print(f"    [step={step_sec}s] {type(e).__name__} "
                  f"(attempt {attempt + 1}/{MAX_RETRIES}), backoff {wait:.0f}s")
            time.sleep(wait)

    raise RuntimeError(f"Failed after {MAX_RETRIES} retries: {last_err}")


def klines_to_rows(raw: list[dict]) -> list[dict]:
    return [
        {
            "timestamp_ms": int(c["timestamp"]) * 1000,
            "open":   float(c["open"]),
            "high":   float(c["high"]),
            "low":    float(c["low"]),
            "close":  float(c["close"]),
            "volume": float(c["volume"]),
        }
        for c in raw
    ]


def init_db(db_path: Path) -> None:
    """Repart à zéro : DROP des deux tables, recrée le schéma."""
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript("""
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=NORMAL;

            DROP TABLE IF EXISTS candles_raw;
            DROP TABLE IF EXISTS candles;

            CREATE TABLE candles_raw (
                timeframe     TEXT    NOT NULL,
                timestamp_ms  INTEGER NOT NULL,
                open          REAL    NOT NULL,
                high          REAL    NOT NULL,
                low           REAL    NOT NULL,
                close         REAL    NOT NULL,
                volume        REAL    NOT NULL,
                PRIMARY KEY (timeframe, timestamp_ms)
            );

            CREATE TABLE candles (
                timeframe         TEXT    NOT NULL,
                timestamp_ms      INTEGER NOT NULL,
                timestamp         TEXT    NOT NULL,
                close             REAL,
                volume            REAL,
                open_ret          REAL,
                high_ret          REAL,
                low_ret           REAL,
                close_ret         REAL,
                volume_ratio      REAL,
                body_ratio_close  REAL,
                body_ratio_avg    REAL,
                obv_pressure      REAL,
                atr_pct           REAL,
                PRIMARY KEY (timeframe, timestamp_ms)
            );

            CREATE INDEX idx_candles_timestamp ON candles (timestamp_ms);
            CREATE INDEX idx_candles_timeframe ON candles (timeframe);
        """)
        conn.commit()
    finally:
        conn.close()


def insert_raw_chunk(db_path: Path, timeframe: str, rows: list[dict]) -> int:
    if not rows:
        return 0

    payload = [
        (timeframe, r["timestamp_ms"], r["open"], r["high"],
         r["low"], r["close"], r["volume"])
        for r in rows
    ]

    with DB_WRITE_LOCK:
        conn = sqlite3.connect(db_path, timeout=30)
        try:
            conn.executemany(
                """
                INSERT OR IGNORE INTO candles_raw
                    (timeframe, timestamp_ms, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                payload,
            )
            conn.commit()
        finally:
            conn.close()

    return len(payload)


def fetch_one_window(pair: str, timeframe: str, step_sec: int,
                     start_sec: int, end_sec: int, db_path: Path) -> int:
    raw = fetch_klines_window(pair, step_sec, start_sec, end_sec)
    if not raw:
        time.sleep(SLEEP_BETWEEN_CALLS_SEC)
        return 0
    rows = klines_to_rows(raw)
    n = insert_raw_chunk(db_path, timeframe, rows)
    time.sleep(SLEEP_BETWEEN_CALLS_SEC)
    return n


def fetch_timeframe(pair: str, timeframe: str, days_back: int, db_path: Path) -> int:
    step_ms, step_sec = TIMEFRAMES[timeframe]
    window_span_ms = step_ms * BITSTAMP_MAX_LIMIT

    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = int((datetime.now(timezone.utc) - timedelta(days=days_back)).timestamp() * 1000)

    d = datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc)
    print(f"  [{timeframe}] Démarrage depuis {d}")

    windows: list[tuple[int, int]] = []
    w_start_ms = start_ms
    while w_start_ms < now_ms:
        w_end_ms = min(w_start_ms + window_span_ms - 1, now_ms)
        windows.append((w_start_ms // 1000, w_end_ms // 1000))
        w_start_ms += window_span_ms
    print(f"  [{timeframe}] {len(windows)} fenêtres à fetch ({WORKERS_PER_TF} workers)")

    total_inserted = 0
    completed = 0
    with ThreadPoolExecutor(max_workers=WORKERS_PER_TF) as ex:
        futures = {
            ex.submit(fetch_one_window, pair, timeframe, step_sec, ws, we, db_path): (ws, we)
            for ws, we in windows
        }
        for fut in as_completed(futures):
            n = fut.result()
            total_inserted += n
            completed += 1
            if completed % 10 == 0 or completed == len(windows):
                pct = 100.0 * completed / len(windows)
                print(f"  [{timeframe}] {completed}/{len(windows)} fenêtres "
                      f"({pct:.0f}%), {total_inserted:,} bougies insérées")

    print(f"  [{timeframe}] Terminé : {total_inserted:,} bougies brutes insérées.")
    return total_inserted


# ============================================================================
# Phase 2 — Calcul des features normalisées + persistance close/volume bruts
# ============================================================================

def compute_normalized_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Entrée : df trié par timestamp_ms avec colonnes
        [timestamp_ms, open, high, low, close, volume].
    Sortie : df avec [timestamp_ms, timestamp, close, volume, + 9 cols normalisées].

    `close` et `volume` sont gardés bruts pour permettre le calcul de MACD/BB/MFI
    en phase 4. features.py les exclura de feature_cols via NON_FEATURE_COLS.
    """
    df = df.copy()
    close_prev = df["close"].shift(1)

    out = pd.DataFrame({
        "timestamp_ms": df["timestamp_ms"].astype("int64"),
        "timestamp":    pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
                          .dt.strftime("%Y-%m-%d %H:%M:%S"),
        # Bruts (utilisés par MACD/BB/MFI en phase 4, exclus de feature_cols).
        "close":  df["close"].astype(float),
        "volume": df["volume"].astype(float),
        # 4 OHLC normalisés vs close précédent.
        "open_ret":  (df["open"]  - close_prev) / close_prev,
        "high_ret":  (df["high"]  - close_prev) / close_prev,
        "low_ret":   (df["low"]   - close_prev) / close_prev,
        "close_ret": (df["close"] - close_prev) / close_prev,
    })

    # volume / SMA(volume, 20)
    vol_avg = df["volume"].rolling(VOLUME_AVG_WINDOW, min_periods=VOLUME_AVG_WINDOW).mean()
    out["volume_ratio"] = df["volume"] / vol_avg

    # body_ratio_close ∈ [0,1] + SMA
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    body_ratio_close = ((df["close"] - df["open"]).abs() / rng).clip(0, 1)
    out["body_ratio_close"] = body_ratio_close
    out["body_ratio_avg"]   = body_ratio_close.rolling(BODY_RATIO_AVG_WINDOW).mean()

    # obv_pressure ∈ [-1, 1]
    direction = np.sign(df["close"].diff().fillna(0))
    obv = (direction * df["volume"]).cumsum()
    obv_diff = obv - obv.shift(OBV_PRESSURE_WINDOW)
    vol_sum = df["volume"].rolling(OBV_PRESSURE_WINDOW, min_periods=OBV_PRESSURE_WINDOW).sum()
    out["obv_pressure"] = obv_diff / vol_sum.clip(lower=1)

    # atr_pct = ATR_Wilder(14) / close
    high = df["high"]
    low = df["low"]
    tr = pd.concat([
        (high - low).abs(),
        (high - close_prev).abs(),
        (low  - close_prev).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / ATR_PERIOD, adjust=False).mean()
    out["atr_pct"] = atr / df["close"]

    return out


def process_timeframe_to_candles(db_path: Path, timeframe: str) -> int:
    """Lit candles_raw du TF, calcule features, INSERT dans candles.

    Cas spécial 5m (contexte fin du modele 30m) : on calcule les features sur la
    série COMPLÈTE (les rolling SMA20/EWMA14/etc. ont besoin de toutes les bougies
    contiguës), puis on FILTRE pour ne garder que la 5m close-aligned avec chaque
    30m (open_time = T + 25min). Ce sont exactement les 5m qui ferment a
    T + 30min = close du 30m. On passe ainsi de ~1.4M rows à ~470k rows en DB &
    CSV, sans dégrader les features (qui ont vu la série complète au calcul).
    Ceci reproduit le contexte fin 5m du modele 15min (parite features).
    """
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(
            "SELECT timestamp_ms, open, high, low, close, volume "
            "FROM candles_raw WHERE timeframe = ? ORDER BY timestamp_ms",
            conn, params=(timeframe,),
        )
    finally:
        conn.close()

    if df.empty:
        return 0

    feat = compute_normalized_features(df)

    if timeframe == "5m":
        # Garde les 5m dont open_time tombe à T + 25min pour un T multiple de 30min.
        # Ce sont exactement les 5m qui ferment à T + 30min = close du 30m.
        offset_ms = feat["timestamp_ms"].to_numpy(dtype=np.int64) % INTERVAL_1H_MS
        keep_mask = offset_ms == (55 * 60 * 1000)
        n_before = len(feat)
        feat = feat.loc[keep_mask].reset_index(drop=True)
        print(f"    [5m] filtre close-align : {n_before:,} -> {len(feat):,} rows "
              f"(garde 1/6, les autres ne sont pas utilisées par le 30m)")

    def _f(v):
        return None if pd.isna(v) else float(v)

    payload = [
        (
            timeframe,
            int(r.timestamp_ms),
            r.timestamp,
            _f(r.close), _f(r.volume),
            _f(r.open_ret), _f(r.high_ret), _f(r.low_ret), _f(r.close_ret),
            _f(r.volume_ratio),
            _f(r.body_ratio_close), _f(r.body_ratio_avg),
            _f(r.obv_pressure),
            _f(r.atr_pct),
        )
        for r in feat.itertuples(index=False)
    ]

    with DB_WRITE_LOCK:
        conn = sqlite3.connect(db_path, timeout=30)
        try:
            conn.executemany(
                """
                INSERT INTO candles
                    (timeframe, timestamp_ms, timestamp,
                     close, volume,
                     open_ret, high_ret, low_ret, close_ret,
                     volume_ratio, body_ratio_close, body_ratio_avg,
                     obv_pressure, atr_pct)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                payload,
            )
            conn.commit()
        finally:
            conn.close()

    return len(feat)


def drop_raw_table(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript("DROP TABLE IF EXISTS candles_raw; VACUUM;")
        conn.commit()
    finally:
        conn.close()


# ============================================================================
# Phase 3 — Per-TF Parquet (debug uniquement, pas utilisé par train.py)
# ============================================================================

def export_timeframe_parquet(db_path: Path, timeframe: str, parquet_path: Path) -> int:
    """Export per-TF Parquet (debug uniquement) — features normalisées seulement.
    Les valeurs brutes close/volume restent en SQLite (utilisées par phase 4
    pour calculer MACD/BB/MFI), elles ne sont jamais écrites dans ces exports.
    """
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(
            """
            SELECT timestamp_ms, timestamp,
                   open_ret, high_ret, low_ret, close_ret,
                   volume_ratio, body_ratio_close, body_ratio_avg,
                   obv_pressure, atr_pct
            FROM candles WHERE timeframe = ? ORDER BY timestamp_ms
            """,
            conn, params=(timeframe,),
        )
    finally:
        conn.close()

    df.to_parquet(parquet_path, index=False)
    return len(df)


# ============================================================================
# Phase 4 — Master 15m feature DataFrame (Tier 0 + 5m + MTF + Tier 1 + Tier 2)
# ============================================================================

def _load_tf(db_path: Path, timeframe: str) -> pd.DataFrame:
    """Charge un TF depuis candles, trié par timestamp_ms."""
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(
            """
            SELECT timestamp_ms, timestamp, close, volume,
                   open_ret, high_ret, low_ret, close_ret,
                   volume_ratio, body_ratio_close, body_ratio_avg,
                   obv_pressure, atr_pct
            FROM candles WHERE timeframe = ? ORDER BY timestamp_ms
            """,
            conn, params=(timeframe,),
        )
    finally:
        conn.close()
    return df.reset_index(drop=True)


def _add_lookup_features(df_15m: pd.DataFrame, df_other: pd.DataFrame,
                         tf_label: str, tf_step_ms: int,
                         kept_cols: list[str],
                         base_step_ms: int = INTERVAL_1H_MS) -> int:
    """
    Pour chaque row de base 30m, ajoute les colonnes `kept_cols` du TF supérieur
    (ou contexte fin 15m), suffixées par tf_label. Lookup no-lookahead :
    on prend la dernière bougie `df_other` *fermée* à T = open_30m + 30m.

    Pour 15m (contexte fin) : T - 15m est l'open de la dernière 15m fermée à T
    (= la 15m qui ferme exactement à T, open_time = T - 15m = open_30m + 15m).
    Pour 1h+ : T - tf_step est l'open de la dernière bougie supérieure fermée à T.
    """
    times_15m = df_15m["timestamp_ms"].to_numpy(dtype=np.int64)
    times_oth = df_other["timestamp_ms"].to_numpy(dtype=np.int64)
    decision_times = times_15m + base_step_ms
    target = decision_times - tf_step_ms
    indices = np.searchsorted(times_oth, target, side="right") - 1
    valid = indices >= 0

    n = len(df_15m)
    n_added = 0
    for col in kept_cols:
        new_col = f"{col}_{tf_label}"
        values = np.full(n, np.nan, dtype=float)
        if valid.any():
            src = df_other[col].to_numpy(dtype=float)
            values[valid] = src[indices[valid]]
        df_15m[new_col] = values
        n_added += 1
    return n_added


def _ema(values: np.ndarray, period: int) -> np.ndarray:
    """EMA standard alpha = 2/(period+1). Renvoie un array de même longueur."""
    n = len(values)
    if n == 0:
        return np.empty(0, dtype=float)
    out = np.empty(n, dtype=float)
    alpha = 2.0 / (period + 1)
    out[0] = values[0]
    for i in range(1, n):
        out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return out


def _add_tier1_derived(df: pd.DataFrame) -> int:
    """
    Ajoute les 7 features Tier 1 KEPT (post-drop) :
        dist_ema_20, return_3, return_96,
        hour_sin, hour_cos, upper_wick, lower_wick.

    Note : rsi_14_norm, return_12, dist_ema_50, dist_ema_200 sont DROP (audit).
    Les calculs sont scale-invariants : on utilise close brut directement.
    """
    n_added = 0
    n = len(df)
    close = df["close"].to_numpy(dtype=float)

    # dist_ema_20 = (close - EMA20) / close
    ema_20 = _ema(close, EMA_PERIOD_DIST)
    df["dist_ema_20"] = (close - ema_20) / close
    n_added += 1

    # Multi-bar returns (close ratio)
    for period in RETURN_PERIODS_KEPT:
        col = f"return_{period}"
        ret = np.full(n, np.nan, dtype=float)
        if n > period:
            ret[period:] = close[period:] / close[:-period] - 1.0
        df[col] = ret
        n_added += 1

    # Time of day cyclique
    ts_sec = (df["timestamp_ms"].to_numpy() // 1000)
    seconds_in_day = ts_sec % 86400
    hours = seconds_in_day / 3600.0
    df["hour_sin"] = np.sin(2 * np.pi * hours / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hours / 24)
    n_added += 2

    # Wicks (calculs scale-invariants, fonctionnent sur les % comme sur les prix)
    h = df["high_ret"].to_numpy(dtype=float)
    l = df["low_ret"].to_numpy(dtype=float)
    o = df["open_ret"].to_numpy(dtype=float)
    c = df["close_ret"].to_numpy(dtype=float)
    rng = h - l
    rng_safe = np.where(rng > 1e-12, rng, np.nan)
    df["upper_wick"] = np.clip((h - np.maximum(o, c)) / rng_safe, 0.0, 1.0)
    df["lower_wick"] = np.clip((np.minimum(o, c) - l)  / rng_safe, 0.0, 1.0)
    n_added += 2

    return n_added


def _add_tier2(df: pd.DataFrame) -> int:
    """
    Ajoute les 9 features Tier 2 (15m only) :
        macd, macd_signal, macd_hist
        bb_position, bb_width
        mfi_norm
        rv_20, rv_100, rv_ratio

    Tous les calculs utilisent close & volume BRUTS de phase 2.
    """
    n_added = 0
    n = len(df)
    close = df["close"].to_numpy(dtype=float)
    high  = (df["high_ret"].to_numpy(dtype=float) + 1.0) * np.roll(close, 1)
    low   = (df["low_ret"].to_numpy(dtype=float)  + 1.0) * np.roll(close, 1)
    # Première row : pas de close_prev → high/low non-définis. Mis à NaN.
    high[0] = np.nan
    low[0]  = np.nan
    volume = df["volume"].to_numpy(dtype=float)

    # ---- MACD : (EMA12 - EMA26) / close, signal = EMA9 sur macd ----
    ema_fast = _ema(close, EMA_PERIOD_FAST)
    ema_slow = _ema(close, EMA_PERIOD_SLOW)
    macd = (ema_fast - ema_slow) / close
    macd_signal = _ema(macd, EMA_PERIOD_SIG)
    macd_hist = macd - macd_signal
    df["macd"]        = macd
    df["macd_signal"] = macd_signal
    df["macd_hist"]   = macd_hist
    n_added += 3

    # ---- Bollinger Bands ----
    close_s = pd.Series(close)
    bb_mid = close_s.rolling(BB_PERIOD, min_periods=BB_PERIOD).mean()
    bb_std = close_s.rolling(BB_PERIOD, min_periods=BB_PERIOD).std(ddof=0)
    band_width_abs = (2 * BB_NSIGMA) * bb_std            # = 4 * std
    bb_lower = bb_mid - BB_NSIGMA * bb_std
    # bb_position ∈ ~[0,1] dans la bande, hors bande possible (pas clampé).
    df["bb_position"] = ((close_s - bb_lower) / band_width_abs).to_numpy()
    df["bb_width"]    = (band_width_abs / bb_mid).to_numpy()
    n_added += 2

    # ---- MFI (Money Flow Index) ----
    typical = (high + low + close) / 3.0
    raw_mf  = typical * volume
    delta_tp = np.diff(typical, prepend=np.nan)
    pos_flow = np.where(delta_tp > 0, raw_mf, 0.0)
    neg_flow = np.where(delta_tp < 0, raw_mf, 0.0)
    pos_s = pd.Series(pos_flow).rolling(MFI_PERIOD, min_periods=MFI_PERIOD).sum()
    neg_s = pd.Series(neg_flow).rolling(MFI_PERIOD, min_periods=MFI_PERIOD).sum()
    # Évite div par zéro : si neg_flow == 0 → MFI = 100 ; si pos_flow == 0 → MFI = 0.
    pos_arr = pos_s.to_numpy()
    neg_arr = neg_s.to_numpy()
    mfi = np.full(n, np.nan, dtype=float)
    valid_mask = ~(np.isnan(pos_arr) | np.isnan(neg_arr))
    both_zero = valid_mask & (pos_arr == 0) & (neg_arr == 0)
    only_neg = valid_mask & (pos_arr == 0) & (neg_arr > 0)
    only_pos = valid_mask & (pos_arr > 0) & (neg_arr == 0)
    normal   = valid_mask & (pos_arr > 0) & (neg_arr > 0)
    mfi[both_zero] = 50.0
    mfi[only_neg]  = 0.0
    mfi[only_pos]  = 100.0
    ratio = np.where(normal, pos_arr / np.where(neg_arr == 0, 1, neg_arr), np.nan)
    mfi[normal] = 100.0 - 100.0 / (1.0 + ratio[normal])
    df["mfi_norm"] = (mfi - 50.0) / 50.0
    n_added += 1

    # ---- Realized volatility (sur close_ret) ----
    close_ret = df["close_ret"].to_numpy(dtype=float)
    cr_s = pd.Series(close_ret)
    rv_short = cr_s.rolling(RV_SHORT_WINDOW, min_periods=RV_SHORT_WINDOW).std(ddof=0)
    rv_long  = cr_s.rolling(RV_LONG_WINDOW,  min_periods=RV_LONG_WINDOW).std(ddof=0)
    df["rv_20"]    = rv_short.to_numpy()
    df["rv_100"]   = rv_long.to_numpy()
    # rv_ratio : NaN si rv_long est 0 ou NaN
    rv_long_safe = np.where(rv_long.to_numpy() > 0, rv_long.to_numpy(), np.nan)
    df["rv_ratio"] = rv_short.to_numpy() / rv_long_safe
    n_added += 3

    return n_added


def _add_microstructure(df: pd.DataFrame) -> int:
    """Ajoute les 5 features microstructure du modele 15min (parite features) :
        dist_to_high_50, dist_to_low_50, autocorr_24, acc_4, volume_rel_30d.

    Port EXACT du builder runtime
    live/modele_15m/live_helpers/features_down.py (phase 4 + volume_rel_30d).
    Les fenetres sont gardees EN NOMBRE DE BOUGIES identiques au 15min :
      - 50 bougies pour dist_to_high/low_50
      - 24 bougies pour autocorr_24
      - lag 4/8 bougies pour acc_4
      - 2880 bougies pour volume_rel_30d (= la meme fenetre en NB de bougies que
        le 15min ; sur base 30m cela couvre ~60 jours calendaires)
    """
    n_added = 0
    close = df["close"].to_numpy(dtype=float)
    close_s = pd.Series(close)

    # dist_to_high_50 / dist_to_low_50 : reconstruit high/low absolus a partir
    # des rets normalises (high = (1+high_ret)*close_prev), rolling 50.
    high_50 = pd.Series((df["high_ret"] + 1) * close_s.shift(1)).rolling(50).max()
    low_50  = pd.Series((df["low_ret"]  + 1) * close_s.shift(1)).rolling(50).min()
    df["dist_to_high_50"] = (close - high_50.to_numpy()) / close
    df["dist_to_low_50"]  = (close - low_50.to_numpy())  / close
    n_added += 2

    # autocorr_24 : correlation glissante (24) entre close_ret et close_ret lag 1.
    cr = df["close_ret"]
    df["autocorr_24"] = cr.rolling(24).corr(cr.shift(1))
    n_added += 1

    # acc_4 : "acceleration" = (close - 2*close[-4] + close[-8]) / close[-8].
    df["acc_4"] = ((close_s - 2 * close_s.shift(4) + close_s.shift(8))
                   / close_s.shift(8)).to_numpy()
    n_added += 1

    # volume_rel_30d : volume / SMA(volume, 2880). 2880 = meme NB de bougies
    # que le 15min (porte de modele_up).
    vol = df["volume"].astype(float)
    df["volume_rel_30d"] = (vol / vol.rolling(2880, min_periods=2880).mean()).to_numpy()
    n_added += 1

    return n_added


def build_master_15m(db_path: Path, out_path: Path) -> tuple[int, int]:
    """
    Construit le DataFrame final 15m (Tier 0 + 5m + MTF + Tier 1 + Tier 2)
    et l'écrit selon l'extension de out_path :
      - .parquet -> df.to_parquet (recommandé, ~5-10x plus petit + lecture plus rapide)
      - .csv     -> df.to_csv     (texte, plus gros mais lisible)
    Renvoie (n_rows, n_cols).
    """
    print("\n--- Phase 4/5 : construction du master 30m DataFrame ---")
    t0 = time.time()

    # 4.1 — base 30m (df_15m garde son nom de variable historique = le DataFrame de base)
    df_15m = _load_tf(db_path, "1h")
    print(f"  [base 30m] {len(df_15m):,} rows × {len(df_15m.columns)} cols (avec close+volume)")

    # 4.2 — 5m lookups (CONTEXTE FIN, no-lookahead) : la 5m qui ferme a T+30m.
    df_5ctx = _load_tf(db_path, "5m")
    if not df_5ctx.empty:
        n_5 = _add_lookup_features(df_15m, df_5ctx, "5m",
                                   INTERVAL_5M_MS, MTF_KEPT_COLS["5m"])
        print(f"  [5m lookup] +{n_5} cols (contexte fin)")
    else:
        print(f"  [5m lookup] skip (table 5m vide)")

    # 4.3 — MTF lookups (30m, 1h, 4h, 1d) — colonnes filtrées post-drop.
    # 30m = la base elle-meme (close_ret_30m ~= close_ret), voulu pour la parite
    # de colonnes avec le modele 15min.
    for tf in ["30m", "1h", "4h", "1d"]:
        df_oth = _load_tf(db_path, tf)
        if df_oth.empty:
            print(f"  [{tf} lookup] skip (table vide)")
            continue
        step_ms, _ = TIMEFRAMES[tf]
        n = _add_lookup_features(df_15m, df_oth, tf, step_ms, MTF_KEPT_COLS[tf])
        print(f"  [{tf} lookup] +{n} cols (post-drop)")

    # 4.4 — Tier 1 derived (7 cols)
    n_t1 = _add_tier1_derived(df_15m)
    print(f"  [Tier 1 derived] +{n_t1} cols (dist_ema_20, return_3/96, hour_sin/cos, wicks)")

    # 4.5 — Tier 2 (9 cols : MACD ×3, BB ×2, MFI ×1, rv ×3)
    n_t2 = _add_tier2(df_15m)
    print(f"  [Tier 2] +{n_t2} cols (MACD ×3, BB ×2, MFI ×1, rv ×3)")

    # 4.5b — Microstructure (5 cols : dist_to_high/low_50, autocorr_24, acc_4,
    # volume_rel_30d) — parite features 15min.
    n_ms = _add_microstructure(df_15m)
    print(f"  [Microstructure] +{n_ms} cols (dist_to_high/low_50, autocorr_24, acc_4, volume_rel_30d)")

    # 4.6 — Precalcule des colonnes derivees (label, volatility_state, is_weekend).
    # Avant : features.py les calculait au load-time, ce qui le forcait a relire
    # atr_pct et open_ret depuis la SQLite. Maintenant on les ecrit directement
    # dans le parquet -> features.py n'a plus besoin de la SQLite du tout.

    # is_weekend (UTC, samedi/dimanche)
    ts_dt = pd.to_datetime(df_15m["timestamp_ms"], unit="ms", utc=True)
    df_15m["is_weekend"] = (ts_dt.dt.dayofweek >= 5).astype(int)

    # volatility_state : ratio atr_pct / SMA(atr_pct, ATR_LONG_PERIOD)
    #   0 = low (< VOLATILITY_LOW), 1 = mid (default), 2 = high (> VOLATILITY_HIGH)
    atr_long = df_15m["atr_pct"].rolling(window=ATR_LONG_PERIOD, min_periods=20).mean()
    atr_ratio = df_15m["atr_pct"] / atr_long
    vol_state = np.where(atr_ratio > VOLATILITY_HIGH, 2,
                np.where(atr_ratio < VOLATILITY_LOW, 0, 1))
    vol_state = np.where(np.isnan(atr_ratio.to_numpy()), 1, vol_state)
    df_15m["volatility_state"] = vol_state.astype(int)

    # label : 1 si bougie i+1 VERTE (close[i+1] > open[i+1])
    # NaN pour : flat current, flat next, derniere row (pas de "next"),
    # ou current/next NaN (warmup). features.py drop les NaN avant le training.
    cr = df_15m["close_ret"].to_numpy()
    or_ = df_15m["open_ret"].to_numpy()
    cr_next = np.concatenate([cr[1:], [np.nan]])
    or_next = np.concatenate([or_[1:], [np.nan]])
    invalid = (
        (cr == or_) | (cr_next == or_next)        # flat current ou next
        | np.isnan(cr) | np.isnan(or_)
        | np.isnan(cr_next) | np.isnan(or_next)
    )
    label = np.where(invalid, np.nan, (cr_next > or_next).astype(float))
    df_15m["label"] = label
    n_valid_label = int(np.sum(~np.isnan(label)))
    print(f"  [derived] +3 cols : is_weekend, volatility_state, label "
          f"({n_valid_label:,} labels valides / {len(df_15m):,})")

    # 4.7 — Drop des colonnes inutiles APRES le precalcul.
    # - close/volume : valeurs brutes USD, fuite d'echelle pour XGBoost.
    # - atr_pct, open_ret : ont servi au precalcul ci-dessus, plus utiles ensuite.
    # Toutes restent disponibles dans candles (SQLite) pour diagnostics.
    df_15m = df_15m.drop(columns=["close", "volume", "atr_pct", "open_ret"])

    # 4.7 — Export selon l'extension
    suffix = out_path.suffix.lower()
    if suffix == ".parquet":
        df_15m.to_parquet(out_path, index=False)
    elif suffix == ".csv":
        df_15m.to_csv(out_path, index=False)
    else:
        raise ValueError(
            f"Extension non supportée pour {out_path} : '{suffix}'. "
            f"Attendu : .parquet ou .csv."
        )
    elapsed = time.time() - t0
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"  -> {out_path.name} : {len(df_15m):,} rows x {len(df_15m.columns)} cols, "
          f"{size_mb:.1f} MB ({elapsed:.1f}s)")

    return len(df_15m), len(df_15m.columns)


# ============================================================================
# Pipeline complet
# ============================================================================

def run(pair: str, days_back: int, db_path: Path) -> None:
    pair_norm = pair.lower()

    print(f"=== modele_updown — pipeline data.py (Tier 0 + Tier 1 + Tier 2) ===")
    print(f"Pair        : {pair_norm}")
    print(f"Période     : {days_back} jours en arrière")
    print(f"Timeframes  : {', '.join(TIMEFRAMES.keys())}  ({WORKERS_PER_TF} workers/TF)")
    print(f"DB          : {db_path}")

    init_db(db_path)

    # ---- Phase 1 — fetch parallèle vers candles_raw ----
    t0 = time.time()
    print(f"\n--- Phase 1/5 : téléchargement OHLCV bruts (-> candles_raw) ---")
    totals: dict[str, int] = {}
    for tf in TIMEFRAMES:
        try:
            totals[tf] = fetch_timeframe(pair_norm, tf, days_back, db_path)
        except Exception as e:
            print(f"  [{tf}] ECHEC : {e}")
            totals[tf] = 0
    elapsed_fetch = time.time() - t0
    print(f"\nFetch terminé en {elapsed_fetch:.1f}s")

    # ---- Phase 2 — features normalisées + close/volume bruts ----
    print(f"\n--- Phase 2/5 : normalisation des features (-> candles, avec close+volume) ---")
    t1 = time.time()
    for tf in TIMEFRAMES:
        n = process_timeframe_to_candles(db_path, tf)
        print(f"  [{tf}] {n:,} bougies normalisées et insérées")
    elapsed_feat = time.time() - t1
    print(f"Normalisation terminée en {elapsed_feat:.1f}s")

    # ---- Phase 3 — DROP candles_raw + per-TF Parquet (debug) ----
    print(f"\n--- Phase 3/5 : nettoyage + export per-TF Parquet ---")
    t2 = time.time()
    drop_raw_table(db_path)
    print(f"  candles_raw supprimée + VACUUM")
    csv_dir = db_path.parent
    db_stem = db_path.stem
    for tf in TIMEFRAMES:
        # "_debug" suffix : evite la collision avec le master "{db_stem}_1h.parquet"
        # (le TF "1h" est a la fois un export debug ET le nom du master -- meme
        # extension .parquet pour les deux depuis la conversion CSV->parquet).
        parquet_path = csv_dir / f"{db_stem}_{tf}_debug.parquet"
        n = export_timeframe_parquet(db_path, tf, parquet_path)
        print(f"  [{tf}] {n:,} lignes -> {parquet_path.name}")
    elapsed_csv = time.time() - t2
    print(f"Export per-TF terminé en {elapsed_csv:.1f}s")

    # ---- Phase 4 — master 15m parquet (= dataset d'entrée de train.py) ----
    # Parquet : ~5-10x plus petit que CSV + lecture plus rapide. features.py
    # auto-détecte l'extension via Path.suffix, donc DB_PATH dans config.py
    # peut pointer indifféremment sur .parquet ou .csv.
    master_path = csv_dir / f"{db_stem}_1h.parquet"
    n_rows, n_cols = build_master_15m(db_path, master_path)

    # ---- Phase 5 — récap ----
    print(f"\n--- Phase 5/5 : récap ---")
    db_size_mb = db_path.stat().st_size / (1024 * 1024)
    total_elapsed = elapsed_fetch + elapsed_feat + elapsed_csv
    print(f"Bougies brutes téléchargées : {sum(totals.values()):,}")
    print(f"DB                          : {db_path.resolve()} ({db_size_mb:.1f} MB)")
    print(f"Master 30m (parquet)        : {master_path.resolve()}")
    print(f"  {n_rows:,} rows x {n_cols} cols (base 30m + 4 ctx 5m + MTF 30m/1h/4h/1d "
          f"+ 7 Tier1 + 9 Tier2 + 5 microstructure ; close/volume bruts en SQLite "
          f"uniquement). Parite features 15min via FEATURE_WHITELIST dans train.py.")

    conn = sqlite3.connect(db_path)
    summary = pd.read_sql_query(
        """
        SELECT timeframe, COUNT(*) AS n_rows,
               MIN(timestamp) AS first_ts, MAX(timestamp) AS last_ts
        FROM candles GROUP BY timeframe
        ORDER BY CASE timeframe
            WHEN '5m'  THEN 1 WHEN '15m' THEN 2 WHEN '30m' THEN 3
            WHEN '1h'  THEN 4 WHEN '4h'  THEN 5 WHEN '1d'  THEN 6 END
        """,
        conn,
    )
    conn.close()

    print(f"\nRécap par timeframe :")
    print(summary.to_string(index=False))

    print(f"\n=== Terminé. Lance maintenant : python train.py ===")


# ============================================================================
# CLI
# ============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="modele_updown — fetch + features (Tier 0 + 1 + 2) en un seul script."
    )
    p.add_argument("--pair", default=DEFAULT_PAIR,
                   help=f"Code paire Bitstamp en minuscules (défaut : {DEFAULT_PAIR}).")
    p.add_argument("--days", type=int, default=DEFAULT_DAYS_BACK,
                   help=f"Jours d'historique (défaut : {DEFAULT_DAYS_BACK})")
    p.add_argument("--db", type=Path, default=Path(DEFAULT_DB_PATH),
                   help=f"Chemin SQLite (défaut : {DEFAULT_DB_PATH})")
    return p.parse_args()


# Patches de features appliques AUTOMATIQUEMENT a la fin du build, pour qu'UNE
# seule commande `python data.py` produise la DB 100% prete (toutes features
# comprises). Chaque patch est lance comme s'il etait lance a la main (subprocess
# -> meme comportement exact, zero changement du calcul des features).
# Chaine canonique modele_up (cf. add_alpha_features.py) : add_eth (cree/charge le
# parquet ETH lu par add_alpha) -> add_engineered (stoch_k_14, rsi_6) -> add_alpha
# (ema_compression_20, bars_since_low_50, ret_skew_24). Les 3 sont requis : le
# modele UP attend stoch_k_14/rsi_6 (add_engineered) ET les 3 alpha (add_alpha).
_FEATURE_PATCHES = ("add_eth_features.py", "add_engineered_features.py", "add_alpha_features.py")


def _apply_feature_patches() -> None:
    import subprocess
    here = Path(__file__).resolve().parent
    for patch in _FEATURE_PATCHES:
        script = here / patch
        if not script.exists():
            print(f"[data.py] patch features absent, ignore : {patch}")
            continue
        print(f"[data.py] application du patch features : {patch} ...")
        subprocess.run([sys.executable, str(script)], cwd=str(here), check=True)
    print("[data.py] DB prete a 100% (donnees + toutes les features).")


def main() -> int:
    args = parse_args()
    args.db.parent.mkdir(parents=True, exist_ok=True)
    run(pair=args.pair, days_back=args.days, db_path=args.db)
    _apply_feature_patches()
    return 0


if __name__ == "__main__":
    sys.exit(main())
