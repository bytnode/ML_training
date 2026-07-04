"""
add_engineered_features.py — patch d'INGENIERIE de features pour le master 30m.

Objectif : enrichir le parquet avec un LARGE pool de candidats ORTHOGONAUX
(familles de signal distinctes), pour qu'un selecteur (optimize/select_features.py)
puisse composer un jeu final >= 13 features ou CHAQUE feature porte un signal
unique (ablation leave-one-out > bande de bruit => 0 feature "inutile").

Ne touche QUE le parquet (idempotent : drop ses propres colonnes puis recompute).
Reconstruit l'OHLC brut par timeframe depuis la SQLite `candles`
(open = (1+open_ret)*close_prev, etc.) — AUCUN re-fetch reseau.

NO-LOOKAHEAD (strict) :
  - base 30m : rolling/shift n'utilisent que des bougies <= i ; la bougie i
    ferme a T = open_i + 30m = decision time, donc close[i] est connu a T.
  - MTF (1h/4h/1d) : derniere bougie HTF *fermee* a T (searchsorted sur
    T - tf_step), identique a data.py::_add_lookup_features.
  - ETH : bougie ETH 30m concurrente (meme open_time) fermee a T, join ts exact.

Chaine de build :
    python data.py
    python add_eth_features.py
    python add_engineered_features.py     # <-- ce script
    python train.py

Toutes les colonnes ajoutees sont scale-invariantes (returns/ratios/oscillateurs
normalises). Le parquet peut contenir des colonnes en trop : la whitelist
FEATURE_WHITELIST (config.py) selectionne le jeu final ; les autres sont ignorees.
"""
from __future__ import annotations

import shutil
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DB_SQLITE = ROOT / "DB" / "btc_multi_tf.db"
MASTER_PARQUET = ROOT / "DB" / "btc_multi_tf_1h.parquet"
ETH_CSV = ROOT / "DB" / "eth_1h.csv"

INTERVAL_1H_MS = 60 * 60 * 1000
TF_STEP_MS = {"1h": 60 * 60 * 1000, "4h": 4 * 60 * 60 * 1000, "1d": 24 * 60 * 60 * 1000}

# ---------------------------------------------------------------------------
# Liste EXHAUSTIVE des colonnes produites par ce patch (sert a l'idempotence).
# Prefixe par famille pour lisibilite ; toutes uniques vs les 68 cols existantes.
# ---------------------------------------------------------------------------
NEW_FEATURE_COLS = [
    # --- Momentum multi-horizon (base 30m) ---
    "ret_6", "ret_12", "ret_24", "ret_48",
    # --- Tendance / moyennes mobiles ---
    "dist_ema_50", "dist_ema_100", "ema_slope_20", "dist_vwap_48",
    # --- Oscillateurs / mean-reversion ---
    "rsi_14", "rsi_6", "stoch_k_14", "cci_20", "zscore_close_48",
    # --- Volatilite ---
    "atr_pct_cur", "parkinson_20", "rv_ratio_fast", "range_pct", "range_rel_20",
    # --- Volume / flux ---
    "cmf_20", "vol_z_20", "force_index_13", "eom_14", "vol_price_corr_20", "signed_vol_mom_6",
    # --- Forme de bougie / microstructure ---
    "streak", "body_pos", "wick_asym", "body_signed_3",
    # --- Acceleration / regime ---
    "acc_8", "efficiency_ratio_20", "autocorr_12", "trend_slope_24",
    # --- MTF no-lookahead (1h / 4h / 1d) ---
    "rsi_14_1h", "dist_ema_20_1h", "ret_1h_6",
    "ret_4h_3", "rsi_14_4h",
    "ret_1d_1", "dist_ema_10_1d", "mtf_align",
    # --- Cross-asset ETH / BTC-ETH ---
    "eth_rsi_14", "eth_dist_ema_20", "eth_stoch_14", "eth_vol_ratio",
    "btc_eth_div", "eth_btc_ratio_mom_8",
]


# ===========================================================================
# Helpers indicateurs (numpy / pandas, vectorises, past-only)
# ===========================================================================
def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def _rsi(close: pd.Series, period: int) -> pd.Series:
    """RSI Wilder, renvoye normalise dans [-1, 1] = (rsi - 50) / 50."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    rsi = rsi.where(avg_loss != 0, 100.0)   # loss=0 -> RSI 100
    return (rsi - 50.0) / 50.0


def _streak(close_ret: pd.Series) -> np.ndarray:
    """Run length signe : +k apres k bougies vertes, -k apres k rouges."""
    sign = np.sign(close_ret.to_numpy())
    s = pd.Series(sign)
    grp = (s != s.shift()).cumsum()
    run = s.groupby(grp).cumcount().to_numpy() + 1
    return run * sign


def _stoch_k(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    hh = high.rolling(period, min_periods=period).max()
    ll = low.rolling(period, min_periods=period).min()
    rng = (hh - ll).replace(0, np.nan)
    return ((close - ll) / rng).clip(0, 1)


def _efficiency_ratio(close: pd.Series, period: int) -> pd.Series:
    """Kaufman efficiency ratio : |net change| / somme(|changes|) sur period."""
    net = (close - close.shift(period)).abs()
    vol = close.diff().abs().rolling(period, min_periods=period).sum()
    return net / vol.replace(0, np.nan)


def _trend_slope(close: pd.Series, period: int) -> pd.Series:
    """Pente OLS de close sur une fenetre `period`, normalisee par close (par bar)."""
    x = np.arange(period, dtype=float)
    x = x - x.mean()
    denom = (x * x).sum()
    c = close.to_numpy(dtype=float)
    n = len(c)
    out = np.full(n, np.nan)
    if n >= period and denom > 0:
        for i in range(period - 1, n):
            window = c[i - period + 1: i + 1]
            slope = (x * (window - window.mean())).sum() / denom
            out[i] = slope / window[-1] if window[-1] != 0 else np.nan
    return pd.Series(out, index=close.index)


# ===========================================================================
# Chargement OHLCV reconstruit depuis SQLite candles
# ===========================================================================
def load_tf_ohlcv(conn, timeframe: str) -> pd.DataFrame:
    """Reconstruit l'OHLCV brut d'un TF depuis `candles` :
        open  = (1 + open_ret)  * close_prev
        high  = (1 + high_ret)  * close_prev
        low   = (1 + low_ret)   * close_prev
        close = close (brut)
    """
    df = pd.read_sql_query(
        "SELECT timestamp_ms, close, volume, open_ret, high_ret, low_ret, close_ret "
        "FROM candles WHERE timeframe = ? ORDER BY timestamp_ms",
        conn, params=(timeframe,),
    ).reset_index(drop=True)
    close = df["close"].astype(float)
    close_prev = close.shift(1)
    df["open"] = (1.0 + df["open_ret"]) * close_prev
    df["high"] = (1.0 + df["high_ret"]) * close_prev
    df["low"] = (1.0 + df["low_ret"]) * close_prev
    return df


# ===========================================================================
# Features base 30m
# ===========================================================================
def compute_base_30m(df: pd.DataFrame) -> dict:
    """Renvoie {col: np.ndarray} aligne sur df (base 30m, par timestamp_ms)."""
    o, h, l, c = df["open"], df["high"], df["low"], df["close"].astype(float)
    vol = df["volume"].astype(float)
    cr = df["close_ret"].astype(float)
    cp = c.shift(1)
    out = {}

    # Momentum multi-horizon
    for n in (6, 12, 24, 48):
        out[f"ret_{n}"] = (c / c.shift(n) - 1.0).to_numpy()

    # Tendance / MA
    out["dist_ema_50"] = ((c - _ema(c, 50)) / c).to_numpy()
    out["dist_ema_100"] = ((c - _ema(c, 100)) / c).to_numpy()
    ema20 = _ema(c, 20)
    out["ema_slope_20"] = ((ema20 - ema20.shift(6)) / c).to_numpy()
    vwap48 = (c * vol).rolling(48, min_periods=48).sum() / vol.rolling(48, min_periods=48).sum()
    out["dist_vwap_48"] = ((c - vwap48) / c).to_numpy()

    # Oscillateurs / mean-reversion
    out["rsi_14"] = _rsi(c, 14).to_numpy()
    out["rsi_6"] = _rsi(c, 6).to_numpy()
    out["stoch_k_14"] = _stoch_k(h, l, c, 14).to_numpy()
    tp = (h + l + c) / 3.0
    sma_tp = tp.rolling(20, min_periods=20).mean()
    mad = (tp - sma_tp).abs().rolling(20, min_periods=20).mean()
    out["cci_20"] = ((tp - sma_tp) / (0.015 * mad.replace(0, np.nan))).clip(-300, 300).to_numpy() / 100.0
    sma48 = c.rolling(48, min_periods=48).mean()
    std48 = c.rolling(48, min_periods=48).std(ddof=0)
    out["zscore_close_48"] = ((c - sma48) / std48.replace(0, np.nan)).to_numpy()

    # Volatilite
    tr = pd.concat([(h - l).abs(), (h - cp).abs(), (l - cp).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / 14, adjust=False).mean()
    out["atr_pct_cur"] = (atr / c).to_numpy()
    log_hl = np.log((h / l).replace([np.inf, -np.inf], np.nan))
    park = (log_hl ** 2 / (4.0 * np.log(2.0)))
    out["parkinson_20"] = np.sqrt(park.rolling(20, min_periods=20).mean()).to_numpy()
    rv_fast = cr.rolling(10, min_periods=10).std(ddof=0)
    rv_slow = cr.rolling(50, min_periods=50).std(ddof=0)
    out["rv_ratio_fast"] = (rv_fast / rv_slow.replace(0, np.nan)).to_numpy()
    range_pct = ((h - l) / c)
    out["range_pct"] = range_pct.to_numpy()
    out["range_rel_20"] = (range_pct / range_pct.rolling(20, min_periods=20).mean()).to_numpy()

    # Volume / flux
    rng = (h - l).replace(0, np.nan)
    mfm = ((c - l) - (h - c)) / rng          # money flow multiplier [-1,1]
    mfv = mfm * vol
    out["cmf_20"] = (mfv.rolling(20, min_periods=20).sum()
                     / vol.rolling(20, min_periods=20).sum().replace(0, np.nan)).to_numpy()
    vmean = vol.rolling(20, min_periods=20).mean()
    vstd = vol.rolling(20, min_periods=20).std(ddof=0)
    out["vol_z_20"] = ((vol - vmean) / vstd.replace(0, np.nan)).to_numpy()
    fi = (c - cp) * vol
    fi_ema = _ema(fi.fillna(0.0), 13)
    out["force_index_13"] = (fi_ema / (c * vmean.replace(0, np.nan))).to_numpy()
    dist_moved = ((h + l) / 2.0) - ((h.shift(1) + l.shift(1)) / 2.0)
    box = (vol / 1e6) / rng
    out["eom_14"] = (dist_moved / box.replace(0, np.nan)).rolling(14, min_periods=14).mean().to_numpy()
    vol_chg = vol.pct_change().replace([np.inf, -np.inf], np.nan)
    out["vol_price_corr_20"] = cr.rolling(20, min_periods=20).corr(vol_chg).to_numpy()
    out["signed_vol_mom_6"] = (np.sign(cr) * (vol / vmean)).rolling(6, min_periods=6).mean().to_numpy()

    # Forme de bougie / microstructure
    out["streak"] = _streak(cr)
    out["body_pos"] = ((c - l) / rng).clip(0, 1).to_numpy()
    upper = ((h - np.maximum(o, c)) / rng).clip(0, 1)
    lower = ((np.minimum(o, c) - l) / rng).clip(0, 1)
    out["wick_asym"] = (upper - lower).to_numpy()
    body_signed = (c - o) / cp
    out["body_signed_3"] = body_signed.rolling(3, min_periods=3).sum().to_numpy()

    # Acceleration / regime
    out["acc_8"] = ((c - 2 * c.shift(8) + c.shift(16)) / c.shift(16)).to_numpy()
    out["efficiency_ratio_20"] = _efficiency_ratio(c, 20).to_numpy()
    out["autocorr_12"] = cr.rolling(12, min_periods=12).corr(cr.shift(1)).to_numpy()
    out["trend_slope_24"] = _trend_slope(c, 24).to_numpy()

    return out


# ===========================================================================
# Features MTF no-lookahead (1h / 4h / 1d)
# ===========================================================================
def _map_htf(values_htf: np.ndarray, times_htf: np.ndarray,
             decision_times: np.ndarray, tf_step_ms: int) -> np.ndarray:
    """Pour chaque decision_time T, prend la valeur HTF de la derniere bougie
    HTF *fermee* a T (open_htf <= T - tf_step)."""
    target = decision_times - tf_step_ms
    idx = np.searchsorted(times_htf, target, side="right") - 1
    out = np.full(len(decision_times), np.nan, dtype=float)
    valid = idx >= 0
    out[valid] = values_htf[idx[valid]]
    return out


def compute_htf(df30_times: np.ndarray, conn) -> dict:
    out = {}
    dt = df30_times + INTERVAL_1H_MS   # decision time T

    # --- 1h ---
    h1 = load_tf_ohlcv(conn, "1h")
    t1 = h1["timestamp_ms"].to_numpy(dtype=np.int64)
    c1 = h1["close"].astype(float)
    rsi1 = _rsi(c1, 14).to_numpy()
    dist_ema20_1h = ((c1 - _ema(c1, 20)) / c1).to_numpy()
    ret1h6 = (c1 / c1.shift(6) - 1.0).to_numpy()
    out["rsi_14_1h"] = _map_htf(rsi1, t1, dt, TF_STEP_MS["1h"])
    out["dist_ema_20_1h"] = _map_htf(dist_ema20_1h, t1, dt, TF_STEP_MS["1h"])
    out["ret_1h_6"] = _map_htf(ret1h6, t1, dt, TF_STEP_MS["1h"])
    sign_1h = _map_htf(np.sign(dist_ema20_1h), t1, dt, TF_STEP_MS["1h"])

    # --- 4h ---
    h4 = load_tf_ohlcv(conn, "4h")
    t4 = h4["timestamp_ms"].to_numpy(dtype=np.int64)
    c4 = h4["close"].astype(float)
    ret4h3 = (c4 / c4.shift(3) - 1.0).to_numpy()
    rsi4 = _rsi(c4, 14).to_numpy()
    dist_ema20_4h = ((c4 - _ema(c4, 20)) / c4).to_numpy()
    out["ret_4h_3"] = _map_htf(ret4h3, t4, dt, TF_STEP_MS["4h"])
    out["rsi_14_4h"] = _map_htf(rsi4, t4, dt, TF_STEP_MS["4h"])
    sign_4h = _map_htf(np.sign(dist_ema20_4h), t4, dt, TF_STEP_MS["4h"])

    # --- 1d ---
    d1 = load_tf_ohlcv(conn, "1d")
    td = d1["timestamp_ms"].to_numpy(dtype=np.int64)
    cd = d1["close"].astype(float)
    ret1d1 = (cd / cd.shift(1) - 1.0).to_numpy()
    dist_ema10_1d = ((cd - _ema(cd, 10)) / cd).to_numpy()
    out["ret_1d_1"] = _map_htf(ret1d1, td, dt, TF_STEP_MS["1d"])
    out["dist_ema_10_1d"] = _map_htf(dist_ema10_1d, td, dt, TF_STEP_MS["1d"])
    sign_1d = _map_htf(np.sign(dist_ema10_1d), td, dt, TF_STEP_MS["1d"])

    out["mtf_align"] = (np.nan_to_num(sign_1h) + np.nan_to_num(sign_4h)
                        + np.nan_to_num(sign_1d))
    return out


# ===========================================================================
# Cross-asset ETH / BTC-ETH
# ===========================================================================
def compute_eth_cross(df30: pd.DataFrame) -> dict:
    """ETH 1h (natif, DB/eth_1h.csv) -> oscillateurs ETH + divergence BTC-ETH,
    joins ts exact (concurrent). Pas de resample : deja a la meme granularite
    que le master BTC."""
    eth = pd.read_csv(ETH_CSV)
    h, l, c, v = eth["high"].astype(float), eth["low"].astype(float), eth["close"].astype(float), eth["volume"].astype(float)
    cr = (c - c.shift(1)) / c.shift(1)

    feat = pd.DataFrame({"timestamp_ms": eth["timestamp_ms"].to_numpy(dtype=np.int64)})
    feat["eth_rsi_14"] = _rsi(c, 14).to_numpy()
    feat["eth_dist_ema_20"] = ((c - _ema(c, 20)) / c).to_numpy()
    feat["eth_stoch_14"] = _stoch_k(h, l, c, 14).to_numpy()
    vmean = v.rolling(20, min_periods=20).mean()
    feat["eth_vol_ratio"] = (v / vmean.replace(0, np.nan)).to_numpy()
    feat["eth_close_ret"] = cr.to_numpy()

    eth_map = {int(ts): i for i, ts in enumerate(feat["timestamp_ms"].to_numpy(dtype=np.int64))}
    btc_ts = df30["timestamp_ms"].to_numpy(dtype=np.int64)
    idx = np.array([eth_map.get(int(ts), -1) for ts in btc_ts])
    valid = idx >= 0

    out = {}
    for col in ("eth_rsi_14", "eth_dist_ema_20", "eth_stoch_14", "eth_vol_ratio"):
        src = feat[col].to_numpy(dtype=float)
        arr = np.full(len(df30), np.nan)
        arr[valid] = src[idx[valid]]
        out[col] = arr

    eth_cr = np.full(len(df30), np.nan)
    eth_cr[valid] = feat["eth_close_ret"].to_numpy(dtype=float)[idx[valid]]
    btc_cr = df30["close_ret"].astype(float).to_numpy()
    out["btc_eth_div"] = btc_cr - eth_cr

    btc_close = df30["close"].astype(float).to_numpy()
    eth_close = np.full(len(df30), np.nan)
    eth_close[valid] = c.to_numpy(dtype=float)[idx[valid]]
    ratio = pd.Series(eth_close / btc_close)
    out["eth_btc_ratio_mom_8"] = (ratio / ratio.shift(8) - 1.0).to_numpy()
    return out


# ===========================================================================
# Main
# ===========================================================================
def main() -> int:
    if not MASTER_PARQUET.exists():
        print(f"ERREUR : master parquet introuvable : {MASTER_PARQUET}\nLance d'abord : python data.py + python add_eth_features.py")
        return 1
    if not DB_SQLITE.exists():
        print(f"ERREUR : SQLite introuvable : {DB_SQLITE}")
        return 1
    if not ETH_CSV.exists():
        print(f"ERREUR : ETH CSV introuvable : {ETH_CSV}")
        return 1

    print("=== add_engineered_features.py (1h) ===")
    master = pd.read_parquet(MASTER_PARQUET)
    print(f"Master 1h : {len(master):,} rows x {len(master.columns)} cols")

    present = [c for c in NEW_FEATURE_COLS if c in master.columns]
    if present:
        print(f"  features deja presentes ({len(present)}), reconstruction.")
        master = master.drop(columns=present)
    else:
        bak = MASTER_PARQUET.with_suffix(MASTER_PARQUET.suffix + ".preeng.bak")
        if not bak.exists():
            shutil.copy2(MASTER_PARQUET, bak)
            print(f"  backup one-off : {bak.name}")

    conn = sqlite3.connect(DB_SQLITE)
    try:
        df30 = load_tf_ohlcv(conn, "1h")
        print(f"  OHLCV 30m reconstruit depuis SQLite : {len(df30):,} rows")
        base = compute_base_30m(df30)
        htf = compute_htf(df30["timestamp_ms"].to_numpy(dtype=np.int64), conn)
        cross = compute_eth_cross(df30)
    finally:
        conn.close()

    all_feats = {**base, **htf, **cross}
    feat_df = pd.DataFrame(all_feats)
    feat_df["timestamp_ms"] = df30["timestamp_ms"].to_numpy(dtype=np.int64)

    missing = [c for c in NEW_FEATURE_COLS if c not in feat_df.columns]
    if missing:
        print(f"  ATTENTION colonnes non produites : {missing}")
    extra = [c for c in feat_df.columns if c not in NEW_FEATURE_COLS and c != "timestamp_ms"]
    if extra:
        feat_df = feat_df.drop(columns=extra)

    merged = master.merge(feat_df, on="timestamp_ms", how="left")
    assert len(merged) == len(master), "merge a change le nombre de lignes !"

    tail = merged.tail(len(merged) - 2880)
    print(f"\n  {len(NEW_FEATURE_COLS)} features ajoutees. NaN% (post-warmup) :")
    for c in NEW_FEATURE_COLS:
        if c in merged.columns:
            pct = float(tail[c].isna().mean()) * 100
            flag = "  <-- ELEVE" if pct > 15 else ""
            print(f"    {c:<22} {pct:5.1f}%{flag}")

    merged.to_parquet(MASTER_PARQUET, index=False)
    print(f"\n-> {MASTER_PARQUET.name} : {len(merged):,} rows x {len(merged.columns)} cols")
    print("=== Termine. Lance maintenant : python optimize/select_features.py ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
