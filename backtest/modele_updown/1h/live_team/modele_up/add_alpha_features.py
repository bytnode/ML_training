"""
add_alpha_features.py — patch d'INGENIERIE de NOUVELLES features (alpha) pour
le master 1h de modele_up.

Cree 17 features candidates ABSENTES des 114 colonnes existantes, choisies pour
etre ORTHOGONALES au pool sature (technique BTC+ETH) et adaptees a la prediction
de la DIRECTION de la prochaine bougie 1h (UP). Familles :

  - Saisonnalite calendaire (ancrage prix-temps) :
      ret_since_dopen   = close / open_du_jour_UTC - 1     (derive intraday)
      ret_since_wopen   = close / open_de_la_semaine - 1   (derive hebdo)
  - Structure de la veille (breakout / range) :
      dist_pday_high    = (close - high_veille) / close
      dist_pday_low     = (close - low_veille)  / close
      pday_range_pos    = position de close dans le range de la veille [0,1]
  - Order-flow / pression (proxy depuis OHLCV) :
      signed_vol_24/6   = somme(vol*signe(ret)) / somme(vol) sur 24 / 6 bars
      vol_hod_ratio     = vol / moyenne EXPANDING du volume a CETTE heure-du-jour
  - Forme de distribution des returns :
      ret_skew_24       = skew roulant des close_ret (24)
      ret_kurt_48       = kurtosis roulant (48)
  - Regime de volatilite :
      vov_20            = vol-of-vol = std(std_roulant_10, 20)
      atr_accel_12      = atr_pct / atr_pct.shift(12) - 1   (expansion de vol)
  - Temps depuis extreme :
      bars_since_high_50 / bars_since_low_50  (en fraction de 50)
  - Structure de tendance :
      ema_compression_20 = (max - min des EMA 10/20/50) / close  (squeeze)
      donchian_mid_50    = (close - milieu du canal 50) / close  (signe = biais)
  - Couplage cross-asset :
      btc_eth_corr_48   = correlation roulante des returns BTC vs ETH (48)

NO-LOOKAHEAD (strict, identique a add_engineered_features.py) :
  - base 1h : la bougie i ferme a T = open_i + 1h = decision time, donc
    close/high/low/volume[i] sont connus a T ; rolling/shift n'utilisent que <= i.
  - daily/weekly open : open de la 1ere bougie de la periode, connu des le debut
    de la periode (passe). Veille = jour calendaire entierement clos a T.
  - vol_hod_ratio : moyenne EXPANDING .shift(1) (exclut la bougie courante).
  - ETH : bougie ETH 1h concurrente (meme open_time) fermee a T, join ts exact.

Idempotent : drop ses propres colonnes puis recompute. Ne touche QUE le parquet.
Toutes les colonnes sont scale-invariantes. Chaine :
    python data.py ; python add_eth_features.py ; python add_engineered_features.py
    python add_alpha_features.py     # <-- ce script
    python analyse/validate_alpha.py # walk-forward : garde les >=5 qui tiennent
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
ETH_PARQUET = ROOT / "eth_data" / "eth_15m.parquet"

H_MS = 60 * 60 * 1000
DAY_MS = 24 * H_MS
WEEK_MS = 7 * DAY_MS

NEW_ALPHA_COLS = [
    "ret_since_dopen", "ret_since_wopen",
    "dist_pday_high", "dist_pday_low", "pday_range_pos",
    "signed_vol_24", "signed_vol_6", "vol_hod_ratio",
    "ret_skew_24", "ret_kurt_48",
    "vov_20", "atr_accel_12",
    "bars_since_high_50", "bars_since_low_50",
    "ema_compression_20", "donchian_mid_50",
    "btc_eth_corr_48",
]


def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def load_tf_ohlcv(conn, timeframe: str) -> pd.DataFrame:
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


def _resample_eth_1h(eth_fine: pd.DataFrame) -> pd.DataFrame:
    eth = eth_fine.sort_values("timestamp_ms").reset_index(drop=True).copy()
    eth["bucket"] = (eth["timestamp_ms"].to_numpy(dtype=np.int64) // H_MS) * H_MS
    agg = eth.groupby("bucket", sort=True).agg(close=("close", "last")).reset_index()
    agg = agg.rename(columns={"bucket": "timestamp_ms"})
    return agg


def compute_alpha(df: pd.DataFrame, conn) -> dict:
    o, h, l, c = df["open"], df["high"], df["low"], df["close"].astype(float)
    vol = df["volume"].astype(float)
    cr = df["close_ret"].astype(float)
    cp = c.shift(1)
    ts = df["timestamp_ms"].to_numpy(dtype=np.int64)
    out = {}

    # --- Saisonnalite calendaire ---
    day_id = pd.Series(ts // DAY_MS, index=df.index)
    week_id = pd.Series(ts // WEEK_MS, index=df.index)
    day_open = o.groupby(day_id).transform("first")
    week_open = o.groupby(week_id).transform("first")
    out["ret_since_dopen"] = (c / day_open - 1.0).to_numpy()
    out["ret_since_wopen"] = (c / week_open - 1.0).to_numpy()

    # --- Structure de la veille (jour calendaire precedent, entierement clos) ---
    daily = pd.DataFrame({"day": day_id.to_numpy(), "high": h.to_numpy(),
                          "low": l.to_numpy()})
    dh = daily.groupby("day")["high"].max()
    dl = daily.groupby("day")["low"].min()
    pday_high = day_id.map(dh.shift(1))
    pday_low = day_id.map(dl.shift(1))
    out["dist_pday_high"] = ((c - pday_high.to_numpy()) / c).to_numpy()
    out["dist_pday_low"] = ((c - pday_low.to_numpy()) / c).to_numpy()
    rng_pd = (pday_high - pday_low).replace(0, np.nan)
    out["pday_range_pos"] = (((c - pday_low.to_numpy()) / rng_pd.to_numpy())
                             .clip(-0.5, 1.5)).to_numpy()

    # --- Order-flow / pression ---
    sign = np.sign(cr).fillna(0.0)
    sv = (sign * vol)
    for n in (24, 6):
        num = sv.rolling(n, min_periods=n).sum()
        den = vol.rolling(n, min_periods=n).sum().replace(0, np.nan)
        out[f"signed_vol_{n}"] = (num / den).to_numpy()
    hour = pd.Series((ts // H_MS) % 24, index=df.index)
    exp_mean_hour = vol.groupby(hour).transform(
        lambda s: s.shift(1).expanding(min_periods=5).mean())
    out["vol_hod_ratio"] = (vol / exp_mean_hour.replace(0, np.nan)).to_numpy()

    # --- Forme de distribution ---
    out["ret_skew_24"] = cr.rolling(24, min_periods=24).skew().to_numpy()
    out["ret_kurt_48"] = cr.rolling(48, min_periods=48).kurt().to_numpy()

    # --- Regime de volatilite ---
    rstd10 = cr.rolling(10, min_periods=10).std(ddof=0)
    out["vov_20"] = (rstd10.rolling(20, min_periods=20).std(ddof=0)
                     / rstd10.rolling(20, min_periods=20).mean().replace(0, np.nan)).to_numpy()
    tr = pd.concat([(h - l).abs(), (h - cp).abs(), (l - cp).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / 14, adjust=False).mean()
    atr_pct = atr / c
    out["atr_accel_12"] = (atr_pct / atr_pct.shift(12) - 1.0).to_numpy()

    # --- Temps depuis extreme (fraction de la fenetre) ---
    win = 50
    bsh = h.rolling(win, min_periods=win).apply(lambda x: (win - 1 - np.argmax(x)), raw=True)
    bsl = l.rolling(win, min_periods=win).apply(lambda x: (win - 1 - np.argmin(x)), raw=True)
    out["bars_since_high_50"] = (bsh / win).to_numpy()
    out["bars_since_low_50"] = (bsl / win).to_numpy()

    # --- Structure de tendance ---
    e10, e20, e50 = _ema(c, 10), _ema(c, 20), _ema(c, 50)
    emax = pd.concat([e10, e20, e50], axis=1).max(axis=1)
    emin = pd.concat([e10, e20, e50], axis=1).min(axis=1)
    out["ema_compression_20"] = ((emax - emin) / c).to_numpy()
    hh50 = h.rolling(win, min_periods=win).max()
    ll50 = l.rolling(win, min_periods=win).min()
    out["donchian_mid_50"] = ((c - (hh50 + ll50) / 2.0) / c).to_numpy()

    # --- Couplage cross-asset BTC/ETH (corr roulante des returns, concurrent) ---
    eth = _resample_eth_1h(pd.read_parquet(ETH_PARQUET))
    eth["eth_cr"] = eth["close"].astype(float).pct_change()
    eth_map = dict(zip(eth["timestamp_ms"].to_numpy(dtype=np.int64), eth["eth_cr"].to_numpy()))
    eth_cr = pd.Series([eth_map.get(int(t), np.nan) for t in ts], index=df.index)
    out["btc_eth_corr_48"] = cr.rolling(48, min_periods=48).corr(eth_cr).to_numpy()

    return out


def main() -> int:
    for p, label in [(MASTER_PARQUET, "master parquet"), (DB_SQLITE, "SQLite"),
                     (ETH_PARQUET, "ETH parquet")]:
        if not p.exists():
            print(f"ERREUR : {label} introuvable : {p}")
            return 1

    print("=== add_alpha_features.py (modele_up, 1h) ===")
    master = pd.read_parquet(MASTER_PARQUET)
    print(f"Master 1h : {len(master):,} rows x {len(master.columns)} cols")

    present = [c for c in NEW_ALPHA_COLS if c in master.columns]
    if present:
        print(f"  features alpha deja presentes ({len(present)}), reconstruction.")
        master = master.drop(columns=present)
    else:
        bak = MASTER_PARQUET.with_suffix(MASTER_PARQUET.suffix + ".prealpha.bak")
        if not bak.exists():
            shutil.copy2(MASTER_PARQUET, bak)
            print(f"  backup one-off : {bak.name}")

    conn = sqlite3.connect(DB_SQLITE)
    try:
        df1h = load_tf_ohlcv(conn, "1h")
        print(f"  OHLCV 1h reconstruit depuis SQLite : {len(df1h):,} rows")
        feats = compute_alpha(df1h, conn)
    finally:
        conn.close()

    feat_df = pd.DataFrame(feats)
    feat_df["timestamp_ms"] = df1h["timestamp_ms"].to_numpy(dtype=np.int64)

    missing = [c for c in NEW_ALPHA_COLS if c not in feat_df.columns]
    if missing:
        print(f"  ATTENTION colonnes non produites : {missing}")
    keep = [c for c in NEW_ALPHA_COLS if c in feat_df.columns] + ["timestamp_ms"]
    feat_df = feat_df[keep]

    merged = master.merge(feat_df, on="timestamp_ms", how="left")
    assert len(merged) == len(master), "merge a change le nombre de lignes !"

    tail = merged.tail(len(merged) - 2880)
    print(f"\n  {len(NEW_ALPHA_COLS)} features ajoutees. NaN% (post-warmup) :")
    for c in NEW_ALPHA_COLS:
        if c in merged.columns:
            pct = float(tail[c].isna().mean()) * 100
            flag = "  <-- ELEVE" if pct > 15 else ""
            print(f"    {c:<22} {pct:5.1f}%{flag}")

    merged.to_parquet(MASTER_PARQUET, index=False)
    print(f"\n-> {MASTER_PARQUET.name} : {len(merged):,} rows x {len(merged.columns)} cols")
    print("=== Termine. Lance : python analyse/validate_alpha.py ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
