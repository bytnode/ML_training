"""
add_eth_features.py — patch ETH/USD pour le master 30m (parite features 15min).

Porte le builder ETH du modele 15min (live/modele_15m/live_helpers/features_down.py
:: _compute_eth_features) vers la base 30m. La seule difference : l'ETH 15m
brut (eth_data/eth_15m.parquet) est d'abord RESAMPLE en bougies 30m
(open=first, high=max, low=min, close=last, volume=sum) avant le calcul des
features ef_*, de sorte qu'une bougie ETH 30m partage exactement l'open_time
d'une bougie BTC 30m et ferme au meme instant T = open_30m + 30m (decision time).

Les ef_* sont ensuite alignes NO-LOOKAHEAD sur chaque BTC 30m par join de
TIMESTAMP EXACT (la bougie ETH 30m concurrente, fermee a T). Les BTC bars
anterieurs a l'historique ETH recoivent NaN (XGBoost route NaN nativement).

Le modele 30m n'utilise que 5 des features (ef_ret2, ef_ret8, ef_streak,
ef_macd, ef_body — la whitelist FEATURE_WHITELIST), mais on produit les 14
comme le 15min pour rester fidele au patch d'origine et permettre l'exploration.

Idempotent : garde un <parquet>.bak la premiere fois ; re-executer data.py
reconstruit le parquet SANS les ef_*, donc relancer ce script ensuite.

Usage :
    cd backtest/modele_updown/30min/modele_down
    python data.py
    python add_eth_features.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
MASTER_PARQUET = ROOT / "DB" / "btc_multi_tf_1h.parquet"
ETH_PARQUET = ROOT / "eth_data" / "eth_15m.parquet"

INTERVAL_1H_MS = 60 * 60 * 1000

# Les 14 features ETH du patch 15min d'origine (le modele 30m n'en garde que 5
# via la whitelist, mais on produit les 14 par fidelite au patch).
ETH_FEATURE_COLS = [
    "ef_ret1", "ef_ret2", "ef_ret4", "ef_ret8", "ef_ret16", "ef_ret48", "ef_ret96",
    "ef_streak", "ef_macd", "ef_macd_hist", "ef_vol_z", "ef_upwick", "ef_lowick",
    "ef_body",
]


def _eth_ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def _eth_streak(close_ret: pd.Series) -> np.ndarray:
    """Signed run length: +k apres k bougies ETH vertes, -k apres k rouges."""
    sign = np.sign(close_ret.to_numpy())
    s = pd.Series(sign)
    grp = (s != s.shift()).cumsum()
    run = s.groupby(grp).cumcount().to_numpy() + 1
    return run * sign


def resample_eth_to_30m(eth_15m: pd.DataFrame) -> pd.DataFrame:
    """ETH 15m OHLCV -> bougies 30m (open=first/high=max/low=min/close=last/
    volume=sum), keyees par l'open_time du bucket 30m (multiple de 30min).

    Un bucket 30m d'open_time T agrege les bougies 15m d'open_time T et T+15m.
    L'open_time du bucket 30m s'aligne EXACTEMENT sur l'open_time des BTC 30m.
    """
    eth = eth_15m.sort_values("timestamp_ms").reset_index(drop=True).copy()
    eth["bucket"] = (eth["timestamp_ms"].to_numpy(dtype=np.int64)
                     // INTERVAL_1H_MS) * INTERVAL_1H_MS
    agg = eth.groupby("bucket", sort=True).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    ).reset_index().rename(columns={"bucket": "timestamp_ms"})
    return agg


def compute_eth_features(eth: pd.DataFrame) -> pd.DataFrame:
    """ETH 30m OHLCV -> DataFrame[timestamp_ms + 14 ef_ feature columns].

    Port verbatim de _compute_eth_features (builder live 15min), etendu aux 14
    features d'origine. Toutes les features sont scale-invariantes / backward-
    looking sur la serie ETH 30m.
    """
    eth = eth.sort_values("timestamp_ms").reset_index(drop=True)
    o = eth["open"].astype(float)
    h = eth["high"].astype(float)
    l = eth["low"].astype(float)
    c = eth["close"].astype(float)
    cp = c.shift(1)
    cr = (c - cp) / cp
    vol = eth["volume"].astype(float)

    f = {"timestamp_ms": eth["timestamp_ms"].to_numpy(dtype=np.int64)}

    # momentum : ef_retN = close / close[-N] - 1
    for n in (1, 2, 4, 8, 16, 48, 96):
        f[f"ef_ret{n}"] = (c / c.shift(n) - 1.0).to_numpy()

    # trend
    f["ef_streak"] = _eth_streak(cr)
    macd = (_eth_ema(c, 12) - _eth_ema(c, 26)) / c
    macd_signal = _eth_ema(macd, 9)
    f["ef_macd"] = macd.to_numpy()
    f["ef_macd_hist"] = (macd - macd_signal).to_numpy()

    # volume z-score (rolling 20)
    vol_mean = vol.rolling(20, min_periods=20).mean()
    vol_std = vol.rolling(20, min_periods=20).std(ddof=0)
    f["ef_vol_z"] = ((vol - vol_mean) / vol_std.replace(0, np.nan)).to_numpy()

    # microstructure
    rng = (h - l).replace(0, np.nan)
    upper = (h - np.maximum(o, c)) / rng
    lower = (np.minimum(o, c) - l) / rng
    f["ef_upwick"] = upper.clip(0, 1).to_numpy()
    f["ef_lowick"] = lower.clip(0, 1).to_numpy()
    f["ef_body"] = ((c - o) / cp).to_numpy()

    return pd.DataFrame(f)


def main() -> int:
    if not MASTER_PARQUET.exists():
        print(f"ERREUR : master parquet introuvable : {MASTER_PARQUET}")
        print("Lance d'abord : python data.py")
        return 1
    if not ETH_PARQUET.exists():
        print(f"ERREUR : ETH parquet introuvable : {ETH_PARQUET}")
        return 1

    print(f"=== add_eth_features.py (30m) ===")
    master = pd.read_parquet(MASTER_PARQUET)
    print(f"Master 30m : {len(master):,} rows x {len(master.columns)} cols")

    # Si les ef_* sont deja presents (re-run sans rebuild), on les drop d'abord
    # pour reconstruire proprement.
    present = [c for c in ETH_FEATURE_COLS if c in master.columns]
    if present:
        print(f"  ef_* deja presents ({len(present)}), reconstruction.")
        master = master.drop(columns=present)
    else:
        # Premier passage -> backup one-off.
        bak = MASTER_PARQUET.with_suffix(MASTER_PARQUET.suffix + ".bak")
        if not bak.exists():
            shutil.copy2(MASTER_PARQUET, bak)
            print(f"  backup : {bak.name}")

    eth_15m = pd.read_parquet(ETH_PARQUET)
    eth_30m = resample_eth_to_30m(eth_15m)
    print(f"ETH 15m : {len(eth_15m):,} rows -> ETH 30m resample : {len(eth_30m):,} rows")

    eth_feat = compute_eth_features(eth_30m)

    # Join no-lookahead par TIMESTAMP EXACT : la bougie ETH 30m concurrente
    # (meme open_time que la BTC 30m) ferme a T = open_30m + 30m = decision time.
    master_ts = master["timestamp_ms"].to_numpy(dtype=np.int64)
    eth_map = {int(ts): i for i, ts in enumerate(eth_feat["timestamp_ms"].to_numpy(dtype=np.int64))}
    idx = np.array([eth_map.get(int(ts), -1) for ts in master_ts])
    valid = idx >= 0

    n_matched = int(valid.sum())
    for col in ETH_FEATURE_COLS:
        src = eth_feat[col].to_numpy(dtype=float)
        vals = np.full(len(master), np.nan, dtype=float)
        if valid.any():
            vals[valid] = src[idx[valid]]
        master[col] = vals

    print(f"ETH features ajoutees : {len(ETH_FEATURE_COLS)} cols, "
          f"{n_matched:,}/{len(master):,} BTC bars matchees (le reste = NaN, "
          f"pre-historique ETH).")

    master.to_parquet(MASTER_PARQUET, index=False)
    print(f"-> {MASTER_PARQUET.name} : {len(master):,} rows x {len(master.columns)} cols")
    print(f"=== Termine. Lance maintenant : python train.py ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
