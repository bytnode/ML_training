"""
08_cascade_check.py - Test empirique de l'hypothese "asymetrie du marche".

L'utilisateur affirme que les baisses BTC sont souvent PLUS fortes et
PRÉDICTIBLES que les hausses (cascades de liquidations). Si vrai, alors le
modele DOWN devrait pouvoir capturer ces cascades avec haute confiance.

Procedure :
  1. Pour chaque modele (UP, DOWN), trouve les 100 plus grosses bougies du test :
     - UP : les 100 plus grosses bougies VERTES (close - open le plus eleve en %)
     - DOWN : les 100 plus grosses bougies ROUGES (open - close le plus eleve)
  2. Mesure : sur ces "extreme moves", quelle etait la p_up / p_down du modele ?
     - Si modele DOWN voyait les cascades arriver : p_down sur les 100 grosses rouges = haute
     - Si modele DOWN NE voyait pas les cascades : p_down sur ces rouges ~ 0.50

  3. Bonus : pour les 100 rows avec p_down LE PLUS HAUT, regarde si la bougie suivante
     etait reellement une cascade ou juste une petite baisse. Compare avec UP.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "08_cascade_check.md"
DOWN_PARQUET = ROOT.parent / "db" / "btc_multi_tf_15m.parquet"
UP_PARQUET = ROOT.parent.parent / "modele_up" / "db" / "btc_multi_tf_15m.parquet"

TRAIN_DAYS = 4000
TEST_DAYS = 365
VAL_FRACTION = 0.1
WARMUP = 500
HP = dict(
    n_estimators=1000, max_depth=5, learning_rate=0.05,
    min_child_weight=20, subsample=0.8, colsample_bytree=0.8, gamma=1.5,
    objective="binary:logistic", eval_metric="logloss",
    early_stopping_rounds=50, random_state=42, n_jobs=-1,
)
FEATURES = ['acc_4', 'autocorr_24', 'bb_position', 'close_ret', 'close_ret_1h',
            'close_ret_30m', 'close_ret_5m', 'dist_ema_20', 'dist_to_low_50',
            'high_ret', 'high_ret_5m', 'low_ret_1h', 'low_ret_30m', 'low_ret_5m',
            'macd', 'macd_hist', 'mfi_norm', 'return_3', 'is_weekend']


def load(path):
    df = pd.read_parquet(path)
    df = df.iloc[WARMUP:].reset_index(drop=True)
    df = df.dropna(subset=["label"]).reset_index(drop=True)
    df["label"] = df["label"].astype(int)
    if "timestamp" in df.columns:
        df = df.drop(columns=["timestamp"])
    df = df.rename(columns={"timestamp_ms": "timestamp"})
    return df


def split(df):
    df = df.sort_values("timestamp").reset_index(drop=True)
    ms_per_day = 86_400 * 1000
    ts_max = df["timestamp"].max()
    test_cutoff = ts_max - TEST_DAYS * ms_per_day
    test_start = int(df["timestamp"].searchsorted(test_cutoff, side="left"))
    tv_cutoff = test_cutoff - TRAIN_DAYS * ms_per_day
    tv_start = int(df["timestamp"].searchsorted(tv_cutoff, side="left"))
    n_tv = test_start - tv_start
    val_size = int(n_tv * VAL_FRACTION)
    train_end = test_start - val_size
    return df.iloc[tv_start:train_end], df.iloc[train_end:test_start], df.iloc[test_start:]


def auto_spw(y):
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    return n_neg / n_pos if n_pos > 0 else 1.0


def fit(train, val, test, feats):
    spw = auto_spw(train["label"])
    m = xgb.XGBClassifier(scale_pos_weight=spw, **HP)
    m.fit(train[feats], train["label"], eval_set=[(val[feats], val["label"])], verbose=False)
    return m.predict_proba(test[feats])[:, 1], m


def main():
    df_d = load(DOWN_PARQUET)
    df_u = load(UP_PARQUET)
    train_u, val_u, test_u = split(df_u)
    train_d, val_d, test_d = split(df_d)
    print(f"test size : {len(test_u):,} (UP) / {len(test_d):,} (DOWN)")

    # On a besoin de la TAILLE des bougies. C'est la "next candle" qui est predite.
    # On a deja close_ret (= close[i]/close[i-1] - 1) mais on veut la taille de la
    # bougie suivante. Reconstruisons-la a partir de l'index.
    # Test contient timestamp, label, ET high_ret/low_ret/close_ret de la bougie i.
    # MAIS on veut SIZE(bougie i+1). Sans le close[i+1], on peut juste utiliser le
    # LABEL (= sign of next bar), pas sa taille.

    # On va donc procurer une approximation : pour chaque row du test, lire la row
    # SUIVANTE dans le parquet et calculer son close_ret = (next.close - next.open)/next.open
    # = body_signed de la prochaine bougie.

    # Reload le parquet complet pour avoir acces a close/open
    df_full = pd.read_parquet(DOWN_PARQUET)
    # Trier
    df_full = df_full.sort_values("timestamp_ms").reset_index(drop=True)

    if "open" not in df_full.columns or "close" not in df_full.columns:
        print("WARN : open/close pas dispo dans le parquet, on va se rabattre sur high_ret/low_ret de la bougie suivante.")
        # Calcule next_close_ret comme close_ret[i+1]
        df_full["next_close_ret"] = df_full["close_ret"].shift(-1)
    else:
        df_full["next_size"] = (df_full["close"].shift(-1) - df_full["open"].shift(-1)) / df_full["open"].shift(-1)

    # Train models
    print("\n[fit UP and DOWN]...")
    p_u, m_u = fit(train_u, val_u, test_u, FEATURES)
    p_d, m_d = fit(train_d, val_d, test_d, FEATURES)

    # Test set timestamps
    test_ts = test_d["timestamp"].to_numpy()

    # Pour chaque row de test, recupere la "next candle size" via merge sur timestamp
    next_size_map = {}
    if "next_size" in df_full.columns:
        for ts, ns in zip(df_full["timestamp_ms"].to_numpy(), df_full["next_size"].to_numpy()):
            if not np.isnan(ns):
                next_size_map[int(ts)] = float(ns)
    else:
        # fallback : use close_ret of next bar (which is approximately close_ret of i+1)
        for ts, ns in zip(df_full["timestamp_ms"].to_numpy(), df_full["next_close_ret"].to_numpy()):
            if not np.isnan(ns):
                next_size_map[int(ts)] = float(ns)

    next_size = np.array([next_size_map.get(int(ts), np.nan) for ts in test_ts])

    # Filter valid
    valid = ~np.isnan(next_size)
    next_size = next_size[valid]
    test_d_v = test_d.iloc[np.where(valid)[0]].reset_index(drop=True)
    test_u_v = test_u.iloc[np.where(valid)[0]].reset_index(drop=True)
    p_u_v = p_u[valid]
    p_d_v = p_d[valid]

    print(f"\n[test stats] valid rows = {valid.sum():,}")
    print(f"  next candle body distribution : mean={next_size.mean()*100:.4f}%, std={next_size.std()*100:.4f}%, min={next_size.min()*100:.4f}%, max={next_size.max()*100:.4f}%")

    # Top 100 bougies vertes (les plus grosses hausses)
    sorted_by_size = np.argsort(next_size)
    top_red = sorted_by_size[:100]   # plus negatifs = plus rouges
    top_green = sorted_by_size[-100:][::-1]  # plus positifs = plus verts

    print(f"\n=== TOP 100 BIGGEST RED CANDLES (= cascades potentielles) ===")
    print(f"  taille moyenne : {next_size[top_red].mean()*100:.3f}% (= chute moyenne)")
    print(f"  taille min     : {next_size[top_red].min()*100:.3f}% (la plus grosse chute)")
    print(f"  p_down moyen   : {p_d_v[top_red].mean()*100:.2f}%  (= confiance moyenne DOWN sur ces cascades)")
    print(f"  p_down median  : {np.median(p_d_v[top_red])*100:.2f}%")
    print(f"  p_down max     : {p_d_v[top_red].max()*100:.2f}%")
    print(f"  p_down min     : {p_d_v[top_red].min()*100:.2f}%")
    print(f"  p_up moyen     : {p_u_v[top_red].mean()*100:.2f}%  (= confiance moyenne UP sur ces cascades)")

    print(f"\n=== TOP 100 BIGGEST GREEN CANDLES (= rallies) ===")
    print(f"  taille moyenne : {next_size[top_green].mean()*100:.3f}%")
    print(f"  taille max     : {next_size[top_green].max()*100:.3f}%")
    print(f"  p_up moyen     : {p_u_v[top_green].mean()*100:.2f}%  (= confiance moyenne UP sur ces rallies)")
    print(f"  p_up median    : {np.median(p_u_v[top_green])*100:.2f}%")
    print(f"  p_up max       : {p_u_v[top_green].max()*100:.2f}%")
    print(f"  p_up min       : {p_u_v[top_green].min()*100:.2f}%")
    print(f"  p_down moyen   : {p_d_v[top_green].mean()*100:.2f}%")

    # Top 100 by predicted p
    top_p_up = np.argsort(p_u_v)[-100:][::-1]
    top_p_down = np.argsort(p_d_v)[-100:][::-1]

    print(f"\n=== TOP 100 highest p_up predictions ===")
    print(f"  p_up moyen       : {p_u_v[top_p_up].mean()*100:.2f}%")
    print(f"  next candle size : mean={next_size[top_p_up].mean()*100:.4f}%, median={np.median(next_size[top_p_up])*100:.4f}%")
    print(f"  % of these were green : {(next_size[top_p_up] > 0).mean()*100:.2f}%")
    print(f"  big greens (>0.5%) parmi ces 100 : {((next_size[top_p_up] > 0.005)).sum()}/100")

    print(f"\n=== TOP 100 highest p_down predictions ===")
    print(f"  p_down moyen     : {p_d_v[top_p_down].mean()*100:.2f}%")
    print(f"  next candle size : mean={next_size[top_p_down].mean()*100:.4f}%, median={np.median(next_size[top_p_down])*100:.4f}%")
    print(f"  % of these were red : {(next_size[top_p_down] < 0).mean()*100:.2f}%")
    print(f"  big reds (<-0.5%) parmi ces 100 : {((next_size[top_p_down] < -0.005)).sum()}/100")

    # Symmetry test : the "size" of bigger green vs bigger red bars
    print(f"\n=== SYMMETRY CHECK : taille des bougies extremes ===")
    print(f"  abs(min red)  = {abs(next_size.min())*100:.3f}% (plus grosse rouge)")
    print(f"  abs(max green)= {next_size.max()*100:.3f}% (plus grosse verte)")
    print(f"  std reds (only)  = {next_size[next_size < 0].std()*100:.4f}%")
    print(f"  std greens (only)= {next_size[next_size > 0].std()*100:.4f}%")
    print(f"  count |size|>1%  red  : {((next_size < -0.01)).sum()}")
    print(f"  count |size|>1%  green: {((next_size >  0.01)).sum()}")
    print(f"  count |size|>2%  red  : {((next_size < -0.02)).sum()}")
    print(f"  count |size|>2%  green: {((next_size >  0.02)).sum()}")

    # Write report
    lines = ["# Test empirique : asymmetrie REELLE des baisses vs hausses BTC\n\n"]
    lines.append("## Setup\n")
    lines.append(f"- Test set : {valid.sum():,} bougies (TEST_DAYS=365)\n")
    lines.append(f"- Memes 19 features pour les 2 modeles, memes hyperparams\n\n")

    lines.append("## Symetrie des bougies extremes\n\n")
    lines.append(f"| metric | red | green |\n|---|---|---|\n")
    lines.append(f"| |size| max | {abs(next_size.min())*100:.3f}% | {next_size.max()*100:.3f}% |\n")
    lines.append(f"| std (only own sign) | {next_size[next_size < 0].std()*100:.4f}% | {next_size[next_size > 0].std()*100:.4f}% |\n")
    lines.append(f"| count > 1% | {((next_size < -0.01)).sum()} | {((next_size >  0.01)).sum()} |\n")
    lines.append(f"| count > 2% | {((next_size < -0.02)).sum()} | {((next_size >  0.02)).sum()} |\n\n")

    lines.append("## Test : le modele voit-il les vraies grosses bougies ?\n\n")
    lines.append("### TOP 100 BIG RED candles (= les chutes les plus fortes)\n\n")
    lines.append(f"- taille moyenne : {next_size[top_red].mean()*100:.3f}%, max chute : {next_size[top_red].min()*100:.3f}%\n")
    lines.append(f"- **p_down moyen attribue par le modele DOWN : {p_d_v[top_red].mean()*100:.2f}%**\n")
    lines.append(f"- p_down median : {np.median(p_d_v[top_red])*100:.2f}%, max : {p_d_v[top_red].max()*100:.2f}%\n")
    lines.append(f"- p_up moyen (modele UP, miroir) : {p_u_v[top_red].mean()*100:.2f}%\n\n")

    lines.append("### TOP 100 BIG GREEN candles (= les rallies les plus forts)\n\n")
    lines.append(f"- taille moyenne : {next_size[top_green].mean()*100:.3f}%, max rally : {next_size[top_green].max()*100:.3f}%\n")
    lines.append(f"- **p_up moyen attribue par le modele UP : {p_u_v[top_green].mean()*100:.2f}%**\n")
    lines.append(f"- p_up median : {np.median(p_u_v[top_green])*100:.2f}%, max : {p_u_v[top_green].max()*100:.2f}%\n")
    lines.append(f"- p_down moyen : {p_d_v[top_green].mean()*100:.2f}%\n\n")

    lines.append("## Test : ce que le modele predit comme tres haut, est-ce reellement grand ?\n\n")
    lines.append("### TOP 100 highest p_up\n")
    lines.append(f"- p_up moyen : {p_u_v[top_p_up].mean()*100:.2f}%\n")
    lines.append(f"- taille de la bougie suivante : mean={next_size[top_p_up].mean()*100:.4f}%, median={np.median(next_size[top_p_up])*100:.4f}%\n")
    lines.append(f"- % qui sont effectivement vertes : {(next_size[top_p_up] > 0).mean()*100:.2f}%\n")
    lines.append(f"- bougies vraiment grosses (>0.5%) : {((next_size[top_p_up] > 0.005)).sum()}/100\n\n")

    lines.append("### TOP 100 highest p_down\n")
    lines.append(f"- p_down moyen : {p_d_v[top_p_down].mean()*100:.2f}%\n")
    lines.append(f"- taille de la bougie suivante : mean={next_size[top_p_down].mean()*100:.4f}%, median={np.median(next_size[top_p_down])*100:.4f}%\n")
    lines.append(f"- % qui sont effectivement rouges : {(next_size[top_p_down] < 0).mean()*100:.2f}%\n")
    lines.append(f"- bougies vraiment grosses (<-0.5%) : {((next_size[top_p_down] < -0.005)).sum()}/100\n\n")

    OUT.write_text("".join(lines), encoding="utf-8")
    print(f"\n[ok] -> {OUT}")


if __name__ == "__main__":
    main()
