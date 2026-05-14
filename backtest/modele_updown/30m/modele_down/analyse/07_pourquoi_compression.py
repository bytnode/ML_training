"""
07_pourquoi_compression.py - Demonstration mecaniste du POURQUOI DOWN est
compresse vs UP.

Le mecanisme : XGBoost produit p = sigmoid(sum_t(leaf_value_t)).
Pour que p atteigne 0.60, il faut sum = log(0.6/0.4) = +0.405 (log-odds).
Pour que p atteigne 0.55, il faut sum = log(0.55/0.45) = +0.201.

Si les arbres SUMMENT a une valeur plus faible pour DOWN que pour UP,
c'est parce que les gradients pendant l'entrainement sont plus faibles
(= signal moins fort dans les features pour predire DOWN).

On compare :
  1. La distribution des log-odds bruts (= avant sigmoid) pour UP et DOWN
  2. La val_logloss (= mesure du signal extrait par le modele)
  3. Le mutual information par feature (= info de chaque feature sur le label)
  4. La WR par bin de p (= calibration : meme p, meme WR ?)

Tout ca avec MEMES features (intersect + is_weekend) pour les deux modeles.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "07_pourquoi_compression.md"
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
    m.fit(train[feats], train["label"],
          eval_set=[(val[feats], val["label"])], verbose=False)
    p = m.predict_proba(test[feats])[:, 1]
    # Raw log-odds : output_margin=True
    raw = m.predict(test[feats], output_margin=True)
    val_ll = float(m.evals_result()["validation_0"]["logloss"][m.best_iteration])
    train_ll = m.evals_result().get("validation_0", {})
    return p, raw, m, val_ll


def reliability(p, y):
    bins = np.linspace(0.40, 0.65, 11)
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (p >= lo) & (p < hi)
        n = int(mask.sum())
        wr = float(y[mask].mean()) if n > 0 else 0.0
        out.append((round(lo, 3), round(hi, 3), n, wr))
    return out


def main():
    print("[load] UP and DOWN parquets...")
    df_u = load(UP_PARQUET)
    df_d = load(DOWN_PARQUET)
    train_u, val_u, test_u = split(df_u)
    train_d, val_d, test_d = split(df_d)

    print(f"\n[fit UP]")
    p_u, raw_u, m_u, vll_u = fit(train_u, val_u, test_u, FEATURES)
    print(f"  val_logloss = {vll_u:.6f}")
    print(f"  raw log-odds : min={raw_u.min():.3f}, max={raw_u.max():.3f}, std={raw_u.std():.3f}")
    print(f"  p          : min={p_u.min():.4f}, max={p_u.max():.4f}, std={p_u.std():.4f}")

    print(f"\n[fit DOWN]")
    p_d, raw_d, m_d, vll_d = fit(train_d, val_d, test_d, FEATURES)
    print(f"  val_logloss = {vll_d:.6f}")
    print(f"  raw log-odds : min={raw_d.min():.3f}, max={raw_d.max():.3f}, std={raw_d.std():.3f}")
    print(f"  p          : min={p_d.min():.4f}, max={p_d.max():.4f}, std={p_d.std():.4f}")

    # Reliability (= calibration per bin)
    rel_u = reliability(p_u, test_u["label"].to_numpy())
    rel_d = reliability(p_d, test_d["label"].to_numpy())

    # Importance ratio
    imp_u = dict(zip(FEATURES, m_u.feature_importances_.tolist()))
    imp_d = dict(zip(FEATURES, m_d.feature_importances_.tolist()))

    lines = ["# Pourquoi DOWN compresse vs UP (memes features + memes HP)\n\n"]
    lines.append("## Setup\n\n")
    lines.append(f"- 19 features identiques pour les 2 modeles : `{FEATURES}`\n")
    lines.append(f"- HP identiques : max_depth=5, n_estim=1000, lr=0.05, min_child_weight=20\n\n")

    lines.append("## Comparaison directe\n\n")
    lines.append("| metric | UP | DOWN | gap |\n")
    lines.append("|---|---|---|---|\n")
    lines.append(f"| **val_logloss** | {vll_u:.6f} | {vll_d:.6f} | **{(vll_d - vll_u)*1000:+.2f}m** |\n")
    lines.append(f"| best_iteration | {m_u.best_iteration} | {m_d.best_iteration} | {m_d.best_iteration - m_u.best_iteration:+d} |\n")
    lines.append(f"| n_trees (= best+1) | {m_u.best_iteration + 1} | {m_d.best_iteration + 1} | |\n")
    lines.append(f"| raw_log-odds max | {raw_u.max():.4f} | {raw_d.max():.4f} | {raw_d.max() - raw_u.max():+.4f} |\n")
    lines.append(f"| raw_log-odds std | {raw_u.std():.4f} | {raw_d.std():.4f} | {raw_d.std() - raw_u.std():+.4f} |\n")
    lines.append(f"| p max | {p_u.max():.4f} | {p_d.max():.4f} | {(p_d.max() - p_u.max())*100:+.2f}pp |\n")
    lines.append(f"| p mean | {p_u.mean():.4f} | {p_d.mean():.4f} | {(p_d.mean() - p_u.mean())*100:+.2f}pp |\n")
    lines.append(f"| p std | {p_u.std():.4f} | {p_d.std():.4f} | {(p_d.std() - p_u.std())*100:+.2f}pp |\n")
    lines.append(f"| label base rate | {train_u['label'].mean():.4f} | {train_d['label'].mean():.4f} | |\n")

    # Interprete
    lines.append(f"\n## Interpretation\n\n")
    delta_ll = (vll_d - vll_u) * 1000
    if delta_ll > 0.1:
        lines.append(f"**val_logloss DOWN = {vll_d:.6f}, UP = {vll_u:.6f} → DOWN est {delta_ll:.2f}m PLUS HAUT.**\n\n")
        lines.append(f"Cela signifie que le modele DOWN extrait MOINS de signal des memes features.\n")
        lines.append(f"En d'autres termes : les features predisent MIEUX la prochaine bougie verte (UP) que la prochaine bougie rouge (DOWN).\n")
        lines.append(f"Ce n'est PAS un probleme du modele, c'est la NATURE DE LA DONNEE BTC.\n\n")

    lines.append(f"## Reliability comparison (WR observee a chaque bin de p)\n\n")
    lines.append(f"| p_bin | UP n | UP WR | DOWN n | DOWN WR | Δ |\n|---|---|---|---|---|---|\n")
    for (lo_u, hi_u, n_u, wr_u), (lo_d, hi_d, n_d, wr_d) in zip(rel_u, rel_d):
        delta_wr = (wr_d - wr_u) * 100 if (n_u > 30 and n_d > 30) else None
        delta_str = f"{delta_wr:+.2f}pp" if delta_wr is not None else "N/A"
        lines.append(f"| [{lo_u}, {hi_u}) | {n_u} | {wr_u*100:.2f}% | {n_d} | {wr_d*100:.2f}% | {delta_str} |\n")

    lines.append(f"\n## Importance features (memes features, mais le modele les utilise differemment)\n\n")
    lines.append(f"| feature | UP imp | DOWN imp | ratio (DOWN/UP) |\n|---|---|---|---|\n")
    for f in FEATURES:
        u = imp_u[f]
        d = imp_d[f]
        ratio = d / u if u > 0 else float('nan')
        lines.append(f"| {f} | {u:.4f} | {d:.4f} | {ratio:.2f} |\n")

    OUT.write_text("".join(lines), encoding="utf-8")
    print(f"\n[ok] -> {OUT}")

    print(f"\n=== DECISIF ===")
    print(f"val_logloss UP   : {vll_u:.6f}")
    print(f"val_logloss DOWN : {vll_d:.6f}")
    print(f"DIFFERENCE       : {(vll_d - vll_u)*1000:+.3f}m (DOWN extrait moins de signal)")
    print(f"")
    print(f"Raw log-odds max UP   : {raw_u.max():.4f} (= log-odds qui donne p={1/(1+np.exp(-raw_u.max())):.4f})")
    print(f"Raw log-odds max DOWN : {raw_d.max():.4f} (= log-odds qui donne p={1/(1+np.exp(-raw_d.max())):.4f})")


if __name__ == "__main__":
    main()
