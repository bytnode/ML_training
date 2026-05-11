"""
02_train_variants.py - Entraine plusieurs variantes du modele et compare les
distributions de p_positive (= p_up ou p_down selon le label).

Objectif : isoler la cause de la sous-confiance du modele DOWN.

Variantes :
  A. UP_full      : modele UP avec toutes les features UP (current state)
  B. UP_no_vol30d : modele UP sans volume_rel_30d (= reste = intersect+is_weekend? non,
                    UP excluait is_weekend, donc reste = features UP - volume_rel_30d)
  C. DOWN_full    : modele DOWN avec toutes les features DOWN + is_weekend (current state)
  D. DOWN_intersect : DOWN avec seulement features communes UP/DOWN (sans is_weekend)
  E. DOWN_no_isweekend : DOWN current state moins is_weekend
  F. DOWN_only_intersect_excl_isweekend : DOWN avec EXACTEMENT les memes features que UP_no_vol30d

Hypotheses testees :
  - Si C reste compresse mais F a la meme distribution que B → c'est les features supplementaires de DOWN qui compriment
  - Si F est encore compresse → c'est la nature du label (DOWN minoritaire) qui compresse, pas les features

Compare la distribution complete : mean, std, max, q90, q99, q99.9.
"""
from __future__ import annotations
import time
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path

import sys
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT.parent.parent / "modele_up"))

OUT = ROOT / "02_train_variants.md"
DOWN_PARQUET = ROOT.parent / "db" / "btc_multi_tf_15m.parquet"
UP_PARQUET = ROOT.parent.parent / "modele_up" / "db" / "btc_multi_tf_15m.parquet"

# Hyperparams identiques pour tous (= modele_up actuels = modele_down actuels)
HP = dict(
    n_estimators=1000, max_depth=5, learning_rate=0.05,
    min_child_weight=20, subsample=0.8, colsample_bytree=0.8,
    objective="binary:logistic", eval_metric="logloss",
    early_stopping_rounds=50, random_state=42, n_jobs=-1,
    gamma=1.5,
)

TRAIN_DAYS = 4000
TEST_DAYS = 365
VAL_FRACTION = 0.1
WARMUP = 500


def load_parquet(path):
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
    test_cutoff_ts = ts_max - TEST_DAYS * ms_per_day
    test_start = int(df["timestamp"].searchsorted(test_cutoff_ts, side="left"))
    train_val_cutoff_ts = test_cutoff_ts - TRAIN_DAYS * ms_per_day
    train_val_start = int(df["timestamp"].searchsorted(train_val_cutoff_ts, side="left"))
    n_train_val = test_start - train_val_start
    val_size = int(n_train_val * VAL_FRACTION)
    train_end = test_start - val_size
    return df.iloc[train_val_start:train_end], df.iloc[train_end:test_start], df.iloc[test_start:]


def auto_spw(y_train):
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    return n_neg / n_pos if n_pos > 0 else 1.0


def fit(train, val, test, feature_cols):
    spw = auto_spw(train["label"])
    m = xgb.XGBClassifier(scale_pos_weight=spw, **HP)
    m.fit(train[feature_cols], train["label"],
          eval_set=[(val[feature_cols], val["label"])], verbose=False)
    p_test = m.predict_proba(test[feature_cols])[:, 1]
    return p_test, m


def stats(p, label=None):
    p = np.asarray(p)
    out = {
        "n": len(p),
        "mean": float(p.mean()),
        "median": float(np.median(p)),
        "std": float(p.std()),
        "min": float(p.min()),
        "max": float(p.max()),
        "q90": float(np.quantile(p, 0.90)),
        "q99": float(np.quantile(p, 0.99)),
        "q99_9": float(np.quantile(p, 0.999)),
    }
    if label is not None:
        # Reliability : pour chaque bin de p, calcule la WR observee
        bins = np.linspace(0.40, 0.65, 11)
        out["reliability"] = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (p >= lo) & (p < hi)
            n = int(m.sum())
            wr = float(label[m].mean()) if n > 0 else 0.0
            out["reliability"].append((round(lo, 3), round(hi, 3), n, round(wr, 4)))
    return out


def main():
    print("[load] loading parquets...")
    df_d = load_parquet(DOWN_PARQUET)
    df_u = load_parquet(UP_PARQUET)
    print(f"  DOWN: {len(df_d):,} rows, {len(df_d.columns)} cols")
    print(f"  UP  : {len(df_u):,} rows, {len(df_u.columns)} cols")

    # Splits separes pour chaque parquet
    train_d, val_d, test_d = split(df_d)
    train_u, val_u, test_u = split(df_u)
    print(f"  DOWN split: train={len(train_d):,} val={len(val_d):,} test={len(test_d):,}")
    print(f"  UP   split: train={len(train_u):,} val={len(val_u):,} test={len(test_u):,}")

    # Feature sets
    excluded_for_up = ("timestamp", "label", "volatility_state", "is_weekend")
    excluded_for_down_current = ("timestamp", "label", "volatility_state")

    features_up_full = sorted(c for c in df_u.columns if c not in excluded_for_up)
    features_down_full = sorted(c for c in df_d.columns if c not in excluded_for_down_current)

    common = set(df_u.columns) & set(df_d.columns)
    common_no_meta = sorted(c for c in common if c not in ("timestamp", "label", "volatility_state", "is_weekend"))

    print(f"\n  features UP full ({len(features_up_full)}): {features_up_full}")
    print(f"  features DOWN full ({len(features_down_full)}): {features_down_full}")
    print(f"  common (excl is_weekend): {len(common_no_meta)} = {common_no_meta}")

    variants = {}

    # A. UP_full
    print("\n[A] UP_full...")
    p_test, m = fit(train_u, val_u, test_u, features_up_full)
    variants["A_UP_full"] = {
        "n_features": len(features_up_full),
        "features": features_up_full,
        "p_stats": stats(p_test, test_u["label"].to_numpy()),
        "best_iteration": int(m.best_iteration),
    }

    # B. UP_no_vol30d
    print("[B] UP_no_vol30d...")
    feats_b = [c for c in features_up_full if c != "volume_rel_30d"]
    p_test, m = fit(train_u, val_u, test_u, feats_b)
    variants["B_UP_no_vol30d"] = {
        "n_features": len(feats_b),
        "features": feats_b,
        "p_stats": stats(p_test, test_u["label"].to_numpy()),
        "best_iteration": int(m.best_iteration),
    }

    # C. DOWN_full (current state)
    print("[C] DOWN_full...")
    p_test, m = fit(train_d, val_d, test_d, features_down_full)
    variants["C_DOWN_full"] = {
        "n_features": len(features_down_full),
        "features": features_down_full,
        "p_stats": stats(p_test, test_d["label"].to_numpy()),
        "best_iteration": int(m.best_iteration),
    }

    # D. DOWN_intersect (features communes only, sans is_weekend)
    print("[D] DOWN_intersect (= same features as UP_no_vol30d)...")
    feats_d = [c for c in common_no_meta]
    p_test, m = fit(train_d, val_d, test_d, feats_d)
    variants["D_DOWN_intersect_excl_isweekend"] = {
        "n_features": len(feats_d),
        "features": feats_d,
        "p_stats": stats(p_test, test_d["label"].to_numpy()),
        "best_iteration": int(m.best_iteration),
    }

    # E. DOWN_no_isweekend (DOWN full sans is_weekend)
    print("[E] DOWN_no_isweekend...")
    feats_e = [c for c in features_down_full if c != "is_weekend"]
    p_test, m = fit(train_d, val_d, test_d, feats_e)
    variants["E_DOWN_no_isweekend"] = {
        "n_features": len(feats_e),
        "features": feats_e,
        "p_stats": stats(p_test, test_d["label"].to_numpy()),
        "best_iteration": int(m.best_iteration),
    }

    # F. DOWN_intersect_with_isweekend (intersect + is_weekend, sans les features DOWN-only)
    print("[F] DOWN_intersect_with_isweekend...")
    feats_f = [c for c in common_no_meta] + ["is_weekend"]
    feats_f = [c for c in feats_f if c in df_d.columns]
    p_test, m = fit(train_d, val_d, test_d, feats_f)
    variants["F_DOWN_intersect_with_isweekend"] = {
        "n_features": len(feats_f),
        "features": feats_f,
        "p_stats": stats(p_test, test_d["label"].to_numpy()),
        "best_iteration": int(m.best_iteration),
    }

    # Write report
    lines = ["# Audit des variantes UP/DOWN\n\n"]
    lines.append("## Hyperparams (identiques pour toutes les variantes)\n")
    lines.append(f"```\n{json.dumps(HP, indent=2)}\n```\n\n")
    lines.append("## Splits\n")
    lines.append(f"- DOWN: train={len(train_d):,} val={len(val_d):,} test={len(test_d):,}\n")
    lines.append(f"- UP  : train={len(train_u):,} val={len(val_u):,} test={len(test_u):,}\n\n")
    lines.append("## Distributions p_positive sur test\n\n")
    lines.append("| variant | n_feat | best_iter | mean | std | max | q90 | q99 | q99.9 |\n")
    lines.append("|---|---|---|---|---|---|---|---|---|\n")
    for name, v in variants.items():
        s = v["p_stats"]
        lines.append(
            f"| **{name}** | {v['n_features']} | {v['best_iteration']} | "
            f"{s['mean']*100:.2f}% | {s['std']*100:.2f}pp | "
            f"**{s['max']*100:.2f}%** | {s['q90']*100:.2f}% | "
            f"{s['q99']*100:.2f}% | {s['q99_9']*100:.2f}% |\n"
        )
    lines.append("\n## Reliability per variant (WR observee par bin de p_positive)\n\n")
    for name, v in variants.items():
        lines.append(f"### {name}\n\n")
        lines.append(f"| p_bin | n | WR |\n|---|---|---|\n")
        for lo, hi, n, wr in v["p_stats"]["reliability"]:
            lines.append(f"| [{lo:.3f}, {hi:.3f}) | {n} | {wr*100:.2f}% |\n")
        lines.append("\n")
    lines.append("\n## Features par variant\n\n")
    for name, v in variants.items():
        lines.append(f"### {name} ({v['n_features']})\n```\n")
        for f in v["features"]:
            lines.append(f"  {f}\n")
        lines.append("```\n\n")

    OUT.write_text("".join(lines), encoding="utf-8")
    print(f"\n[ok] -> {OUT}")
    print(f"\n=== SUMMARY ===")
    print(f"{'variant':<40} {'n_feat':>6} {'mean':>7} {'std':>7} {'max':>7}")
    for name, v in variants.items():
        s = v["p_stats"]
        print(f"{name:<40} {v['n_features']:>6} {s['mean']*100:>6.2f}% {s['std']*100:>6.2f}pp {s['max']*100:>6.2f}%")


if __name__ == "__main__":
    main()
