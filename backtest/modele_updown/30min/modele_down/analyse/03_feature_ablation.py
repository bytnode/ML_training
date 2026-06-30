"""
03_feature_ablation.py - Ablation feature par feature pour identifier laquelle
des 6 features DOWN-only degrade la spread du modele.

Setup :
  Baseline = variant F (intersect + is_weekend) → max ~57.82%
  Pour chaque feature DOWN-only, on l'ajoute a la baseline et on mesure la
  nouvelle distribution. Si max baisse → feature mauvaise. Si max monte →
  feature utile.

Features DOWN-only a tester individuellement :
  - body_ratio_close_30m
  - dist_to_high_50
  - hour_sin
  - low_ret
  - obv_pressure_5m
  - volume_ratio_30m
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "03_feature_ablation.md"
DOWN_PARQUET = ROOT.parent / "db" / "btc_multi_tf_15m.parquet"

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


def stats(p):
    p = np.asarray(p)
    return {
        "mean": float(p.mean()),
        "std": float(p.std()),
        "max": float(p.max()),
        "q99": float(np.quantile(p, 0.99)),
        "q99_9": float(np.quantile(p, 0.999)),
    }


def main():
    df_d = load_parquet(DOWN_PARQUET)
    train, val, test = split(df_d)
    print(f"DOWN split: train={len(train):,} val={len(val):,} test={len(test):,}")

    # Baseline F = intersect + is_weekend
    common = ['acc_4', 'autocorr_24', 'bb_position', 'close_ret', 'close_ret_1h',
              'close_ret_30m', 'close_ret_5m', 'dist_ema_20', 'dist_to_low_50',
              'high_ret', 'high_ret_5m', 'low_ret_1h', 'low_ret_30m', 'low_ret_5m',
              'macd', 'macd_hist', 'mfi_norm', 'return_3']
    baseline_features = common + ['is_weekend']
    down_only = ['body_ratio_close_30m', 'dist_to_high_50', 'hour_sin',
                  'low_ret', 'obv_pressure_5m', 'volume_ratio_30m']

    results = {}

    # Baseline (variant F)
    print(f"\n[baseline] {len(baseline_features)} features (intersect + is_weekend)")
    p, m = fit(train, val, test, baseline_features)
    importance_baseline = dict(zip(baseline_features, m.feature_importances_))
    results["baseline_F"] = {
        "n_features": len(baseline_features),
        "best_iter": int(m.best_iteration),
        **stats(p),
    }

    # Add each DOWN-only feature one at a time
    for f in down_only:
        print(f"[ablation+] baseline + {f}")
        feats = baseline_features + [f]
        p, m = fit(train, val, test, feats)
        results[f"baseline_plus_{f}"] = {
            "n_features": len(feats),
            "best_iter": int(m.best_iteration),
            "added": f,
            "feature_importance_of_added": float(dict(zip(feats, m.feature_importances_))[f]),
            **stats(p),
        }

    # All DOWN-only added (= variant C current state)
    print(f"\n[full] baseline + all 6 DOWN-only features")
    p, m = fit(train, val, test, baseline_features + down_only)
    results["baseline_plus_all_down_only"] = {
        "n_features": len(baseline_features) + len(down_only),
        "best_iter": int(m.best_iteration),
        **stats(p),
    }

    # Also test: remove each common feature one at a time from baseline (sanity check)
    # Skipped for now to keep audit tight

    # Report
    lines = ["# Ablation des features DOWN-only\n\n"]
    lines.append("## Setup\n")
    lines.append(f"- Baseline = 18 features intersect + `is_weekend` = 19 features.\n")
    lines.append(f"- Pour chaque feature DOWN-only, on l'ajoute (= +1 feature) et on mesure la nouvelle distribution.\n\n")
    lines.append(f"## Resultats\n\n")
    lines.append(f"| config | n_feat | best_iter | mean | std | max | q99 | q99.9 | importance |\n")
    lines.append(f"|---|---|---|---|---|---|---|---|---|\n")
    base = results["baseline_F"]
    base_max = base["max"]
    lines.append(f"| **baseline (F)** | {base['n_features']} | {base['best_iter']} | "
                 f"{base['mean']*100:.2f}% | {base['std']*100:.2f}pp | {base['max']*100:.2f}% | "
                 f"{base['q99']*100:.2f}% | {base['q99_9']*100:.2f}% | - |\n")
    for f in down_only:
        v = results[f"baseline_plus_{f}"]
        delta_max = (v["max"] - base_max) * 100
        marker = "↑" if delta_max > 0.1 else ("↓" if delta_max < -0.1 else "=")
        lines.append(f"| +{f} | {v['n_features']} | {v['best_iter']} | "
                     f"{v['mean']*100:.2f}% | {v['std']*100:.2f}pp | {v['max']*100:.2f}% **({delta_max:+.2f}pp {marker})** | "
                     f"{v['q99']*100:.2f}% | {v['q99_9']*100:.2f}% | {v['feature_importance_of_added']:.4f} |\n")
    full = results["baseline_plus_all_down_only"]
    delta_max = (full["max"] - base_max) * 100
    lines.append(f"| **+ALL 6** | {full['n_features']} | {full['best_iter']} | "
                 f"{full['mean']*100:.2f}% | {full['std']*100:.2f}pp | {full['max']*100:.2f}% **({delta_max:+.2f}pp)** | "
                 f"{full['q99']*100:.2f}% | {full['q99_9']*100:.2f}% | - |\n")
    OUT.write_text("".join(lines), encoding="utf-8")
    print(f"\n[ok] -> {OUT}")

    print(f"\n=== SUMMARY ===")
    print(f"{'config':<50} {'max':>8} {'delta_max':>10} {'importance':>10}")
    print(f"{'baseline_F (intersect+is_weekend)':<50} {base['max']*100:>7.2f}% {' ':>10}")
    for f in down_only:
        v = results[f"baseline_plus_{f}"]
        delta = (v["max"] - base_max) * 100
        print(f"{'+'+f:<50} {v['max']*100:>7.2f}% {delta:>+9.2f}pp {v['feature_importance_of_added']:>10.4f}")
    print(f"{'+all_6_down_only':<50} {full['max']*100:>7.2f}% {(full['max']-base_max)*100:>+9.2f}pp")


if __name__ == "__main__":
    main()
