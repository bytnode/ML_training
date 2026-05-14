"""
04_hyperparam_sweep.py - Sweep des hyperparams sur la baseline F pour voir si
on peut etendre la spread du modele DOWN au-dela de max=57.82%.

Hypotheses :
  - min_child_weight plus bas → splits plus fins → spread plus large
  - max_depth plus haut → splits plus precis → spread plus large
  - gamma plus bas → splits plus libres → spread plus large
  - n_estimators (sans early stopping) → continue d'entrainer plus longtemps
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "04_hyperparam_sweep.md"
DOWN_PARQUET = ROOT.parent / "db" / "btc_multi_tf_15m.parquet"

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


def fit(train, val, test, feature_cols, **hp_overrides):
    hp = dict(
        n_estimators=1000, max_depth=5, learning_rate=0.05,
        min_child_weight=20, subsample=0.8, colsample_bytree=0.8,
        gamma=1.5,
        objective="binary:logistic", eval_metric="logloss",
        early_stopping_rounds=50, random_state=42, n_jobs=-1,
    )
    hp.update(hp_overrides)
    spw = auto_spw(train["label"])
    m = xgb.XGBClassifier(scale_pos_weight=spw, **hp)
    m.fit(train[feature_cols], train["label"],
          eval_set=[(val[feature_cols], val["label"])], verbose=False)
    p_test = m.predict_proba(test[feature_cols])[:, 1]
    val_logloss = float(m.evals_result()["validation_0"]["logloss"][m.best_iteration])
    return p_test, m, val_logloss


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

    common = ['acc_4', 'autocorr_24', 'bb_position', 'close_ret', 'close_ret_1h',
              'close_ret_30m', 'close_ret_5m', 'dist_ema_20', 'dist_to_low_50',
              'high_ret', 'high_ret_5m', 'low_ret_1h', 'low_ret_30m', 'low_ret_5m',
              'macd', 'macd_hist', 'mfi_norm', 'return_3']
    baseline_features = common + ['is_weekend']

    sweeps = []

    # Baseline
    print(f"[baseline] hp default")
    p, m, ll = fit(train, val, test, baseline_features)
    sweeps.append(("baseline", {**stats(p), "best_iter": int(m.best_iteration), "val_logloss": ll}))

    # Vary min_child_weight
    for mcw in [5, 10, 30, 50, 100]:
        print(f"[mcw={mcw}]")
        p, m, ll = fit(train, val, test, baseline_features, min_child_weight=mcw)
        sweeps.append((f"min_child_weight={mcw}",
                        {**stats(p), "best_iter": int(m.best_iteration), "val_logloss": ll}))

    # Vary max_depth
    for d in [3, 4, 6, 7, 8]:
        print(f"[max_depth={d}]")
        p, m, ll = fit(train, val, test, baseline_features, max_depth=d)
        sweeps.append((f"max_depth={d}",
                        {**stats(p), "best_iter": int(m.best_iteration), "val_logloss": ll}))

    # Vary gamma
    for g in [0.0, 0.5, 1.0, 2.0, 3.0]:
        print(f"[gamma={g}]")
        p, m, ll = fit(train, val, test, baseline_features, gamma=g)
        sweeps.append((f"gamma={g}",
                        {**stats(p), "best_iter": int(m.best_iteration), "val_logloss": ll}))

    # Vary learning rate
    for lr in [0.01, 0.02, 0.10, 0.20]:
        print(f"[learning_rate={lr}]")
        p, m, ll = fit(train, val, test, baseline_features, learning_rate=lr)
        sweeps.append((f"learning_rate={lr}",
                        {**stats(p), "best_iter": int(m.best_iteration), "val_logloss": ll}))

    # No early stopping (= force more trees)
    print(f"[no_early_stop n=1000]")
    p, m, ll = fit(train, val, test, baseline_features, early_stopping_rounds=2000)
    sweeps.append(("no_early_stop_n1000",
                    {**stats(p), "best_iter": int(m.best_iteration), "val_logloss": ll}))

    # Ensemble : various seeds
    print(f"[ensemble 5 seeds]")
    p_ens = np.zeros(len(test))
    val_ll_ens = []
    for seed in [42, 123, 456, 789, 1001]:
        p, m, ll = fit(train, val, test, baseline_features, random_state=seed)
        p_ens += p
        val_ll_ens.append(ll)
    p_ens /= 5
    sweeps.append(("ensemble_5_seeds_avg",
                    {**stats(p_ens), "best_iter": -1, "val_logloss": np.mean(val_ll_ens)}))

    # Report
    base_max = sweeps[0][1]["max"]
    lines = ["# Hyperparam sweep sur DOWN baseline F\n\n"]
    lines.append("## Setup\n")
    lines.append("- Features = intersect (18) + is_weekend = 19\n")
    lines.append("- Variation 1 hyperparam a la fois (cetereis paribus)\n\n")
    lines.append(f"| config | best_iter | val_ll | mean | std | max | Δmax | q99 | q99.9 |\n")
    lines.append(f"|---|---|---|---|---|---|---|---|---|\n")
    for name, s in sweeps:
        delta = (s["max"] - base_max) * 100
        marker = "↑" if delta > 0.2 else ("↓" if delta < -0.2 else "=")
        lines.append(f"| {name} | {s['best_iter']} | {s['val_logloss']:.5f} | "
                     f"{s['mean']*100:.2f}% | {s['std']*100:.2f}pp | "
                     f"**{s['max']*100:.2f}%** | {delta:+.2f}pp {marker} | "
                     f"{s['q99']*100:.2f}% | {s['q99_9']*100:.2f}% |\n")

    OUT.write_text("".join(lines), encoding="utf-8")
    print(f"\n[ok] -> {OUT}")

    print(f"\n=== SUMMARY (sorted by max desc) ===")
    sweeps_sorted = sorted(sweeps, key=lambda x: -x[1]["max"])
    for name, s in sweeps_sorted:
        print(f"  {name:<30}  max={s['max']*100:.2f}%  std={s['std']*100:.2f}pp  val_ll={s['val_logloss']:.5f}  best_iter={s['best_iter']}")


if __name__ == "__main__":
    main()
