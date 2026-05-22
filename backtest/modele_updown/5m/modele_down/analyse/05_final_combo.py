"""
05_final_combo.py - Test du combo optimal trouve : features intersect+is_weekend
+ max_depth=6, avec la strategie V2 complete (quantile + hour_boost).

Compare a la production actuelle (toutes features + max_depth=5).

But : confirmer que le changement ameliore reellement les stats P(down)
des paris pris et la performance.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "05_final_combo.md"
DOWN_PARQUET = ROOT.parent / "db" / "btc_multi_tf_15m.parquet"

TRAIN_DAYS = 4000
TEST_DAYS = 365
VAL_FRACTION = 0.1
WARMUP = 500
ASYM_VOL_FRAC_WEEKDAY = 0.01
ASYM_VOL_FRAC_WEEKEND = 0.10
ASYM_SKIP_CELLS = ((2, 0), (2, 1))
ASYM_HOUR_BOOST_GOOD_HOURS = (20, 21, 22)
ASYM_HOUR_BOOST_DELTA = 0.020


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
    p_val = m.predict_proba(val[feature_cols])[:, 1]
    p_test = m.predict_proba(test[feature_cols])[:, 1]
    return p_val, p_test, m


def hour_utc(ts_ms):
    ts_ms = np.asarray(ts_ms, dtype=np.int64)
    return ((ts_ms // (15 * 60 * 1000)) % 96 // 4).astype(int)


def apply_strategy(val, test, p_val, p_test):
    """Calibration quantile + hour boost + skip cells. Returns dict de stats."""
    skip_set = set((int(v), int(w)) for (v, w) in ASYM_SKIP_CELLS)
    good_hours = set(ASYM_HOUR_BOOST_GOOD_HOURS)

    val_wknd = val["is_weekend"].to_numpy().astype(int)
    val_vol = val["volatility_state"].to_numpy().astype(int)
    val_y = val["label"].to_numpy().astype(int)
    non_skip_val = np.array([(int(v), int(w)) not in skip_set for v, w in zip(val_vol, val_wknd)])

    # Quantile per axis
    mask_wd = non_skip_val & (val_wknd == 0)
    mask_we = non_skip_val & (val_wknd == 1)
    T_wd = float(np.quantile(p_val[mask_wd], 1 - ASYM_VOL_FRAC_WEEKDAY))
    T_we = float(np.quantile(p_val[mask_we], 1 - ASYM_VOL_FRAC_WEEKEND))

    # Apply on test
    test_wknd = test["is_weekend"].to_numpy().astype(int)
    test_vol = test["volatility_state"].to_numpy().astype(int)
    test_y = test["label"].to_numpy().astype(int)
    test_hour = hour_utc(test["timestamp"].to_numpy())

    eff_th = np.full(len(test), 2.0)
    for i in range(len(test)):
        if (int(test_vol[i]), int(test_wknd[i])) in skip_set:
            continue
        T = T_wd if test_wknd[i] == 0 else T_we
        if int(test_hour[i]) in good_hours:
            T -= ASYM_HOUR_BOOST_DELTA
        eff_th[i] = T

    take = p_test >= eff_th
    n_taken = int(take.sum())
    wr = float(test_y[take].mean()) if n_taken > 0 else 0.0
    p_mean_taken = float(p_test[take].mean()) if n_taken > 0 else 0.0
    win_mask = take & (test_y == 1)
    loss_mask = take & (test_y == 0)
    p_mean_wins = float(p_test[win_mask].mean()) if win_mask.any() else 0.0
    p_mean_loss = float(p_test[loss_mask].mean()) if loss_mask.any() else 0.0

    return {
        "T_wd": T_wd, "T_we": T_we,
        "n_taken": n_taken, "pct_vol": n_taken / len(test) * 100, "wr": wr,
        "p_max": float(p_test.max()), "p_std": float(p_test.std()),
        "p_mean_taken": p_mean_taken,
        "p_mean_wins": p_mean_wins, "p_mean_loss": p_mean_loss,
        "n_wins": int(win_mask.sum()), "n_loss": int(loss_mask.sum()),
    }


def main():
    df_d = load_parquet(DOWN_PARQUET)
    train, val, test = split(df_d)
    print(f"DOWN split: train={len(train):,} val={len(val):,} test={len(test):,}")

    # All features (= current production state, max_depth=5)
    features_full = [c for c in df_d.columns
                      if c not in ("timestamp", "label", "volatility_state")]

    # Intersect features + is_weekend
    common = ['acc_4', 'autocorr_24', 'bb_position', 'close_ret', 'close_ret_1h',
              'close_ret_30m', 'close_ret_5m', 'dist_ema_20', 'dist_to_low_50',
              'high_ret', 'high_ret_5m', 'low_ret_1h', 'low_ret_30m', 'low_ret_5m',
              'macd', 'macd_hist', 'mfi_norm', 'return_3']
    features_clean = common + ['is_weekend']

    configs = [
        ("CURRENT (full features, max_depth=5)", features_full, dict(max_depth=5)),
        ("clean features, max_depth=5", features_clean, dict(max_depth=5)),
        ("clean features, max_depth=6", features_clean, dict(max_depth=6)),
        ("clean features, max_depth=7", features_clean, dict(max_depth=7)),
        ("clean features, max_depth=8", features_clean, dict(max_depth=8)),
        ("full features, max_depth=6", features_full, dict(max_depth=6)),
    ]

    results = []
    for name, feats, hp in configs:
        print(f"\n[{name}]")
        p_val, p_test, m = fit(train, val, test, feats, **hp)
        strat = apply_strategy(val, test, p_val, p_test)
        print(f"  n_features={len(feats)}, max_depth={hp.get('max_depth')}, best_iter={m.best_iteration}")
        print(f"  T_wd={strat['T_wd']:.4f}, T_we={strat['T_we']:.4f}")
        print(f"  n_taken={strat['n_taken']} ({strat['pct_vol']:.2f}%) WR={strat['wr']*100:.2f}%")
        print(f"  p_mean_taken={strat['p_mean_taken']*100:.2f}%, max={strat['p_max']*100:.2f}%, std={strat['p_std']*100:.2f}pp")
        print(f"  p_wins={strat['p_mean_wins']*100:.2f}% ({strat['n_wins']}), p_loss={strat['p_mean_loss']*100:.2f}% ({strat['n_loss']})")
        results.append((name, feats, hp, m, strat))

    # Write report
    lines = ["# Combo final : ablation features + max_depth\n\n"]
    lines.append("## Strategie\n")
    lines.append(f"- vol_frac_weekday = {ASYM_VOL_FRAC_WEEKDAY}, vol_frac_weekend = {ASYM_VOL_FRAC_WEEKEND}\n")
    lines.append(f"- skip = {list(ASYM_SKIP_CELLS)}, hour_boost = {list(ASYM_HOUR_BOOST_GOOD_HOURS)} @ {ASYM_HOUR_BOOST_DELTA}\n\n")
    lines.append("## Resultats\n\n")
    lines.append("| config | n_feat | best_iter | T_wd | T_we | n_taken | %vol | WR | p_mean_taken | p_max | p_wins/loss |\n")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|\n")
    for name, feats, hp, m, s in results:
        lines.append(
            f"| {name} | {len(feats)} | {m.best_iteration} | "
            f"{s['T_wd']:.4f} | {s['T_we']:.4f} | "
            f"{s['n_taken']} | {s['pct_vol']:.2f}% | **{s['wr']*100:.2f}%** | "
            f"**{s['p_mean_taken']*100:.2f}%** | {s['p_max']*100:.2f}% | "
            f"{s['p_mean_wins']*100:.2f}/{s['p_mean_loss']*100:.2f}% |\n"
        )

    OUT.write_text("".join(lines), encoding="utf-8")
    print(f"\n[ok] -> {OUT}")


if __name__ == "__main__":
    main()
