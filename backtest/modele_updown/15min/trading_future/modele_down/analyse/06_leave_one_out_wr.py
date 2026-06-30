"""
06_leave_one_out_wr.py - Test "leave-one-out" : retire chaque feature
individuellement (parmi les 25 features actuelles) et mesure l'impact sur la
WR + n_taken de la strategie complete.

Si retirer une feature AUGMENTE la WR : cette feature nuit au modele.
Si retirer une feature DIMINUE la WR : cette feature aide le modele.

Cible : trouver les features qui nuisent reellement a la capacite du modele
(pas juste celles qui compriment les probas).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "06_leave_one_out_wr.md"
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


def fit(train, val, test, feature_cols):
    hp = dict(
        n_estimators=1000, max_depth=5, learning_rate=0.05,
        min_child_weight=20, subsample=0.8, colsample_bytree=0.8, gamma=1.5,
        objective="binary:logistic", eval_metric="logloss",
        early_stopping_rounds=50, random_state=42, n_jobs=-1,
    )
    spw = auto_spw(train["label"])
    m = xgb.XGBClassifier(scale_pos_weight=spw, **hp)
    m.fit(train[feature_cols], train["label"],
          eval_set=[(val[feature_cols], val["label"])], verbose=False)
    return m.predict_proba(val[feature_cols])[:, 1], m.predict_proba(test[feature_cols])[:, 1], m


def hour_utc(ts_ms):
    ts_ms = np.asarray(ts_ms, dtype=np.int64)
    return ((ts_ms // (15 * 60 * 1000)) % 96 // 4).astype(int)


def apply_strategy(val, test, p_val, p_test):
    skip_set = set((int(v), int(w)) for (v, w) in ASYM_SKIP_CELLS)
    good_hours = set(ASYM_HOUR_BOOST_GOOD_HOURS)
    val_wknd = val["is_weekend"].to_numpy().astype(int)
    val_vol = val["volatility_state"].to_numpy().astype(int)
    non_skip_val = np.array([(int(v), int(w)) not in skip_set for v, w in zip(val_vol, val_wknd)])
    mask_wd = non_skip_val & (val_wknd == 0)
    mask_we = non_skip_val & (val_wknd == 1)
    T_wd = float(np.quantile(p_val[mask_wd], 1 - ASYM_VOL_FRAC_WEEKDAY))
    T_we = float(np.quantile(p_val[mask_we], 1 - ASYM_VOL_FRAC_WEEKEND))
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
    n = int(take.sum())
    wr = float(test_y[take].mean()) if n > 0 else 0.0
    ev = n * (wr - 0.5)
    return {"n": n, "wr": wr, "ev": ev,
            "p_max": float(p_test.max()),
            "p_mean_taken": float(p_test[take].mean()) if n > 0 else 0.0,
            "T_wd": T_wd, "T_we": T_we}


def main():
    df_d = load_parquet(DOWN_PARQUET)
    train, val, test = split(df_d)
    print(f"DOWN split: train={len(train):,} val={len(val):,} test={len(test):,}")

    all_features = [c for c in df_d.columns
                     if c not in ("timestamp", "label", "volatility_state")]
    print(f"\nAll features ({len(all_features)}): {all_features}")

    # Baseline = current state (all features)
    print(f"\n[baseline] all {len(all_features)} features")
    p_val, p_test, m = fit(train, val, test, all_features)
    base = apply_strategy(val, test, p_val, p_test)
    base["importance"] = dict(zip(all_features, m.feature_importances_.tolist()))
    base["best_iter"] = int(m.best_iteration)
    print(f"  baseline: n={base['n']} WR={base['wr']*100:.2f}% EV={base['ev']:+.2f}  p_max={base['p_max']*100:.2f}%")

    # Leave one out (LOO) for every feature
    results = []
    for f in all_features:
        feats = [x for x in all_features if x != f]
        p_val, p_test, m = fit(train, val, test, feats)
        s = apply_strategy(val, test, p_val, p_test)
        delta_wr = (s["wr"] - base["wr"]) * 100
        delta_n = s["n"] - base["n"]
        delta_ev = s["ev"] - base["ev"]
        s["delta_wr"] = delta_wr
        s["delta_n"] = delta_n
        s["delta_ev"] = delta_ev
        s["importance"] = base["importance"][f]
        s["removed"] = f
        s["best_iter"] = int(m.best_iteration)
        results.append(s)
        flag = "+" if delta_wr > 0.2 else ("-" if delta_wr < -0.2 else "=")
        print(f"  [LOO -{f:>25}] n={s['n']:>5} WR={s['wr']*100:.2f}% ({delta_wr:+.2f}pp {flag}) EV={s['ev']:+.2f} ({delta_ev:+.2f})")

    # Sort by delta_wr descending : features that HURT WR (= removing them HELPS) at top
    results_sorted = sorted(results, key=lambda x: -x["delta_wr"])

    lines = ["# Leave-One-Out feature impact sur WR (modele DOWN)\n\n"]
    lines.append("## Setup\n")
    lines.append(f"- Baseline = config CURRENT ({len(all_features)} features, max_depth=5)\n")
    lines.append(f"- Strategie : quantile (vw={ASYM_VOL_FRAC_WEEKDAY}, we={ASYM_VOL_FRAC_WEEKEND}) + hour_boost\n")
    lines.append(f"- Baseline: n={base['n']} WR={base['wr']*100:.2f}% EV={base['ev']:+.2f}  p_max={base['p_max']*100:.2f}%\n\n")
    lines.append("## Resultats LOO (trie par Δ WR descendant = retirer LES PIRES en haut)\n\n")
    lines.append("| feature retiree | importance | n_taken | Δn | WR | **Δ WR** | EV | Δ EV | p_max | verdict |\n")
    lines.append("|---|---|---|---|---|---|---|---|---|---|\n")
    for s in results_sorted:
        verdict = "HURTS (retire = +)" if s["delta_wr"] > 0.2 else ("HELPS (retire = -)" if s["delta_wr"] < -0.2 else "neutral")
        lines.append(
            f"| `{s['removed']}` | {s['importance']:.4f} | {s['n']} | {s['delta_n']:+d} | "
            f"{s['wr']*100:.2f}% | **{s['delta_wr']:+.2f}pp** | "
            f"{s['ev']:+.2f} | {s['delta_ev']:+.2f} | {s['p_max']*100:.2f}% | {verdict} |\n"
        )

    # Top hurts / helps
    hurts = [s for s in results if s["delta_wr"] > 0.2]
    helps = [s for s in results if s["delta_wr"] < -0.2]
    lines.append(f"\n## Synthese\n")
    lines.append(f"- **Features qui NUISENT** (retirer ameliore WR de plus de 0.2pp) : **{len(hurts)}**\n")
    for s in hurts:
        lines.append(f"  - `{s['removed']}` : retirer → WR {base['wr']*100:.2f}% → {s['wr']*100:.2f}% ({s['delta_wr']:+.2f}pp)\n")
    lines.append(f"- **Features qui AIDENT** (retirer baisse WR de plus de 0.2pp) : **{len(helps)}**\n")
    for s in helps:
        lines.append(f"  - `{s['removed']}` : retirer → WR {base['wr']*100:.2f}% → {s['wr']*100:.2f}% ({s['delta_wr']:+.2f}pp)\n")

    OUT.write_text("".join(lines), encoding="utf-8")
    print(f"\n[ok] -> {OUT}")
    print(f"\n=== HURTS ({len(hurts)} features) — retirer ameliore WR ===")
    for s in hurts:
        print(f"  -{s['removed']:<25}: Δ WR = {s['delta_wr']:+.2f}pp")
    print(f"\n=== HELPS ({len(helps)} features) — retirer baisse WR ===")
    for s in helps:
        print(f"  -{s['removed']:<25}: Δ WR = {s['delta_wr']:+.2f}pp")


if __name__ == "__main__":
    main()
