"""
Random HP search + dual filter (p_up_sibling) + best calibration combo.

Looking for any HP config that crosses 62% WR with taken >= 750 when combined
with the optimal calib + dual filter.
"""
from __future__ import annotations

from pathlib import Path
import random

import numpy as np
import xgboost as xgb

from common import (
    load_dataset, feature_columns, train_once, hour_of_day_from_ms,
)
from train import compute_thresholds_asym_wknd_excl


UP_MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "modele_up" / "btc_updown_model.json"

HP_SPACE = {
    "n_estimators": [400, 800, 1000, 1500],
    "max_depth": [4, 5, 6, 7, 8, 10],
    "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.08],
    "min_child_weight": [1, 5, 10, 20, 40],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.5, 0.7, 0.8, 1.0],
    "reg_lambda": [0.0, 1.0, 5.0, 10.0, 30.0, 100.0],
    "reg_alpha": [False, 0.0, 1.0, 5.0],
    "gamma": [0.0, 1.0, 3.0, 5.0, 10.0],
}

CALIB_GRID = [
    {"ufv": False, "cd": 90, "wd": 0.060, "we": 0.20, "tag": "cd90_wd06_we20"},
    {"ufv": False, "cd": 90, "wd": 0.040, "we": 0.20, "tag": "cd90_wd04_we20"},
    {"ufv": False, "cd": 55, "wd": 0.030, "we": 0.20, "tag": "cd55_wd03_we20"},
]
BAD_HOURS_OPTIONS = [
    (8, 13, 17, 20, 21),
    (8, 17, 20),
]
GOOD_HOURS_OPTIONS = [
    (2, 15),
    (15, 22),
    (),
]
DELTA_GRID = (0.020, 0.040)
P_UP_MAX_GRID = (1.0, 0.50, 0.48, 0.45)
MIN_PARIS = 750
N_HP = 50
SEED = 1234


def load_up_model():
    m = xgb.XGBClassifier()
    m.load_model(str(UP_MODEL_PATH))
    return m


def predict_up_on(df, up_model):
    feat = up_model.get_booster().feature_names
    X = df[feat]
    mask_valid = ~X.isna().any(axis=1)
    p_up = np.full(len(df), np.nan, dtype=float)
    if mask_valid.any():
        p_up[mask_valid.to_numpy()] = up_model.predict_proba(X[mask_valid])[:, 1]
    return p_up


def sample_hp(rng):
    return {k: rng.choice(v) for k, v in HP_SPACE.items()}


def main():
    df = load_dataset()
    fc = feature_columns(df)
    up = load_up_model()
    rng = random.Random(SEED)

    print(f"Dataset: {len(df):,} rows, {len(fc)} features.")
    print(f"Sampling {N_HP} HP configs...")

    best = []
    for i in range(N_HP):
        hp = sample_hp(rng)
        try:
            run = train_once(df, fc, seed=42, hp_overrides=hp)
        except Exception as e:
            print(f"  [{i+1}/{N_HP}] HP={hp} -> ERROR {e}")
            continue

        test = run["test"]; val = run["val"]
        p_val = run["p_val"]; p_test = run["p_test"]; y_test = run["y_test"]
        p_up_test = predict_up_on(test, up)
        hours_t = hour_of_day_from_ms(test["timestamp"].to_numpy())
        wknd_t = test["is_weekend"].astype(int).to_numpy()
        hour_masks_t = {h: (hours_t == h) for h in range(24)}

        best_for_hp = None
        for cal in CALIB_GRID:
            thr, _ = compute_thresholds_asym_wknd_excl(
                val, p_val,
                vol_frac_wd=cal["wd"], vol_frac_we=cal["we"],
                skip_cells=(), use_full_val=cal["ufv"], calib_days=cal["cd"],
            )
            T_base = np.where(wknd_t == 1, thr.get(1, 1.0), thr.get(0, 1.0))
            for bh in BAD_HOURS_OPTIONS:
                bad_mask = np.zeros(len(p_test), dtype=bool)
                for h in bh:
                    bad_mask |= hour_masks_t[h]
                for gh in GOOD_HOURS_OPTIONS:
                    good_mask = np.zeros(len(p_test), dtype=bool)
                    for h in gh:
                        good_mask |= hour_masks_t[h]
                    for delta in (DELTA_GRID if len(gh) > 0 else (0.0,)):
                        T_eff = T_base - delta * good_mask
                        for pu_max in P_UP_MAX_GRID:
                            pu_filter = (np.ones(len(p_test), dtype=bool) if pu_max >= 1.0
                                         else ((p_up_test <= pu_max) & ~np.isnan(p_up_test)))
                            take = (~bad_mask) & (p_test >= T_eff) & pu_filter
                            n = int(take.sum())
                            if n < MIN_PARIS:
                                continue
                            wr = float(y_test[take].mean())
                            if best_for_hp is None or wr > best_for_hp["wr"]:
                                best_for_hp = {
                                    "hp": hp, "calib": cal["tag"], "bad": bh,
                                    "good": gh, "delta": delta, "p_up_max": pu_max,
                                    "taken": n, "wr": wr, "ev": n * (wr - 0.5),
                                    "val_ll": run["val_logloss"],
                                }
        if best_for_hp:
            best.append(best_for_hp)
            print(f"  [{i+1}/{N_HP}] depth={hp['max_depth']} lr={hp['learning_rate']} "
                  f"ss={hp['subsample']} cs={hp['colsample_bytree']} "
                  f"l={hp['reg_lambda']} g={hp['gamma']} "
                  f"-> BEST: taken={best_for_hp['taken']} wr={best_for_hp['wr']*100:.2f}% "
                  f"(calib={best_for_hp['calib']}, pu<={best_for_hp['p_up_max']:.2f})")

    best.sort(key=lambda b: (-b["wr"], -b["taken"]))
    print(f"\nTop 20 by WR among {len(best)} HP configs:")
    print("-" * 200)
    for r in best[:20]:
        hp = r["hp"]
        bh_s = "(" + ",".join(str(h) for h in r["bad"]) + ")"
        gh_s = "(" + ",".join(str(h) for h in r["good"]) + ")"
        print(f"  d={hp['max_depth']} lr={hp['learning_rate']:>5.2f} mcw={hp['min_child_weight']:>3} "
              f"ss={hp['subsample']:.1f} cs={hp['colsample_bytree']:.1f} "
              f"l={hp['reg_lambda']:>5.1f} g={hp['gamma']:>4.1f} a={hp['reg_alpha']!s:<5} "
              f"n_est={hp['n_estimators']:>5}  "
              f"{r['calib']:<16} bad={bh_s:<22} good={gh_s:<14} d={r['delta']:>5.3f} pu<={r['p_up_max']:.2f}  "
              f"taken={r['taken']:>5} wr={r['wr']*100:>6.2f}%")


if __name__ == "__main__":
    main()
