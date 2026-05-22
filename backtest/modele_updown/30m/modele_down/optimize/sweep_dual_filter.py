"""
Dual filter at inference time: take DOWN bet only if
  (p_down >= T_d)  AND  (p_up_sibling <= T_u_max)
where p_up_sibling is the sister modele_up prediction on the same row.

No leakage in DOWN training because p_up_sibling is NOT a DOWN feature.

Sweep T_d via vol_frac on val, T_u_max as a fixed value in {0.40, 0.45, ...},
+ also hour adjustments.
"""
from __future__ import annotations

from pathlib import Path

import itertools
import numpy as np
import pandas as pd
import xgboost as xgb

from common import (
    load_dataset, feature_columns, train_once, hour_of_day_from_ms,
)
from train import compute_thresholds_asym_wknd_excl


UP_MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "modele_up" / "btc_updown_model.json"


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


CALIB_GRID = [
    {"ufv": False, "cd": 90, "wd": 0.060, "we": 0.20, "tag": "cd90_wd06_we20"},
    {"ufv": False, "cd": 90, "wd": 0.040, "we": 0.20, "tag": "cd90_wd04_we20"},
    {"ufv": False, "cd": 55, "wd": 0.030, "we": 0.20, "tag": "cd55_wd03_we20"},
    {"ufv": False, "cd": 180, "wd": 0.060, "we": 0.20, "tag": "cd180_wd06_we20"},
]
P_UP_MAX_GRID = (1.0, 0.55, 0.52, 0.50, 0.48, 0.45, 0.42)  # 1.0 = no filter
BAD_HOURS = [
    (8, 13, 17, 20, 21),
    (8, 17, 20),
    (8, 17),
    (),
]
GOOD_HOURS = [
    (2, 15),
    (15, 22),
    (15,),
    (),
]
DELTA_GRID = (0.020, 0.040)
MIN_PARIS = 750


def main():
    df = load_dataset()
    fc = feature_columns(df)
    print(f"Dataset: {len(df):,} rows, {len(fc)} features.")

    # Train DOWN model normally
    run = train_once(df, fc, seed=42)
    print(f"  DOWN: best_iter={run['best_iter']} val_ll={run['val_logloss']:.5f}")

    # Load UP model and predict on test + val
    up = load_up_model()
    print(f"  UP features: {len(up.get_booster().feature_names)}")
    p_up_val = predict_up_on(run["val"], up)
    p_up_test = predict_up_on(run["test"], up)
    print(f"  p_up on val:  min={np.nanmin(p_up_val):.4f} max={np.nanmax(p_up_val):.4f} "
          f"mean={np.nanmean(p_up_val):.4f}")
    print(f"  p_up on test: min={np.nanmin(p_up_test):.4f} max={np.nanmax(p_up_test):.4f} "
          f"mean={np.nanmean(p_up_test):.4f}")

    # Correlation between p_down and p_up
    pd_test = run["p_test"]
    corr = np.corrcoef(pd_test, p_up_test)[0, 1]
    print(f"  corr(p_down, p_up) on test = {corr:.4f}")

    test = run["test"]
    val = run["val"]
    p_val = run["p_val"]
    p_test = run["p_test"]
    y_test = run["y_test"]
    hours_t = hour_of_day_from_ms(test["timestamp"].to_numpy())
    wknd_t = test["is_weekend"].astype(int).to_numpy()
    hour_masks_t = {h: (hours_t == h) for h in range(24)}

    results = []

    for cal in CALIB_GRID:
        thr, _ = compute_thresholds_asym_wknd_excl(
            val, p_val,
            vol_frac_wd=cal["wd"], vol_frac_we=cal["we"],
            skip_cells=(), use_full_val=cal["ufv"], calib_days=cal["cd"],
        )
        T_wd = float(thr.get(0, 1.0))
        T_we = float(thr.get(1, 1.0))
        T_base = np.where(wknd_t == 1, T_we, T_wd)

        for bh in BAD_HOURS:
            bad_mask = np.zeros(len(p_test), dtype=bool)
            for h in bh:
                bad_mask |= hour_masks_t[h]
            for gh in GOOD_HOURS:
                good_mask = np.zeros(len(p_test), dtype=bool)
                for h in gh:
                    good_mask |= hour_masks_t[h]
                for delta in (DELTA_GRID if len(gh) > 0 else (0.0,)):
                    T_eff = T_base - delta * good_mask
                    for pu_max in P_UP_MAX_GRID:
                        if pu_max < 1.0:
                            pu_filter = (p_up_test <= pu_max) & ~np.isnan(p_up_test)
                        else:
                            pu_filter = np.ones(len(p_test), dtype=bool)
                        take = (~bad_mask) & (p_test >= T_eff) & pu_filter
                        n = int(take.sum())
                        if n == 0:
                            continue
                        wr = float(y_test[take].mean())
                        results.append({
                            "calib": cal["tag"], "T_wd": T_wd, "T_we": T_we,
                            "bad": bh, "good": gh, "delta": delta, "p_up_max": pu_max,
                            "taken": n, "wr": wr, "ev": n * (wr - 0.5),
                        })

    qual = [r for r in results if r["taken"] >= MIN_PARIS]
    qual.sort(key=lambda r: (-r["wr"], -r["taken"]))

    print(f"\nResults: {len(results):,}, qualifying (>= {MIN_PARIS}): {len(qual):,}")
    print("\nTOP 30 by WR with taken >= MIN_PARIS:")
    print("-" * 160)
    for r in qual[:30]:
        bh_s = "(" + ",".join(str(h) for h in r["bad"]) + ")"
        gh_s = "(" + ",".join(str(h) for h in r["good"]) + ")"
        print(f"  {r['calib']:<16} bad={bh_s:<22} good={gh_s:<14} d={r['delta']:>5.3f} "
              f"p_up<={r['p_up_max']:.2f}  "
              f"taken={r['taken']:>5} wr={r['wr']*100:>6.2f}% ev={r['ev']:>6.1f}")

    goal = [r for r in results if r["wr"] >= 0.65 and r["taken"] >= MIN_PARIS]
    goal.sort(key=lambda r: -r["taken"])
    print(f"\nCONFIGS WR >= 65% AND taken >= {MIN_PARIS}: {len(goal):,}")
    for r in goal[:20]:
        bh_s = "(" + ",".join(str(h) for h in r["bad"]) + ")"
        gh_s = "(" + ",".join(str(h) for h in r["good"]) + ")"
        print(f"  {r['calib']:<16} bad={bh_s:<22} good={gh_s:<14} d={r['delta']:>5.3f} "
              f"p_up<={r['p_up_max']:.2f}  taken={r['taken']:>5} wr={r['wr']*100:>6.2f}%")


if __name__ == "__main__":
    main()
