"""
Try different TRAIN_DAYS values. Less history may match current crypto regime better.
"""
from __future__ import annotations
import sys
from pathlib import Path
import time
import numpy as np
import pandas as pd

_PKG_ROOT = Path(__file__).resolve().parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from sklearn.metrics import log_loss
from common import (
    load_dataset, feature_columns, build_model, evaluate_with_calibration,
)
from config import TEST_DAYS, VAL_FRACTION, XGB_RANDOM_STATE
from train import compute_thresholds_asym_wknd_excl, apply_thresholds_asym_hour_boost


WD_GRID = (0.025, 0.030, 0.035, 0.040)
WE_GRID = (0.20, 0.22, 0.25, 0.28, 0.30)
GOOD_HOURS_GRID = [(20,), (20, 21), (20, 21, 22)]
DELTA_GRID = (0.020, 0.025, 0.030, 0.035)
MIN_PARIS = 750


def split_with_train_days(df, train_days):
    df = df.sort_values("timestamp").reset_index(drop=True)
    ms_per_day = 86_400_000
    ts_max = df["timestamp"].max()
    test_cutoff = ts_max - TEST_DAYS * ms_per_day
    test_start = int(df["timestamp"].searchsorted(test_cutoff, side="left"))
    if train_days is None:
        tv_start = 0
    else:
        tv_cutoff = test_cutoff - train_days * ms_per_day
        tv_start = int(df["timestamp"].searchsorted(tv_cutoff, side="left"))
    n_tv = test_start - tv_start
    val_size = int(n_tv * VAL_FRACTION)
    train_end = test_start - val_size
    return df.iloc[tv_start:train_end], df.iloc[train_end:test_start], df.iloc[test_start:]


def train_with_days(df, fcols, train_days):
    train, val, test = split_with_train_days(df, train_days)
    X_train, y_train = train[fcols], train["label"]
    X_val, y_val = val[fcols], val["label"]
    X_test, y_test = test[fcols], test["label"]
    model = build_model(y_train, seed=XGB_RANDOM_STATE)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    p_val = model.predict_proba(X_val)[:, 1]
    p_test = model.predict_proba(X_test)[:, 1]
    return {
        "model": model, "best_iter": int(model.best_iteration),
        "val_logloss": float(model.evals_result()["validation_0"]["logloss"][int(model.best_iteration)]),
        "train": train, "val": val, "test": test,
        "p_val": p_val, "p_test": p_test,
        "y_test": y_test.to_numpy().astype(int),
        "n_train": len(train), "n_val": len(val), "n_test": len(test),
    }


def best_calibration_for_run(run):
    best = None
    for wd in WD_GRID:
        for we in WE_GRID:
            for gh in GOOD_HOURS_GRID:
                for d in DELTA_GRID:
                    res = evaluate_with_calibration(run, wd, we,
                                                    good_hours=gh,
                                                    hour_boost_delta=d)
                    if res["taken"] < MIN_PARIS:
                        continue
                    if best is None or res["wr"] > best["wr"]:
                        best = {**res, "wd": wd, "we": we, "gh": gh, "delta": d}
    return best


def main():
    df = load_dataset()
    fcols = feature_columns(df)
    print(f"=== TRAIN_DAYS sweep === ({len(fcols)} features)\n")

    train_days_grid = [None, 5000, 4500, 4000, 3600, 3000, 2500, 2000, 1500, 1000, 750]
    results = []
    for td in train_days_grid:
        t0 = time.time()
        run = train_with_days(df, fcols, td)
        best = best_calibration_for_run(run)
        elapsed = time.time() - t0
        if best is None:
            print(f"  TD={td:>5} n_train={run['n_train']:>7,} : no 750+ config ({elapsed:.1f}s)")
            continue
        results.append({"td": td, "n_train": run["n_train"],
                        "best_iter": run["best_iter"],
                        "val_ll": run["val_logloss"], **best})
        marker = " <-- MISSION" if best["wr"] >= 0.60 else ""
        print(f"  TD={str(td):>5} n_train={run['n_train']:>7,} bi={run['best_iter']:>3} "
              f"vll={run['val_logloss']:.5f} -> "
              f"taken={best['taken']} WR={best['wr']*100:.2f}% EV={best['ev']:+.2f} "
              f"(wd={best['wd']:.3f} we={best['we']:.3f} gh={best['gh']} d={best['delta']:.3f}){marker}")

    results.sort(key=lambda r: -r["wr"])
    print(f"\n=== Top 5 by WR ===")
    for r in results[:5]:
        print(f"  TD={r['td']} n_train={r['n_train']:,}  taken={r['taken']} "
              f"WR={r['wr']*100:.2f}% EV={r['ev']:+.2f}")


if __name__ == "__main__":
    main()
