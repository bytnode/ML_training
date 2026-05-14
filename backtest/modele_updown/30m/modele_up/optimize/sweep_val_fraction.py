"""Test different VAL_FRACTION values."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

_PKG_ROOT = Path(__file__).resolve().parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from common import load_dataset, feature_columns, build_model
from config import TRAIN_DAYS, TEST_DAYS, XGB_RANDOM_STATE
from train import compute_thresholds_asym_wknd_excl, apply_thresholds_asym_hour_boost

# Calibration grid mostly stable around current winner
WD_GRID = (0.018, 0.020, 0.022, 0.025, 0.030)
WE_GRID = (0.12, 0.15, 0.18, 0.20)
GH_GRID = [(20, 21)]
DELTA_GRID = (0.050, 0.060)
CD_GRID = (15, 20, 25, 30, 40, 60)
MIN_PARIS = 750


def split_for_vf(df, vf):
    df = df.sort_values("timestamp").reset_index(drop=True)
    ms_per_day = 86_400_000
    ts_max = df["timestamp"].max()
    test_cutoff = ts_max - TEST_DAYS * ms_per_day
    test_start = int(df["timestamp"].searchsorted(test_cutoff, side="left"))
    tv_cutoff = test_cutoff - TRAIN_DAYS * ms_per_day
    tv_start = int(df["timestamp"].searchsorted(tv_cutoff, side="left"))
    n_tv = test_start - tv_start
    val_size = int(n_tv * vf)
    train_end = test_start - val_size
    return df.iloc[tv_start:train_end], df.iloc[train_end:test_start], df.iloc[test_start:]


def train_for_vf(df, fcols, vf):
    train, val, test = split_for_vf(df, vf)
    X_train, y_train = train[fcols], train["label"]
    X_val, y_val = val[fcols], val["label"]
    X_test, y_test = test[fcols], test["label"]
    model = build_model(y_train, seed=XGB_RANDOM_STATE)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return {
        "model": model,
        "best_iter": int(model.best_iteration),
        "val_logloss": float(model.evals_result()["validation_0"]["logloss"][int(model.best_iteration)]),
        "n_train": len(train), "n_val": len(val), "n_test": len(test),
        "val": val, "test": test,
        "p_val": model.predict_proba(X_val)[:, 1],
        "p_test": model.predict_proba(X_test)[:, 1],
        "y_test": y_test.to_numpy().astype(int),
    }


def evaluate(run, wd, we, gh, delta, cd):
    th, _ = compute_thresholds_asym_wknd_excl(
        run["val"], run["p_val"],
        vol_frac_wd=wd, vol_frac_we=we,
        skip_cells=(), use_full_val=False, calib_days=cd,
    )
    take = apply_thresholds_asym_hour_boost(
        run["test"], run["p_test"], th,
        skip_cells=(), good_hours=gh, hour_boost_delta=delta,
    )
    n = int(take.sum())
    if n < MIN_PARIS:
        return None
    y = run["y_test"]
    wr = float(y[take].mean())
    return {"taken": n, "wr": wr, "ev": n * (wr - 0.5)}


def main():
    df = load_dataset()
    fcols = feature_columns(df)
    print(f"=== VAL_FRACTION sweep === {len(fcols)} features\n")

    vf_grid = (0.05, 0.075, 0.10, 0.15, 0.20, 0.25)
    for vf in vf_grid:
        run = train_for_vf(df, fcols, vf)
        print(f"\n  VF={vf:.3f} n_train={run['n_train']:,} n_val={run['n_val']:,} "
              f"best_iter={run['best_iter']} val_ll={run['val_logloss']:.5f}")
        best = None
        for cd in CD_GRID:
            for wd in WD_GRID:
                for we in WE_GRID:
                    for gh in GH_GRID:
                        for d in DELTA_GRID:
                            res = evaluate(run, wd, we, gh, d, cd)
                            if res is None:
                                continue
                            if best is None or res["wr"] > best["wr"]:
                                best = {**res, "wd": wd, "we": we, "gh": gh,
                                        "delta": d, "cd": cd}
        if best:
            m = " <-- MISSION" if best["wr"] >= 0.60 else ""
            print(f"  -> best : cd={best['cd']:>3} wd={best['wd']:.3f} we={best['we']:.3f} "
                  f"d={best['delta']:.3f} -> {best['taken']} {best['wr']*100:.2f}% {best['ev']:+.2f}{m}")
        else:
            print(f"  -> no 750+ config")


if __name__ == "__main__":
    main()
