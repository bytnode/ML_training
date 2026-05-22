"""
Sweep TRAIN_DAYS. BTC microstructure has changed massively over 13 years;
short recent training windows may produce sharper signal.

For each TRAIN_DAYS, train, then apply best calibration combo. Print WR / taken.
"""
from __future__ import annotations

import numpy as np

from common import (
    load_dataset, feature_columns, train_once, hour_of_day_from_ms,
)
import config as config_mod  # noqa: E402 (common adds path)
from train import compute_thresholds_asym_wknd_excl


CALIB = {
    "ufv": False, "cd": 90, "wd": 0.040, "we": 0.20,
    "bad": (8, 13, 17, 20, 21), "good": (2, 15), "delta": 0.040,
}
TRAIN_DAYS_GRID = (500, 800, 1200, 1800, 2400, 3000, 3600, 4500, None)


def eval_with(run, calib):
    test = run["test"]; val = run["val"]
    p_val = run["p_val"]; p_test = run["p_test"]; y_test = run["y_test"]
    hours_t = hour_of_day_from_ms(test["timestamp"].to_numpy())
    wknd_t = test["is_weekend"].astype(int).to_numpy()

    thr, _ = compute_thresholds_asym_wknd_excl(
        val, p_val,
        vol_frac_wd=calib["wd"], vol_frac_we=calib["we"],
        skip_cells=(), use_full_val=calib["ufv"], calib_days=calib["cd"],
    )
    T_base = np.where(wknd_t == 1, thr.get(1, 1.0), thr.get(0, 1.0))
    bad_mask = np.isin(hours_t, list(calib["bad"]))
    good_mask = np.isin(hours_t, list(calib["good"]))
    T_eff = T_base - calib["delta"] * good_mask
    take = (~bad_mask) & (p_test >= T_eff)
    n_taken = int(take.sum())
    wr = float(y_test[take].mean()) if n_taken > 0 else 0.0
    return n_taken, wr, float(thr.get(0, 1.0)), float(thr.get(1, 1.0))


def main():
    df = load_dataset()
    fc = feature_columns(df)
    print(f"Dataset: {len(df):,} rows, {len(fc)} features.\n")

    rows = []
    print(f"{'TRAIN_DAYS':>10} {'taken':>6} {'wr':>7} {'ev':>7} {'val_ll':>8} {'best_it':>7}")
    for td in TRAIN_DAYS_GRID:
        orig_td = config_mod.TRAIN_DAYS
        config_mod.TRAIN_DAYS = td
        # The common module imports TRAIN_DAYS at load — need to reload
        import importlib, common as common_mod
        importlib.reload(common_mod)
        run = common_mod.train_once(df, fc, seed=42)
        n, wr, T_wd, T_we = eval_with(run, CALIB)
        ev = n * (wr - 0.5)
        rows.append((td, n, wr, ev, run["val_logloss"], run["best_iter"]))
        td_str = "ALL" if td is None else str(td)
        print(f"{td_str:>10} {n:>6} {wr*100:>6.2f}% {ev:>7.1f} "
              f"{run['val_logloss']:>8.5f} {run['best_iter']:>7}")
        config_mod.TRAIN_DAYS = orig_td


if __name__ == "__main__":
    main()
