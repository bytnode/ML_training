"""
Sweep scale_pos_weight. Auto = n_neg/n_pos = ~1.06.

Hypothesis: with SPW < 1 (down-weight DOWN class), the model has to be
extra-confident before predicting label=1. This may push P(down) higher
when it does fire, lifting WR on confident bets.

Best calib from previous sweeps applied to each SPW run.
"""
from __future__ import annotations

import numpy as np

from common import (
    load_dataset, feature_columns, train_once, hour_of_day_from_ms,
)
from train import compute_thresholds_asym_wknd_excl


CALIB = {
    "ufv": False, "cd": 90, "wd": 0.040, "we": 0.20,
    "bad": (8, 13, 17, 20, 21), "good": (2, 15), "delta": 0.040,
}
SPW_GRID = (0.3, 0.5, 0.7, 0.85, "auto", 1.2, 1.5, 2.0)


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
    n = int(take.sum())
    wr = float(y_test[take].mean()) if n > 0 else 0.0
    return n, wr, float(thr.get(0, 1.0)), float(thr.get(1, 1.0))


def main():
    df = load_dataset()
    fc = feature_columns(df)
    print(f"Dataset: {len(df):,} rows, {len(fc)} features.\n")
    print(f"{'spw':>10} {'taken':>6} {'wr':>7} {'p_min':>7} {'p_max':>7} {'val_ll':>8}")
    for spw in SPW_GRID:
        hp_over = {"scale_pos_weight": spw}
        run = train_once(df, fc, seed=42, hp_overrides=hp_over)
        n, wr, T_wd, T_we = eval_with(run, CALIB)
        p_test = run["p_test"]
        print(f"{str(spw):>10} {n:>6} {wr*100:>6.2f}% {p_test.min():>7.4f} {p_test.max():>7.4f} {run['val_logloss']:>8.5f}")


if __name__ == "__main__":
    main()
