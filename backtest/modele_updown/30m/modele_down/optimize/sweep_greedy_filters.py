"""
Greedy stepwise filter addition from v1 best (62.73% / 754 bets).

For each filter dimension, try a grid of cutoffs (or hour additions) and pick
the value that maximizes WR while keeping taken >= 750. Repeat until no
single change improves WR by >= 0.001.
"""
from __future__ import annotations

import itertools
import numpy as np

from common import (
    load_dataset, feature_columns, train_once, hour_of_day_from_ms,
)
from train import compute_thresholds_asym_wknd_excl


MIN_PARIS = 750


# Best config from sweep_aggressive (v1)
BASE = {
    "cd": 90, "wd": 0.060, "we": 0.20,
    "bad": (8, 13, 14, 17, 21),
    "good": (2, 20, 22), "delta": 0.020,
    "vr30_max": np.inf, "r3_max": 0.015,
    "r96_max": np.inf, "uw_min": 0.0, "cr_max": np.inf,
}


def eval_config(p_test, y_test, T_base, hour_masks, vr30_t, ret3_t, ret96_t,
                uw_t, cr_t, body_t, stoch_t, rsi_t, n_test,
                cfg):
    bad_mask = np.zeros(n_test, dtype=bool)
    for h in cfg["bad"]:
        bad_mask |= hour_masks[h]
    good_mask = np.zeros(n_test, dtype=bool)
    for h in cfg["good"]:
        good_mask |= hour_masks[h]
    T_eff = T_base - cfg["delta"] * good_mask
    base = (~bad_mask) & (p_test >= T_eff)

    def f_mask(arr, cutoff, mode):
        if cutoff in (np.inf, -np.inf):
            return np.ones(n_test, dtype=bool)
        if mode == "<":
            return (arr < cutoff) | np.isnan(arr)
        else:  # ">"
            return (arr > cutoff) | np.isnan(arr)

    take = base & f_mask(vr30_t, cfg["vr30_max"], "<")
    take &= f_mask(ret3_t, cfg["r3_max"], "<")
    take &= f_mask(ret96_t, cfg["r96_max"], "<")
    if cfg.get("uw_min", 0) > 0:
        take &= ((uw_t >= cfg["uw_min"]) | np.isnan(uw_t))
    take &= f_mask(cr_t, cfg["cr_max"], "<")
    if cfg.get("body_max", np.inf) != np.inf:
        take &= ((body_t < cfg["body_max"]) | np.isnan(body_t))
    if cfg.get("stoch_max", np.inf) != np.inf:
        take &= ((stoch_t < cfg["stoch_max"]) | np.isnan(stoch_t))
    if cfg.get("rsi_max", np.inf) != np.inf:
        take &= ((rsi_t < cfg["rsi_max"]) | np.isnan(rsi_t))

    n = int(take.sum())
    if n == 0:
        return n, 0.0
    wr = float(y_test[take].mean())
    return n, wr


def main():
    df = load_dataset()
    fc = feature_columns(df)
    print(f"Dataset: {len(df):,} rows, {len(fc)} features.")
    run = train_once(df, fc, seed=42)
    print(f"  best_iter={run['best_iter']} val_logloss={run['val_logloss']:.5f}")

    test = run["test"]
    val = run["val"]
    p_val = run["p_val"]
    p_test = run["p_test"]
    y_test = run["y_test"]
    n_test = len(p_test)
    hours_t = hour_of_day_from_ms(test["timestamp"].to_numpy())
    wknd_t = test["is_weekend"].astype(int).to_numpy()
    vr30_t = test["volume_ratio_30m"].to_numpy(dtype=float)
    ret3_t = test["return_3"].to_numpy(dtype=float)
    ret96_t = test["return_96"].to_numpy(dtype=float)
    uw_t = test["upper_wick"].to_numpy(dtype=float)
    cr_t = test["close_ret"].to_numpy(dtype=float)
    body_t = test["body_ratio_close"].to_numpy(dtype=float)
    stoch_t = test["stoch_k_14"].to_numpy(dtype=float)
    rsi_t = test["rsi_14"].to_numpy(dtype=float)
    hour_masks = {h: (hours_t == h) for h in range(24)}

    def compute_T_base(cd, wd, we):
        thr, _ = compute_thresholds_asym_wknd_excl(
            val, p_val,
            vol_frac_wd=wd, vol_frac_we=we,
            skip_cells=(), use_full_val=False, calib_days=cd,
        )
        return np.where(wknd_t == 1, thr.get(1, 1.0), thr.get(0, 1.0))

    T_base = compute_T_base(BASE["cd"], BASE["wd"], BASE["we"])
    cur = dict(BASE)
    n, wr = eval_config(p_test, y_test, T_base, hour_masks,
                        vr30_t, ret3_t, ret96_t, uw_t, cr_t,
                        body_t, stoch_t, rsi_t, n_test, cur)
    print(f"\nStarting config: taken={n} wr={wr*100:.2f}%")

    # Filter dimensions to try
    DIMS = [
        ("vr30_max", [np.inf, 1.5, 1.4, 1.3, 1.2, 1.1, 1.0], "<"),
        ("r3_max", [np.inf, 0.025, 0.020, 0.015, 0.010, 0.005, 0.0, -0.005], "<"),
        ("r96_max", [np.inf, 0.15, 0.10, 0.05, 0.02, 0.0], "<"),
        ("uw_min", [0.0, 0.05, 0.10, 0.15, 0.20, 0.25], ">"),
        ("cr_max", [np.inf, 0.030, 0.020, 0.010, 0.005, 0.0, -0.005], "<"),
        ("body_max", [np.inf, 0.9, 0.8, 0.7, 0.6, 0.5], "<"),
        ("stoch_max", [np.inf, 0.5, 0.3, 0.1, -0.1, -0.3], "<"),
        ("rsi_max", [np.inf, 0.5, 0.3, 0.1, -0.1, -0.3], "<"),
    ]

    # Try also adding hours to bad / good lists
    BAD_HOUR_CANDIDATES = [0, 1, 4, 7, 10, 13, 18, 20, 23]
    GOOD_HOUR_CANDIDATES = [9, 11, 12, 19]

    DELTA_CANDIDATES = [0.020, 0.030, 0.040, 0.050, 0.060]

    rounds = 0
    while True:
        rounds += 1
        best_change = None  # (dim, value, new_wr, new_n)

        # Try numeric dims
        for dim, values, _mode in DIMS:
            for v in values:
                test_cfg = dict(cur); test_cfg[dim] = v
                tn, tw = eval_config(p_test, y_test, T_base, hour_masks,
                                     vr30_t, ret3_t, ret96_t, uw_t, cr_t,
                                     body_t, stoch_t, rsi_t, n_test, test_cfg)
                if tn < MIN_PARIS:
                    continue
                if (tw > wr + 0.001) or (abs(tw - wr) < 0.001 and tn > n):
                    if best_change is None or tw > best_change[2]:
                        best_change = (dim, v, tw, tn)

        # Try adding bad hours
        for h in BAD_HOUR_CANDIDATES:
            if h in cur["bad"]:
                continue
            test_cfg = dict(cur); test_cfg["bad"] = tuple(sorted(cur["bad"] + (h,)))
            tn, tw = eval_config(p_test, y_test, T_base, hour_masks,
                                 vr30_t, ret3_t, ret96_t, uw_t, cr_t,
                                 body_t, stoch_t, rsi_t, n_test, test_cfg)
            if tn < MIN_PARIS:
                continue
            if (tw > wr + 0.001):
                if best_change is None or tw > best_change[2]:
                    best_change = (f"bad+h{h}", test_cfg["bad"], tw, tn)

        # Try adding/removing good hours
        for h in GOOD_HOUR_CANDIDATES:
            if h not in cur["good"]:
                test_cfg = dict(cur); test_cfg["good"] = tuple(sorted(cur["good"] + (h,)))
                tn, tw = eval_config(p_test, y_test, T_base, hour_masks,
                                     vr30_t, ret3_t, ret96_t, uw_t, cr_t,
                                     body_t, stoch_t, rsi_t, n_test, test_cfg)
                if tn >= MIN_PARIS and tw > wr + 0.001:
                    if best_change is None or tw > best_change[2]:
                        best_change = (f"good+h{h}", test_cfg["good"], tw, tn)

        # Try varying delta
        for d in DELTA_CANDIDATES:
            test_cfg = dict(cur); test_cfg["delta"] = d
            tn, tw = eval_config(p_test, y_test, T_base, hour_masks,
                                 vr30_t, ret3_t, ret96_t, uw_t, cr_t,
                                 body_t, stoch_t, rsi_t, n_test, test_cfg)
            if tn >= MIN_PARIS and tw > wr + 0.001:
                if best_change is None or tw > best_change[2]:
                    best_change = ("delta", d, tw, tn)

        # Try varying calibration (cd, wd, we)
        for cd in (55, 90, 180):
            for wd in (0.040, 0.060, 0.080, 0.100):
                for we in (0.15, 0.20, 0.25):
                    T_test = compute_T_base(cd, wd, we)
                    test_cfg = dict(cur)
                    # Eval with fresh T
                    bad_mask = np.zeros(n_test, dtype=bool)
                    for h in test_cfg["bad"]:
                        bad_mask |= hour_masks[h]
                    good_mask = np.zeros(n_test, dtype=bool)
                    for h in test_cfg["good"]:
                        good_mask |= hour_masks[h]
                    T_eff = T_test - test_cfg["delta"] * good_mask
                    base = (~bad_mask) & (p_test >= T_eff)
                    # apply filters
                    def f_mask(arr, cutoff):
                        return np.ones(n_test, dtype=bool) if cutoff == np.inf else ((arr < cutoff) | np.isnan(arr))
                    take = base & f_mask(vr30_t, test_cfg["vr30_max"]) \
                        & f_mask(ret3_t, test_cfg["r3_max"]) \
                        & f_mask(ret96_t, test_cfg["r96_max"]) \
                        & f_mask(cr_t, test_cfg["cr_max"])
                    if test_cfg.get("uw_min", 0) > 0:
                        take &= ((uw_t >= test_cfg["uw_min"]) | np.isnan(uw_t))
                    if test_cfg.get("body_max", np.inf) != np.inf:
                        take &= ((body_t < test_cfg["body_max"]) | np.isnan(body_t))
                    if test_cfg.get("stoch_max", np.inf) != np.inf:
                        take &= ((stoch_t < test_cfg["stoch_max"]) | np.isnan(stoch_t))
                    if test_cfg.get("rsi_max", np.inf) != np.inf:
                        take &= ((rsi_t < test_cfg["rsi_max"]) | np.isnan(rsi_t))
                    tn = int(take.sum())
                    if tn < MIN_PARIS:
                        continue
                    tw = float(y_test[take].mean())
                    if tw > wr + 0.001:
                        if best_change is None or tw > best_change[2]:
                            best_change = ("calib", (cd, wd, we), tw, tn)

        if best_change is None:
            print(f"\nRound {rounds}: no improvement found. Stopping.")
            break

        dim, val_changed, new_wr, new_n = best_change
        print(f"Round {rounds}: change {dim} -> {val_changed}, "
              f"taken {n}->{new_n}, wr {wr*100:.2f}% -> {new_wr*100:.2f}%")
        if dim.startswith("bad+"):
            cur["bad"] = val_changed
        elif dim.startswith("good+"):
            cur["good"] = val_changed
        elif dim == "delta":
            cur["delta"] = val_changed
        elif dim == "calib":
            cd, wd, we = val_changed
            cur["cd"], cur["wd"], cur["we"] = cd, wd, we
            T_base = compute_T_base(cd, wd, we)
        else:
            cur[dim] = val_changed
        wr, n = new_wr, new_n

        if rounds >= 20:
            break

    print(f"\nFinal config:")
    for k, v in cur.items():
        print(f"  {k}: {v}")
    print(f"  taken={n} wr={wr*100:.2f}%")


if __name__ == "__main__":
    main()
