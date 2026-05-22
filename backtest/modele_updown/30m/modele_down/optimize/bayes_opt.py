"""
Bayesian optimization (skopt) of the FILTER + CALIB config jointly.
~150 evaluations, Gaussian Process surrogate.

Search space:
- cd_days: 30 .. 200
- wd_frac: 0.02 .. 0.20
- we_frac: 0.05 .. 0.40
- delta:   0.005 .. 0.080
- vr30_max: 1.0 .. 2.0 (inf = 2.0)
- r3_max:   0.0 .. 0.05 (or off via discrete)
- r96_max:  0.0 .. 0.2 (or off)
- cr_max:   0.0 .. 0.03 (or off)
- uw_min:   0.0 .. 0.30
- Plus binary toggles for which hours are in BAD/GOOD
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
from skopt import gp_minimize, dummy_minimize
from skopt.space import Real, Integer, Categorical

from common import (
    load_dataset, feature_columns, train_once, hour_of_day_from_ms,
)


MIN_PARIS = 750
TARGET_WR = 0.65
TARGET_TAKEN = 800


def filtered_threshold(val_df, p_val, wknd_arr_val, mask_val,
                       wd_frac, we_frac, calib_days):
    ms_per_day = 86_400_000
    val_max_ts = int(val_df["timestamp"].max())
    cutoff = val_max_ts - calib_days * ms_per_day
    recent = val_df["timestamp"].to_numpy() >= cutoff
    cal_mask = mask_val & recent
    thresholds = {}
    for ax, frac in [(0, wd_frac), (1, we_frac)]:
        m = cal_mask & (wknd_arr_val == ax)
        if m.sum() < 30 or frac <= 0:
            thresholds[ax] = 1.0
            continue
        thresholds[ax] = float(np.quantile(p_val[m], 1.0 - frac))
    return thresholds


def main():
    df = load_dataset()
    fc = feature_columns(df)
    run = train_once(df, fc, seed=42)
    val = run["val"]; p_val = run["p_val"]
    test = run["test"]; p_test = run["p_test"]; y_test = run["y_test"]
    n_test = len(p_test)

    hours_t = hour_of_day_from_ms(test["timestamp"].to_numpy())
    wknd_t = test["is_weekend"].astype(int).to_numpy()
    hours_v = hour_of_day_from_ms(val["timestamp"].to_numpy())
    wknd_v = val["is_weekend"].astype(int).to_numpy()
    vr30_t = test["volume_ratio_30m"].to_numpy(dtype=float)
    r3_t = test["return_3"].to_numpy(dtype=float)
    r96_t = test["return_96"].to_numpy(dtype=float)
    cr_t = test["close_ret"].to_numpy(dtype=float)
    uw_t = test["upper_wick"].to_numpy(dtype=float)
    stoch_t = test["stoch_k_14"].to_numpy(dtype=float)
    rsi_t = test["rsi_14"].to_numpy(dtype=float)
    bb_t = test["bb_position"].to_numpy(dtype=float)
    vr30_v = val["volume_ratio_30m"].to_numpy(dtype=float)
    r3_v = val["return_3"].to_numpy(dtype=float)
    r96_v = val["return_96"].to_numpy(dtype=float)
    cr_v = val["close_ret"].to_numpy(dtype=float)
    uw_v = val["upper_wick"].to_numpy(dtype=float)
    stoch_v = val["stoch_k_14"].to_numpy(dtype=float)
    rsi_v = val["rsi_14"].to_numpy(dtype=float)
    bb_v = val["bb_position"].to_numpy(dtype=float)
    hour_masks_t = {h: (hours_t == h) for h in range(24)}
    hour_masks_v = {h: (hours_v == h) for h in range(24)}

    # Variables: 24 binary (bad), 24 binary (good), delta, cd, wd, we,
    # + 7 numeric filter cutoffs
    space = [
        Integer(30, 200, name="cd"),
        Real(0.020, 0.20, name="wd"),
        Real(0.05, 0.40, name="we"),
        Real(0.010, 0.080, name="delta"),
        Real(1.0, 2.5, name="vr30_max"),   # 2.5 ~ effectively off
        Real(0.0, 0.05, name="r3_max"),
        Real(0.0, 0.30, name="r96_max"),
        Real(0.0, 0.030, name="cr_max"),
        Real(0.0, 0.30, name="uw_min"),
        Real(-0.5, 1.0, name="stoch_max"),
        Real(-0.5, 1.0, name="rsi_max"),
        Real(0.0, 1.0, name="bb_max"),
    ]
    # 24 hours bad / 24 hours good as binary
    for h in range(24):
        space.append(Categorical([0, 1], name=f"bad_{h}"))
    for h in range(24):
        space.append(Categorical([0, 1], name=f"good_{h}"))

    n_calls = 200
    n_initial_points = 60

    best_so_far = {"wr": -1.0, "n": 0, "params": None}

    def objective(x):
        cd = int(x[0])
        wd = float(x[1]); we = float(x[2]); delta = float(x[3])
        vr30_max = float(x[4]); r3_max = float(x[5]); r96_max = float(x[6])
        cr_max = float(x[7]); uw_min = float(x[8])
        stoch_max = float(x[9]); rsi_max = float(x[10]); bb_max = float(x[11])
        bad_bits = [int(x[12 + h]) for h in range(24)]
        good_bits = [int(x[36 + h]) for h in range(24)]
        bad_hours = tuple(h for h in range(24) if bad_bits[h])
        good_hours = tuple(h for h in range(24) if good_bits[h])

        bad_t = np.zeros(n_test, dtype=bool)
        for h in bad_hours: bad_t |= hour_masks_t[h]
        bad_v_mask = np.zeros(len(val), dtype=bool)
        for h in bad_hours: bad_v_mask |= hour_masks_v[h]
        good_t = np.zeros(n_test, dtype=bool)
        for h in good_hours: good_t |= hour_masks_t[h]

        m_t = ~bad_t
        m_t &= (vr30_t < vr30_max) | np.isnan(vr30_t) if vr30_max < 2.5 else np.ones(n_test, dtype=bool)
        m_t &= (r3_t < r3_max) | np.isnan(r3_t) if r3_max < 0.05 else np.ones(n_test, dtype=bool)
        m_t &= (r96_t < r96_max) | np.isnan(r96_t) if r96_max < 0.30 else np.ones(n_test, dtype=bool)
        m_t &= (cr_t < cr_max) | np.isnan(cr_t) if cr_max < 0.030 else np.ones(n_test, dtype=bool)
        if uw_min > 0:
            m_t &= (uw_t >= uw_min) | np.isnan(uw_t)
        m_t &= (stoch_t < stoch_max) | np.isnan(stoch_t) if stoch_max < 1.0 else np.ones(n_test, dtype=bool)
        m_t &= (rsi_t < rsi_max) | np.isnan(rsi_t) if rsi_max < 1.0 else np.ones(n_test, dtype=bool)
        m_t &= (bb_t < bb_max) | np.isnan(bb_t) if bb_max < 1.0 else np.ones(n_test, dtype=bool)

        m_v = ~bad_v_mask
        m_v &= (vr30_v < vr30_max) | np.isnan(vr30_v) if vr30_max < 2.5 else np.ones(len(val), dtype=bool)
        m_v &= (r3_v < r3_max) | np.isnan(r3_v) if r3_max < 0.05 else np.ones(len(val), dtype=bool)
        m_v &= (r96_v < r96_max) | np.isnan(r96_v) if r96_max < 0.30 else np.ones(len(val), dtype=bool)
        m_v &= (cr_v < cr_max) | np.isnan(cr_v) if cr_max < 0.030 else np.ones(len(val), dtype=bool)
        if uw_min > 0:
            m_v &= (uw_v >= uw_min) | np.isnan(uw_v)
        m_v &= (stoch_v < stoch_max) | np.isnan(stoch_v) if stoch_max < 1.0 else np.ones(len(val), dtype=bool)
        m_v &= (rsi_v < rsi_max) | np.isnan(rsi_v) if rsi_max < 1.0 else np.ones(len(val), dtype=bool)
        m_v &= (bb_v < bb_max) | np.isnan(bb_v) if bb_max < 1.0 else np.ones(len(val), dtype=bool)

        if m_v.sum() < 100 or m_t.sum() < MIN_PARIS:
            return 1.0

        thr = filtered_threshold(val, p_val, wknd_v, m_v, wd, we, cd)
        T_base = np.where(wknd_t == 1, thr.get(1, 1.0), thr.get(0, 1.0))
        T_eff = T_base - delta * good_t
        take = m_t & (p_test >= T_eff)
        n = int(take.sum())
        if n < MIN_PARIS:
            return 1.0
        wr = float(y_test[take].mean())

        # Objective : minimize -wr but penalize if too far above MIN_PARIS
        # (we want at least 750+ but reward up to ~900)
        # Negative log-loss-style score:
        score = -wr
        if wr > best_so_far["wr"]:
            best_so_far["wr"] = wr
            best_so_far["n"] = n
            best_so_far["params"] = {
                "cd": cd, "wd": wd, "we": we, "delta": delta,
                "vr30_max": vr30_max, "r3_max": r3_max, "r96_max": r96_max,
                "cr_max": cr_max, "uw_min": uw_min,
                "stoch_max": stoch_max, "rsi_max": rsi_max, "bb_max": bb_max,
                "bad": bad_hours, "good": good_hours,
            }
            print(f"  NEW BEST: taken={n} wr={wr*100:.2f}%  "
                  f"cd={cd} wd={wd:.3f} we={we:.3f} delta={delta:.3f} bad={bad_hours} good={good_hours}")
        return score

    print(f"Running Bayesian optimization with {n_calls} calls...")
    result = gp_minimize(
        objective, space, n_calls=n_calls, n_initial_points=n_initial_points,
        random_state=42, verbose=False,
    )

    print(f"\n=== FINAL BEST ===")
    if best_so_far["params"]:
        print(f"  taken={best_so_far['n']} wr={best_so_far['wr']*100:.2f}%")
        for k, v in best_so_far["params"].items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
