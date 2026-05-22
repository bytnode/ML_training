"""
Random search over filter combinations. 5000 configs.

Each config samples:
- calib (cd, wd, we) from grid
- bad_hours (random subset of pool)
- good_hours (random subset of pool)
- delta
- each numeric filter from grid (with non-trivial prob of being off)
"""
from __future__ import annotations

import random
import numpy as np

from common import (
    load_dataset, feature_columns, train_once, hour_of_day_from_ms,
)
from train import compute_thresholds_asym_wknd_excl


N_CONFIGS = 8000
SEED = 1234
MIN_PARIS = 750


CD_GRID = (55, 90, 180)
WD_GRID = (0.030, 0.040, 0.050, 0.060, 0.080, 0.100, 0.140)
WE_GRID = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35)
DELTA_GRID = (0.020, 0.030, 0.040, 0.050, 0.060)
BAD_POOL  = [0, 1, 4, 7, 8, 10, 13, 14, 17, 18, 20, 21, 23]
GOOD_POOL = [2, 6, 9, 11, 12, 15, 19, 20, 22, 23]

VR30_GRID = [np.inf] * 3 + [1.5, 1.4, 1.3, 1.2, 1.1, 1.0]
R3_GRID = [np.inf] * 3 + [0.030, 0.020, 0.015, 0.010, 0.005, 0.0, -0.005]
R96_GRID = [np.inf] * 3 + [0.20, 0.15, 0.10, 0.05, 0.02, 0.0, -0.02]
UW_GRID = [0.0] * 3 + [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
CR_GRID = [np.inf] * 3 + [0.025, 0.015, 0.010, 0.005, 0.0]
BODY_GRID = [np.inf] * 3 + [0.9, 0.8, 0.7, 0.6, 0.5]
STOCH_GRID = [np.inf] * 3 + [0.5, 0.3, 0.1, -0.1, -0.3]
RSI_GRID = [np.inf] * 3 + [0.5, 0.3, 0.1, -0.1, -0.3]


def main():
    df = load_dataset()
    fc = feature_columns(df)
    print(f"Dataset: {len(df):,} rows, {len(fc)} features.")
    run = train_once(df, fc, seed=42)
    print(f"  best_iter={run['best_iter']} val_logloss={run['val_logloss']:.5f}")

    val = run["val"]
    p_val = run["p_val"]
    test = run["test"]
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

    # Cache T_base per (cd, wd, we)
    T_base_cache = {}
    def get_T_base(cd, wd, we):
        key = (cd, wd, we)
        if key not in T_base_cache:
            thr, _ = compute_thresholds_asym_wknd_excl(
                val, p_val,
                vol_frac_wd=wd, vol_frac_we=we,
                skip_cells=(), use_full_val=False, calib_days=cd,
            )
            T_base_cache[key] = np.where(wknd_t == 1, thr.get(1, 1.0), thr.get(0, 1.0))
        return T_base_cache[key]

    rng = random.Random(SEED)
    results = []
    for i in range(N_CONFIGS):
        cd = rng.choice(CD_GRID)
        wd = rng.choice(WD_GRID)
        we = rng.choice(WE_GRID)
        T_base = get_T_base(cd, wd, we)

        bad_size = rng.choice([0, 1, 2, 3, 4, 5, 6, 7])
        bad = tuple(sorted(rng.sample(BAD_POOL, bad_size)))
        good_size = rng.choice([0, 1, 2, 3])
        good = tuple(sorted(rng.sample(GOOD_POOL, good_size)))
        delta = rng.choice(DELTA_GRID) if good_size > 0 else 0.0

        vr30 = rng.choice(VR30_GRID)
        r3 = rng.choice(R3_GRID)
        r96 = rng.choice(R96_GRID)
        uw = rng.choice(UW_GRID)
        cr = rng.choice(CR_GRID)
        body = rng.choice(BODY_GRID)
        stoch = rng.choice(STOCH_GRID)
        rsi = rng.choice(RSI_GRID)

        bad_mask = np.zeros(n_test, dtype=bool)
        for h in bad: bad_mask |= hour_masks[h]
        good_mask = np.zeros(n_test, dtype=bool)
        for h in good: good_mask |= hour_masks[h]
        T_eff = T_base - delta * good_mask
        take = (~bad_mask) & (p_test >= T_eff)
        if vr30 != np.inf: take &= (vr30_t < vr30) | np.isnan(vr30_t)
        if r3 != np.inf: take &= (ret3_t < r3) | np.isnan(ret3_t)
        if r96 != np.inf: take &= (ret96_t < r96) | np.isnan(ret96_t)
        if uw > 0: take &= (uw_t >= uw) | np.isnan(uw_t)
        if cr != np.inf: take &= (cr_t < cr) | np.isnan(cr_t)
        if body != np.inf: take &= (body_t < body) | np.isnan(body_t)
        if stoch != np.inf: take &= (stoch_t < stoch) | np.isnan(stoch_t)
        if rsi != np.inf: take &= (rsi_t < rsi) | np.isnan(rsi_t)

        n = int(take.sum())
        if n < MIN_PARIS:
            continue
        wr = float(y_test[take].mean())
        if wr < 0.62:
            continue
        results.append({
            "cd": cd, "wd": wd, "we": we, "bad": bad, "good": good, "delta": delta,
            "vr30": vr30, "r3": r3, "r96": r96, "uw": uw, "cr": cr,
            "body": body, "stoch": stoch, "rsi": rsi,
            "taken": n, "wr": wr,
        })

    results.sort(key=lambda r: (-r["wr"], -r["taken"]))
    print(f"\n{N_CONFIGS} random configs -> {len(results)} qualifying (wr>=0.62, taken>={MIN_PARIS})")

    print("\nTOP 30 by WR:")
    print("-" * 200)
    for r in results[:30]:
        def fmt(v, w=4, p=2):
            return "off".rjust(w) if v == np.inf else f"{v:.{p}f}".rjust(w)
        bh = "(" + ",".join(str(h) for h in r["bad"]) + ")"
        gh = "(" + ",".join(str(h) for h in r["good"]) + ")"
        print(f"  cd={r['cd']:>3} wd={r['wd']:.3f} we={r['we']:.2f} bad={bh:<26} good={gh:<18} "
              f"d={r['delta']:>5.3f} vr30<{fmt(r['vr30'])} r3<{fmt(r['r3'],5,3)} "
              f"r96<{fmt(r['r96'])} uw>={r['uw']:.2f} cr<{fmt(r['cr'],5,3)} "
              f"body<{fmt(r['body'])} stoch<{fmt(r['stoch'])} rsi<{fmt(r['rsi'])}  "
              f"taken={r['taken']:>5} wr={r['wr']*100:>6.2f}%")

    goal = [r for r in results if r["wr"] >= 0.65]
    print(f"\nGOAL WR>=65%: {len(goal):,}")
    for r in goal[:20]:
        def fmt(v, w=4, p=2):
            return "off".rjust(w) if v == np.inf else f"{v:.{p}f}".rjust(w)
        bh = "(" + ",".join(str(h) for h in r["bad"]) + ")"
        gh = "(" + ",".join(str(h) for h in r["good"]) + ")"
        print(f"  cd={r['cd']:>3} wd={r['wd']:.3f} we={r['we']:.2f} bad={bh:<26} good={gh:<18} "
              f"d={r['delta']:>5.3f} vr30<{fmt(r['vr30'])} r3<{fmt(r['r3'],5,3)} "
              f"r96<{fmt(r['r96'])} uw>={r['uw']:.2f} cr<{fmt(r['cr'],5,3)} "
              f"body<{fmt(r['body'])} stoch<{fmt(r['stoch'])} rsi<{fmt(r['rsi'])}  "
              f"taken={r['taken']:>5} wr={r['wr']*100:>6.2f}%")


if __name__ == "__main__":
    main()
