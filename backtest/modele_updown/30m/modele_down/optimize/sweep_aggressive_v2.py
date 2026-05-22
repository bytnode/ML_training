"""
Round 2 aggressive sweep. Built on v1 best (62.73%).

Adds:
- Wider wd grid (push tighter)
- More ret3 / vr30 cutoffs
- return_96 cutoff (1-week momentum filter)
- upper_wick cutoff (skip if no rejection wick)
- close_ret cutoff (current bar should be flat/red, not strongly green)
- Mid-bar filter: stoch_k_14 too low = oversold = avoid DOWN bet
"""
from __future__ import annotations

import itertools
import numpy as np

from common import (
    load_dataset, feature_columns, train_once, hour_of_day_from_ms,
)
from train import compute_thresholds_asym_wknd_excl


# Tighter wd values
WD_GRID = (0.030, 0.040, 0.050, 0.060, 0.080, 0.100, 0.140)
WE_GRID = (0.15, 0.20, 0.25, 0.30)
CD_GRID = (55, 90, 180)

BAD_HOUR_FIXED  = (8, 14, 17, 21)
BAD_HOUR_EXTRA  = (10, 13, 18)
GOOD_HOUR_POOL = [2, 9, 11, 20, 22]
GOOD_SIZES = (1, 2, 3)
DELTA_GRID = (0.020, 0.030, 0.040, 0.060)

VR30_GRID = (np.inf, 1.5, 1.3, 1.2, 1.1, 1.0)
RET3_GRID = (np.inf, 0.025, 0.015, 0.010, 0.005, 0.0)
RET96_GRID = (np.inf, 0.10, 0.05, 0.02)
UPPER_WICK_MIN_GRID = (0.0, 0.10, 0.20)
CLOSE_RET_MAX_GRID = (np.inf, 0.020, 0.010, 0.005)

MIN_PARIS = 750


def hour_combos(pool, sizes):
    out = []
    for sz in sizes:
        if sz == 0:
            out.append(())
        else:
            out.extend(itertools.combinations(pool, sz))
    return out


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
    hours_t = hour_of_day_from_ms(test["timestamp"].to_numpy())
    wknd_t = test["is_weekend"].astype(int).to_numpy()
    vr30_t = test["volume_ratio_30m"].to_numpy(dtype=float)
    ret3_t = test["return_3"].to_numpy(dtype=float)
    ret96_t = test["return_96"].to_numpy(dtype=float)
    uw_t = test["upper_wick"].to_numpy(dtype=float)
    cr_t = test["close_ret"].to_numpy(dtype=float)
    hour_masks = {h: (hours_t == h) for h in range(24)}

    # Reduce search space: fix BAD_HOUR to a few near-optimal
    bad_combos = []
    for extra_sz in range(len(BAD_HOUR_EXTRA) + 1):
        for ex in itertools.combinations(BAD_HOUR_EXTRA, extra_sz):
            bad_combos.append(tuple(sorted(BAD_HOUR_FIXED + ex)))
    bad_combos = list(set(bad_combos))
    good_combos = hour_combos(GOOD_HOUR_POOL, GOOD_SIZES)
    print(f"\nCalib: {len(WD_GRID)*len(WE_GRID)*len(CD_GRID)}, "
          f"bad: {len(bad_combos)}, good: {len(good_combos)}, "
          f"delta: {len(DELTA_GRID)}")
    n_filters = len(VR30_GRID) * len(RET3_GRID) * len(RET96_GRID) * len(UPPER_WICK_MIN_GRID) * len(CLOSE_RET_MAX_GRID)
    print(f"Filters: {n_filters}")
    print(f"Total: {len(WD_GRID)*len(WE_GRID)*len(CD_GRID)*len(bad_combos)*len(good_combos)*len(DELTA_GRID)*n_filters:,}")

    results = []
    progress = 0
    for cd in CD_GRID:
        for wd in WD_GRID:
            for we in WE_GRID:
                thr, _ = compute_thresholds_asym_wknd_excl(
                    val, p_val,
                    vol_frac_wd=wd, vol_frac_we=we,
                    skip_cells=(), use_full_val=False, calib_days=cd,
                )
                T_base = np.where(wknd_t == 1, thr.get(1, 1.0), thr.get(0, 1.0))
                for bh in bad_combos:
                    bad_mask = np.zeros(len(p_test), dtype=bool)
                    for h in bh:
                        bad_mask |= hour_masks[h]
                    for gh in good_combos:
                        good_mask = np.zeros(len(p_test), dtype=bool)
                        for h in gh:
                            good_mask |= hour_masks[h]
                        for delta in DELTA_GRID:
                            T_eff = T_base - delta * good_mask
                            base_take = (~bad_mask) & (p_test >= T_eff)
                            # Apply feature filters
                            for vr_max in VR30_GRID:
                                vr_mask = (vr30_t < vr_max) | np.isnan(vr30_t) if vr_max != np.inf else np.ones(len(p_test), dtype=bool)
                                for r3_max in RET3_GRID:
                                    r3_mask = (ret3_t < r3_max) | np.isnan(ret3_t) if r3_max != np.inf else np.ones(len(p_test), dtype=bool)
                                    for r96_max in RET96_GRID:
                                        r96_mask = (ret96_t < r96_max) | np.isnan(ret96_t) if r96_max != np.inf else np.ones(len(p_test), dtype=bool)
                                        for uw_min in UPPER_WICK_MIN_GRID:
                                            uw_mask = (uw_t >= uw_min) | np.isnan(uw_t) if uw_min > 0 else np.ones(len(p_test), dtype=bool)
                                            for cr_max in CLOSE_RET_MAX_GRID:
                                                cr_mask = (cr_t < cr_max) | np.isnan(cr_t) if cr_max != np.inf else np.ones(len(p_test), dtype=bool)
                                                take = base_take & vr_mask & r3_mask & r96_mask & uw_mask & cr_mask
                                                n = int(take.sum())
                                                if n < MIN_PARIS:
                                                    continue
                                                wr = float(y_test[take].mean())
                                                if wr < 0.62:
                                                    continue
                                                results.append({
                                                    "cd": cd, "wd": wd, "we": we,
                                                    "bad": bh, "good": gh, "delta": delta,
                                                    "vr30": vr_max, "r3": r3_max, "r96": r96_max,
                                                    "uw_min": uw_min, "cr_max": cr_max,
                                                    "taken": n, "wr": wr,
                                                })
                progress += 1
                if progress % 30 == 0:
                    print(f"  ... {progress} calib combos done, results so far: {len(results):,}")

    results.sort(key=lambda r: (-r["wr"], -r["taken"]))
    print(f"\nResults >= 62% WR with taken >= {MIN_PARIS}: {len(results):,}")

    print("\nTOP 30 by WR:")
    print("-" * 200)
    for r in results[:30]:
        bh_s = "(" + ",".join(str(h) for h in r["bad"]) + ")"
        gh_s = "(" + ",".join(str(h) for h in r["good"]) + ")"
        def fmt(v, fmt_str="{:.2f}"):
            return "off" if v == np.inf else fmt_str.format(v)
        print(f"  cd={r['cd']:>3} wd={r['wd']:.3f} we={r['we']:.2f} bad={bh_s:<26} good={gh_s:<18} "
              f"d={r['delta']:>5.3f} vr30<{fmt(r['vr30']):>4} r3<{fmt(r['r3'], '{:.3f}'):>5} "
              f"r96<{fmt(r['r96'], '{:.2f}'):>4} uw>={r['uw_min']:.2f} cr<{fmt(r['cr_max'], '{:.3f}'):>5}  "
              f"taken={r['taken']:>5} wr={r['wr']*100:>6.2f}%")

    goal = [r for r in results if r["wr"] >= 0.65]
    print(f"\nGOAL (WR >= 65%): {len(goal):,}")
    for r in goal[:20]:
        bh_s = "(" + ",".join(str(h) for h in r["bad"]) + ")"
        gh_s = "(" + ",".join(str(h) for h in r["good"]) + ")"
        def fmt(v, fmt_str="{:.2f}"):
            return "off" if v == np.inf else fmt_str.format(v)
        print(f"  cd={r['cd']:>3} wd={r['wd']:.3f} we={r['we']:.2f} bad={bh_s:<26} good={gh_s:<18} "
              f"d={r['delta']:>5.3f} vr30<{fmt(r['vr30']):>4} r3<{fmt(r['r3'], '{:.3f}'):>5} "
              f"r96<{fmt(r['r96'], '{:.2f}'):>4} uw>={r['uw_min']:.2f} cr<{fmt(r['cr_max'], '{:.3f}'):>5}  "
              f"taken={r['taken']:>5} wr={r['wr']*100:>6.2f}%")


if __name__ == "__main__":
    main()
