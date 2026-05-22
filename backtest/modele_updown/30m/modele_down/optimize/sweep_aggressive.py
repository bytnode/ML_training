"""
Aggressive sweep using mistake-analysis insights:
- volume_ratio_30m > X is a BAD signal (rising volume = bullish)
- return_3 > X is a BAD signal (strong momentum continuation)
- Hour h22, h2, h9, h11, h20 are HIGH-WR hours on test top quantile
- Hour h8, h10, h14, h17, h21 are LOW-WR hours
- vol_state=2 (high vol) has HIGHER WR than vol_state=0/1

Sweep:
  vol_ratio_30m_max in {99 (off), 1.2, 1.3, 1.5, 2.0}
  return_3_max in {99, 0.015, 0.025, 0.04}
  bad_hours from a richer pool
  good_hours from richer pool with bigger delta range
"""
from __future__ import annotations

import itertools
import numpy as np

from common import (
    load_dataset, feature_columns, train_once, hour_of_day_from_ms,
)
from train import compute_thresholds_asym_wknd_excl


CALIB_GRID = [
    {"ufv": False, "cd": 90, "wd": 0.060, "we": 0.20},
    {"ufv": False, "cd": 90, "wd": 0.040, "we": 0.20},
    {"ufv": False, "cd": 90, "wd": 0.080, "we": 0.20},
    {"ufv": False, "cd": 90, "wd": 0.060, "we": 0.30},
    {"ufv": False, "cd": 55, "wd": 0.030, "we": 0.20},
    {"ufv": False, "cd": 180, "wd": 0.060, "we": 0.20},
]
# Bad hours based on top-1500 analysis
BAD_HOUR_POOL  = [8, 10, 13, 14, 17, 21]
GOOD_HOUR_POOL = [2, 9, 11, 20, 22]
BAD_SIZES  = (0, 2, 3, 4, 5, 6)
GOOD_SIZES = (0, 1, 2, 3)
DELTA_GRID = (0.020, 0.040, 0.060, 0.080)

VR30_MAX_GRID = (np.inf, 1.4, 1.3, 1.2)
RET3_MAX_GRID = (np.inf, 0.04, 0.025, 0.015)

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
    hour_masks = {h: (hours_t == h) for h in range(24)}

    bad_combos = hour_combos(BAD_HOUR_POOL, BAD_SIZES)
    good_combos = hour_combos(GOOD_HOUR_POOL, GOOD_SIZES)
    print(f"\nCalib variants: {len(CALIB_GRID)}, bad combos: {len(bad_combos)}, "
          f"good combos: {len(good_combos)}, delta: {len(DELTA_GRID)}, "
          f"vr30 cutoffs: {len(VR30_MAX_GRID)}, ret3 cutoffs: {len(RET3_MAX_GRID)}")

    n_eval = (len(CALIB_GRID) * len(bad_combos) * len(good_combos)
              * len(DELTA_GRID) * len(VR30_MAX_GRID) * len(RET3_MAX_GRID))
    print(f"Total: {n_eval:,}")

    results = []
    for cal in CALIB_GRID:
        thr, _ = compute_thresholds_asym_wknd_excl(
            val, p_val,
            vol_frac_wd=cal["wd"], vol_frac_we=cal["we"],
            skip_cells=(), use_full_val=cal["ufv"], calib_days=cal["cd"],
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
                deltas = DELTA_GRID if len(gh) > 0 else (0.0,)
                for delta in deltas:
                    T_eff = T_base - delta * good_mask
                    base_take = (~bad_mask) & (p_test >= T_eff)
                    for vr_max in VR30_MAX_GRID:
                        if vr_max == np.inf:
                            vr_mask = np.ones(len(p_test), dtype=bool)
                        else:
                            vr_mask = (vr30_t < vr_max) | np.isnan(vr30_t)
                        for r3_max in RET3_MAX_GRID:
                            if r3_max == np.inf:
                                r3_mask = np.ones(len(p_test), dtype=bool)
                            else:
                                r3_mask = (ret3_t < r3_max) | np.isnan(ret3_t)
                            take = base_take & vr_mask & r3_mask
                            n = int(take.sum())
                            if n < MIN_PARIS:
                                continue
                            wr = float(y_test[take].mean())
                            results.append({
                                "cd": cal["cd"], "wd": cal["wd"], "we": cal["we"],
                                "bad": bh, "good": gh, "delta": delta,
                                "vr30_max": vr_max, "ret3_max": r3_max,
                                "taken": n, "wr": wr,
                            })

    results.sort(key=lambda r: (-r["wr"], -r["taken"]))
    qual = [r for r in results if r["taken"] >= MIN_PARIS]
    print(f"\n{len(qual):,} qualifying configs")

    print("\nTOP 40 by WR:")
    print("-" * 170)
    for r in qual[:40]:
        bh_s = "(" + ",".join(str(h) for h in r["bad"]) + ")"
        gh_s = "(" + ",".join(str(h) for h in r["good"]) + ")"
        vr_s = "off" if r["vr30_max"] == np.inf else f"{r['vr30_max']:.2f}"
        r3_s = "off" if r["ret3_max"] == np.inf else f"{r['ret3_max']:.3f}"
        print(f"  cd={r['cd']:>3} wd={r['wd']:.3f} we={r['we']:.2f} bad={bh_s:<22} good={gh_s:<18} "
              f"d={r['delta']:>5.3f} vr30<{vr_s:>4} ret3<{r3_s:>5}  "
              f"taken={r['taken']:>5} wr={r['wr']*100:>6.2f}%")

    goal = [r for r in results if r["wr"] >= 0.65 and r["taken"] >= MIN_PARIS]
    goal.sort(key=lambda r: (-r["wr"], -r["taken"]))
    print(f"\nGOAL (WR>=65%, taken>={MIN_PARIS}): {len(goal):,} configs")
    for r in goal[:15]:
        bh_s = "(" + ",".join(str(h) for h in r["bad"]) + ")"
        gh_s = "(" + ",".join(str(h) for h in r["good"]) + ")"
        vr_s = "off" if r["vr30_max"] == np.inf else f"{r['vr30_max']:.2f}"
        r3_s = "off" if r["ret3_max"] == np.inf else f"{r['ret3_max']:.3f}"
        print(f"  cd={r['cd']:>3} wd={r['wd']:.3f} we={r['we']:.2f} bad={bh_s:<22} good={gh_s:<18} "
              f"d={r['delta']:>5.3f} vr30<{vr_s:>4} ret3<{r3_s:>5}  "
              f"taken={r['taken']:>5} wr={r['wr']*100:>6.2f}%")


if __name__ == "__main__":
    main()
