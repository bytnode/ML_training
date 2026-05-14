"""
Random hp search at the NEW config (spw=auto, skip=(), cd=90).
Search space refined around current winner.
"""
from __future__ import annotations
import time
import numpy as np

from common import (
    load_dataset, feature_columns, train_once,
)
from train import compute_thresholds_asym_wknd_excl, apply_thresholds_asym_hour_boost

GRID = {
    "max_depth":            [4, 5, 6, 7, 8, 10],
    "learning_rate":        [0.015, 0.020, 0.025, 0.030, 0.040, 0.050],
    "min_child_weight":     [5, 10, 15, 25, 40],
    "reg_lambda":           [1.0, 3.0, 5.0, 10.0, 20.0, 50.0],
    "reg_alpha":            [False, 0.1, 0.5, 2.0],
    "gamma":                [0.5, 1.5, 3.0, 5.0, 8.0],
    "colsample_bytree":     [0.7, 0.8, 0.9, 1.0],
    "subsample":            [0.7, 0.8, 0.9, 1.0],
    "n_estimators":         [800, 1000, 1500, 2000],
}

# Calibration grid at the new config (skip=(), cd=90)
WD_GRID = (0.025, 0.030, 0.033, 0.035, 0.040, 0.050)
WE_GRID = (0.18, 0.20, 0.22, 0.25, 0.28)
GH_GRID = [(20,), (20, 21), (20, 21, 22)]
DELTA_GRID = (0.020, 0.025, 0.030, 0.035)

MIN_PARIS = 750
CALIB_DAYS = 90


def evaluate(run, wd, we, gh, delta):
    th, _ = compute_thresholds_asym_wknd_excl(
        run["val"], run["p_val"],
        vol_frac_wd=wd, vol_frac_we=we,
        skip_cells=(), use_full_val=False, calib_days=CALIB_DAYS,
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
    return {"taken": n, "wr": wr, "ev": n * (wr - 0.5),
            "wd": wd, "we": we, "gh": gh, "delta": delta}


def best_for_run(run):
    best = None
    for wd in WD_GRID:
        for we in WE_GRID:
            for gh in GH_GRID:
                for d in DELTA_GRID:
                    res = evaluate(run, wd, we, gh, d)
                    if res is None:
                        continue
                    if best is None or res["wr"] > best["wr"]:
                        best = res
    return best


def random_sample(n, rng_seed=1):
    rng = np.random.default_rng(rng_seed)
    keys = list(GRID.keys())
    samples = []
    seen = set()
    while len(samples) < n:
        cfg = {k: GRID[k][rng.integers(0, len(GRID[k]))] for k in keys}
        sig = tuple(sorted(cfg.items()))
        if sig in seen:
            continue
        seen.add(sig)
        samples.append(cfg)
    return samples


def main(n_samples=80):
    df = load_dataset()
    fcols = feature_columns(df)
    print(f"=== HP v3 sweep (n={n_samples}, cd={CALIB_DAYS}) ===\n")
    samples = random_sample(n_samples)
    t0 = time.time()
    results = []
    for i, hp in enumerate(samples, 1):
        try:
            run = train_once(df, fcols, seed=42, hp_overrides=hp)
            best = best_for_run(run)
            if best is None:
                print(f"  [{i:3}/{n_samples}] no 750+")
                continue
            results.append({"hp": hp, **best})
            mark = " <-- MISSION" if best["wr"] >= 0.60 else ""
            print(f"  [{i:3}/{n_samples}] WR={best['wr']*100:5.2f}% taken={best['taken']:>4} "
                  f"EV={best['ev']:+6.2f} wd={best['wd']:.3f} we={best['we']:.3f} "
                  f"gh={best['gh']} d={best['delta']:.3f}{mark}")
        except Exception as e:
            print(f"  [{i:3}/{n_samples}] FAIL: {e}")
        if i % 20 == 0:
            elapsed = time.time() - t0
            print(f"  ... {elapsed:.0f}s elapsed")

    results.sort(key=lambda r: -r["wr"])
    print(f"\n=== Top 15 by WR ===")
    for r in results[:15]:
        m = " <-- MISSION" if r["wr"] >= 0.60 else ""
        print(f"  WR={r['wr']*100:5.2f}% taken={r['taken']:>4} EV={r['ev']:+6.2f} "
              f"wd={r['wd']:.3f} we={r['we']:.3f} gh={r['gh']} d={r['delta']:.3f}{m}")
        print(f"    {r['hp']}")


if __name__ == "__main__":
    main()
