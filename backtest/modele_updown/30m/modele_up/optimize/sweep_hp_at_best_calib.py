"""HP sweep at the CURRENT best calibration (cd=20, wd=0.018, we=0.200,
gh=(19,20,21), d=0.060). Find hp combos that maximize WR at this calib."""
from __future__ import annotations
import time
import numpy as np

from common import (
    load_dataset, feature_columns, train_once,
)
from train import compute_thresholds_asym_wknd_excl, apply_thresholds_asym_hour_boost

GRID = {
    "max_depth":            [4, 5, 6, 7, 8, 10, 12],
    "learning_rate":        [0.015, 0.020, 0.025, 0.030, 0.040, 0.050],
    "min_child_weight":     [3, 5, 10, 15, 25],
    "reg_lambda":           [1.0, 5.0, 10.0, 20.0, 50.0],
    "reg_alpha":            [False, 0.1, 0.5, 2.0],
    "gamma":                [0.5, 1.5, 3.0, 5.0, 8.0],
    "colsample_bytree":     [0.7, 0.85, 1.0],
    "subsample":            [0.7, 0.85, 1.0],
}

# Reduced calibration grid centered on current best
CALIB_VARIANTS = [
    {"cd": 20, "wd": 0.018, "we": 0.200, "gh": (19, 20, 21), "d": 0.060},
    {"cd": 20, "wd": 0.020, "we": 0.180, "gh": (19, 20, 21), "d": 0.060},
    {"cd": 30, "wd": 0.025, "we": 0.200, "gh": (19, 20, 21), "d": 0.060},
    {"cd": 60, "wd": 0.030, "we": 0.150, "gh": (20, 21), "d": 0.060},
    {"cd": 40, "wd": 0.025, "we": 0.180, "gh": (20, 21), "d": 0.060},
]

MIN_PARIS = 750


def evaluate(run, cal):
    th, _ = compute_thresholds_asym_wknd_excl(
        run["val"], run["p_val"],
        vol_frac_wd=cal["wd"], vol_frac_we=cal["we"],
        skip_cells=(), use_full_val=False, calib_days=cal["cd"],
    )
    take = apply_thresholds_asym_hour_boost(
        run["test"], run["p_test"], th,
        skip_cells=(), good_hours=cal["gh"], hour_boost_delta=cal["d"],
    )
    n = int(take.sum())
    if n < MIN_PARIS:
        return None
    y = run["y_test"]
    wr = float(y[take].mean())
    return {"taken": n, "wr": wr, "ev": n * (wr - 0.5)}


def best_calib(run):
    best = None
    for cal in CALIB_VARIANTS:
        res = evaluate(run, cal)
        if res is None:
            continue
        if best is None or res["wr"] > best["wr"]:
            best = {**res, "cal": cal}
    return best


def random_sample(n, rng_seed=2):
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
    print(f"=== HP sweep @ best-calib variants ({n_samples} samples, {len(fcols)} feat) ===\n")
    samples = random_sample(n_samples)
    t0 = time.time()
    results = []
    for i, hp in enumerate(samples, 1):
        try:
            run = train_once(df, fcols, seed=42, hp_overrides=hp)
            best = best_calib(run)
            if best is None:
                continue
            results.append({"hp": hp, **best})
            mark = " <-- 63.5+" if best["wr"] >= 0.635 else ""
            print(f"  [{i:3}/{n_samples}] WR={best['wr']*100:5.2f}% taken={best['taken']:>4} "
                  f"EV={best['ev']:+6.2f}{mark}")
        except Exception as e:
            print(f"  [{i:3}/{n_samples}] FAIL: {e}")

    results.sort(key=lambda r: -r["wr"])
    print(f"\n=== Top 15 ===")
    for r in results[:15]:
        print(f"  WR={r['wr']*100:5.2f}% taken={r['taken']:>4} EV={r['ev']:+6.2f} cal={r['cal']}")
        print(f"    {r['hp']}")


if __name__ == "__main__":
    main()
