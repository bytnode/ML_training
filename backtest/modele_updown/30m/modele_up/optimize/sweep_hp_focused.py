"""
Focused hyperparam search around the winner config.
Adds scale_pos_weight to the grid.

Winner from sweep_hyperparams.py :
  max_depth=6, learning_rate=0.03, min_child_weight=10,
  reg_lambda=10.0, reg_alpha=False, gamma=3.0,
  colsample_bytree=1.0, subsample=1.0

Usage : cd modele_up && python optimize/sweep_hp_focused.py
"""
from __future__ import annotations
import time
import numpy as np

from common import (
    load_dataset, feature_columns, train_once, evaluate_with_calibration,
)


GRID = {
    "max_depth":            [5, 6, 7, 8],
    "learning_rate":        [0.015, 0.020, 0.025, 0.030, 0.040, 0.050],
    "min_child_weight":     [5, 10, 15, 25],
    "reg_lambda":           [5.0, 10.0, 20.0, 30.0, 50.0],
    "reg_alpha":            [False, 0.1, 0.5, 1.0],
    "gamma":                [1.5, 2.0, 3.0, 5.0, 8.0],
    "colsample_bytree":     [0.8, 0.9, 1.0],
    "subsample":            [0.8, 0.9, 1.0],
    "scale_pos_weight":     [0.95, 1.00, 1.03, 1.05, 1.10],
    "n_estimators":         [1000, 1500],
}

WD_GRID = (0.020, 0.025, 0.030, 0.035, 0.040, 0.045, 0.050)
WE_GRID = (0.15, 0.18, 0.20, 0.22, 0.25, 0.30)


def best_wr_at_min_paris(run, min_paris: int = 750) -> dict | None:
    best = None
    for wd in WD_GRID:
        for we in WE_GRID:
            res = evaluate_with_calibration(run, wd, we)
            if res["taken"] < min_paris:
                continue
            if best is None or res["wr"] > best["wr"]:
                best = {**res, "wd_frac": wd, "we_frac": we}
    return best


def score_hp_config(df, fcols, hp_overrides: dict, n_seeds: int = 2) -> dict:
    seeds_results = []
    for s in range(n_seeds):
        run = train_once(df, fcols, seed=42 + s, hp_overrides=hp_overrides)
        best = best_wr_at_min_paris(run)
        if best is None:
            return {"hp": hp_overrides, "wr": None, "taken": 0, "ev": 0,
                    "wd_frac": None, "we_frac": None}
        seeds_results.append(best)
    return {
        "hp": hp_overrides,
        "wr": float(np.mean([r["wr"] for r in seeds_results])),
        "taken": float(np.mean([r["taken"] for r in seeds_results])),
        "ev": float(np.mean([r["ev"] for r in seeds_results])),
        "wd_frac": seeds_results[0]["wd_frac"],
        "we_frac": seeds_results[0]["we_frac"],
    }


def random_sample_grid(n_samples: int, rng_seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(rng_seed)
    keys = list(GRID.keys())
    samples = []
    seen = set()
    while len(samples) < n_samples:
        cfg = {k: GRID[k][rng.integers(0, len(GRID[k]))] for k in keys}
        sig = tuple(sorted(cfg.items()))
        if sig in seen:
            continue
        seen.add(sig)
        samples.append(cfg)
    return samples


def main(n_samples: int = 120, n_seeds: int = 2):
    print(f"=== Focused HP sweep (n_samples={n_samples}, n_seeds={n_seeds}) ===")
    df = load_dataset()
    fcols = feature_columns(df)
    print(f"  dataset : {len(df):,} rows x {len(fcols)} features\n")

    samples = random_sample_grid(n_samples)
    results = []
    t0 = time.time()
    for i, hp in enumerate(samples, 1):
        try:
            r = score_hp_config(df, fcols, hp, n_seeds=n_seeds)
            results.append(r)
            if r["wr"] is not None:
                marker = " <-- MISSION!" if r["wr"] >= 0.60 and r["taken"] >= 750 else ""
                print(f"  [{i:3d}/{n_samples}] WR={r['wr']*100:5.2f}% "
                      f"taken={r['taken']:6.1f} EV={r['ev']:+6.2f} "
                      f"wd={r['wd_frac']:.3f} we={r['we_frac']:.3f}{marker}")
            else:
                print(f"  [{i:3d}/{n_samples}] NO_750")
        except Exception as e:
            print(f"  [{i:3d}/{n_samples}] FAIL : {e}")
        if i % 20 == 0:
            elapsed = time.time() - t0
            eta = elapsed / i * (n_samples - i)
            print(f"  ... {elapsed:.0f}s elapsed, ETA {eta:.0f}s")

    valid = [r for r in results if r["wr"] is not None]
    valid.sort(key=lambda r: -r["wr"])

    print(f"\n=== Top 15 by WR (taken>=750) ===")
    for r in valid[:15]:
        marker = " *** MISSION ***" if r["wr"] >= 0.60 and r["taken"] >= 750 else ""
        print(f"  WR={r['wr']*100:5.2f}% taken={r['taken']:6.1f} EV={r['ev']:+6.2f} "
              f"wd={r['wd_frac']:.3f} we={r['we_frac']:.3f}{marker}")
        print(f"    {r['hp']}")

    return results


if __name__ == "__main__":
    main()
