"""
Sweep XGBoost hyperparams. For each combo:
  - train n_seeds models
  - sweep vol_frac
  - report : best WR with taken >= 750

Goal : find hp combo that lifts WR while staying compatible with the
mission target (>=750 paris on 365d).

Usage : cd modele_up && python optimize/sweep_hyperparams.py
"""
from __future__ import annotations
import itertools
import time
import numpy as np

from common import (
    load_dataset, feature_columns, train_once, evaluate_with_calibration,
)


# Grid centered around current values + spread
GRID = {
    "max_depth":            [3, 4, 5, 6],
    "learning_rate":        [0.03, 0.05, 0.08],
    "min_child_weight":     [10, 20, 40, 80],
    "reg_lambda":           [False, 1.0, 3.0, 10.0],
    "reg_alpha":            [False, 0.5, 2.0],
    "gamma":                [0.5, 1.5, 3.0],
    "colsample_bytree":     [0.6, 0.8, 1.0],
    "subsample":            [0.6, 0.8, 1.0],
}

# Calibration sweep to find best WR per model
WD_GRID = (0.025, 0.030, 0.035, 0.040, 0.050, 0.060)
WE_GRID = (0.15, 0.20, 0.25, 0.30, 0.35)


def best_wr_at_min_paris(run, min_paris: int = 750) -> dict:
    """For a trained run, find the calibration giving the best WR among
    those with taken >= min_paris. Returns dict with best result or None."""
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
    """For one hp config, train n_seeds, score = avg WR at taken>=750."""
    seeds_results = []
    for s in range(n_seeds):
        run = train_once(df, fcols, seed=42 + s, hp_overrides=hp_overrides)
        best = best_wr_at_min_paris(run)
        if best is None:
            return {"hp": hp_overrides, "wr": None, "taken": 0,
                    "val_logloss": run["val_logloss"]}
        seeds_results.append(best)
    return {
        "hp": hp_overrides,
        "wr": float(np.mean([r["wr"] for r in seeds_results])),
        "taken": float(np.mean([r["taken"] for r in seeds_results])),
        "ev": float(np.mean([r["ev"] for r in seeds_results])),
        "val_logloss": float(np.mean([r["val_logloss"] if False else 0 for r in seeds_results]) or 0),
    }


def random_sample_grid(n_samples: int = 80, rng_seed: int = 0) -> list[dict]:
    """Random search over GRID (more efficient than full cartesian)."""
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


def main(n_samples: int = 60, n_seeds: int = 2):
    print(f"=== Hyperparam sweep (random search, n_samples={n_samples}, n_seeds={n_seeds}) ===")
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
                marker = " <-- MISSION" if r["wr"] >= 0.60 and r["taken"] >= 750 else ""
                print(f"  [{i}/{n_samples}] WR={r['wr']*100:5.2f}% "
                      f"taken={r['taken']:6.1f} EV={r['ev']:+6.2f}  hp={hp}{marker}")
            else:
                print(f"  [{i}/{n_samples}] NO_750_CONFIG hp={hp}")
        except Exception as e:
            print(f"  [{i}/{n_samples}] FAIL : {e}")
        if i % 10 == 0:
            elapsed = time.time() - t0
            eta = elapsed / i * (n_samples - i)
            print(f"  ... {elapsed:.0f}s elapsed, ETA {eta:.0f}s")

    # Filter + sort
    valid = [r for r in results if r["wr"] is not None]
    valid.sort(key=lambda r: -r["wr"])

    print(f"\n=== Top 10 by WR (subject to taken>=750) ===")
    for r in valid[:10]:
        marker = " *** MISSION ***" if r["wr"] >= 0.60 and r["taken"] >= 750 else ""
        print(f"  WR={r['wr']*100:5.2f}% taken={r['taken']:6.1f} EV={r['ev']:+6.2f}{marker}")
        print(f"    hp={r['hp']}")

    return results


if __name__ == "__main__":
    main()
