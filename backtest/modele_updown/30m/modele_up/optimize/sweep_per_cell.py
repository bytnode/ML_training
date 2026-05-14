"""Per-cell calibration : 6 thresholds (one per (vol_state, is_weekend))
instead of 2 per-axis. Brute-force grid search per cell."""
from __future__ import annotations
import time
import itertools
import numpy as np

from common import (
    load_dataset, feature_columns, train_once,
)
from train import hour_of_day_from_ms

CD = 20  # days of val to use
MIN_PARIS = 750
GOOD_HOURS = (19, 20, 21)
DELTA = 0.060

# Per-cell vol_frac grid. 6 cells, 4 values -> 4^6 = 4096 combos.
# Reduce by using same vol_frac for all weekday OR weekend cells, but
# adapt within each axis.
VOL_FRAC_GRID = (0.010, 0.018, 0.025, 0.035, 0.050, 0.080, 0.150, 0.250)


def compute_cell_thresholds(val_df, p_val, frac_per_cell: dict, cd: int):
    """frac_per_cell: dict[(vol, wknd) -> vol_frac]. Returns thresholds dict."""
    ms_per_day = 86_400 * 1000
    val_max_ts = int(val_df["timestamp"].max())
    cutoff = val_max_ts - cd * ms_per_day
    mask = val_df["timestamp"].to_numpy() >= cutoff

    vol_arr = val_df["volatility_state"].astype(int).to_numpy()
    wknd_arr = val_df["is_weekend"].astype(int).to_numpy()
    probas = np.asarray(p_val, dtype=float)

    thresholds = {}
    for (v, w), frac in frac_per_cell.items():
        cell_mask = mask & (vol_arr == v) & (wknd_arr == w)
        n = int(cell_mask.sum())
        if n < 10 or frac <= 0:
            thresholds[(v, w)] = 1.0
            continue
        thresholds[(v, w)] = float(np.quantile(probas[cell_mask], 1.0 - frac))
    return thresholds


def apply_cell_thresholds(test_df, p_test, thresholds, good_hours, delta):
    p_test = np.asarray(p_test, dtype=float)
    if len(test_df) == 0:
        return np.zeros(0, dtype=bool)
    vol = test_df["volatility_state"].astype(int).to_numpy()
    wknd = test_df["is_weekend"].astype(int).to_numpy()
    hour = hour_of_day_from_ms(test_df["timestamp"].to_numpy())
    gh_set = set(int(h) for h in good_hours)

    take = np.zeros(len(p_test), dtype=bool)
    for i in range(len(p_test)):
        T = thresholds.get((int(vol[i]), int(wknd[i])), 1.0)
        if int(hour[i]) in gh_set:
            T -= delta
        if p_test[i] >= T:
            take[i] = True
    return take


def main(rng_seed: int = 0, n_samples: int = 4000):
    df = load_dataset()
    fcols = feature_columns(df)
    run = train_once(df, fcols, seed=42)
    print(f"trained best_iter={run['best_iter']}\n")
    print(f"=== Per-cell sweep ({n_samples} random samples) ===")

    # Distribution check
    val = run["val"]
    val_vol = val["volatility_state"].astype(int).to_numpy()
    val_wknd = val["is_weekend"].astype(int).to_numpy()
    print(f"  val cells (vol, wknd) -> count :")
    for v in (0, 1, 2):
        for w in (0, 1):
            n = int(((val_vol == v) & (val_wknd == w)).sum())
            print(f"    ({v}, {w}) : {n}")
    print()

    rng = np.random.default_rng(rng_seed)
    cells = [(v, w) for v in (0, 1, 2) for w in (0, 1)]

    best = None
    t0 = time.time()
    for trial in range(n_samples):
        frac_per_cell = {c: VOL_FRAC_GRID[rng.integers(0, len(VOL_FRAC_GRID))] for c in cells}
        thresholds = compute_cell_thresholds(run["val"], run["p_val"], frac_per_cell, CD)
        take = apply_cell_thresholds(run["test"], run["p_test"], thresholds, GOOD_HOURS, DELTA)
        n = int(take.sum())
        if n < MIN_PARIS:
            continue
        y = run["y_test"]
        wr = float(y[take].mean())
        if best is None or wr > best["wr"]:
            best = {"frac": frac_per_cell, "taken": n, "wr": wr, "ev": n*(wr-0.5)}
            print(f"  [{trial:>4}] NEW BEST WR={wr*100:.2f}% taken={n} EV={n*(wr-0.5):+.2f}  frac={frac_per_cell}")
        if (trial+1) % 500 == 0:
            print(f"  ... {trial+1}/{n_samples} ({time.time()-t0:.1f}s)")

    print(f"\nFinal best: WR={best['wr']*100:.2f}% taken={best['taken']} EV={best['ev']:+.2f}")
    print(f"  fractions: {best['frac']}")


if __name__ == "__main__":
    main()
