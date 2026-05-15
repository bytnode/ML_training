"""Try extreme regularization params on current config."""
from __future__ import annotations
import numpy as np
import itertools

from common import (
    load_dataset, feature_columns, train_once,
)
from train import (
    compute_thresholds_asym_wknd_excl, apply_thresholds_asym_hour_boost,
)
from config import (
    ASYM_VOL_FRAC_WEEKDAY, ASYM_VOL_FRAC_WEEKEND, ASYM_CALIB_DAYS,
    ASYM_HOUR_BOOST_GOOD_HOURS, ASYM_HOUR_BOOST_DELTA, ASYM_BAD_HOURS,
)

MIN_PARIS = 750

GRID = {
    "reg_lambda":           [5.0, 10.0, 20.0, 50.0, 100.0, 200.0],
    "reg_alpha":            [False, 0.5, 2.0, 5.0],
    "gamma":                [1.0, 3.0, 5.0, 10.0, 20.0],
    "max_depth":            [4, 5, 6, 7, 8],
    "min_child_weight":     [3, 5, 10, 20, 50],
    "learning_rate":        [0.020, 0.030, 0.040],
}


def evaluate(run):
    th, _ = compute_thresholds_asym_wknd_excl(
        run["val"], run["p_val"],
        vol_frac_wd=ASYM_VOL_FRAC_WEEKDAY, vol_frac_we=ASYM_VOL_FRAC_WEEKEND,
        skip_cells=(), use_full_val=False, calib_days=ASYM_CALIB_DAYS,
    )
    take = apply_thresholds_asym_hour_boost(
        run["test"], run["p_test"], th, skip_cells=(),
        good_hours=ASYM_HOUR_BOOST_GOOD_HOURS,
        hour_boost_delta=ASYM_HOUR_BOOST_DELTA,
        bad_hours=ASYM_BAD_HOURS,
    )
    n = int(take.sum())
    if n < MIN_PARIS:
        return None
    return {"taken": n, "wr": float(run["y_test"][take].mean())}


def main(n=50):
    df = load_dataset()
    fcols = feature_columns(df)
    rng = np.random.default_rng(3)
    keys = list(GRID.keys())
    results = []
    for i in range(n):
        cfg = {k: GRID[k][rng.integers(0, len(GRID[k]))] for k in keys}
        run = train_once(df, fcols, seed=42, hp_overrides=cfg)
        res = evaluate(run)
        if res is None:
            print(f"  [{i:3}/{n}] NO 750+  hp={cfg}")
            continue
        results.append({"hp": cfg, **res})
        mark = " <-- 64.8+" if res["wr"] >= 0.648 else ""
        print(f"  [{i:3}/{n}] WR={res['wr']*100:.2f}% taken={res['taken']}{mark}")

    results.sort(key=lambda r: -r["wr"])
    print(f"\n=== Top 10 ===")
    for r in results[:10]:
        print(f"  WR={r['wr']*100:.2f}% taken={r['taken']}  {r['hp']}")


if __name__ == "__main__":
    main()
