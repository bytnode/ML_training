"""
High-WR sweep on logs/test_predictions.csv.

Goal: from the 1966 bets / 60.38 % WR baseline, find a config that lands
at 750-1000 bets / 61-65 % WR by being MORE SELECTIVE.

The script:
  1. loads the test predictions (one row per test bar) ;
  2. computes per-bucket (volatility_state, is_weekend, hour) WR at the
     current baseline threshold so we can spot losing buckets ;
  3. sweeps a grid of (T_weekday, T_weekend, skip_cells, good_hours, delta)
     and reports configs in the [750, 1000] bet range, sorted by WR.

It does NOT touch the model or the parquet -- pure logs/test_predictions.csv
analysis. Run from the package root:

    cd backtest/modele_updown/15min/modele_down
    python optimize/sweep_high_wr.py
"""

from __future__ import annotations
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

PKG_ROOT = Path(__file__).resolve().parents[1]
TEST_PRED = PKG_ROOT / "logs" / "test_predictions.csv"

# Target window for bet count.
TARGET_LO, TARGET_HI = 750, 1000

# Baseline reference (= what the current config gives on this CSV).
BASELINE_T_WKD = 0.56626
BASELINE_T_WKEND = 0.54769
BASELINE_GOOD_HOURS = (20, 21, 22)
BASELINE_DELTA = 0.020


# ----------------------------------------------------------------------
def _add_hour(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour"] = ((df["timestamp"] // (60 * 60 * 1000)) % 24).astype(int)
    return df


def _apply_config(
    df: pd.DataFrame,
    t_wkd: float,
    t_wkend: float,
    skip_cells: tuple,
    good_hours: tuple,
    delta: float,
) -> pd.Series:
    """Replicates train.apply_thresholds_asym_hour_boost on test rows.

    Returns a boolean mask of bars that would be TAKEN.
    """
    p = df["p_down"].to_numpy()
    is_wkend = df["is_weekend"].to_numpy().astype(int)
    vol = df["volatility_state"].to_numpy().astype(int)
    hour = df["hour"].to_numpy().astype(int)

    # Per-axis base threshold.
    T = np.where(is_wkend == 1, t_wkend, t_wkd).astype(float)

    # Skip cells: force threshold = 1.0 (= never take).
    skip = np.zeros_like(T, dtype=bool)
    for v, w in skip_cells:
        skip |= (vol == v) & (is_wkend == w)
    T = np.where(skip, np.inf, T)

    # Hour boost: lower threshold on good hours (skip cells immune).
    if delta and good_hours:
        good = np.isin(hour, good_hours)
        T = np.where(good & (~skip), T - delta, T)

    return p >= T


def _wr_count(df: pd.DataFrame, mask: np.ndarray) -> tuple[int, float]:
    taken = df.loc[mask, "label"]
    n = int(taken.size)
    wr = float(taken.mean()) if n else float("nan")
    return n, wr


# ----------------------------------------------------------------------
def per_bucket_wr(df: pd.DataFrame) -> None:
    """Print per-bucket WR at the current baseline threshold."""
    mask = _apply_config(
        df,
        t_wkd=BASELINE_T_WKD,
        t_wkend=BASELINE_T_WKEND,
        skip_cells=(),
        good_hours=BASELINE_GOOD_HOURS,
        delta=BASELINE_DELTA,
    )
    taken = df[mask].copy()
    n_total, wr_total = _wr_count(df, mask)
    print(f"\nBaseline (full test, current config): {n_total} bets, WR={wr_total:.4f}\n")

    print("--- WR by (volatility_state x is_weekend) bucket ---")
    g = taken.groupby(["volatility_state", "is_weekend"])
    bucket = g["label"].agg(["count", "mean"]).rename(columns={"count": "n", "mean": "wr"})
    bucket["wr"] = bucket["wr"].round(4)
    print(bucket.to_string())

    print("\n--- WR by hour (UTC), all volatility/weekend pooled ---")
    g = taken.groupby("hour")
    h = g["label"].agg(["count", "mean"]).rename(columns={"count": "n", "mean": "wr"})
    h["wr"] = h["wr"].round(4)
    print(h.to_string())

    print("\n--- WR by hour x is_weekend ---")
    g = taken.groupby(["is_weekend", "hour"])
    hw = g["label"].agg(["count", "mean"]).rename(columns={"count": "n", "mean": "wr"})
    hw["wr"] = hw["wr"].round(4)
    print(hw.to_string())


def sweep(df: pd.DataFrame) -> pd.DataFrame:
    """Grid sweep of all candidate configs. Returns a DataFrame sorted by WR."""
    # Threshold grid: 0.50 ... 0.70 in 0.005 steps (extended downward so we can
    # explore "weekend-only" configs where weekday is forced out via skip_cells
    # and weekend keeps a permissive threshold).
    t_wkd_grid = np.round(np.arange(0.50, 0.701, 0.005), 4)
    t_wkend_grid = np.round(np.arange(0.50, 0.701, 0.005), 4)

    # Skip-cell candidates: empty, 1-cell, 2-cell, plus the structural choices
    # "all weekday" and "all weekend" (the latter as a sanity probe).
    cells = [(v, w) for v in (0, 1, 2) for w in (0, 1)]
    skip_choices = [()]
    skip_choices += [(c,) for c in cells]
    # All-weekday and 2-cell weekday subsets (weekday is empirically weaker).
    skip_choices += [
        ((0, 0), (1, 0), (2, 0)),    # skip ALL weekday
        ((0, 0), (2, 0)),            # skip weekday low+high vol (keep mid)
        ((2, 0), (2, 1)),            # skip ALL high vol
        ((0, 0), (1, 0)),            # skip weekday low+mid
    ]

    # Hour-boost candidates: include "MALUS" inverted variants (boost the
    # GOOD hours, not the bad ones). Hour 22 was the worst at baseline -- the
    # current boost (20,21,22) is dragging the WR down.
    hour_choices = [
        ((), 0.0),                              # disable boost
        ((20, 21, 22), 0.010),                  # original boost, smaller
        ((20, 21, 22), 0.020),                  # original baseline
        ((20, 21, 22, 23), 0.020),              # boost extended
        ((19, 20, 21, 22), 0.020),              # boost shifted earlier
        ((20, 21), 0.020),                      # drop hour 22 (the bad one)
        ((20, 21), 0.030),                      # stronger boost on the 2 good ones
        ((6, 7, 10, 12, 18), 0.015),            # boost the empirically-best hours
        ((6, 7, 10, 12), 0.020),
    ]

    rows = []
    for t_wkd, t_wkend in product(t_wkd_grid, t_wkend_grid):
        for skip in skip_choices:
            for good_h, d in hour_choices:
                mask = _apply_config(df, t_wkd, t_wkend, skip, good_h, d)
                n, wr = _wr_count(df, mask)
                if TARGET_LO <= n <= TARGET_HI:
                    rows.append({
                        "t_wkd":   t_wkd,
                        "t_wkend": t_wkend,
                        "skip":    skip,
                        "hours":   good_h,
                        "delta":   d,
                        "bets":    n,
                        "wr":      wr,
                    })

    if not rows:
        print(f"\nNo configs landed in [{TARGET_LO}, {TARGET_HI}] bets.")
        return pd.DataFrame()

    out = pd.DataFrame(rows).sort_values("wr", ascending=False).reset_index(drop=True)
    print(f"\nSweep: {len(out)} configs land in [{TARGET_LO}, {TARGET_HI}] bets. Top 20:\n")
    print(out.head(20).to_string())
    return out


# ----------------------------------------------------------------------
def main() -> None:
    if not TEST_PRED.exists():
        sys.exit(f"missing {TEST_PRED} -- run python train.py first")

    df = pd.read_csv(TEST_PRED)
    df = _add_hour(df)
    print(f"loaded {len(df):,} test bars from {TEST_PRED.relative_to(PKG_ROOT)}")
    print(f"label distribution: down={df['label'].mean():.4f}  base WR if blind=DOWN")

    per_bucket_wr(df)
    out = sweep(df)

    if not out.empty:
        # Top-3 with diversity in skip/hours to make the next decision easier.
        print("\n--- Top-3 distinct configs (one per (skip, hours, delta) tuple) ---")
        seen = set()
        picks = []
        for _, r in out.iterrows():
            key = (r["skip"], r["hours"], r["delta"])
            if key in seen:
                continue
            seen.add(key)
            picks.append(r)
            if len(picks) == 5:
                break
        for r in picks:
            print(
                f"  T_wkd={r['t_wkd']:.4f}  T_wkend={r['t_wkend']:.4f}  "
                f"skip={r['skip']}  hours={r['hours']}  delta={r['delta']}  "
                f"-> {r['bets']} bets / WR={r['wr']:.4f}"
            )


if __name__ == "__main__":
    main()
