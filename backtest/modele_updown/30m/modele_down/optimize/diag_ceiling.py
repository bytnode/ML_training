"""
Diagnostic: what's the highest WR achievable on test, regardless of how we
calibrate, by taking the top N test bets by P(down)?

This is the WR ceiling the model can reach. Anything calibration does has
to live below this. If the ceiling is 65%, our threshold approach can reach
goal. If the ceiling is 58%, the model itself is the bottleneck.
"""
from __future__ import annotations

import numpy as np

from common import (
    load_dataset, feature_columns, train_once, hour_of_day_from_ms,
)


def main():
    df = load_dataset()
    fc = feature_columns(df)
    print(f"Dataset: {len(df):,} rows, {len(fc)} features.")
    run = train_once(df, fc, seed=42)
    print(f"  best_iter={run['best_iter']} val_ll={run['val_logloss']:.5f}\n")

    y_test = run["y_test"]
    p_test = run["p_test"]
    n_test = len(p_test)

    # Sort indices by P(down) descending
    order = np.argsort(-p_test)

    print("Ceiling profile on TEST (sorted by P(down) descending):")
    print("  top_N      P_min   wr      cum_wins")
    for n in (50, 100, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 5000):
        if n > n_test:
            continue
        sel = order[:n]
        wr = float(y_test[sel].mean())
        p_min = float(p_test[sel].min())
        print(f"  {n:>5}    {p_min:.4f}   {wr*100:>6.2f}%   {int(y_test[sel].sum()):>4}/{n}")

    # Also: best possible at >= 750 paris
    print("\nWR by P(down) quantile on TEST:")
    print("  pct      P_min     n      wr")
    for q in (99.5, 99, 98, 97, 95, 92, 90, 85, 80, 70):
        T = float(np.quantile(p_test, q/100))
        sel = p_test >= T
        n = int(sel.sum())
        wr = float(y_test[sel].mean()) if n > 0 else 0.0
        print(f"  q{q:>4}    {T:.4f}   {n:>5}   {wr*100:>6.2f}%")

    # What if we sub-select by hour AND P(down)?
    print("\n  Best 750 by P(down) regardless of hour: see top_N=750 above")

    # Distribution of P(down): histogram
    print("\nP(down) test histogram:")
    bins = np.array([0.0, 0.40, 0.45, 0.48, 0.50, 0.52, 0.54, 0.56, 0.58, 0.60, 0.65, 1.0])
    h, _ = np.histogram(p_test, bins=bins)
    for i, c in enumerate(h):
        sel = (p_test >= bins[i]) & (p_test < bins[i+1])
        wr = float(y_test[sel].mean()) if sel.sum() > 0 else 0.0
        print(f"  P in [{bins[i]:.2f}, {bins[i+1]:.2f}):  n={int(sel.sum()):>5}  wr={wr*100:>6.2f}%")


if __name__ == "__main__":
    main()
