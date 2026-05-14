"""
Discover which UTC hours are most favorable for UP bets, then sweep
hour_boost (good_hours, delta) configurations.

Steps :
  1. Train model
  2. For each hour h, compute val_WR conditional on p_val being in the
     top quantile (= would-be-bet zone) at that hour
  3. Pick top-K hours by val_WR
  4. Sweep delta for the picked hours + a few hand-crafted alternatives

Usage : cd modele_up && python optimize/sweep_hour_boost.py
"""
from __future__ import annotations
import time
import numpy as np

from common import (
    load_dataset, feature_columns, train_once, evaluate_with_calibration,
)
from train import hour_of_day_from_ms

WD = 0.030
WE = 0.250
MIN_PARIS = 750


def per_hour_val_wr(run) -> list[tuple[int, int, float, float]]:
    """For each hour h, return (h, n_above_quantile, p_quantile, wr_above_quantile).
    Quantile = 1 - WE = 0.75 (= top 25% of val probas), pour matcher la zone
    de pari potentielle."""
    p_val = run["p_val"]
    val_df = run["val"]
    labels = val_df["label"].to_numpy().astype(int)
    hours = hour_of_day_from_ms(val_df["timestamp"].to_numpy())

    # Use the 0.97 quantile (matches weekday) to evaluate "tight bets"
    q = 0.97
    p_thresh = float(np.quantile(p_val, q))
    rows = []
    for h in range(24):
        mask = (hours == h) & (p_val >= p_thresh)
        n = int(mask.sum())
        wr = float(labels[mask].mean()) if n > 0 else float("nan")
        rows.append((h, n, p_thresh, wr))
    return rows


def main():
    print("=== Hour-boost discovery + sweep ===")
    df = load_dataset()
    fcols = feature_columns(df)
    print(f"  dataset : {len(df):,} rows x {len(fcols)} features\n")

    # Train one model (deterministic at subsample=colsample=1.0)
    t0 = time.time()
    run = train_once(df, fcols, seed=42)
    print(f"  trained in {time.time()-t0:.1f}s, best_iter={run['best_iter']}")

    # --- Step 1 : per-hour val WR at high quantile ---
    print(f"\n--- Per-hour val WR at p_val >= 0.97 quantile ---")
    rows = per_hour_val_wr(run)
    rows.sort(key=lambda r: -r[3] if not np.isnan(r[3]) else 1.0)
    print(f"  {'hour':>4}  {'n':>5}  {'WR':>7}")
    for h, n, _, wr in rows:
        if n >= 3:
            print(f"  {h:>4}  {n:>5}  {wr*100:>6.2f}%")

    # --- Step 2 : sweep delta x candidate good_hours sets ---
    best_hours_sorted = [r[0] for r in rows if r[1] >= 5]  # need >=5 to trust WR
    candidates = []
    # Top-K consecutive groupings
    for K in (1, 2, 3, 4, 5, 6, 8):
        if len(best_hours_sorted) >= K:
            candidates.append(tuple(sorted(best_hours_sorted[:K])))
    # Hand-crafted
    candidates.extend([
        (20, 21, 22),                   # current
        (19, 20, 21, 22, 23),           # US-evening extended
        (7, 8, 9, 10),                  # Europe morning
        (13, 14, 15, 16),                # US morning
        (0, 1, 2, 22, 23),               # midnight + late
        tuple(range(24)),                # all hours = uniform boost
    ])
    # Dedupe
    seen = set()
    unique_candidates = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            unique_candidates.append(c)

    delta_grid = (0.00, 0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.040, 0.050)

    print(f"\n--- Sweep : {len(unique_candidates)} good_hour sets x {len(delta_grid)} deltas ---")
    results = []
    for good_hours in unique_candidates:
        for delta in delta_grid:
            res = evaluate_with_calibration(run, WD, WE,
                                            good_hours=good_hours,
                                            hour_boost_delta=delta)
            if res["taken"] < MIN_PARIS:
                continue
            results.append({
                "good_hours": good_hours,
                "delta": delta,
                **res,
            })

    results.sort(key=lambda r: -r["wr"])
    print(f"\nTop 20 by WR (taken>={MIN_PARIS}) :")
    print(f"  {'good_hours':<40}  {'delta':>6}  {'taken':>5}  {'WR':>7}  {'EV':>7}")
    for r in results[:20]:
        marker = " <-- MISSION" if r["wr"] >= 0.60 and r["taken"] >= 750 else ""
        gh_str = str(r["good_hours"])[:38]
        print(f"  {gh_str:<40}  {r['delta']:>6.3f}  {r['taken']:>5}  "
              f"{r['wr']*100:>6.2f}%  {r['ev']:>+6.2f}{marker}")


if __name__ == "__main__":
    main()
