"""Focused sweep around the (20,) hour-boost winner."""
import numpy as np
from common import (
    load_dataset, feature_columns, train_once, evaluate_with_calibration,
)

WD_GRID = (0.020, 0.022, 0.025, 0.028, 0.030, 0.033, 0.035, 0.038, 0.040, 0.045, 0.050)
WE_GRID = (0.18, 0.20, 0.22, 0.25, 0.28, 0.30, 0.35)
GOOD_HOURS_GRID = [
    (20,),
    (20, 21),
    (3, 20),
    (8, 20),
    (0, 8, 20),
    (3, 8, 20),
    (8, 20, 21),
    (20, 21, 22),
]
DELTA_GRID = (0.00, 0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040, 0.050, 0.060, 0.080)

MIN_PARIS = 750


def main():
    df = load_dataset()
    fcols = feature_columns(df)
    run = train_once(df, fcols, seed=42)
    print(f"  trained best_iter={run['best_iter']}, val_ll={run['val_logloss']:.5f}\n")

    rows = []
    for gh in GOOD_HOURS_GRID:
        for delta in DELTA_GRID:
            for wd in WD_GRID:
                for we in WE_GRID:
                    res = evaluate_with_calibration(
                        run, wd, we, good_hours=gh, hour_boost_delta=delta,
                    )
                    if res["taken"] < MIN_PARIS:
                        continue
                    rows.append({
                        "good_hours": gh, "delta": delta,
                        "wd": wd, "we": we, **res,
                    })
    rows.sort(key=lambda r: -r["wr"])
    print(f"Top 30 by WR (taken>={MIN_PARIS}) :")
    print(f"  {'good_hours':<24} {'d':>6} {'wd':>6} {'we':>6} {'taken':>6} {'WR':>8} {'EV':>7}")
    for r in rows[:30]:
        marker = " <-- MISSION" if r["wr"] >= 0.60 else ""
        print(f"  {str(r['good_hours']):<24} {r['delta']:>6.3f} {r['wd']:>6.3f} {r['we']:>6.3f} "
              f"{r['taken']:>6}  {r['wr']*100:>6.2f}%  {r['ev']:>+6.2f}{marker}")

    # Filter mission-hitters
    mission = [r for r in rows if r["wr"] >= 0.60]
    print(f"\n>>> {len(mission)} mission-passing config(s) found <<<")
    if mission:
        # Maximize EV among them
        mission.sort(key=lambda r: -r["ev"])
        print(f"\nTop 10 mission-passing by EV :")
        for r in mission[:10]:
            print(f"  {str(r['good_hours']):<24} d={r['delta']:>5.3f} wd={r['wd']:>5.3f} we={r['we']:>5.3f}  "
                  f"taken={r['taken']:>4}  WR={r['wr']*100:>6.2f}%  EV={r['ev']:>+6.2f}")


if __name__ == "__main__":
    main()
