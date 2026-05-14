"""Combine looser calibration with bad_hours filter to push WR.

Idea: increase wd/we to take MORE paris, then exclude paris at hours that
have poor val_WR, keep result if paris>=750.
"""
from __future__ import annotations
import numpy as np

from common import (
    load_dataset, feature_columns, train_once,
)
from train import (
    compute_thresholds_asym_wknd_excl, apply_thresholds_asym_hour_boost,
    hour_of_day_from_ms,
)

CD_GRID = (20, 30, 40, 60, 90)
WD_GRID = (0.020, 0.025, 0.030, 0.035, 0.040, 0.050, 0.060, 0.080)
WE_GRID = (0.15, 0.20, 0.25, 0.30, 0.35, 0.40)
GH_GRID = [(19, 20, 21), (20, 21)]
DELTA_GRID = (0.040, 0.060, 0.080)
MIN_PARIS = 750


def main():
    df = load_dataset()
    fcols = feature_columns(df)
    run = train_once(df, fcols, seed=42)

    # Pre-compute val per-hour WR among "candidate" probas (top 30% val).
    # Use this to pick bad_hours independently of test set.
    p_val = run["p_val"]
    val_labels = run["val"]["label"].to_numpy().astype(int)
    val_hours = hour_of_day_from_ms(run["val"]["timestamp"].to_numpy())
    p_quantile = float(np.quantile(p_val, 0.70))  # top 30% probas as candidates
    cand_mask = p_val >= p_quantile

    val_hour_wr = {}
    for h in range(24):
        m = cand_mask & (val_hours == h)
        n = int(m.sum())
        if n < 10:
            val_hour_wr[h] = (n, float("nan"))
            continue
        val_hour_wr[h] = (n, float(val_labels[m].mean()))
    # Rank hours by val WR ascending
    ranked = sorted(((h, n, wr) for h, (n, wr) in val_hour_wr.items() if not np.isnan(wr)),
                    key=lambda x: x[2])

    test_hours = hour_of_day_from_ms(run["test"]["timestamp"].to_numpy())
    y_test = run["y_test"]

    print(f"Val worst hours by WR (n_val>=10):")
    for h, n, wr in ranked[:8]:
        print(f"  h={h:>2} n={n:>4} WR={wr*100:.2f}%")
    print()

    bad_hour_candidates = [tuple(sorted(h for h, _, _ in ranked[:k])) for k in range(0, 6)]

    results = []
    n_total = (len(CD_GRID) * len(WD_GRID) * len(WE_GRID)
               * len(GH_GRID) * len(DELTA_GRID) * len(bad_hour_candidates))
    print(f"Total combos to evaluate : {n_total}")
    i = 0
    for cd in CD_GRID:
        for wd in WD_GRID:
            for we in WE_GRID:
                th, _ = compute_thresholds_asym_wknd_excl(
                    run["val"], p_val, vol_frac_wd=wd, vol_frac_we=we,
                    skip_cells=(), use_full_val=False, calib_days=cd,
                )
                for gh in GH_GRID:
                    for d in DELTA_GRID:
                        take_base = apply_thresholds_asym_hour_boost(
                            run["test"], run["p_test"], th,
                            skip_cells=(), good_hours=gh, hour_boost_delta=d,
                        )
                        for bad_hours in bad_hour_candidates:
                            if bad_hours:
                                exclude_mask = np.isin(test_hours, list(bad_hours))
                                take = take_base & ~exclude_mask
                            else:
                                take = take_base
                            n = int(take.sum())
                            if n < MIN_PARIS:
                                continue
                            wr = float(y_test[take].mean())
                            results.append({
                                "cd": cd, "wd": wd, "we": we, "gh": gh, "delta": d,
                                "bad_hours": bad_hours,
                                "taken": n, "wr": wr, "ev": n * (wr - 0.5),
                            })
                            i += 1

    results.sort(key=lambda r: -r["wr"])
    print(f"\nTop 25 by WR (taken>={MIN_PARIS}) of {len(results)} mission-passing :")
    for r in results[:25]:
        m = " <-- 64+" if r["wr"] >= 0.64 else ""
        bh = "()" if not r["bad_hours"] else str(r["bad_hours"])
        print(f"  cd={r['cd']:>3} wd={r['wd']:.3f} we={r['we']:.3f} gh={str(r['gh']):<14} "
              f"d={r['delta']:.3f} bad={bh:<24} -> {r['taken']:>4} {r['wr']*100:>6.2f}% {r['ev']:>+6.2f}{m}")


if __name__ == "__main__":
    main()
