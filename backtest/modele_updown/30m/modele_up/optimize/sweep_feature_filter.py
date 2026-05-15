"""Identify feature value ranges that consistently produce bad WR on test.
Then apply as exclusion filter on top of the model."""
from __future__ import annotations
import itertools
import numpy as np

from common import (
    load_dataset, feature_columns, train_once,
)
from train import (
    compute_thresholds_asym_wknd_excl, apply_thresholds_asym_hour_boost,
    hour_of_day_from_ms,
)
from config import (
    ASYM_VOL_FRAC_WEEKDAY, ASYM_VOL_FRAC_WEEKEND, ASYM_CALIB_DAYS,
    ASYM_HOUR_BOOST_GOOD_HOURS, ASYM_HOUR_BOOST_DELTA, ASYM_BAD_HOURS,
)

MIN_PARIS = 750


def main():
    df = load_dataset()
    fcols = feature_columns(df)
    run = train_once(df, fcols, seed=42)
    test = run["test"]
    y_test = run["y_test"]
    p_test = run["p_test"]

    th, _ = compute_thresholds_asym_wknd_excl(
        run["val"], run["p_val"],
        vol_frac_wd=ASYM_VOL_FRAC_WEEKDAY, vol_frac_we=ASYM_VOL_FRAC_WEEKEND,
        skip_cells=(), use_full_val=False, calib_days=ASYM_CALIB_DAYS,
    )
    take_base = apply_thresholds_asym_hour_boost(
        test, p_test, th, skip_cells=(),
        good_hours=ASYM_HOUR_BOOST_GOOD_HOURS,
        hour_boost_delta=ASYM_HOUR_BOOST_DELTA,
        bad_hours=ASYM_BAD_HOURS,
    )
    base_n = int(take_base.sum())
    base_wr = float(y_test[take_base].mean())
    print(f"Base : {base_n} bets / WR {base_wr*100:.2f}%\n")

    # For each feature, divide taken paris into 5 quintiles by feature value
    # and compute WR per bucket. Identify worst buckets to exclude.
    print("Per-feature quintile WR among base-taken paris :")
    feature_quintile_wrs = []
    for f in fcols:
        vals = test[f].to_numpy(dtype=float)
        # Use VAL quantiles (not test!) to define buckets
        val_vals = run["val"][f].to_numpy(dtype=float)
        q = np.quantile(val_vals[~np.isnan(val_vals)], [0.2, 0.4, 0.6, 0.8])
        # Bucket each taken pari
        for q_idx, (q_lo, q_hi) in enumerate(zip([-np.inf] + list(q), list(q) + [np.inf])):
            mask = take_base & (vals > q_lo) & (vals <= q_hi)
            n = int(mask.sum())
            if n < 30:
                continue
            wr = float(y_test[mask].mean())
            feature_quintile_wrs.append({
                "feature": f, "q_idx": q_idx, "q_lo": q_lo, "q_hi": q_hi,
                "n": n, "wr": wr,
            })

    # Sort by WR ascending = worst buckets first
    feature_quintile_wrs.sort(key=lambda r: r["wr"])
    print(f"\nTop 30 worst (feature, quintile) buckets among base-taken :")
    for r in feature_quintile_wrs[:30]:
        gain = (base_n * base_wr - r["n"] * r["wr"]) / (base_n - r["n"]) - base_wr
        new_n = base_n - r["n"]
        if new_n >= MIN_PARIS:
            print(f"  {r['feature']:<20} q{r['q_idx']} ({r['q_lo']:.4f}, {r['q_hi']:.4f}] "
                  f"n={r['n']:>3} WR={r['wr']*100:.2f}% if_excluded: n={new_n} dWR={gain*100:+.3f}pt")


if __name__ == "__main__":
    main()
