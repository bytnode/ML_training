"""Test : add a 'bad_hours' filter that excludes paris at hours where the
model historically performs poorly (validated on val set)."""
from __future__ import annotations
import numpy as np

from common import (
    load_dataset, feature_columns, train_once,
)
from train import (
    compute_thresholds_asym_wknd_excl, apply_thresholds_asym_hour_boost,
    hour_of_day_from_ms,
)

CD = 20
WD = 0.018
WE = 0.200
GH = (19, 20, 21)
DELTA = 0.060
MIN_PARIS = 750


def main():
    df = load_dataset()
    fcols = feature_columns(df)
    run = train_once(df, fcols, seed=42)
    print(f"trained best_iter={run['best_iter']}\n")

    # Base: get the taken mask without bad_hours filter
    th, _ = compute_thresholds_asym_wknd_excl(
        run["val"], run["p_val"],
        vol_frac_wd=WD, vol_frac_we=WE, skip_cells=(),
        use_full_val=False, calib_days=CD,
    )
    take_base = apply_thresholds_asym_hour_boost(
        run["test"], run["p_test"], th,
        skip_cells=(), good_hours=GH, hour_boost_delta=DELTA,
    )

    # Per-hour val WR (at would-be-bet probas)
    p_val = run["p_val"]
    val_labels = run["val"]["label"].to_numpy().astype(int)
    val_hours = hour_of_day_from_ms(run["val"]["timestamp"].to_numpy())
    # Use threshold from weekend axis (looser) for sample size
    test_hours = hour_of_day_from_ms(run["test"]["timestamp"].to_numpy())
    y_test = run["y_test"]

    # Per-hour test WR among base-taken paris (to see what we'd lose if we exclude)
    print("Per-hour analysis among base-taken paris (test):")
    print(f"  {'hour':>4}  {'n':>5}  {'WR':>7}  {'gain_if_excluded':>17}")
    base_n = int(take_base.sum())
    base_wr = float(y_test[take_base].mean())
    for h in range(24):
        m = take_base & (test_hours == h)
        n_h = int(m.sum())
        if n_h == 0:
            continue
        wr_h = float(y_test[m].mean()) if n_h > 0 else float("nan")
        # If we exclude this hour: new_n = base_n - n_h, new_correct = base_correct - n_h*wr_h
        if base_n - n_h >= MIN_PARIS:
            new_wr = (base_n * base_wr - n_h * wr_h) / (base_n - n_h)
            gain = (new_wr - base_wr) * 100
            print(f"  {h:>4}  {n_h:>5}  {wr_h*100:>6.2f}%  {gain:>+15.3f}pt  (n={base_n - n_h})")

    print(f"\nBase WR : {base_wr*100:.2f}%, taken : {base_n}")

    # Now sweep : exclude all combinations of 1-3 worst hours
    # Find hours ranked by negative gain
    test_wrs = []
    for h in range(24):
        m = take_base & (test_hours == h)
        n_h = int(m.sum())
        if n_h == 0:
            test_wrs.append((h, 0, 1.0))
            continue
        wr_h = float(y_test[m].mean())
        test_wrs.append((h, n_h, wr_h))

    test_wrs.sort(key=lambda x: x[2])  # ascending WR = worst first

    print(f"\n--- Sweep : exclude worst N hours (by val WR proxy) ---")
    # But: we should use VAL WR not TEST WR (to avoid overfitting!). Compute val WR per hour.
    val_taken = (np.asarray(p_val) >= 0.55)  # rough proxy of would-be-bet on val
    val_hours_taken = val_hours[val_taken]
    val_labels_taken = val_labels[val_taken]
    val_wrs = []
    for h in range(24):
        m = val_hours_taken == h
        n_h = int(m.sum())
        if n_h < 5:
            val_wrs.append((h, n_h, float("nan")))
            continue
        wr_h = float(val_labels_taken[m].mean())
        val_wrs.append((h, n_h, wr_h))
    val_wrs.sort(key=lambda x: x[2] if not np.isnan(x[2]) else 1.0)

    print("Val-based worst hours (by val WR among >0.55 probas):")
    for h, n, wr in val_wrs[:10]:
        wrs = f"{wr*100:.2f}%" if not np.isnan(wr) else "N/A"
        print(f"  h={h:>2} n_val={n:>4} val_WR={wrs}")

    # Try excluding bad hours one at a time, then 2, then 3
    print(f"\nResult of excluding worst hours :")
    for k in (1, 2, 3, 4, 5):
        worst_k = set(h for h, n, wr in val_wrs[:k] if n >= 5)
        exclude_mask = np.isin(test_hours, list(worst_k))
        new_take = take_base & ~exclude_mask
        n_new = int(new_take.sum())
        if n_new < MIN_PARIS:
            print(f"  excl {worst_k} : n={n_new} < {MIN_PARIS} (skip)")
            continue
        wr_new = float(y_test[new_take].mean())
        print(f"  excl {sorted(worst_k)} : n={n_new} WR={wr_new*100:.2f}% (delta = {(wr_new-base_wr)*100:+.3f}pt)")


if __name__ == "__main__":
    main()
