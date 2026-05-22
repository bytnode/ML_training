"""
Diagnostic des heures UTC pour DOWN.

Train une fois, puis pour chaque heure UTC h calcule sur la VAL :
  - n_total : nombre de bougies a cette heure dans la val
  - p_down_mean : moyenne de P(down) brute
  - wr_brute : fraction de label=1 (bougies rouges)
  - wr_top_q : WR du top quantile P(down) (= si on prenait les top q% pour ces heures)

Permet d'identifier :
  - bad hours : WR brute basse + WR top_q basse -> SKIP
  - good hours : WR brute haute + WR top_q haute -> BOOST

NB : le top quantile est specifique a chaque heure pour eviter le biais
volume. Pour chaque heure on simule "et si on prenait les meilleurs X% du
val a cette heure".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import (
    load_dataset, feature_columns, train_once, hour_of_day_from_ms,
)


def main():
    print("Loading dataset...")
    df = load_dataset()
    fc = feature_columns(df)
    print(f"  {len(df):,} rows, {len(fc)} features.")

    print("\nTraining one model with current config...")
    run = train_once(df, fc, seed=42)
    print(f"  best_iter={run['best_iter']} val_logloss={run['val_logloss']:.5f} "
          f"test_logloss={run['test_logloss']:.5f}")

    val = run["val"]
    p_val = run["p_val"]

    hours = hour_of_day_from_ms(val["timestamp"].to_numpy())
    labels = val["label"].to_numpy().astype(int)

    p_down_mean_global = float(p_val.mean())
    wr_global = float(labels.mean())
    print(f"\n  val:  P(down)_mean={p_down_mean_global:.4f}  WR_brute={wr_global:.4f}  "
          f"(n={len(val):,})")

    # WR par heure
    print("\n" + "=" * 110)
    print("  Per-hour diagnostic on VAL")
    print("=" * 110)
    print(f"  {'h':>3}  {'n_total':>8}  {'wr_brute':>8}  {'p_mean':>8}  "
          f"{'top1%_wr':>9}  {'top1%_n':>8}  {'top3%_wr':>9}  {'top3%_n':>8}  "
          f"{'top5%_wr':>9}  {'top5%_n':>8}")
    print("  " + "-" * 108)

    rows = []
    for h in range(24):
        mask = hours == h
        n = int(mask.sum())
        if n == 0:
            continue
        wr_h = float(labels[mask].mean())
        p_mean_h = float(p_val[mask].mean())
        pb = p_val[mask]
        yb = labels[mask]

        def top_q_wr(q):
            if len(pb) == 0:
                return 0.0, 0
            T = float(np.quantile(pb, 1.0 - q))
            sel = pb >= T
            n_sel = int(sel.sum())
            wr_sel = float(yb[sel].mean()) if n_sel > 0 else 0.0
            return wr_sel, n_sel

        t1, n1 = top_q_wr(0.01)
        t3, n3 = top_q_wr(0.03)
        t5, n5 = top_q_wr(0.05)

        rows.append({
            "h": h, "n_total": n, "wr_brute": wr_h, "p_mean": p_mean_h,
            "top1pct_wr": t1, "top1pct_n": n1,
            "top3pct_wr": t3, "top3pct_n": n3,
            "top5pct_wr": t5, "top5pct_n": n5,
        })
        marker = ""
        if wr_h > wr_global + 0.025: marker += " ++"
        elif wr_h > wr_global + 0.010: marker += " +"
        elif wr_h < wr_global - 0.025: marker += " --"
        elif wr_h < wr_global - 0.010: marker += " -"
        print(f"  {h:>3}  {n:>8}  {wr_h:>8.4f}  {p_mean_h:>8.4f}  "
              f"{t1:>9.4f}  {n1:>8}  {t3:>9.4f}  {n3:>8}  "
              f"{t5:>9.4f}  {n5:>8}{marker}")

    print("  " + "=" * 108)
    print("  Legend: ++ wr_brute > global+2.5pp  + wr_brute > global+1pp  "
          "-- wr_brute < global-2.5pp  - wr_brute < global-1pp")

    # Same diagnostic on TEST (out of sample, only for sanity check, do NOT
    # use to pick hours)
    test = run["test"]
    p_test = run["p_test"]
    hours_t = hour_of_day_from_ms(test["timestamp"].to_numpy())
    labels_t = run["y_test"]

    print("\n" + "=" * 110)
    print("  Per-hour diagnostic on TEST (for INFO only, do NOT calibrate on this)")
    print("=" * 110)
    print(f"  {'h':>3}  {'n_total':>8}  {'wr_brute':>8}  {'p_mean':>8}  "
          f"{'top3%_wr':>9}  {'top3%_n':>8}")
    print("  " + "-" * 60)
    wr_global_t = float(labels_t.mean())
    for h in range(24):
        mask = hours_t == h
        n = int(mask.sum())
        if n == 0:
            continue
        wr_h = float(labels_t[mask].mean())
        p_mean_h = float(p_test[mask].mean())
        pb = p_test[mask]
        yb = labels_t[mask]
        T = float(np.quantile(pb, 0.97))
        sel = pb >= T
        n_sel = int(sel.sum())
        wr_sel = float(yb[sel].mean()) if n_sel > 0 else 0.0
        marker = ""
        if wr_h > wr_global_t + 0.025: marker += " ++"
        elif wr_h > wr_global_t + 0.010: marker += " +"
        elif wr_h < wr_global_t - 0.025: marker += " --"
        elif wr_h < wr_global_t - 0.010: marker += " -"
        print(f"  {h:>3}  {n:>8}  {wr_h:>8.4f}  {p_mean_h:>8.4f}  "
              f"{wr_sel:>9.4f}  {n_sel:>8}{marker}")


if __name__ == "__main__":
    main()
