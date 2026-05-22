"""
Analyze when the model makes mistakes among its top-confidence predictions.

For the 1500 highest P(down) test bets, look at what features differ between
correct (label=1) and incorrect (label=0) predictions. Find features where
the mean differs meaningfully between wins and losses.

This guides hard-filter design at inference: skip bets where a feature is
in the "loser region".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import (
    load_dataset, feature_columns, train_once,
)


def main():
    df = load_dataset()
    fc = feature_columns(df)
    print(f"Dataset: {len(df):,} rows, {len(fc)} features.")
    run = train_once(df, fc, seed=42)
    p_test = run["p_test"]
    y_test = run["y_test"]
    test = run["test"].reset_index(drop=True)

    # Top 1500 by P(down)
    N = 1500
    order = np.argsort(-p_test)[:N]
    p_top = p_test[order]
    y_top = y_test[order]
    test_top = test.iloc[order].reset_index(drop=True)
    test_top["p_down"] = p_top
    test_top["correct"] = y_top

    print(f"\nTop {N} test bets: WR = {y_top.mean()*100:.2f}%")
    print(f"  win avg P(down) = {p_top[y_top == 1].mean():.4f}")
    print(f"  loss avg P(down) = {p_top[y_top == 0].mean():.4f}")

    # Compare features between wins and losses
    print(f"\n{'feature':<30} {'win_mean':>10} {'loss_mean':>10} {'diff':>10}  {'t':>7}")
    print("-" * 80)
    stats = []
    for col in fc:
        win_v = test_top.loc[test_top["correct"] == 1, col].dropna()
        loss_v = test_top.loc[test_top["correct"] == 0, col].dropna()
        if len(win_v) < 10 or len(loss_v) < 10:
            continue
        w_mean = float(win_v.mean())
        l_mean = float(loss_v.mean())
        diff = w_mean - l_mean
        w_std = float(win_v.std())
        l_std = float(loss_v.std())
        pooled = np.sqrt((w_std**2 / len(win_v)) + (l_std**2 / len(loss_v))) + 1e-9
        t = diff / pooled
        stats.append((col, w_mean, l_mean, diff, t))

    # Sort by abs(t)
    stats.sort(key=lambda r: -abs(r[4]))
    for s in stats[:15]:
        col, w, l, d, t = s
        print(f"  {col:<28} {w:>10.5f} {l:>10.5f} {d:>+10.5f}  {t:>+7.2f}")

    # Look at hours where mistakes cluster
    test_top["hour"] = (test_top["timestamp"].astype("int64") // (3600 * 1000)) % 24
    print(f"\n{'hour':>5} {'n':>5} {'wr':>7}")
    for h, g in test_top.groupby("hour"):
        wr = g["correct"].mean()
        if len(g) >= 20:
            marker = " <-- bad" if wr < 0.5 else (" <-- good" if wr > 0.7 else "")
            print(f"  {h:>3}  {len(g):>5}  {wr*100:>5.2f}%{marker}")

    # Look at wknd
    print("\n  weekend split:")
    for w, g in test_top.groupby(test_top["is_weekend"]):
        print(f"    is_weekend={w}: n={len(g)}, wr={g['correct'].mean()*100:.2f}%")

    # Look at volatility_state
    print("\n  volatility split:")
    for v, g in test_top.groupby(test_top["volatility_state"]):
        print(f"    vol_state={v}: n={len(g)}, wr={g['correct'].mean()*100:.2f}%")


if __name__ == "__main__":
    main()
