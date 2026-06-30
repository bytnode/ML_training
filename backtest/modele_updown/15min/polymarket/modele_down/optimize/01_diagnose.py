"""Diagnostic: where is the DOWN edge, and how far can volume go at WR>=60%?

Answers:
  Q1  current model + current V2 strategy: honest WR-vs-volume curve (test).
  Q2  test-fitted frontier (optimistic ceiling) at 2500 bets.
  Q3  edge breakdown by bucket (volatility_state, is_weekend, hour).
"""
import numpy as np
import pandas as pd

import lib

TARGET_N, TARGET_WR = 2500, 0.60


def main():
    print("Loading master + split...")
    df = lib.load_master()
    train, val, test = lib.temporal_split(df)
    fc = lib.feature_cols(df)
    print(f"  features={len(fc)}  train={len(train):,}  val={len(val):,}  test={len(test):,}")

    print("Training current model...")
    model = lib.train_model(train, val, fc)
    p_val = lib.predict(model, val, fc)
    p_test = lib.predict(model, test, fc)
    y_test = test["label"].to_numpy().astype(int)
    print(f"  best_iteration={model.best_iteration}  baseline_wr_down={y_test.mean()*100:.2f}%")

    # ---- Q1: current V2 strategy, honest curve ----
    print("\n=== Q1  Current V2 strategy (skip vol2, axis=weekend, hour-boost 20-22) ===")
    print("     weekday vol_frac swept, weekend frac=0.10 fixed")
    curve = lib.current_strategy_curve(val, p_val, test, p_test)
    print(f"  {'wd_frac':>8} {'test_n':>8} {'test_wr':>9}")
    best_ok = None
    for r in curve:
        flag = "  <-- >=60%" if r["wr"] >= TARGET_WR else ""
        print(f"  {r['wd_frac']:>8.3f} {r['n']:>8} {r['wr']*100:>8.2f}%{flag}")
        if r["wr"] >= TARGET_WR and (best_ok is None or r["n"] > best_ok["n"]):
            best_ok = r
    if best_ok:
        print(f"  >> max honest volume at WR>=60%: n={best_ok['n']} "
              f"(wd_frac={best_ok['wd_frac']})")

    # ---- Q2: test-fitted frontier (ceiling) ----
    print("\n=== Q2  Test-fitted frontier = optimistic ceiling ===")
    vol = test["volatility_state"].to_numpy().astype(int)
    allowed_all = np.ones(len(test), dtype=bool)
    allowed_novol2 = vol != 2
    for label, mask in (("all buckets allowed", allowed_all),
                        ("vol2 skipped (production)", allowed_novol2)):
        fr = lib.test_frontier(p_test, y_test, mask,
                               [1500, 1800, 2000, 2200, 2500, 2800, 3000, 3500])
        s = "  ".join(f"n{d['n']}={d['wr']*100:.1f}%" for d in fr)
        print(f"  [{label}]\n     {s}")

    # ---- Q3: edge by bucket ----
    print("\n=== Q3  Edge by bucket (top-X% p_down WR within each bucket) ===")
    hour = lib.hour_utc(test)
    wknd = test["is_weekend"].to_numpy().astype(int)

    print("\n  -- by volatility_state --")
    _bucket_table(p_test, y_test, vol, ["vol0", "vol1", "vol2"], [0, 1, 2])

    print("\n  -- by is_weekend --")
    _bucket_table(p_test, y_test, wknd, ["weekday", "weekend"], [0, 1])

    print("\n  -- by hour (UTC), vol!=2 only --")
    nv2 = vol != 2
    _hour_table(p_test[nv2], y_test[nv2], hour[nv2])


def _bucket_table(p, y, keys, names, vals):
    print(f"  {'bucket':<10} {'n_bars':>8} {'base':>7}  "
          f"{'top20%':>16} {'top10%':>16} {'top5%':>16}")
    for name, v in zip(names, vals):
        m = keys == v
        n = int(m.sum())
        if n == 0:
            continue
        pm, ym = p[m], y[m]
        base = ym.mean()
        cells = []
        for frac in (0.20, 0.10, 0.05):
            k = max(1, int(n * frac))
            order = np.argsort(-pm)[:k]
            cells.append(f"{k:>5}@{ym[order].mean()*100:5.1f}%")
        print(f"  {name:<10} {n:>8} {base*100:>6.1f}%  "
              f"{cells[0]:>16} {cells[1]:>16} {cells[2]:>16}")


def _hour_table(p, y, hour):
    print(f"  {'hr':>3} {'n':>6} {'base':>7} {'top15%':>14} {'top8%':>14} {'top4%':>14}")
    rows = []
    for h in range(24):
        m = hour == h
        n = int(m.sum())
        if n < 50:
            continue
        pm, ym = p[m], y[m]
        cells = {}
        for frac in (0.15, 0.08, 0.04):
            k = max(1, int(n * frac))
            order = np.argsort(-pm)[:k]
            cells[frac] = (k, ym[order].mean())
        rows.append((h, n, ym.mean(), cells))
    # sort by top8% wr desc to see best hours
    rows.sort(key=lambda r: -r[3][0.08][1])
    for h, n, base, cells in rows:
        c15, c8, c4 = cells[0.15], cells[0.08], cells[0.04]
        print(f"  {h:>3} {n:>6} {base*100:>6.1f}% "
              f"{c15[0]:>4}@{c15[1]*100:5.1f}% "
              f"{c8[0]:>4}@{c8[1]*100:5.1f}% "
              f"{c4[0]:>4}@{c4[1]*100:5.1f}%")


if __name__ == "__main__":
    main()
