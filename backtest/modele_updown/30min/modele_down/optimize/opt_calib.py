"""Honest calibration sweep for modele_down (reads whatever parquet is live).

Trains one production-mirror model per yearly walk-forward window
(train_days=4000, test_days=365, val_fraction=0.1, seed 42), caches each
window's val/test predictions, then sweeps the V2 calibration parameters
WITHOUT retraining.

Per config:
  recent       : window 0 = the production headline window -> (n, wr)
  older_pooled : windows 1..5 pooled -> (n, wr)  [headline-excluded => honest]

Sweeps wd_frac x we_frac for skip-mode in {skip vol2, no skip}. Hour boost
kept at the production setting (20-22 @ 0.02).

Production point to beat: recent = (1800 bets, 60.06% WR).
"""
import numpy as np
import pandas as pd

import lib

MS_DAY = 86_400_000
BASE_N, BASE_WR = 1800, 0.6006
PROD_GOOD_HOURS = (20, 21, 22)
PROD_DELTA = 0.02


def build_windows(df, k=6, test_days=365, train_days=4000):
    ts_max = int(df["timestamp"].max())
    wins = []
    for i in range(k):
        end_ts = ts_max - i * test_days * MS_DAY
        train, val, test = lib.temporal_split(df, train_days=train_days,
                                              test_days=test_days, end_ts=end_ts)
        if len(test) < 5000 or len(val) < 2000 or len(train) < 20000:
            continue
        wins.append((i, train, val, test))
    return wins


def train_cache(wins, fc):
    cache = []
    for i, train, val, test in wins:
        m = lib.train_model(train, val, fc, hp={"min_child_weight": 50})
        cache.append(dict(
            i=i, val=val, p_val=lib.predict(m, val, fc),
            test=test, p_test=lib.predict(m, test, fc),
            best_iter=int(m.best_iteration)))
        print(f"  window {i}: val={len(val):,} test={len(test):,} "
              f"best_iter={m.best_iteration}")
    return cache


def eval_config(cache, wd_frac, we_frac, skip_fn, boost_fn):
    per, older_takes, recent = [], [], None
    for c in cache:
        thr = lib.calibrate(c["val"], c["p_val"], skip_fn=skip_fn,
                            axis_fn=lib.axis_weekend,
                            vol_frac={0: wd_frac, 1: we_frac})
        res = lib.evaluate(c["test"], c["p_test"], thr, skip_fn=skip_fn,
                           axis_fn=lib.axis_weekend, boost_fn=boost_fn)
        y = c["test"]["label"].to_numpy().astype(int)
        per.append((c["i"], res["n"], res["wr"]))
        if c["i"] == 0:
            recent = (res["n"], res["wr"])
        else:
            older_takes.append(y[res["take"]])
    allo = np.concatenate(older_takes) if older_takes else np.array([])
    return dict(recent_n=recent[0], recent_wr=recent[1],
                older_n=len(allo),
                older_wr=float(allo.mean()) if len(allo) else float("nan"),
                per=per)


def main():
    print("Loading master + building 6 yearly windows...")
    df = lib.load_master()
    fc = lib.feature_cols(df)
    wins = build_windows(df)
    print(f"  features={len(fc)}  windows={len(wins)}")
    print("Training one production-mirror model per window (seed 42)...")
    cache = train_cache(wins, fc)

    boost = lib.make_hour_boost(PROD_GOOD_HOURS, PROD_DELTA)
    no_skip = lambda d: np.zeros(len(d), dtype=bool)
    skip_modes = {"skip_vol2": lib.skip_vol2, "no_skip": no_skip}

    # production baseline reference (skip vol2, wd 0.025, we 0.10)
    base = eval_config(cache, 0.025, 0.10, lib.skip_vol2, boost)
    print("\n=== PRODUCTION baseline (skip vol2, wd=0.025 we=0.10 hourboost) ===")
    print(f"  recent      : n={base['recent_n']}  wr={base['recent_wr']*100:.2f}%")
    print(f"  older_pooled: n={base['older_n']}  wr={base['older_wr']*100:.2f}%")
    per = "  ".join(f"w{i}:{n}@{wr*100:.1f}" for i, n, wr in base["per"])
    print(f"  per-window  : {per}")
    base_older = base["older_wr"]

    wd_grid = np.round(np.arange(0.0075, 0.0601, 0.0025), 4)
    we_grid = np.round(np.arange(0.04, 0.281, 0.01), 4)

    all_wins = []
    for sk_name, sk_fn in skip_modes.items():
        for wd in wd_grid:
            for we in we_grid:
                r = eval_config(cache, float(wd), float(we), sk_fn, boost)
                pareto = (r["recent_n"] >= BASE_N and r["recent_wr"] >= BASE_WR
                          and (r["recent_n"] > BASE_N or r["recent_wr"] > BASE_WR))
                honest = r["older_wr"] >= base_older - 0.0003
                if pareto and honest:
                    all_wins.append((sk_name, float(wd), float(we), r))

    print(f"\n=== {len(all_wins)} honest Pareto wins "
          f"(recent>=1800/60.06% AND older_wr>=baseline) ===")
    # rank by older_wr (honest) then recent_wr
    all_wins.sort(key=lambda x: (-x[3]["older_wr"], -x[3]["recent_wr"]))
    print(f"  {'skip':<10}{'wd':>7}{'we':>7} | {'rec_n':>6}{'rec_wr':>8} | "
          f"{'old_n':>6}{'old_wr':>8}")
    for sk, wd, we, r in all_wins[:25]:
        print(f"  {sk:<10}{wd:>7.4f}{we:>7.4f} | {r['recent_n']:>6}"
              f"{r['recent_wr']*100:>7.2f}% | {r['older_n']:>6}"
              f"{r['older_wr']*100:>7.2f}%")

    if all_wins:
        # recommend: best older_wr with recent comfortably Pareto
        cand = [w for w in all_wins
                if w[3]["recent_n"] >= 1810 and w[3]["recent_wr"] >= 0.602]
        pick = cand[0] if cand else all_wins[0]
        sk, wd, we, r = pick
        per = "  ".join(f"w{i}:{n}@{wr*100:.1f}" for i, n, wr in r["per"])
        print(f"\n=== RECOMMENDED ===")
        print(f"  skip={sk}  wd_frac={wd}  we_frac={we}")
        print(f"  recent      : n={r['recent_n']}  wr={r['recent_wr']*100:.2f}%  "
              f"(baseline 1800 / 60.06%)")
        print(f"  older_pooled: n={r['older_n']}  wr={r['older_wr']*100:.2f}%  "
              f"(baseline {base_older*100:.2f}%)")
        print(f"  per-window  : {per}")


if __name__ == "__main__":
    main()
