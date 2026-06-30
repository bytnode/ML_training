"""Final candidate comparison on the locked model (ETH features + mcw=50).

Trains the 6 production-mirror windows once, then evaluates a short list of
named calibration candidates with full per-window detail. Metric notes:
  recent  = window 0 (production headline window)
  older   = windows 1..5 pooled (honest)
  EV      = n * (wr - 0.5)  (relative profit proxy)
Original production point: 1800 bets, 60.06% WR (EV 181.1).
"""
import numpy as np

import lib
import opt_calib as oc


def main():
    df = lib.load_master()
    fc = lib.feature_cols(df)
    wins = oc.build_windows(df)
    print(f"features={len(fc)}  windows={len(wins)}  (model = ETH + mcw50)")
    cache = oc.train_cache(wins, fc)   # train_cache hardcodes min_child_weight=50

    boost = lib.make_hour_boost(oc.PROD_GOOD_HOURS, oc.PROD_DELTA)
    no_skip = lambda d: np.zeros(len(d), dtype=bool)

    candidates = [
        ("skip_vol2 wd.025 we.10 (=prod calib)", lib.skip_vol2, 0.025, 0.10),
        ("no_skip   wd.020 we.10", no_skip, 0.020, 0.10),
        ("no_skip   wd.0225 we.10", no_skip, 0.0225, 0.10),
        ("no_skip   wd.025 we.10", no_skip, 0.025, 0.10),
        ("no_skip   wd.030 we.10", no_skip, 0.030, 0.10),
        ("no_skip   wd.035 we.10", no_skip, 0.035, 0.10),
        ("no_skip   wd.0225 we.11", no_skip, 0.0225, 0.11),
        ("no_skip   wd.0225 we.12", no_skip, 0.0225, 0.12),
        ("no_skip   wd.040 we.11", no_skip, 0.040, 0.11),
    ]

    print(f"\n  {'config':<40}{'rec_n':>6}{'rec_wr':>8}{'rec_EV':>8}"
          f"{'old_n':>7}{'old_wr':>8}  per-window")
    print("  " + "-" * 110)
    print(f"  {'ORIGINAL production (BTC-only)':<40}{1800:>6}{60.06:>7.2f}%"
          f"{181.1:>8.1f}{'':>7}{56.54:>7.2f}%")
    for name, sk, wd, we in candidates:
        r = oc.eval_config(cache, wd, we, sk, boost)
        rn, rw = r["recent_n"], r["recent_wr"]
        ev = rn * (rw - 0.5)
        per = " ".join(f"{n}@{wr*100:.1f}" for i, n, wr in r["per"])
        flag = ""
        if rn >= 1800 and rw >= 0.6006:
            flag = "  <-- PARETO"
        print(f"  {name:<40}{rn:>6}{rw*100:>7.2f}%{ev:>8.1f}"
              f"{r['older_n']:>7}{r['older_wr']*100:>7.2f}%  [{per}]{flag}")


if __name__ == "__main__":
    main()
