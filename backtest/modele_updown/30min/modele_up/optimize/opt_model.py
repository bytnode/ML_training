"""Walk-forward hyperparameter / train-window sweep on the live parquet.

For each model variant: train one production-mirror model per yearly window
(6 windows), evaluate at a FIXED calibration (no skip, wd=0.02 we=0.10, hour
boost 20-22) so the comparison isolates the MODEL. Rank by older-pooled WR
(windows 1..5, headline-excluded => honest).

A variant is adopted only if older_wr genuinely beats baseline AND the recent
window stays Pareto vs (1800, 60.06%).
"""
import numpy as np
import pandas as pd

import lib

MS_DAY = 86_400_000
PROD_GOOD_HOURS = (20, 21, 22)
PROD_DELTA = 0.02
CALIB_WD, CALIB_WE = 0.02, 0.10


def windows(df, train_days=4000):
    ts_max = int(df["timestamp"].max())
    out = []
    for i in range(6):
        end_ts = ts_max - i * 365 * MS_DAY
        tr, va, te = lib.temporal_split(df, train_days=train_days,
                                        test_days=365, end_ts=end_ts)
        if len(te) < 5000 or len(va) < 2000 or len(tr) < 20000:
            continue
        out.append((i, tr, va, te))
    return out


def score(df, fc, hp, train_days, tag):
    boost = lib.make_hour_boost(PROD_GOOD_HOURS, PROD_DELTA)
    no_skip = lambda d: np.zeros(len(d), dtype=bool)
    per, older_takes, recent, iters = [], [], None, []
    for i, tr, va, te in windows(df, train_days):
        m = lib.train_model(tr, va, fc, hp=hp)
        iters.append(int(m.best_iteration))
        p_val = lib.predict(m, va, fc)
        p_test = lib.predict(m, te, fc)
        thr = lib.calibrate(va, p_val, skip_fn=no_skip, axis_fn=lib.axis_weekend,
                            vol_frac={0: CALIB_WD, 1: CALIB_WE})
        res = lib.evaluate(te, p_test, thr, skip_fn=no_skip,
                           axis_fn=lib.axis_weekend, boost_fn=boost)
        y = te["label"].to_numpy().astype(int)
        per.append((i, res["n"], res["wr"]))
        if i == 0:
            recent = (res["n"], res["wr"])
        else:
            older_takes.append(y[res["take"]])
    allo = np.concatenate(older_takes) if older_takes else np.array([])
    on, ow = len(allo), (float(allo.mean()) if len(allo) else float("nan"))
    print(f"  {tag:<26} it~{int(np.median(iters)):>3}  recent={recent[0]:>5}@"
          f"{recent[1]*100:5.2f}%  older={on:>5}@{ow*100:5.2f}%")
    return ow, recent


def main():
    df = lib.load_master()
    fc = lib.feature_cols(df)
    print(f"features={len(fc)}\n")

    print("=== baseline (current config HP, train_days=4000) ===")
    base_ow, base_rec = score(df, fc, None, 4000, "baseline")

    print("\n=== max_depth ===")
    for d in (4, 6, 7):
        score(df, fc, {"max_depth": d}, 4000, f"depth={d}")

    print("\n=== learning_rate ===")
    for lr in (0.03, 0.04, 0.07, 0.10):
        score(df, fc, {"learning_rate": lr}, 4000, f"lr={lr}")

    print("\n=== min_child_weight ===")
    for m in (5, 10, 30, 50, 80):
        score(df, fc, {"min_child_weight": m}, 4000, f"mcw={m}")

    print("\n=== regularization ===")
    for hp, tag in [({"reg_lambda": 1.0}, "reg_lambda=1"),
                    ({"reg_lambda": 3.0}, "reg_lambda=3"),
                    ({"reg_lambda": 5.0}, "reg_lambda=5"),
                    ({"reg_alpha": 1.0}, "reg_alpha=1"),
                    ({"gamma": 0.5}, "gamma=0.5"),
                    ({"gamma": 3.0}, "gamma=3.0")]:
        score(df, fc, hp, 4000, tag)

    print("\n=== subsample / colsample ===")
    for hp, tag in [({"subsample": 0.7}, "subsample=0.7"),
                    ({"subsample": 1.0}, "subsample=1.0"),
                    ({"colsample_bytree": 0.7}, "colsample=0.7"),
                    ({"colsample_bytree": 1.0}, "colsample=1.0")]:
        score(df, fc, hp, 4000, tag)

    print("\n=== scale_pos_weight ===")
    for hp, tag in [({"scale_pos_weight": False}, "spw=False"),
                    ({"scale_pos_weight": 1.05}, "spw=1.05")]:
        score(df, fc, hp, 4000, tag)

    print("\n=== train_days (ETH history starts 2017) ===")
    for td in (2200, 2800, 3400, 4600, None):
        score(df, fc, None, td, f"train_days={td}")

    print(f"\nbaseline older={base_ow*100:.2f}%  recent={base_rec[0]}@"
          f"{base_rec[1]*100:.2f}%  -- adopt a variant only if it clearly beats this")


if __name__ == "__main__":
    main()
