"""Experiment 02: verify bucket stability (val vs test) + test whether adding
hour/cyclic features lifts the model's ranking frontier.

Frontier = test-fitted WR ceiling at 2500 bets (vol2 skipped). If a feature
set raises it toward 60%, the model is genuinely ranking better.
"""
import numpy as np

import lib
import lib2


def evaluate_featureset(name, train, val, test, fc):
    model = lib.train_model(train, val, fc)
    p_val = lib.predict(model, val, fc)
    p_test = lib.predict(model, test, fc)
    yv = val["label"].to_numpy().astype(int)
    yt = test["label"].to_numpy().astype(int)
    av = (val["volatility_state"].to_numpy().astype(int) != 2)
    at = (test["volatility_state"].to_numpy().astype(int) != 2)

    fv = lib2.frontier_at(p_val, yv, av, 2700)     # val-equiv of 2500 test
    ft = lib2.frontier_at(p_test, yt, at, 2500)

    # honest current-V2 strategy at ~2500 test bets
    boost = lib.make_hour_boost()
    r = lib2.wr_at_volume(val, p_val, test, p_test,
                          skip_fn=lib.skip_vol2, axis_fn=lib.axis_weekend,
                          boost_fn=boost, target_n=2500,
                          axis_frac_ratio={0: 1.0, 1: 4.0})
    print(f"  {name:<26} nfeat={len(fc):>2} iter={model.best_iteration:>3}  "
          f"front_val={fv*100:5.2f}%  front_test={ft*100:5.2f}%  "
          f"V2@{r['n']}={r['wr']*100:5.2f}%")
    return model, p_val, p_test


def main():
    df = lib.load_master()
    train0, val0, test0 = lib.temporal_split(df)
    base_fc = lib.feature_cols(df)
    print(f"base features = {len(base_fc)}")

    # ---- baseline ----
    print("\n=== Frontier by feature set (target 2500 test bets, vol2 skip) ===")
    model, p_val, p_test = evaluate_featureset("baseline", train0, val0, test0, base_fc)

    # ---- bucket stability on baseline model ----
    print("\n=== Bucket stability (is the edge real?) ===")
    lib2.bucket_stability(val0, p_val, test0, p_test,
                          lambda d: d["is_weekend"].to_numpy().astype(int),
                          [("weekday", 0), ("weekend", 1)], title="is_weekend")
    lib2.bucket_stability(val0, p_val, test0, p_test,
                          lambda d: d["volatility_state"].to_numpy().astype(int),
                          [("vol0", 0), ("vol1", 1), ("vol2", 2)], title="volatility_state")
    lib2.bucket_stability(val0, p_val, test0, p_test, lib.hour_utc,
                          [(f"h{h:02d}", h) for h in range(24)],
                          frac=0.12, title="hour UTC")

    # ---- hour feature variants ----
    print("\n=== Hour/cyclic feature variants ===")
    variants = {
        "+hour_cos": ["hour_cos"],
        "+hour(int)": ["hour"],
        "+hour_cos+sin2+cos2": ["hour_cos", "hour_sin2", "hour_cos2"],
        "+hour+dow": ["hour", "dow"],
        "+hour+dow+vol_state": ["hour", "dow", "vol_state"],
        "+hour(int)+hour_cos": ["hour", "hour_cos"],
        "+qhour(int)": ["qhour"],
        "+all_cyclic": ["hour_cos", "hour_sin2", "hour_cos2",
                        "hour_sin3", "hour_cos3", "dow_sin", "dow_cos"],
    }
    for name, extra in variants.items():
        tr = lib2.add_time_features(train0, extra)
        va = lib2.add_time_features(val0, extra)
        te = lib2.add_time_features(test0, extra)
        fc = base_fc + extra
        evaluate_featureset(name, tr, va, te, fc)


if __name__ == "__main__":
    main()
