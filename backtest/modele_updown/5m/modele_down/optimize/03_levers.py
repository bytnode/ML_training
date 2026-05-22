"""Experiment 03: sweep the cheap levers (train window, hyperparameters) on
the production test window. Metric = honest WR at 2500 test bets.

honest strategy for comparison ("V2clean"): skip vol2, axis=weekend,
weekend frac = 4x weekday frac, NO hour boost (hour boost is overfit per exp02).
Calibrate on val, target 2500 test bets, report test WR. Also frontier ceiling.
"""
import numpy as np

import lib
import lib2


def score(train, val, test, fc, hp=None, tag=""):
    model = lib.train_model(train, val, fc, hp=hp)
    p_val = lib.predict(model, val, fc)
    p_test = lib.predict(model, test, fc)
    yv = val["label"].to_numpy().astype(int)
    yt = test["label"].to_numpy().astype(int)
    av = val["volatility_state"].to_numpy().astype(int) != 2
    at = test["volatility_state"].to_numpy().astype(int) != 2

    fv = lib2.frontier_at(p_val, yv, av, 2700)
    ft = lib2.frontier_at(p_test, yt, at, 2500)

    r = lib2.wr_at_volume(val, p_val, test, p_test,
                          skip_fn=lib.skip_vol2, axis_fn=lib.axis_weekend,
                          boost_fn=None, target_n=2500,
                          axis_frac_ratio={0: 1.0, 1: 4.0})
    r18 = lib2.wr_at_volume(val, p_val, test, p_test,
                            skip_fn=lib.skip_vol2, axis_fn=lib.axis_weekend,
                            boost_fn=None, target_n=1800,
                            axis_frac_ratio={0: 1.0, 1: 4.0})
    print(f"  {tag:<30} iter={model.best_iteration:>3}  "
          f"front_val={fv*100:5.2f}  front_test={ft*100:5.2f}  "
          f"clean@1800={r18['wr']*100:5.2f}  clean@2500={r['wr']*100:5.2f}")
    return r["wr"]


def main():
    df = lib.load_master()
    fc = lib.feature_cols(df)

    print("=== TRAIN_DAYS sweep (production window) ===")
    for td in (800, 1200, 1800, 2500, 3200, 4000, None):
        train, val, test = lib.temporal_split(df, train_days=td)
        score(train, val, test, fc, tag=f"train_days={td}  n_tr={len(train)}")

    print("\n=== Hyperparameter sweep (train_days=4000) ===")
    train, val, test = lib.temporal_split(df, train_days=4000)
    score(train, val, test, fc, tag="baseline")
    for hp, tag in [
        ({"max_depth": 4}, "max_depth=4"),
        ({"max_depth": 6}, "max_depth=6"),
        ({"max_depth": 7}, "max_depth=7"),
        ({"max_depth": 8}, "max_depth=8"),
        ({"max_depth": 6, "learning_rate": 0.03}, "depth6 lr0.03"),
        ({"max_depth": 6, "learning_rate": 0.02}, "depth6 lr0.02"),
        ({"max_depth": 6, "learning_rate": 0.10}, "depth6 lr0.10"),
        ({"min_child_weight": 5}, "min_child_weight=5"),
        ({"min_child_weight": 50}, "min_child_weight=50"),
        ({"min_child_weight": 100}, "min_child_weight=100"),
        ({"subsample": 0.6}, "subsample=0.6"),
        ({"subsample": 1.0}, "subsample=1.0"),
        ({"colsample_bytree": 0.6}, "colsample=0.6"),
        ({"colsample_bytree": 1.0}, "colsample=1.0"),
        ({"reg_lambda": 1.0}, "reg_lambda=1"),
        ({"reg_lambda": 5.0}, "reg_lambda=5"),
        ({"reg_alpha": 1.0}, "reg_alpha=1"),
        ({"gamma": 0.0}, "gamma=0"),
        ({"gamma": 3.0}, "gamma=3"),
        ({"scale_pos_weight": False}, "spw=False"),
        ({"scale_pos_weight": 1.1}, "spw=1.1"),
        ({"max_depth": 6, "min_child_weight": 50, "colsample_bytree": 0.7}, "depth6 mcw50 cs0.7"),
    ]:
        score(train, val, test, fc, hp=hp, tag=tag)


if __name__ == "__main__":
    main()
