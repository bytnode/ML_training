"""Experiment 04: test OHLCV-derived feature groups. Metric = front_val
(global top-2700 val WR by p_down) + honest clean@2500 test WR.

A group "helps" only if it lifts front_val (honest model quality). front_test
gains alone can be test-window luck (see exp02).
"""
import numpy as np

import lib
import lib2
import feats


def score(train, val, test, fc, tag):
    model = lib.train_model(train, val, fc)
    p_val = lib.predict(model, val, fc)
    p_test = lib.predict(model, test, fc)
    yv = val["label"].to_numpy().astype(int)
    yt = test["label"].to_numpy().astype(int)
    av = val["volatility_state"].to_numpy().astype(int) != 2
    at = test["volatility_state"].to_numpy().astype(int) != 2
    fv = lib2.frontier_at(p_val, yv, av, 2700)
    ft = lib2.frontier_at(p_test, yt, at, 2500)
    r = lib2.wr_at_volume(val, p_val, test, p_test, skip_fn=lib.skip_vol2,
                          axis_fn=lib.axis_weekend, target_n=2500,
                          axis_frac_ratio={0: 1.0, 1: 4.0})
    print(f"  {tag:<26} nf={len(fc):>3} it={model.best_iteration:>3}  "
          f"front_val={fv*100:5.2f}  front_test={ft*100:5.2f}  "
          f"clean@2500={r['wr']*100:5.2f}")
    return fv, r["wr"]


def main():
    df = lib.load_master()
    base_fc = lib.feature_cols(df)
    cand = feats.build_candidates()

    # merge candidates onto master by timestamp
    merged = df.merge(cand, left_on="timestamp", right_on="timestamp_ms",
                      how="left").drop(columns=["timestamp_ms"])
    print(f"merged: {len(merged):,} rows, {len(merged.columns)} cols\n")

    train0, val0, test0 = lib.temporal_split(merged)
    print("=== baseline (25 features) ===")
    base_fv, _ = score(train0, val0, test0, base_fc, "baseline")

    print("\n=== add one feature group at a time ===")
    group_results = {}
    for gname, gcols in feats.GROUPS.items():
        cols = [c for c in gcols if c in merged.columns]
        miss = [c for c in gcols if c not in merged.columns]
        if miss:
            print(f"  ({gname}: missing {miss})")
        fv, wr = score(train0, val0, test0, base_fc + cols, f"+{gname}")
        group_results[gname] = (fv - base_fv, wr)

    # cumulative: add groups that helped front_val, sorted by gain
    helpers = sorted([g for g, (d, _) in group_results.items() if d > 0.05],
                     key=lambda g: -group_results[g][0])
    print(f"\n=== cumulative add (groups lifting front_val): {helpers} ===")
    acc = []
    for g in helpers:
        acc += [c for c in feats.GROUPS[g] if c in merged.columns]
        score(train0, val0, test0, base_fc + acc, f"+{'+'.join(helpers[:helpers.index(g)+1])}")

    print("\n=== all groups together ===")
    allcols = []
    for gcols in feats.GROUPS.values():
        allcols += [c for c in gcols if c in merged.columns]
    allcols = list(dict.fromkeys(allcols))
    score(train0, val0, test0, base_fc + allcols, "+ALL")


if __name__ == "__main__":
    main()
