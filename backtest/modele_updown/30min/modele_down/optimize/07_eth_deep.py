"""Experiment 07: squeeze ETH for all it's worth. Test ETH feature groups,
then greedily build the best subset by front_val.
"""
import numpy as np

import lib
import lib2
import ethfeats


def score(train, val, test, fc, tag, base_fv=None):
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
    d = "" if base_fv is None else f" ({(fv-base_fv)*100:+.2f})"
    print(f"  {tag:<28} nf={len(fc):>3} it={model.best_iteration:>3}  "
          f"front_val={fv*100:5.2f}{d:<9} front_test={ft*100:5.2f}  "
          f"clean@2500={r['wr']*100:5.2f}")
    return fv


def main():
    df = lib.load_master()
    base_fc = lib.feature_cols(df)
    eth = ethfeats.build_eth(df["timestamp"].to_numpy())
    merged = df.merge(eth, left_on="timestamp", right_on="timestamp_ms",
                      how="left").drop(columns=["timestamp_ms"])
    print(f"merged: {len(merged):,} rows, {len(merged.columns)} cols\n")

    train, val, test = lib.temporal_split(merged)
    print("=== ETH groups ===")
    base_fv = score(train, val, test, base_fc, "baseline", None)
    base_fv0 = base_fv

    for g, cols in ethfeats.ETH_GROUPS.items():
        c = [x for x in cols if x in merged.columns]
        score(train, val, test, base_fc + c, f"+{g}", base_fv0)

    # greedy group selection by front_val
    print("\n=== greedy group accumulation ===")
    remaining = dict(ethfeats.ETH_GROUPS)
    chosen, acc = [], []
    cur_fv = base_fv0
    while remaining:
        best_g, best_fv, best_cols = None, cur_fv, None
        for g, cols in remaining.items():
            c = [x for x in cols if x in merged.columns]
            fv = score(train, val, test, base_fc + acc + c,
                       f"  try {g}", None)
            if fv > best_fv + 0.0005:
                best_g, best_fv, best_cols = g, fv, c
        if best_g is None:
            break
        chosen.append(best_g)
        acc += best_cols
        cur_fv = best_fv
        print(f"  --> picked {best_g}, front_val={cur_fv*100:.2f}, chosen={chosen}")
        del remaining[best_g]

    print(f"\n=== FINAL ETH subset: {chosen} ({len(acc)} cols) ===")
    score(train, val, test, base_fc + acc, "baseline+ethbest", base_fv0)
    # save chosen columns
    with open("eth_best_cols.txt", "w") as f:
        f.write("\n".join(acc))
    print(f"wrote eth_best_cols.txt ({len(acc)} cols)")


if __name__ == "__main__":
    main()
