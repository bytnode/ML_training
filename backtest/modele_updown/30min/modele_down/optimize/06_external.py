"""Experiment 06: do external data sources (ETH, Coinbase, funding) lift the
model's honest ranking frontier? This is the decisive test for new signal.

Metric: front_val (honest top-2700 val WR). A source only counts if it lifts
front_val beyond noise (~+0.3pp). Also reports clean@2500 test.
"""
import numpy as np

import lib
import lib2
import extfeats


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
    delta = "" if base_fv is None else f" ({(fv-base_fv)*100:+.2f})"
    print(f"  {tag:<24} nf={len(fc):>3} it={model.best_iteration:>3}  "
          f"front_val={fv*100:5.2f}{delta:<9}  front_test={ft*100:5.2f}  "
          f"clean@2500={r['wr']*100:5.2f}")
    return fv


def main():
    df = lib.load_master()
    base_fc = lib.feature_cols(df)
    ext = extfeats.build_external(df["timestamp"].to_numpy())

    merged = df.merge(ext, left_on="timestamp", right_on="timestamp_ms",
                      how="left").drop(columns=["timestamp_ms"])
    # derived cross features
    if "xf_eth_close_ret" in merged.columns:
        merged["xf_eth_btc_div"] = merged["xf_eth_close_ret"] - merged["close_ret"]
    if "xf_cb_close_ret" in merged.columns:
        merged["xf_cb_btc_div"] = merged["xf_cb_close_ret"] - merged["close_ret"]
    print(f"merged: {len(merged):,} rows, {len(merged.columns)} cols\n")

    train, val, test = lib.temporal_split(merged)
    print("=== external data groups ===")
    base_fv = score(train, val, test, base_fc, "baseline")

    avail = {}
    for g, cols in extfeats.XF_GROUPS.items():
        c = [x for x in cols if x in merged.columns]
        if not c:
            print(f"  ({g}: no columns available)")
            continue
        avail[g] = c
        score(train, val, test, base_fc + c, f"+{g}", base_fv)

    # cross-divergence features
    for extra, tag in (
        (["xf_eth_btc_div"], "+eth_btc_div"),
        (["xf_cb_btc_div"], "+cb_btc_div"),
    ):
        c = [x for x in extra if x in merged.columns]
        if c:
            score(train, val, test, base_fc + c, tag, base_fv)

    # all external together
    allc = []
    for c in avail.values():
        allc += c
    for d in ("xf_eth_btc_div", "xf_cb_btc_div"):
        if d in merged.columns:
            allc.append(d)
    if allc:
        print()
        score(train, val, test, base_fc + allc, "+ALL_external", base_fv)


if __name__ == "__main__":
    main()
