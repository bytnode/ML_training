"""Experiment 08: trustworthy walk-forward comparison (6 windows, ~15k pooled
bets, SE ~0.4pp). Settles: does ETH help? skip vol2 or not? depth? seeds?
"""
import numpy as np

import lib
import ethfeats
import wf

ETH_COLS = open(lib.PKG / "eth_best_cols.txt").read().split()


def build():
    df = lib.load_master()
    base_fc = lib.feature_cols(df)
    eth = ethfeats.build_eth(df["timestamp"].to_numpy(), verbose=False)
    merged = df.merge(eth, left_on="timestamp", right_on="timestamp_ms",
                      how="left").drop(columns=["timestamp_ms"])
    return merged, base_fc


def main():
    merged, base_fc = build()
    eth_fc = base_fc + [c for c in ETH_COLS if c in merged.columns]
    print(f"base={len(base_fc)} feats, +ETH={len(eth_fc)} feats\n")

    print("=== baseline vs +ETH (skip vol2, depth5, 1 seed) ===")
    wf.walk_forward(merged, base_fc, label="baseline")
    wf.walk_forward(merged, eth_fc, label="+ETH")

    print("\n=== skip vol2 vs include vol2 (+ETH) ===")
    wf.walk_forward(merged, eth_fc, skip_vol2=True, label="+ETH skip vol2")
    wf.walk_forward(merged, eth_fc, skip_vol2=False, label="+ETH incl vol2")

    print("\n=== depth (+ETH, include vol2) ===")
    for d in (5, 6, 7):
        wf.walk_forward(merged, eth_fc, skip_vol2=False, hp={"max_depth": d},
                        label=f"+ETH depth{d}")

    print("\n=== weekend ratio (+ETH, depth6, include vol2) ===")
    for wr_ratio in (1.0, 2.0, 4.0, 6.0):
        wf.walk_forward(merged, eth_fc, skip_vol2=False, hp={"max_depth": 6},
                        we_ratio=wr_ratio, label=f"+ETH we_ratio={wr_ratio}")

    print("\n=== multi-seed ensemble (+ETH, depth6, include vol2) ===")
    wf.walk_forward(merged, eth_fc, skip_vol2=False, hp={"max_depth": 6},
                    seeds=(42,), label="1 seed")
    wf.walk_forward(merged, eth_fc, skip_vol2=False, hp={"max_depth": 6},
                    seeds=(42, 1, 2, 3, 4), label="5 seeds")


if __name__ == "__main__":
    main()
