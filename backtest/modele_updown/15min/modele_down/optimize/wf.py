"""Walk-forward pooled evaluation — the trustworthy metric.

WR at 2500 bets on one window has ~1pp sampling noise. Pooling ~6 windows
(~15k bets) cuts that to ~0.4pp, enough to compare configs reliably.

Each window: train on ALL prior data, calibrate the weekend-aware quantile
strategy on val, take ~target_n bets on test. Pool taken bets across windows.
"""
import numpy as np

import lib
import lib2

MS_DAY = 86_400_000


def default_windows(merged, k=6, test_days=365):
    ts_max = int(merged["timestamp"].max())
    return [ts_max - i * test_days * MS_DAY for i in range(k)]


def walk_forward(merged, feat_cols, *, window_ends=None, test_days=365,
                 hp=None, skip_vol2=True, target_n=2500, we_ratio=4.0,
                 seeds=(42,), label="", verbose=True):
    if window_ends is None:
        window_ends = default_windows(merged, k=6, test_days=test_days)

    skip_fn = lib.skip_vol2 if skip_vol2 else (lambda d: np.zeros(len(d), bool))
    pooled = []
    rows = []
    for end_ts in window_ends:
        train, val, test = lib.temporal_split(merged, train_days=None,
                                              test_days=test_days, end_ts=end_ts)
        if len(test) < 5000 or len(val) < 3000:
            continue
        p_val = np.zeros(len(val))
        p_test = np.zeros(len(test))
        for s in seeds:
            h = dict(hp or {})
            h["random_state"] = s
            m = lib.train_model(train, val, feat_cols, hp=h)
            p_val += lib.predict(m, val, feat_cols)
            p_test += lib.predict(m, test, feat_cols)
        p_val /= len(seeds)
        p_test /= len(seeds)

        r = lib2.wr_at_volume(val, p_val, test, p_test, skip_fn=skip_fn,
                              axis_fn=lib.axis_weekend, target_n=target_n,
                              axis_frac_ratio={0: 1.0, 1: we_ratio})
        yt = test["label"].to_numpy().astype(int)
        taken = yt[r["take"]]
        pooled.append(taken)
        import pandas as pd
        wend = pd.to_datetime(end_ts, unit="ms").date()
        rows.append(dict(end=str(wend), n=r["n"], wr=r["wr"]))

    allp = np.concatenate(pooled) if pooled else np.array([])
    pooled_n = len(allp)
    pooled_wr = float(allp.mean()) if pooled_n else float("nan")
    worst = min((x["wr"] for x in rows), default=float("nan"))

    if verbose:
        wstr = "  ".join(f"{x['end']}:{x['n']}@{x['wr']*100:.1f}" for x in rows)
        print(f"  {label:<30} pooled n={pooled_n:>6} WR={pooled_wr*100:6.2f}%  "
              f"worst={worst*100:5.2f}%")
        print(f"       windows: {wstr}")
    return dict(pooled_n=pooled_n, pooled_wr=pooled_wr, worst=worst, rows=rows)
