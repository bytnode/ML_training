"""Extra helpers built on lib.py: hour/cyclic feature builders, honest
WR-at-volume search, and walk-forward evaluation."""
import numpy as np
import pandas as pd

import lib


# ============================================================
#  Time-derived feature columns (computable from `timestamp` ms)
# ============================================================
def add_time_features(df, cols):
    """Return a copy of df with the requested time-derived feature columns.
    All are pure functions of the ms `timestamp`, so no parquet patch needed."""
    df = df.copy()
    ts = df["timestamp"].to_numpy(dtype=np.int64)
    hour = ((ts // (15 * 60 * 1000)) % 96 // 4).astype(int)
    dt = pd.to_datetime(ts, unit="ms", utc=True)
    dow = dt.dayofweek.to_numpy().astype(int)         # 0=Mon .. 6=Sun
    minute = ((ts // (60 * 1000)) % 60).astype(int)
    qhour = (hour * 4 + minute // 15).astype(int)     # 0..95 quarter index

    avail = {
        "hour": hour,
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "hour_sin2": np.sin(4 * np.pi * hour / 24),
        "hour_cos2": np.cos(4 * np.pi * hour / 24),
        "hour_sin3": np.sin(6 * np.pi * hour / 24),
        "hour_cos3": np.cos(6 * np.pi * hour / 24),
        "dow": dow,
        "dow_sin": np.sin(2 * np.pi * dow / 7),
        "dow_cos": np.cos(2 * np.pi * dow / 7),
        "qhour": qhour,
        "qhour_sin": np.sin(2 * np.pi * qhour / 96),
        "qhour_cos": np.cos(2 * np.pi * qhour / 96),
        "vol_state": df["volatility_state"].to_numpy().astype(float),
    }
    for c in cols:
        if c not in avail:
            raise KeyError(f"unknown time feature {c!r}")
        df[c] = avail[c]
    return df


# ============================================================
#  Honest WR at a target volume
# ============================================================
def wr_at_volume(val_df, p_val, eval_df, p_eval, *, skip_fn, axis_fn,
                 boost_fn=None, target_n=2500, axis_frac_ratio=None,
                 lo=1e-4, hi=0.6, iters=40):
    """Binary-search a global vol_frac so the strategy takes ~target_n bets on
    eval_df. axis_frac_ratio: dict{axis_key: multiplier} to keep per-axis frac
    proportions (e.g. weekend frac = 4x weekday). Returns dict(n, wr, frac).
    Calibration always done on val; eval_df is where volume is targeted."""
    axes_all = np.unique(np.asarray(axis_fn(val_df)))
    if axis_frac_ratio is None:
        axis_frac_ratio = {k: 1.0 for k in axes_all}

    def run(g):
        vf = {k: min(0.999, g * axis_frac_ratio.get(k, 1.0)) for k in axes_all}
        thr = lib.calibrate(val_df, p_val, skip_fn=skip_fn, axis_fn=axis_fn,
                            vol_frac=vf)
        return lib.evaluate(eval_df, p_eval, thr, skip_fn=skip_fn,
                            axis_fn=axis_fn, boost_fn=boost_fn)

    best = None
    for _ in range(iters):
        mid = (lo + hi) / 2
        res = run(mid)
        best = res, mid
        if res["n"] < target_n:
            lo = mid
        else:
            hi = mid
        if abs(res["n"] - target_n) <= 2:
            break
    res, g = best
    return dict(n=res["n"], wr=res["wr"], frac=g, take=res["take"])


def frontier_at(p, y, allowed, n):
    """Test/val-fitted WR ceiling at exactly n bets among `allowed` rows."""
    r = lib.test_frontier(p, y, allowed, [n])
    return r[0]["wr"] if r else float("nan")


# ============================================================
#  Bucket stability printer (val vs eval side by side)
# ============================================================
def bucket_stability(val_df, p_val, eval_df, p_eval, key_fn, names,
                     frac=0.10, title=""):
    """For each bucket: top-frac p_down WR on val vs eval. Reveals whether a
    bucket's edge is real (consistent) or test-window noise."""
    print(f"\n  -- bucket stability: {title} (top {frac*100:.0f}% p_down) --")
    print(f"  {'bucket':<12} {'val_n':>7} {'val_wr':>8}   {'eval_n':>7} {'eval_wr':>8}   diff")
    yv = val_df["label"].to_numpy().astype(int)
    ye = eval_df["label"].to_numpy().astype(int)
    kv = np.asarray(key_fn(val_df))
    ke = np.asarray(key_fn(eval_df))
    for name, val in names:
        mv, me = kv == val, ke == val
        if mv.sum() == 0 or me.sum() == 0:
            continue
        kv_n = max(1, int(mv.sum() * frac))
        ke_n = max(1, int(me.sum() * frac))
        ov = np.argsort(-p_val[mv])[:kv_n]
        oe = np.argsort(-p_eval[me])[:ke_n]
        wv = yv[mv][ov].mean()
        we = ye[me][oe].mean()
        print(f"  {name:<12} {kv_n:>7} {wv*100:>7.2f}%   "
              f"{ke_n:>7} {we*100:>7.2f}%   {(we-wv)*100:>+6.2f}")
