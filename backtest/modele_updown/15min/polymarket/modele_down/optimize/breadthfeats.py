"""Altcoin-breadth features: aggregate concurrent moves across many coins
(ETH + alts) into robust regime signals. Breadth is less noisy than any
single pair. All concurrent bars close at decision time T -> no lookahead.
Columns prefixed `bf_`.
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent / "data"
MS15 = 15 * 60 * 1000


def _align_exact(base_ts, src_ts, vals):
    idx = np.searchsorted(src_ts, base_ts)
    out = np.full(len(base_ts), np.nan)
    clip = np.clip(idx, 0, len(src_ts) - 1)
    valid = (idx < len(src_ts)) & (src_ts[clip] == base_ts)
    out[valid] = vals[idx[valid]]
    return out


def build_breadth(base_ts, verbose=True):
    """DataFrame keyed by timestamp_ms (= base_ts) with bf_ breadth columns."""
    base_ts = np.asarray(base_ts, dtype=np.int64)
    files = sorted(DATA.glob("*_15m.parquet"))
    rets, rets4, rets16 = [], [], []
    names = []
    for f in files:
        df = pd.read_parquet(f).sort_values("timestamp_ms").reset_index(drop=True)
        c = df["close"].astype(float)
        cr = (c / c.shift(1) - 1).to_numpy()
        cr4 = (c / c.shift(4) - 1).to_numpy()
        cr16 = (c / c.shift(16) - 1).to_numpy()
        sts = df["timestamp_ms"].to_numpy(np.int64)
        rets.append(_align_exact(base_ts, sts, cr))
        rets4.append(_align_exact(base_ts, sts, cr4))
        rets16.append(_align_exact(base_ts, sts, cr16))
        names.append(f.stem.replace("_15m", ""))

    if not rets:
        return pd.DataFrame({"timestamp_ms": base_ts})

    R = np.column_stack(rets)        # [n_bars, n_coins]
    R4 = np.column_stack(rets4)
    R16 = np.column_stack(rets16)

    def _stats(M, sfx):
        with np.errstate(invalid="ignore"):
            up = np.nanmean((M > 0).astype(float), axis=1)
            mean = np.nanmean(M, axis=1)
            med = np.nanmedian(M, axis=1)
            disp = np.nanstd(M, axis=1)
            mn = np.nanmin(M, axis=1)
            mx = np.nanmax(M, axis=1)
        return {f"bf_breadth{sfx}": up, f"bf_mean{sfx}": mean,
                f"bf_med{sfx}": med, f"bf_disp{sfx}": disp,
                f"bf_min{sfx}": mn, f"bf_max{sfx}": mx}

    cols = {"timestamp_ms": base_ts}
    cols.update(_stats(R, ""))
    cols.update(_stats(R4, "4"))
    cols.update(_stats(R16, "16"))
    # n_coins available at each bar (regime: more coins = more recent data)
    cols["bf_n"] = np.sum(~np.isnan(R), axis=1).astype(float)

    res = pd.DataFrame(cols)
    if verbose:
        print(f"build_breadth: {len(res):,} rows x {len(res.columns)-1} cols "
              f"from {len(names)} coins: {names}")
    return res


BREADTH_GROUPS = {
    "b_now": ["bf_breadth", "bf_mean", "bf_med", "bf_disp", "bf_min", "bf_max"],
    "b_4": ["bf_breadth4", "bf_mean4", "bf_med4", "bf_disp4"],
    "b_16": ["bf_breadth16", "bf_mean16", "bf_med16"],
}
