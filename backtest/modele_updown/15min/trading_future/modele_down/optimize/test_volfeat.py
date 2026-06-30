"""Test: does a 30-day volatility feature help modele_down?

Four causal, scale-invariant forms of "30-day volatility" are tried as an
extra model feature, each evaluated by walk-forward over 6 yearly windows
at a fixed calibration (= opt_model.score) so the comparison isolates the
feature. Adopt only if it lifts the honest older-pooled WR clearly
(>~0.4 pt, the walk-forward SE).

  vol_norm_30d  : mean(|close_ret|,30d) / mean(|close_ret|,365d)   (the spec)
  vol30_raw     : mean(|close_ret|,30d)                            (raw level)
  vol30_z       : (mean30 - mean1y) / std(|close_ret|,365d)        (z-score)
  vol30_range   : mean(high_ret-low_ret,30d) / mean(...,365d)      (range-based)
"""
import numpy as np
import pandas as pd

import lib
import opt_model as om

DAY = 96   # 15m bars per day


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    av = pd.Series(np.abs(df["close_ret"].to_numpy(dtype=float)))
    m30 = av.rolling(30 * DAY,  min_periods=20 * DAY).mean()
    m1y = av.rolling(365 * DAY, min_periods=120 * DAY).mean()
    s1y = av.rolling(365 * DAY, min_periods=120 * DAY).std()
    df["vol_norm_30d"] = (m30 / m1y.replace(0, np.nan)).to_numpy()
    df["vol30_raw"]    = m30.to_numpy()
    df["vol30_z"]      = ((m30 - m1y) / s1y.replace(0, np.nan)).to_numpy()

    rng = pd.Series((df["high_ret"] - df["low_ret"]).to_numpy(dtype=float))
    r30 = rng.rolling(30 * DAY,  min_periods=20 * DAY).mean()
    r1y = rng.rolling(365 * DAY, min_periods=120 * DAY).mean()
    df["vol30_range"] = (r30 / r1y.replace(0, np.nan)).to_numpy()
    return df


def main() -> int:
    df = lib.load_master()
    fc = lib.feature_cols(df)
    df = add_features(df)

    print("=== walk-forward (6 windows, fixed calibration) ===")
    om.score(df, fc, None, 4000, "baseline")
    for feat in ("vol_norm_30d", "vol30_raw", "vol30_z", "vol30_range"):
        om.score(df, fc + [feat], None, 4000, "+" + feat)
    print("\nAdopt only if a variant lifts older WR clearly (>~0.4 pt).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
