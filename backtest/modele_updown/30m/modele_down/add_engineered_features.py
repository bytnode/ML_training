"""
Add bearish-engineered features to the existing parquet without rebuilding
the SQLite chain.

All new features are computed from columns already present in
DB/btc_multi_tf_1h.parquet (no raw OHLCV needed). The script reads the
parquet, computes the new columns, and writes it back IN PLACE.

Idempotent: existing columns are overwritten if recomputed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
PARQUET_PATH = ROOT / "DB" / "btc_multi_tf_1h.parquet"


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add ~15 bearish-engineered features. All scale-invariant."""
    n = len(df)
    close_ret = df["close_ret"].to_numpy(dtype=float)
    upper_w = df["upper_wick"].to_numpy(dtype=float)
    lower_w = df["lower_wick"].to_numpy(dtype=float)
    body_close = df["body_ratio_close"].to_numpy(dtype=float)
    volume_ratio = df["volume_ratio"].to_numpy(dtype=float)
    rsi_14 = df["rsi_14"].to_numpy(dtype=float)
    stoch = df["stoch_k_14"].to_numpy(dtype=float)
    bb_pos = df["bb_position"].to_numpy(dtype=float)
    macd = df["macd"].to_numpy(dtype=float)
    macd_hist = df["macd_hist"].to_numpy(dtype=float)
    cr_4h = df["close_ret_4h"].to_numpy(dtype=float)
    cr_1d = df["high_ret_1d"].to_numpy(dtype=float) + df["low_ret_1d"].to_numpy(dtype=float)  # proxy
    cr_1d_alt = df["open_ret_1d"].to_numpy(dtype=float)  # available
    obv = df["obv_pressure"].to_numpy(dtype=float)
    obv_4h = df["obv_pressure_4h"].to_numpy(dtype=float)
    rv_20 = df["rv_20"].to_numpy(dtype=float)
    rv_100 = df["rv_100"].to_numpy(dtype=float)
    dist_high_20 = df["dist_high_20"].to_numpy(dtype=float)
    dist_high_50 = df["dist_high_50"].to_numpy(dtype=float)
    dist_low_20 = df["dist_low_20"].to_numpy(dtype=float)
    price_in_range_20 = df["price_in_range_20"].to_numpy(dtype=float)
    dist_ema_20 = df["dist_ema_20"].to_numpy(dtype=float)
    dist_ema_100 = df["dist_ema_100"].to_numpy(dtype=float)

    # ---- 1. Wick asymmetry signed (positive = upper wick dominant = bearish rejection)
    df["wick_signed"] = (upper_w - lower_w)

    # ---- 2. Bearish rejection composite: upper wick large + body small + close near low
    # = top-shadow pattern (shooting star, dragonfly inverted)
    body_safe = np.where(body_close > 0.01, body_close, 0.01)
    df["bear_reject"] = (upper_w / body_safe).clip(0, 5) / 5.0  # rescaled [0,1]

    # ---- 3. Consecutive same-color streak (signed). Positive = consecutive reds, neg = consecutive greens.
    sign = np.where(close_ret > 0, 1, np.where(close_ret < 0, -1, 0))
    streak = np.zeros(n, dtype=float)
    cur = 0
    for i in range(n):
        if sign[i] == 0:
            cur = 0
        elif i == 0:
            cur = sign[i]
        elif sign[i] == sign[i - 1]:
            cur = cur + sign[i]
        else:
            cur = sign[i]
        streak[i] = cur
    df["streak_signed"] = np.clip(streak, -8, 8) / 8.0  # rescaled [-1,1]

    # ---- 4. MTF bearish alignment: close_ret 4h + 1d momentum scaled
    df["mtf_align"] = np.tanh((cr_4h + cr_1d_alt) * 50.0)  # squashed [-1,1]

    # ---- 5. RSI overbought score: only triggers when RSI > 0.2 (normalized [-1,1])
    df["rsi_ob_score"] = np.clip(rsi_14 - 0.2, 0.0, 0.8) / 0.8

    # ---- 6. RSI slope (3-bar diff)
    rsi_3 = np.full(n, np.nan, dtype=float)
    if n > 3:
        rsi_3[3:] = rsi_14[3:] - rsi_14[:-3]
    df["rsi_slope_3"] = rsi_3

    # ---- 7. Stoch_d (smoothed stoch_k)
    s = pd.Series(stoch)
    df["stoch_d_14"] = s.rolling(3, min_periods=3).mean().to_numpy()

    # ---- 8. Volume-price divergence: positive when price up but volume down
    # vol_ratio is around 1.0 at average; > 1 = high vol, < 1 = low vol
    vol_dev = volume_ratio - 1.0
    df["vp_divergence"] = -1.0 * np.sign(close_ret) * np.sign(vol_dev)
    # = +1 if price up + low vol  (bear distribution)  or  price down + high vol  (capitulation)
    # = -1 if price up + high vol  (bullish confirm)  or  price down + low vol  (no follow)

    # ---- 9. Day of week (cyclic)
    ts = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
    dow = ts.dt.dayofweek.to_numpy(dtype=float)  # 0=Monday, 6=Sunday
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)

    # ---- 10. BB upper-band proximity (positive when near/above upper band, BEARISH)
    df["bb_upper_pressure"] = np.clip(bb_pos - 0.7, 0.0, 0.5) / 0.5

    # ---- 11. EMA over-extension (price far above EMA20 + EMA100)
    # dist_ema_X = (close - ema_X)/close. Positive = above EMA.
    df["ema_overext"] = np.clip(dist_ema_20 + dist_ema_100, 0.0, 0.1) / 0.1

    # ---- 12. Realized vol expansion (rv_20 > rv_100 = vol expanding)
    rv100_safe = np.where(rv_100 > 1e-9, rv_100, np.nan)
    df["vol_expand"] = np.tanh((rv_20 / rv100_safe - 1.0) * 2.0)

    # ---- 13. OBV momentum + 4h alignment (negative obv = bearish)
    df["obv_align"] = -(obv + obv_4h) / 2.0  # positive = both selling
    df["obv_align"] = np.tanh(df["obv_align"] * 3.0)

    # ---- 14. MACD bearish cross intensity (macd_hist < 0 + macd just turned down)
    macd_prev = np.concatenate([[np.nan], macd[:-1]])
    df["macd_bear_turn"] = np.where((macd_hist < 0) & (macd < macd_prev), -macd_hist * 100, 0.0)
    df["macd_bear_turn"] = np.clip(df["macd_bear_turn"], 0.0, 1.0)

    # ---- 15. Range-high failure: price reached recent high but body is mostly red/small
    near_high = (dist_high_20 > -0.005).astype(float)
    df["range_fail_top"] = near_high * (upper_w - body_close).clip(-1, 1)

    # ---- 16. Combined "tradable top" composite
    df["bear_setup"] = (
        df["rsi_ob_score"] * 0.25 +
        df["bb_upper_pressure"] * 0.25 +
        df["bear_reject"] * 0.15 +
        df["streak_signed"].clip(0, None) * 0.15 +  # only count green streaks
        df["wick_signed"] * 0.10 +
        df["obv_align"].clip(0, None) * 0.10
    )

    return df


def main():
    if not PARQUET_PATH.exists():
        raise FileNotFoundError(f"{PARQUET_PATH} not found. Run data.py first.")
    print(f"Loading {PARQUET_PATH}...")
    df = pd.read_parquet(PARQUET_PATH)
    print(f"  Before: {len(df):,} rows x {len(df.columns)} cols")

    df = add_features(df)
    print(f"  After:  {len(df):,} rows x {len(df.columns)} cols")

    new_cols = [c for c in df.columns if c in (
        "wick_signed", "bear_reject", "streak_signed", "mtf_align",
        "rsi_ob_score", "rsi_slope_3", "stoch_d_14",
        "vp_divergence", "dow_sin", "dow_cos", "bb_upper_pressure",
        "ema_overext", "vol_expand", "obv_align", "macd_bear_turn",
        "range_fail_top", "bear_setup",
    )]
    print(f"\n  Added/refreshed columns: {len(new_cols)}")
    for c in new_cols:
        v = df[c].dropna()
        print(f"    {c:<22} min={v.min():>8.4f} max={v.max():>7.4f} "
              f"mean={v.mean():>7.4f} std={v.std():>7.4f}")

    df.to_parquet(PARQUET_PATH, index=False)
    print(f"\nWrote back {PARQUET_PATH}")


if __name__ == "__main__":
    main()
