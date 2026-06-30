"""Experiment 05: calibration / decision-layer redesign.

Tests honestly (nested: calibrate on val_cal=first 70% of val, evaluate on
val_hold=last 30% — never touches test) whether finer bucket structure or
per-bucket isotonic recalibration beats the production weekday/weekend split.
Also reports test WR at 2500 as confirmation.
"""
import numpy as np
from sklearn.isotonic import IsotonicRegression

import lib
import lib2


def bucket(df, kind):
    we = df["is_weekend"].to_numpy().astype(int)
    vol = df["volatility_state"].to_numpy().astype(int)
    if kind == "none":
        return np.zeros(len(df), int)
    if kind == "weekend":
        return we
    if kind == "vol":
        return vol
    if kind == "wkvol":
        return we * 3 + vol
    if kind == "dow":
        ts = df["timestamp"].to_numpy(np.int64)
        import pandas as pd
        return pd.to_datetime(ts, unit="ms", utc=True).dayofweek.to_numpy().astype(int)
    raise ValueError(kind)


def isotonic_score(p_cal, y_cal, b_cal, p_eval, b_eval, min_n=300):
    """Per-bucket isotonic recalibration -> comparable cross-bucket scores."""
    score = np.full(len(p_eval), np.nan)
    for b in np.unique(b_cal):
        mc = b_cal == b
        if mc.sum() < min_n:
            continue
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(p_cal[mc], y_cal[mc])
        me = b_eval == b
        score[me] = ir.predict(p_eval[me])
    nan = np.isnan(score)
    if nan.any():
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(p_cal, y_cal)
        score[nan] = ir.predict(p_eval[nan])
    return score


def wr_at(score, y, allowed, n):
    s = score[allowed]
    yy = y[allowed]
    order = np.argsort(-s)[:n]
    return float(yy[order].mean()), len(order)


def main():
    df = lib.load_master()
    fc = lib.feature_cols(df)
    train, val, test = lib.temporal_split(df)
    model = lib.train_model(train, val, fc)
    p_val = lib.predict(model, val, fc)
    p_test = lib.predict(model, test, fc)
    yv = val["label"].to_numpy().astype(int)
    yt = test["label"].to_numpy().astype(int)

    # nested val split (time-ordered)
    cut = int(len(val) * 0.70)
    vcal, vhold = val.iloc[:cut], val.iloc[cut:]
    p_vcal, p_vhold = p_val[:cut], p_val[cut:]
    y_vcal = yv[:cut]
    y_vhold = yv[cut:]

    # volume targets: 2500 test ~ 7.2% -> val_hold equiv
    n_hold = int(len(vhold) * 0.072)
    print(f"val_cal={len(vcal):,}  val_hold={len(vhold):,} (target {n_hold})  "
          f"test={len(test):,} (target 2500)\n")

    print(f"{'method':<26} {'skip':<7} {'valhold_wr':>11} {'test_wr@2500':>13}")
    print("-" * 62)

    vol_vhold = vhold["volatility_state"].to_numpy().astype(int)
    vol_test = test["volatility_state"].to_numpy().astype(int)

    for skip_name, skip_vh, skip_te in (
        ("none", np.zeros(len(vhold), bool), np.zeros(len(test), bool)),
        ("vol2", vol_vhold == 2, vol_test == 2),
        ("vol0", vol_vhold == 0, vol_test == 0),
    ):
        allowed_vh = ~skip_vh
        allowed_te = ~skip_te

        # raw p_down ranking (global frontier)
        wh, nh = wr_at(p_vhold, y_vhold, allowed_vh, n_hold)
        wt, nt = wr_at(p_test, yt, allowed_te, 2500)
        print(f"  {'raw p_down':<24} {skip_name:<7} {wh*100:>9.2f}%  {wt*100:>11.2f}%")

        # isotonic per bucket
        for bk in ("weekend", "vol", "wkvol", "dow"):
            sc_vh = isotonic_score(p_vcal, y_vcal, bucket(vcal, bk),
                                   p_vhold, bucket(vhold, bk))
            sc_te = isotonic_score(p_val, yv, bucket(val, bk),
                                   p_test, bucket(test, bk))
            wh, nh = wr_at(sc_vh, y_vhold, allowed_vh, n_hold)
            wt, nt = wr_at(sc_te, yt, allowed_te, 2500)
            print(f"  {'iso['+bk+']':<24} {skip_name:<7} "
                  f"{wh*100:>9.2f}%  {wt*100:>11.2f}%")
        print()


if __name__ == "__main__":
    main()
