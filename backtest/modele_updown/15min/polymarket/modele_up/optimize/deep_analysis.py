"""
Deep diagnostic of modele_up's current bet pool.

Same shape as modele_down/optimize/deep_analysis.py : per-bucket WR
on logs/test_predictions.csv, per-hour, p_up internal calibration,
feature WR spread, and cross-model join with modele_down's p_down.

Run from the package root :
    cd backtest/modele_updown/15min/modele_up
    python optimize/deep_analysis.py
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

PKG_ROOT = Path(__file__).resolve().parents[1]
TEST_PRED = PKG_ROOT / "logs" / "test_predictions.csv"
DOWN_TEST_PRED = PKG_ROOT.parent / "modele_down" / "logs" / "test_predictions.csv"

# Current modele_up production filter (from sidecar 2026-05-24T00:30:59Z).
SKIP_CELLS = {(2, 0), (2, 1)}            # ASYM_SKIP_CELLS = high-vol both days
T_WEEKDAY = 0.5963
T_WEEKEND = 0.5711
HOUR_BOOST_HOURS = {20, 21, 22, 23}
HOUR_BOOST_DELTA = 0.02


def hour_of(ts_ms: np.ndarray) -> np.ndarray:
    return ((ts_ms // (15 * 60 * 1000)) % 96 // 4).astype(int)


def apply_production_filter(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour"] = hour_of(df["timestamp"].to_numpy())
    mask_skip = df.apply(
        lambda r: (int(r["volatility_state"]), int(r["is_weekend"])) in SKIP_CELLS,
        axis=1,
    )
    T = np.where(df["is_weekend"] == 1, T_WEEKEND, T_WEEKDAY).astype(float)
    boost = df["hour"].isin(HOUR_BOOST_HOURS).to_numpy()
    T = np.where(boost & (~mask_skip.to_numpy()), T - HOUR_BOOST_DELTA, T)
    take = (~mask_skip.to_numpy()) & (df["p_up"].to_numpy() >= T)
    return df[take].copy()


def per_bucket_wr(taken: pd.DataFrame) -> None:
    print(f"\nTaken pool: {len(taken)} bets, WR={taken['label'].mean():.4f}\n")
    print("--- WR by (volatility_state x is_weekend) bucket ---")
    g = taken.groupby(["volatility_state", "is_weekend"])["label"].agg(["count", "mean"])
    g.columns = ["n", "wr"]
    g["wr"] = g["wr"].round(4)
    print(g.to_string())

    print("\n--- WR by hour (UTC) ---")
    g = taken.groupby("hour")["label"].agg(["count", "mean"])
    g.columns = ["n", "wr"]
    g["wr"] = g["wr"].round(4)
    print(g.to_string())

    print("\n--- WR by hour x is_weekend ---")
    g = taken.groupby(["is_weekend", "hour"])["label"].agg(["count", "mean"])
    g.columns = ["n", "wr"]
    g["wr"] = g["wr"].round(4)
    print(g.to_string())


def p_up_calibration(taken: pd.DataFrame) -> None:
    print("\n--- p_up internal calibration (within taken) ---")
    bins = pd.qcut(taken["p_up"], q=8, duplicates="drop")
    g = taken.groupby(bins, observed=True)["label"].agg(["count", "mean"])
    g.columns = ["n", "wr"]
    g["wr"] = g["wr"].round(4)
    print(g.to_string())


def per_month_wr(taken: pd.DataFrame) -> None:
    print("\n--- Per-month WR ---")
    taken = taken.copy()
    taken["month"] = pd.to_datetime(taken["timestamp"], unit="ms").dt.to_period("M")
    g = taken.groupby("month")["label"].agg(["count", "mean"])
    g.columns = ["n", "wr"]
    g["wr"] = g["wr"].round(4)
    print(g.to_string())


def feature_wr_spread(taken: pd.DataFrame) -> pd.DataFrame:
    exclude = {"timestamp", "label", "p_up", "confidence", "hour",
               "is_weekend", "volatility_state"}
    feat_cols = [c for c in taken.columns if c not in exclude
                 and pd.api.types.is_numeric_dtype(taken[c])]
    rows = []
    for f in feat_cols:
        vals = taken[f].astype(float)
        if vals.nunique() < 4:
            continue
        try:
            q = np.quantile(vals, [1/3, 2/3])
        except Exception:
            continue
        lo = taken[vals <= q[0]]
        mid = taken[(vals > q[0]) & (vals <= q[1])]
        hi = taken[vals > q[1]]
        if len(lo) < 30 or len(hi) < 30:
            continue
        wrs = (lo["label"].mean(), mid["label"].mean(), hi["label"].mean())
        rows.append({
            "feature": f,
            "n_lo": len(lo), "wr_lo": wrs[0],
            "n_mid": len(mid), "wr_mid": wrs[1],
            "n_hi": len(hi), "wr_hi": wrs[2],
            "spread": float(max(wrs) - min(wrs)),
        })
    return pd.DataFrame(rows).sort_values("spread", ascending=False).reset_index(drop=True)


def cross_model_analysis(taken: pd.DataFrame) -> None:
    if not DOWN_TEST_PRED.exists():
        print("  (modele_down test_predictions.csv missing, skipping)")
        return
    dn = pd.read_csv(DOWN_TEST_PRED, usecols=["timestamp", "p_down"])
    merged = taken.merge(dn, on="timestamp", how="left")
    n_match = merged["p_down"].notna().sum()
    print(f"\n--- Cross-model: modele_down p_down join ({n_match}/{len(merged)}) ---")
    bins = pd.qcut(merged["p_down"], q=5, duplicates="drop")
    g = merged.groupby(bins, observed=True)["label"].agg(["count", "mean"])
    g.columns = ["n", "wr"]
    g["wr"] = g["wr"].round(4)
    print(g.to_string())

    print("\n--- p_down bins (low p_down = modele_down bearish UP signal = high-conviction UP) ---")
    for lo, hi in [(0.30, 0.40), (0.40, 0.45), (0.45, 0.47), (0.47, 0.50),
                   (0.50, 0.53), (0.53, 1.00)]:
        sub = merged[(merged["p_down"] >= lo) & (merged["p_down"] < hi)]
        if len(sub) == 0:
            continue
        print(f"  p_down in [{lo:.2f}, {hi:.2f})  n={len(sub):4d}  wr={sub['label'].mean():.4f}")


def main() -> None:
    df = pd.read_csv(TEST_PRED)
    taken = apply_production_filter(df)
    per_bucket_wr(taken)
    p_up_calibration(taken)
    per_month_wr(taken)
    print("\n--- Feature WR spread (top 20) ---")
    print(feature_wr_spread(taken).head(20).to_string())
    cross_model_analysis(taken)


if __name__ == "__main__":
    main()
