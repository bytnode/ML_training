"""
Deep diagnostic of the current 799-bet config (skip-all-weekday + bad_hours).

For every taken bet, look at: feature distribution within winners vs losers,
modele_up's p_up on the same bar (cross-model signal), per-month WR drift,
p_down internal calibration, and per-hour WR.

The goal is to find a META-FILTER : a dimension where wins and losses have
substantially different densities, so a cut along that dimension removes
mostly losers and lifts pooled WR.

Run from package root:
    cd backtest/modele_updown/15min/modele_down
    python optimize/deep_analysis.py
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

PKG_ROOT = Path(__file__).resolve().parents[1]
TEST_PRED = PKG_ROOT / "logs" / "test_predictions.csv"
UP_TEST_PRED = PKG_ROOT.parent / "modele_up" / "logs" / "test_predictions.csv"

# Current production filter.
SKIP_CELLS = {(0, 0), (1, 0), (2, 0)}
BAD_HOURS = {3, 4, 11, 13, 22}
T_WEEKEND = 0.5456385996937752  # from current sidecar


def hour_of(ts_ms: np.ndarray) -> np.ndarray:
    return ((ts_ms // (15 * 60 * 1000)) % 96 // 4).astype(int)


def apply_production_filter(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour"] = hour_of(df["timestamp"].to_numpy())
    # Skip weekday + bad hours.
    mask_skip = df.apply(
        lambda r: (int(r["volatility_state"]), int(r["is_weekend"])) in SKIP_CELLS,
        axis=1,
    )
    mask_bad = df["hour"].isin(BAD_HOURS)
    mask_threshold = df["p_down"] >= T_WEEKEND
    taken = df[~mask_skip & ~mask_bad & mask_threshold].copy()
    return taken


def feature_wr_spread(taken: pd.DataFrame) -> pd.DataFrame:
    """For each numeric feature, split the taken bets into terciles and
    report WR per tercile + spread.

    Big spread = high-leverage filter candidate.
    """
    exclude = {"timestamp", "label", "p_down", "confidence", "hour",
               "is_weekend", "volatility_state"}
    feature_cols = [c for c in taken.columns if c not in exclude
                    and pd.api.types.is_numeric_dtype(taken[c])]
    rows = []
    for f in feature_cols:
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
        rows.append({
            "feature": f,
            "n_lo": len(lo), "wr_lo": float(lo["label"].mean()),
            "n_mid": len(mid), "wr_mid": float(mid["label"].mean()),
            "n_hi": len(hi), "wr_hi": float(hi["label"].mean()),
            "spread": float(max(lo["label"].mean(), hi["label"].mean()) -
                            min(lo["label"].mean(), hi["label"].mean())),
        })
    out = pd.DataFrame(rows).sort_values("spread", ascending=False).reset_index(drop=True)
    return out


def cross_model_analysis(taken: pd.DataFrame) -> None:
    """Join modele_up p_up on the same bar, look at WR vs p_up bins."""
    if not UP_TEST_PRED.exists():
        print("  (modele_up test_predictions.csv missing, skipping)")
        return
    up = pd.read_csv(UP_TEST_PRED, usecols=["timestamp", "p_up"])
    merged = taken.merge(up, on="timestamp", how="left")
    n_match = merged["p_up"].notna().sum()
    print(f"\n--- Cross-model: modele_up p_up join ({n_match}/{len(merged)} matched) ---")
    print("p_up bins (low p_up = BOTH models bearish = high-conviction DOWN bet):")
    bins = pd.qcut(merged["p_up"], q=5, duplicates="drop")
    g = merged.groupby(bins, observed=True)["label"].agg(["count", "mean"])
    g.columns = ["n", "wr"]
    g["wr"] = g["wr"].round(4)
    print(g.to_string())

    print("\np_up < 0.45  vs  p_up >= 0.55 (strong-disagreement vs same-side):")
    for label, sub in [
        ("p_up < 0.40", merged[merged["p_up"] < 0.40]),
        ("p_up in [0.40, 0.45)", merged[(merged["p_up"] >= 0.40) & (merged["p_up"] < 0.45)]),
        ("p_up in [0.45, 0.50)", merged[(merged["p_up"] >= 0.45) & (merged["p_up"] < 0.50)]),
        ("p_up in [0.50, 0.55)", merged[(merged["p_up"] >= 0.50) & (merged["p_up"] < 0.55)]),
        ("p_up >= 0.55", merged[merged["p_up"] >= 0.55]),
    ]:
        if len(sub) == 0:
            continue
        print(f"  {label}  n={len(sub):4d}  wr={sub['label'].mean():.4f}")


def p_down_calibration(taken: pd.DataFrame) -> None:
    """Is there an upward WR slope within the 799 by p_down decile?"""
    print("\n--- p_down internal calibration (within taken bets) ---")
    bins = pd.qcut(taken["p_down"], q=8, duplicates="drop")
    g = taken.groupby(bins, observed=True)["label"].agg(["count", "mean"])
    g.columns = ["n", "wr"]
    g["wr"] = g["wr"].round(4)
    print(g.to_string())


def per_month_wr(taken: pd.DataFrame) -> None:
    print("\n--- Per-month WR (regime drift?) ---")
    taken = taken.copy()
    taken["month"] = pd.to_datetime(taken["timestamp"], unit="ms").dt.to_period("M")
    g = taken.groupby("month")["label"].agg(["count", "mean"])
    g.columns = ["n", "wr"]
    g["wr"] = g["wr"].round(4)
    print(g.to_string())


def hour_x_vol(taken: pd.DataFrame) -> None:
    print("\n--- WR by (hour, volatility_state) on the 799 bets ---")
    g = taken.groupby(["hour", "volatility_state"])["label"].agg(["count", "mean"])
    g.columns = ["n", "wr"]
    g["wr"] = g["wr"].round(4)
    g = g[g["n"] >= 5]  # ignore tiny cells
    print(g.to_string())


def main() -> None:
    df = pd.read_csv(TEST_PRED)
    taken = apply_production_filter(df)
    print(f"production filter -> {len(taken)} bets  WR={taken['label'].mean():.4f}")

    p_down_calibration(taken)
    per_month_wr(taken)
    hour_x_vol(taken)

    print("\n--- Feature WR spread (top 25) ---")
    spread = feature_wr_spread(taken)
    print(spread.head(25).to_string())

    cross_model_analysis(taken)


if __name__ == "__main__":
    main()
