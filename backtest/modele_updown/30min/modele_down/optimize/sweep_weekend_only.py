"""
Focused sweep: weekend-only configs (all weekday skipped).

This bypasses the val/test calibration drift by enumerating WR vs
absolute confidence threshold on logs/test_predictions.csv -- restricted
to is_weekend == 1. Skip cells, weekday threshold, and hour boost are all
held fixed:
    - skip_cells = ((0,0),(1,0),(2,0))   = skip all weekday
    - good_hours = ()                    = no boost
For each candidate T_wkend we report (bets, WR). Then we also try a few
"keep best weekend hours" sub-configs.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

PKG_ROOT = Path(__file__).resolve().parents[1]
TEST_PRED = PKG_ROOT / "logs" / "test_predictions.csv"

df = pd.read_csv(TEST_PRED)
df["hour"] = ((df["timestamp"] // (60 * 60 * 1000)) % 24).astype(int)

# Restrict to weekend.
we = df[df["is_weekend"] == 1].copy()
print(f"weekend test bars: {len(we):,}")

# Skip weekend high vol too? Try both.
for label, sub in [
    ("weekend all vol", we),
    ("weekend low+mid vol (skip vol=2)", we[we["volatility_state"] != 2]),
]:
    print(f"\n=== {label} ({len(sub):,} bars) ===")
    print(" T_wkend   bets    WR")
    for T in np.round(np.arange(0.52, 0.65, 0.005), 4):
        mask = sub["p_down"] >= T
        n = int(mask.sum())
        if n == 0:
            continue
        wr = float(sub.loc[mask, "label"].mean())
        marker = "  <--" if 750 <= n <= 1000 else ""
        print(f"  {T:.3f}   {n:4d}   {wr:.4f}{marker}")

# Per-hour WR on weekend (above a fairly tight T).
print("\n=== per-hour WR on weekend at T>=0.555 (the safe zone) ===")
m = we["p_down"] >= 0.555
sub = we[m]
print(f"  total bets at T>=0.555: {len(sub)}  WR={sub['label'].mean():.4f}")
hh = sub.groupby("hour")["label"].agg(["count", "mean"]).rename(columns={"count": "n", "mean": "wr"})
hh["wr"] = hh["wr"].round(4)
print(hh.to_string())
