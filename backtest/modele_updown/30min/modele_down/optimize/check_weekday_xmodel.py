"""
Check: do weekday bets ALSO get the +3pt WR lift from cross-model (p_up < 0.47),
or is the cross-model signal weekend-specific?

If weekday WR after cross-model jumps to 64-66 %, we can ADD weekday back
into the bet pool (with tight T_wkd) and reach 750+ bets at high WR.
"""
from pathlib import Path
import numpy as np
import pandas as pd

PKG_ROOT = Path(__file__).resolve().parents[1]
TEST_PRED = PKG_ROOT / "logs" / "test_predictions.csv"
UP_TEST_PRED = PKG_ROOT.parent / "modele_up" / "logs" / "test_predictions.csv"

BAD_HOURS = {3, 4, 11, 13, 22}

df = pd.read_csv(TEST_PRED)
up = pd.read_csv(UP_TEST_PRED, usecols=["timestamp", "p_up"])
df = df.merge(up, on="timestamp")
df["hour"] = ((df["timestamp"] // (15 * 60 * 1000)) % 96 // 4).astype(int)
df = df[~df["hour"].isin(BAD_HOURS)].copy()

print("--- WR per (is_weekend, p_up bin, p_down bin) ---")
for wknd in [0, 1]:
    sub = df[df["is_weekend"] == wknd]
    name = "WEEKEND" if wknd == 1 else "WEEKDAY"
    print(f"\n{name} (n={len(sub)})")
    # For each p_down quintile, split by p_up cutoff and compare WR
    for T_min in [0.51, 0.53, 0.55, 0.56, 0.57]:
        s = sub[sub["p_down"] >= T_min]
        if len(s) < 30:
            continue
        n_total = len(s)
        wr_all = s["label"].mean()
        n_pu_lo = (s["p_up"] < 0.47).sum()
        wr_pu_lo = s.loc[s["p_up"] < 0.47, "label"].mean() if n_pu_lo > 0 else np.nan
        n_pu_hi = (s["p_up"] >= 0.47).sum()
        wr_pu_hi = s.loc[s["p_up"] >= 0.47, "label"].mean() if n_pu_hi > 0 else np.nan
        print(f"  p_down>={T_min:.2f}: total {n_total:5d} WR={wr_all:.4f}  | "
              f"p_up<0.47 n={n_pu_lo:4d} WR={wr_pu_lo:.4f}  | "
              f"p_up>=0.47 n={n_pu_hi:4d} WR={wr_pu_hi:.4f}")
