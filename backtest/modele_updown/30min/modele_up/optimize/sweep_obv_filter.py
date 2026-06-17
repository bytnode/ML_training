"""Sweep over (T_weekend, obv_pressure_5m min) to find configs that land
at >800 bets AND >65 % WR. The latest test_predictions.csv has p_up_xmodel
already integrated (the model trained with it), so this sweep starts from
that improved p_down distribution."""
from __future__ import annotations
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

PKG_ROOT = Path(__file__).resolve().parents[1]
TEST_PRED = PKG_ROOT / "logs" / "test_predictions.csv"

SKIP_CELLS = {(0, 0), (1, 0), (2, 0)}
BAD_HOURS = {3, 4, 11, 13, 22}

TARGET_LO, TARGET_HI = 800, 1200


def hour_of(ts_ms: np.ndarray) -> np.ndarray:
    return ((ts_ms // (15 * 60 * 1000)) % 96 // 4).astype(int)


df = pd.read_csv(TEST_PRED)
df["hour"] = hour_of(df["timestamp"].to_numpy())
# p_up_xmodel IS modele_up's p_up (just renamed for the feature column).
df["p_up"] = df["p_up_xmodel"]

# Apply skip_cells + bad_hours up-front (these are non-negotiable filters).
skip = df.apply(lambda r: (int(r["volatility_state"]), int(r["is_weekend"])) in SKIP_CELLS, axis=1)
bad = df["hour"].isin(BAD_HOURS)
df = df[(~skip) & (~bad)].copy()
print(f"pool after skip+bad_hours: {len(df)} bars")

# Threshold grid : combinations of T_weekend + min obv_pressure_5m + max p_up.
T_grid = np.round(np.arange(0.520, 0.560, 0.002), 4)
obv_quantiles = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.33, 0.40]
max_p_up_grid = [1.0, 0.50, 0.49, 0.48, 0.47, 0.46]

# Compute obv terciles + quantiles on the FILTERED pool (so threshold = absolute value).
obv_vals = df["obv_pressure_5m"].dropna()
obv_thresholds = {q: float(obv_vals.quantile(q)) for q in obv_quantiles}
print(f"obv_pressure_5m quantile thresholds: {obv_thresholds}")

rows = []
for T in T_grid:
    for q in obv_quantiles:
        obv_min = obv_thresholds[q]
        for max_p_up in max_p_up_grid:
            m = (df["p_down"] >= T) & (df["obv_pressure_5m"] >= obv_min) & (df["p_up"] < max_p_up)
            n = int(m.sum())
            if n == 0:
                continue
            wr = float(df.loc[m, "label"].mean())
            if TARGET_LO <= n <= TARGET_HI:
                rows.append({"T": T, "obv_q": q, "obv_min": round(obv_min, 4),
                             "max_p_up": max_p_up, "bets": n, "wr": wr})

out = pd.DataFrame(rows).sort_values("wr", ascending=False).reset_index(drop=True)
print(f"\nconfigs in [{TARGET_LO}, {TARGET_HI}] bets: {len(out)}")
print("\nTop 20 by WR:")
print(out.head(20).to_string())

print("\n--- Top configs with >65% WR ---")
above_65 = out[out["wr"] >= 0.65]
print(f"  {len(above_65)} configs at >=65% WR")
if not above_65.empty:
    print(above_65.head(20).to_string())
