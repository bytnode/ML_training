"""
Add p_up as a feature to the modele_down parquet, computed by the sibling
modele_up XGBoost model.

The UP model is loaded from ../modele_up/btc_updown_model.json. It is run
on all 1h candles to produce p_up. The parquet gets a new column 'p_up_sibling'.

Idempotent: re-running overwrites the existing column.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb


ROOT = Path(__file__).resolve().parent
DOWN_PARQUET = ROOT / "DB" / "btc_multi_tf_1h.parquet"
UP_MODEL = ROOT.parent / "modele_up" / "btc_updown_model.json"


def main():
    if not UP_MODEL.exists():
        raise FileNotFoundError(f"UP model not found: {UP_MODEL}")
    if not DOWN_PARQUET.exists():
        raise FileNotFoundError(f"DOWN parquet not found: {DOWN_PARQUET}")

    print(f"Loading UP model: {UP_MODEL}")
    up = xgb.XGBClassifier()
    up.load_model(str(UP_MODEL))
    up_features = up.get_booster().feature_names
    print(f"  UP features ({len(up_features)}): {up_features[:5]}...")

    print(f"Loading DOWN parquet: {DOWN_PARQUET}")
    df = pd.read_parquet(DOWN_PARQUET)
    print(f"  {len(df):,} rows, {len(df.columns)} cols")

    missing = [c for c in up_features if c not in df.columns]
    if missing:
        raise RuntimeError(f"DOWN parquet missing UP features: {missing}")

    X = df[up_features].copy()
    # XGBoost predict_proba can handle NaN, fill anyway for safety
    mask_valid = ~X.isna().any(axis=1)
    p_up = np.full(len(df), np.nan, dtype=float)
    p_up_valid = up.predict_proba(X[mask_valid])[:, 1]
    p_up[mask_valid.to_numpy()] = p_up_valid

    df["p_up_sibling"] = p_up
    df["p_down_sibling"] = 1.0 - p_up

    valid_count = int(mask_valid.sum())
    print(f"\nPredicted p_up on {valid_count:,} / {len(df):,} valid rows")
    print(f"  p_up_sibling   : min={p_up_valid.min():.4f} max={p_up_valid.max():.4f} "
          f"mean={p_up_valid.mean():.4f}")
    print(f"  p_down_sibling : min={(1-p_up_valid).max() if len(p_up_valid)>0 else 0:.4f} "
          f"max={1-p_up_valid.min() if len(p_up_valid)>0 else 0:.4f}")

    df.to_parquet(DOWN_PARQUET, index=False)
    print(f"\nWrote back {DOWN_PARQUET} with new columns [p_up_sibling, p_down_sibling]")


if __name__ == "__main__":
    main()
