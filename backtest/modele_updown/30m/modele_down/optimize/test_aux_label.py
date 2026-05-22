"""
Test training on an AUXILIARY target (sharper bearish signal), then evaluating
WR on the original label "next bar red".

Auxiliary targets to try:
  - label_streak2: 1 if next 2 bars are both red
  - label_strong: 1 if next bar is red AND |close-open|/open > 0.5%
  - label_dump: 1 if next bar is red AND close < open - 0.3% AND volume_ratio > 1.2

If a sharper auxiliary target produces better-spread predictions, the threshold
approach should bite harder.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import xgboost as xgb

_PKG = Path(__file__).resolve().parent.parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from common import load_dataset, feature_columns, temporal_split, hour_of_day_from_ms
from train import compute_thresholds_asym_wknd_excl, _resolve_scale_pos_weight, build_hyperparams


def add_aux_labels(df):
    """Adds label_streak2, label_strong, label_dump on top of existing label.
    Requires we have raw close_ret in the parquet."""
    cr = df["close_ret"].to_numpy()
    cr_next = np.concatenate([cr[1:], [np.nan]])
    cr_next2 = np.concatenate([cr[2:], [np.nan, np.nan]])
    vol_next = np.concatenate([df["volume_ratio"].to_numpy()[1:], [np.nan]])

    # We use close_ret < 0 as "red" proxy. The original label uses
    # cr_next < or_next (close less than open) but here we just use close_ret < 0.
    next_red = cr_next < 0
    next2_red = cr_next2 < 0
    df["label_streak2"] = (next_red & next2_red).astype(float)
    df["label_strong"] = (next_red & (cr_next < -0.005)).astype(float)  # red > 0.5%
    df["label_dump"]   = (next_red & (cr_next < -0.003) & (vol_next > 1.2)).astype(float)
    df.loc[df["label"].isna(), ["label_streak2", "label_strong", "label_dump"]] = np.nan
    return df


def train_with_label(df, fc, label_col, seed=42, hp_overrides=None):
    train, val, test = temporal_split(df)
    X_train, y_train = train[fc], train[label_col].dropna()
    valid_idx = train[label_col].dropna().index
    X_train = train.loc[valid_idx, fc]
    X_val, y_val = val[fc], val[label_col]
    # Drop rows with NaN in aux label
    val_mask = ~y_val.isna()
    X_val = X_val[val_mask]
    y_val = y_val[val_mask].astype(int)

    hp = build_hyperparams(hp_overrides)
    opt = {}
    if hp["reg_lambda"] is not False: opt["reg_lambda"] = hp["reg_lambda"]
    if hp["reg_alpha"]  is not False: opt["reg_alpha"]  = hp["reg_alpha"]
    if hp["gamma"]      is not False: opt["gamma"]      = hp["gamma"]
    spw = _resolve_scale_pos_weight(hp["scale_pos_weight"], y_train.astype(int))
    if spw is not None:
        opt["scale_pos_weight"] = spw

    model = xgb.XGBClassifier(
        n_estimators=hp["n_estimators"], max_depth=hp["max_depth"],
        learning_rate=hp["learning_rate"], min_child_weight=hp["min_child_weight"],
        subsample=hp["subsample"], colsample_bytree=hp["colsample_bytree"],
        objective="binary:logistic", eval_metric="logloss",
        early_stopping_rounds=hp["early_stopping_rounds"],
        random_state=seed, n_jobs=-1, **opt,
    )
    model.fit(X_train, y_train.astype(int), eval_set=[(X_val, y_val)], verbose=False)
    return model, train, val, test


def main():
    df = load_dataset()
    df = add_aux_labels(df)
    # IMPORTANT: also exclude all aux labels from features
    fc = [c for c in feature_columns(df)
          if c not in ("label_streak2", "label_strong", "label_dump")]
    print(f"Dataset: {len(df):,} rows, {len(fc)} features.")

    # Distribution of aux labels
    for col in ("label", "label_streak2", "label_strong", "label_dump"):
        v = df[col].dropna()
        if len(v) > 0:
            print(f"  {col:<16} mean={v.mean():.4f}  n={len(v):,}")

    for label_col in ("label", "label_streak2", "label_strong", "label_dump"):
        print(f"\n=== Training on {label_col} ===")
        model, train, val, test = train_with_label(df, fc, label_col)
        # Predict on val and test (P(aux_label = 1))
        # Filter NaN in val features
        val_clean = val.dropna(subset=fc)
        test_clean = test.dropna(subset=fc)
        p_val = model.predict_proba(val_clean[fc])[:, 1]
        p_test = model.predict_proba(test_clean[fc])[:, 1]

        print(f"  p_test range: min={p_test.min():.4f} max={p_test.max():.4f} mean={p_test.mean():.4f}")

        # Evaluate WR on ORIGINAL label
        y_test = test_clean["label"].astype(int).to_numpy()
        # Sort by p_test desc, look at top-N WR (original label)
        order = np.argsort(-p_test)
        print(f"  WR on ORIGINAL label by top-N (sorted by P({label_col})):")
        for n in (200, 500, 750, 1000, 1500):
            if n > len(p_test):
                continue
            sel = order[:n]
            wr = float(y_test[sel].mean())
            p_min = float(p_test[sel].min())
            print(f"    top {n:>5} (p_aux>={p_min:.4f}):  WR={wr*100:.2f}%")


if __name__ == "__main__":
    main()
