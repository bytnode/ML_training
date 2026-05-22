"""Experiment 10: model-diversity ensemble. XGBoost + LightGBM +
HistGradientBoosting, all NaN-native. Decorrelated models rank-averaged can
beat any single model. Screened on production window, then walk-forwarded.
"""
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import HistGradientBoostingClassifier

import lib
import lib2
import ethfeats

ETH_COLS = open(lib.PKG / "eth_best_cols.txt").read().split()


def train_xgb(Xtr, ytr, Xv, yv, seed=42):
    n_neg, n_pos = int((ytr == 0).sum()), int((ytr == 1).sum())
    m = xgb.XGBClassifier(n_estimators=1000, max_depth=6, learning_rate=0.05,
                          min_child_weight=20, subsample=0.8, colsample_bytree=0.8,
                          gamma=1.5, objective="binary:logistic",
                          eval_metric="logloss", early_stopping_rounds=50,
                          random_state=seed, n_jobs=-1,
                          scale_pos_weight=n_neg / n_pos)
    m.fit(Xtr, ytr, eval_set=[(Xv, yv)], verbose=False)
    return lambda X: m.predict_proba(X)[:, 1]


def train_lgb(Xtr, ytr, Xv, yv, seed=42):
    m = lgb.LGBMClassifier(n_estimators=1500, max_depth=6, num_leaves=48,
                           learning_rate=0.04, min_child_samples=60,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                           reg_lambda=1.0, random_state=seed, n_jobs=-1,
                           verbose=-1)
    m.fit(Xtr, ytr, eval_set=[(Xv, yv)],
          callbacks=[lgb.early_stopping(60, verbose=False)])
    return lambda X: m.predict_proba(X)[:, 1]


def train_hgb(Xtr, ytr, Xv, yv, seed=42):
    m = HistGradientBoostingClassifier(max_iter=500, max_depth=7,
                                       learning_rate=0.05, min_samples_leaf=80,
                                       l2_regularization=1.0,
                                       early_stopping=True, validation_fraction=0.12,
                                       random_state=seed)
    m.fit(Xtr, ytr)
    return lambda X: m.predict_proba(X)[:, 1]


def rank(p):
    return pd.Series(p).rank(pct=True).to_numpy()


def report(val, test, p_val, p_test, tag):
    yv = val["label"].to_numpy().astype(int)
    yt = test["label"].to_numpy().astype(int)
    av = val["volatility_state"].to_numpy().astype(int) != 2
    at = test["volatility_state"].to_numpy().astype(int) != 2
    fv = lib2.frontier_at(p_val, yv, av, 2700)
    ft = lib2.frontier_at(p_test, yt, at, 2500)
    r = lib2.wr_at_volume(val, p_val, test, p_test, skip_fn=lib.skip_vol2,
                          axis_fn=lib.axis_weekend, target_n=2500,
                          axis_frac_ratio={0: 1.0, 1: 4.0})
    print(f"  {tag:<28} front_val={fv*100:5.2f}  front_test={ft*100:5.2f}  "
          f"clean@2500={r['wr']*100:5.2f}")


def main():
    df = lib.load_master()
    base_fc = lib.feature_cols(df)
    eth = ethfeats.build_eth(df["timestamp"].to_numpy(), verbose=False)
    merged = df.merge(eth, left_on="timestamp", right_on="timestamp_ms",
                      how="left").drop(columns=["timestamp_ms"])
    fc = base_fc + [c for c in ETH_COLS if c in merged.columns]
    train, val, test = lib.temporal_split(merged)
    Xtr, ytr = train[fc], train["label"]
    Xv, yv = val[fc], val["label"]
    Xte = test[fc]
    print(f"features={len(fc)}  train={len(train):,}\n")

    preds_v, preds_t = {}, {}
    for name, fn in (("xgb", train_xgb), ("lgb", train_lgb), ("hgb", train_hgb)):
        f = fn(Xtr, ytr, Xv, yv)
        preds_v[name] = f(Xv)
        preds_t[name] = f(Xte)
        report(val, test, preds_v[name], preds_t[name], name)

    print()
    # raw-average ensembles
    for combo in (("xgb", "lgb"), ("xgb", "hgb"), ("xgb", "lgb", "hgb")):
        pv = np.mean([preds_v[m] for m in combo], axis=0)
        pt = np.mean([preds_t[m] for m in combo], axis=0)
        report(val, test, pv, pt, "avg[" + "+".join(combo) + "]")
    # rank-average ensemble
    pv = np.mean([rank(preds_v[m]) for m in ("xgb", "lgb", "hgb")], axis=0)
    pt = np.mean([rank(preds_t[m]) for m in ("xgb", "lgb", "hgb")], axis=0)
    report(val, test, pv, pt, "rankavg[xgb+lgb+hgb]")

    # multi-seed xgb
    print()
    pv = np.zeros(len(val))
    pt = np.zeros(len(test))
    for s in (42, 1, 2, 3, 4):
        f = train_xgb(Xtr, ytr, Xv, yv, seed=s)
        pv += f(Xv)
        pt += f(Xte)
    report(val, test, pv / 5, pt / 5, "xgb x5 seeds")


if __name__ == "__main__":
    main()
