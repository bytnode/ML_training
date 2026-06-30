"""Experiment 09: training-side tricks on the baseline+ETH feature set.
  - decisive label: train on (next red AND |body| >= percentile)
  - sample weighting by next-candle |body|
  - regression on next signed body (rank by predicted body, ascending)
Screened on the production window (front_val + clean@2500). Survivors get
walk-forwarded later.
"""
import numpy as np
import xgboost as xgb

import lib
import lib2
import lib3
import ethfeats

ETH_COLS = open(lib.PKG / "eth_best_cols.txt").read().split()


def evaluate_scores(val, test, p_val, p_test, tag):
    yv = val["label"].to_numpy().astype(int)
    yt = test["label"].to_numpy().astype(int)
    av = val["volatility_state"].to_numpy().astype(int) != 2
    at = test["volatility_state"].to_numpy().astype(int) != 2
    fv = lib2.frontier_at(p_val, yv, av, 2700)
    ft = lib2.frontier_at(p_test, yt, at, 2500)
    r = lib2.wr_at_volume(val, p_val, test, p_test, skip_fn=lib.skip_vol2,
                          axis_fn=lib.axis_weekend, target_n=2500,
                          axis_frac_ratio={0: 1.0, 1: 4.0})
    print(f"  {tag:<30} front_val={fv*100:5.2f}  front_test={ft*100:5.2f}  "
          f"clean@2500={r['wr']*100:5.2f}")
    return fv


def main():
    df = lib.load_master()
    df = lib3.attach_next_body(df)
    base_fc = lib.feature_cols(df)
    base_fc = [c for c in base_fc if c not in ("next_body_abs", "next_body_signed")]
    eth = ethfeats.build_eth(df["timestamp"].to_numpy(), verbose=False)
    merged = df.merge(eth, left_on="timestamp", right_on="timestamp_ms",
                      how="left").drop(columns=["timestamp_ms"])
    fc = base_fc + [c for c in ETH_COLS if c in merged.columns]
    print(f"features: {len(fc)}\n")

    train, val, test = lib.temporal_split(merged)
    Xtr, Xv, Xte = train[fc], val[fc], test[fc]

    print("=== baseline+ETH (standard label) ===")
    m = lib.train_model(train, val, fc)
    evaluate_scores(val, test, lib.predict(m, val, fc),
                    lib.predict(m, test, fc), "standard label")

    # ---- decisive label ----
    print("\n=== decisive label: train on (red AND |body|>=pct) ----")
    nb = train["next_body_abs"].to_numpy()
    for q in (30, 40, 50, 60):
        thr = np.nanpercentile(nb, q)
        y2_tr = ((train["label"].to_numpy() == 1) & (nb >= thr)).astype(int)
        nbv = val["next_body_abs"].to_numpy()
        y2_v = ((val["label"].to_numpy() == 1) & (nbv >= thr)).astype(int)
        mm = lib3.train_raw(Xtr, y2_tr, Xv, y2_v)
        evaluate_scores(val, test, mm.predict_proba(Xv)[:, 1],
                        mm.predict_proba(Xte)[:, 1], f"decisive q={q}")

    # ---- sample weighting by |body| ----
    print("\n=== sample weighting by next |body| ===")
    nb_tr = train["next_body_abs"].to_numpy()
    med = np.nanmedian(nb_tr)
    for power in (0.5, 1.0, 2.0):
        w = np.power(np.clip(nb_tr / med, 0.05, 20.0), power)
        w = np.where(np.isnan(w), 1.0, w)
        mm = lib3.train_raw(Xtr, train["label"], Xv, val["label"],
                            sample_weight=w)
        evaluate_scores(val, test, mm.predict_proba(Xv)[:, 1],
                        mm.predict_proba(Xte)[:, 1], f"weight^{power}")

    # ---- regression on signed body ----
    print("\n=== regression on next signed body (score = -pred) ===")
    ytr_r = train["next_body_signed"].to_numpy()
    yv_r = val["next_body_signed"].to_numpy()
    ok_tr = ~np.isnan(ytr_r)
    ok_v = ~np.isnan(yv_r)
    reg = xgb.XGBRegressor(n_estimators=1000, max_depth=5, learning_rate=0.05,
                           min_child_weight=20, subsample=0.8,
                           colsample_bytree=0.8, gamma=1.5,
                           early_stopping_rounds=50, random_state=42, n_jobs=-1)
    reg.fit(Xtr[ok_tr], ytr_r[ok_tr], eval_set=[(Xv[ok_v], yv_r[ok_v])],
            verbose=False)
    evaluate_scores(val, test, -reg.predict(Xv), -reg.predict(Xte),
                    "regression(signed body)")


if __name__ == "__main__":
    main()
