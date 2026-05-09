"""
check_new_features.py
=====================

Outil d'analyse d'impact d'une feature sur la pipeline de production.

Mode par defaut = ABLATION FULL (production-parity) :
  Pour chaque feature cible :
    1. Entraine baseline = TOUS les features (1000 trees + early stopping
       + scale_pos_weight + adaptive threshold per bucket = exact miroir
       de train.py).
    2. Entraine ablated = baseline SANS la feature cible.
    3. Compare sur les metrics qui comptent reellement :
        - val_logloss_best   (signal global, ce qu'apprend le modele)
        - test_logloss       (generalisation hors echantillon)
        - adaptive_total_taken (volume de paris reellement pris)
        - adaptive_total_wr  (WR sur les paris pris)
        - EV = taken * (WR - 0.5) (profit attendu, le seul metric qui paie)

  Convention de signe :
    delta = baseline - ablated = ce que la feature AJOUTE
    delta_EV > 0  -> feature ajoute de l'EV    -> KEEP
    delta_EV < 0  -> feature degrade l'EV       -> DROP

Mode `--quick` :
  Permutation importance sur un modele 300 trees (l'ancienne v1).
  Donne le delta_logloss isole. Rapide (5s par feature) mais ne reflete
  pas la pipeline production : peut classer KEEP une feature qui DROP en
  ablation full (cas typique : feature qui shift la distribution entre val
  et test sans aider la queue ou vivent les bets).

Mode `--multi-seed N` :
  Moyenne sur N seeds pour stabiliser le verdict si delta est borderline.
  Default 1 (XGB_RANDOM_STATE de config.py).

Usage :
    cd backtest/modele_updown
    python check_new_features.py FEATURE [FEATURE ...]            # ablation
    python check_new_features.py --quick FEATURE [FEATURE ...]    # rapide
    python check_new_features.py --multi-seed 3 FEATURE
    python check_new_features.py --list                            # liste features
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from xgboost import XGBClassifier

from config import (
    DB_PATH, TRAIN_DAYS, TEST_DAYS, VAL_FRACTION,
    ADAPTIVE_TARGET_WR, XGB_RANDOM_STATE,
)
from features import build_training_dataframe
from train import (
    build_hyperparams,
    _resolve_scale_pos_weight,
    compute_adaptive_thresholds,
    apply_adaptive_threshold,
)

EXCLUDED = {"timestamp", "label", "volatility_state"}


# ============================================================================
#  Splits + train  (production-parity)
# ============================================================================

def _split(df: pd.DataFrame):
    df = df.sort_values("timestamp").reset_index(drop=True)
    ms_per_day = 86_400_000
    ts_max = df["timestamp"].max()
    test_cutoff = ts_max - TEST_DAYS * ms_per_day
    test_start = int(df["timestamp"].searchsorted(test_cutoff, side="left"))
    if TRAIN_DAYS is None:
        tv_start = 0
    else:
        tv_cutoff = test_cutoff - TRAIN_DAYS * ms_per_day
        tv_start = int(df["timestamp"].searchsorted(tv_cutoff, side="left"))
    n_tv = test_start - tv_start
    val_size = int(n_tv * VAL_FRACTION)
    train_end = test_start - val_size
    return df.iloc[tv_start:train_end], df.iloc[train_end:test_start], df.iloc[test_start:]


def _run_one(df: pd.DataFrame, feature_cols: list[str], seed: int) -> dict:
    """Production-parity train + measure. Returns dict of metrics."""
    train, val, test = _split(df)
    X_train, y_train = train[feature_cols], train["label"]
    X_val,   y_val   = val[feature_cols],   val["label"]
    X_test,  y_test  = test[feature_cols],  test["label"]

    hp = build_hyperparams()
    optional_kwargs = {}
    if hp["reg_lambda"] is not False: optional_kwargs["reg_lambda"] = hp["reg_lambda"]
    if hp["reg_alpha"]  is not False: optional_kwargs["reg_alpha"]  = hp["reg_alpha"]
    if hp["gamma"]      is not False: optional_kwargs["gamma"]      = hp["gamma"]
    spw = _resolve_scale_pos_weight(hp["scale_pos_weight"], y_train)
    if spw is not None:
        optional_kwargs["scale_pos_weight"] = spw

    model = XGBClassifier(
        n_estimators=hp["n_estimators"],
        max_depth=hp["max_depth"],
        learning_rate=hp["learning_rate"],
        min_child_weight=hp["min_child_weight"],
        subsample=hp["subsample"],
        colsample_bytree=hp["colsample_bytree"],
        objective="binary:logistic",
        eval_metric="logloss",
        early_stopping_rounds=hp["early_stopping_rounds"],
        random_state=seed,
        n_jobs=-1,
        **optional_kwargs,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    best = int(model.best_iteration)
    val_ll = float(model.evals_result()["validation_0"]["logloss"][best])

    p_val  = model.predict_proba(X_val)[:, 1]
    p_test = model.predict_proba(X_test)[:, 1]
    test_ll = float(log_loss(y_test, p_test))

    deltas, _ = compute_adaptive_thresholds(val, p_val, ADAPTIVE_TARGET_WR)
    take = apply_adaptive_threshold(test, p_test, deltas, ADAPTIVE_TARGET_WR)
    n_taken = int(take.sum())
    wr = float(y_test.to_numpy()[take].mean()) if n_taken > 0 else 0.0
    ev = n_taken * (wr - 0.5)

    # Bucket breakdown (vol x wknd) sur les paris pris
    bucket_breakdown = {}
    if n_taken > 0 and "volatility_state" in test.columns and "is_weekend" in test.columns:
        vol  = test["volatility_state"].astype(int).to_numpy()
        wknd = test["is_weekend"].astype(int).to_numpy()
        labels = y_test.to_numpy().astype(int)
        for v in (0, 1, 2):
            for w in (0, 1):
                mask = take & (vol == v) & (wknd == w)
                n = int(mask.sum())
                if n > 0:
                    wr_b = float(labels[mask].mean())
                    bucket_breakdown[f"v{v}_w{w}"] = (n, wr_b)

    return {
        "best_iter":    best,
        "val_logloss":  val_ll,
        "test_logloss": test_ll,
        "taken":        n_taken,
        "wr":           wr,
        "ev":           ev,
        "buckets":      bucket_breakdown,
    }


def _run_multiseed(df: pd.DataFrame, feature_cols: list[str],
                    n_seeds: int) -> dict:
    """Moyenne les metrics sur N seeds. Si n_seeds=1, equivalent a _run_one."""
    if n_seeds == 1:
        return _run_one(df, feature_cols, seed=XGB_RANDOM_STATE)

    results = []
    seeds = [XGB_RANDOM_STATE + i for i in range(n_seeds)]
    for s in seeds:
        results.append(_run_one(df, feature_cols, seed=s))

    avg = {
        "best_iter":    np.mean([r["best_iter"] for r in results]),
        "val_logloss":  np.mean([r["val_logloss"] for r in results]),
        "test_logloss": np.mean([r["test_logloss"] for r in results]),
        "taken":        np.mean([r["taken"] for r in results]),
        "wr":           np.mean([r["wr"] for r in results]),
        "ev":           np.mean([r["ev"] for r in results]),
        "ev_std":       np.std([r["ev"] for r in results], ddof=0),
        "buckets":      results[0]["buckets"],
    }
    return avg


# ============================================================================
#  Output
# ============================================================================

def _print_run(name: str, m: dict, n_features: int) -> None:
    print(f"  {name:<14} ({n_features} features)")
    print(f"    best_iter    : {m['best_iter']:>7.0f}")
    print(f"    val_logloss  : {m['val_logloss']:.6f}")
    print(f"    test_logloss : {m['test_logloss']:.6f}")
    print(f"    taken        : {m['taken']:>7,.0f}")
    print(f"    WR           : {m['wr'] * 100:>6.2f}%")
    print(f"    EV           : {m['ev']:+8.2f}")
    if "ev_std" in m:
        print(f"    EV std (multi-seed) : {m['ev_std']:.2f}")


def _print_diff(feature: str, baseline: dict, ablated: dict) -> str:
    """Affiche delta = baseline - ablated = ce que la feature ajoute."""
    d_val   = baseline["val_logloss"]  - ablated["val_logloss"]
    d_test  = baseline["test_logloss"] - ablated["test_logloss"]
    d_taken = baseline["taken"]        - ablated["taken"]
    d_wr    = baseline["wr"]           - ablated["wr"]
    d_ev    = baseline["ev"]           - ablated["ev"]

    print(f"  Delta (baseline avec '{feature}' MOINS ablated sans) :")
    print(f"    val_logloss  : {d_val:+.6f}  "
          f"({'feature aide val' if d_val < 0 else 'feature nuit val'})")
    print(f"    test_logloss : {d_test:+.6f}  "
          f"({'feature aide test' if d_test < 0 else 'feature nuit test'})")
    print(f"    taken        : {d_taken:+,.0f}")
    print(f"    WR           : {d_wr * 100:+.2f}pt")
    print(f"    EV           : {d_ev:+.2f}")

    if d_ev > 1.0:
        verdict = "KEEP"
    elif d_ev > -1.0:
        verdict = "NEUTRE"
    else:
        verdict = "DROP"

    print(f"  Verdict (sur EV) : {verdict}")
    return verdict


# ============================================================================
#  Quick mode (permutation importance, v1)
# ============================================================================

def quick_mode(df: pd.DataFrame, fcols: list[str], targets: list[str]) -> int:
    """Mode v1 : permutation importance sur 300 trees. Rapide mais peu fiable
    pour les features qui shift la distribution."""
    QUICK_HP = dict(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        min_child_weight=50, subsample=0.8, colsample_bytree=0.8,
        gamma=1, random_state=XGB_RANDOM_STATE, n_jobs=-1,
        eval_metric="logloss",
    )

    train, val, test = _split(df)
    print(f"Quick mode (300 trees) : train={len(train):,} test={len(test):,}\n")
    print(f"Training... ", end="", flush=True)
    t0 = time.time()
    m = XGBClassifier(**QUICK_HP)
    m.fit(train[fcols], train["label"], verbose=False)
    print(f"{time.time() - t0:.1f}s")

    p_base = m.predict_proba(test[fcols])[:, 1]
    ll_base = log_loss(test["label"], p_base)
    print(f"Baseline test logloss : {ll_base:.6f}\n")

    imp = dict(zip(fcols, m.feature_importances_))
    ranking = sorted(fcols, key=lambda c: -imp[c])

    print(f"  {'feature':<22}{'gain':>9}{'rank':>9}{'delta_logloss':>17}   verdict")
    print("  " + "-" * 70)
    rng = np.random.default_rng(0)
    for f in targets:
        X = test[fcols].copy()
        X[f] = rng.permutation(X[f].values)
        ll = log_loss(test["label"], m.predict_proba(X)[:, 1])
        delta = ll - ll_base
        if delta > 1e-5:
            verdict = "KEEP"
        elif delta > -1e-5:
            verdict = "NEUTRE"
        else:
            verdict = "DROP"
        rank = ranking.index(f) + 1
        print(f"  {f:<22}{imp[f]:>9.4f}{rank:>5}/{len(fcols):<3}"
              f"   {delta:+.6f}      {verdict}")

    print(f"\n  ATTENTION : ce mode ne reflete pas la pipeline production.")
    print(f"  Pour un verdict fiable, relance sans --quick (ablation full).")
    return 0


# ============================================================================
#  Main
# ============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("features", nargs="*",
                        help="Features cibles. Si vide -> --list affiche.")
    parser.add_argument("--quick", action="store_true",
                        help="Mode permutation rapide (v1, peu fiable).")
    parser.add_argument("--multi-seed", type=int, default=1, metavar="N",
                        help="Moyenne sur N seeds (default 1).")
    parser.add_argument("--list", action="store_true",
                        help="Affiche la liste des features disponibles.")
    args = parser.parse_args()

    print(f"Loading parquet...")
    df = build_training_dataframe(DB_PATH, verbose=False)
    fcols = [c for c in df.columns if c not in EXCLUDED]
    print(f"  {len(df):,} rows x {df.shape[1]} cols   ({len(fcols)} features training)")

    if args.list or not args.features:
        print(f"\nFeatures disponibles :")
        for i, c in enumerate(fcols, 1):
            print(f"  {i:>2}. {c}")
        if not args.features:
            print(f"\nUsage : python check_new_features.py FEATURE [FEATURE ...]")
            return 0
        return 0

    missing = [f for f in args.features if f not in fcols]
    if missing:
        print(f"\nERROR : feature(s) absente(s) du parquet : {missing}",
              file=sys.stderr)
        return 1

    if args.quick:
        return quick_mode(df, fcols, args.features)

    print(f"\nMode : ABLATION FULL  (production-parity, "
          f"{args.multi_seed} seed{'s' if args.multi_seed > 1 else ''})")
    print(f"Hyperparams : 1000 trees + early stopping + adaptive threshold "
          f"(target_wr={ADAPTIVE_TARGET_WR})")
    train, val, test = _split(df)
    print(f"Split : train={len(train):,} | val={len(val):,} | test={len(test):,}")
    print()

    print(f"Baseline (avec toutes les features)... ", end="", flush=True)
    t0 = time.time()
    baseline = _run_multiseed(df, fcols, args.multi_seed)
    print(f"{time.time() - t0:.1f}s")
    _print_run("baseline", baseline, len(fcols))

    summary = []
    for f in args.features:
        ablated_cols = [c for c in fcols if c != f]
        print(f"\nAblation sans '{f}' ({len(ablated_cols)} features)... ",
              end="", flush=True)
        t0 = time.time()
        ablated = _run_multiseed(df, ablated_cols, args.multi_seed)
        print(f"{time.time() - t0:.1f}s")
        _print_run(f"no_{f}", ablated, len(ablated_cols))

        print()
        verdict = _print_diff(f, baseline, ablated)
        summary.append((f, baseline["ev"] - ablated["ev"],
                        baseline["taken"] - ablated["taken"],
                        baseline["wr"] - ablated["wr"], verdict))

    if len(summary) > 1:
        print()
        print("=" * 80)
        print(f"  SYNTHESE")
        print("=" * 80)
        print(f"  {'feature':<24}{'dEV':>10}{'dTaken':>10}{'dWR':>10}   verdict")
        print("  " + "-" * 70)
        for f, dev, dtk, dwr, v in summary:
            print(f"  {f:<24}{dev:>+10.2f}{dtk:>+10.0f}{dwr * 100:>+9.2f}pt    {v}")

        n_keep = sum(1 for _, _, _, _, v in summary if v == "KEEP")
        n_drop = sum(1 for _, _, _, _, v in summary if v == "DROP")
        n_neutre = len(summary) - n_keep - n_drop
        print(f"\n  {n_keep} KEEP  |  {n_neutre} NEUTRE  |  {n_drop} DROP")

    return 0


if __name__ == "__main__":
    sys.exit(main())
