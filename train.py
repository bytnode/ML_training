"""
Entrainement + evaluation du modele XGBoost.

Usage CLI :
    python train.py
        Entraine avec les hyperparams de config.py, sauvegarde le modele,
        affiche l'evaluation, et appende un row dans logs/runs.csv.

API librairie :
    from train import train_and_evaluate
    metrics = train_and_evaluate(
        overrides={"max_depth": 6, "learning_rate": 0.1},
        save_model=False,
        verbose=False,
        log_run=True,
        run_label="sweep:max_depth=6",
    )
    # metrics est un dict avec hyperparams + best_iteration + val_logloss + WR par seuil + breakdown
"""

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from config import (
    CSV_PATH, MODEL_OUTPUT,
    TRAIN_DAYS, TEST_DAYS, VAL_FRACTION,
    THRESHOLDS, BREAKDOWN_THRESHOLD,
    XGB_N_ESTIMATORS, XGB_MAX_DEPTH, XGB_LEARNING_RATE,
    XGB_MIN_CHILD_WEIGHT, XGB_SUBSAMPLE, XGB_COLSAMPLE_BYTREE,
    XGB_REG_LAMBDA, XGB_REG_ALPHA, XGB_GAMMA,
    XGB_EARLY_STOPPING_ROUNDS, XGB_RANDOM_STATE,
)

ROOT = Path(__file__).resolve().parent
LOGS_DIR = ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)
RUNS_CSV = LOGS_DIR / "runs.csv"

# Ensemble de seuils logges dans runs.csv. Independant de THRESHOLDS (config) pour
# garantir un schema CSV stable meme si l'utilisateur change THRESHOLDS.
LOG_THRESHOLDS = [0.50, 0.55, 0.58, 0.60, 0.62]


# ============================================================
#  Hyperparameters builder
# ============================================================
def build_hyperparams(overrides: dict | None = None) -> dict:
    """Construit le dict d'hyperparams en partant de config.py + overrides."""
    hp = {
        "n_estimators": XGB_N_ESTIMATORS,
        "max_depth": XGB_MAX_DEPTH,
        "learning_rate": XGB_LEARNING_RATE,
        "min_child_weight": XGB_MIN_CHILD_WEIGHT,
        "subsample": XGB_SUBSAMPLE,
        "colsample_bytree": XGB_COLSAMPLE_BYTREE,
        "reg_lambda": XGB_REG_LAMBDA,
        "reg_alpha": XGB_REG_ALPHA,
        "gamma": XGB_GAMMA,
        "early_stopping_rounds": XGB_EARLY_STOPPING_ROUNDS,
        "random_state": XGB_RANDOM_STATE,
    }
    if overrides:
        hp.update(overrides)
    return hp


# ============================================================
#  Train + evaluate (fonction principale, appelable depuis sweep.py)
# ============================================================
def train_and_evaluate(
    overrides: dict | None = None,
    save_model: bool = True,
    verbose: bool = True,
    log_run: bool = True,
    run_label: str = "manual",
) -> dict:
    """
    Entraine et evalue le modele.

    overrides   : dict d'hyperparams a surcharger sur ceux de config.py.
    save_model  : si True, ecrit le modele dans MODEL_OUTPUT (config).
    verbose     : si True, imprime l'evaluation complete (table seuils + breakdown + importance).
    log_run     : si True, append une ligne dans logs/runs.csv.
    run_label   : etiquette du run (visible dans runs.csv pour filtrer ulterieurement).

    Retourne un dict de metriques (hyperparams + WR par seuil + breakdown + meta).
    """
    t0 = time.time()

    # ---- Load + split ----
    df = pd.read_csv(CSV_PATH)
    df = df.sort_values("timestamp").reset_index(drop=True)

    ms_per_day = 86_400 * 1000
    ts_max = df["timestamp"].max()

    # 1. test = les TEST_DAYS derniers jours
    test_cutoff_ts = ts_max - TEST_DAYS * ms_per_day
    test_start = int(df["timestamp"].searchsorted(test_cutoff_ts, side="left"))

    # 2. train+val = les TRAIN_DAYS jours juste avant le test (TRAIN_DAYS=None => tout l'historique)
    if TRAIN_DAYS is None:
        train_val_start = 0
    else:
        train_val_cutoff_ts = test_cutoff_ts - TRAIN_DAYS * ms_per_day
        train_val_start = int(df["timestamp"].searchsorted(train_val_cutoff_ts, side="left"))

    # 3. val = les VAL_FRACTION derniers du bloc train+val
    n_train_val = test_start - train_val_start
    val_size = int(n_train_val * VAL_FRACTION)
    train_end = test_start - val_size
    val_end = test_start

    train = df.iloc[train_val_start:train_end]
    val   = df.iloc[train_end:val_end]
    test  = df.iloc[val_end:]

    if verbose:
        ignored = train_val_start
        print(f"\nSplit temporel : TRAIN_DAYS={TRAIN_DAYS}, TEST_DAYS={TEST_DAYS}, VAL_FRACTION={VAL_FRACTION}")
        print(f"  ignore (donnees plus vieilles que train) : {ignored:,} rows")
        print(f"  train : {len(train):,} rows  ({pd.to_datetime(train['timestamp'].min(), unit='ms').date()} -> {pd.to_datetime(train['timestamp'].max(), unit='ms').date()})")
        print(f"  val   : {len(val):,} rows  ({pd.to_datetime(val['timestamp'].min(), unit='ms').date()} -> {pd.to_datetime(val['timestamp'].max(), unit='ms').date()})")
        print(f"  test  : {len(test):,} rows  ({pd.to_datetime(test['timestamp'].min(), unit='ms').date()} -> {pd.to_datetime(test['timestamp'].max(), unit='ms').date()})")

    feature_cols = [c for c in df.columns if c not in ("timestamp", "label")]
    X_train, y_train = train[feature_cols], train["label"]
    X_val, y_val = val[feature_cols], val["label"]
    X_test, y_test = test[feature_cols], test["label"]

    # ---- Build model ----
    hp = build_hyperparams(overrides)

    optional_kwargs = {}
    if hp["reg_lambda"] is not False:
        optional_kwargs["reg_lambda"] = hp["reg_lambda"]
    if hp["reg_alpha"] is not False:
        optional_kwargs["reg_alpha"] = hp["reg_alpha"]
    if hp["gamma"] is not False:
        optional_kwargs["gamma"] = hp["gamma"]

    model = xgb.XGBClassifier(
        n_estimators=hp["n_estimators"],
        max_depth=hp["max_depth"],
        learning_rate=hp["learning_rate"],
        min_child_weight=hp["min_child_weight"],
        subsample=hp["subsample"],
        colsample_bytree=hp["colsample_bytree"],
        objective="binary:logistic",
        eval_metric="logloss",
        early_stopping_rounds=hp["early_stopping_rounds"],
        random_state=hp["random_state"],
        n_jobs=-1,
        **optional_kwargs,
    )

    if verbose:
        print(f"\nTraining XGBoost  ({len(train):,} train / {len(val):,} val / {len(test):,} test)...")
        print(f"  Hyperparams : {hp}")

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100 if verbose else False,
    )

    best_iter = int(model.best_iteration)
    val_logloss_best = float(model.evals_result()["validation_0"]["logloss"][best_iter])

    # ---- Test predictions ----
    y_pred = model.predict_proba(X_test)[:, 1]
    baseline_wr = float(y_test.mean())

    # WR + n + pct_vol pour chaque seuil de LOG_THRESHOLDS
    threshold_metrics = {}
    for th in LOG_THRESHOLDS:
        mask = y_pred >= th
        n = int(mask.sum())
        wr = float(y_test[mask].mean()) if n > 0 else None
        pct_vol = n / len(y_test) if len(y_test) else 0.0
        threshold_metrics[th] = {"n": n, "wr": wr, "pct_vol": pct_vol}

    # Breakdown au seuil BREAKDOWN_THRESHOLD
    th = BREAKDOWN_THRESHOLD
    test_full = test.copy()
    test_full["pred"] = y_pred
    test_full["taken"] = test_full["pred"] >= th
    total_taken = int(test_full["taken"].sum())
    total_wr = float(test_full[test_full["taken"]]["label"].mean()) if total_taken > 0 else None
    breakdown_per_seq = {}
    for sl in [1, 2, 3, 4]:
        sub = test_full[(test_full["seq_len"] == sl) & test_full["taken"]]
        breakdown_per_seq[sl] = {
            "n": int(len(sub)),
            "wr": float(sub["label"].mean()) if len(sub) > 0 else None,
        }

    duration = time.time() - t0

    # ---- Save ----
    if save_model:
        model.save_model(MODEL_OUTPUT)

    # ---- Build metrics dict (tout flat, pour le CSV) ----
    metrics = {
        "timestamp_iso": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_label": run_label,
        "duration_sec": round(duration, 2),
        "csv_path": str(CSV_PATH),
        "train_days": TRAIN_DAYS,
        "test_days": TEST_DAYS,
        "val_fraction": VAL_FRACTION,
        "n_features": len(feature_cols),
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "best_iteration": best_iter,
        "val_logloss_best": round(val_logloss_best, 6),
        "baseline_wr": round(baseline_wr, 5),
        "breakdown_threshold": th,
        "breakdown_total_taken": total_taken,
        "breakdown_total_wr": round(total_wr, 5) if total_wr is not None else None,
    }
    for k, v in hp.items():
        metrics[f"hp_{k}"] = v
    for th_, m in threshold_metrics.items():
        metrics[f"wr_at_{th_:.2f}"] = round(m["wr"], 5) if m["wr"] is not None else None
        metrics[f"n_at_{th_:.2f}"] = m["n"]
        metrics[f"pct_at_{th_:.2f}"] = round(m["pct_vol"], 5)
    for sl, m in breakdown_per_seq.items():
        metrics[f"wr_seq{sl}"] = round(m["wr"], 5) if m["wr"] is not None else None
        metrics[f"n_seq{sl}"] = m["n"]

    # ---- Print + log ----
    if verbose:
        _print_evaluation(metrics, breakdown_per_seq, model, feature_cols, train, test, y_pred, y_test)

    if log_run:
        append_to_runs_csv(metrics)

    return metrics


# ============================================================
#  CSV logging (gere l'evolution de schema en relisant + reecrivant)
# ============================================================
def append_to_runs_csv(metrics: dict) -> None:
    """
    Append le dict de metriques a logs/runs.csv. Si le CSV existe deja avec
    des colonnes differentes, fait l'union des colonnes (NaN pour les manquantes)
    et reecrit le fichier complet — c'est OK car le fichier reste petit (< 1000 runs).
    """
    new_row = pd.DataFrame([metrics])
    if RUNS_CSV.exists():
        existing = pd.read_csv(RUNS_CSV)
        # Aligner les colonnes manuellement pour eviter le FutureWarning de pandas
        # sur les colonnes all-NA dans concat.
        all_cols = list(existing.columns) + [c for c in new_row.columns if c not in existing.columns]
        existing = existing.reindex(columns=all_cols)
        new_row = new_row.reindex(columns=all_cols)
        df = pd.concat([existing, new_row], ignore_index=True)
    else:
        df = new_row
    df.to_csv(RUNS_CSV, index=False)


# ============================================================
#  Pretty-print de l'evaluation (preserve l'output console historique)
# ============================================================
def _print_evaluation(metrics, breakdown_per_seq, model, feature_cols, train, test, y_pred, y_test):
    print("\n" + "=" * 70)
    print("  EVALUATION SUR LE TEST SET (out-of-sample)")
    print("=" * 70)
    print(f"\nBaseline (filtre body_ratio seul) : {metrics['baseline_wr'] * 100:.2f}% WR sur {metrics['n_test']} paris")

    print(f"\n{'Seuil':<8}{'Paris pris':<14}{'Win rate':<12}{'% volume conserve'}")
    print("-" * 55)
    # Affiche les seuils de THRESHOLDS (config) pour preserver l'output historique.
    # y_pred est deja calcule dans train_and_evaluate, on le reutilise.
    for th in THRESHOLDS:
        mask = y_pred >= th
        n_bets = int(mask.sum())
        if n_bets == 0:
            print(f"{th:<8}{'0':<14}{'N/A':<12}{'0.00%'}")
            continue
        wr = y_test[mask].mean() * 100
        pct_vol = n_bets / len(y_test) * 100
        print(f"{th:<8}{n_bets:<14}{wr:>6.2f}%{'':<5}{pct_vol:>6.2f}%")

    print(f"\n--- Breakdown par longueur de sequence (seuil={metrics['breakdown_threshold']}) ---")
    total_wr_pct = (metrics['breakdown_total_wr'] or 0) * 100
    print(f"  TOTAL paris pris par l'IA : {metrics['breakdown_total_taken']} (WR = {total_wr_pct:.2f}%)")
    for sl, m in breakdown_per_seq.items():
        if m["n"] > 0:
            print(f"  Seq {sl}b : {m['n']:>5} paris, WR = {m['wr'] * 100:.2f}%")

    print("\n--- Feature importance (top 10) ---")
    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)
    print(importance.head(10).to_string(index=False))

    print(f"\nDuration : {metrics['duration_sec']}s | best_iteration : {metrics['best_iteration']} | val_logloss : {metrics['val_logloss_best']}")


# ============================================================
#  CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Train + evaluate XGBoost reversal model.")
    parser.add_argument("--label", default="manual", help="Run label written to runs.csv (defaut : 'manual').")
    parser.add_argument("--no-save", action="store_true", help="Ne pas sauvegarder le modele sur disque.")
    parser.add_argument("--no-log", action="store_true", help="Ne pas appender a runs.csv.")
    args = parser.parse_args()

    print("Loading data...")
    metrics = train_and_evaluate(
        overrides=None,
        save_model=not args.no_save,
        verbose=True,
        log_run=not args.no_log,
        run_label=args.label,
    )

    if not args.no_save:
        print(f"\nModel saved to {MODEL_OUTPUT}")
    if not args.no_log:
        print(f"Run logged to {RUNS_CSV}")


if __name__ == "__main__":
    main()
