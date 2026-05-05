import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

from config import (
    CSV_PATH, MODEL_OUTPUT,
    TEST_DAYS, VAL_FRACTION,
    THRESHOLDS, BREAKDOWN_THRESHOLD,
    XGB_N_ESTIMATORS, XGB_MAX_DEPTH, XGB_LEARNING_RATE,
    XGB_MIN_CHILD_WEIGHT, XGB_SUBSAMPLE, XGB_COLSAMPLE_BYTREE,
    XGB_REG_LAMBDA, XGB_REG_ALPHA, XGB_GAMMA,
    XGB_EARLY_STOPPING_ROUNDS, XGB_RANDOM_STATE,
)

def main():
    print("Loading data...")
    df = pd.read_csv(CSV_PATH)
    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"Total examples : {len(df)}")

    # Split temporel (PAS DE SHUFFLE) :
    #   test  = les TEST_DAYS derniers jours du CSV
    #   train = tout ce qui précède, moins VAL_FRACTION pour la validation
    ms_per_day = 86_400 * 1000
    test_cutoff_ts = df["timestamp"].max() - TEST_DAYS * ms_per_day
    test_start = int(df["timestamp"].searchsorted(test_cutoff_ts, side="left"))

    val_size = int(test_start * VAL_FRACTION)
    train_end = test_start - val_size
    val_end = test_start

    train = df.iloc[:train_end]
    val   = df.iloc[train_end:val_end]
    test  = df.iloc[val_end:]

    print(f"Train : {len(train)} | Val : {len(val)} | Test : {len(test)}")
    print(f"Train period : {pd.to_datetime(train['timestamp'].min(), unit='ms')} → {pd.to_datetime(train['timestamp'].max(), unit='ms')}")
    print(f"Test period  : {pd.to_datetime(test['timestamp'].min(), unit='ms')} → {pd.to_datetime(test['timestamp'].max(), unit='ms')}")

    feature_cols = [c for c in df.columns if c not in ("timestamp", "label")]
    X_train, y_train = train[feature_cols], train["label"]
    X_val, y_val     = val[feature_cols],   val["label"]
    X_test, y_test   = test[feature_cols],  test["label"]

    print("\nTraining XGBoost...")
    # Régularisations optionnelles : si la valeur est False dans config.py,
    # on ne passe pas l'argument et XGBoost utilise son défaut.
    optional_kwargs = {}
    if XGB_REG_LAMBDA is not False:
        optional_kwargs["reg_lambda"] = XGB_REG_LAMBDA
    if XGB_REG_ALPHA is not False:
        optional_kwargs["reg_alpha"] = XGB_REG_ALPHA
    if XGB_GAMMA is not False:
        optional_kwargs["gamma"] = XGB_GAMMA

    model = xgb.XGBClassifier(
        n_estimators=XGB_N_ESTIMATORS,
        max_depth=XGB_MAX_DEPTH,
        learning_rate=XGB_LEARNING_RATE,
        min_child_weight=XGB_MIN_CHILD_WEIGHT,
        subsample=XGB_SUBSAMPLE,
        colsample_bytree=XGB_COLSAMPLE_BYTREE,
        objective="binary:logistic",
        eval_metric="logloss",
        early_stopping_rounds=XGB_EARLY_STOPPING_ROUNDS,
        random_state=XGB_RANDOM_STATE,
        n_jobs=-1,
        **optional_kwargs,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100,
    )

    # ========== ÉVALUATION ==========
    print("\n" + "="*70)
    print("  ÉVALUATION SUR LE TEST SET (out-of-sample)")
    print("="*70)

    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Baseline : tout le monde prend le pari
    baseline_wr = y_test.mean() * 100
    print(f"\nBaseline (filtre body_ratio seul) : {baseline_wr:.2f}% WR sur {len(y_test)} paris")

    print(f"\n{'Seuil':<8}{'Paris pris':<14}{'Win rate':<12}{'% volume conservé'}")
    print("-" * 55)
    for th in THRESHOLDS:
        mask = y_pred_proba >= th
        n_bets = mask.sum()
        if n_bets == 0:
            print(f"{th:<8}{'0':<14}{'N/A':<12}{'0.00%'}")
            continue
        wr = y_test[mask].mean() * 100
        pct_vol = n_bets / len(y_test) * 100
        print(f"{th:<8}{n_bets:<14}{wr:>6.2f}%{'':<5}{pct_vol:>6.2f}%")

    # Détails par longueur de séquence
    th = BREAKDOWN_THRESHOLD
    print(f"\n--- Breakdown par longueur de séquence (seuil={th}) ---")
    test_full = test.copy()
    test_full["pred_proba"] = y_pred_proba
    test_full["taken"] = test_full["pred_proba"] >= th
    total_taken = int(test_full["taken"].sum())
    total_wr = test_full[test_full["taken"]]["label"].mean() * 100 if total_taken > 0 else 0.0
    print(f"  TOTAL paris pris par l'IA : {total_taken} (WR = {total_wr:.2f}%)")
    for sl in sorted(test_full["seq_len"].unique()):
        sub = test_full[test_full["seq_len"] == sl]
        sub_taken = sub[sub["taken"]]
        if len(sub_taken) > 0:
            wr = sub_taken["label"].mean() * 100
            print(f"  Seq {sl}b : {len(sub_taken):>5} paris, WR = {wr:.2f}%")

    # Feature importance
    print("\n--- Feature importance (top 10) ---")
    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)
    print(importance.head(10).to_string(index=False))

    # Sauvegarde
    model.save_model(MODEL_OUTPUT)
    print(f"\nModel saved to {MODEL_OUTPUT}")

if __name__ == "__main__":
    main()