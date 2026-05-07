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
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from config import (
    CSV_PATH, MODEL_OUTPUT,
    TRAIN_DAYS, TEST_DAYS, VAL_FRACTION, SEQ_TEST, SYMBOL_TEST,
    SYMBOL_TO_ID,
    THRESHOLDS, BREAKDOWN_THRESHOLD,
    ADAPTIVE_THRESHOLD, ADAPTIVE_TARGET_WR, ADAPTIVE_CALIB_FRACTION,
    ADAPTIVE_MIN_N_VAL, ADAPTIVE_MIN_N_VAL_SOFT, ADAPTIVE_MIN_N_AT_THRESHOLD,
    ADAPTIVE_DELTA_CLAMP,
    XGB_N_ESTIMATORS, XGB_MAX_DEPTH, XGB_LEARNING_RATE,
    XGB_MIN_CHILD_WEIGHT, XGB_SUBSAMPLE, XGB_COLSAMPLE_BYTREE,
    XGB_REG_LAMBDA, XGB_REG_ALPHA, XGB_GAMMA,
    XGB_EARLY_STOPPING_ROUNDS, XGB_RANDOM_STATE,
)

# Grille de seuils candidats pour la calibration par bucket. Couvre [0.45, 0.75]
# pour permettre au sweep de "voir" des optima en dehors du clamp ; le clamp final
# est applique apres choix du threshold.
ADAPTIVE_TH_GRID = np.round(np.arange(0.45, 0.7501, 0.005), 4)

# Buckets fixes : 3 etats de volatilite x 2 valeurs de is_weekend = 6 buckets.
ADAPTIVE_BUCKETS = [(v, w) for v in (0, 1, 2) for w in (0, 1)]

# Fenetre de calib resolue : fraction de TEST_DAYS (ex: 0.15 * 365 = 54 jours).
# Conserve une borne basse de 1 jour pour eviter une fenetre vide si l'utilisateur
# met une fraction extremement petite avec un TEST_DAYS court.
ADAPTIVE_CALIB_DAYS = max(1, int(TEST_DAYS * ADAPTIVE_CALIB_FRACTION))

ROOT = Path(__file__).resolve().parent
LOGS_DIR = ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)
RUNS_CSV = LOGS_DIR / "runs.csv"
PREDICTIONS_CSV = LOGS_DIR / "test_predictions.csv"

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
#  Calibration du seuil adaptatif par bucket (volatility_state, is_weekend)
# ============================================================
def _compute_bucket_threshold(probas, labels, target_wr, base_th, min_n_at_th, clamp):
    """
    Choisit le plus petit threshold dont la WR realisee >= target_wr (avec n >= min_n_at_th).
    Si aucun threshold n'atteint la cible, fallback : threshold avec la WR maximale
    (toujours sous contrainte n >= min_n_at_th). Renvoie None si aucun candidat ne
    satisfait min_n_at_th (bucket trop sparse).
    """
    candidates = []
    for th in ADAPTIVE_TH_GRID:
        mask = probas >= th
        n = int(mask.sum())
        if n < min_n_at_th:
            continue
        wr = float(labels[mask].mean())
        candidates.append((float(th), n, wr))

    if not candidates:
        return None

    above_target = [c for c in candidates if c[2] >= target_wr]
    if above_target:
        chosen = min(above_target, key=lambda c: c[0])
        flag = "target_hit"
    else:
        chosen = max(candidates, key=lambda c: c[2])
        flag = "max_wr_fallback"

    th, n, wr = chosen
    delta_raw = th - base_th
    delta = max(clamp[0], min(clamp[1], delta_raw))
    return {
        "threshold_chosen": th,
        "threshold_clamped": base_th + delta,
        "n_at_th": n,
        "wr_at_th": wr,
        "delta_raw": float(delta_raw),
        "delta": float(delta),
        "source_flag": flag,
    }


def compute_adaptive_thresholds(
    val_df, y_pred_val, base_th,
    calib_days=ADAPTIVE_CALIB_DAYS,
    min_n_val=ADAPTIVE_MIN_N_VAL,
    min_n_val_soft=ADAPTIVE_MIN_N_VAL_SOFT,
    target_wr=ADAPTIVE_TARGET_WR,
    min_n_at_th=ADAPTIVE_MIN_N_AT_THRESHOLD,
    clamp=ADAPTIVE_DELTA_CLAMP,
):
    """
    Calcule un delta de seuil par bucket (volatility_state, is_weekend) sur les
    `calib_days` derniers jours de val. Si un bucket a < min_n_val rows dans la
    fenetre de calib, fallback sur l'integralite de val. Si toujours insuffisant,
    delta = 0 (fallback final = comportement BREAKDOWN_THRESHOLD pur).

    Renvoie (deltas_dict, report_list) :
      deltas_dict = {"vol_wknd": delta_float, ...} (cles "0_0" .. "2_1")
      report_list = liste de dicts par bucket pour pretty-print
    """
    if len(val_df) == 0 or "volatility_state" not in val_df.columns or "is_weekend" not in val_df.columns:
        return {}, []

    ms_per_day = 86_400 * 1000
    val_max_ts = int(val_df["timestamp"].max())
    calib_cutoff = val_max_ts - calib_days * ms_per_day

    timestamps = val_df["timestamp"].to_numpy()
    vol_arr = val_df["volatility_state"].astype(int).to_numpy()
    wknd_arr = val_df["is_weekend"].astype(int).to_numpy()
    labels = val_df["label"].to_numpy()
    probas = np.asarray(y_pred_val)

    calib_mask = timestamps >= calib_cutoff

    deltas = {}
    report = []
    for vol, wknd in ADAPTIVE_BUCKETS:
        key = f"{vol}_{wknd}"
        bucket_full = (vol_arr == vol) & (wknd_arr == wknd)
        bucket_calib = bucket_full & calib_mask
        n_calib = int(bucket_calib.sum())
        n_full = int(bucket_full.sum())

        # Cascade de fallback :
        #   1. n_calib >= MIN_N_VAL                  -> high confidence, fenetre recente
        #   2. n_full  >= MIN_N_VAL                  -> high confidence, full val (recence sacrifiee)
        #   3. n_full  >= MIN_N_VAL_SOFT             -> low confidence, full val (peu de data, on essaie quand meme)
        #   4. sinon                                 -> insufficient_data, delta = 0
        if n_calib >= min_n_val:
            use_mask, source, n_used = bucket_calib, "calib_window", n_calib
        elif n_full >= min_n_val:
            use_mask, source, n_used = bucket_full, "fallback_full_val", n_full
        elif n_full >= min_n_val_soft:
            use_mask, source, n_used = bucket_full, "low_conf_full_val", n_full
        else:
            deltas[key] = 0.0
            report.append({
                "bucket": (vol, wknd), "n_calib": n_calib, "n_full": n_full,
                "n_used": 0, "source_flag": "insufficient_data",
                "delta": 0.0, "delta_raw": 0.0,
                "threshold_chosen": base_th, "threshold_clamped": base_th,
                "n_at_th": 0, "wr_at_th": None,
            })
            continue

        res = _compute_bucket_threshold(
            probas[use_mask], labels[use_mask], target_wr, base_th, min_n_at_th, clamp,
        )
        if res is None:
            deltas[key] = 0.0
            report.append({
                "bucket": (vol, wknd), "n_calib": n_calib, "n_full": n_full,
                "n_used": n_used, "source_flag": f"{source}:no_candidates",
                "delta": 0.0, "delta_raw": 0.0,
                "threshold_chosen": base_th, "threshold_clamped": base_th,
                "n_at_th": 0, "wr_at_th": None,
            })
            continue

        deltas[key] = res["delta"]
        report.append({
            "bucket": (vol, wknd), "n_calib": n_calib, "n_full": n_full,
            "n_used": n_used, "source_flag": f"{source}:{res['source_flag']}",
            **res,
        })

    return deltas, report


def save_thresholds_sidecar(model_path, deltas, base_th, metadata=None):
    """Ecrit le sidecar JSON a cote du modele (btc_reversal_model.thresholds.json)."""
    sidecar = Path(model_path).with_suffix(".thresholds.json")
    payload = {
        "base_threshold": float(base_th),
        "deltas": {k: float(v) for k, v in deltas.items()},
        "metadata": metadata or {},
    }
    with open(sidecar, "w") as f:
        json.dump(payload, f, indent=2)
    return sidecar


def apply_adaptive_threshold(df, probas, deltas, base_th):
    """
    Applique les deltas par bucket aux probas et renvoie un mask boolean :
      take[i] = probas[i] >= base_th + deltas[bucket(i)]
    Si volatility_state / is_weekend manquent dans df, fallback : probas >= base_th.
    """
    probas = np.asarray(probas)
    if len(df) == 0 or not deltas:
        return probas >= base_th
    if "volatility_state" not in df.columns or "is_weekend" not in df.columns:
        return probas >= base_th

    vol = df["volatility_state"].astype(int).to_numpy()
    wknd = df["is_weekend"].astype(int).to_numpy()
    deltas_per_row = np.array(
        [deltas.get(f"{int(v)}_{int(w)}", 0.0) for v, w in zip(vol, wknd)],
        dtype=float,
    )
    return probas >= (base_th + deltas_per_row)


# ============================================================
#  Train + evaluate (fonction principale, appelable depuis sweep.py)
# ============================================================
def train_and_evaluate(
    overrides: dict | None = None,
    save_model: bool = True,
    verbose: bool = True,
    log_run: bool = True,
    run_label: str = "manual",
    log_predictions: bool = False,
) -> dict:
    """
    Entraine et evalue le modele.

    overrides       : dict d'hyperparams a surcharger sur ceux de config.py.
    save_model      : si True, ecrit le modele dans MODEL_OUTPUT (config).
    verbose         : si True, imprime l'evaluation complete (table seuils + breakdown + importance).
    log_run         : si True, append une ligne dans logs/runs.csv.
    run_label       : etiquette du run (visible dans runs.csv pour filtrer ulterieurement).
    log_predictions : si True, ecrit logs/test_predictions.csv (test set + colonne confidence).

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

    # Filtre optionnel sur le test set par seq_len (n'affecte ni train ni val).
    seq_test_active = SEQ_TEST if SEQ_TEST else None
    if seq_test_active:
        n_before = len(test)
        test = test[test["seq_len"].isin(seq_test_active)].reset_index(drop=True)
        if verbose:
            print(f"\nSEQ_TEST = {seq_test_active} : test filtre de {n_before:,} -> {len(test):,} rows")

    # Filtre optionnel sur le test set par symbole (n'affecte ni train ni val).
    # Silencieux si le CSV n'a pas de colonne symbol_id (mode single-pair legacy).
    symbol_test_active = SYMBOL_TEST if SYMBOL_TEST else None
    if symbol_test_active:
        if "symbol_id" not in test.columns:
            if verbose:
                print(f"\nSYMBOL_TEST = {symbol_test_active} ignore (CSV sans colonne symbol_id)")
            symbol_test_active = None
        else:
            target_ids = [SYMBOL_TO_ID[s] for s in symbol_test_active]
            n_before = len(test)
            test = test[test["symbol_id"].isin(target_ids)].reset_index(drop=True)
            if verbose:
                print(f"SYMBOL_TEST = {symbol_test_active} : test filtre de {n_before:,} -> {len(test):,} rows")

    if verbose:
        ignored = train_val_start
        print(f"\nSplit temporel : TRAIN_DAYS={TRAIN_DAYS}, TEST_DAYS={TEST_DAYS}, VAL_FRACTION={VAL_FRACTION}")
        print(f"  ignore (donnees plus vieilles que train) : {ignored:,} rows")
        print(f"  train : {len(train):,} rows  ({pd.to_datetime(train['timestamp'].min(), unit='ms').date()} -> {pd.to_datetime(train['timestamp'].max(), unit='ms').date()})")
        print(f"  val   : {len(val):,} rows  ({pd.to_datetime(val['timestamp'].min(), unit='ms').date()} -> {pd.to_datetime(val['timestamp'].max(), unit='ms').date()})")
        if len(test) > 0:
            print(f"  test  : {len(test):,} rows  ({pd.to_datetime(test['timestamp'].min(), unit='ms').date()} -> {pd.to_datetime(test['timestamp'].max(), unit='ms').date()})")
        else:
            print(f"  test  : 0 rows  (filtres : SEQ_TEST={seq_test_active}, SYMBOL_TEST={symbol_test_active} ne matchent aucune row)")

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

    # ---- Adaptive threshold calibration (sur val, jamais sur test) ----
    adaptive_deltas = {}
    adaptive_report = []
    if ADAPTIVE_THRESHOLD and len(val) > 0:
        y_pred_val = model.predict_proba(X_val)[:, 1]
        adaptive_deltas, adaptive_report = compute_adaptive_thresholds(
            val, y_pred_val, BREAKDOWN_THRESHOLD,
        )

    # ---- Test predictions ----
    y_pred = model.predict_proba(X_test)[:, 1]
    baseline_wr = float(y_test.mean())

    if log_predictions:
        _write_test_predictions(test, y_pred, PREDICTIONS_CSV)

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

    # ---- Evaluation adaptative sur test (head-to-head vs threshold fixe) ----
    if ADAPTIVE_THRESHOLD and adaptive_deltas:
        adaptive_take = apply_adaptive_threshold(test_full, y_pred, adaptive_deltas, BREAKDOWN_THRESHOLD)
        adaptive_total_taken = int(adaptive_take.sum())
        adaptive_total_wr = float(y_test[adaptive_take].mean()) if adaptive_total_taken > 0 else None
    else:
        adaptive_take = None
        adaptive_total_taken = 0
        adaptive_total_wr = None

    duration = time.time() - t0

    # ---- Save model + sidecar ----
    if save_model:
        model.save_model(MODEL_OUTPUT)
        if ADAPTIVE_THRESHOLD and adaptive_deltas:
            save_thresholds_sidecar(
                MODEL_OUTPUT, adaptive_deltas, BREAKDOWN_THRESHOLD,
                metadata={
                    "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "calib_fraction": ADAPTIVE_CALIB_FRACTION,
                    "calib_days_resolved": ADAPTIVE_CALIB_DAYS,
                    "test_days": TEST_DAYS,
                    "target_wr": ADAPTIVE_TARGET_WR,
                    "min_n_val": ADAPTIVE_MIN_N_VAL,
                    "min_n_at_threshold": ADAPTIVE_MIN_N_AT_THRESHOLD,
                    "clamp": list(ADAPTIVE_DELTA_CLAMP),
                },
            )

    # ---- Build metrics dict (tout flat, pour le CSV) ----
    metrics = {
        "timestamp_iso": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_label": run_label,
        "duration_sec": round(duration, 2),
        "csv_path": str(CSV_PATH),
        "train_days": TRAIN_DAYS,
        "test_days": TEST_DAYS,
        "val_fraction": VAL_FRACTION,
        "seq_test": ",".join(str(s) for s in seq_test_active) if seq_test_active else "all",
        "symbol_test": ",".join(symbol_test_active) if symbol_test_active else "all",
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

    # Adaptive threshold metrics (toujours ajoutees, NaN si feature off)
    metrics["adaptive_threshold_enabled"] = bool(ADAPTIVE_THRESHOLD)
    metrics["adaptive_calib_fraction"] = ADAPTIVE_CALIB_FRACTION
    metrics["adaptive_calib_days"] = ADAPTIVE_CALIB_DAYS
    metrics["adaptive_target_wr"] = ADAPTIVE_TARGET_WR
    metrics["adaptive_total_taken"] = adaptive_total_taken
    metrics["adaptive_total_wr"] = round(adaptive_total_wr, 5) if adaptive_total_wr is not None else None
    for vol, wknd in ADAPTIVE_BUCKETS:
        key = f"{vol}_{wknd}"
        metrics[f"delta_v{vol}_w{wknd}"] = round(adaptive_deltas[key], 5) if key in adaptive_deltas else None

    # ---- Print + log ----
    if verbose:
        _print_evaluation(metrics, breakdown_per_seq, model, feature_cols, train, test, y_pred, y_test)
        if ADAPTIVE_THRESHOLD:
            _print_adaptive_section(
                adaptive_report, adaptive_total_taken, adaptive_total_wr,
                metrics["breakdown_total_taken"], metrics["breakdown_total_wr"],
                len(y_test),
            )

    if log_run:
        append_to_runs_csv(metrics)

    return metrics


# ============================================================
#  Per-bet predictions CSV (ecrase a chaque run)
# ============================================================
def _write_test_predictions(test_df: pd.DataFrame, y_pred: np.ndarray, output_path: Path) -> None:
    """
    Ecrit le test set + une colonne 'confidence' dans output_path (ecrase).
    Une row = un setup evalue par le modele en phase de test.
    """
    out = test_df.copy()
    out["confidence"] = y_pred
    out.to_csv(output_path, index=False)


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


def _print_adaptive_section(report, ada_taken, ada_wr, fixed_taken, fixed_wr, n_test):
    """Print : (1) calibration par bucket, (2) head-to-head adaptatif vs fixe sur test."""
    print("\n" + "=" * 70)
    print(f"  CALIBRATION SEUIL ADAPTATIF (sur {ADAPTIVE_CALIB_DAYS}j de val "
          f"= {ADAPTIVE_CALIB_FRACTION:.0%} de TEST_DAYS={TEST_DAYS}, target_wr={ADAPTIVE_TARGET_WR})")
    print("=" * 70)
    if not report:
        print("  (aucun bucket calibre - val vide ou colonnes manquantes)")
        return

    print(f"  {'(vol, wknd)':<14}{'n_calib':<10}{'n_used':<10}{'th_pick':<10}"
          f"{'wr@th':<10}{'delta':<10}{'source'}")
    print("  " + "-" * 80)
    for r in report:
        vol, wknd = r["bucket"]
        wr_str = f"{r['wr_at_th']:.4f}" if r["wr_at_th"] is not None else "  N/A "
        print(f"  ({vol}, {wknd}){'':<7}{r['n_calib']:<10}{r['n_used']:<10}"
              f"{r['threshold_clamped']:<10.3f}{wr_str:<10}"
              f"{r['delta']:+.3f}    {r['source_flag']}")

    print("\n" + "=" * 70)
    print("  HEAD-TO-HEAD SUR TEST : seuil fixe vs seuil adaptatif")
    print("=" * 70)
    fixed_pct = fixed_taken / n_test * 100 if n_test else 0.0
    ada_pct = ada_taken / n_test * 100 if n_test else 0.0
    fixed_wr_str = f"{(fixed_wr or 0) * 100:.2f}%" if fixed_wr is not None else "N/A"
    ada_wr_str = f"{(ada_wr or 0) * 100:.2f}%" if ada_wr is not None else "N/A"
    print(f"  Fixe    (BREAKDOWN_THRESHOLD={BREAKDOWN_THRESHOLD}) : "
          f"{fixed_taken:>5} paris ({fixed_pct:.2f}% volume), WR = {fixed_wr_str}")
    print(f"  Adaptatif                                        : "
          f"{ada_taken:>5} paris ({ada_pct:.2f}% volume), WR = {ada_wr_str}")
    if fixed_wr is not None and ada_wr is not None:
        wr_delta = (ada_wr - fixed_wr) * 100
        n_delta = ada_taken - fixed_taken
        sign = "+" if wr_delta >= 0 else ""
        n_sign = "+" if n_delta >= 0 else ""
        print(f"  Delta                                            : "
              f"{n_sign}{n_delta} paris, {sign}{wr_delta:.2f}pp WR")


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
        log_predictions=True,
    )

    if not args.no_save:
        print(f"\nModel saved to {MODEL_OUTPUT}")
    if not args.no_log:
        print(f"Run logged to {RUNS_CSV}")
    print(f"Test predictions written to {PREDICTIONS_CSV}")


if __name__ == "__main__":
    main()
