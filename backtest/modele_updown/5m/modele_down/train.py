"""
Entrainement + evaluation du modele XGBoost DOWN-only.

Le modele predit P(next is down) (label=1 si la prochaine bougie est rouge).
Decision DOWN : take ssi P(down) >= seuil_effectif(row), avec seuil determine
par ADAPTIVE_STRATEGY (asym_wknd_excl V1 ou asym_wknd_excl_hour_boost V2).

Calibration par QUANTILE : T_axis = val_pool quantile (1 - ASYM_VOL_FRAC_axis).
Stable cross-window contrairement a l'ancienne target_wr/target_hit.

Usage CLI :
    python train.py
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
    DB_PATH, MODEL_OUTPUT,
    TRAIN_DAYS, TEST_DAYS, VAL_FRACTION,
    ADAPTIVE_STRATEGY,
    ASYM_VOL_FRAC_WEEKDAY, ASYM_VOL_FRAC_WEEKEND, ASYM_SKIP_CELLS,
    ASYM_USE_FULL_VAL,
    ASYM_HOUR_BOOST_GOOD_HOURS, ASYM_HOUR_BOOST_DELTA,
    XGB_N_ESTIMATORS, XGB_MAX_DEPTH, XGB_LEARNING_RATE,
    XGB_MIN_CHILD_WEIGHT, XGB_SUBSAMPLE, XGB_COLSAMPLE_BYTREE,
    XGB_REG_LAMBDA, XGB_REG_ALPHA, XGB_GAMMA,
    XGB_EARLY_STOPPING_ROUNDS, XGB_RANDOM_STATE,
    XGB_SCALE_POS_WEIGHT,
)
from features import build_training_dataframe

ROOT = Path(__file__).resolve().parent
LOGS_DIR = ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)
RUNS_CSV = LOGS_DIR / "runs.csv"
PREDICTIONS_CSV = LOGS_DIR / "test_predictions.csv"


# ============================================================
#  Hyperparams
# ============================================================
def build_hyperparams(overrides: dict | None = None) -> dict:
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
        "scale_pos_weight": XGB_SCALE_POS_WEIGHT,
        "early_stopping_rounds": XGB_EARLY_STOPPING_ROUNDS,
        "random_state": XGB_RANDOM_STATE,
    }
    if overrides:
        hp.update(overrides)
    return hp


def _resolve_scale_pos_weight(spw_cfg, y_train) -> float | None:
    """
    Resolve XGB_SCALE_POS_WEIGHT into a numeric value (or None to skip).
      False  -> None       (pas passé à XGBoost = comportement par défaut 1.0)
      "auto" -> n_up / n_down sur y_train (équilibrage exact des classes)
      number -> tel quel
    """
    if spw_cfg is False:
        return None
    if spw_cfg == "auto":
        n_pos = int((y_train == 1).sum())
        n_neg = int((y_train == 0).sum())
        if n_pos == 0:
            return 1.0
        return n_neg / n_pos
    if isinstance(spw_cfg, (int, float)) and not isinstance(spw_cfg, bool):
        return float(spw_cfg)
    raise ValueError(
        f"XGB_SCALE_POS_WEIGHT doit être False, 'auto' ou un nombre ; reçu {spw_cfg!r}"
    )


# ============================================================
#  Sidecar writers
# ============================================================
def save_thresholds_sidecar_asym(model_path, thresholds_per_wknd: dict, skip_cells: list,
                                  metadata=None):
    """Sidecar V1 (asym_wknd_excl) : 2 seuils (weekday/weekend) + skip cells."""
    sidecar = Path(model_path).with_suffix(".thresholds.json")
    payload = {
        "strategy": "asym_wknd_excl",
        "thresholds_per_wknd": {str(int(k)): float(v) for k, v in thresholds_per_wknd.items()},
        "skip_cells": [[int(v), int(w)] for (v, w) in skip_cells],
        "metadata": metadata or {},
    }
    with open(sidecar, "w") as f:
        json.dump(payload, f, indent=2)
    return sidecar


def save_thresholds_sidecar_asym_hour_boost(model_path, thresholds_per_wknd: dict,
                                             skip_cells: list, good_hours: tuple,
                                             hour_boost_delta: float, metadata=None):
    """Sidecar V2 (asym_wknd_excl_hour_boost) : V1 + champ optional `hour_boost`."""
    sidecar = Path(model_path).with_suffix(".thresholds.json")
    payload = {
        "strategy": "asym_wknd_excl_hour_boost",
        "thresholds_per_wknd": {str(int(k)): float(v) for k, v in thresholds_per_wknd.items()},
        "skip_cells": [[int(v), int(w)] for (v, w) in skip_cells],
        "hour_boost": {
            "good_hours": [int(h) for h in good_hours],
            "delta": float(hour_boost_delta),
        },
        "metadata": metadata or {},
    }
    with open(sidecar, "w") as f:
        json.dump(payload, f, indent=2)
    return sidecar


# ============================================================
#  Hour-of-day helper (UTC, derive du timestamp en ms)
# ============================================================
def hour_of_day_from_ms(ts_ms: np.ndarray) -> np.ndarray:
    """Renvoie l'heure UTC (0..23) pour chaque timestamp ms.
    Compatible avec les bougies 5m (timestamp = open time, aligne sur 5m).
    """
    ts_ms = np.asarray(ts_ms, dtype=np.int64)
    return ((ts_ms // (5 * 60 * 1000)) % 288 // 12).astype(int)


# ============================================================
#  Asymmetric weekday/weekend strategy with cell exclusions (V1)
# ============================================================
def compute_thresholds_asym_wknd_excl(
    val_df,
    p_down_val,
    vol_frac_wd: float = ASYM_VOL_FRAC_WEEKDAY,
    vol_frac_we: float = ASYM_VOL_FRAC_WEEKEND,
    skip_cells=ASYM_SKIP_CELLS,
    use_full_val: bool = ASYM_USE_FULL_VAL,
    calib_days: int = 55,
):
    """Calibration par quantile :
      T_axis = val_pool_quantile(1 - vol_frac_axis)
             = on garde le top vol_frac_axis du val non-skip de l'axe.

    Stable cross-window car le quantile s'auto-ajuste a la distribution
    p_down de la fenetre val.

    Renvoie :
        thresholds_per_wknd : {0: T_wd, 1: T_we}
        report : list[dict]
    """
    skip_set = set((int(v), int(w)) for (v, w) in skip_cells)

    if len(val_df) == 0 or "volatility_state" not in val_df.columns or "is_weekend" not in val_df.columns:
        return {0: 1.0, 1: 1.0}, []

    if use_full_val:
        mask = np.ones(len(val_df), dtype=bool)
    else:
        ms_per_day = 86_400 * 1000
        val_max_ts = int(val_df["timestamp"].max())
        cutoff = val_max_ts - calib_days * ms_per_day
        mask = val_df["timestamp"].to_numpy() >= cutoff

    probas = np.asarray(p_down_val, dtype=float)
    labels = val_df["label"].to_numpy().astype(int)
    vol_arr = val_df["volatility_state"].astype(int).to_numpy()
    wknd_arr = val_df["is_weekend"].astype(int).to_numpy()

    not_skipped = np.ones(len(val_df), dtype=bool)
    for (v, w) in skip_set:
        not_skipped &= ~((vol_arr == v) & (wknd_arr == w))
    cal_mask = mask & not_skipped

    thresholds_per_wknd = {}
    report = []
    for wknd_axis, vol_frac in ((0, vol_frac_wd), (1, vol_frac_we)):
        axis_mask = cal_mask & (wknd_arr == wknd_axis)
        n_axis = int(axis_mask.sum())
        if n_axis < 30 or vol_frac <= 0:
            thresholds_per_wknd[wknd_axis] = 1.0
            report.append({
                "axis": "weekday" if wknd_axis == 0 else "weekend",
                "wknd": wknd_axis, "vol_frac": vol_frac, "n_val": n_axis,
                "T": None, "wr_val_at_T": None, "n_at_T": 0,
                "source": "insufficient_data" if n_axis < 30 else "vol_frac=0",
            })
            continue

        pb = probas[axis_mask]
        yb = labels[axis_mask]
        T = float(np.quantile(pb, 1.0 - vol_frac))
        bet = pb >= T
        n_at_T = int(bet.sum())
        wr_val = float(yb[bet].mean()) if n_at_T > 0 else 0.0

        thresholds_per_wknd[wknd_axis] = T
        report.append({
            "axis": "weekday" if wknd_axis == 0 else "weekend",
            "wknd": wknd_axis, "vol_frac": vol_frac, "n_val": n_axis,
            "T": T, "wr_val_at_T": wr_val, "n_at_T": n_at_T, "source": "quantile",
        })

    return thresholds_per_wknd, report


def apply_thresholds_asym_wknd_excl(df, p_down, thresholds_per_wknd: dict,
                                     skip_cells=ASYM_SKIP_CELLS) -> np.ndarray:
    """V1 application : take[i] = (cell pas dans skip) AND (p_down[i] >= T_wknd(i))."""
    p_down = np.asarray(p_down, dtype=float)
    if len(df) == 0:
        return np.zeros(0, dtype=bool)
    if "volatility_state" not in df.columns or "is_weekend" not in df.columns:
        return np.zeros(len(p_down), dtype=bool)

    skip_set = set((int(v), int(w)) for (v, w) in skip_cells)
    vol = df["volatility_state"].astype(int).to_numpy()
    wknd = df["is_weekend"].astype(int).to_numpy()

    take = np.zeros(len(p_down), dtype=bool)
    for i in range(len(p_down)):
        if (int(vol[i]), int(wknd[i])) in skip_set:
            continue
        T = thresholds_per_wknd.get(int(wknd[i]), 1.0)
        if p_down[i] >= T:
            take[i] = True
    return take


def apply_thresholds_asym_hour_boost(df, p_down, thresholds_per_wknd: dict,
                                      skip_cells=ASYM_SKIP_CELLS,
                                      good_hours=ASYM_HOUR_BOOST_GOOD_HOURS,
                                      hour_boost_delta: float = ASYM_HOUR_BOOST_DELTA) -> np.ndarray:
    """V2 application : V1 thresholds + boost horaire.

    Pour les rows dont l'heure UTC ∈ good_hours, le seuil effectif est abaisse
    de `hour_boost_delta` (skip cells non boostees).
    """
    p_down = np.asarray(p_down, dtype=float)
    if len(df) == 0:
        return np.zeros(0, dtype=bool)
    if "volatility_state" not in df.columns or "is_weekend" not in df.columns:
        return np.zeros(len(p_down), dtype=bool)
    if "timestamp" not in df.columns:
        return np.zeros(len(p_down), dtype=bool)

    skip_set = set((int(v), int(w)) for (v, w) in skip_cells)
    good_hours_set = set(int(h) for h in good_hours)
    vol = df["volatility_state"].astype(int).to_numpy()
    wknd = df["is_weekend"].astype(int).to_numpy()
    hour = hour_of_day_from_ms(df["timestamp"].to_numpy())

    take = np.zeros(len(p_down), dtype=bool)
    for i in range(len(p_down)):
        if (int(vol[i]), int(wknd[i])) in skip_set:
            continue
        T = thresholds_per_wknd.get(int(wknd[i]), 1.0)
        if int(hour[i]) in good_hours_set:
            T = T - hour_boost_delta
        if p_down[i] >= T:
            take[i] = True
    return take


# ============================================================
#  Train + evaluate
# ============================================================
def train_and_evaluate(
    overrides: dict | None = None,
    save_model: bool = True,
    verbose: bool = True,
    log_run: bool = True,
    run_label: str = "manual",
    log_predictions: bool = False,
) -> dict:
    t0 = time.time()

    # ---- Load + split ----
    df = build_training_dataframe(DB_PATH, verbose=verbose)
    if df.empty:
        raise RuntimeError("Aucune row produite par build_training_dataframe.")
    df = df.sort_values("timestamp").reset_index(drop=True)

    ms_per_day = 86_400 * 1000
    ts_max = df["timestamp"].max()

    test_cutoff_ts = ts_max - TEST_DAYS * ms_per_day
    test_start = int(df["timestamp"].searchsorted(test_cutoff_ts, side="left"))

    if TRAIN_DAYS is None:
        train_val_start = 0
    else:
        train_val_cutoff_ts = test_cutoff_ts - TRAIN_DAYS * ms_per_day
        train_val_start = int(df["timestamp"].searchsorted(train_val_cutoff_ts, side="left"))

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
        print(f"  ignore : {ignored:,} rows")
        print(f"  train  : {len(train):,} rows  ({pd.to_datetime(train['timestamp'].min(), unit='ms').date()} -> {pd.to_datetime(train['timestamp'].max(), unit='ms').date()})")
        print(f"  val    : {len(val):,} rows  ({pd.to_datetime(val['timestamp'].min(), unit='ms').date()} -> {pd.to_datetime(val['timestamp'].max(), unit='ms').date()})")
        print(f"  test   : {len(test):,} rows  ({pd.to_datetime(test['timestamp'].min(), unit='ms').date()} -> {pd.to_datetime(test['timestamp'].max(), unit='ms').date()})")

    # volatility_state reste dans le parquet (cle de bucket pour le seuil
    # adaptatif) mais n'est PAS donne au modele en feature : delta_logloss
    # negatif lors de l'analyse, le modele s'en sert tres peu.
    feature_cols = [c for c in df.columns if c not in ("timestamp", "label", "volatility_state")]
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

    # Resolve scale_pos_weight ("auto" -> numeric ratio computed sur y_train)
    spw_resolved = _resolve_scale_pos_weight(hp["scale_pos_weight"], y_train)
    if spw_resolved is not None:
        optional_kwargs["scale_pos_weight"] = spw_resolved
        # On stocke la valeur résolue dans hp pour qu'elle apparaisse dans
        # logs/runs.csv et dans le print Hyperparams.
        hp["scale_pos_weight"] = round(float(spw_resolved), 6)
        if verbose:
            n_pos = int((y_train == 1).sum())
            n_neg = int((y_train == 0).sum())
            print(f"  scale_pos_weight = {spw_resolved:.4f} "
                  f"(n_up={n_neg:,} / n_down={n_pos:,})")

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
        print(f"\nTraining XGBoost DOWN-only "
              f"({len(train):,} train / {len(val):,} val / {len(test):,} test)...")
        print(f"  Hyperparams : {hp}")

    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100 if verbose else False)

    best_iter = int(model.best_iteration)
    val_logloss_best = float(model.evals_result()["validation_0"]["logloss"][best_iter])

    if ADAPTIVE_STRATEGY not in ("asym_wknd_excl", "asym_wknd_excl_hour_boost"):
        raise ValueError(
            f"ADAPTIVE_STRATEGY={ADAPTIVE_STRATEGY!r} non supporte. "
            f"Choix valides : 'asym_wknd_excl' (V1) ou 'asym_wknd_excl_hour_boost' (V2)."
        )

    # ---- Calibration adaptative (asym pooled par quantile sur val) ----
    thresholds_per_wknd: dict = {}
    report: list[dict] = []
    if len(val) > 0:
        p_down_val = model.predict_proba(X_val)[:, 1]
        thresholds_per_wknd, report = compute_thresholds_asym_wknd_excl(val, p_down_val)

    # ---- Test predictions ----
    p_down_test = model.predict_proba(X_test)[:, 1]
    baseline_wr_down = float(y_test.mean())

    # ---- Decision sur test (V1 ou V2) ----
    if ADAPTIVE_STRATEGY == "asym_wknd_excl_hour_boost":
        take_test = apply_thresholds_asym_hour_boost(test, p_down_test, thresholds_per_wknd)
    else:
        take_test = apply_thresholds_asym_wknd_excl(test, p_down_test, thresholds_per_wknd)
    label_arr = y_test.to_numpy().astype(int)

    n_test = len(test)
    n_total = int(take_test.sum())
    wr_total = float(label_arr[take_test].mean()) if n_total > 0 else None

    if log_predictions:
        _write_test_predictions(test, p_down_test, take_test, PREDICTIONS_CSV)

    duration = time.time() - t0

    # ---- Save model + sidecar ----
    if save_model:
        model.save_model(MODEL_OUTPUT)
        common_metadata = {
            "trained_at":   datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "test_days":    TEST_DAYS,
            "model_kind":   "down_only",
        }
        calib_meta = {
            "calib_method":      "quantile",
            "vol_frac_weekday":  ASYM_VOL_FRAC_WEEKDAY,
            "vol_frac_weekend":  ASYM_VOL_FRAC_WEEKEND,
            "use_full_val":      ASYM_USE_FULL_VAL,
        }
        if ADAPTIVE_STRATEGY == "asym_wknd_excl_hour_boost":
            save_thresholds_sidecar_asym_hour_boost(
                MODEL_OUTPUT, thresholds_per_wknd, list(ASYM_SKIP_CELLS),
                ASYM_HOUR_BOOST_GOOD_HOURS, ASYM_HOUR_BOOST_DELTA,
                metadata={
                    **common_metadata,
                    **calib_meta,
                    "hour_boost_good_hours": list(ASYM_HOUR_BOOST_GOOD_HOURS),
                    "hour_boost_delta":      ASYM_HOUR_BOOST_DELTA,
                },
            )
        else:
            save_thresholds_sidecar_asym(
                MODEL_OUTPUT, thresholds_per_wknd, list(ASYM_SKIP_CELLS),
                metadata={**common_metadata, **calib_meta},
            )

    # ---- Metrics ----
    metrics = {
        "timestamp_iso":     datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_label":         run_label,
        "duration_sec":      round(duration, 2),
        "db_path":           str(DB_PATH),
        "train_days":        TRAIN_DAYS,
        "test_days":         TEST_DAYS,
        "val_fraction":      VAL_FRACTION,
        "n_features":        len(feature_cols),
        "n_train":           len(train),
        "n_val":             len(val),
        "n_test":            n_test,
        "best_iteration":    best_iter,
        "val_logloss_best":  round(val_logloss_best, 6),
        "baseline_wr_down":  round(baseline_wr_down, 5),
    }
    for k, v in hp.items():
        metrics[f"hp_{k}"] = v

    metrics["adaptive_strategy"]    = ADAPTIVE_STRATEGY
    metrics["adaptive_total_taken"] = n_total
    metrics["adaptive_total_wr"]    = round(wr_total, 5) if wr_total is not None else None
    metrics["calib_method"]    = "quantile"
    metrics["vol_frac_weekday"] = ASYM_VOL_FRAC_WEEKDAY
    metrics["vol_frac_weekend"] = ASYM_VOL_FRAC_WEEKEND
    metrics["asym_T_weekday"]  = round(thresholds_per_wknd.get(0, 1.0), 5) if thresholds_per_wknd else None
    metrics["asym_T_weekend"]  = round(thresholds_per_wknd.get(1, 1.0), 5) if thresholds_per_wknd else None
    metrics["asym_skip_cells"] = str(list(ASYM_SKIP_CELLS))
    if ADAPTIVE_STRATEGY == "asym_wknd_excl_hour_boost":
        metrics["asym_hour_boost_good_hours"] = str(list(ASYM_HOUR_BOOST_GOOD_HOURS))
        metrics["asym_hour_boost_delta"]      = ASYM_HOUR_BOOST_DELTA

    # ---- Print + log ----
    if verbose:
        _print_evaluation(metrics, model, feature_cols)
        _print_adaptive_section(report)
        _print_bet_samples(test, p_down_test, take_test, label_arr, n_samples=5)

    if log_run:
        append_to_runs_csv(metrics)

    return metrics


# ============================================================
#  Per-bet predictions CSV
# ============================================================
def _write_test_predictions(
    test_df: pd.DataFrame,
    p_down: np.ndarray,
    take: np.ndarray,
    output_path: Path,
) -> None:
    """
    Ecrit le test set + 2 colonnes :
      - p_down     : probabilité brute "next is down" (∈ [0, 1])
      - confidence : alias de p_down (= score utilisé par le seuil adaptatif)

    Le label brut (1 = next is down) est déjà dans la colonne `label`.
    Direction est implicitement "down" pour tous les paris pris.
    """
    out = test_df.copy()
    out["p_down"]     = p_down
    out["confidence"] = p_down
    out.to_csv(output_path, index=False)


# ============================================================
#  CSV logging
# ============================================================
def append_to_runs_csv(metrics: dict) -> None:
    new_row = pd.DataFrame([metrics])
    if RUNS_CSV.exists():
        existing = pd.read_csv(RUNS_CSV)
        all_cols = list(existing.columns) + [c for c in new_row.columns if c not in existing.columns]
        existing = existing.reindex(columns=all_cols)
        new_row = new_row.reindex(columns=all_cols)
        df = pd.concat([existing, new_row], ignore_index=True)
    else:
        df = new_row
    df.to_csv(RUNS_CSV, index=False)


# ============================================================
#  Pretty-print
# ============================================================
def _print_evaluation(metrics, model, feature_cols):
    print("\n" + "=" * 70)
    print("  EVALUATION SUR LE TEST SET (DOWN-only, décision adaptative)")
    print("=" * 70)
    print(f"\nBaseline P(next is down) sur le test set : "
          f"{metrics['baseline_wr_down'] * 100:.2f}% (sur {metrics['n_test']} bougies)")

    n_total = metrics["adaptive_total_taken"]
    pct = n_total / metrics["n_test"] * 100 if metrics["n_test"] else 0.0
    fmt_wr = lambda x: f"{x * 100:.2f}%" if x is not None else "N/A"

    print(f"\n--- Décision adaptative ---")
    print(f"  Paris DOWN pris : {n_total} ({pct:.2f}% volume), "
          f"WR = {fmt_wr(metrics['adaptive_total_wr'])}")

    print("\n--- Feature importance (top 10) ---")
    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    print(importance.head(10).to_string(index=False))

    print(f"\nDuration : {metrics['duration_sec']}s | best_iteration : {metrics['best_iteration']} "
          f"| val_logloss : {metrics['val_logloss_best']}")


def _print_adaptive_section(report):
    print("\n" + "=" * 70)
    name = "ASYM_WKND_EXCL_HOUR_BOOST (V2)" if ADAPTIVE_STRATEGY == "asym_wknd_excl_hour_boost" else "ASYM_WKND_EXCL (V1)"
    print(f"  CALIBRATION {name} (DOWN-only) - methode quantile")
    print(f"  vol_frac_weekday={ASYM_VOL_FRAC_WEEKDAY}  vol_frac_weekend={ASYM_VOL_FRAC_WEEKEND}")
    print(f"  skip_cells={list(ASYM_SKIP_CELLS)}  use_full_val={ASYM_USE_FULL_VAL}")
    if ADAPTIVE_STRATEGY == "asym_wknd_excl_hour_boost":
        print(f"  HOUR BOOST: good_hours={list(ASYM_HOUR_BOOST_GOOD_HOURS)}  delta={ASYM_HOUR_BOOST_DELTA}")
    print("=" * 70)
    if not report:
        print("  (aucun seuil calibre - val vide ou colonnes manquantes)")
        return
    print(f"  {'axis':<10} {'vol_frac':<10} {'n_val':<8} {'T':<10} {'wr_val@T':<10} {'n_val@T':<8} {'source'}")
    print("  " + "-" * 72)
    for r in report:
        T_str = f"{r['T']:.4f}" if r["T"] is not None else "N/A"
        wr_str = f"{r['wr_val_at_T']:.4f}" if r["wr_val_at_T"] is not None else "N/A"
        print(f"  {r['axis']:<10} {r['vol_frac']:<10.4f} {r['n_val']:<8} "
              f"{T_str:<10} {wr_str:<10} {r['n_at_T']:<8} {r['source']}")


def _print_bet_samples(test_df, p_down, take, label_raw, n_samples=5):
    """
    Affiche n_samples paris DOWN tirés au hasard parmi ceux pris sur le test
    set, avec leur résultat. Pas de seed : tirage différent à chaque run.
    """
    take_arr = np.asarray(take, dtype=bool)
    p_down_arr = np.asarray(p_down, dtype=float)
    label_arr = np.asarray(label_raw, dtype=int)
    timestamps_arr = test_df["timestamp"].to_numpy()

    bet_idx = np.where(take_arr)[0]

    print("\n" + "=" * 70)
    print(f"  ECHANTILLON ALEATOIRE - {n_samples} paris DOWN")
    print("=" * 70)

    # Moyennes de P(down) sur les paris effectivement pris :
    #   - all   : sur tous les paris pris
    #   - wins  : sur les paris gagnants (label = 1 = bougie rouge)
    #   - loss  : sur les paris perdants (label = 0 = bougie verte)
    win_mask = take_arr & (label_arr == 1)
    loss_mask = take_arr & (label_arr == 0)
    n_taken = int(take_arr.sum())
    n_wins = int(win_mask.sum())
    n_loss = int(loss_mask.sum())
    mean_p_taken = float(p_down_arr[take_arr].mean()) if n_taken > 0 else 0.0
    mean_p_wins = float(p_down_arr[win_mask].mean()) if n_wins > 0 else 0.0
    mean_p_loss = float(p_down_arr[loss_mask].mean()) if n_loss > 0 else 0.0
    print(f"\n  Moyenne P(down) sur les {n_taken} paris pris   : {mean_p_taken*100:.2f}%")
    print(f"  Moyenne P(down) sur les {n_wins:>5} paris gagnants : {mean_p_wins*100:.2f}%")
    print(f"  Moyenne P(down) sur les {n_loss:>5} paris perdants : {mean_p_loss*100:.2f}%  (delta = {(mean_p_wins - mean_p_loss)*100:+.2f}pp)")

    print(f"\n  Paris DOWN pris (sur {len(bet_idx)} au total) :")
    if len(bet_idx) == 0:
        print("    (aucun pari pris)")
        return

    rng = np.random.default_rng()
    sample_size = min(n_samples, len(bet_idx))
    sampled = rng.choice(bet_idx, size=sample_size, replace=False)
    sampled = np.sort(sampled)

    print(f"    {'date (UTC)':<22}{'P(down)':<14}{'resultat'}")
    print(f"    {'-' * 50}")
    for i in sampled:
        ts_str = pd.to_datetime(int(timestamps_arr[i]), unit="ms", utc=True) \
                   .strftime("%Y-%m-%d %H:%M")
        p = p_down_arr[i]
        ok = label_arr[i] == 1   # label=1 si la bougie suivante etait bien rouge
        res = "[OK] correct (down)" if ok else "[KO] rate (up)"
        print(f"    {ts_str:<22}{p*100:>6.2f}%       {res}")


# ============================================================
#  CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Train + evaluate XGBoost DOWN-only model.")
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
