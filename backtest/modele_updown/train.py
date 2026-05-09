"""
Entrainement + evaluation du modele XGBoost DOWN-only.

Le modèle est entraîné à prédire :
    label = 1 si la prochaine bougie est ROUGE (close < open)
            0 si verte
    -> sa sortie P(label=1) = P(next is down) = confidence DOWN

Décision adaptative (axe unique, identique à modele_15m) :
  - Le sidecar contient un seul `deltas` dict (1 delta par bucket vol×wknd).
  - On prend le pari DOWN ssi
        P(down) >= ADAPTIVE_TARGET_WR + delta(bucket)
  - Plus de logique UP/DOWN à arbitrer : tous les paris sont des paris DOWN.

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
    ADAPTIVE_TARGET_WR, ADAPTIVE_CALIB_FRACTION,
    ADAPTIVE_MIN_N_VAL, ADAPTIVE_MIN_N_VAL_SOFT, ADAPTIVE_MIN_N_AT_THRESHOLD,
    ADAPTIVE_DELTA_CLAMP,
    XGB_N_ESTIMATORS, XGB_MAX_DEPTH, XGB_LEARNING_RATE,
    XGB_MIN_CHILD_WEIGHT, XGB_SUBSAMPLE, XGB_COLSAMPLE_BYTREE,
    XGB_REG_LAMBDA, XGB_REG_ALPHA, XGB_GAMMA,
    XGB_EARLY_STOPPING_ROUNDS, XGB_RANDOM_STATE,
    XGB_SCALE_POS_WEIGHT,
)
from features import build_training_dataframe

# Grille de seuils candidats. La sortie du modèle est P(down) qu'on calibre
# directement comme un score de confiance.
ADAPTIVE_TH_GRID = np.round(np.arange(0.50, 0.7501, 0.005), 4)

ADAPTIVE_BUCKETS = [(v, w) for v in (0, 1, 2) for w in (0, 1)]
ADAPTIVE_CALIB_DAYS = max(1, int(TEST_DAYS * ADAPTIVE_CALIB_FRACTION))

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
#  Calibration du seuil adaptatif (axe unique : P(down))
# ============================================================
def _compute_bucket_threshold(probas, labels, target_wr, base_th, min_n_at_th, clamp):
    """
    Pour un bucket donné, calibre un threshold T en deux étapes :

    1. **target_hit** : si un threshold T atteint WR >= target_wr (avec
       n_at_T >= min_n_at_th), on prend le PLUS PETIT T qui satisfait
       la cible -> max de bets à WR >= target.

    2. **strict_fallback** : sinon, target_wr est inatteignable. On respecte
       la rigueur demandée en n'autorisant que les thresholds T >= base_th
       (= au moins aussi stricts que le seuil de base demandé). Parmi ceux-là,
       on prend celui à plus haute WR observée. Si aucun T >= base_th n'a
       assez de bets, on renvoie None et le bucket est skippé (0 pari).

    Cette logique garantit que **augmenter target_wr resserre toujours
    le threshold effectif**, jamais l'inverse.
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
        # Cas idéal : on prend le plus petit T qui atteint la cible WR.
        chosen = min(above_target, key=lambda c: c[0])
        flag = "target_hit"
    else:
        # Cible inatteignable : on n'accepte que les T >= base_th, càd
        # au moins aussi stricts que ce que l'utilisateur a demandé.
        strict_candidates = [c for c in candidates if c[0] >= base_th]
        if not strict_candidates:
            # Même base_th est trop strict pour avoir assez de bets.
            # On skippe le bucket entièrement.
            return None
        chosen = max(strict_candidates, key=lambda c: c[2])
        flag = "strict_fallback"

    th, n, wr = chosen
    delta_raw = th - base_th
    delta = max(clamp[0], min(clamp[1], delta_raw))
    return {
        "threshold_chosen":  th,
        "threshold_clamped": base_th + delta,
        "n_at_th":           n,
        "wr_at_th":          wr,
        "delta_raw":         float(delta_raw),
        "delta":             float(delta),
        "source_flag":       flag,
    }


def compute_adaptive_thresholds(
    val_df, p_down_val, base_th,
    calib_days=ADAPTIVE_CALIB_DAYS,
    min_n_val=ADAPTIVE_MIN_N_VAL,
    min_n_val_soft=ADAPTIVE_MIN_N_VAL_SOFT,
    target_wr=ADAPTIVE_TARGET_WR,
    min_n_at_th=ADAPTIVE_MIN_N_AT_THRESHOLD,
    clamp=ADAPTIVE_DELTA_CLAMP,
):
    """
    Calibre un delta par bucket (volatility_state, is_weekend) directement
    sur P(down) du val set.
    """
    if len(val_df) == 0 or "volatility_state" not in val_df.columns or "is_weekend" not in val_df.columns:
        return {}, []

    probas = np.asarray(p_down_val, dtype=float)
    labels = val_df["label"].to_numpy().astype(int)

    ms_per_day = 86_400 * 1000
    val_max_ts = int(val_df["timestamp"].max())
    calib_cutoff = val_max_ts - calib_days * ms_per_day

    timestamps = val_df["timestamp"].to_numpy()
    vol_arr = val_df["volatility_state"].astype(int).to_numpy()
    wknd_arr = val_df["is_weekend"].astype(int).to_numpy()
    calib_mask = timestamps >= calib_cutoff

    deltas: dict[str, float] = {}
    report: list[dict] = []
    for vol, wknd in ADAPTIVE_BUCKETS:
        key = f"{vol}_{wknd}"
        bucket_full = (vol_arr == vol) & (wknd_arr == wknd)
        bucket_calib = bucket_full & calib_mask
        n_calib = int(bucket_calib.sum())
        n_full = int(bucket_full.sum())

        if n_calib >= min_n_val:
            use_mask, source, n_used = bucket_calib, "calib_window", n_calib
        elif n_full >= min_n_val:
            use_mask, source, n_used = bucket_full, "fallback_full_val", n_full
        elif n_full >= min_n_val_soft:
            use_mask, source, n_used = bucket_full, "low_conf_full_val", n_full
        else:
            deltas[key] = 0.0
            report.append({
                "bucket": (vol, wknd),
                "n_calib": n_calib, "n_full": n_full,
                "n_used": 0, "source_flag": "insufficient_data",
                "delta": 0.0, "delta_raw": 0.0,
                "threshold_chosen": base_th, "threshold_clamped": base_th,
                "n_at_th": 0, "wr_at_th": None,
            })
            continue

        res = _compute_bucket_threshold(
            probas[use_mask], labels[use_mask],
            target_wr, base_th, min_n_at_th, clamp,
        )
        if res is None:
            deltas[key] = 0.0
            report.append({
                "bucket": (vol, wknd),
                "n_calib": n_calib, "n_full": n_full,
                "n_used": n_used, "source_flag": f"{source}:no_candidates",
                "delta": 0.0, "delta_raw": 0.0,
                "threshold_chosen": base_th, "threshold_clamped": base_th,
                "n_at_th": 0, "wr_at_th": None,
            })
            continue

        deltas[key] = res["delta"]
        report.append({
            "bucket": (vol, wknd),
            "n_calib": n_calib, "n_full": n_full,
            "n_used": n_used, "source_flag": f"{source}:{res['source_flag']}",
            **res,
        })

    return deltas, report


def save_thresholds_sidecar(model_path, deltas, metadata=None):
    """Sidecar JSON : un seul dict `deltas` (1 delta par bucket vol×wknd).

    Format :
        {"deltas": {"vol_wknd": float, ...}, "metadata": {...}}
    """
    sidecar = Path(model_path).with_suffix(".thresholds.json")
    payload = {
        "deltas":   {k: float(v) for k, v in deltas.items()},
        "metadata": metadata or {},
    }
    with open(sidecar, "w") as f:
        json.dump(payload, f, indent=2)
    return sidecar


def apply_adaptive_threshold(df, p_down, deltas, base_th):
    """
    Renvoie un mask bool de même longueur que p_down :
      take[i] = P(down)[i] >= base_th + delta(bucket(i))

    Si volatility_state / is_weekend manquent : delta = 0 partout.
    """
    p_down = np.asarray(p_down, dtype=float)
    if len(df) == 0:
        return np.zeros(0, dtype=bool)
    if "volatility_state" not in df.columns or "is_weekend" not in df.columns:
        return p_down >= base_th

    vol = df["volatility_state"].astype(int).to_numpy()
    wknd = df["is_weekend"].astype(int).to_numpy()
    delta_per_row = np.array(
        [deltas.get(f"{int(v)}_{int(w)}", 0.0) for v, w in zip(vol, wknd)],
        dtype=float,
    )
    return p_down >= (base_th + delta_per_row)


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

    # ---- Calibration adaptative (sur val, axe unique = P(down)) ----
    deltas: dict[str, float] = {}
    report: list[dict] = []
    if len(val) > 0:
        p_down_val = model.predict_proba(X_val)[:, 1]
        deltas, report = compute_adaptive_thresholds(val, p_down_val, ADAPTIVE_TARGET_WR)

    # ---- Test predictions ----
    p_down_test = model.predict_proba(X_test)[:, 1]
    baseline_wr_down = float(y_test.mean())  # P(label=1=down) sur le test

    # ---- Décision adaptative sur test ----
    take_test = apply_adaptive_threshold(test, p_down_test, deltas, ADAPTIVE_TARGET_WR)
    label_arr = y_test.to_numpy().astype(int)

    n_test = len(test)
    n_total = int(take_test.sum())
    # WR = fraction des paris DOWN qui étaient corrects (= label=1=down)
    wr_total = float(label_arr[take_test].mean()) if n_total > 0 else None

    if log_predictions:
        _write_test_predictions(test, p_down_test, take_test, PREDICTIONS_CSV)

    duration = time.time() - t0

    # ---- Save model + sidecar ----
    if save_model:
        model.save_model(MODEL_OUTPUT)
        save_thresholds_sidecar(
            MODEL_OUTPUT, deltas,
            metadata={
                "trained_at":          datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "calib_fraction":      ADAPTIVE_CALIB_FRACTION,
                "calib_days_resolved": ADAPTIVE_CALIB_DAYS,
                "test_days":           TEST_DAYS,
                "target_wr":           ADAPTIVE_TARGET_WR,
                "min_n_val":           ADAPTIVE_MIN_N_VAL,
                "min_n_at_threshold":  ADAPTIVE_MIN_N_AT_THRESHOLD,
                "clamp":               list(ADAPTIVE_DELTA_CLAMP),
                "model_kind":          "down_only",
            },
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

    metrics["adaptive_calib_fraction"] = ADAPTIVE_CALIB_FRACTION
    metrics["adaptive_calib_days"]     = ADAPTIVE_CALIB_DAYS
    metrics["adaptive_target_wr"]      = ADAPTIVE_TARGET_WR
    metrics["adaptive_total_taken"]    = n_total
    metrics["adaptive_total_wr"]       = round(wr_total, 5) if wr_total is not None else None
    for vol, wknd in ADAPTIVE_BUCKETS:
        key = f"{vol}_{wknd}"
        metrics[f"delta_v{vol}_w{wknd}"] = round(deltas[key], 5) if key in deltas else None

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
    print(f"  CALIBRATION SEUIL ADAPTATIF (DOWN-only, calibre P(down))")
    print(f"  sur {ADAPTIVE_CALIB_DAYS}j de val "
          f"= {ADAPTIVE_CALIB_FRACTION:.0%} de TEST_DAYS={TEST_DAYS}, target_wr={ADAPTIVE_TARGET_WR}")
    print("=" * 70)
    if not report:
        print("  (aucun bucket calibré — val vide ou colonnes manquantes)")
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
    print(f"  ÉCHANTILLON ALÉATOIRE — {n_samples} paris DOWN")
    print("=" * 70)

    print(f"\n  Paris DOWN pris (sur {len(bet_idx)} au total) :")
    if len(bet_idx) == 0:
        print("    (aucun pari pris)")
        return

    rng = np.random.default_rng()
    sample_size = min(n_samples, len(bet_idx))
    sampled = rng.choice(bet_idx, size=sample_size, replace=False)
    sampled = np.sort(sampled)

    print(f"    {'date (UTC)':<22}{'P(down)':<14}{'résultat'}")
    print(f"    {'-' * 50}")
    for i in sampled:
        ts_str = pd.to_datetime(int(timestamps_arr[i]), unit="ms", utc=True) \
                   .strftime("%Y-%m-%d %H:%M")
        p = p_down_arr[i]
        ok = label_arr[i] == 1   # label=1 si la bougie suivante était bien rouge
        res = "✓ correct (down)" if ok else "✗ raté (up)"
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
