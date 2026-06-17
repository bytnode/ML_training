"""
Entrainement + evaluation du modele XGBoost UP-only.

Le modele predit P(next is up) (label=1 si la prochaine bougie est verte).
Decision UP : take ssi P(up) >= seuil_effectif(row), avec seuil determine
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
    DB_PATH, MODEL_OUTPUT, DROP_FEATURES, FEATURE_WHITELIST,
    TRAIN_DAYS, TEST_DAYS, VAL_FRACTION, BRIDGE_DAYS,
    ADAPTIVE_STRATEGY,
    ASYM_VOL_FRAC_WEEKDAY, ASYM_VOL_FRAC_WEEKEND, ASYM_SKIP_CELLS,
    ASYM_USE_FULL_VAL,
    ASYM_UNANIMOUS_WKND, ASYM_UNANIMOUS_THRESHOLD,
    ASYM_HOUR_BOOST_GOOD_HOURS, ASYM_HOUR_BOOST_DELTA,
    ADAPTIVE_BAD_BUCKETS, ADAPTIVE_BAD_WINDOW_DAYS,
    ADAPTIVE_N_BAD_HOURS, ADAPTIVE_N_SOFT_HOURS, ADAPTIVE_N_BAD_DAYS,
    ADAPTIVE_SOFT_HOURS_DELTA, ADAPTIVE_BAD_DAY_DELTA,
    ASYM_MAX_P_UP, ASYM_MODELE_UP_PREDS,
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
      "auto" -> n_down / n_up sur y_train (équilibrage exact des classes)
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
                                             hour_boost_delta: float,
                                             bad_hours: tuple = (),
                                             bad_hours_soft: tuple = (),
                                             bad_days: tuple = (),
                                             max_p_up_runtime: float = 1.0,
                                             metadata=None):
    """Sidecar V2 (asym_wknd_excl_hour_boost) : V1 + champ optional `hour_boost`
    + champ optional `bad_hours` (= heures UTC ou on ne parie jamais)
    + champ optional `max_p_up_runtime` (= filtre cross-model qui rejette
    le bet UP si modele_up's p_up >= ce seuil. 1.0 = filtre desactive)."""
    sidecar = Path(model_path).with_suffix(".thresholds.json")
    payload = {
        "strategy": "asym_wknd_excl_hour_boost",
        "thresholds_per_wknd": {str(int(k)): float(v) for k, v in thresholds_per_wknd.items()},
        "skip_cells": [[int(v), int(w)] for (v, w) in skip_cells],
        "hour_boost": {
            "good_hours": [int(h) for h in good_hours],
            "delta": float(hour_boost_delta),
        },
        "bad_hours": [int(h) for h in bad_hours],
        "bad_hours_soft": [int(h) for h in bad_hours_soft],
        "bad_days": [int(d) for d in bad_days],
        "max_p_up_runtime": float(max_p_up_runtime),
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
    Compatible avec les bougies 30m (timestamp = open time, aligne sur 30m).
    """
    ts_ms = np.asarray(ts_ms, dtype=np.int64)
    return ((ts_ms // (30 * 60 * 1000)) % 48 // 2).astype(int)


def isoweekday_from_ms(ts_ms: np.ndarray) -> np.ndarray:
    """1=Monday ... 7=Sunday (UTC isoweekday). Unix epoch 1970-01-01 was a
    Thursday (=4), so iso = ((days_since_epoch + 3) % 7) + 1."""
    ts_ms = np.asarray(ts_ms, dtype=np.int64)
    return ((ts_ms // 86_400_000 + 3) % 7 + 1).astype(int)


# ============================================================
#  Bad-hours / bad-days ADAPTATIFS (calcules au train, ecrits dans le sidecar)
# ============================================================
def compute_adaptive_bad_buckets(val_df, p_up_val, test_df, p_up_test,
                                  window_days: int = 365, n_bad_hours: int = 10,
                                  n_soft_hours: int = 7, n_bad_days: int = 2,
                                  bet_threshold=None, skip_cells=(), min_takes: int = 20):
    """Classe les heures (0-23) et les jours (isoweekday 1-7) par PERF REELLE DU
    SYSTEME sur les `window_days` derniers jours du dataset (val + test).

    DEUX MODES :
      - bet_threshold fourni (mode SYSTEME, defaut quand le seuil est connu) :
        on ne regarde QUE les paris que le systeme prendrait reellement, i.e.
        `p_up >= bet_threshold` ET la cellule (volatility_state, is_weekend) hors
        `skip_cells`. Le WR d'un bucket = part de paris GAGNANTS (label==1) parmi
        ces PARIS PRIS (= le WR reel du systeme dans ce bucket). On ne tient pas
        compte de la couche bad-hours elle-meme (c'est elle qu'on calcule).
        Un bucket avec < min_takes paris est NON CLASSABLE (donnees insuffisantes
        -> jamais classe "pire" : pas de skip a l'aveugle sur du bruit).
      - bet_threshold=None (mode DIRECTIONNEL, retrocompat) : ancien calcul
        p_up>=0.5 vs label sur TOUTES les bougies (modele nu).

    Renvoie (bad_hours, bad_hours_soft, bad_days, diag). Tri (WR croissant,
    index croissant) = deterministe.

      - bad_hours      : les n_bad_hours pires heures            -> skip dur
      - bad_hours_soft : les n_soft_hours heures suivantes        -> boost doux
      - bad_days       : les n_bad_days pires jours (isoweekday)  -> boost doux
    """
    frames = []
    for d, p in ((val_df, p_up_val), (test_df, p_up_test)):
        if d is None or len(d) == 0 or "timestamp" not in d.columns or "label" not in d.columns:
            continue
        f = pd.DataFrame({
            "timestamp": d["timestamp"].to_numpy(),
            "label": d["label"].astype(int).to_numpy(),
            "p_up": np.asarray(p, dtype=float),
        })
        f["volatility_state"] = (d["volatility_state"].to_numpy()
                                 if "volatility_state" in d.columns else -1)
        f["is_weekend"] = (d["is_weekend"].to_numpy()
                           if "is_weekend" in d.columns else -1)
        frames.append(f)
    if not frames:
        return (), (), (), None
    comb = pd.concat(frames, ignore_index=True)
    cutoff = int(comb["timestamp"].max()) - int(window_days) * 24 * 3600 * 1000
    comb = comb[comb["timestamp"] >= cutoff]
    if comb.empty:
        return (), (), (), None

    hours = hour_of_day_from_ms(comb["timestamp"].to_numpy())
    days = isoweekday_from_ms(comb["timestamp"].to_numpy())
    labels = comb["label"].to_numpy()
    p_up = comb["p_up"].to_numpy()
    mode = "takes" if bet_threshold is not None else "directional"

    if mode == "takes":
        # Paris reellement pris par le systeme (hors couche bad-hours/soft qu'on
        # determine ici) : p_up >= seuil ET cellule hors skip_cells.
        take = p_up >= float(bet_threshold)
        skip_set = {(int(v), int(w)) for (v, w) in skip_cells}
        if skip_set:
            vs = comb["volatility_state"].to_numpy()
            iw = comb["is_weekend"].to_numpy()
            in_skip = np.array([(int(a), int(b)) in skip_set for a, b in zip(vs, iw)])
            take = take & ~in_skip

        def stats(mask):
            sel = take & mask
            n = int(sel.sum())
            wr = float(labels[sel].mean()) if n > 0 else float("nan")  # WR = part gagnante des paris pris
            return wr, n
        floor = min_takes
    else:
        pred_up = (p_up >= 0.5).astype(int)
        correct = (pred_up == labels).astype(int)

        def stats(mask):
            n = int(mask.sum())
            wr = float(correct[mask].mean()) if n > 0 else float("nan")
            return wr, n
        floor = 1

    hour_wr, hour_n = {}, {}
    for h in range(24):
        hour_wr[h], hour_n[h] = stats(hours == h)
    rank_h = sorted([h for h in range(24) if hour_n[h] >= floor and not np.isnan(hour_wr[h])],
                    key=lambda h: (hour_wr[h], h))
    bad_hours = tuple(rank_h[:n_bad_hours])
    bad_hours_soft = tuple(rank_h[n_bad_hours:n_bad_hours + n_soft_hours])

    day_wr, day_n = {}, {}
    for dd in range(1, 8):
        day_wr[dd], day_n[dd] = stats(days == dd)
    rank_d = sorted([dd for dd in range(1, 8) if day_n[dd] >= floor and not np.isnan(day_wr[dd])],
                    key=lambda dd: (day_wr[dd], dd))
    bad_days = tuple(rank_d[:n_bad_days])

    diag = {"mode": mode, "hour_wr": hour_wr, "hour_n": hour_n,
            "day_wr": day_wr, "day_n": day_n, "window_days": int(window_days),
            "bet_threshold": (float(bet_threshold) if bet_threshold is not None else None),
            "min_takes": int(min_takes)}
    return bad_hours, bad_hours_soft, bad_days, diag


# ============================================================
#  Cross-model p_up loader (V3 extension)
# ============================================================
def _load_p_up_keep_mask(test_ts_ms: np.ndarray, preds_path: str,
                         max_p_up: float) -> np.ndarray:
    """Renvoie un mask boolean (1 par bar de `test_ts_ms`) :
       True ssi le modele_up a predit `p_up < max_p_up` sur le meme timestamp.
    Bars sans match (timestamp absent du CSV modele_up) sont REJETEs
    (mask = False) -- conservatif : pas de pari sans confirmation cross-model.
    """
    p_up_path = (ROOT / preds_path).resolve() if not Path(preds_path).is_absolute() \
                else Path(preds_path)
    if not p_up_path.exists():
        print(f"  WARN: modele_up preds not found at {p_up_path}, cross-model filter DISABLED")
        return np.ones(len(test_ts_ms), dtype=bool)
    up = pd.read_csv(p_up_path, usecols=["timestamp", "p_up"])
    # Map timestamp -> p_up.
    up_map = dict(zip(up["timestamp"].astype("int64").to_numpy(),
                       up["p_up"].astype(float).to_numpy()))
    keep = np.zeros(len(test_ts_ms), dtype=bool)
    n_missing = 0
    for i, ts in enumerate(test_ts_ms):
        p_up = up_map.get(int(ts))
        if p_up is None:
            n_missing += 1
            continue
        keep[i] = (p_up < max_p_up)
    if n_missing > 0:
        print(f"  cross-model: {n_missing}/{len(test_ts_ms)} test bars had no modele_up match (rejected)")
    return keep


# ============================================================
#  Asymmetric weekday/weekend strategy with cell exclusions (V1)
# ============================================================
def compute_thresholds_asym_wknd_excl(
    val_df,
    p_up_val,
    vol_frac_wd: float = ASYM_VOL_FRAC_WEEKDAY,
    vol_frac_we: float = ASYM_VOL_FRAC_WEEKEND,
    skip_cells=ASYM_SKIP_CELLS,
    use_full_val: bool = ASYM_USE_FULL_VAL,
    calib_days: int = 55,
    bad_hours: tuple = (),
):
    """Calibration par quantile :
      T_axis = val_pool_quantile(1 - vol_frac_axis)
             = on garde le top vol_frac_axis du val non-skip de l'axe.

    Stable cross-window car le quantile s'auto-ajuste a la distribution
    p_up de la fenetre val.

    `bad_hours` : heures UTC ou on ne parie jamais -> exclues du pool de
    calibration pour que le quantile reflete la distribution effective des
    bars sur lesquels on parie reellement.

    Renvoie :
        thresholds_per_wknd : {0: T_wd, 1: T_we}
        report : list[dict]
    """
    skip_set = set((int(v), int(w)) for (v, w) in skip_cells)
    bad_hours_set = set(int(h) for h in bad_hours)

    if len(val_df) == 0 or "volatility_state" not in val_df.columns or "is_weekend" not in val_df.columns:
        return {0: 1.0, 1: 1.0}, []

    if use_full_val:
        mask = np.ones(len(val_df), dtype=bool)
    else:
        ms_per_day = 86_400 * 1000
        val_max_ts = int(val_df["timestamp"].max())
        cutoff = val_max_ts - calib_days * ms_per_day
        mask = val_df["timestamp"].to_numpy() >= cutoff

    probas = np.asarray(p_up_val, dtype=float)
    labels = val_df["label"].to_numpy().astype(int)
    vol_arr = val_df["volatility_state"].astype(int).to_numpy()
    wknd_arr = val_df["is_weekend"].astype(int).to_numpy()
    hour_arr = hour_of_day_from_ms(val_df["timestamp"].to_numpy())

    not_skipped = np.ones(len(val_df), dtype=bool)
    for (v, w) in skip_set:
        not_skipped &= ~((vol_arr == v) & (wknd_arr == w))
    if bad_hours_set:
        bad_mask = np.isin(hour_arr, list(bad_hours_set))
        not_skipped &= ~bad_mask
    cal_mask = mask & not_skipped

    thresholds_per_wknd = {}
    report = []

    # --- Mode UNANIME HARDCODE : un seul seuil ecrit en dur (identique a up) ---
    # T = ASYM_UNANIMOUS_THRESHOLD pris tel quel (PAS de quantile). On l'assigne
    # aux deux axes -> sidecar {"0": T, "1": T}. La boucle par axe ci-dessous
    # (vol_frac_wd / vol_frac_we) n'est jamais executee : l'autre mode est OFF.
    # Les stats val (n_at_T, wr_val) sont purement informatives pour le report.
    if ASYM_UNANIMOUS_WKND:
        T = float(ASYM_UNANIMOUS_THRESHOLD)
        pb = probas[cal_mask]
        yb = labels[cal_mask]
        bet = pb >= T
        n_at_T = int(bet.sum())
        wr_val = float(yb[bet].mean()) if n_at_T > 0 else 0.0
        report.append({
            "axis": "unanimous", "wknd": -1, "vol_frac": None,
            "n_val": int(cal_mask.sum()), "T": T, "wr_val_at_T": wr_val,
            "n_at_T": n_at_T, "source": "hardcoded",
        })
        return {0: T, 1: T}, report

    for wknd_axis, vol_frac in ((0, vol_frac_wd), (1, vol_frac_we)):
        axis_mask = cal_mask & (wknd_arr == wknd_axis)
        n_axis = int(axis_mask.sum())
        # Detect "axis fully skipped" : all (vol, wknd_axis) cells are in
        # skip_set. This is the intentional case (e.g. skip-all-weekday) and
        # should be labeled as such -- NOT "insufficient_data" which suggests
        # a data bug.
        axis_cells_skipped = all((int(v), int(wknd_axis)) in skip_set for v in (0, 1, 2))
        if n_axis < 30 or vol_frac <= 0:
            thresholds_per_wknd[wknd_axis] = 1.0
            if axis_cells_skipped:
                source = "axis_fully_skipped"
            elif vol_frac <= 0:
                source = "vol_frac=0"
            else:
                source = "insufficient_data"
            report.append({
                "axis": "weekday" if wknd_axis == 0 else "weekend",
                "wknd": wknd_axis, "vol_frac": vol_frac, "n_val": n_axis,
                "T": None, "wr_val_at_T": None, "n_at_T": 0,
                "source": source,
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


def apply_thresholds_asym_wknd_excl(df, p_up, thresholds_per_wknd: dict,
                                     skip_cells=ASYM_SKIP_CELLS) -> np.ndarray:
    """V1 application : take[i] = (cell pas dans skip) AND (p_up[i] >= T_wknd(i))."""
    p_up = np.asarray(p_up, dtype=float)
    if len(df) == 0:
        return np.zeros(0, dtype=bool)
    if "volatility_state" not in df.columns or "is_weekend" not in df.columns:
        return np.zeros(len(p_up), dtype=bool)

    skip_set = set((int(v), int(w)) for (v, w) in skip_cells)
    vol = df["volatility_state"].astype(int).to_numpy()
    wknd = df["is_weekend"].astype(int).to_numpy()

    take = np.zeros(len(p_up), dtype=bool)
    for i in range(len(p_up)):
        if (int(vol[i]), int(wknd[i])) in skip_set:
            continue
        T = thresholds_per_wknd.get(int(wknd[i]), 1.0)
        if p_up[i] >= T:
            take[i] = True
    return take


def apply_thresholds_asym_hour_boost(df, p_up, thresholds_per_wknd: dict,
                                      skip_cells=ASYM_SKIP_CELLS,
                                      good_hours=ASYM_HOUR_BOOST_GOOD_HOURS,
                                      hour_boost_delta: float = ASYM_HOUR_BOOST_DELTA,
                                      bad_hours: tuple = (),
                                      bad_days: tuple = (),
                                      wr_bad_day: float = ADAPTIVE_BAD_DAY_DELTA,
                                      adaptive_bad_hours: tuple = (),
                                      bad_hours_delta: float = ADAPTIVE_SOFT_HOURS_DELTA) -> np.ndarray:
    """V2 application : V1 thresholds + boost horaire + bad_hours exclusion.

    Pour les rows dont l'heure UTC in good_hours, le seuil effectif est abaisse
    de `hour_boost_delta` (skip cells non boostees).
    Pour les rows dont l'heure UTC in bad_hours, take = False quel que soit
    le p_up (filtre dur d'heures losers, comme un skip_cells horaire).
    Soft penalty (ADDED to T, le pari peut encore etre pris si la confiance
    est assez haute) :
      - `bad_days` (isoweekday 1=Mon..7=Sun) -> T += wr_bad_day
    """
    p_up = np.asarray(p_up, dtype=float)
    if len(df) == 0:
        return np.zeros(0, dtype=bool)
    if "volatility_state" not in df.columns or "is_weekend" not in df.columns:
        return np.zeros(len(p_up), dtype=bool)
    if "timestamp" not in df.columns:
        return np.zeros(len(p_up), dtype=bool)

    skip_set = set((int(v), int(w)) for (v, w) in skip_cells)
    good_hours_set = set(int(h) for h in good_hours)
    bad_hours_set = set(int(h) for h in bad_hours)
    bad_days_set = set(int(d) for d in (bad_days or ()))
    adaptive_bad_hours_set = set(int(h) for h in (adaptive_bad_hours or ()))
    vol = df["volatility_state"].astype(int).to_numpy()
    wknd = df["is_weekend"].astype(int).to_numpy()
    ts_arr = df["timestamp"].to_numpy()
    hour = hour_of_day_from_ms(ts_arr)
    iso_day = isoweekday_from_ms(ts_arr)

    take = np.zeros(len(p_up), dtype=bool)
    for i in range(len(p_up)):
        if (int(vol[i]), int(wknd[i])) in skip_set:
            continue
        h_i = int(hour[i])
        if h_i in bad_hours_set:
            continue
        T = thresholds_per_wknd.get(int(wknd[i]), 1.0)
        if h_i in good_hours_set:
            T = T - hour_boost_delta
        if bad_days_set and int(iso_day[i]) in bad_days_set:
            T = T + wr_bad_day
        if adaptive_bad_hours_set and h_i in adaptive_bad_hours_set:
            T = T + bad_hours_delta
        if p_up[i] >= T:
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

    # Bridge (embargo) : pause de BRIDGE_DAYS jours entre la fin de (train+val) et
    # le debut du test. Les bougies de [val_end, test_start) sont DROPPED (ni val
    # ni test) -> coupe la fuite des features a lookback qui chevauchent le test.
    # BRIDGE_DAYS = 0 -> val colle au test (comportement historique).
    bridge_cutoff_ts = test_cutoff_ts - BRIDGE_DAYS * ms_per_day
    val_end = int(df["timestamp"].searchsorted(bridge_cutoff_ts, side="left"))

    n_train_val = val_end - train_val_start
    val_size = int(n_train_val * VAL_FRACTION)
    train_end = val_end - val_size

    train = df.iloc[train_val_start:train_end]
    val   = df.iloc[train_end:val_end]
    test  = df.iloc[test_start:]

    if verbose:
        ignored = train_val_start
        print(f"\nSplit temporel : TRAIN_DAYS={TRAIN_DAYS}, TEST_DAYS={TEST_DAYS}, VAL_FRACTION={VAL_FRACTION}, BRIDGE_DAYS={BRIDGE_DAYS}")
        print(f"  ignore : {ignored:,} rows")
        print(f"  train  : {len(train):,} rows  ({pd.to_datetime(train['timestamp'].min(), unit='ms').date()} -> {pd.to_datetime(train['timestamp'].max(), unit='ms').date()})")
        print(f"  val    : {len(val):,} rows  ({pd.to_datetime(val['timestamp'].min(), unit='ms').date()} -> {pd.to_datetime(val['timestamp'].max(), unit='ms').date()})")
        if BRIDGE_DAYS > 0:
            print(f"  bridge : {test_start - val_end:,} rows DROPPED (pause {BRIDGE_DAYS}j entre val et test)")
        print(f"  test   : {len(test):,} rows  ({pd.to_datetime(test['timestamp'].min(), unit='ms').date()} -> {pd.to_datetime(test['timestamp'].max(), unit='ms').date()})")

    # volatility_state reste dans le parquet (cle de bucket pour le seuil
    # adaptatif) mais n'est PAS donne au modele en feature : delta_logloss
    # negatif lors de l'analyse, le modele s'en sert tres peu.
    # is_weekend reste dans le parquet (bucket key pour le seuil adaptatif)
    # mais n'est PAS donne au modele en feature (mirror modele_up).
    # PARITE FEATURES 15min : si FEATURE_WHITELIST est non vide, feature_cols est
    # EXACTEMENT cette liste (ordre inclus). Les colonnes en trop du parquet sont
    # ignorees ; une colonne manquante leve une erreur claire. Resultat :
    # booster.feature_names == les 27 features du modele 15min.
    if FEATURE_WHITELIST:
        missing = [c for c in FEATURE_WHITELIST if c not in df.columns]
        if missing:
            raise RuntimeError(
                "FEATURE_WHITELIST : colonnes absentes du parquet "
                f"({len(missing)}/{len(FEATURE_WHITELIST)}) : {missing}. "
                "Re-run 'python data.py' puis 'python add_eth_features.py' "
                "(les ef_* viennent du patch ETH)."
            )
        feature_cols = [c for c in FEATURE_WHITELIST]
        if verbose:
            print(f"\nFEATURE_WHITELIST active : {len(feature_cols)} features "
                  f"(parite stricte modele 15min).")
    else:
        feature_cols = [c for c in df.columns
                        if c not in ("timestamp", "label", "volatility_state", "is_weekend")
                        and c not in DROP_FEATURES]
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
                  f"(n_down={n_neg:,} / n_up={n_pos:,})")

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
        print(f"\nTraining XGBoost UP-only "
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

    # ---- Predictions val + test (calculees AVANT la calibration : necessaires
    #      au calcul des bad-hours/bad-days adaptatifs sur la fenetre val+test) ----
    p_up_val = model.predict_proba(X_val)[:, 1] if len(val) > 0 else np.zeros(0)
    p_up_test = model.predict_proba(X_test)[:, 1]
    baseline_wr_up = float(y_test.mean())

    # ---- Bad-hours / bad-days ADAPTATIFS classes sur la PERF REELLE DU SYSTEME
    #      (WR des paris reellement pris : p_up >= seuil reel, hors skip_cells)
    #      sur les ADAPTIVE_BAD_WINDOW_DAYS derniers jours du dataset = val+test.
    #      Remplacent les tuples hardcodes du config quand ADAPTIVE_BAD_BUCKETS. ----
    if ADAPTIVE_BAD_BUCKETS:
        # Seuil de pari reel du systeme : en mode unanime hardcode c'est
        # ASYM_UNANIMOUS_THRESHOLD ; sinon None -> retombe sur le WR directionnel.
        bet_thr = ASYM_UNANIMOUS_THRESHOLD if ASYM_UNANIMOUS_WKND else None
        bad_hours_eff, bad_hours_soft_eff, bad_days_eff, adaptive_diag = compute_adaptive_bad_buckets(
            val, p_up_val, test, p_up_test,
            window_days=ADAPTIVE_BAD_WINDOW_DAYS,
            n_bad_hours=ADAPTIVE_N_BAD_HOURS,
            n_soft_hours=ADAPTIVE_N_SOFT_HOURS,
            n_bad_days=ADAPTIVE_N_BAD_DAYS,
            bet_threshold=bet_thr,
            skip_cells=ASYM_SKIP_CELLS,
        )
        if verbose:
            print(f"  [adaptive bad buckets] skip_hours={bad_hours_eff} "
                  f"soft_hours={bad_hours_soft_eff} bad_days={bad_days_eff}")
    else:
        bad_hours_eff = ()
        bad_hours_soft_eff = ()
        bad_days_eff = ()
        adaptive_diag = None

    # ---- Calibration adaptative (asym pooled par quantile sur val) ----
    thresholds_per_wknd: dict = {}
    report: list[dict] = []
    if len(val) > 0:
        thresholds_per_wknd, report = compute_thresholds_asym_wknd_excl(
            val, p_up_val, bad_hours=bad_hours_eff,
        )

    # ---- Decision sur test (V1 ou V2) ----
    if ADAPTIVE_STRATEGY == "asym_wknd_excl_hour_boost":
        take_test = apply_thresholds_asym_hour_boost(
            test, p_up_test, thresholds_per_wknd, bad_hours=bad_hours_eff,
            bad_days=bad_days_eff, adaptive_bad_hours=bad_hours_soft_eff,
        )
    else:
        take_test = apply_thresholds_asym_wknd_excl(test, p_up_test, thresholds_per_wknd)

    # ---- Cross-model filter (V3 extension) ----
    # Si ASYM_MAX_P_UP < 1.0, on lit modele_up's p_up sur les memes timestamps
    # et on n'engage QUE si p_up < ASYM_MAX_P_UP. Bars sans match (p_up
    # manquant) sont REJETES par defaut (= conservative : on ne parie pas
    # sans confirmation cross-model).
    n_pre_cross = int(take_test.sum())
    wr_pre_cross = float(y_test.to_numpy().astype(int)[take_test].mean()) if n_pre_cross > 0 else None
    if ASYM_MAX_P_UP < 1.0:
        cross_keep = _load_p_up_keep_mask(
            test["timestamp"].to_numpy(),
            preds_path=ASYM_MODELE_UP_PREDS,
            max_p_up=ASYM_MAX_P_UP,
        )
        take_test = take_test & cross_keep
        if verbose:
            n_post = int(take_test.sum())
            print(f"  cross-model filter: p_up < {ASYM_MAX_P_UP:.3f}  "
                  f"=> {n_pre_cross} -> {n_post} bets "
                  f"(WR pre={wr_pre_cross:.4f})" if wr_pre_cross else "")

    label_arr = y_test.to_numpy().astype(int)

    n_test = len(test)
    n_total = int(take_test.sum())
    wr_total = float(label_arr[take_test].mean()) if n_total > 0 else None

    if log_predictions:
        _write_test_predictions(test, p_up_test, take_test, PREDICTIONS_CSV)

    duration = time.time() - t0

    # ---- Save model + sidecar ----
    if save_model:
        model.save_model(MODEL_OUTPUT)
        common_metadata = {
            "trained_at":   datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "test_days":    TEST_DAYS,
            "model_kind":   "up_only",
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
                bad_hours=bad_hours_eff,
                bad_hours_soft=bad_hours_soft_eff,
                bad_days=bad_days_eff,
                max_p_up_runtime=ASYM_MAX_P_UP,
                metadata={
                    **common_metadata,
                    **calib_meta,
                    "hour_boost_good_hours": list(ASYM_HOUR_BOOST_GOOD_HOURS),
                    "hour_boost_delta":      ASYM_HOUR_BOOST_DELTA,
                    "bad_hours":             list(bad_hours_eff),
                    "bad_hours_soft":        list(bad_hours_soft_eff),
                    "bad_days":              list(bad_days_eff),
                    "max_p_up_runtime":      ASYM_MAX_P_UP,
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
        "baseline_wr_up":  round(baseline_wr_up, 5),
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
        metrics["asym_bad_hours"]             = str(list(bad_hours_eff))
        metrics["adaptive_bad_hours_soft"]    = str(list(bad_hours_soft_eff))
        metrics["adaptive_bad_days"]          = str(list(bad_days_eff))
        metrics["asym_max_p_up"]              = ASYM_MAX_P_UP

    # ---- Print + log ----
    if verbose:
        _print_evaluation(metrics, model, feature_cols)
        _print_adaptive_section(report)
        _print_bet_samples(test, p_up_test, take_test, label_arr, n_samples=5)
        _print_hourly_wr(test, take_test, label_arr, side="up",
                         bad_hours=bad_hours_eff, bad_hours_soft=bad_hours_soft_eff,
                         good_hours=ASYM_HOUR_BOOST_GOOD_HOURS, bad_days=bad_days_eff)
        _print_directional_wr_ranking(adaptive_diag, skip_hours=bad_hours_eff,
                                      soft_hours=bad_hours_soft_eff, bad_days=bad_days_eff)

    if log_run:
        append_to_runs_csv(metrics)

    return metrics


# ============================================================
#  Per-bet predictions CSV
# ============================================================
def _write_test_predictions(
    test_df: pd.DataFrame,
    p_up: np.ndarray,
    take: np.ndarray,
    output_path: Path,
) -> None:
    """
    Ecrit le test set + 2 colonnes :
      - p_up     : probabilité brute "next is up" (∈ [0, 1])
      - confidence : alias de p_up (= score utilisé par le seuil adaptatif)

    Le label brut (1 = next is up) est déjà dans la colonne `label`.
    Direction est implicitement "up" pour tous les paris pris.
    """
    out = test_df.copy()
    out["p_up"]     = p_up
    out["confidence"] = p_up
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
    print("  EVALUATION SUR LE TEST SET (UP-only, décision adaptative)")
    print("=" * 70)
    print(f"\nBaseline P(next is up) sur le test set : "
          f"{metrics['baseline_wr_up'] * 100:.2f}% (sur {metrics['n_test']} bougies)")

    n_total = metrics["adaptive_total_taken"]
    pct = n_total / metrics["n_test"] * 100 if metrics["n_test"] else 0.0
    fmt_wr = lambda x: f"{x * 100:.2f}%" if x is not None else "N/A"

    print(f"\n--- Décision adaptative ---")
    print(f"  Paris UP pris : {n_total} ({pct:.2f}% volume), "
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
    print(f"  CALIBRATION {name} (UP-only) - methode quantile")
    if ASYM_UNANIMOUS_WKND:
        print(f"  MODE UNANIME HARDCODE: T={ASYM_UNANIMOUS_THRESHOLD} pour weekday ET weekend "
              f"(mode par axe DESACTIVE)")
    else:
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
        vf_str = f"{r['vol_frac']:.4f}" if r["vol_frac"] is not None else "N/A"
        print(f"  {r['axis']:<10} {vf_str:<10} {r['n_val']:<8} "
              f"{T_str:<10} {wr_str:<10} {r['n_at_T']:<8} {r['source']}")


def _print_hourly_wr(test_df, take, label_raw, side: str = "up",
                     bad_hours=(), bad_hours_soft=(), good_hours=(), bad_days=()):
    """Affiche un tableau par heure UTC : n bets pris, n wins, WR, + la colonne
    Adapt. = l'adaptation de seuil que subit l'heure :
        SKIP dur  -> heure dans bad_hours (jamais de pari, T force a 1.0)
        +soft     -> heure dans bad_hours_soft (seuil releve de ADAPTIVE_SOFT_HOURS_DELTA)
        -boost    -> heure dans good_hours (seuil abaisse de ASYM_HOUR_BOOST_DELTA)
    bad_hours / hour_boost sont DEJA appliques dans take : on lit juste le take final.
    """
    take_arr = np.asarray(take, dtype=bool)
    label_arr = np.asarray(label_raw, dtype=int)
    ts_arr = test_df["timestamp"].to_numpy()
    hours = hour_of_day_from_ms(ts_arr)

    skip_set = set(int(h) for h in bad_hours)
    soft_set = set(int(h) for h in bad_hours_soft)
    good_set = set(int(h) for h in good_hours)

    def _adapt(h: int) -> str:
        if h in skip_set:
            return "SKIP dur"
        if h in good_set:
            return "-boost"
        if h in soft_set:
            return "+soft"
        return ""

    print("\n" + "=" * 82)
    print(f"  WR PAR HEURE UTC ({side.upper()}-only, paris pris sur le test)")
    print("=" * 82)
    print(f"  {'Heure':<8}{'Pris':>7}{'Wins':>8}{'WR':>10}{'% du total':>14}{'Adapt.':>12}")
    print(f"  {'-' * 59}")
    total_taken = int(take_arr.sum())
    total_wins = int((take_arr & (label_arr == 1)).sum())
    for h in range(24):
        h_mask = hours == h
        h_taken_mask = h_mask & take_arr
        n_taken_h = int(h_taken_mask.sum())
        n_wins_h = int((h_taken_mask & (label_arr == 1)).sum())
        if n_taken_h > 0:
            wr_h = n_wins_h / n_taken_h * 100
            pct = n_taken_h / total_taken * 100 if total_taken else 0.0
            wr_str = f"{wr_h:6.2f}%"
            pct_str = f"{pct:6.2f}%"
        else:
            wr_str = "   n/a"
            pct_str = "   n/a"
        print(f"  {f'{h:02d}h':<8}{n_taken_h:>7}{n_wins_h:>8}{wr_str:>10}{pct_str:>14}{_adapt(h):>12}")
    print(f"  {'-' * 59}")
    total_wr = (total_wins / total_taken * 100) if total_taken else float("nan")
    total_wr_str = "   n/a" if total_taken == 0 else f"{total_wr:6.2f}%"
    print(f"  {'Total':<8}{total_taken:>7}{total_wins:>8}{total_wr_str:>10}{'100.00%':>14}{'':>12}")
    # Legende + jours adaptes
    print(f"\n  Adapt. :  SKIP dur = jamais de pari (les {len(skip_set)} pires heures)  |"
          f"  +soft = seuil releve (+ADAPTIVE_SOFT_HOURS_DELTA, {len(soft_set)} heures suivantes)  |"
          f"  -boost = seuil abaisse (hour_boost)")
    if bad_days:
        jours = {1: "Lun", 2: "Mar", 3: "Mer", 4: "Jeu", 5: "Ven", 6: "Sam", 7: "Dim"}
        print(f"  Jours a seuil releve (+ADAPTIVE_BAD_DAY_DELTA) : "
              f"{', '.join(jours.get(int(d), str(d)) for d in bad_days)}")


def _print_directional_wr_ranking(diag, skip_hours=(), soft_hours=(), bad_days=()):
    """Affiche le WR par heure/jour qui SERT DE BASE au classement bad-buckets.
    Si diag['mode']=='takes' : WR des PARIS REELLEMENT PRIS par le systeme
    (p_up>=seuil, hors skip_cells) = perf reelle. Sinon : WR directionnel brut du
    modele nu (p_up>=0.5 sur toutes les bougies). La colonne 'Classement' montre
    lesquels ont ete retenus -> verifie que ce sont bien les pires qui sautent.
    """
    if not diag:
        print("\n  (classement indisponible : ADAPTIVE_BAD_BUCKETS=False ou fenetre vide)")
        return
    skip_set = set(int(h) for h in skip_hours)
    soft_set = set(int(h) for h in soft_hours)
    bad_days_set = set(int(d) for d in bad_days)
    takes = diag.get("mode") == "takes"

    def _wr_str(wr):
        return "    n/a" if wr is None or (isinstance(wr, float) and np.isnan(wr)) \
            else f"{wr * 100:6.2f}%"

    def _key(wr):
        return (1, 0.0) if wr is None or (isinstance(wr, float) and np.isnan(wr)) else (0, float(wr))

    print("\n" + "=" * 70)
    if takes:
        print(f"  WR DES PARIS PRIS (p_up>={diag['bet_threshold']:.3f}, perf REELLE du systeme) "
              f"sur les {diag['window_days']} derniers jours")
        print(f"  = base du classement bad-buckets (min {diag['min_takes']} paris pour etre classable)")
        col_n = "Paris"
    else:
        print(f"  WR DIRECTIONNEL BRUT (p_up>=0.5 vs label, modele nu) sur les "
              f"{diag['window_days']} derniers jours")
        col_n = "Bougies"
    print("=" * 70)

    print("  -- 24 HEURES UTC (triees par WR croissant = pires d'abord) --")
    print(f"  {'Heure':<8}{col_n:>9}{'WR':>10}{'Classement':>14}")
    print(f"  {'-' * 41}")
    for h in sorted(range(24), key=lambda x: (_key(diag['hour_wr'][x]), x)):
        tag = "SKIP dur" if h in skip_set else ("+soft" if h in soft_set else "")
        print(f"  {f'{h:02d}h':<8}{diag['hour_n'][h]:>9}{_wr_str(diag['hour_wr'][h]):>10}{tag:>14}")

    jours = {1: "Lun", 2: "Mar", 3: "Mer", 4: "Jeu", 5: "Ven", 6: "Sam", 7: "Dim"}
    print("\n  -- 7 JOURS (isoweekday, tries par WR croissant) --")
    print(f"  {'Jour':<8}{col_n:>9}{'WR':>10}{'Classement':>14}")
    print(f"  {'-' * 41}")
    for d in sorted(range(1, 8), key=lambda x: (_key(diag['day_wr'][x]), x)):
        tag = "bad-day" if d in bad_days_set else ""
        print(f"  {jours[d]:<8}{diag['day_n'][d]:>9}{_wr_str(diag['day_wr'][d]):>10}{tag:>14}")


def _print_bet_samples(test_df, p_up, take, label_raw, n_samples=5):
    """
    Affiche n_samples paris UP tirés au hasard parmi ceux pris sur le test
    set, avec leur résultat. Pas de seed : tirage différent à chaque run.
    """
    take_arr = np.asarray(take, dtype=bool)
    p_up_arr = np.asarray(p_up, dtype=float)
    label_arr = np.asarray(label_raw, dtype=int)
    timestamps_arr = test_df["timestamp"].to_numpy()

    bet_idx = np.where(take_arr)[0]

    print("\n" + "=" * 70)
    print(f"  ECHANTILLON ALEATOIRE - {n_samples} paris UP")
    print("=" * 70)

    # Moyennes de P(up) sur les paris effectivement pris :
    #   - all   : sur tous les paris pris
    #   - wins  : sur les paris gagnants (label = 1 = bougie verte)
    #   - loss  : sur les paris perdants (label = 0 = bougie rouge)
    win_mask = take_arr & (label_arr == 1)
    loss_mask = take_arr & (label_arr == 0)
    n_taken = int(take_arr.sum())
    n_wins = int(win_mask.sum())
    n_loss = int(loss_mask.sum())
    mean_p_taken = float(p_up_arr[take_arr].mean()) if n_taken > 0 else 0.0
    mean_p_wins = float(p_up_arr[win_mask].mean()) if n_wins > 0 else 0.0
    mean_p_loss = float(p_up_arr[loss_mask].mean()) if n_loss > 0 else 0.0
    print(f"\n  Moyenne P(up) sur les {n_taken} paris pris   : {mean_p_taken*100:.2f}%")
    print(f"  Moyenne P(up) sur les {n_wins:>5} paris gagnants : {mean_p_wins*100:.2f}%")
    print(f"  Moyenne P(up) sur les {n_loss:>5} paris perdants : {mean_p_loss*100:.2f}%  (delta = {(mean_p_wins - mean_p_loss)*100:+.2f}pp)")

    print(f"\n  Paris UP pris (sur {len(bet_idx)} au total) :")
    if len(bet_idx) == 0:
        print("    (aucun pari pris)")
        return

    rng = np.random.default_rng()
    sample_size = min(n_samples, len(bet_idx))
    sampled = rng.choice(bet_idx, size=sample_size, replace=False)
    sampled = np.sort(sampled)

    print(f"    {'date (UTC)':<22}{'P(up)':<14}{'resultat'}")
    print(f"    {'-' * 50}")
    for i in sampled:
        ts_str = pd.to_datetime(int(timestamps_arr[i]), unit="ms", utc=True) \
                   .strftime("%Y-%m-%d %H:%M")
        p = p_up_arr[i]
        ok = label_arr[i] == 1   # label=1 si la bougie suivante etait bien verte
        res = "[OK] correct (up)" if ok else "[KO] rate (down)"
        print(f"    {ts_str:<22}{p*100:>6.2f}%       {res}")


# ============================================================
#  CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Train + evaluate XGBoost UP-only model.")
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
