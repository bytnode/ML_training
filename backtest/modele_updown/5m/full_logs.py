"""Full per-bet logs for the team backtest.

`main.py` calls `write_full_logs()` right after `detailed_log.write_detailed_logs()`.
It produces three wide CSVs inside `5m/logs/full_logs/` :

    lost_bets.csv  - every team bet that LOST
    won_bets.csv   - every team bet that WON
    all_bets.csv   - every team bet (won + lost)

One row per bet the team actually placed (team_decision UP or DOWN). Each row
carries everything the backtest knows about that bet :

  - the analyzed 5m bar   : OHLC + volume + atr_pct + microstructure ratios
  - the target (next) bar : open / close / colour
  - both models' full feature vectors            (prefixed up_ / dn_)
  - both models' probability, threshold, take flag and edge
  - the team decision and the win/loss result
  - the constant hyperparameters / config / sidecar of both packages
        (prefixed up_hp_ up_cfg_ up_sc_  and  dn_hp_ dn_cfg_ dn_sc_)

Where `detailed_log.py` is the thin 21-column per-bar view, this is the wide
"give me all of it" log for offline analysis. It reuses the parquet / model /
sidecar already cached by each package's `api.py` - no extra prediction pass,
no redundant parquet read.

Working-directory contract : same as `detailed_log.py` - imported and called
from `5m/main.py`, which has already put `5m/` on sys.path.
"""
from __future__ import annotations

import importlib
import logging
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

INTERVAL_5M_MS = 5 * 60 * 1000
FULL_LOGS_SUBDIR = "full_logs"

# Output filenames. Order = how the run report lists them.
OUTPUT_FILES = {
    "lost": "lost_bets.csv",
    "won": "won_bets.csv",
    "all": "all_bets.csv",
}

# Raw-bar stats pulled from the SQLite `candles` table (5m timeframe). The
# parquet keeps only scale-invariant features, so price / volume / atr live here.
_BAR_STAT_COLUMNS = ["volume", "atr_pct", "volume_ratio",
                     "body_ratio_close", "body_ratio_avg", "obv_pressure"]

# config.py constants surfaced as constant columns : (output_name, config_attr).
_HYPERPARAM_KEYS = [
    ("n_estimators", "XGB_N_ESTIMATORS"),
    ("max_depth", "XGB_MAX_DEPTH"),
    ("learning_rate", "XGB_LEARNING_RATE"),
    ("min_child_weight", "XGB_MIN_CHILD_WEIGHT"),
    ("subsample", "XGB_SUBSAMPLE"),
    ("colsample_bytree", "XGB_COLSAMPLE_BYTREE"),
    ("reg_lambda", "XGB_REG_LAMBDA"),
    ("reg_alpha", "XGB_REG_ALPHA"),
    ("gamma", "XGB_GAMMA"),
    ("early_stopping_rounds", "XGB_EARLY_STOPPING_ROUNDS"),
    ("random_state", "XGB_RANDOM_STATE"),
    ("scale_pos_weight", "XGB_SCALE_POS_WEIGHT"),
]
_CONFIG_KEYS = [
    ("train_days", "TRAIN_DAYS"),
    ("test_days", "TEST_DAYS"),
    ("val_fraction", "VAL_FRACTION"),
    ("warmup", "WARMUP"),
    ("atr_long_period", "ATR_LONG_PERIOD"),
    ("volatility_high", "VOLATILITY_HIGH"),
    ("volatility_low", "VOLATILITY_LOW"),
    ("adaptive_strategy", "ADAPTIVE_STRATEGY"),
    ("asym_vol_frac_weekday", "ASYM_VOL_FRAC_WEEKDAY"),
    ("asym_vol_frac_weekend", "ASYM_VOL_FRAC_WEEKEND"),
    ("asym_skip_cells", "ASYM_SKIP_CELLS"),
    ("asym_use_full_val", "ASYM_USE_FULL_VAL"),
    ("asym_hour_boost_good_hours", "ASYM_HOUR_BOOST_GOOD_HOURS"),
    ("asym_hour_boost_delta", "ASYM_HOUR_BOOST_DELTA"),
]

# Per-bet columns, in reading order. Feature columns (model-dependent) and the
# constant config columns are appended after these at build time.
_HEAD_COLUMNS = [
    "timestamp_ms", "datetime_utc",
    "team_decision", "result",
    "bet_confidence", "bet_threshold", "bet_edge",
    "p_up", "th_up", "take_up", "edge_up",
    "p_down", "th_down", "take_down", "edge_down",
    "volatility_state", "is_weekend", "hour",
    "open", "high", "low", "close",
    "volume", "atr_pct", "volume_ratio",
    "body_ratio_close", "body_ratio_avg", "obv_pressure",
    "target_timestamp_ms", "next_open", "next_close", "next_candle",
]


def _scalar(value):
    """CSV-safe form of a config value : tuples / lists -> one string cell."""
    if isinstance(value, (tuple, list)):
        return str(value)
    return value


def _package_name(api_mod) -> str:
    """'modele_up.api' -> 'modele_up' (holds for namespace packages too)."""
    return api_mod.__name__.rsplit(".", 1)[0]


def _feature_frame(api_mod, prefix: str) -> tuple[pd.DataFrame, list[str]]:
    """Return (frame, prefixed_cols) : the package's full feature matrix keyed
    on timestamp_ms, every feature column renamed to `<prefix>_<feature>`.

    Reuses the model + parquet already cached by api.py - no extra I/O. The
    feature list is read straight off the booster, so it always matches what
    the model was actually trained on.
    """
    model = api_mod._load_model()
    feats = list(model.get_booster().feature_names)
    parquet = api_mod._load_parquet()
    prefixed = [f"{prefix}_{c}" for c in feats]
    frame = (parquet[["timestamp_ms"] + feats]
             .rename(columns={c: f"{prefix}_{c}" for c in feats})
             .astype({"timestamp_ms": "int64"}))
    return frame, prefixed


def _bar_stats(api_down) -> pd.DataFrame:
    """Raw 5m bar stats (volume / atr_pct / microstructure) from the SQLite
    `candles` table built by modele_down/data.py. Empty (with the right
    columns) if the DB is missing - the caller then leaves those columns
    blank, exactly like detailed_log.py does for OHLC.
    """
    db_path = api_down._MODEL_DIR / "DB" / "btc_multi_tf.db"
    cols = ["timestamp_ms"] + _BAR_STAT_COLUMNS
    if not db_path.exists():
        logging.warning("full_logs: candles DB not found (%s); volume / atr / "
                        "microstructure columns will be blank.", db_path)
        return pd.DataFrame(columns=cols)

    query = (
        "SELECT timestamp_ms, volume, atr_pct, volume_ratio, body_ratio_close, "
        "body_ratio_avg, obv_pressure FROM candles WHERE timeframe = '5m'"
    )
    conn = sqlite3.connect(db_path)
    try:
        stats = pd.read_sql_query(query, conn)
    finally:
        conn.close()
    stats["timestamp_ms"] = stats["timestamp_ms"].astype("int64")
    return stats


def _package_static(api_mod, prefix: str) -> dict:
    """Flat dict of the package's constant columns : XGBoost hyperparameters +
    training/calibration config (both from config.py) + the trained sidecar.
    Keys are already prefixed (up_hp_* / up_cfg_* / up_sc_*, or the dn_ set).
    Same value on every row by design - this is the "what setup produced this
    bet" context the user asked to carry inside each file.
    """
    cfg = importlib.import_module(f"{_package_name(api_mod)}.config")
    out: dict = {}
    for name, attr in _HYPERPARAM_KEYS:
        out[f"{prefix}_hp_{name}"] = _scalar(getattr(cfg, attr, None))
    for name, attr in _CONFIG_KEYS:
        out[f"{prefix}_cfg_{name}"] = _scalar(getattr(cfg, attr, None))

    sidecar = api_mod._load_sidecar()
    thresholds = sidecar.get("thresholds_per_wknd", {})
    hour_boost = sidecar.get("hour_boost", {})
    meta = sidecar.get("metadata", {})
    out[f"{prefix}_sc_strategy"] = sidecar.get("strategy")
    out[f"{prefix}_sc_threshold_weekday"] = thresholds.get("0")
    out[f"{prefix}_sc_threshold_weekend"] = thresholds.get("1")
    out[f"{prefix}_sc_skip_cells"] = _scalar(sidecar.get("skip_cells"))
    out[f"{prefix}_sc_hour_boost_good_hours"] = _scalar(hour_boost.get("good_hours"))
    out[f"{prefix}_sc_hour_boost_delta"] = hour_boost.get("delta")
    out[f"{prefix}_sc_trained_at"] = meta.get("trained_at")
    out[f"{prefix}_sc_calib_method"] = meta.get("calib_method")
    return out


def _write_three(table: pd.DataFrame, out_dir: Path) -> dict:
    """Split `table` into lost / won / all and write the three CSVs. `table`
    holds only taken bets, so won + lost partition it exactly."""
    lost = table[table["result"] == "loss"]
    won = table[table["result"] == "win"]
    lost.to_csv(out_dir / OUTPUT_FILES["lost"], index=False)
    won.to_csv(out_dir / OUTPUT_FILES["won"], index=False)
    table.to_csv(out_dir / OUTPUT_FILES["all"], index=False)
    return {
        f"{FULL_LOGS_SUBDIR}/{OUTPUT_FILES['lost']}": len(lost),
        f"{FULL_LOGS_SUBDIR}/{OUTPUT_FILES['won']}": len(won),
        f"{FULL_LOGS_SUBDIR}/{OUTPUT_FILES['all']}": len(table),
    }


def write_full_logs(per_bar: pd.DataFrame, api_up, api_down, logs_dir) -> dict:
    """Build and write the three full per-bet CSVs into `logs_dir/full_logs/`.

    `per_bar` is the per-bar table from detailed_log.build_per_bar_table (every
    analyzed bar). We keep only the taken bets, enrich them with both models'
    feature vectors, the raw bar stats and the constant config, then split the
    result into lost / won / all. Overwrites each run. Returns
    {relative_filename: row_count} for the run report.
    """
    out_dir = Path(logs_dir) / FULL_LOGS_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # Constant columns + feature matrices (model-dependent -> built once).
    static_up = _package_static(api_up, "up")
    static_down = _package_static(api_down, "dn")
    up_feats, up_cols = _feature_frame(api_up, "up")
    dn_feats, dn_cols = _feature_frame(api_down, "dn")

    final_cols = (_HEAD_COLUMNS + up_cols + dn_cols
                  + list(static_up) + list(static_down))

    # One row per bet the team actually placed (PASS / CANCEL bars excluded).
    taken = per_bar[per_bar["bet_taken"] == 1].copy()
    if taken.empty:
        return _write_three(pd.DataFrame(columns=final_cols), out_dir)

    taken["timestamp_ms"] = taken["timestamp_ms"].astype("int64")

    table = (taken
             .merge(up_feats, on="timestamp_ms", how="left")
             .merge(dn_feats, on="timestamp_ms", how="left"))

    bar_stats = _bar_stats(api_down)
    if bar_stats.empty:
        for col in _BAR_STAT_COLUMNS:
            table[col] = pd.NA
    else:
        table = table.merge(bar_stats, on="timestamp_ms", how="left")

    # Derived per-bet columns.
    is_up = table["team_decision"].to_numpy() == "UP"
    table["edge_up"] = table["p_up"] - table["th_up"]
    table["edge_down"] = table["p_down"] - table["th_down"]
    table["bet_confidence"] = np.where(is_up, table["p_up"], table["p_down"])
    table["bet_threshold"] = np.where(is_up, table["th_up"], table["th_down"])
    table["bet_edge"] = table["bet_confidence"] - table["bet_threshold"]
    table["target_timestamp_ms"] = table["timestamp_ms"] + INTERVAL_5M_MS

    # Constant columns (hyperparams / config / sidecar - same on every row).
    for key, value in {**static_up, **static_down}.items():
        table[key] = value

    table = table.sort_values("timestamp_ms").reset_index(drop=True)
    return _write_three(table.reindex(columns=final_cols), out_dir)
