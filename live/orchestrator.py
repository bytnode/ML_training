"""Daemon principal de l'orchestrateur Aure live."""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import xgboost as xgb

from settings import (
    MODEL_PATH, THRESHOLDS_PATH, THRESHOLDS_SOURCE,
    LOGS_DIR, ORCHESTRATOR_LOG, REPO_ROOT,
    WARMUP_CANDLES_15M, WARMUP_CANDLES_MTF,
    RECENT_FETCH_15M, RECENT_FETCH_MTF,
    INTERVAL_15M_MS, CYCLE_FUDGE_SECONDS,
    FRESH_CANDLE_RETRY_INTERVAL, FRESH_CANDLE_MAX_RETRIES,
)
from fetch import fetch_warmup_window, fetch_recent_window, merge_klines
from features import find_setups_at_latest
from store import (
    append_training_row, append_decision_row,
    update_pending_labels, update_pending_outcomes,
)
from notify import send_message, format_decision_message

_log = logging.getLogger("orchestrator")


# ---------------------------------------------------------------------------
# .env loader
# ---------------------------------------------------------------------------
def _load_env_file(path: Path) -> int:
    """Charge un fichier .env (KEY=VALUE par ligne) dans os.environ.

    - Skip blank lines, lignes commencant par #, lignes sans '='.
    - Si la VALUE est vide, ne touche pas os.environ pour cette clef.
    - Si KEY est deja set dans os.environ, ne l'ecrase PAS (la vraie env var
      gagne contre le .env -- pratique pour overrider ponctuellement en CLI).
    - Strip un eventuel quoting "..." ou '...' autour de la valeur.

    Returns: nombre de variables chargees.
    """
    if not path.exists():
        return 0
    loaded = 0
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if not key or not value:
            continue
        if key not in os.environ:
            os.environ[key] = value
            loaded += 1
    return loaded


# ---------------------------------------------------------------------------
# Setup logging
# ---------------------------------------------------------------------------
def _setup_logging() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)sZ %(levelname)s  %(message)s"
    datefmt = "%Y-%m-%dT%H:%M:%S"
    file_handler = logging.FileHandler(ORCHESTRATOR_LOG, encoding="utf-8")
    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter(fmt, datefmt=datefmt)
    formatter.converter = time.gmtime
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers = [file_handler, stream_handler]


# ---------------------------------------------------------------------------
# Model + sidecar loading
# ---------------------------------------------------------------------------
def _ensure_sidecar() -> None:
    """Copie backtest/btc_reversal_model.thresholds.json -> aure.thresholds.json
    si ce dernier n'existe pas. Idempotent."""
    if THRESHOLDS_PATH.exists():
        return
    if not THRESHOLDS_SOURCE.exists():
        _log.warning("sidecar_missing source=%s — Aure utilisera un seuil fixe (fallback)",
                     THRESHOLDS_SOURCE)
        return
    shutil.copy2(THRESHOLDS_SOURCE, THRESHOLDS_PATH)
    _log.info("sidecar_copied %s -> %s", THRESHOLDS_SOURCE.name, THRESHOLDS_PATH.name)


def _load_model() -> xgb.XGBClassifier:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"model file not found at {MODEL_PATH} — copy aure.json to repo root"
        )
    model = xgb.XGBClassifier()
    model.load_model(str(MODEL_PATH))
    _log.info("model_loaded %s features=%d", MODEL_PATH.name,
              len(model.get_booster().feature_names))
    return model


def _load_sidecar() -> dict:
    """Returns {'base_threshold': float, 'deltas': dict[str, float]}.

    Si pas de sidecar OU sidecar malforme : fallback
    {'base_threshold': 0.58, 'deltas': {}} avec un log ERROR. On ne crash
    pas le daemon — on tourne avec le seuil fixe et l'operateur fixe le
    sidecar a chaud (et restart pour le reload).
    """
    if not THRESHOLDS_PATH.exists():
        return {"base_threshold": 0.58, "deltas": {}}
    try:
        with open(THRESHOLDS_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return {
            "base_threshold": float(data["base_threshold"]),
            "deltas": {k: float(v) for k, v in data.get("deltas", {}).items()},
        }
    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
        _log.error(
            "sidecar_malformed %s err=%s — fallback to base=0.58 with no deltas",
            THRESHOLDS_PATH, e,
        )
        return {"base_threshold": 0.58, "deltas": {}}


# ---------------------------------------------------------------------------
# Sleep alignment
# ---------------------------------------------------------------------------
def next_boundary_ms(now_ms: int) -> int:
    """Retourne le prochain open_time_ms 15m strictement apres now_ms."""
    return ((now_ms // INTERVAL_15M_MS) + 1) * INTERVAL_15M_MS


def sleep_until(target_ms: int) -> None:
    while True:
        delta = target_ms / 1000 - time.time()
        if delta <= 0:
            return
        time.sleep(min(delta, 30))  # capping a 30s pour eviter sleep trop long


# ---------------------------------------------------------------------------
# Predict + decide
# ---------------------------------------------------------------------------
def _predict_and_decide(model: xgb.XGBClassifier, sidecar: dict, features: dict) -> dict:
    """Calcule confidence, effective_threshold, taken, bet_direction.

    Returns dict utilise par store.append_decision_row + notify.format_decision_message.
    """
    expected_cols = list(model.get_booster().feature_names)
    missing = [col for col in expected_cols if col not in features]
    if missing:
        raise ValueError(
            f"_predict_and_decide: features dict missing keys {missing} — "
            f"verifier que find_setups_at_latest produit le bon schema"
        )
    df = pd.DataFrame([{col: features[col] for col in expected_cols}])
    confidence = float(model.predict_proba(df)[:, 1][0])

    bucket_v = int(features["volatility_state"])
    bucket_w = int(features["is_weekend"])
    bucket_key = f"{bucket_v}_{bucket_w}"
    delta = sidecar["deltas"].get(bucket_key, 0.0)
    effective_threshold = sidecar["base_threshold"] + delta

    bet_direction = "down" if features["seq_dir"] == 1 else "up"
    taken = confidence >= effective_threshold

    return {
        "confidence": confidence,
        "effective_threshold": effective_threshold,
        "bucket_v": bucket_v,
        "bucket_w": bucket_w,
        "bet_direction": bet_direction,
        "taken": taken,
    }


# ---------------------------------------------------------------------------
# Run cycle
# ---------------------------------------------------------------------------
def run_cycle(
    model, sidecar, klines_state: dict, no_notify: bool,
    wait_for_fresh: bool = True,
) -> dict:
    """Execute un cycle complet. Mute klines_state in place avec les nouvelles bougies.

    wait_for_fresh : si True (mode daemon), retry le fetch jusqu'a obtenir une
    bougie 15m strictement plus recente que la derniere connue (= preuve que
    Bitstamp a publie la nouvelle bougie qui vient de fermer). Si epuise, skip
    le cycle. False en mode --dry-run (on veut tester le pipeline meme si le
    fetch ne ramene rien de neuf — typique car warmup vient juste de tourner).

    Returns: stats du cycle (dict) pour log.
    """
    cycle_start = time.time()

    # 1. Fetch recent (+ freshness retry en mode daemon).
    if wait_for_fresh:
        prev_last_open = klines_state["15m"][-1][0] if klines_state.get("15m") else 0
        fresh = None
        fresh_ok = False
        for attempt in range(1, FRESH_CANDLE_MAX_RETRIES + 1):
            fresh = fetch_recent_window(RECENT_FETCH_15M, RECENT_FETCH_MTF)
            for tf, candles in fresh.items():
                klines_state[tf] = merge_klines(klines_state.get(tf, []), candles)
            latest_open = klines_state["15m"][-1][0] if klines_state["15m"] else 0
            if latest_open > prev_last_open:
                fresh_ok = True
                if attempt > 1:
                    _log.info("fresh_candle_obtained attempt=%d", attempt)
                break
            if attempt < FRESH_CANDLE_MAX_RETRIES:
                time.sleep(FRESH_CANDLE_RETRY_INTERVAL)

        if not fresh_ok:
            _log.warning(
                "no_fresh_candle after %d retries (last_known_open=%d) — cycle skipped",
                FRESH_CANDLE_MAX_RETRIES, prev_last_open,
            )
            return {
                "setups": 0,
                "notifs_sent": 0,
                "duration_ms": int((time.time() - cycle_start) * 1000),
                "labels_resolved": 0,
                "outcomes_resolved": 0,
            }
    else:
        # Mode --dry-run : un seul fetch, pas de freshness check.
        fresh = fetch_recent_window(RECENT_FETCH_15M, RECENT_FETCH_MTF)
        for tf, candles in fresh.items():
            klines_state[tf] = merge_klines(klines_state.get(tf, []), candles)

    _log.info("fetched candles_15m=%d candles_30m=%d candles_1h=%d candles_4h=%d candles_1d=%d",
              len(fresh.get("15m", [])), len(fresh.get("30m", [])),
              len(fresh.get("1h", [])), len(fresh.get("4h", [])),
              len(fresh.get("1d", [])))

    # 2. Resolve pendings (avec les klines_15m mises a jour)
    n_labels, _ = update_pending_labels(klines_state["15m"])
    n_outcomes, _ = update_pending_outcomes(klines_state["15m"])
    if n_labels or n_outcomes:
        _log.info("pendings_resolved labels=%d outcomes=%d", n_labels, n_outcomes)

    # 3. Detect setups
    klines_15m = klines_state["15m"]
    klines_mtf = {tf: klines_state[tf] for tf in klines_state if tf != "15m"}
    setups = find_setups_at_latest(klines_15m, klines_mtf)
    _log.info("setups_qualifying=%d", len(setups))

    # 4. Pour chaque setup : append + predict + notif
    notifs_sent = 0
    for seq_len, features in setups:
        append_training_row(features)
        decision = _predict_and_decide(model, sidecar, features)

        notified_at = None
        if decision["taken"] and not no_notify:
            price = float(klines_15m[-1][4])  # close de la derniere bougie
            msg = format_decision_message(features, decision, price=price)
            ok = send_message(msg)
            if ok:
                notified_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                notifs_sent += 1
        decision["notified_at"] = notified_at

        append_decision_row(features, decision)
        _log.info(
            "decision seq_len=%d confidence=%.3f threshold=%.3f bucket=%d_%d taken=%s direction=%s",
            seq_len, decision["confidence"], decision["effective_threshold"],
            decision["bucket_v"], decision["bucket_w"], decision["taken"],
            decision["bet_direction"],
        )

    duration_ms = int((time.time() - cycle_start) * 1000)
    _log.info("cycle_end duration_ms=%d notifs_sent=%d", duration_ms, notifs_sent)
    return {
        "setups": len(setups),
        "notifs_sent": notifs_sent,
        "duration_ms": duration_ms,
        "labels_resolved": n_labels,
        "outcomes_resolved": n_outcomes,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Daemon Aure live orchestrator")
    parser.add_argument("--dry-run", action="store_true",
                        help="Execute UN seul cycle puis quitte (test rapide).")
    parser.add_argument("--no-notify", action="store_true",
                        help="Ne pas envoyer de notif Telegram (decisions seulement loguees).")
    args = parser.parse_args(argv)

    # Charge .env (a la racine du repo) AVANT de setup le logging — comme ca le
    # logging file handler peut ecrire dans LOGS_DIR sans surprises et les
    # env vars (Telegram) sont deja en place quand notify.py en a besoin.
    n_env = _load_env_file(REPO_ROOT / ".env")

    _setup_logging()
    _log.info("orchestrator_start dry_run=%s no_notify=%s", args.dry_run, args.no_notify)
    if n_env:
        _log.info("env_loaded from %s vars=%d", REPO_ROOT / ".env", n_env)

    _ensure_sidecar()
    model = _load_model()
    sidecar = _load_sidecar()

    # Warmup
    _log.info("fetching_warmup candles_15m=%d candles_mtf=%d", WARMUP_CANDLES_15M, WARMUP_CANDLES_MTF)
    klines_state = fetch_warmup_window(WARMUP_CANDLES_15M, WARMUP_CANDLES_MTF)

    # Resolve startup pendings (avec les klines warmup)
    n_labels, rem_labels = update_pending_labels(klines_state["15m"])
    n_outcomes, rem_outcomes = update_pending_outcomes(klines_state["15m"])
    if n_labels or n_outcomes:
        _log.info("startup_pendings_resolved labels=%d outcomes=%d", n_labels, n_outcomes)
    if rem_labels or rem_outcomes:
        # Pendings dont la bougie cible est anterieure a la fenetre de warmup
        # (typiquement daemon down >50h). Restent NaN/pending indefiniment ;
        # l'operateur peut regenerer via backtest/data.py si besoin.
        _log.warning("unresolvable_pending labels=%d outcomes=%d", rem_labels, rem_outcomes)

    if args.dry_run:
        _log.info("dry_run executing single cycle")
        run_cycle(model, sidecar, klines_state, no_notify=args.no_notify, wait_for_fresh=False)
        _log.info("dry_run done")
        return 0

    # Daemon loop
    while True:
        now_ms = int(time.time() * 1000)
        target_ms = next_boundary_ms(now_ms) + CYCLE_FUDGE_SECONDS * 1000
        target_iso = datetime.fromtimestamp(target_ms / 1000, tz=timezone.utc).strftime("%H:%M:%S UTC")
        _log.info("sleeping_until %s", target_iso)
        try:
            sleep_until(target_ms)
        except KeyboardInterrupt:
            _log.info("orchestrator_stopping (keyboard interrupt)")
            return 0

        _log.info("cycle_start")
        try:
            run_cycle(model, sidecar, klines_state, no_notify=args.no_notify)
        except KeyboardInterrupt:
            _log.info("orchestrator_stopping (keyboard interrupt mid-cycle)")
            return 0
        except Exception as e:
            _log.exception("cycle_failed err=%s", e)
            # On continue : prochain cycle dans 15min.


if __name__ == "__main__":
    sys.exit(main())
