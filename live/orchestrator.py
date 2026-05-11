"""Live trader daemon: detect setups, ask the model, place real Polymarket bets.

Cycle (every 15min UTC boundary):
    1. Sleep until next xx:00 / xx:15 / xx:30 / xx:45 UTC
    2. Freshness retry (max 5) until Bitstamp publishes the new candle
    3. Resolve pending bets whose target window has closed
    4. Fetch new candles (15m + MTF), merge into rolling buffer
    5. Detect qualifying setups for seq_len in [2, 3, 4]
    6. Build features per candidate
    7. Decide: predict_proba + should_take_bet (adaptive sidecar) -> pick highest-confidence taken
    8. Place bet on Polymarket if any
    9. Telegram + wallet snapshot

CLI:
    python orchestrator.py            # run forever
    python orchestrator.py --dry-run  # one cycle, log everything, no real order
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import settings  # FIRST — triggers sys.path injection for predict / data imports

import xgboost as xgb

# After settings: predict / data are reachable.
import predict as model_predict  # noqa: E402

import fetch  # noqa: E402
import features as feat_mod  # noqa: E402
import decide  # noqa: E402
import store  # noqa: E402
import notify  # noqa: E402
from polymarket import PolymarketClient, MarketNotFoundError, from_settings  # noqa: E402

# Constants from backtest config (via sys.path injection in settings).
from config import (  # noqa: E402
    INTERVAL_MS, ADAPTIVE_TARGET_WR,
)

logger = logging.getLogger("live.orchestrator")
INTERVAL_15M_MS = INTERVAL_MS["15m"]


# ============================================================
#  Model + sidecar injection (avoids cwd-dependent loading)
# ============================================================
def inject_model_and_sidecar():
    """Pre-populate predict._model and predict._thresholds with absolute paths.

    Avoids predict.py's lazy load of MODEL_OUTPUT (relative path = wrong cwd).
    """
    if not settings.MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {settings.MODEL_PATH}")
    if not settings.SIDECAR_PATH.exists():
        raise FileNotFoundError(
            f"Sidecar not found: {settings.SIDECAR_PATH}. "
            f"Run 'cd backtest/modele_15m && python train.py' to regenerate."
        )

    m = xgb.XGBClassifier()
    m.load_model(str(settings.MODEL_PATH))
    model_predict._model = m

    with open(settings.SIDECAR_PATH) as f:
        model_predict._thresholds = json.load(f)


# ============================================================
#  Cycle helpers
# ============================================================
def next_15m_boundary_seconds(now_s: float | None = None) -> int:
    """Returns the next xx:00 / xx:15 / xx:30 / xx:45 UTC as unix seconds."""
    if now_s is None:
        now_s = time.time()
    return ((int(now_s) // 900) + 1) * 900


def sleep_until_boundary():
    target = next_15m_boundary_seconds()
    delay = target - time.time()
    if delay > 0:
        logger.info("sleeping %.1fs until next 15m boundary (target=%d)", delay, target)
        time.sleep(delay)
    # Safety: time.sleep() can return early due to clock skew (observed: ~5-6s
    # systematic drift in Docker on Windows). Wait the remainder if needed.
    while time.time() < target:
        time.sleep(min(0.5, max(0.0, target - time.time())))


def split_klines(klines: list[list]) -> tuple[list, list, list, list, list, list]:
    opens = [float(k[1]) for k in klines]
    highs = [float(k[2]) for k in klines]
    lows = [float(k[3]) for k in klines]
    closes = [float(k[4]) for k in klines]
    volumes = [float(k[5]) for k in klines]
    times = [int(k[0]) for k in klines]
    return opens, highs, lows, closes, volumes, times


# ============================================================
#  Pending settlement reconciliation (step 3 of cycle)
# ============================================================
def reconcile_pending(client: PolymarketClient, cycle_count_by_bet: dict[str, int]) -> None:
    pendings = store.read_pending_bets()
    if not pendings:
        return

    now_ms = int(time.time() * 1000)
    for row in pendings:
        target_end = int(row["target_window_end_ms"])
        if target_end >= now_ms:
            continue

        bet_id = row["bet_id"]
        market_slug = row["polymarket_slug"]
        cycle_count_by_bet[bet_id] = cycle_count_by_bet.get(bet_id, 0) + 1

        result = client.read_settlement(market_slug)
        if result is None:
            if cycle_count_by_bet[bet_id] >= settings.SETTLEMENT_STUCK_AFTER_CYCLES:
                logger.error("settlement_stuck bet_id=%s slug=%s", bet_id, market_slug)
                store.update_bet_outcome(bet_id, "stuck", 0.0)
                notify.alert(
                    "Settlement stuck",
                    f"bet_id={bet_id} slug={market_slug} "
                    f"({cycle_count_by_bet[bet_id]} cycles past close)",
                )
            continue

        winning = result["winning_outcome"]
        bet_direction = row["bet_direction"]
        shares_filled = float(row.get("shares_filled") or 0)
        usdc_spent = float(row.get("usdc_spent") or 0)

        if winning == bet_direction:
            outcome = "win"
            realized_pnl = shares_filled - usdc_spent
        else:
            outcome = "loss"
            realized_pnl = -usdc_spent

        store.update_bet_outcome(bet_id, outcome, realized_pnl, result.get("resolved_at"))
        notify.bet_resolved(bet_direction, outcome, realized_pnl)
        logger.info(
            "settlement bet_id=%s outcome=%s pnl=%+.4f winning=%s direction=%s",
            bet_id, outcome, realized_pnl, winning, bet_direction,
        )


# ============================================================
#  Bet placement (step 8 of cycle)
# ============================================================
def place_bet(client: PolymarketClient, chosen: dict, target_window_start_ms: int,
              dry_run: bool) -> str | None:
    """Place a real (or simulated) Polymarket order for the chosen setup.
    Returns bet_id if a row was written to bets.csv, else None.
    """
    target_window_end_ms = target_window_start_ms + INTERVAL_15M_MS
    bet_direction = chosen["bet_direction"]

    # Read balance pre-trade
    balance_pre = client.get_usdc_balance()
    bet_usdc = math.floor(balance_pre * settings.BET_FRACTION * 100) / 100  # 2-decimal precision

    if bet_usdc < settings.BET_MIN_USDC:
        logger.error("balance_low usdc=%.4f bet_size=%.4f", balance_pre, bet_usdc)
        notify.alert("Balance low", f"USDC={balance_pre:.4f}, can't place ${bet_usdc:.2f} bet")
        return None
    if bet_usdc > balance_pre * settings.BET_MAX_FRACTION:
        logger.error("sizing_anomaly bet=%.4f balance=%.4f", bet_usdc, balance_pre)
        notify.alert("Sizing anomaly", f"bet=${bet_usdc:.2f} > 95% of balance ${balance_pre:.2f}")
        return None

    # Resolve target market
    try:
        market = client.find_btc_15m_market(target_window_start_ms // 1000)
    except MarketNotFoundError as e:
        logger.warning("market_not_found ts=%d reason=%s", target_window_start_ms, e)
        notify.alert("Market not found", f"window_start={target_window_start_ms}")
        return None

    token_id = market["down_token_id"] if bet_direction == "down" else market["up_token_id"]

    bet_id = str(uuid.uuid4())
    bet_row: dict = {
        "bet_id": bet_id,
        "decision_timestamp": chosen["features"]["timestamp"],
        "target_window_start_ms": target_window_start_ms,
        "target_window_end_ms": target_window_end_ms,
        "polymarket_slug": market["market_slug"],
        "polymarket_condition_id": market["condition_id"],
        "polymarket_token_id": token_id,
        "bet_direction": bet_direction,
        "bet_size_usdc": bet_usdc,
        "wallet_balance_pre_usdc": balance_pre,
        "outcome": "pending",
    }

    if dry_run:
        logger.info("[DRY RUN] would place_market_order token=%s usdc=%.4f", token_id, bet_usdc)
        bet_row.update({
            "polymarket_order_id": "DRY_RUN",
            "order_status": "filled",
            "fill_price_avg": None,
            "shares_filled": 0.0,
            "usdc_spent": 0.0,
            "fee_usdc": 0.0,
            "wallet_balance_post_usdc": balance_pre,
        })
        store.append_bet(bet_row)
        return bet_id

    # Real order
    result = client.place_market_order(token_id, bet_usdc)
    balance_post = client.get_usdc_balance()
    bet_row.update({
        "polymarket_order_id": result.get("order_id"),
        "order_status": result["status"],
        "fill_price_avg": result.get("average_fill_price"),
        "shares_filled": result.get("shares_filled"),
        "usdc_spent": result.get("usdc_spent"),
        "fee_usdc": result.get("fee_usdc"),
        "wallet_balance_post_usdc": balance_post,
    })

    if result["status"] != "filled":
        bet_row["outcome"] = "loss"
        bet_row["realized_pnl_usdc"] = 0.0
        bet_row["outcome_resolved_at_iso"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        notify.alert(
            f"Order {result['status']}",
            f"bet_id={bet_id} reason={result.get('error_msg', 'unknown')}",
        )

    store.append_bet(bet_row)

    if result["status"] == "filled":
        notify.bet_placed(
            direction=bet_direction,
            bet_size_usdc=bet_usdc,
            fill_price=result.get("average_fill_price"),
            confidence=chosen["confidence"],
            effective_threshold=ADAPTIVE_TARGET_WR + 0.0,  # log the adaptive base; per-row delta omitted for now
            target_window_start_ms=target_window_start_ms,
        )

    return bet_id


# ============================================================
#  Cycle (one iteration of the daemon loop)
# ============================================================
def run_cycle(client: PolymarketClient, last_known_open_ms: int | None,
              cycle_settlement_counter: dict, dry_run: bool) -> int | None:
    """Run one cycle. Returns the new last_known_open_ms (or None if cycle skipped)."""
    cycle_t0 = time.time()
    logger.info("cycle_start dry_run=%s", dry_run)

    # ----- Step 3: reconcile pending settlements (before fetch, runs even on skipped cycles) -----
    reconcile_pending(client, cycle_settlement_counter)

    # ----- Step 4: fetch (with freshness retry) -----
    try:
        if last_known_open_ms is None:
            klines_15m = fetch.fetch_warmup_15m()
        else:
            fresh_15m = fetch.freshness_retry(last_known_open_ms)
            klines_15m = fetch.fetch_warmup_15m()
            klines_15m = fetch.merge_klines(klines_15m, fresh_15m)
    except TimeoutError as e:
        logger.warning("cycle_skip reason=stale_candle err=%s", e)
        return None

    if len(klines_15m) < settings.WARMUP_CANDLES_15M // 2:
        logger.warning("cycle_skip reason=insufficient_warmup n=%d", len(klines_15m))
        return None

    klines_mtf = fetch.fetch_warmup_mtf()

    opens, highs, lows, closes, volumes, times = split_klines(klines_15m)

    # The latest candle in the buffer is usually still in progress (its close
    # is the live mid-price at fetch time, not the final close). We only ever
    # analyze a fully-closed candle. Skip the latest if it has not closed yet.
    now_ms = int(time.time() * 1000)
    latest_idx = len(closes) - 1
    latest_open_ms = times[latest_idx]
    if latest_open_ms + INTERVAL_15M_MS <= now_ms:
        last_idx = latest_idx
    else:
        last_idx = latest_idx - 1
    last_open_ms = times[last_idx]

    # ----- Step 5: detect qualifying setups -----
    candidates = feat_mod.detect_qualifying_setups(opens, highs, lows, closes, last_idx)
    if not candidates:
        logger.info("no_setup last_idx=%d last_open=%d", last_idx, last_open_ms)
        store.append_wallet_snapshot({
            "cycle_action": "no_setup",
            "usdc_balance": client.get_usdc_balance(),
            "matic_balance": client.get_matic_balance(),
            "pending_bets_n": len(store.read_pending_bets()),
            "pending_bets_exposure_usdc": sum(
                float(r.get("bet_size_usdc") or 0) for r in store.read_pending_bets()
            ),
        })
        return latest_open_ms

    # ----- Step 6: build features -----
    enriched = []
    for c in candidates:
        f = feat_mod.build_features(
            opens, highs, lows, closes, volumes, times, klines_mtf,
            c["last_idx"], c["seq_len"], c["seq_dir"],
        )
        if f is None:
            logger.warning("feature_build_failed seq_len=%d (insufficient MTF)", c["seq_len"])
            continue
        enriched.append({**c, "features": f})

    if not enriched:
        logger.warning("cycle_skip reason=no_features_built")
        return latest_open_ms

    # ----- Step 7: decide -----
    evaluated = decide.evaluate_candidates(enriched)
    chosen = decide.pick_chosen(evaluated)

    # Append decisions.csv (one row per candidate)
    decision_rows = []
    for c in evaluated:
        is_chosen = (chosen is not None and c is chosen)
        eff_th = ADAPTIVE_TARGET_WR + _delta_for_bucket(c["bucket_v"], c["bucket_w"])
        row = {
            **c["features"],
            "bet_direction": c["bet_direction"],
            "confidence": c["confidence"],
            "effective_threshold": eff_th,
            "bucket_v": c["bucket_v"],
            "bucket_w": c["bucket_w"],
            "taken": c["taken"],
            "is_chosen": is_chosen,
            "bet_id": "",  # filled below if order placed
        }
        decision_rows.append(row)

    if chosen is None:
        store.append_decisions(decision_rows)
        logger.info(
            "no_take candidates=%d (max_conf=%.4f)", len(evaluated),
            max((c["confidence"] for c in evaluated), default=0.0),
        )
        store.append_wallet_snapshot({
            "cycle_action": "no_take",
            "usdc_balance": client.get_usdc_balance(),
            "matic_balance": client.get_matic_balance(),
            "pending_bets_n": len(store.read_pending_bets()),
            "pending_bets_exposure_usdc": sum(
                float(r.get("bet_size_usdc") or 0) for r in store.read_pending_bets()
            ),
        })
        return latest_open_ms

    # ----- Step 8: place bet -----
    target_window_start_ms = last_open_ms + INTERVAL_15M_MS
    bet_id = place_bet(client, chosen, target_window_start_ms, dry_run)
    if bet_id is not None:
        # Stamp the chosen decision row with the bet_id
        for r in decision_rows:
            if r["is_chosen"]:
                r["bet_id"] = bet_id
                break
    store.append_decisions(decision_rows)

    # ----- Step 9: wallet snapshot -----
    pendings = store.read_pending_bets()
    store.append_wallet_snapshot({
        "cycle_action": "bet_placed" if bet_id else "market_not_found",
        "usdc_balance": client.get_usdc_balance(),
        "matic_balance": client.get_matic_balance(),
        "pending_bets_n": len(pendings),
        "pending_bets_exposure_usdc": sum(float(r.get("bet_size_usdc") or 0) for r in pendings),
    })

    logger.info("cycle_end duration_ms=%d", int((time.time() - cycle_t0) * 1000))
    return latest_open_ms


def _delta_for_bucket(vol_state: int, is_weekend: int) -> float:
    """Read the per-bucket delta from the cached sidecar."""
    sidecar = model_predict.get_thresholds()
    delta = sidecar.get("deltas", {}).get(f"{int(vol_state)}_{int(is_weekend)}", 0.0)
    return float(delta) if delta is not None else 0.0


# ============================================================
#  Boot + main loop
# ============================================================
def boot(client: PolymarketClient) -> None:
    """Verify model + sidecar load, log wallet, MATIC sanity check."""
    inject_model_and_sidecar()
    usdc = client.get_usdc_balance()
    matic = client.get_matic_balance()
    logger.info("boot eoa=%s proxy=%s usdc=%.4f matic=%.6f",
                client.eoa_address, client.proxy_address, usdc, matic)
    if matic < settings.MATIC_LOW_THRESHOLD:
        notify.alert("MATIC low", f"balance={matic:.6f} (threshold {settings.MATIC_LOW_THRESHOLD})")


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[
            logging.FileHandler(settings.ORCHESTRATOR_LOG, mode="a", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def main():
    parser = argparse.ArgumentParser(description="Live Polymarket trader for modele_15m.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run one cycle, log everything, don't place a real order.")
    args = parser.parse_args()

    setup_logging()
    settings.assert_env_ready()
    client = from_settings()
    boot(client)

    cycle_settlement_counter: dict[str, int] = {}
    last_known_open_ms: int | None = None

    if args.dry_run:
        last_known_open_ms = run_cycle(client, None, cycle_settlement_counter, dry_run=True)
        logger.info("dry_run done last_open_ms=%s", last_known_open_ms)
        return

    while True:
        sleep_until_boundary()
        try:
            new_open = run_cycle(client, last_known_open_ms, cycle_settlement_counter, dry_run=False)
            if new_open is not None:
                last_known_open_ms = new_open
        except Exception as e:  # noqa: BLE001 — we log + re-raise so Docker can restart us
            logger.exception("cycle_crashed: %s", e)
            notify.alert("Daemon crashed", f"{type(e).__name__}: {e}")
            raise


if __name__ == "__main__":
    main()
