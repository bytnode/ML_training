# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Scope

This is the **self-contained live trading daemon** for the 15min team model. It is the only thing under `live/`. The repo-root `CLAUDE.md` (one level up) is the authority for the broader project — read it for the training pipeline, the 15min/5m/30m sibling trees, the team coherence rule, and the V1/V2 sidecar shape. This file documents only what is local to `live/`. (This folder was flattened up from `live/modele_15m/` — there is now no `modele_15m` intermediate directory.)

## What lives here

The folder is flat and self-contained — no parent-directory imports, no symlinks, no shared state with `backtest/`:

- `live.py` — the loop. One file, ~one entry point.
- `polymarket.py` — **the single file** that touches Polymarket / Polygon / `py-clob-client-v2` / `web3` / `eth-account`. Lazy-imported only when `LIVE_MODE=true`. CLOB **V2** with `signature_type=3` (deposit-wallet) and market FAK orders.
- `live_helpers/` — `fetch.py` (Bitstamp public OHLC), `features_up.py`, `features_down.py`, `notify.py` (Telegram).
- `modele_up/` + `modele_down/` — **copies** of the runtime files needed at inference: `config.py`, `btc_updown_model.json`, `btc_updown_model.thresholds.json`, and (modele_down only) `data.py` for feature helpers imported by `features_down.py`. **These are copies of the training packages in `backtest/modele_updown/15min/modele_*/`** — after a retrain, manually copy `btc_updown_model.json` + `btc_updown_model.thresholds.json` from `backtest/` into `live/modele_*/` here, then `docker compose restart live-team`.
- `Dockerfile` + `docker-compose.yml` + `Dockerfile.dockerignore` — `live-team` service. Build context is **this folder itself** (`build: .`). Same image for both modes — flip `LIVE_MODE` in `.env` and restart.
- `.env` (gitignored) — Telegram creds + (when `LIVE_MODE=true`) Polymarket / Polygon / relayer creds + tunables. There is no checked-in `.env.example` in this folder; copy keys from the root `CLAUDE.md`'s env-var list.

## Two modes, one env var

| `LIVE_MODE` | What runs | Deps loaded | Logs written |
|-------------|-----------|-------------|--------------|
| `false` (default / absent) | Paper trading — model + decision flow only | `polymarket.py` never imported | `logs/live_predictions.csv` only |
| `true` | Same flow + real FAK BUY on the matching `btc-updown-15m-<unix_s>` Polymarket market + on-chain redeem of winners via Polymarket's relayer | `py-clob-client-v2`, `web3`, `eth-account`, `polymarket.py` | `logs/live_predictions.csv` **and** `logs/bets.csv` (full lifecycle: `pending → win/loss/stuck`) |

`logs/bets.csv` paper rows are written **complete** (decision + outcome) at resolution time with `order_status="paper"`. Real rows follow the pending lifecycle. Schema is `live.BETS_COLUMNS` (23 real-order columns + 13 appended context/resolution columns).

CLI : the only flag is `--dry-run` (real-mode but skip the actual `POST /order`). Mode itself is **not** a CLI flag — it lives in `.env`.

## The cycle (per 15m UTC boundary)

1. Sleep until next `xx:00 / xx:15 / xx:30 / xx:45 UTC` boundary.
2. `freshness_retry` to confirm Bitstamp published the just-closed bar.
3. *Real-mode only* — `resolve_pending_bets`: Polymarket settlement first, Bitstamp fallback after 1 cycle, `stuck` after 6. Winning bets are redeemed.
4. Resolve the previous cycle's paper prediction (fills `resolved_correct`).
5. Fetch fresh OHLCV on 4 timeframes (`15m`, `5m`, `30m`, `1h`), build features both sides, predict, apply the team coherence rule (`team_decision`: up-only→`UP`, down-only→`DOWN`, **both fire→`CANCEL`**, neither→`PASS`), append one row to `live_predictions.csv`, render Rich panels. Per-TF OHLCV buffers are cached at module level and refreshed incrementally (`INCREMENTAL_REFRESH_N_BARS` tail bars) each cycle after a cold-start warmup.
6. *Real-mode only* — on `UP` / `DOWN`, `place_real_bet` + append one row to `bets.csv`.
7. Telegram notify via `live_helpers/notify.py`.

Real-mode step 3 also runs **once at startup** to catch stale pending bets across restarts.

## Polymarket auto-redeem (real-mode only)

Deposit-wallet redemption is operator-gated on-chain (`DepositWallet.execute` is `OnlyFactory`; `DepositWalletFactory.proxy` is `OnlyOperator`). The daemon cannot call the on-chain redeem directly. Instead it **builds + EIP-712-signs a `redeemPositions` batch and submits it to Polymarket's relayer** (`POST {RELAYER_HOST}/submit`, `type:"WALLET"`).

Requires `RELAYER_API_KEY` in `.env` (get one at polymarket.com → Settings → API Keys). Without it, `redeem()` returns `MANUAL_REDEEM_REQUIRED` and Telegram-alerts you to claim by hand.

## Bet sizing (real-mode only)

`place_real_bet` sizes each bet as `BET_FRACTION` of the live USDC wallet balance, floored at `BET_MIN_USDC` and capped at `DEFAULT_BET_MAX_FRACTION` (0.95) of the wallet. Defaults: `BET_FRACTION=0.10`, `BET_MIN_USDC=2.50`. `SLIPPAGE_BPS` controls the FAK limit price tolerance. All three are read from `.env` per cycle (no rebuild needed). Threshold/edge logic (`_bucket_threshold`, `_predict_one`) is a deliberate exact mirror of `backtest/.../api.py` — keep them identical.

## Telegram

Two optional bots, both wired in `live_helpers/notify.py`:

- `TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID` — normal cycle notifications (paper resolution, `send_bet_placed`, `send_bet_resolved_real`, `send_stuck`, `send_low_matic`).
- `TELEGRAM_ERROR_BOT_TOKEN` + `TELEGRAM_ERROR_CHAT_ID` — separate bot wired as a `logging.Handler` that catches every `logger.error(...)` / `logger.exception(...)` from the daemon. **Any new error site picks it up automatically** — just call `logger.error` / `logger.exception`, no manual wiring.

If a token/chat-id pair is missing the loop still runs and that notify path becomes a silent no-op.

## Geo-block

**France and most of Western EU are geo-blocked by Polymarket.** `LIVE_MODE=true` only works from a non-blocked-region VPS. Paper mode runs anywhere. See `docs/superpowers/specs/2026-05-14-polymarket-live-betting-design.md` at the repo root for design notes.

## Running

```powershell
# Host (paper mode by default)
cd live
python live.py
python live.py --dry-run    # only meaningful when LIVE_MODE=true

# Docker
cd live
docker compose build
docker compose up -d
docker compose logs -f live-team    # Rich panels render thanks to tty: true
docker compose restart live-team    # after a model retrain
```

`docker-compose.yml` mounts `./logs:/app/logs` so the two CSVs survive rebuilds, and reads `./.env`. No port is published — the container only does outbound calls (Bitstamp + Telegram + Polymarket CLOB/Polygon RPC when real).

### Dependencies & tests

There is **no `requirements.txt`, `pyproject.toml`, test suite, or linter config** in this folder — by repo convention deps are pinned inline in the `Dockerfile` (`requests`, `pandas`, `numpy`, `xgboost`, `scikit-learn`, `scipy`, `rich`, plus real-mode-only `py-clob-client-v2`, `web3`, `eth-account`). To run on the host, `pip install` that pinned set yourself; the Polymarket trio is only needed for `LIVE_MODE=true`. Verification is by running the daemon and watching the Rich panels / CSVs, not by unit tests.

## What NOT to do here

- Don't add Polymarket / `web3` imports outside `polymarket.py` — keep paper mode dependency-free.
- Don't symlink or import from `backtest/` — this folder ships independently. Update the model artifact copies manually after a retrain.
- Don't add a `--live` / `--paper` CLI flag — mode lives in `.env` so swapping is one file edit + `docker compose restart`, no Dockerfile CMD change ([[feedback_env_vars_over_cli_flags]]).
- Don't rename the folder or change the Dockerfile build context — `build: .` and the flat `COPY` layout depend on it.
