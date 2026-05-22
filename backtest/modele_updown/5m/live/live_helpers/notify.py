"""Telegram resolution notifier for live.py.

Sends one message per resolved bet (UP or DOWN that was placed in the previous
cycle and is now closed). PASS / CANCEL cycles do not trigger anything.

Credentials come from env vars (`TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`). If
either is missing, `send_resolution()` becomes a silent no-op so the live loop
keeps running even without Telegram configured.
"""
from __future__ import annotations

import logging
import os
import sys

import requests

logger = logging.getLogger("live.notify")

TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"
HTTP_TIMEOUT_S = 10
# Telegram caps text at 4096 chars; leave headroom for HTML tags + headers.
TELEGRAM_MAX_CHARS = 3800


def _credentials() -> tuple[str | None, str | None]:
    return os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")


def _error_credentials() -> tuple[str | None, str | None]:
    """Credentials for the SEPARATE error-only bot. Independent of the main bot
    so you can route errors to a different chat (or different team member)."""
    return (
        os.environ.get("TELEGRAM_ERROR_BOT_TOKEN"),
        os.environ.get("TELEGRAM_ERROR_CHAT_ID"),
    )


def is_configured() -> bool:
    token, chat_id = _credentials()
    return bool(token and chat_id)


def _send_raw(text: str) -> bool:
    """POST one message to Telegram. Returns True on success, False otherwise."""
    token, chat_id = _credentials()
    if not (token and chat_id):
        return False
    url = TELEGRAM_API.format(token=token)
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    try:
        r = requests.post(url, json=payload, timeout=HTTP_TIMEOUT_S)
        if r.status_code == 404:
            logger.warning(
                "Telegram returned 404 - your TELEGRAM_BOT_TOKEN is likely wrong "
                "(typo / extra whitespace / deleted bot). Expected format: "
                "'<digits>:<letters-and-numbers>'."
            )
            return False
        if r.status_code == 400 and "chat not found" in r.text.lower():
            logger.warning(
                "Telegram returned 400 'chat not found' - your TELEGRAM_CHAT_ID "
                "is wrong, or you haven't sent a /start to the bot yet."
            )
            return False
        if r.status_code >= 400:
            logger.warning("Telegram returned %s: %s", r.status_code, r.text[:200])
            return False
        return True
    except requests.RequestException as e:
        logger.warning("Telegram send failed: %s", e)
        return False


def _fmt_pct(x: float) -> str:
    return f"{x*100:.2f}%"


def _fmt_usd(x: float) -> str:
    return f"{x:,.0f}".replace(",", " ")  # space thousands separator


def send_resolution(
    cycle_n: int,
    bar_close_ts_iso: str,
    direction: str,           # "UP" or "DOWN"
    confidence: float,        # 0..1, the confidence at which we took the bet
    window_open_iso: str,
    window_close_iso: str,
    open_price: float,
    close_price: float,
    correct: bool,
    running_bets: int,
    running_wr: float | None,
) -> None:
    """Send a resolution notif. No-op if Telegram is not configured.

    Never raises - Telegram failures degrade gracefully so the live cycle
    keeps running.
    """
    if not is_configured():
        return

    verdict_emoji = "🟢" if correct else "🔴"
    verdict_word  = "WON" if correct else "LOST"
    dir_arrow = "▲ UP" if direction == "UP" else "▼ DOWN"
    pct_move = (close_price - open_price) / open_price * 100 if open_price else 0.0
    pct_sign = "+" if pct_move >= 0 else ""

    wr_line = (f"{running_bets} bets · WR {_fmt_pct(running_wr)}"
               if running_wr is not None else f"{running_bets} bets")

    text = (
        f"<b>{verdict_emoji} BET {verdict_word}</b>\n"
        f"\n"
        f"<pre>"
        f"cycle    : #{cycle_n} closed at {bar_close_ts_iso}\n"
        f"direction: {dir_arrow}\n"
        f"confidence: {_fmt_pct(confidence)}\n"
        f"window   : {window_open_iso} → {window_close_iso}\n"
        f"open     : {_fmt_usd(open_price)} USD\n"
        f"close    : {_fmt_usd(close_price)} USD  ({pct_sign}{pct_move:.2f}%)\n"
        f"verdict  : {'✓ correct' if correct else '✗ wrong'}\n"
        f"\n"
        f"running  : {wr_line}"
        f"</pre>"
    )
    _send_raw(text)


def _sanity_check_credentials() -> None:
    """Detect the common 'token and chat_id swapped' mistake and warn loudly."""
    token, chat_id = _credentials()
    if not (token and chat_id):
        return
    # Bot tokens always contain ':' (format: '<digits>:<letters/digits>').
    # Chat IDs are pure numbers (sometimes negative for groups).
    if ":" not in token and ":" in chat_id:
        logger.warning(
            "TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID look SWAPPED in your .env "
            "(token must contain ':', chat_id is a plain number). "
            "Fix the .env and `docker compose restart`."
        )


def send_startup(live_mode: bool = False, usdc_balance: float | None = None,
                  matic_balance: float | None = None) -> None:
    """Optional 'I'm up' notif at container boot. Helpful to know the daemon
    actually started without waiting 15min for the first cycle.

    In real-bet mode (LIVE_MODE=true), wallet balances are included in the message.
    """
    if not is_configured():
        logger.info("Telegram not configured (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID); "
                    "resolution notifs disabled.")
        return
    _sanity_check_credentials()
    if live_mode:
        usdc_line = f"USDC : {usdc_balance:.2f}" if usdc_balance is not None else "USDC : n/a"
        matic_line = f"MATIC: {matic_balance:.4f}" if matic_balance is not None else "MATIC: n/a"
        text = (
            f"<b>🟢 LIVE TEAM started</b> (LIVE_MODE=<b>true</b>)\n"
            f"<pre>"
            f"Mode     : REAL Polymarket orders\n"
            f"{usdc_line}\n"
            f"{matic_line}"
            f"</pre>"
        )
    else:
        text = ("<b>🟢 LIVE TEAM started</b> (LIVE_MODE=<i>false</i>)\n"
                "Paper trading on btcusd. Resolution notifs will arrive as bets close.")
    ok = _send_raw(text)
    if ok:
        logger.info("Telegram configured; startup notif sent (live_mode=%s).", live_mode)
    else:
        logger.warning("Telegram configured but startup notif FAILED to send "
                        "(see warning above). Resolution notifs will keep retrying.")


# ============================================================
#  Real-bet (Polymarket) events. Only fired when LIVE_MODE=true.
# ============================================================

def _dir_arrow(direction: str) -> str:
    return "▲ UP" if direction == "UP" else "▼ DOWN"


def send_bet_placed(direction: str, usdc_amount: float, fill_price: float | None,
                    confidence: float, target_open_iso: str, target_close_iso: str,
                    order_id: str | None, status: str,
                    shares_filled: float | None = None,
                    error_msg: str = "") -> None:
    """Sent once per real order attempt - filled, rejected, failed, or skipped.

    `status` is the normalized one from polymarket.place_fak_order:
    'filled' | 'rejected' | 'failed' | 'skipped_market_not_found' | 'skipped_balance_low'.
    """
    if not is_configured():
        return

    if status == "filled":
        emoji, headline = "💸", "BET PLACED"
        body_extra = (
            f"fill_price: {fill_price:.4f} USD/share\n"
            f"shares    : {shares_filled:.2f}"
        ) if fill_price is not None else "fill_price: n/a"
    elif status == "rejected":
        emoji, headline = "⚠️", "BET REJECTED"
        body_extra = f"reason: {error_msg}"
    elif status == "failed":
        emoji, headline = "🔥", "BET FAILED"
        body_extra = f"reason: {error_msg}"
    elif status.startswith("skipped"):
        emoji, headline = "⏭️", "BET SKIPPED"
        body_extra = f"reason: {status}"
    else:
        emoji, headline = "❔", f"BET {status.upper()}"
        body_extra = f"reason: {error_msg or status}"

    text = (
        f"<b>{emoji} {headline}</b>\n"
        f"<pre>"
        f"direction : {_dir_arrow(direction)}\n"
        f"confidence: {_fmt_pct(confidence)}\n"
        f"size      : {usdc_amount:.2f} USDC\n"
        f"window    : {target_open_iso} → {target_close_iso}\n"
        f"{body_extra}\n"
        f"order_id  : {order_id or 'n/a'}"
        f"</pre>"
    )
    _send_raw(text)


def send_bet_resolved_real(direction: str, outcome: str, source: str,
                            realized_pnl_usdc: float | None,
                            fill_price: float | None, target_close_iso: str,
                            running_bets: int, running_pnl: float) -> None:
    """Sent when a real bet's outcome lands in bets.csv (win/loss).

    `outcome` is one of: 'win' / 'loss' / 'bitstamp_win' / 'bitstamp_loss'.
    `source` is one of: 'polymarket' / 'bitstamp_fallback'.
    """
    if not is_configured():
        return

    is_win = outcome.endswith("win")
    emoji = "💰" if is_win else "💔"
    verdict = "WON" if is_win else "LOST"

    source_label = "Polymarket" if source == "polymarket" else "Bitstamp fallback"
    pnl_str = f"{realized_pnl_usdc:+.2f}" if realized_pnl_usdc is not None else "n/a"
    fill_str = f"{fill_price:.4f}" if fill_price is not None else "n/a"

    # On a win, winnings sit as CTF shares in the deposit wallet until claimed.
    # Auto-redemption is not implemented for the CLOB V2 deposit-wallet flow
    # (see polymarket.redeem), so remind the operator to claim it by hand.
    redeem_note = ("\n\n⚠️ CLAIM: redeem this win manually on polymarket.com"
                   if is_win else "")

    text = (
        f"<b>{emoji} REAL BET {verdict}</b>\n"
        f"<pre>"
        f"direction : {_dir_arrow(direction)}\n"
        f"fill_price: {fill_str} USD/share\n"
        f"pnl       : {pnl_str} USDC\n"
        f"source    : {source_label}\n"
        f"closed at : {target_close_iso}\n"
        f"\n"
        f"running   : {running_bets} bets · pnl {running_pnl:+.2f} USDC"
        f"{redeem_note}"
        f"</pre>"
    )
    _send_raw(text)


def send_stuck(slug: str, target_close_iso: str, cycles_pending: int) -> None:
    """Sent once when a bet has been pending for > SETTLEMENT_STUCK_AFTER_CYCLES."""
    if not is_configured():
        return
    text = (
        f"<b>🛑 BET STUCK</b>\n"
        f"<pre>"
        f"slug       : {slug}\n"
        f"closed at  : {target_close_iso}\n"
        f"pending for: {cycles_pending} cycles\n"
        f"\n"
        f"Polymarket has not resolved this market and the Bitstamp fallback\n"
        f"was inconclusive. Manual investigation required."
        f"</pre>"
    )
    _send_raw(text)


def send_low_matic(matic_balance: float, threshold: float = 0.5) -> None:
    """One-time alert at boot (or after a failed redeem) when MATIC is low."""
    if not is_configured():
        return
    text = (
        f"<b>⛽ LOW MATIC</b>\n"
        f"<pre>"
        f"matic_balance: {matic_balance:.4f} (threshold: {threshold})\n"
        f"\n"
        f"redeemPositions() needs gas. Top up the EOA so winning shares can be\n"
        f"converted back to USDC."
        f"</pre>"
    )
    _send_raw(text)


# ============================================================
#  Dedicated error-only bot
# ============================================================
#
# Architecture:
#   The main bot (TELEGRAM_BOT_TOKEN) handles operational events: bet placed,
#   bet resolved, stuck bets, low MATIC, startup.
#
#   A SEPARATE bot (TELEGRAM_ERROR_BOT_TOKEN) handles errors only. It is wired
#   to the root Python logger via TelegramErrorHandler, so every
#   logger.error(...) and logger.exception(...) anywhere in the project
#   automatically sends a message to this bot. No need to instrument each error
#   site manually.
#
# Recursion safety: failures in the error-bot send path itself fall back to
# print(..., file=sys.stderr) so they never trigger another logger.error.

def is_error_configured() -> bool:
    token, chat_id = _error_credentials()
    return bool(token and chat_id)


def _send_error_raw(text: str) -> bool:
    """POST one message to the error bot. Short timeout (3s) so a slow Telegram
    never blocks the live cycle. No-op if creds missing.

    Returns True on 2xx, False otherwise. Never raises.
    """
    token, chat_id = _error_credentials()
    if not (token and chat_id):
        return False
    url = TELEGRAM_API.format(token=token)
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    try:
        r = requests.post(url, json=payload, timeout=3)
        if r.status_code >= 400:
            # Use print(stderr) NOT logger.warning, to keep the error-bot path
            # outside Python's logging system (which the handler is hooked to).
            print(f"[error-bot] Telegram returned {r.status_code}: {r.text[:200]}",
                  file=sys.stderr)
            return False
        return True
    except requests.RequestException as e:
        print(f"[error-bot] send failed: {e}", file=sys.stderr)
        return False


def _html_escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def send_error(level: str, logger_name: str, message: str,
                exc_text: str | None = None) -> None:
    """Send one error message to the dedicated error bot. No-op if not configured.

    Called by TelegramErrorHandler.emit, but can also be invoked directly if
    you want to force an alert from code that doesn't use Python's logging
    (e.g., a hook that catches stderr).
    """
    if not is_error_configured():
        return

    body = message
    if exc_text:
        body = f"{message}\n\n{exc_text}"
    # Trim - keep the tail because that's usually where the root cause sits.
    if len(body) > TELEGRAM_MAX_CHARS:
        body = "... (truncated) ...\n" + body[-(TELEGRAM_MAX_CHARS - 100):]

    emoji = "🔥" if level == "CRITICAL" else "⚠️"
    text = (
        f"<b>{emoji} {level}</b> <code>{_html_escape(logger_name)}</code>\n"
        f"<pre>{_html_escape(body)}</pre>"
    )
    _send_error_raw(text)


class TelegramErrorHandler(logging.Handler):
    """logging.Handler that pipes ERROR-and-above records to the error bot.

    Recursion-safe: the emit path swallows its own exceptions and never calls
    logger.error / logger.warning (which would re-enter this same handler).
    """

    def __init__(self, level: int = logging.ERROR):
        super().__init__(level=level)
        self._sending = False

    def emit(self, record: logging.LogRecord) -> None:
        if self._sending:
            return
        if not is_error_configured():
            return
        try:
            self._sending = True
            try:
                msg = self.format(record)
            except Exception as fmt_err:   # noqa: BLE001 - we *must* swallow
                msg = f"<format error: {fmt_err}> raw={record.msg!r}"
            exc_text = None
            if record.exc_info:
                import traceback
                exc_text = "".join(traceback.format_exception(*record.exc_info))
            send_error(
                level=record.levelname,
                logger_name=record.name,
                message=msg,
                exc_text=exc_text,
            )
        except Exception as e:   # noqa: BLE001
            print(f"[error-bot] handler emit failed: {e}", file=sys.stderr)
        finally:
            self._sending = False


def install_error_handler(level: int = logging.ERROR) -> bool:
    """Attach a TelegramErrorHandler to the root logger so every
    logger.error(...) / logger.exception(...) in the project sends a Telegram
    message to the dedicated error bot.

    No-op if TELEGRAM_ERROR_BOT_TOKEN / TELEGRAM_ERROR_CHAT_ID are unset.

    Returns True if the handler was installed, False otherwise. Sends a one-time
    'error bot online' ping on successful install so the user knows the channel
    is alive without waiting for the first real failure.
    """
    if not is_error_configured():
        logger.info("Error bot not configured (TELEGRAM_ERROR_BOT_TOKEN / "
                    "TELEGRAM_ERROR_CHAT_ID); errors stay in stdout only.")
        return False
    h = TelegramErrorHandler(level=level)
    h.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger().addHandler(h)
    logger.info("Error bot configured. Errors at %s and above will be sent.",
                logging.getLevelName(level))
    # One-time alive ping. Bypasses the handler (direct _send_error_raw call)
    # so it doesn't show up as a fake ERROR record.
    _send_error_raw(
        "<b>🟢 Error bot online</b>\n"
        "<pre>This is the dedicated error channel for live-team.\n"
        "Errors from live.py and polymarket.py will arrive here.</pre>"
    )
    return True
