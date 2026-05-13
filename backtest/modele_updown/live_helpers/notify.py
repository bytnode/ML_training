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

import requests

logger = logging.getLogger("live.notify")

TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"
HTTP_TIMEOUT_S = 10


def _credentials() -> tuple[str | None, str | None]:
    return os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")


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


def send_startup() -> None:
    """Optional 'I'm up' notif at container boot. Helpful to know the daemon
    actually started without waiting 15min for the first cycle.
    """
    if not is_configured():
        logger.info("Telegram not configured (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID); "
                    "resolution notifs disabled.")
        return
    _sanity_check_credentials()
    ok = _send_raw("<b>🟢 LIVE TEAM started</b>\nPaper trading on btcusd. Resolution "
                    "notifs will arrive as bets close.")
    if ok:
        logger.info("Telegram configured; startup notif sent.")
    else:
        logger.warning("Telegram configured but startup notif FAILED to send "
                        "(see warning above). Resolution notifs will keep retrying.")
