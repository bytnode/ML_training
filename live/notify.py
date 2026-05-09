"""Telegram notifier — best-effort, never blocks the trading loop."""
from __future__ import annotations

import logging

import requests

import settings

logger = logging.getLogger("live.notify")

_TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"


def _enabled() -> bool:
    return bool(settings.TELEGRAM_BOT_TOKEN and settings.TELEGRAM_CHAT_ID)


def send(text: str) -> None:
    """Send a Telegram message. Never raises."""
    if not _enabled():
        return
    url = _TELEGRAM_API.format(token=settings.TELEGRAM_BOT_TOKEN)
    try:
        requests.post(
            url,
            json={
                "chat_id": settings.TELEGRAM_CHAT_ID,
                "text": text,
                "parse_mode": "HTML",
            },
            timeout=settings.TELEGRAM_TIMEOUT,
        )
    except requests.exceptions.RequestException as e:
        logger.warning("telegram_failed reason=%s", e)


def bet_placed(direction: str, bet_size_usdc: float, fill_price: float | None,
               confidence: float, effective_threshold: float,
               target_window_start_ms: int) -> None:
    fp = f"{fill_price:.4f}" if fill_price is not None else "N/A"
    emoji = "🟢" if direction == "up" else "🔴"
    send(
        f"{emoji} <b>BET PLACED {direction.upper()}</b>\n"
        f"Size  : ${bet_size_usdc:.2f}\n"
        f"Fill  : {fp} per share\n"
        f"Conf  : {confidence * 100:.2f}% (th {effective_threshold * 100:.2f}%)\n"
        f"Window: {target_window_start_ms}"
    )


def bet_resolved(direction: str, outcome: str, realized_pnl_usdc: float) -> None:
    emoji = "✅" if outcome == "win" else "❌"
    send(
        f"{emoji} <b>BET {outcome.upper()}</b> ({direction})\n"
        f"PnL: ${realized_pnl_usdc:+.2f}"
    )


def alert(reason: str, details: str = "") -> None:
    send(f"⚠️ <b>ALERT</b> {reason}\n{details}")
