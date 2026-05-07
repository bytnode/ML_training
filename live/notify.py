"""Client Telegram pour les notifs Aure.

Best-effort : ne raise jamais. Retourne True si la notif est partie, False sinon.
"""
import logging
import os
import time

import requests

from settings import (
    TELEGRAM_BOT_TOKEN_ENV,
    TELEGRAM_CHAT_ID_ENV,
    TELEGRAM_TIMEOUT,
)

_log = logging.getLogger(__name__)

# Etat global :
# - _bot_token, _chat_id : lus a chaque appel pour permettre des changements
#   d'env vars sans relancer le daemon (rare mais utile en debug).
# - _disabled : passe a True si une 4xx d'auth survient, pour eviter de
#   spammer Telegram qui rejettera tout. Reset au prochain restart.
_disabled = False


def _read_env() -> tuple[str | None, str | None]:
    return os.environ.get(TELEGRAM_BOT_TOKEN_ENV), os.environ.get(TELEGRAM_CHAT_ID_ENV)


def is_configured() -> bool:
    token, chat_id = _read_env()
    return bool(token and chat_id) and not _disabled


def send_message(text: str) -> bool:
    """Envoie text via sendMessage (parse_mode=HTML). Retourne True si OK.

    Si bot token / chat_id sont missing : log WARN une fois, retourne False
    sans faire d'appel reseau.
    Si 4xx d'auth : log ERROR, set _disabled=True, retourne False.
    Si timeout / 5xx : retry 1x, puis log WARN et retourne False.
    """
    global _disabled
    if _disabled:
        return False

    token, chat_id = _read_env()
    if not token or not chat_id:
        _log.warning("telegram_disabled (missing %s or %s env var)",
                     TELEGRAM_BOT_TOKEN_ENV, TELEGRAM_CHAT_ID_ENV)
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text, "parse_mode": "HTML"}

    for attempt in (1, 2):
        try:
            resp = requests.post(url, json=payload, timeout=TELEGRAM_TIMEOUT)
        except (requests.Timeout, requests.ConnectionError) as e:
            if attempt == 1:
                time.sleep(2)
                continue
            _log.warning("telegram_failed network attempt=%d err=%s", attempt, e)
            return False

        if resp.status_code == 200:
            return True
        if 400 <= resp.status_code < 500:
            _log.error("telegram_misconfigured status=%d body=%s — disabling notif until restart",
                       resp.status_code, resp.text[:200])
            _disabled = True
            return False
        # 5xx : retry une fois
        if attempt == 1:
            time.sleep(2)
            continue
        _log.warning("telegram_failed status=%d", resp.status_code)
        return False

    return False


def format_decision_message(features: dict, decision: dict, price: float) -> str:
    """Genere le HTML d'une notif Aure.

    features : dict avec seq_len, seq_dir (1=up, 0=down), volatility_state, is_weekend.
    decision : dict avec bet_direction ('up'|'down'), confidence, effective_threshold.
    price : prix BTC au moment du setup (= close de la derniere bougie du streak).
    """
    bet = decision["bet_direction"]
    emoji = "🟢" if bet == "up" else "🔴"
    action = "BUY" if bet == "up" else "SELL"

    streak_dir_word = "up" if features["seq_dir"] == 1 else "down"
    vol_word = {0: "low", 1: "normal", 2: "high"}.get(features["volatility_state"], "?")
    weekend_word = "yes" if features["is_weekend"] == 1 else "no"

    timestamp_ms = features["timestamp"]
    from datetime import datetime, timezone
    ts_str = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    return (
        f"{emoji} <b>{action} BTC</b>\n"
        f"<i>Aure setup</i>\n"
        f"Price : <b>${price:,.0f}</b>\n"
        f"Confidence : {decision['confidence']*100:.1f}% (th {decision['effective_threshold']*100:.1f}%)\n"
        f"Streak : {features['seq_len']} candles {streak_dir_word}\n"
        f"Bucket : vol={vol_word} weekend={weekend_word}\n"
        f"Time : {ts_str}"
    )


if __name__ == "__main__":
    # Smoke test : verifie format_decision_message sur un setup factice,
    # ET tente un send_message si les env vars sont presentes.
    fake_features = {
        "timestamp": 1746000000000, "seq_len": 3, "seq_dir": 0,
        "volatility_state": 1, "is_weekend": 0,
    }
    fake_decision = {
        "bet_direction": "up", "confidence": 0.624, "effective_threshold": 0.580,
    }
    msg = format_decision_message(fake_features, fake_decision, price=67234.0)
    print("--- formatted message ---")
    print(msg)
    print("-------------------------")
    if is_configured():
        print("Telegram configured. Sending...")
        ok = send_message("(smoke test) " + msg)
        print(f"  send_message returned: {ok}")
    else:
        print("Telegram not configured (env vars manquantes). Skipping send.")
