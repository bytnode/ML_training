"""notify.py — notifications Telegram (2 bots) pour le daemon live MEXC.

Deux bots indépendants :
  - BOT PRINCIPAL  (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID)             -> events opérationnels
  - BOT D'ERREUR   (TELEGRAM_ERROR_BOT_TOKEN / TELEGRAM_ERROR_CHAT_ID) -> tout logger.error/exception

Dégradation gracieuse : si un couple token/chat-id manque, l'envoi correspondant
devient un no-op silencieux et la boucle continue. Aucune exception n'est jamais
propagée depuis ce module (un échec Telegram ne doit pas tuer le daemon).
"""

import logging
import os

import requests

logger = logging.getLogger(__name__)

_TIMEOUT_S = 3
_MAX_LEN = 3800   # cap Telegram = 4096 ; on garde la QUEUE du message (root cause)


def _creds(prefix_token, prefix_chat):
    return os.getenv(prefix_token), os.getenv(prefix_chat)


def _post(token, chat_id, text):
    """Envoi bas-niveau. Avale toute exception (no-op si creds absentes/erreur réseau)."""
    if not token or not chat_id:
        return
    if len(text) > _MAX_LEN:
        text = "…" + text[-_MAX_LEN:]   # garde la fin (souvent la root cause)
    try:
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": "HTML",
                  "disable_web_page_preview": True},
            timeout=_TIMEOUT_S,
        )
    except Exception:   # noqa: BLE001 — notify ne doit JAMAIS propager
        pass


def _send_main(text):
    token, chat = _creds("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID")
    _post(token, chat, text)


# ============================================================
#  Events opérationnels (bot principal)
# ============================================================

def send_startup(balance, symbol, mode):
    _send_main(
        f"🟢 <b>live_team MEXC démarré</b>\n"
        f"symbol: <code>{symbol}</code>\n"
        f"mode: <b>{mode}</b>\n"
        f"solde futures: <b>{balance:.2f} USDT</b>"
    )


def send_position_opened(side, entry, vol, notional):
    arrow = "📈 LONG" if side == "up" else "📉 SHORT"
    _send_main(
        f"{arrow} <b>position ouverte</b>\n"
        f"entrée: <code>{entry:.2f}</code>\n"
        f"vol: <b>{vol}</b> contrats (~{notional:.2f} USDT notionnel)"
    )


def send_position_closed(side, exit_reason, entry, exit_price, pp, pnl_usdc, balance,
                         mfe_pp=0.0, mae_pp=0.0, mfe_usdc=0.0, mae_usdc=0.0):
    sign = "✅" if pnl_usdc >= 0 else "❌"
    arrow = "LONG" if side == "up" else "SHORT"
    _send_main(
        f"{sign} <b>{arrow} fermée</b> — <i>{exit_reason}</i>\n"
        f"entrée <code>{entry:.2f}</code> → sortie <code>{exit_price:.2f}</code>\n"
        f"mouvement: <b>{pp * 100:+.3f}%</b> prix\n"
        f"📈 plus haut profit atteint: <b>{mfe_pp * 100:+.3f}%</b> "
        f"(<b>{mfe_usdc:+.2f} USDT</b>)\n"
        f"📉 plus forte perte atteinte: <b>{mae_pp * 100:+.3f}%</b> "
        f"(<b>{mae_usdc:+.2f} USDT</b>)\n"
        f"PnL: <b>{pnl_usdc:+.2f} USDT</b>\n"
        f"solde: <b>{balance:.2f} USDT</b>"
    )


def send_entry_missed(side):
    arrow = "LONG" if side == "up" else "SHORT"
    _send_main(f"⏭️ <b>entrée {arrow} ratée</b> (3 tentatives maker non remplies → opportunité annulée)")


# ============================================================
#  Bot d'erreur séparé (logging.Handler auto-câblé)
# ============================================================

class _TelegramErrorHandler(logging.Handler):
    """Route tout logger.error()/exception() vers le bot d'erreur. Recursion-safe."""

    def __init__(self):
        super().__init__(level=logging.ERROR)
        self._sending = False

    def emit(self, record):
        if self._sending:
            return   # évite la récursion si _post logue une erreur
        token, chat = _creds("TELEGRAM_ERROR_BOT_TOKEN", "TELEGRAM_ERROR_CHAT_ID")
        if not token or not chat:
            return
        self._sending = True
        try:
            msg = self.format(record)
            _post(token, chat, f"🛑 <b>{record.levelname}</b>\n<pre>{_escape(msg)}</pre>")
        except Exception:   # noqa: BLE001
            try:
                print(f"[notify] échec envoi erreur Telegram: {record.getMessage()}")
            except Exception:
                pass
        finally:
            self._sending = False


def _escape(text):
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


_error_handler_installed = False


def install_error_handler():
    """À appeler EN PREMIER dans main() : capte même les erreurs de bootstrap."""
    global _error_handler_installed
    if _error_handler_installed:
        return
    handler = _TelegramErrorHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(name)s\n%(message)s"))
    logging.getLogger().addHandler(handler)
    _error_handler_installed = True
