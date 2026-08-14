"""notify.py -- notifications Telegram du daemon live OKX.

Meme pattern que les autres daemons du repo : un bot "operationnel" (ouverture/
fermeture de position, resume de cycle) + un bot "erreur" optionnel et separe,
cable via un logging.Handler pour que TOUT logger.error/logger.exception emis
n'importe ou dans live/ remonte automatiquement -- pas besoin d'appeler notify
explicitement a chaque site d'erreur.

Best-effort : credentials absentes -> no-op silencieux (le daemon tourne quand
meme). Erreur reseau Telegram -> logged, jamais leve (une notif ratee ne doit
jamais faire planter le cycle de trading)."""
from __future__ import annotations

import logging
import os

import requests

logger = logging.getLogger("live.notify")

_TIMEOUT_S = 10.0


def _send(token: str | None, chat_id: str | None, text: str) -> None:
    if not token or not chat_id:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
            timeout=_TIMEOUT_S,
        )
    except Exception as exc:  # noqa: BLE001 -- notif best-effort, jamais bloquant
        logger.warning(f"Envoi Telegram echoue : {exc}")


def _op_creds() -> tuple[str | None, str | None]:
    return os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")


def _error_creds() -> tuple[str | None, str | None]:
    """Bot d'erreur separe si configure ; sinon retombe sur le bot operationnel
    (au moins UNE notification part, jamais un silence total sur une erreur)."""
    token = os.environ.get("TELEGRAM_ERROR_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_ERROR_CHAT_ID")
    if token and chat_id:
        return token, chat_id
    return _op_creds()


def send_bet_placed(pos_side: str, mode: str, sz: str, px: float | None) -> None:
    token, chat_id = _op_creds()
    px_txt = f"{px:.2f}" if px is not None else "market"
    _send(token, chat_id,
          f"\U0001F7E2 <b>Position ouverte</b> {pos_side.upper()} "
          f"({mode}) sz={sz} px={px_txt}")


def send_position_widened(pos_side: str) -> None:
    token, chat_id = _op_creds()
    _send(token, chat_id,
          f"\U0001F535 <b>Position elargie</b> {pos_side.upper()} "
          f"(signal du meme sens confirme, position deja en profit -- "
          f"TP/trailing/hold etendus)")


def send_bet_resolved(pos_side: str, reason: str, pp: float, entry_px: float, exit_px: float,
                       peak_pp: float, worst_pp: float, notional_usd: float,
                       wins: int, losses: int, balance: float | None) -> None:
    emoji = "✅" if pp > 0 else "❌"
    side_label = "LONG" if pos_side == "long" else "SHORT"
    total = wins + losses
    wr_pct = (wins / total * 100.0) if total else 0.0
    pnl_usdt = pp * notional_usd
    peak_usdt = peak_pp * notional_usd
    worst_usdt = worst_pp * notional_usd
    balance_txt = f"{balance:.2f} USDT" if balance is not None else "n/a"
    token, chat_id = _op_creds()
    _send(token, chat_id,
          f"{emoji} <b>{side_label} fermee</b> — {reason}\n"
          f"entree {entry_px:.2f} → sortie {exit_px:.2f}\n"
          f"mouvement: {pp*100:+.2f}% prix\n"
          f"\U0001F4C8 plus haut profit atteint: {peak_pp*100:+.2f}% ({peak_usdt:+.2f} USDT)\n"
          f"\U0001F4C9 plus forte perte atteinte: {worst_pp*100:+.2f}% ({worst_usdt:+.2f} USDT)\n"
          f"PnL: {pnl_usdt:+.2f} USDT\n"
          f"WR depuis lancement: {wins}/{total} ({wr_pct:.1f}%)\n"
          f"capital : {balance_txt}")


def send_position_adopted(pos_side: str, avg_px: float, sz: str) -> None:
    token, chat_id = _error_creds()
    _send(token, chat_id,
          f"❗ <b>Position adoptee</b> {pos_side.upper()} avg_px={avg_px:.2f} sz={sz}\n"
          f"(ouverte chez OKX mais absente de l'etat local au dernier tick -- "
          f"redemarrage sur etat corrompu/tronque, ou crash pendant l'ouverture. "
          f"Surveillance SL/TP/trailing/timestop reprend a partir de maintenant, "
          f"horizon repart de zero.)")


def send_entry_abandoned(pos_side: str, error: str) -> None:
    token, chat_id = _error_creds()
    _send(token, chat_id,
          f"\U0001F6A8 <b>Entree abandonnee</b> {pos_side.upper()} : fallback taker "
          f"rejete apres expiration de la fenetre maker, aucune position ouverte : {error}")


def send_desync(pos_side: str) -> None:
    token, chat_id = _error_creds()
    _send(token, chat_id,
          f"⚠️ <b>Desync position</b> {pos_side.upper()} attendue "
          f"ouverte mais absente chez OKX (liquidation ou fermeture manuelle ?).")


def send_close_failed(pos_side: str, reason: str, error: str) -> None:
    token, chat_id = _error_creds()
    _send(token, chat_id,
          f"\U0001F6A8 <b>Echec fermeture</b> {pos_side.upper()} (reason={reason}) : {error}")


def send_cycle_error(context: str, error: str) -> None:
    token, chat_id = _error_creds()
    _send(token, chat_id, f"\U0001F6A8 <b>Erreur cycle</b> [{context}] : {error}")


class TelegramErrorHandler(logging.Handler):
    """Handler logging : route tout record >= ERROR vers le bot d'erreur (ou
    le bot operationnel en fallback). Attache-le au root logger dans live.py."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            token, chat_id = _error_creds()
            _send(token, chat_id, f"\U0001F6A8 <b>{record.name}</b>\n{self.format(record)}")
        except Exception:  # noqa: BLE001 -- un handler de logging ne doit JAMAIS lever
            # BUG FIXE (2026-07-15, audit erreurs silencieuses) : un `pass` nu
            # ici ne laissait AUCUNE trace si _error_creds()/self.format(record)
            # echouait de facon inattendue (au-dela de ce que _send() capture
            # deja) -- pas meme dans la console/live.log. `self.handleError`
            # est le mecanisme standard de logging.Handler pour ce cas exact :
            # imprime sur stderr, ne leve jamais, ne re-boucle jamais dans
            # d'autres handlers (contrairement a un logger.error() ici, qui
            # remonterait au root logger et re-declencherait CE MEME handler).
            self.handleError(record)
