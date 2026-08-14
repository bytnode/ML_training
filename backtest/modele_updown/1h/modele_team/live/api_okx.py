"""api_okx.py -- SEUL fichier de live/ qui touche l'API OKX.

Client REST maison pour OKX API v5 (SWAP perpetuel, mode HEDGE, marge ISOLATED).
Signature HMAC-SHA256 (base64), meme schema que okx_fees.py (modele_team/) mais
duplique volontairement ici (petit helper, pas la peine de partager -- meme
convention que _wilder_dmi/_bucket_threshold ailleurs dans ce repo).

Convention HEDGE : pour ouvrir un LONG -> side="buy", posSide="long". Pour
ouvrir un SHORT -> side="sell", posSide="short". Pour FERMER un LONG ->
side="sell", posSide="long" (reduceOnly implicite via posSide en mode hedge,
PAS besoin du flag reduceOnly qui est reserve au mode net). Pour fermer un
SHORT -> side="buy", posSide="short".

DRY_RUN (cf config_live.py / .env) : si True, toute methode d'ECRITURE
(place_order / cancel_order / amend_order / set_leverage / set_position_mode)
logue ce qu'elle AURAIT fait et renvoie un dict simule, SANS appeler l'API.
Les methodes de LECTURE (get_positions / get_balance / get_ticker / get_order /
get_candles) appellent toujours l'API reelle (lecture seule, sans risque),
meme en DRY_RUN -- necessaire pour que le cycle live tourne "pour de vrai"
en mode observation.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import time
from datetime import datetime, timezone

import requests

logger = logging.getLogger("live.api_okx")

REST_HOST = "https://www.okx.com"
_TIMEOUT_S = 10.0


def _iso_timestamp_ms() -> str:
    now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"


class OkxError(RuntimeError):
    """Leve sur toute reponse OKX code != '0' (hors lecture best-effort)."""


class OkxClient:
    def __init__(self, api_key: str, api_secret: str, passphrase: str,
                dry_run: bool = True):
        self.api_key = api_key
        self.api_secret = api_secret
        self.passphrase = passphrase
        self.dry_run = dry_run
        self._session = requests.Session()

    # ------------------------------------------------------------------ #
    #  Signature / requete bas niveau
    # ------------------------------------------------------------------ #
    def _sign(self, timestamp: str, method: str, request_path: str, body: str) -> str:
        msg = f"{timestamp}{method}{request_path}{body}".encode("utf-8")
        digest = hmac.new(self.api_secret.encode("utf-8"), msg, hashlib.sha256).digest()
        return base64.b64encode(digest).decode("utf-8")

    def _request(self, method: str, path: str, params: dict | None = None,
                body_obj: dict | None = None) -> dict:
        """Requete signee generique. `path` SANS query string ; `params` est
        serialise en query string pour GET (inclus dans le sign)."""
        query = ""
        if params:
            items = [f"{k}={v}" for k, v in params.items() if v is not None]
            query = "?" + "&".join(items) if items else ""
        request_path = f"{path}{query}"
        body = json.dumps(body_obj, separators=(",", ":")) if body_obj else ""
        timestamp = _iso_timestamp_ms()
        signature = self._sign(timestamp, method, request_path, body)
        headers = {
            "OK-ACCESS-KEY": self.api_key,
            "OK-ACCESS-SIGN": signature,
            "OK-ACCESS-TIMESTAMP": timestamp,
            "OK-ACCESS-PASSPHRASE": self.passphrase,
            "Content-Type": "application/json",
        }
        url = f"{REST_HOST}{request_path}"
        r = self._session.request(method, url, headers=headers,
                                  data=body if body else None, timeout=_TIMEOUT_S)
        try:
            r.raise_for_status()
        except requests.exceptions.HTTPError:
            # BUG FIXE (2026-07-15, audit complet) : raise_for_status() jette
            # AVANT toute lecture du corps -- le sCode/sMsg d'OKX (ex. distinguer
            # cle invalide / IP non whitelistee / signature invalide) etait
            # perdu, ne laissant que "401 Client Error" cote appelant. Le corps
            # est souvent exploitable meme sur un code d'erreur HTTP (OKX y met
            # son propre message).
            logger.error(f"{method} {request_path} -> HTTP {r.status_code} : {r.text[:500]}")
            raise
        payload = r.json()
        if not isinstance(payload, dict) or payload.get("code") != "0":
            raise OkxError(f"{method} {request_path} -> {payload}")
        return payload

    # ------------------------------------------------------------------ #
    #  Lecture (toujours reelle, meme en DRY_RUN)
    # ------------------------------------------------------------------ #
    def get_positions(self, inst_id: str | None = None) -> list[dict]:
        """GET /api/v5/account/positions -- liste des positions ouvertes.
        Champs cles par position : instId, posSide ('long'/'short'), pos
        (taille, signee selon posSide), avgPx, upl (PnL non realise, devise
        de marge), uplRatio (PnL% -- deja leverage, cf position.py), lever,
        margin, mgnMode."""
        params = {"instType": "SWAP"}
        if inst_id:
            params["instId"] = inst_id
        payload = self._request("GET", "/api/v5/account/positions", params=params)
        return [p for p in payload.get("data", []) if float(p.get("pos", "0") or 0) != 0]

    def get_balance(self, ccy: str = "USDT") -> float:
        """GET /api/v5/account/balance -- solde disponible (equity) en `ccy`."""
        payload = self._request("GET", "/api/v5/account/balance", params={"ccy": ccy})
        data = payload.get("data", [])
        if not data:
            return 0.0
        details = data[0].get("details", [])
        for d in details:
            if d.get("ccy") == ccy:
                return float(d.get("eq", "0") or 0.0)
        return 0.0

    def get_ticker(self, inst_id: str) -> dict:
        """GET /api/v5/market/ticker -- {bidPx, askPx, last, ts (ms, str)}."""
        payload = self._request("GET", "/api/v5/market/ticker", params={"instId": inst_id})
        data = payload.get("data", [])
        if not data:
            raise OkxError(f"ticker vide pour {inst_id}")
        return data[0]

    def get_instrument(self, inst_id: str) -> dict:
        """GET /api/v5/public/instruments -- endpoint PUBLIC (non signe).
        Champs cles : ctVal (valeur d'1 contrat, ex. '0.01' BTC pour
        BTC-USDT-SWAP), lotSz (increment de taille), minSz (taille min en
        contrats), tickSz (increment de prix). Necessaire pour convertir un
        notionnel USD en nombre de CONTRATS avant place_order (sz)."""
        r = self._session.get(f"{REST_HOST}/api/v5/public/instruments",
                              params={"instType": "SWAP", "instId": inst_id}, timeout=_TIMEOUT_S)
        r.raise_for_status()
        payload = r.json()
        data = payload.get("data", [])
        if not isinstance(payload, dict) or payload.get("code") != "0" or not data:
            raise OkxError(f"get_instrument({inst_id}) -> {payload}")
        return data[0]

    def get_order(self, inst_id: str, ord_id: str) -> dict:
        """GET /api/v5/trade/order -- champs cles : state
        ('live'/'partially_filled'/'filled'/'canceled'), fillPx, accFillSz,
        avgPx.

        DRY_RUN (cf _dry_run_stub) : un ordId 'DRYRUN-...' n'existe PAS cote
        OKX -- repond localement 'live' (jamais rempli) sans appel API, pour
        que advance_pending_entry simule proprement la fenetre d'attente puis
        bascule sur le fallback taker (lui aussi stubbe) a l'echeance."""
        if str(ord_id).startswith("DRYRUN-"):
            return {"ordId": ord_id, "instId": inst_id, "state": "live"}
        payload = self._request("GET", "/api/v5/trade/order",
                                params={"instId": inst_id, "ordId": ord_id})
        data = payload.get("data", [])
        if not data:
            raise OkxError(f"ordre {ord_id} introuvable pour {inst_id}")
        return data[0]

    # ------------------------------------------------------------------ #
    #  Ecriture (no-op + log si DRY_RUN)
    # ------------------------------------------------------------------ #
    def _dry_run_stub(self, action: str, **kwargs) -> dict:
        logger.info(f"[DRY_RUN] {action} (aucun appel OKX) params={kwargs}")
        stub = {"dry_run": True, "action": action, **kwargs}
        if action == "place_order":
            # BUG FIXE (audit 2026-07-10) : sans ordId utilisable ici, get_order
            # (methode de LECTURE, jamais stubbee) etait appelee avec un ID vide
            # a CHAQUE tick tant qu'une entree maker simulee restait "en cours" --
            # vrai appel API signe pour rien + exception loggee inutilement. Un
            # ordId prefixe DRYRUN- laisse get_order le reconnaitre et repondre
            # localement (cf get_order ci-dessous), sans jamais toucher l'API.
            stub["ordId"] = f"DRYRUN-{int(time.time() * 1000)}"
        return stub

    def set_position_mode(self, pos_mode: str = "long_short_mode") -> dict:
        if self.dry_run:
            return self._dry_run_stub("set_position_mode", posMode=pos_mode)
        return self._request("POST", "/api/v5/account/set-position-mode",
                             body_obj={"posMode": pos_mode})

    def set_leverage(self, inst_id: str, lever: int, mgn_mode: str = "isolated",
                     pos_side: str | None = None) -> dict:
        body = {"instId": inst_id, "lever": str(lever), "mgnMode": mgn_mode}
        if pos_side:
            body["posSide"] = pos_side
        if self.dry_run:
            return self._dry_run_stub("set_leverage", **body)
        return self._request("POST", "/api/v5/account/set-leverage", body_obj=body)

    def place_order(self, inst_id: str, side: str, pos_side: str, ord_type: str,
                    sz: str, px: str | None = None, td_mode: str = "isolated",
                    cl_ord_id: str | None = None) -> dict:
        """POST /api/v5/trade/order. side: 'buy'/'sell' ; pos_side:
        'long'/'short' (mode HEDGE -- determine seul si c'est une ouverture ou
        une fermeture, cf convention en tete de module) ; ord_type:
        'market'/'limit'/'post_only'/'fok'/'ioc' ; sz: taille en CONTRATS
        (string) ; px requis si ord_type != 'market'."""
        body = {
            "instId": inst_id, "tdMode": td_mode, "side": side,
            "posSide": pos_side, "ordType": ord_type, "sz": sz,
        }
        if px is not None:
            body["px"] = px
        if cl_ord_id:
            body["clOrdId"] = cl_ord_id
        if self.dry_run:
            return self._dry_run_stub("place_order", **body)
        payload = self._request("POST", "/api/v5/trade/order", body_obj=body)
        data = payload.get("data", [{}])[0]
        if data.get("sCode") not in (None, "0"):
            raise OkxError(f"place_order rejete : {data}")
        return data

    def cancel_order(self, inst_id: str, ord_id: str) -> dict:
        body = {"instId": inst_id, "ordId": ord_id}
        if self.dry_run:
            return self._dry_run_stub("cancel_order", **body)
        return self._request("POST", "/api/v5/trade/cancel-order", body_obj=body)

    def amend_order(self, inst_id: str, ord_id: str, new_px: str) -> dict:
        body = {"instId": inst_id, "ordId": ord_id, "newPx": new_px}
        if self.dry_run:
            return self._dry_run_stub("amend_order", **body)
        return self._request("POST", "/api/v5/trade/amend-order", body_obj=body)

    def close_position_market(self, inst_id: str, pos_side: str, sz: str,
                              td_mode: str = "isolated", cl_ord_id: str | None = None) -> dict:
        """Sortie market immediate (toujours taker, cf CLAUDE.md : la sortie
        est TOUJOURS taker, jamais simulee en maker). Convention HEDGE : fermer
        un LONG -> side='sell' ; fermer un SHORT -> side='buy'."""
        side = "sell" if pos_side == "long" else "buy"
        return self.place_order(inst_id, side=side, pos_side=pos_side,
                                ord_type="market", sz=sz, td_mode=td_mode, cl_ord_id=cl_ord_id)
