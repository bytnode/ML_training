"""okx_fees.py -- taux de frais OKX REELS (maker/taker), calcules depuis l'API
plutot qu'estimes en dur dans config_main.py (campagne symetrie 2026-07-09,
remplace mexc_fees.py : l'exchange live devient OKX, donc la calibration des
frais du backtest doit suivre le meme exchange -- cf CLAUDE.md).

Duplique volontairement sa propre logique de signature HMAC-SHA256 plutot que
de partager avec live/api_okx.py (meme convention que _wilder_dmi/_bucket_threshold
ailleurs dans ce repo : petit helper, pas la peine de partager entre packages --
modele_team ne DOIT JAMAIS importer live/, cf CLAUDE.md racine "le backtest n'a
besoin de rien de live").

Lecture SEULE (un seul GET signe, aucun ordre place) -- une cle API OKX scopee
"lecture seule" suffit amplement (pas besoin du droit trading).

Lit OKX_API_KEY / OKX_API_SECRET / OKX_API_PASSPHRASE depuis l'environnement
PROCESSUS -- les memes variables que live/.env. Si ce fichier est deja charge
dans ton shell (ou si tu exportes les cles manuellement), ce module les
recupere gratuitement ; sinon (cles absentes, reseau indisponible, erreur API)
`fetch_real_fee_rate` retourne None -- jamais d'exception, le caller (main.py)
decide du fallback sur les constantes MAKER_FEE_PER_SIDE/FEE_PER_SIDE de
config_main.py et l'affiche CLAIREMENT (jamais silencieux : un fallback perime
ne doit jamais se faire passer pour une valeur reelle).

Reponse OKX : `GET /api/v5/account/trade-fee` renvoie `maker`/`taker` en
STRING, negatifs quand c'est un cout pour le compte (ex. "-0.0005" = 0.05%
preleve), positifs en cas de rebate. On prend la valeur ABSOLUE : ce module
n'exprime qu'un TAUX (>=0), le signe cout/rebate n'a pas de sens dans
config_main.py (MAKER_FEE_PER_SIDE/FEE_PER_SIDE sont deja des taux positifs
soustraits du gross). Si ton compte est en rebate net (maker negatif de
signe... releve improbable mais a verifier une fois la cle API active), le abs()
ci-dessous masquerait un GAIN de frais -- verifie le signe brut retourne par
l'API si `fetch_real_fee_rate` te semble surestimer les frais.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import os
from datetime import datetime, timezone

import requests

REST_HOST = "https://www.okx.com"
_TIMEOUT_S = 5.0


def _iso_timestamp_ms() -> str:
    """Timestamp ISO 8601 millisecondes attendu par OK-ACCESS-TIMESTAMP,
    ex. '2026-07-09T12:34:56.789Z'."""
    now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"


def _sign(secret: str, timestamp: str, method: str, request_path: str, body: str = "") -> str:
    """base64(HMAC_SHA256(secret, timestamp + method + requestPath + body)) --
    schema de signature OKX API v5 (identique GET/POST)."""
    msg = f"{timestamp}{method}{request_path}{body}".encode("utf-8")
    digest = hmac.new(secret.encode("utf-8"), msg, hashlib.sha256).digest()
    return base64.b64encode(digest).decode("utf-8")


def fetch_real_fee_rate(inst_id: str = "BTC-USDT-SWAP") -> dict | None:
    """GET /api/v5/account/trade-fee (instType=SWAP, instId) -- taux EFFECTIF du
    compte (apres remise de tier eventuelle), pas le bareme "de base". Retourne
    {"maker": float, "taker": float} (taux >=0, abs() du signe cout/rebate OKX --
    voir la note en tete de module) ou None si cles absentes / appel en echec
    (best-effort, jamais bloquant pour le backtest)."""
    access_key = os.environ.get("OKX_API_KEY")
    secret_key = os.environ.get("OKX_API_SECRET")
    passphrase = os.environ.get("OKX_API_PASSPHRASE")
    if not access_key or not secret_key or not passphrase:
        return None
    try:
        request_path = f"/api/v5/account/trade-fee?instType=SWAP&instId={inst_id}"
        timestamp = _iso_timestamp_ms()
        signature = _sign(secret_key, timestamp, "GET", request_path)
        headers = {
            "OK-ACCESS-KEY": access_key,
            "OK-ACCESS-SIGN": signature,
            "OK-ACCESS-TIMESTAMP": timestamp,
            "OK-ACCESS-PASSPHRASE": passphrase,
            "Content-Type": "application/json",
        }
        r = requests.get(f"{REST_HOST}{request_path}", headers=headers, timeout=_TIMEOUT_S)
        r.raise_for_status()
        body = r.json()
        if not isinstance(body, dict) or body.get("code") != "0":
            return None
        data = body.get("data") or []
        if not data:
            return None
        row = data[0]
        maker = abs(float(row.get("maker", 0.0) or 0.0))
        taker = abs(float(row.get("taker", 0.0) or 0.0))
        if maker <= 0 or taker <= 0:
            return None
        return {"maker": maker, "taker": taker}
    except Exception:      # noqa: BLE001 - lookup best-effort, jamais d'exception vers le caller
        return None


if __name__ == "__main__":
    rates = fetch_real_fee_rate()
    if rates is None:
        print("OKX_API_KEY/OKX_API_SECRET/OKX_API_PASSPHRASE absentes de l'environnement, "
              "ou appel API en echec.")
    else:
        print(f"maker={rates['maker']*100:.4f}%  taker={rates['taker']*100:.4f}%  (taux reel du compte)")
