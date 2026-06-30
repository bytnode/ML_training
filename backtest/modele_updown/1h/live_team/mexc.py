"""Client REST MEXC Futures - le *seul* fichier du daemon qui touche directement
l'API MEXC (contract / private). Tous les autres modules (live.py, position.py,
EntryManager...) passent par MexcClient. Le jour ou l'API MEXC change, c'est le
seul fichier a auditer.

Meme philosophie que le polymarket.py du daemon 15min (un unique fichier-client),
mais cible MEXC Futures (contrats perpetuels USDT-margined), pas Polymarket.

Decision d'implementation : REST-polling uniquement (pas de WebSocket en v1).
Rationale : solidite avant tout. La cadence du daemon est 1-min, tres en-dessous
des rate limits MEXC ; le WS est une optimisation v2 documentee.

------------------------------------------------------------------------------
ENDPOINTS UTILISES
------------------------------------------------------------------------------
Publics (pas de signature) :
    GET  /api/v1/contract/detail?symbol=BTC_USDT      -> get_contract_detail
    GET  /api/v1/contract/ticker?symbol=BTC_USDT      -> get_ticker
    GET  /api/v1/contract/kline/{symbol}?interval=&start=&end=  -> get_klines

Prives (signes HMAC-SHA256) :
    GET  /api/v1/private/account/asset/{currency}     -> get_assets
    GET  /api/v1/private/position/open_positions      -> get_open_positions
    POST /api/v1/private/position/change_leverage     -> set_leverage
    POST /api/v1/private/order/create                 -> place_order
    POST /api/v1/private/order/cancel                 -> cancel_orders
    POST /api/v1/private/order/cancel_all             -> cancel_all
    GET  /api/v1/private/order/get/{order_id}         -> get_order

------------------------------------------------------------------------------
SIGNATURE HMAC-SHA256 (hex)
------------------------------------------------------------------------------
En-tetes prives :
    ApiKey          : la cle d'acces
    Request-Time    : timestamp ms (str)
    Signature       : HMAC-SHA256 hex (voir ci-dessous)
    Content-Type    : application/json
    Recv-Window     : fenetre de validite (ms)

Chaine a signer :
    POST        : accessKey + ts + jsonBody   (jsonBody = le body EXACT envoye,
                  byte-pour-byte ; on signe la meme string que celle postee).
    GET/DELETE  : accessKey + ts + paramString (params tries alpha, joints
                  "cle=valeur" par "&", valeurs url-encodees).
La signature est hex(HMAC_SHA256(secretKey, chaine)).

------------------------------------------------------------------------------
MAPPING side / type (MEXC Futures, mode hedge)
------------------------------------------------------------------------------
side (int) :
    1 = open long      (ouvrir long)
    3 = open short     (ouvrir short)
    2 = close short    (fermer short)
    4 = close long     (fermer long)
type (int) :
    1 = limit
    2 = post-only      (maker garanti, rejete si croiserait le book)
    5 = market
openType (int) :
    1 = isolated
    2 = cross
positionType (int) :
    1 = long
    2 = short
order state (get_order, int) :
    2 = uncompleted (en cours)
    3 = completed   (rempli)
    4 = cancelled

Voir la spec : docs/superpowers/specs/2026-06-18-mexc-live-team-1h-design.md
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import time
from typing import Any
from urllib.parse import quote

import requests

logger = logging.getLogger("live_team.mexc")

# --- Constantes par defaut (overridables via config_team / env) ---
DEFAULT_REST_HOST = "https://contract.mexc.com"  # FUTURES (api.mexc.com = SPOT) ; verifier avant le reel
DEFAULT_WS_URL = "wss://contract.mexc.com/edge"
DEFAULT_RECV_WINDOW = 10000          # ms : fenetre de validite de la requete signee

# Retry reseau : on retente sur erreur transport (timeout, conn reset) ET sur le
# code HTTP 510 (MEXC l'utilise pour "system busy" / rate-limit transitoire).
_MAX_RETRIES = 3
_BACKOFF_BASE = 0.5                   # s : backoff exponentiel 0.5, 1.0, 2.0...
_RETRY_HTTP_CODES = {510}

# Timeouts (s). Court sur le hot-path (lecture), un peu plus large sur l'ecriture.
_DEFAULT_READ_TIMEOUT = 5.0
_DEFAULT_WRITE_TIMEOUT = 10.0


class MexcError(Exception):
    """Toute erreur retournee par l'API MEXC (champ success=false / code != 0),
    ou une reponse inexploitable. Porte le code MEXC + le message brut."""

    def __init__(self, code: Any, message: str):
        self.code = code
        self.message = message
        super().__init__(f"MEXC error {code}: {message}")


def _normalize_kline(raw: Any) -> list[list[float]]:
    """Normalise la reponse kline MEXC en une liste de bars ASC.

    MEXC renvoie un dict de listes paralleles :
        {"time":[...], "open":[...], "high":[...], "low":[...],
         "close":[...], "vol":[...]}  -- time en SECONDES.
    On convertit en list de [open_ms, o, h, l, c, vol], temps x1000 (->ms),
    triee par open_ms croissant. Tolerant aux champs absents/longueurs inegales.
    """
    if not isinstance(raw, dict):
        return []
    times = raw.get("time") or []
    opens = raw.get("open") or []
    highs = raw.get("high") or []
    lows = raw.get("low") or []
    closes = raw.get("close") or []
    vols = raw.get("vol") or raw.get("volume") or []

    n = min(len(times), len(opens), len(highs), len(lows), len(closes))
    bars: list[list[float]] = []
    for i in range(n):
        try:
            open_ms = int(times[i]) * 1000          # secondes -> millisecondes
            o = float(opens[i])
            h = float(highs[i])
            low = float(lows[i])
            c = float(closes[i])
            v = float(vols[i]) if i < len(vols) else 0.0
        except (TypeError, ValueError):
            continue
        bars.append([open_ms, o, h, low, c, v])
    # MEXC renvoie deja en ASC, mais on garantit l'ordre par securite.
    bars.sort(key=lambda b: b[0])
    return bars


def _as_float(value: Any, default: float = 0.0) -> float:
    """Cast defensif : les champs numeriques MEXC arrivent parfois en str."""
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return default


class MexcClient:
    """Client REST MEXC Futures. Fichier-client unique, REST-polling.

    Construire via `MexcClient.from_env()`. Toute erreur API leve MexcError.
    En mode DRY_RUN (config_team.DRY_RUN), `place_order` signe mais ne POST pas.
    """

    def __init__(
        self,
        access_key: str,
        secret_key: str,
        rest_host: str = DEFAULT_REST_HOST,
        ws_url: str = DEFAULT_WS_URL,
        recv_window: int = DEFAULT_RECV_WINDOW,
        dry_run: bool = False,
        read_timeout: float = _DEFAULT_READ_TIMEOUT,
        write_timeout: float = _DEFAULT_WRITE_TIMEOUT,
    ):
        self.access_key = access_key
        self._secret_key = secret_key                 # garde en memoire uniquement
        self.rest_host = rest_host.rstrip("/")
        self.ws_url = ws_url
        self.recv_window = int(recv_window)
        self.dry_run = bool(dry_run)
        self.read_timeout = read_timeout
        self.write_timeout = write_timeout

        # Session keep-alive : reutilise la connexion TCP entre requetes.
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "live-team-mexc/1.0"})

        # Decalage horloge serveur-client (ms). Sync au boot ; resync ponctuel
        # possible si l'API rejette pour "request expired".
        self._time_offset_ms = 0
        try:
            self.sync_time()
        except Exception as e:                          # noqa: BLE001 - boot best-effort
            logger.warning("sync_time au boot a echoue (offset=0): %s", e)

    # ------------------------------------------------------------------
    #  Construction
    # ------------------------------------------------------------------
    @classmethod
    def from_env(cls) -> "MexcClient":
        """Construit le client depuis l'environnement + config_team.

        Lit MEXC_ACCESS_KEY / MEXC_SECRET_KEY (obligatoires) et reprend host /
        ws / recv_window / dry_run depuis config_team (lui-meme overridable par
        env). Leve RuntimeError en listant chaque variable manquante.
        """
        access_key = os.environ.get("MEXC_ACCESS_KEY")
        secret_key = os.environ.get("MEXC_SECRET_KEY")
        missing = [name for name, val in
                   (("MEXC_ACCESS_KEY", access_key), ("MEXC_SECRET_KEY", secret_key))
                   if not val]
        if missing:
            raise RuntimeError(
                f"MexcClient.from_env: variables d'environnement manquantes {missing}. "
                f"Renseigne-les dans live_team/.env."
            )

        # On lit la config si elle est importable ; sinon defauts + env directs.
        rest_host = DEFAULT_REST_HOST
        ws_url = DEFAULT_WS_URL
        recv_window = DEFAULT_RECV_WINDOW
        dry_run = os.environ.get("DRY_RUN", "false").lower() in ("1", "true", "yes", "y", "on")
        try:
            import config_team as cfg  # type: ignore
            rest_host = getattr(cfg, "REST_HOST", rest_host)
            ws_url = getattr(cfg, "WS_URL", ws_url)
            recv_window = getattr(cfg, "RECV_WINDOW", recv_window)
            dry_run = getattr(cfg, "DRY_RUN", dry_run)
        except Exception:                               # noqa: BLE001
            # config_team non importable (ex : appel isole) -> defauts + env.
            rest_host = os.environ.get("MEXC_REST_HOST", rest_host)
            ws_url = os.environ.get("MEXC_WS_URL", ws_url)

        return cls(
            access_key=access_key,
            secret_key=secret_key,
            rest_host=rest_host,
            ws_url=ws_url,
            recv_window=recv_window,
            dry_run=dry_run,
        )

    # ------------------------------------------------------------------
    #  Horloge
    # ------------------------------------------------------------------
    def sync_time(self) -> int:
        """Synchronise l'horloge avec le serveur MEXC et memorise l'offset (ms).

        Utilise GET /api/v1/contract/ping (renvoie {"data": <server_ms>}).
        Retourne l'offset serveur-client en ms.
        """
        url = f"{self.rest_host}/api/v1/contract/ping"
        local_before = int(time.time() * 1000)
        r = self.session.get(url, timeout=self.read_timeout)
        r.raise_for_status()
        body = r.json()
        # MEXC renvoie le temps serveur sous "data" (ou "serverTime" selon endpoint).
        server_ms = _as_int(body.get("data") if isinstance(body, dict) else None, 0)
        if server_ms <= 0 and isinstance(body, dict):
            server_ms = _as_int(body.get("serverTime"), 0)
        if server_ms <= 0:
            # Pas de temps serveur exploitable : on garde l'offset courant.
            return self._time_offset_ms
        local_after = int(time.time() * 1000)
        # On estime l'instant local au moment de la reponse serveur (milieu RTT).
        local_mid = (local_before + local_after) // 2
        self._time_offset_ms = server_ms - local_mid
        logger.debug("sync_time: offset serveur=%d ms", self._time_offset_ms)
        return self._time_offset_ms

    def _timestamp_ms(self) -> int:
        """Timestamp courant corrige de l'offset serveur (ms str pour Request-Time)."""
        return int(time.time() * 1000) + self._time_offset_ms

    # ------------------------------------------------------------------
    #  Signature
    # ------------------------------------------------------------------
    def _sign(self, payload: str) -> str:
        """HMAC-SHA256 hex de `payload` avec la cle secrete."""
        return hmac.new(
            self._secret_key.encode("utf-8"),
            payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    @staticmethod
    def _build_param_string(params: dict[str, Any]) -> str:
        """Serialise les params GET/DELETE : tri alpha, "cle=valeur" joints par
        "&", valeurs url-encodees. Cle ABSOLUMENT identique a celle envoyee dans
        l'URL (meme tri, meme encodage) sinon la signature ne matche pas."""
        items = sorted((k, params[k]) for k in params if params[k] is not None)
        return "&".join(f"{k}={quote(str(v), safe='')}" for k, v in items)

    def _private_headers(self, ts: int, signature: str) -> dict[str, str]:
        return {
            "ApiKey": self.access_key,
            "Request-Time": str(ts),
            "Signature": signature,
            "Content-Type": "application/json",
            "Recv-Window": str(self.recv_window),
        }

    # ------------------------------------------------------------------
    #  Transport bas-niveau (retry + interpretation)
    # ------------------------------------------------------------------
    def _send(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        signed: bool = False,
        idempotent: bool = True,
        timeout: float | None = None,
    ) -> Any:
        """Effectue une requete REST avec retry 3x backoff exponentiel sur erreur
        reseau et sur le code HTTP 510. Retourne le champ `data` de la reponse
        MEXC, ou leve MexcError.

        Pour POST signe, le body est serialise UNE SEULE FOIS et c'est cette
        string exacte qui est signee ET envoyee (contrainte MEXC : on signe le
        body byte-pour-byte).
        """
        method = method.upper()
        url = f"{self.rest_host}{path}"
        request_url = url
        params = params or {}
        if timeout is None:
            timeout = self.write_timeout if method in ("POST", "DELETE") else self.read_timeout

        # --- Preparation de la requete (body / query + signature) ---
        # On fige body_str / param_str AVANT la boucle de retry : retenter ne doit
        # PAS regenerer un timestamp/body different (idempotence cote signature).
        # En revanche le timestamp est rafraichi a chaque tentative pour eviter
        # un rejet "request expired" sur un retry tardif.
        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            ts = self._timestamp_ms()
            headers: dict[str, str] = {}
            body_str: str | None = None

            if method == "POST":
                # Body JSON compact et deterministe ; signe puis envoye tel quel.
                body_str = json.dumps(params, separators=(",", ":"), sort_keys=False)
                if signed:
                    signature = self._sign(f"{self.access_key}{ts}{body_str}")
                    headers = self._private_headers(ts, signature)
                else:
                    headers = {"Content-Type": "application/json"}
            else:  # GET / DELETE
                param_str = self._build_param_string(params)
                # On accole la query EXACTEMENT signee a l'URL (meme tri, meme
                # encodage) et on passe params=None a requests : sinon requests
                # re-encode le dict a sa facon et la query envoyee differe de la
                # string signee -> signature invalide.
                request_url = f"{url}?{param_str}" if param_str else url
                if signed:
                    signature = self._sign(f"{self.access_key}{ts}{param_str}")
                    headers = self._private_headers(ts, signature)

            try:
                if method == "POST":
                    resp = self.session.post(url, data=body_str, headers=headers,
                                             timeout=timeout)
                elif method == "DELETE":
                    resp = self.session.delete(request_url, headers=headers,
                                               timeout=timeout)
                else:
                    resp = self.session.get(request_url, headers=headers,
                                            timeout=timeout)
            except requests.exceptions.RequestException as e:
                # Erreur transport : retry avec backoff — SAUF si non-idempotent
                # (creation d'ordre) : un timeout peut masquer un ordre DEJA place,
                # re-poster doublerait l'ordre. On leve, le caller/reconcile arbitre.
                last_exc = e
                if idempotent and attempt < _MAX_RETRIES - 1:
                    delay = _BACKOFF_BASE * (2 ** attempt)
                    logger.warning("%s %s erreur reseau (tentative %d/%d), retry dans %.1fs: %s",
                                   method, path, attempt + 1, _MAX_RETRIES, delay, e)
                    time.sleep(delay)
                    continue
                raise MexcError("network", f"{method} {path}: {e}") from e

            # Code 510 = "system busy" transitoire cote MEXC -> retry (sauf
            # creation d'ordre non-idempotente : risque de double ordre).
            if idempotent and resp.status_code in _RETRY_HTTP_CODES and attempt < _MAX_RETRIES - 1:
                delay = _BACKOFF_BASE * (2 ** attempt)
                logger.warning("%s %s HTTP %d (tentative %d/%d), retry dans %.1fs",
                               method, path, resp.status_code, attempt + 1, _MAX_RETRIES, delay)
                time.sleep(delay)
                continue

            return self._interpret(method, path, resp)

        # Boucle epuisee sans retour ni exception explicite : ne devrait pas arriver.
        raise MexcError("network", f"{method} {path}: echec apres {_MAX_RETRIES} tentatives "
                                   f"(derniere: {last_exc})")

    @staticmethod
    def _interpret(method: str, path: str, resp: requests.Response) -> Any:
        """Interprete la reponse HTTP MEXC. Retourne le champ `data`, ou leve
        MexcError sur HTTP>=400, success=false, code!=0 ou JSON invalide."""
        # JSON d'abord (MEXC renvoie souvent un body JSON meme sur erreur).
        try:
            body = resp.json()
        except ValueError:
            if resp.status_code >= 400:
                raise MexcError(resp.status_code,
                                f"{method} {path}: HTTP {resp.status_code} {resp.text[:300]}")
            raise MexcError("decode", f"{method} {path}: reponse non-JSON: {resp.text[:300]}")

        if isinstance(body, dict):
            # Schema MEXC futures : {"success": bool, "code": int, "data": ...,
            # "message"/"msg": str}.
            success = body.get("success")
            code = body.get("code")
            message = body.get("message") or body.get("msg") or ""

            # code==0 ET/OU success==true => OK. On est tolerant : certains
            # endpoints publics ne renvoient pas "success".
            ok = (success is True) or (code in (0, "0")) or \
                 (success is None and code is None and resp.status_code < 400)
            if not ok:
                raise MexcError(code if code is not None else resp.status_code,
                                message or f"HTTP {resp.status_code}")
            if resp.status_code >= 400:
                raise MexcError(code if code is not None else resp.status_code,
                                message or f"HTTP {resp.status_code}")
            # data peut etre absent (ex: cancel renvoie parfois data au niveau racine).
            return body.get("data", body)

        # Reponse JSON non-dict (rare) : on remonte telle quelle si HTTP OK.
        if resp.status_code >= 400:
            raise MexcError(resp.status_code, f"{method} {path}: HTTP {resp.status_code}")
        return body

    # ==================================================================
    #  Endpoints PUBLICS (pas de signature)
    # ==================================================================
    def get_contract_detail(self, symbol: str) -> dict:
        """GET /api/v1/contract/detail?symbol= : details du contrat.

        Expose au moins : contract_size (float, BTC_USDT=0.0001), price_unit
        (float, increment de prix), price_scale (int), vol_scale (int),
        min_vol (float). Le `raw` complet est conserve sous "raw".
        """
        data = self._send("GET", "/api/v1/contract/detail",
                           params={"symbol": symbol}, signed=False)
        # detail peut etre une liste (tous les contrats) ou un dict (un contrat).
        if isinstance(data, list):
            data = next((d for d in data if d.get("symbol") == symbol), data[0] if data else {})
        if not isinstance(data, dict):
            raise MexcError("shape", f"get_contract_detail: reponse inattendue pour {symbol}")
        return {
            "symbol": data.get("symbol", symbol),
            "contract_size": _as_float(data.get("contractSize"), 0.0),
            "price_unit": _as_float(data.get("priceUnit"), 0.0),
            "price_scale": _as_int(data.get("priceScale"), 0),
            "vol_scale": _as_int(data.get("volScale"), 0),
            "min_vol": _as_float(data.get("minVol"), 1.0),
            "raw": data,
        }

    def get_ticker(self, symbol: str) -> dict:
        """GET /api/v1/contract/ticker?symbol= : dernier prix / fair price."""
        data = self._send("GET", "/api/v1/contract/ticker",
                           params={"symbol": symbol}, signed=False)
        if isinstance(data, list):
            data = next((d for d in data if d.get("symbol") == symbol), data[0] if data else {})
        if not isinstance(data, dict):
            raise MexcError("shape", f"get_ticker: reponse inattendue pour {symbol}")
        return {
            "symbol": data.get("symbol", symbol),
            "last_price": _as_float(data.get("lastPrice"), 0.0),
            "fair_price": _as_float(data.get("fairPrice"), 0.0),
            "raw": data,
        }

    def get_klines(self, symbol: str, interval: str,
                   start: int | None = None, end: int | None = None) -> list[list[float]]:
        """GET /api/v1/contract/kline/{symbol} : bougies du contrat.

        `interval` ex "Min1", "Min60" (format MEXC). `start`/`end` en SECONDES
        (l'API kline MEXC attend des secondes). Retourne une liste de bars ASC
        [open_ms, o, h, l, c, vol], temps converti en millisecondes.
        """
        params: dict[str, Any] = {"interval": interval}
        if start is not None:
            params["start"] = int(start)
        if end is not None:
            params["end"] = int(end)
        data = self._send("GET", f"/api/v1/contract/kline/{symbol}",
                          params=params, signed=False)
        return _normalize_kline(data)

    # ==================================================================
    #  Endpoints PRIVES (signes)
    # ==================================================================
    def get_assets(self, currency: str = "USDT") -> dict:
        """GET /api/v1/private/account/asset/{currency} : solde du compte futures."""
        data = self._send("GET", f"/api/v1/private/account/asset/{currency}", signed=True)
        if not isinstance(data, dict):
            raise MexcError("shape", f"get_assets: reponse inattendue pour {currency}")
        return {
            "currency": data.get("currency", currency),
            "available_balance": _as_float(data.get("availableBalance"), 0.0),
            "equity": _as_float(data.get("equity"), 0.0),
            "raw": data,
        }

    def get_open_positions(self, symbol: str | None = None) -> list[dict]:
        """GET /api/v1/private/position/open_positions : positions ouvertes.

        Normalise chaque position : position_id, symbol, position_type (1=long,
        2=short), hold_vol (float), open_avg_price (float), leverage.
        """
        params = {"symbol": symbol} if symbol else {}
        data = self._send("GET", "/api/v1/private/position/open_positions",
                          params=params, signed=True)
        if data is None:
            return []
        if isinstance(data, dict):
            data = [data]
        if not isinstance(data, list):
            raise MexcError("shape", "get_open_positions: reponse inattendue")
        positions: list[dict] = []
        for p in data:
            if not isinstance(p, dict):
                continue
            positions.append({
                "position_id": _as_int(p.get("positionId"), 0),
                "symbol": p.get("symbol"),
                "position_type": _as_int(p.get("positionType"), 0),
                "hold_vol": _as_float(p.get("holdVol"), 0.0),
                "open_avg_price": _as_float(p.get("openAvgPrice")
                                            if p.get("openAvgPrice") is not None
                                            else p.get("holdAvgPrice"), 0.0),
                "leverage": _as_int(p.get("leverage"), 0),
                "raw": p,
            })
        return positions

    def set_leverage(self, symbol: str, leverage: int, open_type: int,
                     position_type: int) -> dict:
        """POST /api/v1/private/position/change_leverage : regle le levier.

        Sans position ouverte, MEXC exige openType + symbol + positionType (en
        plus du leverage). On envoie toujours les 4 champs.
        """
        body = {
            "symbol": symbol,
            "leverage": int(leverage),
            "openType": int(open_type),
            "positionType": int(position_type),
        }
        return self._send("POST", "/api/v1/private/position/change_leverage",
                          params=body, signed=True)

    def place_order(
        self,
        symbol: str,
        side: int,
        vol: int,
        price: float | None,
        order_type: int,
        open_type: int = 1,
        leverage: int | None = None,
        reduce_only: bool = False,
        external_oid: str | None = None,
        position_id: int | None = None,
    ) -> dict:
        """POST /api/v1/private/order/create : place un ordre.

        side : 1=open long, 3=open short, 2=close short, 4=close long.
        order_type (-> champ MEXC `type`) : 1=limit, 2=post-only, 5=market.
        `price` arrondi cote appelant a price_unit ; omis si market (type 5).
        Retourne {"order_id": <str>, "ts": <ms>} (+ "raw").

        FICHIER CRITIQUE (argent reel) : si config_team.DRY_RUN, l'ordre est
        construit + signe via _send... NON. Pour garantir l'absence totale de
        POST en DRY_RUN, on n'appelle PAS _send : on construit le body, on le
        loggue, et on retourne {"order_id": "DRYRUN-"+external_oid, "dry_run": True}.
        """
        body: dict[str, Any] = {
            "symbol": symbol,
            "side": int(side),
            "type": int(order_type),
            "openType": int(open_type),
            "vol": int(vol),
        }
        # price omis pour un ordre market (type 5).
        if order_type != 5 and price is not None:
            body["price"] = price
        if leverage is not None:
            body["leverage"] = int(leverage)
        if reduce_only:
            body["reduceOnly"] = True
        if external_oid is not None:
            body["externalOid"] = external_oid
        if position_id is not None:
            body["positionId"] = int(position_id)

        if self.dry_run:
            # Aucun POST en DRY_RUN : on loggue le body qui AURAIT ete envoye.
            oid = external_oid or f"noext-{int(time.time() * 1000)}"
            logger.info("DRY-RUN place_order: %s (NON envoye)",
                        json.dumps(body, separators=(",", ":")))
            return {
                "order_id": f"DRYRUN-{oid}",
                "ts": self._timestamp_ms(),
                "dry_run": True,
                "raw": {"dry_run": True, "body": body},
            }

        data = self._send("POST", "/api/v1/private/order/create",
                          params=body, signed=True, idempotent=False)
        # MEXC renvoie l'orderId directement sous data (str/int) ou dans un dict.
        if isinstance(data, dict):
            order_id = data.get("orderId") or data.get("order_id") or data
            ts = _as_int(data.get("ts"), self._timestamp_ms())
        else:
            order_id = data
            ts = self._timestamp_ms()
        return {"order_id": str(order_id), "ts": ts, "raw": data}

    def cancel_orders(self, order_ids: list[str]) -> dict:
        """POST /api/v1/private/order/cancel : annule une liste d'ordres.

        Le body MEXC est un ARRAY d'orderId (pas un objet). On serialise donc
        directement la liste comme body signe.
        """
        # _send serialise params en JSON ; on passe la liste telle quelle pour
        # obtenir un body de type array.
        return self._send_array("/api/v1/private/order/cancel",
                                [str(o) for o in order_ids])

    def cancel_all(self, symbol: str) -> dict:
        """POST /api/v1/private/order/cancel_all : annule tous les ordres du symbole."""
        return self._send("POST", "/api/v1/private/order/cancel_all",
                          params={"symbol": symbol}, signed=True)

    def get_order(self, order_id: str) -> dict:
        """GET /api/v1/private/order/get/{order_id} : etat d'un ordre.

        Normalise : state (int : 2=uncompleted, 3=completed, 4=cancelled),
        deal_vol (float rempli), deal_avg_price (float), vol (commande).
        """
        data = self._send("GET", f"/api/v1/private/order/get/{order_id}", signed=True)
        if not isinstance(data, dict):
            raise MexcError("shape", f"get_order: reponse inattendue pour {order_id}")
        return {
            "order_id": str(data.get("orderId") or order_id),
            "state": _as_int(data.get("state"), 0),
            "deal_vol": _as_float(data.get("dealVol"), 0.0),
            "deal_avg_price": _as_float(data.get("dealAvgPrice"), 0.0),
            "vol": _as_float(data.get("vol"), 0.0),
            "raw": data,
        }

    # ------------------------------------------------------------------
    #  POST avec body de type array (cancel) -- signature sur le body exact
    # ------------------------------------------------------------------
    def _send_array(self, path: str, array_body: list[Any]) -> dict:
        """POST signe dont le body est un ARRAY JSON (cas /order/cancel).

        Reproduit la logique retry/signature de _send mais avec un body liste.
        On signe la string JSON exacte envoyee (accessKey + ts + jsonBody).
        """
        url = f"{self.rest_host}{path}"
        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            ts = self._timestamp_ms()
            body_str = json.dumps(array_body, separators=(",", ":"))
            signature = self._sign(f"{self.access_key}{ts}{body_str}")
            headers = self._private_headers(ts, signature)
            try:
                resp = self.session.post(url, data=body_str, headers=headers,
                                         timeout=self.write_timeout)
            except requests.exceptions.RequestException as e:
                last_exc = e
                if attempt < _MAX_RETRIES - 1:
                    delay = _BACKOFF_BASE * (2 ** attempt)
                    logger.warning("POST %s erreur reseau (tentative %d/%d), retry dans %.1fs: %s",
                                   path, attempt + 1, _MAX_RETRIES, delay, e)
                    time.sleep(delay)
                    continue
                raise MexcError("network", f"POST {path}: {e}") from e

            if resp.status_code in _RETRY_HTTP_CODES and attempt < _MAX_RETRIES - 1:
                delay = _BACKOFF_BASE * (2 ** attempt)
                logger.warning("POST %s HTTP %d (tentative %d/%d), retry dans %.1fs",
                               path, resp.status_code, attempt + 1, _MAX_RETRIES, delay)
                time.sleep(delay)
                continue

            return self._interpret("POST", path, resp)

        raise MexcError("network", f"POST {path}: echec apres {_MAX_RETRIES} tentatives "
                                   f"(derniere: {last_exc})")


if __name__ == "__main__":
    # Smoke test : lit les details du contrat + le ticker (endpoints publics,
    # pas besoin de cles). Les endpoints prives necessitent .env renseigne.
    import sys

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    symbol = "BTC_USDT"
    # On instancie en direct (sans cles) pour le test public.
    client = MexcClient(access_key="", secret_key="")
    try:
        detail = client.get_contract_detail(symbol)
        print(f"contract_detail {symbol}: contract_size={detail['contract_size']} "
              f"price_unit={detail['price_unit']} min_vol={detail['min_vol']}")
        ticker = client.get_ticker(symbol)
        print(f"ticker {symbol}: last={ticker['last_price']} fair={ticker['fair_price']}")
        klines = client.get_klines(symbol, "Min60", )
        print(f"klines Min60: {len(klines)} bars ; dernier={klines[-1] if klines else None}")
    except MexcError as e:
        print(f"MEXC error: {e}")
        sys.exit(1)
