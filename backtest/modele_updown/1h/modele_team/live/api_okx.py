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
from datetime import datetime, timedelta, timezone

import requests

logger = logging.getLogger("live.api_okx")

REST_HOST = "https://www.okx.com"

# Pas (ms) des granularites OKX utilisees ici -- cf get_candles_range.
_BAR_MS = {"1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000,
           "30m": 1_800_000, "1H": 3_600_000, "4H": 14_400_000}
_TIMEOUT_S = 10.0


def _iso_timestamp_ms(offset_ms: int = 0) -> str:
    # offset_ms : derive locale vs serveur OKX (cf OkxClient.sync_time, point #11
    # audit 2026-08-03) -- applique a l'horodatage signe pour qu'une horloge hote
    # legerement derivee ne fasse pas rejeter toute requete signee (code 50102).
    now = datetime.now(timezone.utc) + timedelta(milliseconds=offset_ms)
    return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"


# Code OKX renvoye quand l'OK-ACCESS-TIMESTAMP signe est hors de la fenetre
# de tolerance (~30s) du serveur : "Timestamp request expired". Arrive en
# HTTP 401 avec ce code au niveau racine du corps.
_TIMESTAMP_EXPIRED_CODE = "50102"


def _is_timestamp_expired(r: "requests.Response") -> bool:
    """True si la reponse OKX est un rejet 50102 (horodatage expire). Robuste
    a un corps non-JSON (fallback sur le texte brut)."""
    try:
        body = r.json()
    except ValueError:
        return _TIMESTAMP_EXPIRED_CODE in (r.text or "")
    if isinstance(body, dict):
        if body.get("code") == _TIMESTAMP_EXPIRED_CODE:
            return True
        for d in body.get("data") or []:
            if isinstance(d, dict) and d.get("sCode") == _TIMESTAMP_EXPIRED_CODE:
                return True
    return False


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
        self._time_offset_ms = 0   # cale par sync_time() (point #11 audit 2026-08-03)

    # ------------------------------------------------------------------ #
    #  Synchronisation d'horloge (public/time, non signe) -- point #11
    # ------------------------------------------------------------------ #
    def sync_time(self) -> float:
        """GET /api/v5/public/time -- cale `self._time_offset_ms` sur l'heure
        serveur OKX et renvoie la derive locale (skew) en SECONDES (peut etre
        negatif). A appeler au demarrage : OKX rejette une requete signee a
        plus de ~30s de son heure serveur, et une horloge hote derivee ferait
        echouer SILENCIEUSEMENT toute ecriture (order/cancel) une fois en reel."""
        r = self._session.get(f"{REST_HOST}/api/v5/public/time", timeout=_TIMEOUT_S)
        r.raise_for_status()
        data = r.json().get("data", [])
        if not data:
            raise OkxError("public/time -> reponse vide")
        server_ms = int(data[0]["ts"])
        local_ms = int(time.time() * 1000)
        self._time_offset_ms = server_ms - local_ms
        return self._time_offset_ms / 1000.0

    # ------------------------------------------------------------------ #
    #  Signature / requete bas niveau
    # ------------------------------------------------------------------ #
    def _sign(self, timestamp: str, method: str, request_path: str, body: str) -> str:
        msg = f"{timestamp}{method}{request_path}{body}".encode("utf-8")
        digest = hmac.new(self.api_secret.encode("utf-8"), msg, hashlib.sha256).digest()
        return base64.b64encode(digest).decode("utf-8")

    def _request(self, method: str, path: str, params: dict | None = None,
                body_obj: "dict | list | None" = None) -> dict:
        """Requete signee generique. `path` SANS query string ; `params` est
        serialise en query string pour GET (inclus dans le sign). `body_obj`
        peut etre un dict (majorite des POST) OU une liste (ex. cancel-algos,
        dont le corps OKX est un tableau)."""
        query = ""
        if params:
            items = [f"{k}={v}" for k, v in params.items() if v is not None]
            query = "?" + "&".join(items) if items else ""
        request_path = f"{path}{query}"
        body = json.dumps(body_obj, separators=(",", ":")) if body_obj else ""
        url = f"{REST_HOST}{request_path}"

        # BUG FIXE (audit 2026-08-13) : self-heal du code 50102 "Timestamp request
        # expired". `sync_time()` ne cale l'offset horloge qu'UNE fois au boot ;
        # sur un daemon tournant des jours cet offset derive (correction NTP par
        # pas, latence, VM suspend) et finit par franchir la fenetre ~30s d'OKX,
        # faisant rejeter sporadiquement des requetes signees (positions/balance,
        # voire un ordre). On resynchronise l'horloge et on REJOUE une fois : un
        # 50102 est un rejet AVANT tout traitement cote OKX (auth echouee), donc
        # rejouer est sur meme pour une ECRITURE -- l'ordre n'a pas ete place.
        r = None
        for attempt in range(2):
            timestamp = _iso_timestamp_ms(self._time_offset_ms)
            signature = self._sign(timestamp, method, request_path, body)
            headers = {
                "OK-ACCESS-KEY": self.api_key,
                "OK-ACCESS-SIGN": signature,
                "OK-ACCESS-TIMESTAMP": timestamp,
                "OK-ACCESS-PASSPHRASE": self.passphrase,
                "Content-Type": "application/json",
            }
            r = self._session.request(method, url, headers=headers,
                                      data=body if body else None, timeout=_TIMEOUT_S)
            if attempt == 0 and _is_timestamp_expired(r):
                logger.warning(f"{method} {request_path} -> 50102 (horodatage expire, "
                               f"offset horloge derive) -- resync OKX + retry.")
                try:
                    skew_s = self.sync_time()
                    logger.info(f"horloge resynchronisee apres 50102 (derive {skew_s:+.2f}s).")
                except Exception as exc:  # noqa: BLE001 -- retente quand meme avec l'ancien offset
                    logger.error(f"resync horloge apres 50102 a echoue : {exc} -- retry best-effort.")
                continue
            break

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

    def get_balance(self, ccy: str = "USDT", field: str = "eq") -> float:
        """GET /api/v5/account/balance -- champ `field` du solde en `ccy`.
        field='eq' (defaut) = equity totale (reporting). field='availEq' =
        equite DISPONIBLE (hors marge deja gelee par une position isolee) --
        a utiliser pour DIMENSIONNER une nouvelle entree (point #6 audit
        2026-08-03), pour qu'un 2e trade ne re-suppose pas le capital plein.
        Si le champ demande est vide/absent (selon le mode de compte), retombe
        sur `eq` puis `availBal`."""
        payload = self._request("GET", "/api/v5/account/balance", params={"ccy": ccy})
        data = payload.get("data", [])
        if not data:
            return 0.0
        details = data[0].get("details", [])
        for d in details:
            if d.get("ccy") == ccy:
                for key in (field, "eq", "availBal"):
                    val = d.get(key)
                    if val not in (None, ""):
                        return float(val)
                return 0.0
        return 0.0

    def get_leverage(self, inst_id: str, mgn_mode: str = "isolated",
                     pos_side: str | None = None) -> float:
        """GET /api/v5/account/leverage-info -- levier REELLEMENT applique cote
        OKX pour (instId, mgnMode[, posSide]). Sert a verifier que set_leverage
        a bien pris avant de dimensionner (point #7 audit 2026-08-03) : marge =
        notionnel / levier_reel, donc un ecart fausserait la marge requise.
        Renvoie 0.0 si introuvable."""
        payload = self._request("GET", "/api/v5/account/leverage-info",
                                params={"instId": inst_id, "mgnMode": mgn_mode})
        data = payload.get("data", [])
        if pos_side:
            for d in data:
                if d.get("posSide") == pos_side:
                    return float(d.get("lever") or 0.0)
        if data:
            return float(data[0].get("lever") or 0.0)
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

    # -- Bougies 1-min : le PRIX DE REFERENCE du moteur de sortie -------------
    # ALIGNEMENT BACKTEST (2026-08-23) : main.py::_up_managed_exit /
    # _down_managed_exit marchent sur les CLOTURES de sous-bougies 1-min
    # (DB_1m/btc_1m.parquet, cle = open-time), jamais sur un echantillon
    # instantane. Le live lisait markPx une fois par minute -- meme granularite
    # apparente, mais decale et non ancre sur la grille des bougies. On lit
    # desormais les MEMES objets : des bougies 1-min fermees, indexees par leur
    # open-time. Le `j` du backtest devient (candle_open - traded_ts) / 60s.
    def get_candles(self, inst_id: str, bar: str = "1m", limit: int = 100,
                    after_ms: int | None = None) -> list[list]:
        """GET /api/v5/market/candles -- endpoint PUBLIC (non signe).

        Renvoie une liste TRIEE PAR ts CROISSANT (OKX repond du plus recent au
        plus ancien) de bougies au format :
            [ts_open_ms (int), open, high, low, close, volume, confirm]
        Les 6 premiers champs ont EXACTEMENT la disposition des buffers
        Bitstamp de calcule_feature.py, et `[-1]` porte le drapeau OKX
        `confirm` ("1" = bougie terminee, "0" = en formation).

        `ts` est l'OPEN-TIME de la bougie, meme convention que le parquet 1-min
        du backtest (DB_1m/btc_1m.parquet) et que Bitstamp -- une bougie
        d'open-time t couvre [t, t+bar).

        `after_ms` : pagination OKX vers le PASSE (renvoie les bougies
        strictement anterieures a ce ts). `limit` est plafonne a 300 par OKX.
        """
        params = {"instId": inst_id, "bar": bar, "limit": str(min(int(limit), 300))}
        if after_ms is not None:
            params["after"] = str(int(after_ms))
        r = self._session.get(f"{REST_HOST}/api/v5/market/candles",
                              params=params, timeout=_TIMEOUT_S)
        r.raise_for_status()
        payload = r.json()
        if not isinstance(payload, dict) or payload.get("code") != "0":
            raise OkxError(f"get_candles({inst_id}, {bar}) -> {payload}")
        out = []
        for row in payload.get("data", []):
            out.append([int(row[0]), float(row[1]), float(row[2]), float(row[3]),
                        float(row[4]), float(row[5]), row[8]])
        out.sort(key=lambda c: c[0])
        return out

    def get_candles_range(self, inst_id: str, start_ms: int, end_ms: int,
                          bar: str = "1m", closed_only: bool = True) -> list[list]:
        """Toutes les bougies `bar` dont l'open-time est dans [start_ms, end_ms),
        paginees si necessaire (OKX plafonne a 300 par requete), triees par ts
        croissant. `closed_only` ne garde que les bougies TERMINEES (confirm=1)
        -- c'est ce qu'il faut partout ou le backtest lit un `close` (la bougie
        en formation n'a pas de cloture). Mettre False uniquement pour lire
        l'OPEN d'une bougie en cours (ex. l'ancre P0 d'une entree maker : l'open
        est fige des la premiere seconde de la bougie).

        Renvoie [] plutot que de lever si la plage est vide ou invalide.
        """
        if end_ms <= start_ms:
            return []
        step_ms = _BAR_MS.get(bar)
        if step_ms is None:
            raise ValueError(f"bar '{bar}' inconnu (cf _BAR_MS)")
        acc: dict[int, list] = {}
        cursor = None
        # Borne de securite : au pire (plage tres longue) on s'arrete plutot que
        # de boucler -- l'appelant travaille alors sur ce qui a ete recupere.
        for _ in range(20):
            page = self.get_candles(inst_id, bar=bar, limit=300, after_ms=cursor)
            if not page:
                break
            for c in page:
                if start_ms <= c[0] < end_ms:
                    acc[c[0]] = c
            oldest = page[0][0]
            if oldest <= start_ms:
                break
            cursor = oldest
        rows = [acc[t] for t in sorted(acc)]
        if closed_only:
            rows = [c for c in rows if str(c[-1]) == "1"]
        return rows

    def get_order(self, inst_id: str, ord_id: str) -> dict:
        """GET /api/v5/trade/order -- champs cles : state
        ('live'/'partially_filled'/'filled'/'canceled'), fillPx, accFillSz,
        avgPx.

        DRY_RUN (cf _dry_run_stub) : un ordId 'DRYRUN-...' n'existe PAS cote
        OKX -- repond localement 'live' (jamais rempli) sans appel API, pour
        que advance_pending_entry simule proprement la fenetre d'attente puis
        bascule sur le fallback taker (lui aussi stubbe) a l'echeance.

        BUG FIXE (point #2 audit 2026-08-03) : ce raccourci est desormais GARDE
        par `self.dry_run`. Avant, il repondait 'live' pour un ordId 'DRYRUN-'
        MEME EN MODE REEL -- si un pending_entry papier survivait dans
        positions_state.json au moment du passage DRY_RUN=false, advance_pending_
        entry le voyait 'non rempli', atteignait le deadline (dans le passe) et
        postait un VRAI ordre market involontaire. En reel, un ordId 'DRYRUN-'
        tape maintenant l'API (introuvable -> OkxError), attrapee en amont
        (entry_check_failed) : aucun ordre reel ne part."""
        if self.dry_run and str(ord_id).startswith("DRYRUN-"):
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
                              td_mode: str = "isolated") -> dict:
        """Sortie market immediate (toujours taker, cf CLAUDE.md : la sortie
        est TOUJOURS taker, jamais simulee en maker). Convention HEDGE : fermer
        un LONG -> side='sell' ; fermer un SHORT -> side='buy'."""
        side = "sell" if pos_side == "long" else "buy"
        return self.place_order(inst_id, side=side, pos_side=pos_side,
                                ord_type="market", sz=sz, td_mode=td_mode)

    # ------------------------------------------------------------------ #
    #  Ordres ALGO (filet catastrophe natif) -- point #1 audit 2026-08-03
    # ------------------------------------------------------------------ #
    def place_algo_order(self, inst_id: str, pos_side: str, sz: str,
                         sl_trigger_px: str, sl_ord_px: str = "-1",
                         trigger_px_type: str = "mark",
                         td_mode: str = "isolated") -> dict:
        """POST /api/v5/trade/order-algo -- ordre CONDITIONNEL (stop) pose cote
        OKX comme filet de securite. S'execute sur le matching engine d'OKX
        MEME SI le daemon est mort. Convention HEDGE : pour FERMER un LONG ->
        side='sell', posSide='long' ; fermer un SHORT -> side='buy',
        posSide='short'. `sl_ord_px='-1'` = execution au MARCHE (fermeture
        garantie, jamais un stop-limit qui pourrait ne pas remplir).
        `trigger_px_type` : 'last'/'index'/'mark' (mark = coherent avec le pp
        reel et la logique de liquidation). Renvoie le dict data[0] (algoId)."""
        side = "sell" if pos_side == "long" else "buy"
        body = {
            "instId": inst_id, "tdMode": td_mode, "side": side, "posSide": pos_side,
            "ordType": "conditional", "sz": sz,
            "slTriggerPx": sl_trigger_px, "slOrdPx": sl_ord_px,
            "slTriggerPxType": trigger_px_type,
        }
        if self.dry_run:
            stub = self._dry_run_stub("place_algo_order", **body)
            stub["algoId"] = f"DRYRUN-ALGO-{int(time.time() * 1000)}"
            return stub
        payload = self._request("POST", "/api/v5/trade/order-algo", body_obj=body)
        data = payload.get("data", [{}])[0]
        if data.get("sCode") not in (None, "0"):
            raise OkxError(f"place_algo_order rejete : {data}")
        return data

    def cancel_algos(self, inst_id: str, algo_ids) -> dict:
        """POST /api/v5/trade/cancel-algos -- annule un/des ordres algo. Le
        corps OKX est un TABLEAU. Best-effort cote appelant : un algo deja
        declenche/annule renvoie une erreur, a ignorer (l'appelant l'attrape)."""
        if isinstance(algo_ids, str):
            algo_ids = [algo_ids]
        algo_ids = [a for a in algo_ids if a and not str(a).startswith("DRYRUN-")]
        if not algo_ids:
            return {"skipped": True}
        body = [{"instId": inst_id, "algoId": a} for a in algo_ids]
        if self.dry_run:
            return self._dry_run_stub("cancel_algos", algoIds=algo_ids)
        return self._request("POST", "/api/v5/trade/cancel-algos", body_obj=body)

    def get_algo_pending(self, inst_id: str, ord_type: str = "conditional") -> list[dict]:
        """GET /api/v5/trade/orders-algo-pending -- ordres algo EN ATTENTE (non
        declenches). Lecture (toujours reelle) -- sert a la reconciliation au
        demarrage (point #5) : savoir si une position a deja son filet."""
        payload = self._request("GET", "/api/v5/trade/orders-algo-pending",
                                params={"instType": "SWAP", "instId": inst_id, "ordType": ord_type})
        return payload.get("data", [])
