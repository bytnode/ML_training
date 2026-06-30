"""Bitstamp public OHLC — fetcher pour le daemon live 1h.

Format interne des barres : [open_time_ms, open, high, low, close, volume]
Les secondes Bitstamp sont converties en millisecondes ici et nulle part ailleurs.

TFs utilisés par le daemon 1h :
  btcusd : 5m (300s), 30m (1800s), 1h (3600s), 4h (14400s), 1d (86400s)
  ethusd : 15m (900s)

Usage typique :
    buf_1h = fetch_klines("btcusd", 3600, n_bars=3200)
    buf_eth = fetch_klines("ethusd", 900, n_bars=3200)
    results = fetch_many_parallel([
        ("btcusd", 300, 500),
        ("btcusd", 1800, 500),
        ("btcusd", 3600, 3200),
        ("btcusd", 14400, 500),
        ("btcusd", 86400, 500),
        ("ethusd", 900, 3200),
    ])
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Iterable

import requests

# --------------------------------------------------------------------------- #
# Constantes                                                                   #
# --------------------------------------------------------------------------- #

BITSTAMP_OHLC = "https://www.bitstamp.net/api/v2/ohlc/{pair}/"

# Steps (secondes) légitimes pour chaque paire.
# Le daemon 1h n'utilise pas le 15m BTC ni le 1h ETH — on accepte quand même
# des valeurs arbitraires pour rester générique (validate_step est optionnel).
BTC_STEPS = {300, 1800, 3600, 14400, 86400}   # 5m 30m 1h 4h 1d
ETH_STEPS = {900}                              # 15m uniquement

MAX_LIMIT = 1000          # cap hard Bitstamp (maximum barres par requête)
REQUEST_TIMEOUT_S = 15    # timeout par requête
REQUEST_RETRIES = 3       # nombre de ré-essais sur erreur réseau
REQUEST_BACKOFF_S = 0.5   # backoff exponentiel : 0.5 s, 1.0 s, 2.0 s …
PAGE_SLEEP_S = 0.15       # pause inter-pages (politesse)

# Session keep-alive : une seule connexion TCP réutilisée par thread
# (urllib3 pool_connections=10 → assez pour fetch_many_parallel à 6 workers).
_SESSION = requests.Session()


# --------------------------------------------------------------------------- #
# Requête élémentaire avec retry                                               #
# --------------------------------------------------------------------------- #

def _request_chunk(
    pair: str,
    step_sec: int,
    limit: int,
    end_sec: int | None = None,
) -> list[list]:
    """Un appel Bitstamp OHLC avec retry exponentiel.

    Paramètres
    ----------
    pair     : ex. "btcusd", "ethusd"
    step_sec : durée d'une bougie en secondes
    limit    : nombre de barres demandées (≤ MAX_LIMIT)
    end_sec  : si fourni, Bitstamp renvoie les barres se terminant avant end_sec

    Retourne une liste de barres [open_ms, o, h, l, c, v] triées ASC.
    Lève la dernière exception si tous les essais échouent.
    """
    params: dict = {"step": step_sec, "limit": limit}
    if end_sec is not None:
        params["end"] = end_sec
    url = BITSTAMP_OHLC.format(pair=pair)

    last_exc: Exception | None = None
    for attempt in range(REQUEST_RETRIES + 1):
        try:
            r = _SESSION.get(url, params=params, timeout=REQUEST_TIMEOUT_S)
            r.raise_for_status()
            data = r.json().get("data", {}).get("ohlc", [])
            return [
                [
                    int(c["timestamp"]) * 1000,   # secondes → millisecondes
                    float(c["open"]),
                    float(c["high"]),
                    float(c["low"]),
                    float(c["close"]),
                    float(c["volume"]),
                ]
                for c in data
            ]
        except (requests.exceptions.Timeout,
                requests.exceptions.ConnectionError) as exc:
            last_exc = exc
            if attempt < REQUEST_RETRIES:
                time.sleep(REQUEST_BACKOFF_S * (2 ** attempt))
                continue
            raise

    if last_exc is not None:
        raise last_exc
    return []


# --------------------------------------------------------------------------- #
# Pagination                                                                   #
# --------------------------------------------------------------------------- #

def fetch_klines(pair: str, step_s: int, n_bars: int) -> list[list]:
    """Récupère les `n_bars` barres les plus récentes pour `pair` / `step_s`.

    Pagine en arrière via le paramètre `end` (« oldest recule ») jusqu'à avoir
    n_bars ou atteindre la limite historique Bitstamp.

    Terminaison : on s'arrête si la page la plus ancienne ne recule plus
    (oldest_open >= prev_oldest), et non sur `len(chunk) < limit` — Bitstamp
    omet souvent la bougie en cours de formation et renvoie limit-1, ce qui
    ferait avorter la pagination trop tôt.

    Retourne une liste triée ASC, dédupliquée, de longueur min(n_bars, dispo).
    """
    out: dict[int, list] = {}   # open_ms → barre (dédup automatique)
    end_sec: int | None = None
    prev_oldest: int | None = None
    safety_iter = 0

    while len(out) < n_bars and safety_iter < 40:
        safety_iter += 1
        limit = min(n_bars - len(out), MAX_LIMIT)
        chunk = _request_chunk(pair, step_s, limit, end_sec=end_sec)
        if not chunk:
            break  # plus d'historique disponible

        for bar in chunk:
            out[bar[0]] = bar

        oldest_open = min(bar[0] for bar in chunk)

        # Fin d'historique réelle : la page ne recule plus
        if prev_oldest is not None and oldest_open >= prev_oldest:
            break

        prev_oldest = oldest_open
        end_sec = oldest_open // 1000 - 1   # reculer d'un pas
        time.sleep(PAGE_SLEEP_S)

    klines = sorted(out.values(), key=lambda b: b[0])
    return klines[-n_bars:]


# --------------------------------------------------------------------------- #
# Merge (mise à jour incrémentale du buffer)                                   #
# --------------------------------------------------------------------------- #

def merge_klines(buffer: list[list], fresh: list[list]) -> list[list]:
    """Fusionne `fresh` dans `buffer` par open_time_ms (replace-or-append).

    Utilisé pour les rafraîchissements incrémentiels : on fetch quelques barres
    récentes et on met à jour le buffer long existant sans tout re-télécharger.
    """
    by_ts: dict[int, list] = {int(b[0]): b for b in buffer}
    for bar in fresh:
        by_ts[int(bar[0])] = bar
    return sorted(by_ts.values(), key=lambda b: b[0])


# --------------------------------------------------------------------------- #
# Fetch parallèle multi-TF                                                     #
# --------------------------------------------------------------------------- #

def fetch_many_parallel(
    specs: list[tuple[str, int, int]],
) -> dict[tuple[str, int], list[list]]:
    """Fetch parallèle de plusieurs (pair, step_s, n_bars).

    Le temps total est celui du fetch LE PLUS LENT (non la somme).
    Remplace ~6 appels série (~3–4 s) par un batch parallèle (~0.5–1 s).

    Retourne un dict {(pair, step_s): klines}.
    Si un thread lève une exception, elle se propage (sémanctique série sur erreur).
    """
    if not specs:
        return {}

    with ThreadPoolExecutor(max_workers=len(specs)) as ex:
        futures: dict[tuple[str, int], Future] = {
            (pair, step_s): ex.submit(fetch_klines, pair, step_s, n_bars)
            for (pair, step_s, n_bars) in specs
        }
        return {key: fut.result() for key, fut in futures.items()}


# --------------------------------------------------------------------------- #
# Freshness retry                                                              #
# --------------------------------------------------------------------------- #

def freshness_retry(
    pair: str,
    step_s: int,
    target_open_ms: int,
    buf: list[list],
    max_tries: int = 12,
) -> list[list]:
    """Re-fetch la queue jusqu'à ce que `target_open_ms` soit présent dans buf.

    Bitstamp peut publier la bougie avec quelques secondes de retard.
    On attend en boucle (max `max_tries` × ~5 s) que la bougie cible apparaisse,
    puis on retourne le buffer mis à jour.

    Paramètres
    ----------
    pair            : ex. "btcusd"
    step_s          : pas en secondes
    target_open_ms  : open_time_ms de la bougie qu'on veut voir confirmée
    buf             : buffer existant (modifié en place via merge)
    max_tries       : nombre de tentatives (défaut 12 ≈ 60 s de patience)

    Retourne le buffer mis à jour (merge des barres récentes).
    Lève TimeoutError si la bougie n'est pas visible après max_tries essais.
    """
    for attempt in range(1, max_tries + 1):
        # Fetch les 5 barres les plus récentes pour ne pas re-télécharger tout
        fresh = fetch_klines(pair, step_s, n_bars=5)
        buf = merge_klines(buf, fresh)

        # Vérifie la présence de la bougie cible
        latest = buf[-1][0] if buf else 0
        if latest >= target_open_ms:
            return buf

        # Pause progressive (5 s par tentative)
        wait_s = 5
        time.sleep(wait_s)

    raise TimeoutError(
        f"freshness_retry: {pair} step={step_s}s "
        f"target={target_open_ms} non visible après {max_tries} essais"
    )


# --------------------------------------------------------------------------- #
# Helpers utilitaires                                                          #
# --------------------------------------------------------------------------- #

def latest_open_ms(klines: list[list]) -> int:
    """Open_ms de la dernière barre du buffer (0 si vide)."""
    return int(klines[-1][0]) if klines else 0


# --------------------------------------------------------------------------- #
# Smoke test                                                                   #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import datetime as dt

    print("Smoke test — 5 barres par TF (btcusd + ethusd 15m)…")

    # Steps daemon 1h
    btc_steps = [300, 1800, 3600, 14400, 86400]
    labels    = ["5m", "30m", "1h", "4h", "1d"]

    for step, lbl in zip(btc_steps, labels):
        kl = fetch_klines("btcusd", step, n_bars=5)
        ts = dt.datetime.fromtimestamp(kl[-1][0] / 1000, tz=dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        print(f"  btcusd {lbl:3s} ({step:6d}s) : {len(kl)} barres, dernière {ts}  C={kl[-1][4]}")

    kl_eth = fetch_klines("ethusd", 900, n_bars=5)
    ts_eth = dt.datetime.fromtimestamp(kl_eth[-1][0] / 1000, tz=dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(f"  ethusd 15m  (   900s) : {len(kl_eth)} barres, dernière {ts_eth}  C={kl_eth[-1][4]}")

    print("\nParallel fetch (6 specs)…")
    specs = [
        ("btcusd", 300, 10),
        ("btcusd", 1800, 10),
        ("btcusd", 3600, 10),
        ("btcusd", 14400, 10),
        ("btcusd", 86400, 10),
        ("ethusd", 900, 10),
    ]
    results = fetch_many_parallel(specs)
    for (pair, step_s), kl in results.items():
        print(f"  ({pair}, {step_s:6d}s) → {len(kl)} barres")

    print("OK")
