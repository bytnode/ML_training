"""calcule_feature.py -- fetch Bitstamp + calcul des features live pour les
2 modeles (modele_up, modele_down), bar-par-bar (bougie 1h qui vient de fermer).

SKEW-CRITIQUE : reutilise les helpers d'ENTRAINEMENT des packages modele_up/
modele_down (add_engineered_features.py, add_alpha_features.py, add_eth_features.py,
data.py) au lieu de reecrire les formules -- meme garantie de parite train/serve
que le reste du repo (cf CLAUDE.md).

PIEGE EVITE : modele_up/ ET modele_down/ ont chacun un module `add_engineered_
features.py` / `data.py` / `config.py` DE MEME NOM. Un simple sys.path.insert +
`import add_engineered_features` laisserait le PREMIER import gagner (cache
sys.modules) et l'AUTRE package recevrait silencieusement le mauvais code --
un bug de skew invisible. `_import_isolated` purge sys.modules avant/apres
chaque import pour garantir qu'on charge TOUJOURS le fichier du bon package.

Sources de donnees :
  - Bitstamp (btcusd, ethusd) pour TOUTES les features des 2 modeles (identique
    a l'entrainement, zero skew train/live).
  - OKX n'intervient JAMAIS ici -- uniquement dans api_okx.py / position.py
    (execution + monitoring de position, cf CLAUDE.md).
"""
from __future__ import annotations

import importlib
import logging
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger("live.calcule_feature")

# ---------------------------------------------------------------------------
# Import isole des packages modele_up / modele_down (cf note SKEW ci-dessus)
# AUTONOMIE (2026-07-10) : live/ embarque ses PROPRES copies (live/modele_up/,
# live/modele_down/) plutot que d'importer les packages siblings de 1h/ -- un
# seul dossier (`modele_team/`) a deployer, plus besoin de synchroniser 1h/
# entier sur la VPS. Compromis assume (documente dans live/CLAUDE.md) : apres
# un retrain, recopier les fichiers modifies dans live/modele_up|down/, sinon
# le daemon tourne sur un modele/code perime sans avertissement.
# ---------------------------------------------------------------------------
_LIVE_DIR = Path(__file__).resolve().parent
_MODELE_UP_DIR = _LIVE_DIR / "modele_up"
_MODELE_DOWN_DIR = _LIVE_DIR / "modele_down"

_ISOLATED_NAMES = ("config", "data", "add_engineered_features",
                   "add_alpha_features", "add_eth_features")


def _import_isolated(pkg_dir: Path, module_name: str):
    """Importe `module_name` depuis pkg_dir en isolation totale : purge
    sys.modules AVANT (evite un hit de cache venant de l'autre package) et
    APRES (evite de polluer le prochain import isole)."""
    for name in _ISOLATED_NAMES:
        sys.modules.pop(name, None)
    pkg_dir_str = str(pkg_dir)
    sys.path.insert(0, pkg_dir_str)
    try:
        mod = importlib.import_module(module_name)
    finally:
        sys.path.remove(pkg_dir_str)
        for name in _ISOLATED_NAMES:
            sys.modules.pop(name, None)
    return mod


# ---------------------------------------------------------------------------
# Bitstamp OHLC -- fetcher (port de l'ancien live_team/live_helpers/fetch.py)
# ---------------------------------------------------------------------------
BITSTAMP_OHLC = "https://www.bitstamp.net/api/v2/ohlc/{pair}/"
MAX_LIMIT = 1000
REQUEST_TIMEOUT_S = 15
REQUEST_RETRIES = 3
REQUEST_BACKOFF_S = 0.5
PAGE_SLEEP_S = 0.15
_SESSION = requests.Session()


def _request_chunk(pair: str, step_sec: int, limit: int, end_sec: int | None = None) -> list[list]:
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
            return [[int(c["timestamp"]) * 1000, float(c["open"]), float(c["high"]),
                    float(c["low"]), float(c["close"]), float(c["volume"])] for c in data]
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
            last_exc = exc
            if attempt < REQUEST_RETRIES:
                time.sleep(REQUEST_BACKOFF_S * (2 ** attempt))
                continue
            raise
    if last_exc is not None:
        raise last_exc
    return []


def fetch_klines(pair: str, step_s: int, n_bars: int, until_sec: int | None = None) -> list[list]:
    """Recupere les `n_bars` barres les plus recentes pour `pair` / `step_s` --
    ou, si `until_sec` est fourni, les `n_bars` barres se terminant A/AVANT
    `until_sec` (ancre la requete sur un point PASSE precis plutot que sur
    "maintenant"). Pagine en arriere ; s'arrete quand la page la plus ancienne
    ne recule plus (Bitstamp omet parfois la bougie en formation et renvoie
    limit-1).

    BUG FIXE (2026-07-17) : `until_sec=None` (endpoint "les plus recentes")
    a un angle mort observe en prod -- une bougie qui VIENT de clore peut ne
    pas encore apparaitre de facon fiable dans cette reponse, meme si une
    bougie PLUS RECENTE (en formation) y figure deja (cf `backfill_gap_if_needed`/
    `heal_gaps`, qui utilisent `until_sec` pour eviter ce meme angle mort lors
    d'un rattrapage de trou)."""
    out: dict[int, list] = {}
    end_sec: int | None = until_sec
    prev_oldest: int | None = None
    safety_iter = 0
    while len(out) < n_bars and safety_iter < 40:
        safety_iter += 1
        limit = min(n_bars - len(out), MAX_LIMIT)
        chunk = _request_chunk(pair, step_s, limit, end_sec=end_sec)
        if not chunk:
            break
        for bar in chunk:
            out[bar[0]] = bar
        oldest_open = min(bar[0] for bar in chunk)
        if prev_oldest is not None and oldest_open >= prev_oldest:
            break
        prev_oldest = oldest_open
        end_sec = oldest_open // 1000 - 1
        time.sleep(PAGE_SLEEP_S)
    klines = sorted(out.values(), key=lambda b: b[0])
    return klines[-n_bars:]


def merge_klines(buffer: list[list], fresh: list[list]) -> list[list]:
    by_ts: dict[int, list] = {int(b[0]): b for b in buffer}
    for bar in fresh:
        by_ts[int(bar[0])] = bar
    return sorted(by_ts.values(), key=lambda b: b[0])


def fetch_many_parallel(specs: list[tuple[str, int, int]]) -> dict[tuple[str, int], list[list]]:
    if not specs:
        return {}
    with ThreadPoolExecutor(max_workers=len(specs)) as ex:
        futures: dict[tuple[str, int], Future] = {
            (pair, step_s): ex.submit(fetch_klines, pair, step_s, n_bars)
            for (pair, step_s, n_bars) in specs
        }
        return {key: fut.result() for key, fut in futures.items()}


def freshness_retry(pair: str, step_s: int, target_open_ms: int, buf: list[list],
                    max_tries: int = 12) -> list[list]:
    """Re-fetch jusqu'a ce que target_open_ms soit present dans buf (Bitstamp
    publie parfois la bougie avec quelques secondes de retard)."""
    for _ in range(1, max_tries + 1):
        fresh = fetch_klines(pair, step_s, n_bars=5)
        buf = merge_klines(buf, fresh)
        latest = buf[-1][0] if buf else 0
        if latest >= target_open_ms:
            return buf
        time.sleep(5)
    raise TimeoutError(
        f"freshness_retry: {pair} step={step_s}s target={target_open_ms} "
        f"non visible apres {max_tries} essais"
    )


def backfill_gap_if_needed(pair: str, step_s: int, buf: list[list], target_open_ms: int,
                           deeper_bars: int = 60, max_tries: int = 6,
                           retry_sleep_s: float = 5.0) -> list[list]:
    """BUG FIXE (2026-07-15) : `freshness_retry` garantit seulement que la
    DERNIERE bougie du buffer est >= target_open_ms -- PAS que target_open_ms
    lui-meme y figure comme cle exacte. Bitstamp omet parfois une bougie
    precise (cf note dans `fetch_klines` : "Bitstamp omet parfois la bougie en
    formation"). Comme le rafraichissement normal (`LiveMarketData.refresh`) ne
    re-fetch que les REFRESH_TAIL_BARS bougies les plus recentes, un tel trou
    ne se serait JAMAIS comble tout seul -- `_target_row_idx` levait alors
    ValueError a CHAQUE cycle suivant, indefiniment (observe en prod : meme
    target_open_ms echouant en boucle a chaque heure).

    Si `target_open_ms` est dans le RANGE du buffer (entre son premier et son
    dernier timestamp) mais absent en tant que cle exacte, refetch une fenetre
    plus large (`deeper_bars`) et merge (idempotent, upsert par timestamp dans
    `merge_klines`) -- comble le trou sans jamais dupliquer ni re-fetch
    l'integralite de l'historique.

    BUG FIXE (2026-07-17) : le premier essai de ce fix (refetch "les plus
    recentes", until_sec=None) est reste inefficace en prod -- meme trou
    reproduit 2x en 48h, TOUJOURS exactement 1h derriere la fin du buffer.
    Hypothese confirmee : une requete "most recent" retombe sur le MEME angle
    mort Bitstamp qui a cause le trou au depart (une bougie qui vient de
    clore n'est pas encore fiable dans cette reponse, meme si une bougie PLUS
    RECENTE -- en formation -- y figure deja). Fix : ancrer `until_sec` bien
    APRES la cloture du target (force Bitstamp a la traiter comme historique
    ferme, pas comme la limite "temps reel"), et reessayer (jusqu'a
    max_tries, cf freshness_retry) au cas ou le retard d'indexation Bitstamp
    depasse une seule tentative."""
    if not buf:
        return buf
    if target_open_ms in {int(b[0]) for b in buf}:
        return buf
    lo, hi = int(buf[0][0]), int(buf[-1][0])
    if not (lo <= target_open_ms <= hi):
        return buf   # hors range -- freshness_retry gere deja ce cas (pas encore publie)
    until_sec = target_open_ms // 1000 + step_s + 300   # marge 5min apres la cloture du target
    for attempt in range(1, max_tries + 1):
        logger.warning(f"backfill_gap_if_needed: trou detecte pour {pair} step={step_s}s "
                       f"target={target_open_ms} (range {lo}..{hi}) -- refetch ancre "
                       f"until={until_sec} (essai {attempt}/{max_tries}).")
        deeper = fetch_klines(pair, step_s, deeper_bars, until_sec=until_sec)
        buf = merge_klines(buf, deeper)
        if target_open_ms in {int(b[0]) for b in deeper}:
            return buf
        time.sleep(retry_sleep_s)
    logger.error(f"backfill_gap_if_needed: trou PERSISTANT pour {pair} step={step_s}s "
                f"target={target_open_ms} apres {max_tries} tentatives -- Bitstamp ne "
                f"publie toujours pas cette bougie.")
    return buf


def heal_gaps(pair: str, step_s: int, buf: list[list], deeper_bars: int = 60,
              max_tries: int = 6, retry_sleep_s: float = 5.0) -> list[list]:
    """BUG FIXE (2026-07-15, audit complet) : detecte un trou QUELCONQUE dans
    buf (deux bougies consecutives dont l'ecart de timestamp != step_s*1000) et
    le comble par un refetch cible. Contrairement a `backfill_gap_if_needed`
    (qui ne verifie qu'un timestamp PRECIS demande par le caller -- suffisant
    pour un buffer consulte par timestamp exact, comme buf_btc_1h/buf_eth_1h
    via `_target_row_idx`), celle-ci scanne TOUT le buffer -- necessaire pour un
    buffer consomme par POSITION dans le tableau plutot que par timestamp exact
    (ex. buf_btc_4h dans `market_data.dmi_blocks()`, qui passe les colonnes du
    buffer directement a `filters._wilder_dmi`) : un trou N'IMPORTE OU y fausse
    silencieusement le DMI calcule -- pas de crash, pas d'alerte, juste deux
    bougies 8h+ d'ecart traitees comme un seul pas de 4h, ce qui peut faire
    basculer une decision d'entree reelle. Meme cause racine que le trou deja
    corrige sur les buffers 1h/eth (Bitstamp omet parfois une bougie).

    BUG FIXE (2026-07-17) : meme correction que `backfill_gap_if_needed` --
    ancre `until_sec` juste apres le bar qui suit le trou (pas une requete
    "most recent") et reessaie plusieurs fois, cf sa docstring pour le
    diagnostic complet."""
    if len(buf) < 2:
        return buf
    step_ms = step_s * 1000
    for i in range(1, len(buf)):
        if int(buf[i][0]) - int(buf[i - 1][0]) != step_ms:
            gap_ts = int(buf[i][0])
            until_sec = gap_ts // 1000 + 300   # marge 5min apres le bar qui suit le trou
            for attempt in range(1, max_tries + 1):
                logger.warning(f"heal_gaps: trou detecte pour {pair} step={step_s}s entre "
                               f"{buf[i - 1][0]} et {buf[i][0]} -- refetch ancre "
                               f"until={until_sec} (essai {attempt}/{max_tries}).")
                deeper = fetch_klines(pair, step_s, deeper_bars, until_sec=until_sec)
                buf = merge_klines(buf, deeper)
                still_gapped = any(
                    int(buf[j][0]) - int(buf[j - 1][0]) != step_ms for j in range(1, len(buf))
                )
                if not still_gapped:
                    return buf
                time.sleep(retry_sleep_s)
            logger.error(f"heal_gaps: trou PERSISTANT pour {pair} step={step_s}s "
                        f"apres {max_tries} tentatives.")
            return buf
    return buf


# ---------------------------------------------------------------------------
# Reconstruction OHLC -> DataFrame avec returns normalises (identique data.py phase 2)
# ---------------------------------------------------------------------------
def _buf_to_df(buf: Sequence) -> pd.DataFrame:
    if not buf or len(buf) < 2:
        raise ValueError("buffer 1h trop court (au moins 2 barres requises).")
    df = pd.DataFrame(buf, columns=["timestamp_ms", "open", "high", "low", "close", "volume"])
    df["timestamp_ms"] = df["timestamp_ms"].astype(np.int64)
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = df[c].astype(float)
    df = df.sort_values("timestamp_ms").reset_index(drop=True)
    close_prev = df["close"].shift(1)
    df["open_ret"] = (df["open"] - close_prev) / close_prev
    df["high_ret"] = (df["high"] - close_prev) / close_prev
    df["low_ret"] = (df["low"] - close_prev) / close_prev
    df["close_ret"] = (df["close"] - close_prev) / close_prev
    return df


def _target_row_idx(df: pd.DataFrame, target_open_ms: int) -> int:
    pos = np.where(df["timestamp_ms"].to_numpy() == int(target_open_ms))[0]
    if len(pos) == 0:
        raise ValueError(
            f"target_open_ms {target_open_ms} introuvable dans le buffer "
            f"(range {int(df['timestamp_ms'].iloc[0])}..{int(df['timestamp_ms'].iloc[-1])})."
        )
    return int(pos[-1])


# ---------------------------------------------------------------------------
# Buckets (volatility_state, is_weekend) -- PAS des features modele (absentes de
# booster.feature_names), utilises UNIQUEMENT pour le lookup du seuil sidecar
# (`_bucket_threshold` dans modele_up/api.py et modele_down/api.py). Recalcules
# ICI avec la MEME formule que data.py (compute_normalized_features::atr_pct +
# rolling ATR_LONG_PERIOD + seuils VOLATILITY_HIGH/LOW du config.py du package),
# par package (isolation sys.modules identique au reste du fichier).
# ---------------------------------------------------------------------------
def compute_buckets(buf_1h: Sequence, target_open_ms: int, pkg_dir: Path) -> tuple[int, int]:
    """Renvoie (volatility_state, is_weekend) pour la bougie 1h target_open_ms,
    calcule avec les helpers du package `pkg_dir` (modele_up ou modele_down)."""
    raw = pd.DataFrame(buf_1h, columns=["timestamp_ms", "open", "high", "low", "close", "volume"])
    raw["timestamp_ms"] = raw["timestamp_ms"].astype(np.int64)
    for c in ("open", "high", "low", "close", "volume"):
        raw[c] = raw[c].astype(float)
    raw = raw.sort_values("timestamp_ms").reset_index(drop=True)

    data_mod = _import_isolated(pkg_dir, "data")
    cfg_mod = _import_isolated(pkg_dir, "config")
    norm = data_mod.compute_normalized_features(raw)   # -> colonne atr_pct incluse

    idx = _target_row_idx(norm, target_open_ms)
    atr_long = norm["atr_pct"].rolling(window=cfg_mod.ATR_LONG_PERIOD, min_periods=20).mean()
    atr_ratio = float(norm["atr_pct"].iloc[idx] / atr_long.iloc[idx]) if pd.notna(atr_long.iloc[idx]) else float("nan")
    if pd.isna(atr_ratio):
        volatility_state = 1   # mid par defaut (warmup), meme convention que data.py
    elif atr_ratio > cfg_mod.VOLATILITY_HIGH:
        volatility_state = 2
    elif atr_ratio < cfg_mod.VOLATILITY_LOW:
        volatility_state = 0
    else:
        volatility_state = 1

    ts_dt = pd.Timestamp(int(target_open_ms), unit="ms", tz="UTC")
    is_weekend = int(ts_dt.dayofweek >= 5)
    return volatility_state, is_weekend


# ---------------------------------------------------------------------------
# modele_up : stoch_k_14, rsi_6, hour_cos, ema_compression_20, bars_since_low_50, ret_skew_24
# ---------------------------------------------------------------------------
FEATURES_UP = ["stoch_k_14", "rsi_6", "hour_cos", "ema_compression_20",
              "bars_since_low_50", "ret_skew_24"]


def _up_alpha_three(df: pd.DataFrame, alpha_mod) -> dict:
    """Recopie verbatim des lignes ema_compression_20/bars_since_low_50/ret_skew_24
    de add_alpha_features.compute_alpha (on n'appelle pas compute_alpha tel quel :
    il lit aussi le parquet ETH pour btc_eth_corr_48, non utilise par modele_up)."""
    c = df["close"].astype(float)
    l = df["low"]
    cr = df["close_ret"].astype(float)
    out = {"ret_skew_24": cr.rolling(24, min_periods=24).skew().to_numpy()}
    win = 50
    bsl = l.rolling(win, min_periods=win).apply(lambda x: (win - 1 - np.argmin(x)), raw=True)
    out["bars_since_low_50"] = (bsl / win).to_numpy()
    e10, e20, e50 = alpha_mod._ema(c, 10), alpha_mod._ema(c, 20), alpha_mod._ema(c, 50)
    emax = pd.concat([e10, e20, e50], axis=1).max(axis=1)
    emin = pd.concat([e10, e20, e50], axis=1).min(axis=1)
    out["ema_compression_20"] = ((emax - emin) / c).to_numpy()
    return out


def build_features_up(buf_1h: Sequence, target_open_ms: int) -> dict:
    """6 features UP pour la bougie 1h dont l'open == target_open_ms."""
    df = _buf_to_df(buf_1h)
    idx = _target_row_idx(df, target_open_ms)

    eng = _import_isolated(_MODELE_UP_DIR, "add_engineered_features")
    alpha = _import_isolated(_MODELE_UP_DIR, "add_alpha_features")
    data_mod = _import_isolated(_MODELE_UP_DIR, "data")

    base = eng.compute_base_30m(df)
    df_hc = df.copy()
    data_mod._add_tier1_derived(df_hc)
    hour_cos = df_hc["hour_cos"].to_numpy()
    alpha3 = _up_alpha_three(df, alpha)

    out = {
        "stoch_k_14": float(base["stoch_k_14"][idx]),
        "rsi_6": float(base["rsi_6"][idx]),
        "hour_cos": float(hour_cos[idx]),
        "ema_compression_20": float(alpha3["ema_compression_20"][idx]),
        "bars_since_low_50": float(alpha3["bars_since_low_50"][idx]),
        "ret_skew_24": float(alpha3["ret_skew_24"][idx]),
    }
    return {k: out[k] for k in FEATURES_UP}


# ---------------------------------------------------------------------------
# modele_down : stoch_k_14, hour_cos, ef_ret4, rsi_6, autocorr_12, ef_upwick
# ---------------------------------------------------------------------------
FEATURES_DOWN = ["stoch_k_14", "hour_cos", "ef_ret4", "rsi_6", "autocorr_12", "ef_upwick"]


def build_features_down(buf_1h: Sequence, buf_eth_1h: Sequence | None,
                        target_open_ms: int) -> dict:
    """6 features DOWN pour la bougie 1h dont l'open == target_open_ms.

    ef_ret4 / ef_upwick : ETH est deja NATIF 1h cote entrainement (DB/eth_1h.parquet,
    add_eth_features.py -- PAS de resample 15m->1h, contrairement a l'ancienne
    version 30m). buf_eth_1h doit donc etre des barres ETH 1h Bitstamp brutes,
    jointes par TIMESTAMP EXACT (meme open_time que la bougie BTC cible).
    """
    df = _buf_to_df(buf_1h)
    idx = _target_row_idx(df, target_open_ms)

    eng = _import_isolated(_MODELE_DOWN_DIR, "add_engineered_features")
    data_mod = _import_isolated(_MODELE_DOWN_DIR, "data")

    base = eng.compute_base_30m(df)
    data_mod._add_tier1_derived(df)   # mutation in-place, ajoute hour_cos
    hour_cos = df["hour_cos"].to_numpy()

    out = {
        "stoch_k_14": float(base["stoch_k_14"][idx]),
        "rsi_6": float(base["rsi_6"][idx]),
        "autocorr_12": float(base["autocorr_12"][idx]),
        "hour_cos": float(hour_cos[idx]),
    }

    ef_ret4 = ef_upwick = float("nan")
    if buf_eth_1h:
        eth_mod = _import_isolated(_MODELE_DOWN_DIR, "add_eth_features")
        eth_df = pd.DataFrame(buf_eth_1h, columns=["timestamp_ms", "open", "high", "low", "close", "volume"])
        eth_df["timestamp_ms"] = eth_df["timestamp_ms"].astype(np.int64)
        for c in ("open", "high", "low", "close", "volume"):
            eth_df[c] = eth_df[c].astype(float)
        eth_feat = eth_mod.compute_eth_features(eth_df.sort_values("timestamp_ms").reset_index(drop=True))
        hit = eth_feat.loc[eth_feat["timestamp_ms"] == int(target_open_ms)]
        if len(hit) == 1:
            row = hit.iloc[0]
            ef_ret4 = float(row["ef_ret4"])
            ef_upwick = float(row["ef_upwick"])
    out["ef_ret4"] = ef_ret4
    out["ef_upwick"] = ef_upwick

    missing = [c for c in FEATURES_DOWN if c not in out]
    if missing:
        raise KeyError(f"build_features_down: features manquantes : {missing}")
    return {k: out[k] for k in FEATURES_DOWN}


# ---------------------------------------------------------------------------
# Smoke test (nécessite un acces reseau Bitstamp)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Fetch 300 barres 1h BTC + ETH...")
    specs = [("btcusd", 3600, 300), ("ethusd", 3600, 300)]
    results = fetch_many_parallel(specs)
    buf_btc = results[("btcusd", 3600)]
    buf_eth = results[("ethusd", 3600)]
    target = buf_btc[-2][0]   # avant-derniere bougie (la derniere peut etre en formation)
    print(f"target_open_ms={target}")
    feats_up = build_features_up(buf_btc, target)
    print("UP  :", feats_up)
    feats_down = build_features_down(buf_btc, buf_eth, target)
    print("DOWN:", feats_down)
