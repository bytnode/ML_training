"""market_data.py -- buffers Bitstamp PERSISTANTS pour live.py (campagne autonomie
2026-07-10) : fetch complet UNE FOIS au lancement, puis mis a jour de facon
INCREMENTALE a chaque cycle horaire (fetch des quelques bougies les plus recentes
+ merge_klines), jamais un re-fetch complet -- et plus AUCUN acces a la SQLite de
modele_down (ni pour le DMI par-trade, ni pour le regime) une fois demarre.

Buffers maintenus, tous en memoire (jamais persistes sur disque -- un redemarrage
refait un fetch initial complet, rapide) :
  - buf_btc_1h  : features des 2 modeles (calcule_feature.py) + axe tendance/
                  momentum du regime (regime_live.py). Profondeur >= LOOKBACK
                  regime (720h=30j) + marge warmup features -> 800 bars.
  - buf_btc_4h  : filtre DMI PAR-TRADE (config_main.DMI_FILTER_ENABLED, via
                  filters._wilder_dmi) ET axe DMI du regime (regime_live, instance
                  SEPAREE mais MEME buffer source -- juste 2 appels Wilder distincts
                  sur les memes bougies). Profondeur >= ~10x DMI_LENGTH pour
                  convergence -> 300 bars (largement suffisant, cf calcul de
                  convergence EMA de Wilder dans le design).
  - buf_eth_1h  : feature ef_ret4/ef_upwick de modele_down (calcule_feature.py).

Le regime (regime_live.py) a sa PROPRE persistance (logs/regime_state.json,
independante des buffers OHLC ci-dessus qui restent en RAM) -- cf regime_live.py.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

_LIVE_DIR = Path(__file__).resolve().parent
if str(_LIVE_DIR) not in sys.path:
    sys.path.insert(0, str(_LIVE_DIR))

import calcule_feature  # noqa: E402
import filters  # noqa: E402
import regime_live  # noqa: E402
from config_main import DMI_FILTER_ENABLED, DMI_LENGTH  # noqa: E402
from regime import config_regime as cfg_regime  # noqa: E402

logger = logging.getLogger("live.market_data")

INTERVAL_1H_MS = 60 * 60 * 1000
INTERVAL_4H_MS = 4 * 60 * 60 * 1000

# Profondeur des buffers -- cf docstring ci-dessus pour la justification de chaque valeur.
BTC_1H_BARS = 800
BTC_4H_BARS = 300
ETH_1H_BARS = 400
REFRESH_TAIL_BARS = 5   # nb de bougies recentes re-fetchees + mergees a chaque cycle (incremental)


class LiveMarketData:
    def __init__(self, regime_state_path: Path):
        self.regime_state_path = regime_state_path
        self.buf_btc_1h: list = []
        self.buf_btc_4h: list = []
        self.buf_eth_1h: list = []
        self.regime_state: dict | None = None

    # -- Fetch initial complet (une fois, au lancement) -----------------------
    def bootstrap(self) -> None:
        logger.info("Fetch initial des buffers marche (BTC 1h/4h, ETH 1h)...")
        specs = [("btcusd", 3600, BTC_1H_BARS), ("btcusd", 14400, BTC_4H_BARS),
                ("ethusd", 3600, ETH_1H_BARS)]
        results = calcule_feature.fetch_many_parallel(specs)
        self.buf_btc_1h = results[("btcusd", 3600)]
        self.buf_btc_4h = results[("btcusd", 14400)]
        self.buf_eth_1h = results[("ethusd", 3600)]
        logger.info(f"Buffers initiaux : BTC 1h={len(self.buf_btc_1h)} "
                    f"BTC 4h={len(self.buf_btc_4h)} ETH 1h={len(self.buf_eth_1h)}.")

        if cfg_regime.REGIME_WRAPPER_ENABLED:
            self.regime_state = regime_live.load_or_init_state(self.regime_state_path)
        else:
            self.regime_state = None

    # -- Rafraichissement incremental (chaque cycle horaire) -------------------
    def refresh(self, target_ts: int) -> None:
        """Merge les REFRESH_TAIL_BARS bougies les plus recentes dans chaque
        buffer (jamais un re-fetch complet), confirme que `target_ts` est bien
        present (freshness_retry -- attend/re-fetch si Bitstamp est en retard,
        leve TimeoutError sinon, jamais un cycle silencieux sur une bougie
        perimee), puis avance le regime d'un bar (ou comble un trou si le
        daemon a manque des cycles -- meme fonction, cf
        regime_live.bootstrap_and_advance)."""
        fresh = calcule_feature.fetch_many_parallel([
            ("btcusd", 3600, REFRESH_TAIL_BARS), ("btcusd", 14400, REFRESH_TAIL_BARS),
            ("ethusd", 3600, REFRESH_TAIL_BARS),
        ])
        self.buf_btc_1h = calcule_feature.merge_klines(self.buf_btc_1h, fresh[("btcusd", 3600)])
        self.buf_btc_4h = calcule_feature.merge_klines(self.buf_btc_4h, fresh[("btcusd", 14400)])
        self.buf_eth_1h = calcule_feature.merge_klines(self.buf_eth_1h, fresh[("ethusd", 3600)])

        self.buf_btc_1h = calcule_feature.freshness_retry("btcusd", 3600, target_ts, self.buf_btc_1h)
        self.buf_eth_1h = calcule_feature.freshness_retry("ethusd", 3600, target_ts, self.buf_eth_1h)

        # BUG FIXE (2026-07-15) : freshness_retry ci-dessus ne garantit que le
        # DERNIER bar >= target_ts, pas que target_ts lui-meme soit present
        # (trou possible si Bitstamp a omis une bougie precise, cf docstring
        # calcule_feature.backfill_gap_if_needed) -- comble un tel trou avant
        # que run_hourly_cycle n'aille chercher exactement target_ts.
        self.buf_btc_1h = calcule_feature.backfill_gap_if_needed("btcusd", 3600, self.buf_btc_1h, target_ts)
        self.buf_eth_1h = calcule_feature.backfill_gap_if_needed("ethusd", 3600, self.buf_eth_1h, target_ts)

        # BUG FIXE (2026-07-15, audit complet) : buf_btc_4h n'avait AUCUNE
        # protection anti-trou (freshness_retry/backfill_gap_if_needed ci-dessus
        # ne verifient qu'un timestamp PRECIS, alors que buf_btc_4h est consomme
        # par POSITION dans le tableau par dmi_blocks()/filters._wilder_dmi,
        # jamais par timestamp exact) -- ferme une exposition reelle deja
        # prouvee possible (meme cause que le trou 1h/eth deja corrige : Bitstamp
        # omet parfois une bougie). heal_gaps scanne tout le buffer et comble le
        # premier trou trouve ; applique aussi aux 2 autres buffers en defense
        # complementaire (backfill_gap_if_needed ne verifie que LE bar cible du
        # cycle courant, pas tout l'historique consomme par les calculs roulants
        # comme l'ATR de compute_buckets).
        self.buf_btc_1h = calcule_feature.heal_gaps("btcusd", 3600, self.buf_btc_1h)
        self.buf_btc_4h = calcule_feature.heal_gaps("btcusd", 14400, self.buf_btc_4h)
        self.buf_eth_1h = calcule_feature.heal_gaps("ethusd", 3600, self.buf_eth_1h)

        # Purge la queue pour ne pas grossir indefiniment (garde une marge x1.5).
        self.buf_btc_1h = self.buf_btc_1h[-int(BTC_1H_BARS * 1.5):]
        self.buf_btc_4h = self.buf_btc_4h[-int(BTC_4H_BARS * 1.5):]
        self.buf_eth_1h = self.buf_eth_1h[-int(ETH_1H_BARS * 1.5):]

        if self.regime_state is not None:
            # BUG FIXE (2026-07-15, audit complet) : ces closures re-fetchaient
            # TOUJOURS depuis Bitstamp (jusqu'a ~726 bougies 1h + ~142 bougies 4h
            # CHAQUE HEURE, meme pour l'avance normale d'UN SEUL bar) au lieu de
            # reutiliser self.buf_btc_1h/self.buf_btc_4h deja en memoire --
            # contredisait le principe documente en tete de ce fichier ("jamais
            # un re-fetch complet"). Sert depuis le buffer local si suffisant
            # (cas normal), retombe sur un fetch reseau seulement si le buffer
            # n'a pas assez d'historique (ex. apres une longue coupure -- meme
            # comportement qu'avant dans ce cas rare).
            def _fetch_1h(n_bars):
                if len(self.buf_btc_1h) >= n_bars:
                    return self.buf_btc_1h[-n_bars:]
                return calcule_feature.fetch_klines("btcusd", 3600, n_bars)

            def _fetch_4h(n_bars):
                if len(self.buf_btc_4h) >= n_bars:
                    return self.buf_btc_4h[-n_bars:]
                return calcule_feature.fetch_klines("btcusd", 14400, n_bars)

            regime = regime_live.bootstrap_and_advance(self.regime_state, target_ts, _fetch_1h, _fetch_4h)
            regime_live.save_state(self.regime_state, self.regime_state_path)
            logger.info(f"Regime : {regime} (bar {target_ts})")

    # -- Filtre DMI PAR-TRADE (config_main.DMI_FILTER_ENABLED) -----------------
    def dmi_blocks(self, direction: str, decision_ts: int) -> bool:
        """Equivalent live de filters.apply_dmi_filter, mais calcule depuis
        buf_btc_4h (local, Bitstamp) au lieu de filters.load_dmi_map() (SQLite).
        Reutilise filters._wilder_dmi et filters.dmi_blocks_direction TELS QUELS
        -- seule la SOURCE de donnee change, jamais la formule (symetrie preservee)."""
        if not DMI_FILTER_ENABLED or not self.buf_btc_4h:
            return False
        ts = np.array([b[0] for b in self.buf_btc_4h], dtype=np.int64)
        high = np.array([b[2] for b in self.buf_btc_4h], dtype=float)
        low = np.array([b[3] for b in self.buf_btc_4h], dtype=float)
        close = np.array([b[4] for b in self.buf_btc_4h], dtype=float)
        plus_di, minus_di = filters._wilder_dmi(high, low, close, DMI_LENGTH)
        dmi_map = {int(t): (float(p), float(m)) for t, p, m in zip(ts, plus_di, minus_di)}
        return filters.dmi_blocks_direction(direction, dmi_map, decision_ts)

    # -- Regime map (contrat inchange cote filters.py : regime_map.get(ts, "neutral")) --
    def current_regime_map(self) -> "_ConstantRegimeMap | None":
        """Renvoie un objet duck-type-compatible dict.get(ts, default) qui repond
        TOUJOURS le regime COURANT, quel que soit ts demande. Le regime change au
        plus 1x/heure (confirm_bars/cooldown_bars, cf regime/config_regime.py) --
        entre le bar de DECISION d'une entree maker et son FILL (<= quelques
        minutes plus tard, position.py::advance_pending_entry), il n'a physiquement
        pas pu changer. Un dict a une seule cle {target_ts: regime} casserait ce
        lookup des que ts != target_ts exactement (ex: decision_ts_ms d'une entree
        encore pending d'un cycle precedent) -- ce wrapper l'evite."""
        if self.regime_state is None:
            return None
        return _ConstantRegimeMap(str(self.regime_state["state"]["regime"]))


class _ConstantRegimeMap:
    def __init__(self, regime: str):
        self._regime = regime

    def get(self, key, default=None):
        return self._regime
