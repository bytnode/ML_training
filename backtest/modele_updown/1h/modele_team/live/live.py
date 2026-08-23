"""live.py -- daemon de trading live OKX pour l'equipe 1h (up + down).

Boucle 1-minute :
  1. CHAQUE minute : PositionManager.check_positions (SL/TP/trailing/timestop
     sur les positions ouvertes) + advance_pending_entry (fait progresser une
     entree maker en cours, sans jamais bloquer la boucle).
  2. AU TOP DE L'HEURE (minute==0) : rafraichit les buffers marche (market_data),
     predit les 2 modeles, applique la regle d'equipe + filters.py (DMI, 3
     filtres UP, wrapper regime), et si un pari survit + un slot est libre ->
     demarre une entree (start_entry, maker ou taker selon config_main.MAKER).

SYMETRIE : la decision (regle d'equipe, DMI, filtres UP, seuils sidecar) suit
EXACTEMENT la meme logique que modele_team/main.py, via le meme module
`filters.py` et les memes `modele_up.api` / `modele_down.api` (memes modeles,
memes sidecars). Seule l'EXECUTION differe (OKX au lieu d'une simulation).

AUTONOMIE DONNEES (campagne 2026-07-10) : live/ ne touche PLUS JAMAIS la SQLite
de modele_down, ni au demarrage ni en cours de route. Tout vient de Bitstamp,
fetch complet UNE FOIS au lancement puis maintenu incrementalement a chaque
cycle (cf market_data.py) :
  - features des 2 modeles + filtre DMI par-trade : buffers BTC 1h/4h + ETH 1h,
    en RAM, mis a jour par petites requetes incrementales (jamais un re-fetch
    complet).
  - regime (bear/neutral/bull) : calcul incremental depuis regime/regime_seed.json
    (queue glissante 730j + etat de la machine a etats hysteresis, cf
    regime_live.py), avance d'UN bar par cycle et persiste dans
    logs/regime_state.json -- jamais de trou, un redemarrage comble
    automatiquement l'ecart en rejouant les heures manquantes depuis Bitstamp.

Lancement :
    cd backtest/modele_updown/1h/modele_team/live
    python live.py            # DRY_RUN selon live/.env
"""
from __future__ import annotations

import csv
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

_LIVE_DIR = Path(__file__).resolve().parent
_TEAM_DIR = _LIVE_DIR.parent
for _p in (str(_LIVE_DIR), str(_TEAM_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import filters  # noqa: E402
from config_main import (  # noqa: E402
    UP_DROP_HOURS, MAKER, MAX_CONCURRENT_POSITIONS, UP_XMODEL_MAX_PDOWN,
    CHOP_SCALP_ENABLED,
)
from modele_up import api as api_up_mod  # noqa: E402
from modele_down import api as api_down_mod  # noqa: E402

import api_okx  # noqa: E402
import calcule_feature  # noqa: E402
import market_data  # noqa: E402
import position  # noqa: E402
import notify  # noqa: E402
from config_live import (  # noqa: E402
    OKX_INST_ID, OKX_LEVERAGE, OKX_POSITION_FRACTION, OKX_MGN_MODE, OKX_POS_MODE,
    CYCLES_CSV, POSITIONS_CSV, LIVE_LOG_FILE, LOGS_DIR, is_dry_run,
    SIZING_SAFETY_BUFFER, MIN_SIZE_OVERSHOOT_TOL, MAX_CLOCK_SKEW_S, CLOCK_RESYNC_MIN,
)


class SizeTooSmall(Exception):
    """Solde insuffisant pour la taille cible sans depasser minSz (point #9) --
    le trade est saute proprement, ce n'est pas une erreur de cycle."""

INTERVAL_1H_MS = 60 * 60 * 1000
REGIME_STATE_PATH = LOGS_DIR / "regime_state.json"

logger = logging.getLogger("live")


# ----------------------------------------------------------------------------
#  Logging (console + fichier + Telegram sur ERROR)
# ----------------------------------------------------------------------------
def setup_logging() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root.addHandler(console)

    file_handler = logging.FileHandler(LIVE_LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    tg_handler = notify.TelegramErrorHandler()
    tg_handler.setLevel(logging.ERROR)
    tg_handler.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(tg_handler)


# ----------------------------------------------------------------------------
#  CSV logging (cycles.csv, positions.csv)
# ----------------------------------------------------------------------------
def _append_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)


def _handle_position_event(ev: dict) -> None:
    if ev["type"] == "closed":
        logger.info(f"Position fermee {ev['pos_side']} reason={ev['reason']} pp={ev['pp']:+.4%}")
        notify.send_bet_resolved(
            ev["pos_side"], ev["reason"], ev["pp"], ev["avg_px"], ev["mark_px"],
            ev["peak_pp"], ev["worst_pp"], ev["notional_usd"],
            ev["wins"], ev["losses"], ev["balance"],
        )
        _append_csv(POSITIONS_CSV, {
            "ts_ms": ev["exit_ts_ms"], "pos_side": ev["pos_side"], "event": "closed",
            "reason": ev["reason"], "pp": ev["pp"], "avg_px": ev["avg_px"], "mark_px": ev["mark_px"],
            "entry_ts_ms": ev["entry_ts_ms"], "peak_pp": ev["peak_pp"], "worst_pp": ev["worst_pp"],
            "pnl_usdt": ev["pp"] * ev["notional_usd"],
        })
    elif ev["type"] == "desync_closed":
        # logger.warning (pas .error) : notify.send_desync juste en dessous envoie
        # deja ce meme evenement formate -- un .error ici ferait doublon via
        # TelegramErrorHandler (cf bug triplicata 2026-07-15).
        logger.warning(f"Desync {ev['pos_side']} : position absente cote OKX.")
        notify.send_desync(ev["pos_side"])
        _append_csv(POSITIONS_CSV, {"ts_ms": int(time.time() * 1000), "pos_side": ev["pos_side"],
                                    "event": "desync_closed", "reason": "", "pp": "", "avg_px": "",
                                    "mark_px": "", "entry_ts_ms": "", "peak_pp": "", "worst_pp": "",
                                    "pnl_usdt": ""})
    elif ev["type"] == "close_failed":
        # logger.warning (pas .error) : notify.send_close_failed juste en dessous
        # envoie deja ce meme evenement -- meme raison que desync_closed ci-dessus.
        logger.warning(f"Echec fermeture {ev['pos_side']} ({ev['reason']}) : {ev['error']}")
        notify.send_close_failed(ev["pos_side"], ev["reason"], ev["error"])
    elif ev["type"] == "opened":
        px = ev.get("avg_px")
        logger.info(f"Position ouverte {ev['pos_side']} mode={ev['mode']}"
                    + (f" avg_px={px}" if px else ""))
        notify.send_bet_placed(ev["pos_side"], ev["mode"], "", px)
        _append_csv(POSITIONS_CSV, {"ts_ms": int(time.time() * 1000), "pos_side": ev["pos_side"],
                                    "event": "opened", "reason": ev["mode"], "pp": "",
                                    "avg_px": px or "", "mark_px": "", "entry_ts_ms": "",
                                    "peak_pp": "", "worst_pp": "", "pnl_usdt": ""})
    elif ev["type"] == "entry_posted":
        logger.info(f"Entree postee {ev['pos_side']} mode={ev['mode']}")
    elif ev["type"] == "entry_missed":
        # ALIGNEMENT BACKTEST 2026-08-14 : fenetre maker ecoulee sans fill =
        # opportunite RATEE (plus de bascule taker). Symetrie avec le "missed"
        # du backtest (mode MAKER, main.py) -- trace + notif, aucun ordre.
        logger.info(f"Entree maker {ev['pos_side']} NON remplie dans la fenetre -> "
                    f"opportunite ratee (aucune bascule taker, semantique backtest).")
        notify.send_entry_missed(ev["pos_side"])
        _append_csv(POSITIONS_CSV, {"ts_ms": int(time.time() * 1000), "pos_side": ev["pos_side"],
                                    "event": "entry_missed", "reason": "maker_window_expired",
                                    "pp": "", "avg_px": "", "mark_px": "", "entry_ts_ms": "",
                                    "peak_pp": "", "worst_pp": "", "pnl_usdt": ""})
    elif ev["type"] == "widened":
        # ALIGNEMENT GRILLE 1-MIN (2026-08-23) : l'elargissement n'est plus
        # decide au moment ou live.py voit le signal bloque, mais PAR LA MARCHE
        # de sortie, au premier candidat atteint EN PROFIT (regle exacte de
        # main.py). L'evenement remonte donc depuis check_positions.
        logger.info(f"Position {ev['pos_side']} ELARGIE (TP/trailing/hold etendus, SL inchange).")
        notify.send_position_widened(ev["pos_side"])
        _append_csv(POSITIONS_CSV, {"ts_ms": int(time.time() * 1000), "pos_side": ev["pos_side"],
                                    "event": "widened", "reason": "", "pp": "", "avg_px": "",
                                    "mark_px": "", "entry_ts_ms": "", "peak_pp": "",
                                    "worst_pp": "", "pnl_usdt": ""})
    elif ev["type"] == "entry_check_failed":
        # Etat du fill maker inconnu (lecture OKX en echec) : position.py
        # retente au tick suivant sans rien poster -- ici on trace seulement.
        logger.warning(f"Lecture de l'entree maker {ev['pos_side']} en echec : {ev['error']} "
                       f"-- nouvelle tentative au prochain tick.")
    elif ev["type"] == "adopted":
        # Position trouvee ouverte sur OKX au demarrage, reprise en gestion (point #5).
        logger.warning(f"Reconciliation : position {ev['pos_side']} adoptee avg_px={ev['avg_px']}")
        notify.send_position_adopted(ev["pos_side"], ev["avg_px"])
        _append_csv(POSITIONS_CSV, {"ts_ms": int(time.time() * 1000), "pos_side": ev["pos_side"],
                                    "event": "adopted", "reason": "", "pp": "",
                                    "avg_px": ev["avg_px"], "mark_px": "", "entry_ts_ms": "",
                                    "peak_pp": "", "worst_pp": "", "pnl_usdt": ""})
    elif ev["type"] == "reconciled_closed":
        logger.info(f"Reconciliation : {ev['pos_side']} efface (aucune position OKX correspondante).")
        _append_csv(POSITIONS_CSV, {"ts_ms": int(time.time() * 1000), "pos_side": ev["pos_side"],
                                    "event": "reconciled_closed", "reason": "", "pp": "",
                                    "avg_px": "", "mark_px": "", "entry_ts_ms": "",
                                    "peak_pp": "", "worst_pp": "", "pnl_usdt": ""})


# ----------------------------------------------------------------------------
#  Bootstrap OKX (mode position, levier)
# ----------------------------------------------------------------------------
def bootstrap(client: api_okx.OkxClient) -> None:
    # BUG FIXE (audit 2026-07-10) : set_position_mode/set_leverage sont
    # generalement REJETES par OKX s'il y a deja une position ouverte (compte
    # entier pour posMode, cet instId pour set_leverage) -- exactement le cas
    # d'un REDEMARRAGE (crash, redeploy, reboot host) pendant qu'une position
    # tourne. Une exception non rattrapee ici plantait tout le process AVANT
    # meme d'atteindre PositionManager -- la position existante restait alors
    # SANS AUCUNE surveillance SL/TP/trailing/timestop. Chaque appel est donc
    # best-effort : loggue une erreur mais laisse le daemon continuer vers la
    # reconciliation/surveillance des positions.
    # Synchronisation d'horloge AVANT toute requete signee (point #11 audit
    # 2026-08-03) : les lectures signees (positions/balance) tapent l'API reelle
    # meme en DRY_RUN, donc utile dans les deux modes. Une horloge hote derivee
    # de >30s ferait rejeter toute ecriture (order/cancel) une fois en reel.
    try:
        skew_s = client.sync_time()
        if abs(skew_s) > MAX_CLOCK_SKEW_S:
            logger.error(f"Horloge hote derivee de {skew_s:+.1f}s vs serveur OKX "
                        f"(> {MAX_CLOCK_SKEW_S}s) -- offset applique aux signatures, MAIS corrige "
                        f"l'horloge (NTP) : au-dela de ~30s OKX rejette toute requete signee.")
        else:
            logger.info(f"Horloge synchronisee avec OKX (derive {skew_s:+.2f}s).")
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"sync_time() a echoue : {exc} -- signatures sur l'horloge locale.")

    try:
        client.set_position_mode(OKX_POS_MODE)
    except Exception as exc:  # noqa: BLE001
        logger.error(f"set_position_mode({OKX_POS_MODE}) echoue (position deja ouverte ?) : {exc} "
                    f"-- poursuite (le mode est peut-etre deja correct).")
    for pos_side in ("long", "short"):
        try:
            client.set_leverage(OKX_INST_ID, OKX_LEVERAGE, OKX_MGN_MODE, pos_side=pos_side)
        except Exception as exc:  # noqa: BLE001
            logger.error(f"set_leverage({pos_side}) echoue (position deja ouverte ?) : {exc} "
                        f"-- poursuite (le levier est peut-etre deja correct).")
    # Verification best-effort du levier REELLEMENT applique (point #7) -- purement
    # informatif ici ; run_hourly_cycle le re-fixe ET le confirme juste avant
    # chaque entree, et dimensionne sur le levier reel.
    if not client.dry_run:
        for pos_side in ("long", "short"):
            try:
                actual = client.get_leverage(OKX_INST_ID, OKX_MGN_MODE, pos_side=pos_side)
                if actual and int(round(actual)) != int(OKX_LEVERAGE):
                    logger.warning(f"Levier {pos_side} reel={actual} != cible={OKX_LEVERAGE} "
                                   f"-- sera re-tente/confirme avant chaque entree.")
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Lecture levier {pos_side} au bootstrap echouee : {exc}")
    logger.info(f"Bootstrap termine : {OKX_INST_ID} lever={OKX_LEVERAGE} mgnMode={OKX_MGN_MODE} "
                f"posMode={OKX_POS_MODE} dry_run={client.dry_run}")


def build_client() -> api_okx.OkxClient:
    key = os.environ.get("OKX_API_KEY", "")
    secret = os.environ.get("OKX_API_SECRET", "")
    passphrase = os.environ.get("OKX_API_PASSPHRASE", "")
    dry = is_dry_run()
    if not dry and not (key and secret and passphrase):
        raise RuntimeError(
            "DRY_RUN=false mais OKX_API_KEY/OKX_API_SECRET/OKX_API_PASSPHRASE "
            "absentes de l'environnement -- impossible de trader en reel."
        )
    return api_okx.OkxClient(key, secret, passphrase, dry_run=dry)


def _retry(fn, tries: int = 2, delay_s: float = 2.0):
    """BUG FIXE (2026-07-15, audit complet) : get_balance/get_ticker (lecture
    OKX) n'avaient aucun retry -- un blip reseau transitoire pendant le sizing
    de l'entree faisait perdre TOUT le signal de l'heure (fail-safe, pas de
    corruption d'etat, mais une opportunite de trade valide jetee pour rien).
    2 tentatives, 2s d'ecart -- jamais bloquant plus de ~2s dans le pire cas."""
    last_exc: Exception | None = None
    for i in range(tries):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if i < tries - 1:
                time.sleep(delay_s)
    raise last_exc  # type: ignore[misc]


# ----------------------------------------------------------------------------
#  Sizing : notionnel USD -> contrats OKX (ctVal)
# ----------------------------------------------------------------------------
def compute_size_contracts(client: api_okx.OkxClient, notional_usd: float, price: float) -> str:
    instrument = client.get_instrument(OKX_INST_ID)
    ct_val = float(instrument["ctVal"])
    lot_sz = float(instrument["lotSz"])
    min_sz = float(instrument["minSz"])
    if notional_usd <= 0 or price <= 0 or ct_val <= 0:
        # BUG FIXE (2026-07-15, audit complet) : sans ce garde-fou, un notionnel
        # <=0 (solde nul/negatif, OKX_POSITION_FRACTION mal configure) arrondissait
        # quand meme a min_sz via max(min_sz, 0) plus bas -- postait un ordre
        # REEL de taille minimale au lieu de ne rien poster du tout.
        raise ValueError(f"sizing invalide (notional_usd={notional_usd} price={price} "
                         f"ct_val={ct_val}) -- aucun ordre ne doit partir.")
    raw_contracts = notional_usd / (price * ct_val)
    # BUG FIXE (point #9 audit 2026-08-03) : si le notionnel voulu est nettement
    # SOUS la taille minimale d'OKX, arrondir a min_sz posterait un ordre PLUS
    # GROS que voulu (solde trop faible pour la taille cible). On saute alors le
    # trade proprement plutot que de sur-trader. Tolerance : on accepte un leger
    # arrondi vers min_sz (MIN_SIZE_OVERSHOOT_TOL), au-dela on refuse.
    if raw_contracts < min_sz * (1.0 - MIN_SIZE_OVERSHOOT_TOL):
        raise SizeTooSmall(
            f"notionnel {notional_usd:.2f} USD -> {raw_contracts:.6f} contrats, "
            f"sous minSz={min_sz} (tol {MIN_SIZE_OVERSHOOT_TOL:.0%}) -- trade saute.")
    contracts = max(min_sz, round(raw_contracts / lot_sz) * lot_sz)
    # BUG FIXE (2026-07-15, audit complet) : `:g` peut emettre de la notation
    # scientifique (ex. "1e-05") pour un `contracts` tres petit -- OKX attend un
    # nombre decimal classique pour `sz`. Virgule fixe, zeros/point de fin
    # retires -- jamais de notation scientifique.
    return f"{contracts:.8f}".rstrip("0").rstrip(".")


# ----------------------------------------------------------------------------
#  Cycle horaire : rafraichit market_data -> features -> predict -> regle
#  d'equipe -> filtres
# ----------------------------------------------------------------------------
def run_hourly_cycle(client: api_okx.OkxClient, pm: position.PositionManager,
                     md: market_data.LiveMarketData) -> None:
    now_ms = int(time.time() * 1000)
    # `target` DOIT etre calcule depuis l'HORLOGE MURALE, jamais depuis un
    # buffer deja fetch -- sinon la verification de fraicheur (dans md.refresh)
    # deviendrait un no-op. Si Bitstamp est en retard, md.refresh leve
    # TimeoutError plutot que de laisser passer un cycle sur une bougie perimee.
    current_hour_start = (now_ms // INTERVAL_1H_MS) * INTERVAL_1H_MS
    target = current_hour_start - INTERVAL_1H_MS   # bougie 1h qui vient de fermer (open time)

    try:
        md.refresh(target)
    except TimeoutError as exc:
        logger.error(f"Fraicheur Bitstamp non confirmee pour target={target} : {exc} -- cycle annule.")
        return

    buf_btc = md.buf_btc_1h
    buf_eth = md.buf_eth_1h
    regime_map = md.current_regime_map()
    # Contexte de tendance SMA/ER (ALIGNEMENT BACKTEST 2026-08-14) : meme math
    # que main.py (filters.trend_context_from_series), construit depuis le
    # buffer Bitstamp 1h. Alimente les bumps d'entree chop (ci-dessous) et les
    # overrides de sortie (via pm.start_entry -> _decision_ctx). None au warmup
    # -> comportement trend partout, comme les barres warmup du backtest.
    # Gate maitre CHOP_SCALP_ENABLED : MEME interrupteur que main.py:run_backtest
    # (trend_ctx=None si False) -- sans lui, couper le flag cote backtest
    # laisserait le live appliquer tout le mecanisme en silence (finding revue
    # adversariale 2026-08-14).
    trend = md.trend_ctx() if CHOP_SCALP_ENABLED else None
    in_chop = filters.chop_mode(trend, target) if trend is not None else False

    feats_up = calcule_feature.build_features_up(buf_btc, target)
    feats_down = calcule_feature.build_features_down(buf_btc, buf_eth, target)
    vs_up, iw_up = calcule_feature.compute_buckets(buf_btc, target, calcule_feature._MODELE_UP_DIR)
    vs_dn, iw_dn = calcule_feature.compute_buckets(buf_btc, target, calcule_feature._MODELE_DOWN_DIR)

    model_up = api_up_mod._load_model()
    sidecar_up = api_up_mod._load_sidecar()
    cols_up = model_up.get_booster().feature_names
    p_up = float(model_up.predict_proba(pd.DataFrame([feats_up])[cols_up])[:, 1][0])
    th_up = api_up_mod._bucket_threshold(vs_up, iw_up, sidecar_up, ts_ms=target)

    model_down = api_down_mod._load_model()
    sidecar_down = api_down_mod._load_sidecar()
    cols_down = model_down.get_booster().feature_names
    p_down = float(model_down.predict_proba(pd.DataFrame([feats_down])[cols_down])[:, 1][0])
    th_down_base = api_down_mod._bucket_threshold(vs_dn, iw_dn, sidecar_down, ts_ms=target)
    eff_delta = filters.effective_down_delta_1(target, regime_map, trend)
    th_down = th_down_base + eff_delta

    take_up = p_up >= th_up
    take_down = p_down >= th_down

    hour_utc = pd.Timestamp(target, unit="ms", tz="UTC").hour
    # NOUVEAU (2026-07-15, logs lisibles) : capture si cette regle a REELLEMENT
    # force take_up a False (pour le log) -- comportement de decision inchange.
    up_hour_blocked = bool(UP_DROP_HOURS and hour_utc in UP_DROP_HOURS and take_up)
    if up_hour_blocked:
        take_up = False

    # Cross-model UP : PRE-coherence (ALIGNEMENT BACKTEST 2026-08-14 -- main.py
    # applique apply_xmodel_filter sur up_bets AVANT build_team_bets). L'ancien
    # emplacement post-coherence divergeait sur les barres double-pari : le
    # backtest retire le UP et PREND le DOWN ; le live faisait CANCEL et jetait
    # les deux. Meme regle desormais.
    up_xmodel_blocked = bool(take_up and filters.up_xmodel_blocks(p_down))
    if up_xmodel_blocked:
        take_up = False

    if take_up and take_down:
        decision = "CANCEL"
    elif take_up:
        decision = "UP"
    elif take_down:
        decision = "DOWN"
    else:
        decision = "PASS"

    team_decision = decision   # NOUVEAU : decision d'equipe AVANT les filtres, pour le log

    dmi_blocked = False
    if decision in ("UP", "DOWN"):
        direction = "up" if decision == "UP" else "down"
        if md.dmi_blocks(direction, target):
            dmi_blocked = True
            decision = "PASS"

    # NOUVEAU : calcule desormais INCONDITIONNELLEMENT (fonction pure, aucun
    # effet de bord) pour etre toujours disponible dans le log, meme quand
    # decision != "UP" -- comportement de decision inchange. trend (alignement
    # 2026-08-14) : le bump chop CHOP_UP_CONF_MIN_BUMP s'applique aussi en live.
    eff_min = filters.effective_up_conf_min_1(target, regime_map, trend)
    up_conf_blocked = False
    if decision == "UP":
        if p_up < eff_min:
            up_conf_blocked = True
            decision = "PASS"

    _append_csv(CYCLES_CSV, {
        "run_at_ms": now_ms, "target_open_ms": target, "p_up": p_up, "p_down": p_down,
        "th_up": th_up, "th_down": th_down, "team_decision": team_decision,
        "decision": decision, "dmi_blocked": dmi_blocked, "up_hour_blocked": up_hour_blocked,
        "up_conf_blocked": up_conf_blocked, "up_xmodel_blocked": up_xmodel_blocked,
    })

    # NOUVEAU (2026-07-15) : bloc de log multi-lignes lisible d'un coup d'oeil
    # dans `docker compose logs -f live-okx`, remplace l'ancienne ligne unique.
    # Montre les 2 probas vs seuils, la regle d'equipe AVANT filtres, et QUEL
    # filtre precis (s'il y en a un) a bloque le signal.
    # NOTE : up_hour_blocked est traite A PART de team_decision -- il ecarte le
    # signal UP AVANT meme la regle de coherence, donc peut coexister avec un
    # team_decision="DOWN" (DOWN part normalement, ce n'est PAS lui qui est
    # bloque) ou team_decision="PASS" (si aucun cote ne survit). L'attribuer au
    # blocage du signal FINAL choisi par l'equipe serait trompeur.
    notes = []
    if up_hour_blocked:
        notes.append(f"UP aurait declenche (p_up>=seuil) mais heure UTC {hour_utc}h est "
                     f"dans UP_DROP_HOURS -> ecarte avant la regle d'equipe")
    if up_xmodel_blocked:
        # PRE-coherence depuis l'alignement 2026-08-14 : comme UP_DROP_HOURS,
        # peut coexister avec une decision DOWN (le DOWN part normalement).
        notes.append(f"UP ecarte avant la regle d'equipe (cross-model) : p_down={p_down:.4f} "
                     f">= UP_XMODEL_MAX_PDOWN={UP_XMODEL_MAX_PDOWN:.4f}")
    if in_chop:
        notes.append("Contexte : CHOP scalpable (ER<0.25, bear<=30j) -> bumps d'entree "
                     "+ sorties scalp sur ce cycle.")
    if team_decision == "CANCEL":
        notes.append("Coherence : LES DEUX modeles ont parie -> ANNULE (CANCEL, rare).")
    elif team_decision == "PASS":
        notes.append("Coherence : aucun signal exploitable -> PASS.")
    else:
        blocked_msgs = []
        if dmi_blocked:
            blocked_msgs.append(f"DMI contraire au sens {team_decision}")
        if up_conf_blocked:
            blocked_msgs.append(f"confiance UP {p_up:.4f} < seuil effectif {eff_min:.4f}")
        notes.append(f"Signal {team_decision} BLOQUE par : " + " ; ".join(blocked_msgs)
                     if blocked_msgs else f"Signal {team_decision} : aucun filtre n'a bloque.")
    reasoning = "\n".join(f"  {n}" for n in notes)

    logger.info(
        "\n" + "=" * 64 + "\n"
        f"CYCLE {pd.Timestamp(now_ms, unit='ms', tz='UTC').strftime('%Y-%m-%d %H:%M')} UTC  "
        f"(bougie {pd.Timestamp(target, unit='ms', tz='UTC').strftime('%Y-%m-%dT%H:%M:%SZ')})\n"
        + "-" * 64 + "\n"
        f"  UP   : p={p_up:.4f}  seuil={th_up:.4f} (conf_min_eff={eff_min:.4f})  "
        f"vol={vs_up} wknd={iw_up}  -> {'PREND' if p_up >= th_up else 'passe'}\n"
        f"  DOWN : p={p_down:.4f}  seuil={th_down:.4f} (base={th_down_base:.4f} delta={eff_delta:+.4f})  "
        f"vol={vs_dn} wknd={iw_dn}  -> {'PREND' if p_down >= th_down else 'passe'}\n"
        + "-" * 64 + "\n"
        + reasoning + "\n"
        + "-" * 64 + "\n"
        f"  DECISION FINALE : {decision}\n"
        + "=" * 64
    )

    if decision not in ("UP", "DOWN"):
        return

    pos_side = "long" if decision == "UP" else "short"
    traded_ts_ms = target + INTERVAL_1H_MS

    if not pm.slot_free(pos_side):
        # Elargissement sur profit (symetrie main.py, campagne 2026-07-10) : au
        # lieu de jeter le signal, on l'enregistre comme CANDIDAT d'elargissement
        # de la position deja ouverte de ce sens. ALIGNEMENT GRILLE 1-MIN
        # (2026-08-23) : c'est la marche de sortie qui elargit, au premier
        # candidat atteint EN PROFIT (main.py::_up_managed_exit) -- ici on ne
        # fait plus qu'enregistrer le traded_ts, comme le backtest construit son
        # `widen_candidates` avec TOUS les signaux bloques a venir.
        if pm.register_widen_candidate(pos_side, traded_ts_ms):
            logger.info(f"{decision} : slot {pos_side} deja occupe -> signal enregistre comme "
                        f"CANDIDAT d'elargissement (applique des que la position sera en profit).")
        else:
            logger.info(f"{decision} : slot {pos_side} deja occupe, signal ignore "
                        f"(cap {MAX_CONCURRENT_POSITIONS} positions, cf CLAUDE.md).")
        return

    # Point #7 : (re)fixe et CONFIRME le levier de ce sens juste avant l'entree,
    # puis dimensionne avec le levier REELLEMENT applique (marge = notionnel /
    # levier). Un ecart set_leverage silencieux fausserait sinon la marge requise.
    effective_leverage = float(OKX_LEVERAGE)
    if not client.dry_run:
        try:
            client.set_leverage(OKX_INST_ID, OKX_LEVERAGE, OKX_MGN_MODE, pos_side=pos_side)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"set_leverage({pos_side}) avant entree echoue : {exc}")
        try:
            actual = client.get_leverage(OKX_INST_ID, OKX_MGN_MODE, pos_side=pos_side)
            if actual > 0:
                effective_leverage = actual
                if int(round(actual)) != int(OKX_LEVERAGE):
                    logger.warning(f"Entree {pos_side} : levier effectif {actual} != configure "
                                   f"{OKX_LEVERAGE} -- sizing sur le levier REEL.")
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"get_leverage({pos_side}) echoue : {exc} -- sizing sur levier configure.")

    # Point #6 : dimensionne sur la marge DISPONIBLE (availEq), pas l'equity
    # totale -- une position deja ouverte gele sa marge, la suivante ne doit pas
    # re-supposer le capital plein (sinon rejet OKX / 100% engage sans coussin).
    avail = _retry(lambda: client.get_balance("USDT", field="availEq"))
    ticker = _retry(lambda: client.get_ticker(OKX_INST_ID))
    price = float(ticker["last"])
    notional = avail * OKX_POSITION_FRACTION * effective_leverage
    # Plafonne pour que la marge requise ne depasse jamais la marge disponible
    # (coussin frais/funding via SIZING_SAFETY_BUFFER).
    max_margin = avail * SIZING_SAFETY_BUFFER
    if notional / effective_leverage > max_margin:
        capped = max_margin * effective_leverage
        logger.warning(f"Entree {pos_side} : notionnel {notional:.2f} plafonne a {capped:.2f} "
                       f"(marge dispo {avail:.2f} x buffer {SIZING_SAFETY_BUFFER}).")
        notional = capped

    try:
        sz = compute_size_contracts(client, notional, price)
    except SizeTooSmall as exc:
        logger.info(f"{decision} : entree sautee -- {exc}")
        return

    # ANCRE D'ENTREE (ALIGNEMENT GRILLE 1-MIN, 2026-08-23) : en mode MAKER le
    # backtest poste sa limite a P0 = open_ib.get(traded_ts), c.-a-d. l'OPEN de
    # la bougie 1-min d'open-time traded_ts (= l'open de la bougie 1h qu'on
    # trade). On lit desormais ce MEME prix chez OKX au lieu d'un instantane
    # bid/ask : la bougie est en formation a cette seconde (confirm=0), mais son
    # open est fige des sa premiere transaction -> closed_only=False.
    # En mode TAKER le backtest entre aussi a P0 ; le live entre au marche (le
    # slippage reel est irreductible) et enregistre le vrai prix de fill.
    price_hint = float(ticker["bidPx"] if pos_side == "long" else ticker["askPx"])
    if MAKER:
        try:
            p0_rows = client.get_candles_range(OKX_INST_ID, traded_ts_ms,
                                               traded_ts_ms + 60_000, bar="1m",
                                               closed_only=False)
            if p0_rows:
                price_hint = float(p0_rows[0][1])       # open de la bougie traded_ts
            else:
                logger.warning(f"Ancre P0 introuvable pour traded_ts={traded_ts_ms} -- "
                               f"repli sur le bid/ask instantane ({price_hint}).")
        except Exception as exc:  # noqa: BLE001 -- repli documente, jamais bloquant
            logger.warning(f"Lecture de l'ancre P0 echouee ({exc}) -- repli sur le "
                           f"bid/ask instantane ({price_hint}).")

    result = pm.start_entry(pos_side, sz, price_hint, target, traded_ts_ms, regime_map,
                            maker=MAKER, notional_usd=notional, trend_ctx=trend)
    if result:
        _handle_position_event(result)


# ----------------------------------------------------------------------------
#  Boucle principale
# ----------------------------------------------------------------------------
def _sleep_until_next_minute() -> None:
    now = time.time()
    time.sleep(60.0 - (now % 60.0))


def main() -> None:
    setup_logging()
    logger.info(f"Demarrage live.py -- inst={OKX_INST_ID} DRY_RUN={is_dry_run()}")

    client = build_client()
    bootstrap(client)
    pm = position.PositionManager(client)

    md = market_data.LiveMarketData(REGIME_STATE_PATH)
    try:
        md.bootstrap()   # fetch complet des buffers + comble le trou regime (seed -> maintenant)
    except Exception as exc:  # noqa: BLE001
        # BUG FIXE (2026-07-15, audit complet) : sans ce garde-fou, un crash ICI
        # (Bitstamp injoignable, regime_seed.json absent...) tuait le process
        # AVANT la boucle -- ZERO alerte Telegram (TelegramErrorHandler
        # n'intercepte que les logs, pas un process qui meurt), et sous Docker
        # restart:unless-stopped ca crash-loop silencieusement. Si une position
        # etait DEJA ouverte (chargee dans `pm` juste au-dessus), elle se
        # retrouvait sans AUCUNE surveillance SL/TP/trailing/timestop tant que
        # personne ne remarquait le redemarrage en boucle. On alerte
        # explicitement puis on relance -- le process doit quand meme s'arreter
        # (tourner sans donnees marche n'est pas une option), mais au moins
        # l'operateur est prevenu au lieu d'un silence total.
        logger.warning(f"md.bootstrap() a echoue au demarrage : {exc}", exc_info=True)
        notify.send_cycle_error("startup_bootstrap", str(exc))
        raise

    # Reconciliation au demarrage (point #5) : adopte toute position OKX ouverte
    # non suivie localement (etat perdu/corrompu, position hors bande), efface une
    # entree locale sans position OKX (fermee pendant l'arret / etat papier
    # residuel), et garantit le filet catastrophe. No-op en DRY_RUN.
    try:
        for ev in pm.reconcile_positions(int(time.time() * 1000)):
            _handle_position_event(ev)
    except Exception as exc:  # noqa: BLE001
        logger.exception(f"reconcile_positions au demarrage : {exc}")

    processed_hour_key: str | None = None
    last_clock_resync_ms = int(time.time() * 1000)

    while True:
        _sleep_until_next_minute()
        now_ms = int(time.time() * 1000)
        now_dt = datetime.fromtimestamp(now_ms / 1000, tz=timezone.utc)

        # Re-synchronisation periodique de l'offset horloge (audit 2026-08-13) :
        # evite que l'offset cale au boot ne derive jusqu'a franchir la fenetre
        # ~30s d'OKX (rejets 50102). Best-effort -- ne crashe jamais la boucle ;
        # `_request` conserve en plus un self-heal resync+retry sur 50102.
        if CLOCK_RESYNC_MIN > 0 and now_ms - last_clock_resync_ms >= CLOCK_RESYNC_MIN * 60_000:
            last_clock_resync_ms = now_ms
            try:
                skew_s = client.sync_time()
                if abs(skew_s) > MAX_CLOCK_SKEW_S:
                    logger.error(f"Horloge hote derivee de {skew_s:+.1f}s vs OKX (> "
                                f"{MAX_CLOCK_SKEW_S}s) -- offset re-applique, MAIS corrige l'horloge (NTP).")
                else:
                    logger.debug(f"Resync horloge OKX periodique (derive {skew_s:+.2f}s).")
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"resync horloge periodique a echoue : {exc} -- ancien offset conserve.")

        try:
            for ev in pm.check_positions(now_ms):
                _handle_position_event(ev)
        except Exception as exc:  # noqa: BLE001 -- ne jamais crasher la boucle
            logger.exception(f"check_positions : {exc}")

        try:
            ev = pm.advance_pending_entry(now_ms, md.current_regime_map())
            if ev:
                _handle_position_event(ev)
        except Exception as exc:  # noqa: BLE001
            logger.exception(f"advance_pending_entry : {exc}")

        if now_dt.minute == 0:
            hour_key = now_dt.strftime("%Y-%m-%dT%H")
            if hour_key != processed_hour_key:
                processed_hour_key = hour_key
                try:
                    run_hourly_cycle(client, pm, md)
                except Exception as exc:  # noqa: BLE001
                    # logger.warning(exc_info=True) (pas .exception=.error+trace) :
                    # notify.send_cycle_error juste en dessous envoie deja ce meme
                    # evenement -- meme raison que desync_closed/close_failed
                    # ci-dessus. Traceback complet toujours dans live.log (exc_info).
                    logger.warning(f"run_hourly_cycle : {exc}", exc_info=True)
                    notify.send_cycle_error("hourly_cycle", str(exc))


if __name__ == "__main__":
    main()
