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
if str(_LIVE_DIR) not in sys.path:
    sys.path.insert(0, str(_LIVE_DIR))

import filters  # noqa: E402
from config_main import (  # noqa: E402
    UP_DROP_HOURS, MAKER, MAX_CONCURRENT_POSITIONS, UP_XMODEL_MAX_PDOWN,
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
    TICK_INTERVAL_S, CYCLES_CSV, POSITIONS_CSV, LIVE_LOG_FILE, LOGS_DIR, is_dry_run,
)

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
    elif ev["type"] == "entry_repriced":
        logger.info(f"Entree repricee {ev['pos_side']} px={ev['px']}")
    elif ev["type"] == "adopted":
        logger.warning(f"Position {ev['pos_side']} adoptee (avg_px={ev['avg_px']} sz={ev['sz']}) -- "
                       f"absente de l'etat local, reconciliee depuis OKX.")
        notify.send_position_adopted(ev["pos_side"], ev["avg_px"], ev["sz"])
        _append_csv(POSITIONS_CSV, {"ts_ms": int(time.time() * 1000), "pos_side": ev["pos_side"],
                                    "event": "adopted", "reason": "", "pp": "",
                                    "avg_px": ev["avg_px"], "mark_px": "", "entry_ts_ms": "",
                                    "peak_pp": "", "worst_pp": "", "pnl_usdt": ""})
    elif ev["type"] == "entry_abandoned":
        logger.warning(f"Entree {ev['pos_side']} abandonnee (fallback taker rejete) : {ev['error']}")
        notify.send_entry_abandoned(ev["pos_side"], ev["error"])


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
    eff_delta = filters.effective_down_delta_1(target, regime_map)
    th_down = th_down_base + eff_delta

    take_up = p_up >= th_up
    take_down = p_down >= th_down

    hour_utc = pd.Timestamp(target, unit="ms", tz="UTC").hour
    # NOUVEAU (2026-07-15, logs lisibles) : capture si cette regle a REELLEMENT
    # force take_up a False (pour le log) -- comportement de decision inchange.
    up_hour_blocked = bool(UP_DROP_HOURS and hour_utc in UP_DROP_HOURS and take_up)
    if up_hour_blocked:
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
    # decision != "UP" -- comportement de decision inchange.
    eff_min = filters.effective_up_conf_min_1(target, regime_map)
    up_conf_blocked = up_xmodel_blocked = False
    if decision == "UP":
        if p_up < eff_min:
            up_conf_blocked = True
            decision = "PASS"
    if decision == "UP" and filters.up_xmodel_blocks(p_down):
        up_xmodel_blocked = True
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
        if up_xmodel_blocked:
            blocked_msgs.append(f"p_down={p_down:.4f} >= UP_XMODEL_MAX_PDOWN={UP_XMODEL_MAX_PDOWN:.4f}")
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
    if not pm.slot_free(pos_side):
        # Elargissement sur profit (symetrie main.py, campagne 2026-07-10) : au
        # lieu de simplement jeter le signal, on l'utilise comme confirmation
        # de tendance pour elargir la position DEJA ouverte de ce sens, SI elle
        # est actuellement en profit. try_widen() ne fait rien (renvoie False)
        # si WIDEN_ON_PROFIT_ENABLED=False, deja elargie, ou pas en profit.
        widened = pm.try_widen(pos_side)
        if widened:
            logger.info(f"{decision} : slot {pos_side} deja occupe mais position en profit "
                        f"-> ELARGIE (TP/trailing/hold etendus).")
            notify.send_position_widened(pos_side)
            _append_csv(POSITIONS_CSV, {"ts_ms": now_ms, "pos_side": pos_side,
                                        "event": "widened", "reason": "", "pp": "",
                                        "avg_px": "", "mark_px": "", "entry_ts_ms": "",
                                        # BUG FIXE (2026-07-15, audit complet) : ces 3 cles
                                        # manquaient ici (regression de la campagne
                                        # peak_pp/worst_pp/pnl_usdt precedente) alors que
                                        # TOUTES les autres lignes POSITIONS_CSV les ont --
                                        # DictWriter avec un header fixe desalignait
                                        # silencieusement chaque ligne "widened" ecrite.
                                        "peak_pp": "", "worst_pp": "", "pnl_usdt": ""})
        else:
            logger.info(f"{decision} : slot {pos_side} deja occupe, signal ignore "
                        f"(cap {MAX_CONCURRENT_POSITIONS} positions, cf CLAUDE.md).")
        return

    traded_ts_ms = target + INTERVAL_1H_MS
    balance = _retry(lambda: client.get_balance("USDT"))
    ticker = _retry(lambda: client.get_ticker(OKX_INST_ID))
    price = float(ticker["last"])
    notional = balance * OKX_POSITION_FRACTION * OKX_LEVERAGE
    sz = compute_size_contracts(client, notional, price)
    price_hint = float(ticker["bidPx"] if pos_side == "long" else ticker["askPx"])

    result = pm.start_entry(pos_side, sz, price_hint, target, traded_ts_ms, regime_map,
                            maker=MAKER, notional_usd=notional)
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

    processed_hour_key: str | None = None

    while True:
        _sleep_until_next_minute()
        now_ms = int(time.time() * 1000)
        now_dt = datetime.fromtimestamp(now_ms / 1000, tz=timezone.utc)

        try:
            for ev in pm.check_positions(now_ms, md.current_regime_map()):
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
