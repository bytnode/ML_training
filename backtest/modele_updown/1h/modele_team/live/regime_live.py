"""regime_live.py -- calcul INCREMENTAL du regime (bear/neutral/bull), SANS jamais
toucher la SQLite de modele_down, a partir de la graine regime/regime_seed.json.

CONTEXTE : reconstruit ce que faisait l'ancien live_team/regime_live.py -- fichier
jamais commite (untracked) et disparu du working tree pendant cette session, donc
irrecuperable via git. Le mecanisme qu'il consommait, lui, existe toujours et est
reutilise ICI TEL QUEL (jamais reimplemente) pour garantir la symetrie exacte avec
le backtest :
  - regime/export_seed.py documente le format de la graine (raw_axis_tail,
    state, thresholds, tous les parametres de config_regime.py au moment de l'export).
  - regime/state_machine.py::apply_state_machine EST APPELE DIRECTEMENT ici (supporte
    nativement `initial_state=` pour reprendre a partir d'un etat deja avance,
    exactement concu pour ce cas -- rien a reimplementer).
  - Les FORMULES (trend_raw/momentum_raw/dmi_raw, z-score roulant, poids, DMI Wilder)
    sont recopiees depuis regime/scorers.py (meme convention que _wilder_dmi/
    _bucket_threshold ailleurs dans ce repo : petit helper duplique plutot que
    partage entre packages independants).

PRINCIPE : la graine contient la QUEUE GLISSANTE (raw_axis_tail, NORM_WINDOW_DAYS
jours = 730j = 17520 bougies 1h) des valeurs BRUTES (pre-normalisation) de chaque
axe. A chaque nouvelle bougie 1h, on calcule SES valeurs brutes depuis les buffers
Bitstamp DEJA fetches par live.py (1h pour tendance/momentum, 4h pour DMI), on les
ajoute a la queue (en ejectant la plus ancienne pour rester a NORM_WINDOW_DAYS), on
renormalise en z-score sur la queue mise a jour, et on avance la machine a etats
hysteresis d'UN SEUL bar depuis l'etat courant. Le resultat est un dict a un seul
element {target_ts: regime}, directement consommable par
filters.effective_down_exit_params / effective_up_conf_min_1 / effective_down_delta_1
(meme contrat que le regime_map du backtest -- ces fonctions ne changent pas).

PERSISTANCE : l'etat complet (queue + state machine) est sauvegarde dans
logs/regime_state.json apres CHAQUE bar traite -- un redemarrage reprend exactement
ou le daemon s'est arrete. Si aucun etat local n'existe encore (premier lancement),
on part de regime/regime_seed.json et on COMBLE LE TROU entre sa date de generation
et maintenant en rejouant heure par heure (bootstrap_and_advance ci-dessous).
"""
from __future__ import annotations

import json
import logging
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("live.regime_live")

_LIVE_DIR = Path(__file__).resolve().parent
if str(_LIVE_DIR) not in sys.path:
    sys.path.insert(0, str(_LIVE_DIR))

# AUTONOMIE (2026-07-10) : live/regime/ est une copie embarquee (live/regime/),
# plus le package sibling de 1h/regime/ -- cf calcule_feature.py, meme campagne.
from regime import state_machine  # noqa: E402 -- reutilise TEL QUEL, jamais reimplemente

SEED_PATH = _LIVE_DIR / "regime" / "regime_seed.json"
INTERVAL_1H_MS = 60 * 60 * 1000


# ----------------------------------------------------------------------------
#  Wilder DMI -- duplique volontairement de filters.py/regime/scorers.py (meme
#  convention deja suivie partout dans ce repo pour ce helper).
# ----------------------------------------------------------------------------
def _wilder_dmi(high, low, close, n: int):
    h = np.asarray(high, dtype=float)
    l = np.asarray(low, dtype=float)
    cl = np.asarray(close, dtype=float)
    m = len(h)
    plus = np.full(m, np.nan)
    minus = np.full(m, np.nan)
    if m < 2:
        return plus, minus
    up_move = h[1:] - h[:-1]
    down_move = l[:-1] - l[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0.0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0.0), down_move, 0.0)
    prev_close = cl[:-1]
    tr = np.maximum.reduce([h[1:] - l[1:], np.abs(h[1:] - prev_close), np.abs(l[1:] - prev_close)])

    def _rma(x):
        return pd.Series(x).ewm(alpha=1.0 / n, adjust=False).mean().to_numpy()

    atr = _rma(tr)
    s_plus = _rma(plus_dm)
    s_minus = _rma(minus_dm)
    with np.errstate(divide="ignore", invalid="ignore"):
        plus[1:] = 100.0 * s_plus / atr
        minus[1:] = 100.0 * s_minus / atr
    return plus, minus


# ----------------------------------------------------------------------------
#  Chargement / persistance de l'etat local
# ----------------------------------------------------------------------------
def _seed_to_state(seed: dict) -> dict:
    lookback_bars = int(seed["lookback_days"] * 24)
    momentum_bars = max(1, int(lookback_bars * seed["momentum_fraction"]))
    return {
        "last_ts": int(seed["raw_axis_tail"][-1]["ts"]) if seed["raw_axis_tail"] else None,
        "norm_window_bars": int(seed["norm_window_days"] * 24),
        "lookback_bars": lookback_bars,
        "momentum_bars": momentum_bars,
        "regime_dmi_length": int(seed["regime_dmi_length"]),
        "norm_clip": float(seed["norm_clip"]),
        "weight_trend": float(seed["weight_trend"]),
        "weight_momentum": float(seed["weight_momentum"]),
        "weight_dmi": float(seed["weight_dmi"]),
        "thresholds": dict(seed["thresholds"]),
        "confirm_bars": int(seed["confirm_bars"]),
        "cooldown_bars": int(seed["cooldown_bars"]),
        "state": dict(seed["state"]),
        "raw_axis_tail": list(seed["raw_axis_tail"]),
    }


def load_or_init_state(local_state_path: Path) -> dict:
    """Reprend l'etat local persiste (`logs/regime_state.json`) s'il existe deja
    (redemarrage -- aucun trou possible, cf `advance_one_bar` appele chaque heure).
    Sinon, PREMIER LANCEMENT : copie + charge la graine `regime/regime_seed.json`
    (le trou entre sa date de generation et maintenant est comble separement par
    `bootstrap_and_advance`, PAS ici -- ce chargement seul ne fetch rien)."""
    if local_state_path.exists():
        logger.info(f"Etat regime local trouve ({local_state_path}) -- reprise sans trou.")
        return json.loads(local_state_path.read_text())

    if not SEED_PATH.exists():
        raise FileNotFoundError(
            f"{SEED_PATH} introuvable -- lance `cd 1h/regime && python export_seed.py` "
            f"au moins une fois avant de demarrer live.py."
        )
    seed = json.loads(SEED_PATH.read_text())
    state = _seed_to_state(seed)
    logger.info(f"Aucun etat regime local -- parti de la graine {SEED_PATH} "
                f"(generee {pd.Timestamp(seed['generated_at_ts'], unit='ms', tz='UTC')}, "
                f"queue={len(state['raw_axis_tail'])} bars).")
    return state


def save_state(state: dict, local_state_path: Path) -> None:
    local_state_path.parent.mkdir(parents=True, exist_ok=True)
    local_state_path.write_text(json.dumps(state))


# ----------------------------------------------------------------------------
#  Un seul pas (une bougie 1h) -- coeur du calcul incremental
# ----------------------------------------------------------------------------
def _min_periods(window: int) -> int:
    return max(2, window // 4)   # meme formule que regime/scorers.py::_rolling_norm


def _z_last(values: list[float], window: int, clip: float) -> float:
    """Z-score roulant du DERNIER element de `values` (deja tronque a `window`),
    memes conventions que regime/scorers.py::_rolling_norm (rolling().mean()/std()
    ignorent les NaN ; std(ddof=1, pandas par defaut) ; NaN/std=0 -> 0.0, meme
    fallback que compute_score's fillna(0.0) sur un axe indisponible)."""
    arr = np.asarray(values, dtype=float)
    valid = arr[~np.isnan(arr)]
    if len(valid) < _min_periods(window) or np.isnan(arr[-1]):
        return 0.0
    std = valid.std(ddof=1)
    if std == 0.0:
        return 0.0
    z = (arr[-1] - valid.mean()) / std
    return float(np.clip(z, -clip, clip) / clip)


def compute_raw_for_bar(target_ts: int, close_1h_by_ts: dict[int, float],
                        dmi_plus: np.ndarray, dmi_minus: np.ndarray, dmi_ts: np.ndarray,
                        lookback_bars: int, momentum_bars: int) -> tuple[float, float, float]:
    """trend_raw/momentum_raw/dmi_raw pour la bougie 1h `target_ts`, meme formule
    que regime/scorers.py::build_raw_axes (trend = log(close)-log(close[-lookback]),
    momentum = idem sur momentum_bars, dmi = dernier bar REGIME_DMI_TIMEFRAME deja
    clos <= target_ts, via searchsorted -- jamais un bar en cours ou futur)."""
    close_now = close_1h_by_ts.get(target_ts)
    close_lb = close_1h_by_ts.get(target_ts - lookback_bars * INTERVAL_1H_MS)
    close_mb = close_1h_by_ts.get(target_ts - momentum_bars * INTERVAL_1H_MS)
    trend_raw = (float(np.log(close_now) - np.log(close_lb))
                if (close_now is not None and close_lb is not None) else float("nan"))
    momentum_raw = (float(np.log(close_now) - np.log(close_mb))
                    if (close_now is not None and close_mb is not None) else float("nan"))

    idx = int(np.searchsorted(dmi_ts, target_ts, side="right")) - 1
    dmi_raw = float(dmi_plus[idx] - dmi_minus[idx]) if idx >= 0 else float("nan")
    return trend_raw, momentum_raw, dmi_raw


def advance_one_bar(state: dict, target_ts: int, trend_raw: float, momentum_raw: float,
                    dmi_raw: float) -> str:
    """Ajoute la bougie `target_ts` a la queue (ejecte la plus ancienne au-dela de
    norm_window_bars), renormalise en z-score, avance la machine a etats d'UN bar
    depuis state['state']. Mute `state` en place. Renvoie le regime confirme."""
    tail = state["raw_axis_tail"]
    tail.append({"ts": target_ts, "trend_raw": trend_raw, "momentum_raw": momentum_raw, "dmi_raw": dmi_raw})
    window = state["norm_window_bars"]
    if len(tail) > window:
        del tail[: len(tail) - window]

    clip = state["norm_clip"]
    trend_norm = _z_last([r["trend_raw"] for r in tail], window, clip)
    momentum_norm = _z_last([r["momentum_raw"] for r in tail], window, clip)
    dmi_norm = _z_last([r["dmi_raw"] for r in tail], window, clip)

    w_sum = state["weight_trend"] + state["weight_momentum"] + state["weight_dmi"]
    score = (state["weight_trend"] * trend_norm + state["weight_momentum"] * momentum_norm
             + state["weight_dmi"] * dmi_norm) / w_sum
    score = float(np.clip(score, -1.0, 1.0))

    th = state["thresholds"]
    regimes, final_state = state_machine.apply_state_machine(
        pd.Series([score]), th["bear_enter"], th["bear_exit"], th["bull_enter"], th["bull_exit"],
        confirm_bars=state["confirm_bars"], cooldown_bars=state["cooldown_bars"],
        initial_state=state["state"], return_state=True)
    state["state"] = final_state
    state["last_ts"] = target_ts
    return str(regimes[0])


# ----------------------------------------------------------------------------
#  Bootstrap : comble le trou entre state["last_ts"] et maintenant, puis avance
# ----------------------------------------------------------------------------
def bootstrap_and_advance(state: dict, now_target_ts: int, fetch_1h_fn, fetch_4h_fn) -> str:
    """A appeler CHAQUE cycle horaire (bootstrap au sens "rattrape s'il y a un
    trou", pas juste au premier lancement -- un daemon qui redemarre apres une
    coupure de plusieurs heures/jours passe aussi par ici). `fetch_1h_fn(n_bars)`
    / `fetch_4h_fn(n_bars)` : callables renvoyant les n_bars les plus recentes
    barres [open_ms,o,h,l,c,v] (meme contrat que calcule_feature.fetch_klines).
    Renvoie le regime confirme pour `now_target_ts` (deja ajoute a l'etat)."""
    last_ts = state.get("last_ts")
    if last_ts is not None and last_ts >= now_target_ts:
        # Deja a jour (ex: relance rapprochee) -- rien a rejouer, regime courant valide.
        return str(state["state"]["regime"])

    gap_start_ts = (last_ts + INTERVAL_1H_MS) if last_ts is not None else now_target_ts
    n_missing = int((now_target_ts - gap_start_ts) // INTERVAL_1H_MS) + 1
    if n_missing > 1:
        logger.warning(f"Regime : trou de {n_missing} bougies 1h entre {gap_start_ts} et "
                       f"{now_target_ts} -- comblement en cours (peut prendre quelques secondes).")

    lookback_bars = state["lookback_bars"]
    dmi_len = state["regime_dmi_length"]
    # Marge de warmup DMI 4h : ~8x la longueur suffit largement a la convergence
    # de l'EMA de Wilder (cf note calcul dans le design -- alpha=1/n, residu
    # negligeable en quelques dizaines de bars).
    dmi_warmup_bars = max(100, dmi_len * 10)

    n_1h_needed = lookback_bars + n_missing + 5
    buf_1h = fetch_1h_fn(n_1h_needed)
    close_1h_by_ts = {int(b[0]): float(b[4]) for b in buf_1h}

    n_4h_needed = dmi_warmup_bars + (n_missing // 4 + 2)
    buf_4h = fetch_4h_fn(n_4h_needed)
    dmi_ts = np.array([int(b[0]) for b in buf_4h], dtype=np.int64)
    dmi_high = np.array([float(b[2]) for b in buf_4h], dtype=float)
    dmi_low = np.array([float(b[3]) for b in buf_4h], dtype=float)
    dmi_close = np.array([float(b[4]) for b in buf_4h], dtype=float)
    dmi_plus, dmi_minus = _wilder_dmi(dmi_high, dmi_low, dmi_close, dmi_len)

    regime = str(state["state"]["regime"])
    t = gap_start_ts
    while t <= now_target_ts:
        trend_raw, momentum_raw, dmi_raw = compute_raw_for_bar(
            t, close_1h_by_ts, dmi_plus, dmi_minus, dmi_ts,
            lookback_bars, state["momentum_bars"])
        regime = advance_one_bar(state, t, trend_raw, momentum_raw, dmi_raw)
        t += INTERVAL_1H_MS

    return regime
