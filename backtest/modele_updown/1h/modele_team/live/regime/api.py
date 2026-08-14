"""api.py -- contrat standardise du package `regime` (meme convention que
modele_up/modele_down/api.py) : modele_team ne consomme JAMAIS scorers/calibration/
state_machine directement, seulement les fonctions ci-dessous.
"""
from __future__ import annotations

import pandas as pd

try:
    from . import config_regime as _cfg
    from . import scorers
    from . import calibration
    from . import state_machine
except ImportError:
    import config_regime as _cfg
    import scorers
    import calibration
    import state_machine

NAME = "regime"


def evaluate_regime_full(lookback_days: int = None,
                          calibration_method: str = None) -> pd.DataFrame:
    """Regime confirme sur TOUT l'historique 1h disponible (~13 ans) -- la fenetre
    de validation du wrapper est independante de modele_team's TEAM_TEST_DAYS (365j),
    actee pour couvrir plusieurs vrais cycles bear/neutral/bull.

    Renvoie DataFrame[timestamp, trend_norm, momentum_norm, dmi_norm, score, regime].
    `timestamp` = open-time ms de la bougie 1h (meme grille que modele_team/main.py).
    """
    if lookback_days is None:
        lookback_days = _cfg.LOOKBACK_DAYS_DEFAULT
    if calibration_method is None:
        calibration_method = _cfg.CALIBRATION_METHOD

    axes = scorers.compute_score(lookback_days)
    bear_enter, bear_exit, bull_enter, bull_exit = calibration.get_thresholds(
        axes["score"], calibration_method)
    regimes = state_machine.apply_state_machine(
        axes["score"], bear_enter, bear_exit, bull_enter, bull_exit)
    out = axes.rename(columns={"timestamp_ms": "timestamp"}).copy()
    out["regime"] = regimes
    out.attrs["thresholds"] = {
        "bear_enter": bear_enter, "bear_exit": bear_exit,
        "bull_enter": bull_enter, "bull_exit": bull_exit,
    }
    out.attrs["n_flips"] = state_machine.count_flips(regimes)
    return out


def evaluate_regime_on_slice(start_ms: int, end_ms: int,
                              lookback_days: int = None,
                              calibration_method: str = None) -> pd.DataFrame:
    """Meme chose que evaluate_regime_full, restreint a [start_ms, end_ms] -- pour
    un usage futur (ex: live_team) qui n'a besoin que d'une fenetre recente. Le
    score/etat est calcule sur tout l'historique puis tronque (jamais l'inverse :
    le lookback/hysteresis a besoin du passe complet pour etre correct au debut de
    la fenetre demandee)."""
    full = evaluate_regime_full(lookback_days, calibration_method)
    mask = (full["timestamp"] >= start_ms) & (full["timestamp"] <= end_ms)
    return full[mask].reset_index(drop=True)
