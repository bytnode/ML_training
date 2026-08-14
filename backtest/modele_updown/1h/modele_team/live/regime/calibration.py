"""calibration.py -- derivation des seuils bear/bull (enter/exit) a partir du score.

Deux methodes comparees empiriquement par compare_regime_variants.py :
  - "manual"     : seuils fixes (config_regime.MANUAL_*)
  - "calibrated" : percentiles de la distribution HISTORIQUE COMPLETE du score
                   (config_regime.CALIB_*_PCTL), meme asymetrie bear/bull.
Les deux renvoient le meme tuple (bear_enter, bear_exit, bull_enter, bull_exit)
consomme par state_machine.apply_state_machine -- la machine a etats est agnostique
de la methode qui a produit les seuils.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from . import config_regime as _cfg
except ImportError:
    import config_regime as _cfg


def manual_thresholds(score: pd.Series) -> tuple[float, float, float, float]:
    """Seuils fixes, independants de la distribution du score."""
    return (_cfg.MANUAL_BEAR_ENTER, _cfg.MANUAL_BEAR_EXIT,
            _cfg.MANUAL_BULL_ENTER, _cfg.MANUAL_BULL_EXIT)


def calibrated_thresholds(score: pd.Series) -> tuple[float, float, float, float]:
    """Seuils = percentiles de la distribution historique COMPLETE du score (un seul
    calibrage sur tout l'historique disponible, coherent avec la fenetre de
    validation 'tout l'historique' actee pour le wrapper -- pas de calibrage roulant)."""
    s = score.dropna().to_numpy()
    bear_enter = float(np.percentile(s, _cfg.CALIB_BEAR_ENTER_PCTL))
    bear_exit = float(np.percentile(s, _cfg.CALIB_BEAR_EXIT_PCTL))
    bull_enter = float(np.percentile(s, _cfg.CALIB_BULL_ENTER_PCTL))
    bull_exit = float(np.percentile(s, _cfg.CALIB_BULL_EXIT_PCTL))
    return bear_enter, bear_exit, bull_enter, bull_exit


METHODS = {
    "manual": manual_thresholds,
    "calibrated": calibrated_thresholds,
}


def get_thresholds(score: pd.Series, method: str) -> tuple[float, float, float, float]:
    if method not in METHODS:
        raise ValueError(f"methode de calibration inconnue: {method!r} (attendu: {list(METHODS)})")
    return METHODS[method](score)
