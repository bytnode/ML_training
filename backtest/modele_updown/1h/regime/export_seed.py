"""export_seed.py -- genere regime_seed.json pour live_team (campagne symetrie
2026-07-09).

live_team est self-contained par design (pas de dependance runtime a la SQLite
complete de modele_down, cf CLAUDE.md racine) mais le detecteur de regime a besoin
de NORM_WINDOW_DAYS (730j) d'historique pour normaliser chaque axe en z-score
roulant -- bien plus que ce que live_team bufferise pour les predictions (~133j).
Ce script exporte donc une GRAINE (artefact derive, gitignore, comme les .json de
modeles) contenant :
  - la queue glissante (raw_axis_tail) des valeurs BRUTES (pre-normalisation)
    trend_raw/momentum_raw/dmi_raw sur les NORM_WINDOW_DAYS derniers jours,
  - l'etat terminal de la machine a etats hysteresis (regime confirme + compteurs),
  - les seuils calibres courants (percentiles de la distribution historique --
    figes a l'export ; bougent tres peu sur quelques mois, pas critique).

live_team/regime_live.py charge cette graine au boot puis avance la machine a
etats d'UN bar a la fois en live, en calculant les nouvelles valeurs brutes depuis
ses PROPRES buffers deja fetches (1h pour trend/momentum, 4h pour dmi) -- aucun
acces a la SQLite de modele_down requis en live.

A relancer et recopier vers live_team/regime_seed.json :
  - si config_regime.py change (poids, lookback, calibration, fenetre de norm...),
  - periodiquement (ex: tous les 3-6 mois) pour rafraichir les seuils calibres.

Usage:
    cd backtest/modele_updown/1h/regime
    python export_seed.py
    # puis (PowerShell) copier regime_seed.json vers live_team/regime_seed.json
"""
from __future__ import annotations

import json
from pathlib import Path

try:
    from . import config_regime as cfg
    from . import scorers
    from . import calibration
    from . import state_machine
except ImportError:
    import config_regime as cfg
    import scorers
    import calibration
    import state_machine

OUT_PATH = Path(__file__).resolve().parent / "regime_seed.json"


def build_seed() -> dict:
    """Calcule la graine sur tout l'historique 1h disponible (meme calcul que
    api.evaluate_regime_full, mais on a besoin des colonnes BRUTES en plus)."""
    axes = scorers.compute_score(cfg.LOOKBACK_DAYS_DEFAULT)
    bear_enter, bear_exit, bull_enter, bull_exit = calibration.get_thresholds(
        axes["score"], cfg.CALIBRATION_METHOD)
    _regimes, final_state = state_machine.apply_state_machine(
        axes["score"], bear_enter, bear_exit, bull_enter, bull_exit,
        return_state=True)

    window_bars = cfg.NORM_WINDOW_DAYS * 24
    tail = axes.tail(window_bars).copy()
    raw_axis_tail = [
        {"ts": int(r.timestamp_ms), "trend_raw": float(r.trend_raw),
         "momentum_raw": float(r.momentum_raw), "dmi_raw": float(r.dmi_raw)}
        for r in tail.itertuples(index=False)
    ]

    return {
        "generated_at_ts": int(axes["timestamp_ms"].max()),
        "lookback_days": cfg.LOOKBACK_DAYS_DEFAULT,
        "momentum_fraction": cfg.MOMENTUM_FRACTION,
        "norm_window_days": cfg.NORM_WINDOW_DAYS,
        "norm_clip": cfg.NORM_CLIP,
        "regime_dmi_timeframe": cfg.REGIME_DMI_TIMEFRAME,
        "regime_dmi_length": cfg.REGIME_DMI_LENGTH,
        "weight_trend": cfg.WEIGHT_TREND,
        "weight_momentum": cfg.WEIGHT_MOMENTUM,
        "weight_dmi": cfg.WEIGHT_DMI,
        "calibration_method": cfg.CALIBRATION_METHOD,
        "thresholds": {"bear_enter": bear_enter, "bear_exit": bear_exit,
                       "bull_enter": bull_enter, "bull_exit": bull_exit},
        "confirm_bars": cfg.CONFIRM_BARS,
        "cooldown_bars": cfg.COOLDOWN_BARS,
        "state": final_state,
        "raw_axis_tail": raw_axis_tail,
    }


if __name__ == "__main__":
    seed = build_seed()
    OUT_PATH.write_text(json.dumps(seed), encoding="utf-8")
    print(f"regime_seed.json ecrit : {len(seed['raw_axis_tail'])} bars dans la queue, "
         f"regime terminal={seed['state']['regime']}, "
         f"generated_at={seed['generated_at_ts']}")
    print("-> copier vers live_team/regime_seed.json (PowerShell : "
         "copy regime_seed.json ..\\live_team\\regime_seed.json)")
