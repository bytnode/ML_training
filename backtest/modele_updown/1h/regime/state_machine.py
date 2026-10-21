"""state_machine.py -- hysteresis + confirmation + cooldown -> regime confirme.

Generique : prend des seuils deja derives (peu importe leur origine, manuel ou
calibre) et produit un regime discret par bar, avec les 3 mecanismes anti-whipsaw
demandes :
  - hysteresis   : bandes enter/exit distinctes (bear a une bande plus large que
                   bull -> se declenche plus tot ET persiste plus longtemps)
  - confirmation : N bars consecutifs du signal brut avant qu'un changement soit accepte
  - cooldown     : M bars apres un changement avant qu'un nouveau soit accepte
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from . import config_regime as _cfg
except ImportError:
    import config_regime as _cfg


def apply_state_machine(score: pd.Series,
                        bear_enter: float, bear_exit: float,
                        bull_enter: float, bull_exit: float,
                        confirm_bars: int = None,
                        cooldown_bars: int = None,
                        initial_state: dict | None = None,
                        return_state: bool = False):
    """Renvoie un array de regimes confirmes ('bear'/'neutral'/'bull'), un par bar
    de `score` (meme longueur/ordre -- suppose deja trie par timestamp croissant).
    NaN (warmup) -> signal ambigu -> reste dans le regime courant (par defaut
    'neutral' au tout debut) : comportement par defaut sur signal ambigu = rester
    sur la strategie neutre, comme demande.

    initial_state (campagne symetrie 2026-07-09, cf regime/export_seed.py) : reprend
    la marche depuis un etat deja avance ({"regime","candidate","candidate_run",
    "bars_since_change"}) au lieu de repartir de "neutral" -- utilise pour exporter
    l'etat terminal d'un run complet (graine pour live_team/regime_live.py), qui
    avance ensuite la machine d'UN bar a la fois depuis ce point. None (defaut) =
    comportement inchange (part de "neutral").
    return_state=True renvoie (regimes, final_state_dict) au lieu du seul array."""
    if confirm_bars is None:
        confirm_bars = _cfg.CONFIRM_BARS
    if cooldown_bars is None:
        cooldown_bars = _cfg.COOLDOWN_BARS

    s = score.to_numpy(dtype=float)
    n = len(s)
    regimes = np.empty(n, dtype=object)

    if initial_state is None:
        state = "neutral"
        candidate = None
        candidate_run = 0
        bars_since_change = 10 ** 9   # tres grand -> pas de cooldown actif au depart
    else:
        state = initial_state.get("regime", "neutral")
        candidate = initial_state.get("candidate")
        candidate_run = int(initial_state.get("candidate_run", 0))
        bars_since_change = int(initial_state.get("bars_since_change", 10 ** 9))

    for i in range(n):
        val = s[i]
        if np.isnan(val):
            raw = state          # signal ambigu -> ne bouge pas, reste dans le regime courant
        elif state == "bear":
            raw = "neutral" if val > bear_exit else "bear"
        elif state == "bull":
            raw = "neutral" if val < bull_exit else "bull"
        else:  # state == "neutral"
            if val <= bear_enter:
                raw = "bear"
            elif val >= bull_enter:
                raw = "bull"
            else:
                raw = "neutral"

        if raw == state:
            candidate = None
            candidate_run = 0
        else:
            if candidate == raw:
                candidate_run += 1
            else:
                candidate = raw
                candidate_run = 1
            if candidate_run >= confirm_bars and bars_since_change >= cooldown_bars:
                state = raw
                candidate = None
                candidate_run = 0
                bars_since_change = -1   # remis a 0 par le += 1 juste en dessous

        regimes[i] = state
        bars_since_change += 1

    if return_state:
        final_state = {"regime": state, "candidate": candidate,
                       "candidate_run": candidate_run, "bars_since_change": bars_since_change}
        return regimes, final_state
    return regimes


def count_flips(regimes: np.ndarray) -> int:
    """Nombre de changements de regime confirmes (proxy whipsaw pour le harnais)."""
    if len(regimes) < 2:
        return 0
    return int(np.sum(regimes[1:] != regimes[:-1]))
