# -*- coding: utf-8 -*-
"""exp_dnsma.py — test demande user (2026-08-14) : activer le contexte
SMA200/ER (mode chop) AUSSI sur le cote DOWN, par-dessus l'etat actuel
(DOWN prod : widen x4 + regime bear, sans gouverneur, sans cliquets/tpext).

CHOP_DOWN_ENABLED=True reactive 3 choses en chop scalpable (ER<0.25 &
bear<=30j) : sorties scalp DOWN (TP 1.10 / hold 600 / gap 0.70 / act 0.40),
bump d'entree +0.01, et palier de cliquet (0.50 -> 0.10) hors bear installe.
A/B sur 720j, entrees reconstruites par variante (le bump change les takes).

Usage : cd modele_team && python optimize/exp_dnsma.py
"""
from __future__ import annotations
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main as M
import config_main as C
import filters

ROOT = Path(__file__).resolve().parent.parent
# Cache pickle ecrit par optimize/test_720.py (nos propres DataFrames) -> sur.
with open(ROOT / "optimize" / ".cache_prepared_720.pkl", "rb") as f:
    prepared, _ = pickle.load(f)
M._SWING_PIVOTS = None
REGIME_MAP = filters.load_regime_map()
TREND = filters.load_trend_context()
Y1_END = prepared["start_ms"] + 365 * C.DAY_MS


def build_and_run(chop_down: bool):
    M.CHOP_DOWN_ENABLED = chop_down
    filters.CHOP_DOWN_ENABLED = chop_down
    down_full = prepared["down_full"].copy()
    eff = filters.effective_down_delta(down_full["timestamp"].to_numpy(), REGIME_MAP, TREND)
    down_full["take"] = (down_full["confidence"] >= down_full["threshold"] + eff).astype(int)
    up_bets = filters.apply_xmodel_filter(M._taken_bets(prepared["up_full"], "up"), down_full)
    team, _ = M.build_team_bets(up_bets, M._taken_bets(down_full, "down"))
    team, _, _ = filters.apply_dmi_filter(team, prepared.get("dmi_map"))
    team, _ = filters.apply_conf_filter(team, REGIME_MAP, TREND)
    trades, info = M.simulate_trades(team, prepared["body_map"], prepared["prices_intrabar"],
                                     None, prepared["extremes"], REGIME_MAP, None, TREND)
    return trades, info


def side_stats(tr, direction, y1):
    m = (tr.traded_ts < Y1_END) if y1 else (tr.traded_ts >= Y1_END)
    s = tr[m & (tr.direction == direction)]
    net = s["net_ret"].to_numpy(float) * C.EXPOSURE
    if not len(net):
        return "n=0"
    return (f"n={len(net):3d} WR={(net > 0).mean()*100:5.2f}% "
            f"net={(np.prod(1+net)-1)*100:+8.2f}%")


def show(tag, tr):
    net = tr.sort_values("traded_ts")["net_ret"].to_numpy(float) * C.EXPOSURE
    eq = np.cumprod(1 + net)
    dd = float((eq / np.maximum.accumulate(eq) - 1).min()) * 100
    total = (eq[-1] - 1) * 100
    m1 = tr.traded_ts < Y1_END
    a1 = (np.prod(1 + tr[m1].sort_values("traded_ts")["net_ret"].to_numpy(float) * C.EXPOSURE) - 1) * 100
    a2 = (np.prod(1 + tr[~m1].sort_values("traded_ts")["net_ret"].to_numpy(float) * C.EXPOSURE) - 1) * 100
    print(f"\n=== {tag} ===")
    print(f"  TOTAL net={total:+9.2f}%  dd={dd:+6.2f}%  n={len(tr)}  |  annee1={a1:+8.2f}%  annee2={a2:+9.2f}%")
    for y1, lab in ((True, "annee 1"), (False, "annee 2")):
        print(f"  {lab}  UP   : {side_stats(tr, 'up', y1)}")
        print(f"  {lab}  DOWN : {side_stats(tr, 'down', y1)}")


for chop_down, tag in ((False, "A. etat actuel (SMA cote UP seulement)"),
                       (True,  "B. SMA aussi cote DOWN (CHOP_DOWN_ENABLED=True)")):
    tr, info = build_and_run(chop_down)
    show(tag, tr)
M.CHOP_DOWN_ENABLED = filters.CHOP_DOWN_ENABLED = C.CHOP_DOWN_ENABLED
