# -*- coding: utf-8 -*-
"""exp_dntrail.py — campagne POINTS-FAIBLES (2026-08-11) : trailing DOWN en chop.

Diagnostic : en mode chop, 81 sorties trailing DOWN = -134 pts sur l'annee 1
(gap 0.60 arme des l'entree -> whipsaw immediat). On teste : activation
RETARDEE du trailing (le short doit d'abord prouver un profit) x gap, et un
SL DOWN legerement resserre en chop.

Usage : cd modele_team && python optimize/exp_dntrail.py
"""
from __future__ import annotations
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _harness as H
import main as M
import config_main as C
import filters

# Cache pickle ecrit par optimize/test_720.py (nos propres DataFrames) -> sur.
CACHE720 = H.ROOT / "optimize" / ".cache_prepared_720.pkl"
with open(CACHE720, "rb") as f:
    prepared, regime_map = pickle.load(f)
M._SWING_PIVOTS = None

PROD = dict(up_ratchet=C.UP_RATCHET_STEPS, down_ratchet=C.DOWN_RATCHET_STEPS,
            up_prop=C.UP_RATCHET_PROP, down_prop=C.DOWN_RATCHET_PROP)
Y1_END = prepared["start_ms"] + 365 * C.DAY_MS
TREND = filters.load_trend_context()

_TEAM = None


def team_and_meta():
    global _TEAM
    if _TEAM is None:
        team_bets, up_full = H._team_bets(prepared, regime_map)
        uf = up_full[up_full["take"] == 1]
        up_meta = {int(r["timestamp"]):
                   (int(r["volatility_state"]) if pd.notna(r["volatility_state"]) else -1,
                    int(r["is_weekend"]) if pd.notna(r["is_weekend"]) else -1)
                   for _, r in uf.iterrows()}
        _TEAM = (team_bets, up_meta)
    return _TEAM


def run(activate=None, gap=None, dn_sl=None):
    """Overrides appliques SEULEMENT quand la barre de decision est en chop :
    activate (pct) = seuil d'armement du trailing DOWN ; gap (pct) = trailing
    gap scalp ; dn_sl = (start_pct, floor_pct) du SL temporel DOWN."""
    H.reset_params()
    H.set_params(**PROD)
    tb, up_meta = team_and_meta()
    orig_down = M._down_managed_exit

    def down_wrap(prices_intrabar, traded_ts, max_steps=None, trail_gap=None,
                  widen_candidates=None, regime="neutral", swing_pivots=None,
                  take_profit=None):
        dts = traded_ts - C.INTERVAL_1H_MS
        if not filters.chop_mode(TREND, dts):
            return orig_down(prices_intrabar, traded_ts, max_steps, trail_gap,
                             widen_candidates, regime, swing_pivots, take_profit)
        old = (M.DOWN_TRAIL_ACTIVATE, M.DOWN_SL_START_DEC, M.DOWN_SL_FLOOR_DEC)
        if activate is not None:
            M.DOWN_TRAIL_ACTIVATE = activate / 100.0
        if dn_sl is not None:
            M.DOWN_SL_START_DEC = -(dn_sl[0] / 100.0)
            M.DOWN_SL_FLOOR_DEC = -(dn_sl[1] / 100.0)
        tg = (gap / 100.0) if gap is not None else trail_gap
        try:
            return orig_down(prices_intrabar, traded_ts, max_steps, tg,
                             widen_candidates, regime, swing_pivots, take_profit)
        finally:
            M.DOWN_TRAIL_ACTIVATE, M.DOWN_SL_START_DEC, M.DOWN_SL_FLOOR_DEC = old

    M._down_managed_exit = down_wrap
    try:
        trades, _ = M.simulate_trades(tb, prepared["body_map"], prepared["prices_intrabar"],
                                      up_meta, prepared["extremes"], regime_map, None,
                                      TREND)
    finally:
        M._down_managed_exit = orig_down
    return H.metrics(trades, Y1_END), trades


def rec(tag, m):
    print(f"  {tag:44s} a1={m['h1_net']:+8.2f}%  a2={m['h2_net']:+8.2f}%  net={m['net']:+9.2f}%  "
          f"dd={m['dd']:+6.2f}%  worst={m['worst']:+.2f}%  n={m['n']}", flush=True)
    return dict(tag=tag, a1=m["h1_net"], a2=m["h2_net"], net=m["net"], dd=m["dd"],
                worst=m["worst"], n=m["n"])


rows = []
m0, _ = run()
rows.append(rec("champion (prod)", m0))

print("\n=== activation retardee x gap (chop) ===", flush=True)
for act in (0.1, 0.2, 0.3):
    for g in (0.5, 0.6, 0.7):
        m, _ = run(activate=act, gap=g)
        rows.append(rec(f"act{act}/gap{g}", m))

print("\n=== SL DOWN resserre en chop (activate/gap champions) ===", flush=True)
for sl in ((0.8, 0.15), (0.7, 0.10), (0.9, 0.15)):
    m, _ = run(dn_sl=sl)
    rows.append(rec(f"dnSL[{sl[0]}/{sl[1]}]", m))

pd.DataFrame(rows).to_csv(H.ROOT / "logs" / "exp_dntrail.csv", index=False)
print("\n-> logs/exp_dntrail.csv", flush=True)
