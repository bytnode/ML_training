# -*- coding: utf-8 -*-
"""exp_combo.py — campagne POINTS-FAIBLES (2026-08-11) : grilles raffinees
SL UP chop x trailing DOWN chop, puis croisement des meilleurs.

Sorties des sweeps paralleles : SL UP chop optimal vers (1.2/0.6) ; trailing
DOWN chop optimal au COIN (act 0.3 / gap 0.7) -> grilles etendues ici, puis
croisement top2 x top2. Le meilleur combo est sauve AVEC exit_ts pour le
gouverneur exact (exp_governor2.py).

Usage : cd modele_team && python optimize/exp_combo.py
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


def run(up_sl=None, dn_act=None, dn_gap=None):
    """Overrides SEULEMENT quand la barre de decision est en chop."""
    H.reset_params()
    H.set_params(**PROD)
    tb, up_meta = team_and_meta()
    orig_up, orig_down = M._up_managed_exit, M._down_managed_exit

    def up_wrap(prices_intrabar, traded_ts, widen_cands=None, take_profit=None):
        dts = traded_ts - C.INTERVAL_1H_MS
        if up_sl is None or not filters.chop_mode(TREND, dts):
            return orig_up(prices_intrabar, traded_ts, widen_cands, take_profit)
        old = (M.UP_SL_START_DEC, M.UP_SL_FLOOR_DEC)
        M.UP_SL_START_DEC = -(up_sl[0] / 100.0)
        M.UP_SL_FLOOR_DEC = -(up_sl[1] / 100.0)
        try:
            return orig_up(prices_intrabar, traded_ts, widen_cands, take_profit)
        finally:
            M.UP_SL_START_DEC, M.UP_SL_FLOOR_DEC = old

    def down_wrap(prices_intrabar, traded_ts, max_steps=None, trail_gap=None,
                  widen_candidates=None, regime="neutral", swing_pivots=None,
                  take_profit=None):
        dts = traded_ts - C.INTERVAL_1H_MS
        if (dn_act is None and dn_gap is None) or not filters.chop_mode(TREND, dts):
            return orig_down(prices_intrabar, traded_ts, max_steps, trail_gap,
                             widen_candidates, regime, swing_pivots, take_profit)
        old_act = M.DOWN_TRAIL_ACTIVATE
        if dn_act is not None:
            M.DOWN_TRAIL_ACTIVATE = dn_act / 100.0
        tg = (dn_gap / 100.0) if dn_gap is not None else trail_gap
        try:
            return orig_down(prices_intrabar, traded_ts, max_steps, tg,
                             widen_candidates, regime, swing_pivots, take_profit)
        finally:
            M.DOWN_TRAIL_ACTIVATE = old_act

    M._up_managed_exit = up_wrap
    M._down_managed_exit = down_wrap
    try:
        trades, _ = M.simulate_trades(tb, prepared["body_map"], prepared["prices_intrabar"],
                                      up_meta, prepared["extremes"], regime_map, None,
                                      TREND)
    finally:
        M._up_managed_exit, M._down_managed_exit = orig_up, orig_down
    return H.metrics(trades, Y1_END), trades


def rec(rows, tag, m):
    print(f"  {tag:40s} a1={m['h1_net']:+8.2f}%  a2={m['h2_net']:+8.2f}%  net={m['net']:+9.2f}%  "
          f"dd={m['dd']:+6.2f}%  n={m['n']}", flush=True)
    rows.append(dict(tag=tag, a1=m["h1_net"], a2=m["h2_net"], net=m["net"], dd=m["dd"], n=m["n"]))


rows = []
m0, _ = run()
rec(rows, "champion (prod)", m0)

print("\n=== A) SL UP chop raffine ===", flush=True)
up_grid = []
for start in (1.1, 1.2, 1.3):
    for floor in (0.5, 0.6, 0.7):
        m, _ = run(up_sl=(start, floor))
        rec(rows, f"upSL[{start}/{floor}]", m)
        up_grid.append(((start, floor), m["net"], m["h1_net"]))

print("\n=== B) trailing DOWN chop raffine ===", flush=True)
dn_grid = []
for act in (0.2, 0.3, 0.4, 0.5):
    for gap in (0.7, 0.8, 0.9):
        m, _ = run(dn_act=act, dn_gap=gap)
        rec(rows, f"dnTr[act{act}/gap{gap}]", m)
        dn_grid.append(((act, gap), m["net"], m["h1_net"]))

# top2 de chaque axe par net (a1 doit rester > champion)
up_top = [g[0] for g in sorted(up_grid, key=lambda g: -g[1]) if g[2] > m0["h1_net"]][:2]
dn_top = [g[0] for g in sorted(dn_grid, key=lambda g: -g[1]) if g[2] > m0["h1_net"]][:2]
print(f"\ntop UP: {up_top}  top DN: {dn_top}", flush=True)

print("\n=== C) croisement top2 x top2 ===", flush=True)
best = (None, -1e9, None)
for u in up_top:
    for d in dn_top:
        m, tr = run(up_sl=u, dn_act=d[0], dn_gap=d[1])
        rec(rows, f"combo[up{u} x dn{d}]", m)
        if m["net"] > best[1]:
            best = ((u, d), m["net"], tr)

pd.DataFrame(rows).to_csv(H.ROOT / "logs" / "exp_combo.csv", index=False)
if best[2] is not None:
    best[2].to_csv(H.ROOT / "logs" / "exp_combo_best_trades.csv", index=False)
    print(f"\nMEILLEUR COMBO : up_sl={best[0][0]}  dn(act,gap)={best[0][1]}  net={best[1]:+.2f}%", flush=True)
    print("-> logs/exp_combo_best_trades.csv (avec exit_ts, pour exp_governor2)", flush=True)
print("-> logs/exp_combo.csv", flush=True)
