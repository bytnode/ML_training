# -*- coding: utf-8 -*-
"""exp_upsl.py — campagne POINTS-FAIBLES (2026-08-11, apres TOUT-TERRAIN) :
SL UP conditionnel au chop/macro-bull.

Diagnostic : les 14 pires trades de l'annee 1 sont TOUS des stop_loss UP
(-5.9 a -8.7 pts, -138 pts au total). En mode chop le TP UP est 0.60% mais le
SL reste START 1.5% -> risque 2.5x le gain vise. On teste un SL UP resserre
quand la barre de decision est en chop (et variante chop OU macro-bull).

Usage : cd modele_team && python optimize/exp_upsl.py
"""
from __future__ import annotations
import pickle
import sqlite3
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


def run(up_sl=None, scope="chop"):
    """up_sl = (start_pct, floor_pct) applique quand la barre est dans `scope`
    ('chop' | 'chop_or_mbull'). None = champion tel quel (production)."""
    H.reset_params()
    H.set_params(**PROD)
    tb, up_meta = team_and_meta()
    orig_up = M._up_managed_exit

    def up_wrap(prices_intrabar, traded_ts, widen_cands=None, take_profit=None):
        dts = traded_ts - C.INTERVAL_1H_MS
        e = TREND.get(int(dts))
        in_chop = filters.chop_mode(TREND, dts)
        in_scope = in_chop or (scope == "chop_or_mbull" and e is not None and e[2])
        if up_sl is None or not in_scope:
            return orig_up(prices_intrabar, traded_ts, widen_cands, take_profit)
        old = (M.UP_SL_START_DEC, M.UP_SL_FLOOR_DEC)
        M.UP_SL_START_DEC = -(up_sl[0] / 100.0)
        M.UP_SL_FLOOR_DEC = -(up_sl[1] / 100.0)
        try:
            return orig_up(prices_intrabar, traded_ts, widen_cands, take_profit)
        finally:
            M.UP_SL_START_DEC, M.UP_SL_FLOOR_DEC = old

    M._up_managed_exit = up_wrap
    try:
        trades, _ = M.simulate_trades(tb, prepared["body_map"], prepared["prices_intrabar"],
                                      up_meta, prepared["extremes"], regime_map, None,
                                      TREND)
    finally:
        M._up_managed_exit = orig_up
    return H.metrics(trades, Y1_END), trades


def rec(tag, m):
    print(f"  {tag:44s} a1={m['h1_net']:+8.2f}%  a2={m['h2_net']:+8.2f}%  net={m['net']:+9.2f}%  "
          f"dd={m['dd']:+6.2f}%  worst={m['worst']:+.2f}%  n={m['n']}", flush=True)
    return dict(tag=tag, a1=m["h1_net"], a2=m["h2_net"], net=m["net"], dd=m["dd"],
                worst=m["worst"], n=m["n"])


rows = []
m0, tr0 = run()
rows.append(rec("champion (prod)", m0))

# anatomie : les SL UP du champion, chop ou pas ?
tr0 = tr0.copy()
up_sl_tr = tr0[(tr0.direction == "up") & (tr0.exit_reason == "stop_loss")]
flags = []
for t in up_sl_tr["timestamp"]:
    e = TREND.get(int(t))
    flags.append(("chop" if filters.chop_mode(TREND, int(t))
                  else ("mbull" if (e is not None and e[2]) else "autre")))
print("\nSL UP du champion par contexte :", pd.Series(flags).value_counts().to_dict(), flush=True)

print("\n=== SL UP resserre en CHOP seulement ===", flush=True)
for start, floor in ((1.2, 0.6), (1.0, 0.5), (0.8, 0.4), (0.8, 0.6), (0.6, 0.3), (1.0, 0.7)):
    m, _ = run(up_sl=(start, floor), scope="chop")
    rows.append(rec(f"chopSL[{start}/{floor}]", m))

print("\n=== SL UP resserre en CHOP OU MACRO-BULL ===", flush=True)
for start, floor in ((1.2, 0.6), (1.0, 0.5), (0.8, 0.4), (1.0, 0.7)):
    m, _ = run(up_sl=(start, floor), scope="chop_or_mbull")
    rows.append(rec(f"bothSL[{start}/{floor}]", m))

pd.DataFrame(rows).to_csv(H.ROOT / "logs" / "exp_upsl.csv", index=False)
print("\n-> logs/exp_upsl.csv", flush=True)
