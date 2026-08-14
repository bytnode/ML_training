# -*- coding: utf-8 -*-
"""exp_er2.py — PHASE 2 (raffinement) : scalp conditionne a (chop ET macro-bull),
reglage des parametres de scalp, combinaison avec le TP UP serre en macro-bull.

exp_er.py : scalper les sorties en chop (ER30<0.25) rend l'annee 1 positive
(+15.31%) mais coute a l'annee 2 (1653 -> 855) car ses periodes 'chop' (lag ER
pendant les bear rallies) gagnaient avec les sorties trend. Hypothese : le mode
scalp ne doit s'armer QUE quand chop ET macro-bull (sma200) coincident -- en
macro-bear on trend-ride toujours.

Usage : cd modele_team && python optimize/exp_er2.py
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

con = sqlite3.connect(str(filters._candles_db_path()))
try:
    _c = pd.read_sql_query(
        "SELECT timestamp_ms, close FROM candles WHERE timeframe='1h' ORDER BY timestamp_ms",
        con)
finally:
    con.close()
_c = _c.dropna(subset=["close"]).reset_index(drop=True)
TS = _c["timestamp_ms"].astype("int64").to_numpy()
CL = _c["close"].to_numpy(dtype=float)


def er_map(days: int) -> dict[int, float]:
    n, step = days, 24
    num = np.full(len(CL), np.nan)
    den = np.zeros(len(CL))
    idx = np.arange(step * n, len(CL))
    num[idx] = np.abs(CL[idx] - CL[idx - step * n])
    for k in range(1, n + 1):
        den[idx] += np.abs(CL[idx - step * (k - 1)] - CL[idx - step * k])
    er = np.where(den > 0, num / den, np.nan)
    return {int(t): float(e) for t, e in zip(TS[idx], er[idx]) if not np.isnan(e)}


ER20, ER30 = er_map(20), er_map(30)
sma200 = pd.Series(CL).rolling(200 * 24).mean().to_numpy()
MB200 = {int(t): ("mbull" if c > s else "mbear")
         for t, c, s in zip(TS, CL, sma200) if not np.isnan(s)}
sma120 = pd.Series(CL).rolling(120 * 24).mean().to_numpy()
MB120 = {int(t): ("mbull" if c > s else "mbear")
         for t, c, s in zip(TS, CL, sma120) if not np.isnan(s)}

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


def run(gate=None, up_wrap=None, down_wrap=None):
    H.reset_params()
    H.set_params(**PROD)
    tb, up_meta = team_and_meta()
    if gate is not None:
        tb = tb[gate(tb)].reset_index(drop=True)
    orig_up, orig_down = M._up_managed_exit, M._down_managed_exit
    if up_wrap is not None:
        M._up_managed_exit = up_wrap(orig_up)
    if down_wrap is not None:
        M._down_managed_exit = down_wrap(orig_down)
    try:
        trades, _ = M.simulate_trades(tb, prepared["body_map"], prepared["prices_intrabar"],
                                      up_meta, prepared["extremes"], regime_map, None)
    finally:
        M._up_managed_exit, M._down_managed_exit = orig_up, orig_down
    return H.metrics(trades, Y1_END), trades


def er_of(emap, ts):
    return emap.get(int(ts), 1.0)


def make_wraps(cond, up_tp=0.0060, dn_tp=0.0080, dn_hold=240, dn_gap=0.0060,
               up_tp_mbull=None, mbull_map=None):
    """cond(decision_ts)->True = mode scalp. up_tp_mbull : TP UP serre aussi
    hors chop quand macro-bull (None = off)."""
    def up_wrap(orig):
        def f(prices_intrabar, traded_ts, widen_cands=None):
            dts = traded_ts - C.INTERVAL_1H_MS
            tp = None
            if cond(dts):
                tp = up_tp
            elif up_tp_mbull is not None and mbull_map is not None \
                    and mbull_map.get(int(dts), "mbear") == "mbull":
                tp = up_tp_mbull
            if tp is None:
                return orig(prices_intrabar, traded_ts, widen_cands)
            old = M.UP_TAKE_PROFIT
            M.UP_TAKE_PROFIT = tp
            try:
                return orig(prices_intrabar, traded_ts, widen_cands)
            finally:
                M.UP_TAKE_PROFIT = old
        return f

    def down_wrap(orig):
        def f(prices_intrabar, traded_ts, max_steps=None, trail_gap=None,
              widen_candidates=None, regime="neutral", swing_pivots=None):
            dts = traded_ts - C.INTERVAL_1H_MS
            if not cond(dts):
                return orig(prices_intrabar, traded_ts, max_steps, trail_gap,
                            widen_candidates, regime, swing_pivots)
            old = M.DOWN_TAKE_PROFIT
            M.DOWN_TAKE_PROFIT = dn_tp
            try:
                return orig(prices_intrabar, traded_ts, dn_hold // C.DOWN_CHECK_MIN,
                            dn_gap, widen_candidates, "neutral", swing_pivots)
            finally:
                M.DOWN_TAKE_PROFIT = old
        return f
    return up_wrap, down_wrap


rows = []


def rec(tag, m):
    rows.append(dict(tag=tag, n=m["n"], a1=m["h1_net"], a2=m["h2_net"], net=m["net"],
                     a1_up=m["h1_up_net"], a1_dn=m["h1_down_net"],
                     a2_up=m["h2_up_net"], a2_dn=m["h2_down_net"],
                     dd=m["dd"], worst=m["worst"]))
    print(f"  {tag:52s} a1={m['h1_net']:+8.2f}%  a2={m['h2_net']:+8.2f}%  net={m['net']:+9.2f}%  "
          f"dd={m['dd']:+6.2f}%  n={m['n']}")


mB, _ = run()
rec("baseline", mB)

print("\n=== D1) scalp si (ER<thr ET macro-bull) ===")
for name, mb in (("mb200", MB200), ("mb120", MB120)):
    for dname, emap in (("ER20", ER20), ("ER30", ER30)):
        for thr in (0.20, 0.25, 0.30):
            def cond(dts, _e=emap, _t=thr, _m=mb):
                return er_of(_e, dts) < _t and _m.get(int(dts), "mbear") == "mbull"
            uw, dw = make_wraps(cond)
            m, _ = run(up_wrap=uw, down_wrap=dw)
            rec(f"scalp[{dname}<{thr} & {name}]", m)

print("\n=== D2) scalp si (ER30<0.25) tout court (rappel) vs macro-conditionne + upTPmbull ===")
def cond_er(dts):
    return er_of(ER30, dts) < 0.25
uw, dw = make_wraps(cond_er)
m, _ = run(up_wrap=uw, down_wrap=dw)
rec("scalp[ER30<0.25] (rappel)", m)
for thr in (0.25, 0.30):
    def cond(dts, _t=thr):
        return er_of(ER30, dts) < _t and MB200.get(int(dts), "mbear") == "mbull"
    uw, dw = make_wraps(cond, up_tp_mbull=0.0060, mbull_map=MB200)
    m, _ = run(up_wrap=uw, down_wrap=dw)
    rec(f"scalp[ER30<{thr} & mb200]+upTP0.60[mbull]", m)

print("\n=== D3) reglage des parametres du scalp (autour du meilleur D1/D2) ===")
BEST_THR = 0.30
def cond_best(dts):
    return er_of(ER30, dts) < BEST_THR and MB200.get(int(dts), "mbear") == "mbull"
for up_tp in (0.0050, 0.0060, 0.0080):
    for dn_tp in (0.0060, 0.0080, 0.0100):
        uw, dw = make_wraps(cond_best, up_tp=up_tp, dn_tp=dn_tp)
        m, _ = run(up_wrap=uw, down_wrap=dw)
        rec(f"scalpP[upTP{up_tp*100:.2f}/dnTP{dn_tp*100:.2f}]", m)
for dn_hold in (120, 240, 360):
    for dn_gap in (0.0040, 0.0060, 0.0080):
        uw, dw = make_wraps(cond_best, dn_hold=dn_hold, dn_gap=dn_gap)
        m, _ = run(up_wrap=uw, down_wrap=dw)
        rec(f"scalpP[hold{dn_hold}/gap{dn_gap*100:.2f}]", m)

df = pd.DataFrame(rows)
df.to_csv(H.ROOT / "logs" / "exp_er2.csv", index=False)
print("\n-> logs/exp_er2.csv")
