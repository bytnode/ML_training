# -*- coding: utf-8 -*-
"""exp_er.py — PHASE 2 (suite) : conditionnement a la TRENDINESS (Kaufman ER).

exp_macro.py a montre que la direction macro ne separe pas proprement (les
corrections en V de l'annee 1 sont label 'mbear' comme les vraies tendances de
l'annee 2). Hypothese affinee : ce qui separe les 2 annees est l'EFFICIENCE du
chemin (trend vs chop), pas sa direction. ER = |close_t - close_{t-n}| / somme
des |pas quotidiens| -- calcule en pas QUOTIDIENS, rafraichi a chaque barre 1h,
sans lookahead (closes <= barre de decision uniquement).

Usage : cd modele_team && python optimize/exp_er.py
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
    """ER de Kaufman en pas quotidiens (24 barres 1h), rafraichi chaque heure."""
    n = days
    step = 24
    num = np.full(len(CL), np.nan)
    den = np.zeros(len(CL))
    lo = step * n
    idx = np.arange(lo, len(CL))
    num[idx] = np.abs(CL[idx] - CL[idx - step * n])
    for k in range(1, n + 1):
        den[idx] += np.abs(CL[idx - step * (k - 1)] - CL[idx - step * k])
    er = np.where(den > 0, num / den, np.nan)
    return {int(t): float(e) for t, e in zip(TS[idx], er[idx]) if not np.isnan(e)}


ERS = {d: er_map(d) for d in (10, 20, 30, 60)}

# aussi : direction macro sma200 (pour la combinaison avec upTP0.60)
sma200 = pd.Series(CL).rolling(200 * 24).mean().to_numpy()
MB200 = {int(t): ("mbull" if c > s else "mbear")
         for t, c, s in zip(TS, CL, sma200) if not np.isnan(s)}

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


def er_of(emap, ts, default=1.0):
    return emap.get(int(ts), default)      # absent (warmup) -> traite comme trending


def mac_of(ts):
    return MB200.get(int(ts), "mbear")


# ----------------------------------------------------------------------------
#  A) distribution + separation
# ----------------------------------------------------------------------------
mB, trB = run()
print(f"baseline : n={mB['n']}  a1={mB['h1_net']:.2f}%  a2={mB['h2_net']:.2f}%  net={mB['net']:.2f}%")

trB = trB.copy()
trB["year"] = np.where(trB["traded_ts"] < Y1_END, "a1", "a2")
trB["net_pts"] = trB["net_ret"] * C.EXPOSURE * 100

bars = prepared["up_full"]["timestamp"].astype("int64")
print("\n=== distribution ER par annee (barres 1h de la fenetre) ===")
for d, emap in ERS.items():
    for tag, lo, hi in (("a1", prepared["start_ms"], Y1_END), ("a2", Y1_END, prepared["end_ms"])):
        sel = bars[(bars >= lo) & (bars < hi)]
        v = np.array([er_of(emap, t, np.nan) for t in sel])
        v = v[~np.isnan(v)]
        q = np.percentile(v, [25, 50, 75])
        print(f"  ER{d}j {tag}: p25={q[0]:.3f} p50={q[1]:.3f} p75={q[2]:.3f}")

print("\n=== separation trades : annee x bucket ER x sens (net pts / n) ===")
for d, emap in ERS.items():
    trB["er"] = [er_of(emap, t, np.nan) for t in trB["timestamp"]]
    trB["bkt"] = pd.cut(trB["er"], [0, 0.15, 0.30, 1.01], labels=["chop", "mid", "trend"])
    piv = trB.pivot_table(index=["year", "bkt"], columns="direction",
                          values="net_pts", aggfunc=["sum", "count"], observed=False)
    print(f"\n--- ER{d}j ---")
    print(piv.round(1).to_string())

# ----------------------------------------------------------------------------
#  B) experiences
# ----------------------------------------------------------------------------
rows = []


def rec(tag, m):
    rows.append(dict(tag=tag, n=m["n"], a1=m["h1_net"], a2=m["h2_net"], net=m["net"],
                     a1_up=m["h1_up_net"], a1_dn=m["h1_down_net"],
                     a2_up=m["h2_up_net"], a2_dn=m["h2_down_net"],
                     dd=m["dd"], worst=m["worst"]))
    print(f"  {tag:46s} a1={m['h1_net']:+8.2f}%  a2={m['h2_net']:+8.2f}%  net={m['net']:+9.2f}%  n={m['n']}")


def gate_er(emap, thr, sides=("down", "up")):
    def g(tb):
        mask = np.ones(len(tb), dtype=bool)
        for i, b in enumerate(tb.itertuples(index=False)):
            if b.direction in sides and er_of(emap, b.timestamp) < thr:
                mask[i] = False
        return mask
    return g


def up_tp_mbull(tp_dec):
    def wrap(orig):
        def f(prices_intrabar, traded_ts, widen_cands=None):
            if mac_of(traded_ts - C.INTERVAL_1H_MS) != "mbull":
                return orig(prices_intrabar, traded_ts, widen_cands)
            old = M.UP_TAKE_PROFIT
            M.UP_TAKE_PROFIT = tp_dec
            try:
                return orig(prices_intrabar, traded_ts, widen_cands)
            finally:
                M.UP_TAKE_PROFIT = old
        return f
    return wrap


def scalp_chop(emap, thr, up_tp=0.0060, dn_tp=0.0080, dn_hold=240, dn_gap=0.0060):
    """En chop (ER < thr) : TP serres, hold court, pas de tpext."""
    def up_wrap(orig):
        def f(prices_intrabar, traded_ts, widen_cands=None):
            if er_of(emap, traded_ts - C.INTERVAL_1H_MS) >= thr:
                return orig(prices_intrabar, traded_ts, widen_cands)
            old = M.UP_TAKE_PROFIT
            M.UP_TAKE_PROFIT = up_tp
            try:
                return orig(prices_intrabar, traded_ts, widen_cands)
            finally:
                M.UP_TAKE_PROFIT = old
        return f

    def down_wrap(orig):
        def f(prices_intrabar, traded_ts, max_steps=None, trail_gap=None,
              widen_candidates=None, regime="neutral", swing_pivots=None):
            if er_of(emap, traded_ts - C.INTERVAL_1H_MS) >= thr:
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


rec("baseline", mB)
print("\n=== B1) gate ER : skip DOWN si ER < thr ===")
for d in (20, 30, 60):
    for thr in (0.15, 0.20, 0.25):
        m, _ = run(gate=gate_er(ERS[d], thr, sides=("down",)))
        rec(f"skipDN[ER{d}<{thr}]", m)

print("\n=== B2) gate ER : skip LES DEUX sens si ER < thr ===")
for d in (20, 30, 60):
    for thr in (0.15, 0.20, 0.25):
        m, _ = run(gate=gate_er(ERS[d], thr))
        rec(f"skipALL[ER{d}<{thr}]", m)

print("\n=== B3) scalp en chop (pas de gate, sorties adaptees) ===")
for d in (20, 30, 60):
    for thr in (0.20, 0.25):
        uw, dw = scalp_chop(ERS[d], thr)
        m, _ = run(up_wrap=uw, down_wrap=dw)
        rec(f"scalpChop[ER{d}<{thr}]", m)

print("\n=== B4) combos avec upTP0.60[mbull sma200] ===")
m, _ = run(up_wrap=up_tp_mbull(0.0060))
rec("upTP0.60[sma200] (rappel)", m)
for d in (20, 30, 60):
    for thr in (0.15, 0.20, 0.25):
        m, _ = run(gate=gate_er(ERS[d], thr), up_wrap=up_tp_mbull(0.0060))
        rec(f"skipALL[ER{d}<{thr}]+upTP0.60", m)

df = pd.DataFrame(rows)
df.to_csv(H.ROOT / "logs" / "exp_er.csv", index=False)
print("\n-> logs/exp_er.csv")
