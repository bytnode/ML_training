# -*- coding: utf-8 -*-
"""exp_macro.py — PHASE 2 du goal "rentable sur tout type de marche" (2026-08-11).

Diagnostic (diag_year1.py) : le regime LOCAL 30j s'inverse entre les 2 annees
(bear-local DOWN perd en annee 1, gagne en annee 2). La variable qui separe les
deux mondes est la TENDANCE MACRO (plusieurs mois). Ce script :
  A) construit 4 definitions de macro-tendance sans lookahead (SMA/momentum sur
     les closes 1h de la table candles, decision au close de la barre de decision) ;
  B) verifie leur pouvoir de separation sur les trades baseline ;
  C) teste des strategies conditionnees au macro-regime :
     gate DOWN en macro-bull, TP UP serre en macro-bull, DOWN scalp en macro-bull.

Usage : cd modele_team && python optimize/exp_macro.py
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

# ----------------------------------------------------------------------------
#  A) Macro-tendance : closes 1h absolus -> map {decision_ts: 'mbull'|'mbear'}
#     Sans lookahead : la valeur au ts de decision n'utilise que les closes <= ts
#     (la barre de decision vient de cloturer quand on decide, meme convention
#     que le filtre DMI).
# ----------------------------------------------------------------------------
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


def macro_sma(days: int) -> dict[int, str]:
    n = days * 24
    sma = pd.Series(CL).rolling(n).mean().to_numpy()
    return {int(t): ("mbull" if c > s else "mbear")
            for t, c, s in zip(TS, CL, sma) if not np.isnan(s)}


def macro_mom(days: int) -> dict[int, str]:
    n = days * 24
    out = {}
    for i in range(n, len(TS)):
        out[int(TS[i])] = "mbull" if CL[i] >= CL[i - n] else "mbear"
    return out


MACROS = {
    "sma120": macro_sma(120),
    "sma200": macro_sma(200),
    "mom90": macro_mom(90),
    "mom120": macro_mom(120),
}

# ----------------------------------------------------------------------------
#  Runner : chaine de filtres du harnais + gate d'entree + wrappers de sortie
# ----------------------------------------------------------------------------
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
    """gate(team_bets)->mask a garder ; *_wrap(orig)->fonction remplacante."""
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


def mac_of(mmap, ts):
    return mmap.get(int(ts), "mbear")     # absent (warmup) -> comportement actuel


# --- gates d'entree (post-coherence, comme DMI/conf) ---
def gate_down_mbull(mmap, keep_local_bull=False):
    def g(tb):
        mask = np.ones(len(tb), dtype=bool)
        for i, b in enumerate(tb.itertuples(index=False)):
            if b.direction == "down" and mac_of(mmap, b.timestamp) == "mbull":
                if keep_local_bull and regime_map and regime_map.get(int(b.timestamp)) == "bull":
                    continue                      # short de sommet local : on garde
                mask[i] = False
        return mask
    return g


# --- wrappers de sortie ---
def up_tp_mbull(mmap, tp_dec):
    def wrap(orig):
        def f(prices_intrabar, traded_ts, widen_cands=None):
            if mac_of(mmap, traded_ts - C.INTERVAL_1H_MS) != "mbull":
                return orig(prices_intrabar, traded_ts, widen_cands)
            old = M.UP_TAKE_PROFIT
            M.UP_TAKE_PROFIT = tp_dec
            try:
                return orig(prices_intrabar, traded_ts, widen_cands)
            finally:
                M.UP_TAKE_PROFIT = old
        return f
    return wrap


def down_scalp_mbull(mmap, tp_dec, hold_min, gap_dec):
    """En macro-bull, le short devient un scalp mean-reversion : TP court, hold
    court, gap serre, pas d'extension tpext (regime passe 'neutral')."""
    def wrap(orig):
        def f(prices_intrabar, traded_ts, max_steps=None, trail_gap=None,
              widen_candidates=None, regime="neutral", swing_pivots=None):
            if mac_of(mmap, traded_ts - C.INTERVAL_1H_MS) != "mbull":
                return orig(prices_intrabar, traded_ts, max_steps, trail_gap,
                            widen_candidates, regime, swing_pivots)
            old = M.DOWN_TAKE_PROFIT
            M.DOWN_TAKE_PROFIT = tp_dec
            try:
                return orig(prices_intrabar, traded_ts, hold_min // C.DOWN_CHECK_MIN,
                            gap_dec, widen_candidates, "neutral", swing_pivots)
            finally:
                M.DOWN_TAKE_PROFIT = old
        return f
    return wrap


# ----------------------------------------------------------------------------
#  B) Pouvoir de separation des 4 defs (annotation des trades baseline)
# ----------------------------------------------------------------------------
mB, trB = run()
print(f"baseline : n={mB['n']}  net={mB['net']:.2f}%  a1={mB['h1_net']:.2f}%  a2={mB['h2_net']:.2f}%")

trB = trB.copy()
trB["year"] = np.where(trB["traded_ts"] < Y1_END, "a1", "a2")
trB["net_pts"] = trB["net_ret"] * C.EXPOSURE * 100
print("\n=== separation par macro-def (total net pts par annee x macro x sens) ===")
for name, mmap in MACROS.items():
    trB["mac"] = [mac_of(mmap, t) for t in trB["timestamp"]]
    piv = trB.pivot_table(index=["year", "mac"], columns="direction",
                          values="net_pts", aggfunc=["sum", "count"])
    print(f"\n--- {name} ---")
    print(piv.round(1).to_string())
    # part de barres mbull par annee
    bars = prepared["up_full"]["timestamp"].astype("int64")
    for tag, lo, hi in (("a1", prepared["start_ms"], Y1_END), ("a2", Y1_END, prepared["end_ms"])):
        sel = bars[(bars >= lo) & (bars < hi)]
        pb = np.mean([mac_of(mmap, t) == "mbull" for t in sel]) * 100
        print(f"    {tag}: {pb:.0f}% des barres en mbull")

# ----------------------------------------------------------------------------
#  C) Experiences
# ----------------------------------------------------------------------------
rows = []


def rec(tag, m):
    rows.append(dict(tag=tag, n=m["n"], a1=m["h1_net"], a2=m["h2_net"], net=m["net"],
                     a1_up=m["h1_up_net"], a1_dn=m["h1_down_net"],
                     a2_up=m["h2_up_net"], a2_dn=m["h2_down_net"],
                     dd=m["dd"], worst=m["worst"]))
    print(f"  {tag:42s} a1={m['h1_net']:+8.2f}%  a2={m['h2_net']:+8.2f}%  net={m['net']:+9.2f}%  n={m['n']}")


rec("baseline", mB)
print("\n=== C1) gate DOWN en macro-bull ===")
for name, mmap in MACROS.items():
    m, _ = run(gate=gate_down_mbull(mmap))
    rec(f"gateDN[{name}]", m)
    m, _ = run(gate=gate_down_mbull(mmap, keep_local_bull=True))
    rec(f"gateDN[{name}]+keepLocalBull", m)

print("\n=== C2) TP UP serre (0.60%) en macro-bull ===")
for name, mmap in MACROS.items():
    m, _ = run(up_wrap=up_tp_mbull(mmap, 0.0060))
    rec(f"upTP0.60[{name}]", m)

print("\n=== C3) DOWN scalp en macro-bull (TP 0.8 / hold 240 / gap 0.6) ===")
for name, mmap in MACROS.items():
    m, _ = run(down_wrap=down_scalp_mbull(mmap, 0.0080, 240, 0.0060))
    rec(f"dnScalp[{name}]", m)

df = pd.DataFrame(rows)
df.to_csv(H.ROOT / "logs" / "exp_macro.csv", index=False)
print("\n-> logs/exp_macro.csv")
