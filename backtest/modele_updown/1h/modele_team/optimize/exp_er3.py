# -*- coding: utf-8 -*-
"""exp_er3.py — PHASE 2 (raffinement 2) : separer 'macro-bear jeune' (corrections
en V de l'annee 1 -> scalper) du 'macro-bear profond/installe' (annee 2 -> laisser
le trend-ride). Axes : age du bear (heures sous la SMA200) et profondeur
(close/sma200 - 1). Le scalp s'arme si chop ET PAS bear-profond.

Usage : cd modele_team && python optimize/exp_er3.py
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


ER30 = er_map(30)
SMA = pd.Series(CL).rolling(200 * 24).mean().to_numpy()
DEPTH = np.where(np.isnan(SMA), np.nan, CL / SMA - 1.0)     # <0 = sous la SMA200
AGE_H = np.zeros(len(CL))                                    # heures consecutives sous la SMA
for i in range(1, len(CL)):
    if not np.isnan(SMA[i]) and CL[i] < SMA[i]:
        AGE_H[i] = AGE_H[i - 1] + 1
DEPTH_MAP = {int(t): float(d) for t, d in zip(TS, DEPTH) if not np.isnan(d)}
AGE_MAP = {int(t): float(a) for t, a in zip(TS, AGE_H)}
MB200 = {int(t): ("mbull" if d > 0 else "mbear") for t, d in DEPTH_MAP.items()}

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


def er_of(ts):
    return ER30.get(int(ts), 1.0)


def deep_age(ts, hours):
    return AGE_MAP.get(int(ts), 0.0) > hours


def deep_depth(ts, frac):
    return DEPTH_MAP.get(int(ts), 0.0) < -frac


# scalp retenu de exp_er2 D3 : upTP 0.60 / dnTP 0.80 / hold 360 / gap 0.60
def make_wraps(cond, up_scalp=True, dn_scalp=True, up_tp=0.0060, dn_tp=0.0080,
               dn_hold=360, dn_gap=0.0060, up_tp_mbull=None):
    def up_wrap(orig):
        def f(prices_intrabar, traded_ts, widen_cands=None):
            dts = traded_ts - C.INTERVAL_1H_MS
            tp = None
            if up_scalp and cond(dts):
                tp = up_tp
            elif up_tp_mbull is not None and MB200.get(int(dts), "mbear") == "mbull":
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
            if not (dn_scalp and cond(dts)):
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
    print(f"  {tag:56s} a1={m['h1_net']:+8.2f}%  a2={m['h2_net']:+8.2f}%  net={m['net']:+9.2f}%  "
          f"dd={m['dd']:+6.2f}%  n={m['n']}")


mB, _ = run()
rec("baseline", mB)

print("\n=== E1) scalp si chop ET PAS bear-profond ===")
CONDS = {
    "uncond": lambda dts: er_of(dts) < 0.25,
    "!age30d": lambda dts: er_of(dts) < 0.25 and not deep_age(dts, 30 * 24),
    "!age60d": lambda dts: er_of(dts) < 0.25 and not deep_age(dts, 60 * 24),
    "!age90d": lambda dts: er_of(dts) < 0.25 and not deep_age(dts, 90 * 24),
    "!dep5":  lambda dts: er_of(dts) < 0.25 and not deep_depth(dts, 0.05),
    "!dep10": lambda dts: er_of(dts) < 0.25 and not deep_depth(dts, 0.10),
    "!dep15": lambda dts: er_of(dts) < 0.25 and not deep_depth(dts, 0.15),
}
for name, cond in CONDS.items():
    uw, dw = make_wraps(cond)
    m, _ = run(up_wrap=uw, down_wrap=dw)
    rec(f"scalp[ER30<0.25 & {name}]", m)

print("\n=== E2) meilleurs conds : isolation par sens + overlay upTP0.60[mbull] ===")
for name in ("!age60d", "!dep10", "uncond"):
    cond = CONDS[name]
    uw, dw = make_wraps(cond, up_scalp=True, dn_scalp=False)
    m, _ = run(up_wrap=uw, down_wrap=dw)
    rec(f"scalpUPonly[{name}]", m)
    uw, dw = make_wraps(cond, up_scalp=False, dn_scalp=True)
    m, _ = run(up_wrap=uw, down_wrap=dw)
    rec(f"scalpDNonly[{name}]", m)
    uw, dw = make_wraps(cond, up_tp_mbull=0.0060)
    m, _ = run(up_wrap=uw, down_wrap=dw)
    rec(f"scalp[{name}]+upTP0.60[mbull]", m)

df = pd.DataFrame(rows)
df.to_csv(H.ROOT / "logs" / "exp_er3.csv", index=False)
print("\n-> logs/exp_er3.csv")
