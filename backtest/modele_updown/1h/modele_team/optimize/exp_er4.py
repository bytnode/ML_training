# -*- coding: utf-8 -*-
"""exp_er4.py — PHASE 2 (validation/reglage du candidat scalp[chop & !bear-age]).

Candidat : sorties scalp (upTP 0.60 / dnTP 0.80 / hold 360 / gap 0.60) quand
ER30<0.25 ET age_bear<=30j (heures consecutives sous SMA200). a1 +23.16% /
a2 +955.41%. Ici : position exacte de la frontiere d'age (marge), plateau du
seuil ER et de l'age, re-reglage des params de scalp sous LA bonne condition,
overlays (upTP mbull, skip DOWN en chop mort), et profil mensuel/DD du gagnant.

Usage : cd modele_team && python optimize/exp_er4.py
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
AGE_H = np.zeros(len(CL))
for i in range(1, len(CL)):
    if not np.isnan(SMA[i]) and CL[i] < SMA[i]:
        AGE_H[i] = AGE_H[i - 1] + 1
AGE_MAP = {int(t): float(a) for t, a in zip(TS, AGE_H)}
MB200 = {int(t): ("mbull" if c > s else "mbear")
         for t, c, s in zip(TS, CL, SMA) if not np.isnan(s)}

# --- A) position de la frontiere d'age ---
bars = prepared["up_full"]["timestamp"].astype("int64")
for tag, lo, hi in (("a1", prepared["start_ms"], Y1_END), ("a2", Y1_END, prepared["end_ms"])):
    sel = bars[(bars >= lo) & (bars < hi)]
    ages = np.array([AGE_MAP.get(int(t), 0.0) for t in sel]) / 24.0
    print(f"{tag}: age bear max = {ages.max():.1f}j ; p95 = {np.percentile(ages,95):.1f}j ; "
          f"% barres age>20j = {(ages>20).mean()*100:.1f}%  >30j = {(ages>30).mean()*100:.1f}%  "
          f">45j = {(ages>45).mean()*100:.1f}%")

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


def make_cond(er_thr, age_days):
    def cond(dts):
        return (ER30.get(int(dts), 1.0) < er_thr
                and AGE_MAP.get(int(dts), 0.0) <= age_days * 24)
    return cond


def make_wraps(cond, up_tp=0.0060, dn_tp=0.0080, dn_hold=360, dn_gap=0.0060,
               up_tp_mbull=None):
    def up_wrap(orig):
        def f(prices_intrabar, traded_ts, widen_cands=None):
            dts = traded_ts - C.INTERVAL_1H_MS
            tp = None
            if cond(dts):
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


print("\n=== B) plateau seuil ER x frontiere d'age ===")
for er_thr in (0.20, 0.25, 0.30):
    for age_d in (15, 20, 30, 45, 60):
        uw, dw = make_wraps(make_cond(er_thr, age_d))
        m, _ = run(up_wrap=uw, down_wrap=dw)
        rec(f"scalp[ER<{er_thr}, age<={age_d}j]", m)

print("\n=== C) re-reglage params de scalp sous [ER<0.25, age<=30j] ===")
COND = make_cond(0.25, 30)
for dn_tp in (0.0070, 0.0080, 0.0090, 0.0100):
    uw, dw = make_wraps(COND, dn_tp=dn_tp)
    m, _ = run(up_wrap=uw, down_wrap=dw)
    rec(f"C[dnTP{dn_tp*100:.2f}]", m)
for dn_hold in (240, 360, 480, 600):
    uw, dw = make_wraps(COND, dn_hold=dn_hold)
    m, _ = run(up_wrap=uw, down_wrap=dw)
    rec(f"C[hold{dn_hold}]", m)
for dn_gap in (0.0050, 0.0060, 0.0070, 0.0080):
    uw, dw = make_wraps(COND, dn_gap=dn_gap)
    m, _ = run(up_wrap=uw, down_wrap=dw)
    rec(f"C[gap{dn_gap*100:.2f}]", m)
for up_tp in (0.0050, 0.0060, 0.0070, 0.0080):
    uw, dw = make_wraps(COND, up_tp=up_tp)
    m, _ = run(up_wrap=uw, down_wrap=dw)
    rec(f"C[upTP{up_tp*100:.2f}]", m)

print("\n=== D) overlays ===")
uw, dw = make_wraps(COND, up_tp_mbull=0.0060)
m, _ = run(up_wrap=uw, down_wrap=dw)
rec("D[+upTP0.60 mbull]", m)


def gate_deadchop(tb):
    mask = np.ones(len(tb), dtype=bool)
    for i, b in enumerate(tb.itertuples(index=False)):
        if b.direction == "down" and ER30.get(int(b.timestamp), 1.0) < 0.10 \
                and AGE_MAP.get(int(b.timestamp), 0.0) <= 30 * 24:
            mask[i] = False
    return mask


uw, dw = make_wraps(COND)
m, _ = run(gate=gate_deadchop, up_wrap=uw, down_wrap=dw)
rec("D[+skipDN ER<0.10 jeune]", m)

print("\n=== E) profil mensuel du gagnant [ER<0.25, age<=30j] ===")
uw, dw = make_wraps(COND)
mW, trW = run(up_wrap=uw, down_wrap=dw)
trW = trW.copy()
trW["month"] = pd.to_datetime(trW["traded_ts"], unit="ms").dt.strftime("%Y-%m")
trW["net_pts"] = trW["net_ret"] * C.EXPOSURE * 100
mo = trW.groupby("month").agg(n=("net_pts", "size"), tot=("net_pts", "sum"),
                              wr=("win_net", "mean"))
mo["wr"] = (mo["wr"] * 100).round(1)
print(mo.round(1).to_string())

df = pd.DataFrame(rows)
df.to_csv(H.ROOT / "logs" / "exp_er4.csv", index=False)
print("\n-> logs/exp_er4.csv")
