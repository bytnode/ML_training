# -*- coding: utf-8 -*-
"""exp_er5.py — PHASE 2 (finalisation) : combinaisons fines autour du champion
scalp[ER30<0.25, age<=30j], et robustesse par fenetres 365j glissantes.

exp_er4 : hold long paie (600 > 480 > 360) -> tester 720 (defaut) / 840 ; gap 0.5
booste a1 mais coute a2 ; dnTP 0.9 legerement mieux. Ici : grille croisee compacte,
overlay upTP0.60[mbull], puis pour le champion : trimestres de l'annee 1 et MIN
des fenetres 365j glissantes (cible finale = positif sur n'importe quelle periode).

Usage : cd modele_team && python optimize/exp_er5.py
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

n, step = 30, 24
num = np.full(len(CL), np.nan)
den = np.zeros(len(CL))
idx = np.arange(step * n, len(CL))
num[idx] = np.abs(CL[idx] - CL[idx - step * n])
for k in range(1, n + 1):
    den[idx] += np.abs(CL[idx - step * (k - 1)] - CL[idx - step * k])
er = np.where(den > 0, num / den, np.nan)
ER30 = {int(t): float(e) for t, e in zip(TS[idx], er[idx]) if not np.isnan(e)}

SMA = pd.Series(CL).rolling(200 * 24).mean().to_numpy()
AGE_H = np.zeros(len(CL))
for i in range(1, len(CL)):
    if not np.isnan(SMA[i]) and CL[i] < SMA[i]:
        AGE_H[i] = AGE_H[i - 1] + 1
AGE_MAP = {int(t): float(a) for t, a in zip(TS, AGE_H)}
MB200 = {int(t): ("mbull" if c > s else "mbear")
         for t, c, s in zip(TS, CL, SMA) if not np.isnan(s)}


def cond_chop(dts):
    return ER30.get(int(dts), 1.0) < 0.25 and AGE_MAP.get(int(dts), 0.0) <= 30 * 24


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


def make_wraps(up_tp=0.0060, dn_tp=0.0080, dn_hold=360, dn_gap=0.0060,
               up_tp_mbull=None):
    def up_wrap(orig):
        def f(prices_intrabar, traded_ts, widen_cands=None):
            dts = traded_ts - C.INTERVAL_1H_MS
            tp = None
            if cond_chop(dts):
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
            if not cond_chop(dts):
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
                     dd=m["dd"], worst=m["worst"]))
    print(f"  {tag:52s} a1={m['h1_net']:+8.2f}%  a2={m['h2_net']:+8.2f}%  net={m['net']:+9.2f}%  "
          f"dd={m['dd']:+6.2f}%  n={m['n']}")


print("=== F) grille croisee hold x gap x dnTP (upTP 0.60) ===")
for dn_hold in (480, 600, 720, 840):
    for dn_gap in (0.0050, 0.0060):
        for dn_tp in (0.0080, 0.0090):
            uw, dw = make_wraps(dn_tp=dn_tp, dn_hold=dn_hold, dn_gap=dn_gap)
            m, _ = run(up_wrap=uw, down_wrap=dw)
            rec(f"F[h{dn_hold}/g{dn_gap*100:.1f}/tp{dn_tp*100:.1f}]", m)

print("\n=== G) upTP 0.50 et overlay mbull sur les 2 meilleurs F ===")
for dn_hold, dn_gap, dn_tp in ((600, 0.0060, 0.0080), (720, 0.0060, 0.0080),
                               (600, 0.0060, 0.0090), (720, 0.0060, 0.0090)):
    for up_tp in (0.0050, 0.0060):
        uw, dw = make_wraps(up_tp=up_tp, dn_tp=dn_tp, dn_hold=dn_hold, dn_gap=dn_gap)
        m, _ = run(up_wrap=uw, down_wrap=dw)
        rec(f"G[h{dn_hold}/tp{dn_tp*100:.1f}/upTP{up_tp*100:.1f}]", m)
        uw, dw = make_wraps(up_tp=up_tp, dn_tp=dn_tp, dn_hold=dn_hold, dn_gap=dn_gap,
                            up_tp_mbull=0.0060)
        m, _ = run(up_wrap=uw, down_wrap=dw)
        rec(f"G[h{dn_hold}/tp{dn_tp*100:.1f}/upTP{up_tp*100:.1f}]+mbull", m)

df = pd.DataFrame(rows)
df.to_csv(H.ROOT / "logs" / "exp_er5.csv", index=False)

# ----------------------------------------------------------------------------
#  H) champion provisoire : trimestres a1 + fenetres 365j glissantes
# ----------------------------------------------------------------------------
best = df.iloc[1:].sort_values(["a1"], ascending=False)
print("\ntop 5 par a1 :")
print(best.head(5).round(2).to_string(index=False))

# champion = meilleur compromis a1/a2 (choisi apres lecture) : h600/g0.6/tp0.8/upTP0.6
uw, dw = make_wraps(dn_tp=0.0080, dn_hold=600, dn_gap=0.0060)
mW, trW = run(up_wrap=uw, down_wrap=dw)
print(f"\nchampion h600/g0.6/dnTP0.8/upTP0.6 : a1={mW['h1_net']:+.2f}% a2={mW['h2_net']:+.2f}% "
      f"net={mW['net']:+.2f}% dd={mW['dd']:+.2f}%")
trW = trW.copy()
trW["q"] = pd.to_datetime(trW["traded_ts"], unit="ms").dt.to_period("Q").astype(str)
net = trW["net_ret"].to_numpy(float) * C.EXPOSURE
print("\ntrimestres (net compose %) :")
for q, g in trW.groupby("q"):
    s = g["net_ret"].to_numpy(float) * C.EXPOSURE
    print(f"  {q}: {(np.prod(1+s)-1)*100:+7.2f}%  (n={len(s)})")

eq = np.cumprod(1.0 + net)
ts_arr = trW["traded_ts"].to_numpy("int64")
print("\nfenetres 365j glissantes (pas 30j) — net % :")
worst_win = (None, 1e9)
t0 = int(prepared["start_ms"])
while t0 + 365 * C.DAY_MS <= prepared["end_ms"]:
    t1 = t0 + 365 * C.DAY_MS
    msk = (ts_arr >= t0) & (ts_arr < t1)
    v = (np.prod(1 + net[msk]) - 1) * 100 if msk.any() else 0.0
    tag = pd.Timestamp(t0, unit='ms').strftime('%Y-%m-%d')
    if v < worst_win[1]:
        worst_win = (tag, v)
    print(f"  {tag}: {v:+8.2f}%  (n={int(msk.sum())})")
    t0 += 30 * C.DAY_MS
print(f"\nPIRE fenetre 365j : {worst_win[0]} -> {worst_win[1]:+.2f}%")
