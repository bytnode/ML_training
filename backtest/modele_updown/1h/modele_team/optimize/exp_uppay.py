# -*- coding: utf-8 -*-
"""exp_uppay.py — campagne PAYOFF-UP (2026-08-12) : ramener AvgLoss UP vers
AvgWin (annee 1 : 3.354 vs 2.568, ratio 1.31 — demande user ~1.0).

Trois familles JAMAIS testees sous le systeme final (les re-tests SL dataient
d'avant le bump de confiance — la population UP chop a change) :
  A) SL UP chop re-serre (start x floor) ;
  B) hold UP raccourci en chop (jamais teste : TP 0.6 pop vite ou jamais ;
     raccourcir borne le SL temporel ET l'exposition) ;
  C) trailing serre en chop (gap 1.0 actuel > TP 0.6 = trailing mort ;
     un gap 0.3-0.4 convertit des pertes en scratches) ;
  D) croisements des meilleurs.

Usage : cd modele_team && python optimize/exp_uppay.py
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
    """Chaine de filtres PRODUCTION complete (bumps DOWN et UP inclus)."""
    global _TEAM
    if _TEAM is None:
        up_full = prepared["up_full"]
        down_full = prepared["down_full"]
        ts_arr = down_full["timestamp"].to_numpy()
        eff_delta = filters.effective_down_delta(ts_arr, regime_map, TREND)
        down_full = down_full.copy()
        down_full["take"] = (down_full["confidence"] >= down_full["threshold"] + eff_delta).astype(int)
        up_bets = M._taken_bets(up_full, "up")
        up_bets = filters.apply_xmodel_filter(up_bets, down_full)
        down_bets = M._taken_bets(down_full, "down")
        team_bets, _ = M.build_team_bets(up_bets, down_bets)
        team_bets, _, _ = filters.apply_dmi_filter(team_bets, prepared.get("dmi_map"))
        team_bets, _ = filters.apply_conf_filter(team_bets, regime_map, TREND)
        uf = up_full[up_full["take"] == 1]
        up_meta = {int(r["timestamp"]):
                   (int(r["volatility_state"]) if pd.notna(r["volatility_state"]) else -1,
                    int(r["is_weekend"]) if pd.notna(r["is_weekend"]) else -1)
                   for _, r in uf.iterrows()}
        _TEAM = (team_bets, up_meta)
    return _TEAM


def run(sl=None, hold=None, trail=None):
    """Overrides UP appliques SEULEMENT en chop : sl=(start,floor) pct,
    hold=minutes, trail=(activate,gap) pct."""
    H.reset_params()
    H.set_params(**PROD)
    tb, up_meta = team_and_meta()
    orig_up = M._up_managed_exit

    def up_wrap(prices_intrabar, traded_ts, widen_cands=None, take_profit=None,
                sl_override=None):
        dts = traded_ts - C.INTERVAL_1H_MS
        if not filters.chop_mode(TREND, dts):
            return orig_up(prices_intrabar, traded_ts, widen_cands, take_profit, sl_override)
        slo = sl_override
        if sl is not None:
            slo = (-(sl[0] / 100.0), -(sl[1] / 100.0))
        old = (M.UP_MAX_STEPS, M.UP_TRAIL_ACTIVATE, M.UP_TRAIL_GAP)
        if hold is not None:
            M.UP_MAX_STEPS = hold // C.UP_CHECK_MIN
        if trail is not None:
            M.UP_TRAIL_ACTIVATE = trail[0] / 100.0
            M.UP_TRAIL_GAP = trail[1] / 100.0
        try:
            return orig_up(prices_intrabar, traded_ts, widen_cands, take_profit, slo)
        finally:
            M.UP_MAX_STEPS, M.UP_TRAIL_ACTIVATE, M.UP_TRAIL_GAP = old

    M._up_managed_exit = up_wrap
    try:
        trades, _ = M.simulate_trades(tb, prepared["body_map"], prepared["prices_intrabar"],
                                      up_meta, prepared["extremes"], regime_map, None,
                                      TREND)
    finally:
        M._up_managed_exit = orig_up
    if C.GOVERNOR_ENABLED and not trades.empty:
        trades, _ = filters.apply_governor(trades)
    return H.metrics(trades, Y1_END), trades


def up_stats(tr, y1=True):
    m = (tr["traded_ts"] < Y1_END) if y1 else (tr["traded_ts"] >= Y1_END)
    s = tr[m & (tr.direction == "up")]
    net = s["net_ret"].to_numpy(float) * C.EXPOSURE
    if not len(net):
        return 0, 0.0, 0.0, 0.0, 0.0
    w, l = net[net > 0], net[net < 0]
    return len(net), (net > 0).mean() * 100, (np.prod(1 + net) - 1) * 100, \
        (w.mean() * 100 if len(w) else 0.0), (-l.mean() * 100 if len(l) else 0.0)


def rec(rows, tag, m, tr):
    n1, wr1, net1, aw1, al1 = up_stats(tr, True)
    _, _, net2, _, _ = up_stats(tr, False)
    ratio = al1 / aw1 if aw1 else float("nan")
    print(f"  {tag:32s} a1={m['h1_net']:+8.2f}%  a1_up: net={net1:+7.2f}% wr={wr1:5.2f}% "
          f"aw={aw1:.2f} al={al1:.2f} r={ratio:.2f} | a2_up={net2:+7.2f}%  net={m['net']:+9.2f}%  dd={m['dd']:+6.2f}%", flush=True)
    rows.append(dict(tag=tag, a1=m["h1_net"], a1_up_net=net1, a1_up_wr=wr1, aw=aw1, al=al1,
                     ratio=ratio, a2_up=net2, a2=m["h2_net"], net=m["net"], dd=m["dd"]))


rows = []
m0, tr0 = run()
rec(rows, "production", m0, tr0)

print("\n=== A) SL UP chop re-serre (sous le bump 0.035) ===", flush=True)
for start in (0.7, 0.8, 0.9, 1.0):
    for floor in (0.3, 0.4, 0.5):
        m, tr = run(sl=(start, floor))
        rec(rows, f"SL[{start}/{floor}]", m, tr)

print("\n=== B) hold UP raccourci en chop ===", flush=True)
for h in (120, 150, 180):
    m, tr = run(hold=h)
    rec(rows, f"hold[{h}]", m, tr)

print("\n=== C) trailing serre en chop ===", flush=True)
for act, gap in ((0.2, 0.3), (0.2, 0.4), (0.3, 0.4), (0.3, 0.3)):
    m, tr = run(trail=(act, gap))
    rec(rows, f"trail[{act}/{gap}]", m, tr)

pd.DataFrame(rows).to_csv(H.ROOT / "logs" / "exp_uppay.csv", index=False)
print("\n-> logs/exp_uppay.csv", flush=True)
