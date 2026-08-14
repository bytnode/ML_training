# -*- coding: utf-8 -*-
"""exp_upwr.py — campagne UP-PERTES (2026-08-11, 5e goal du jour) : reduire
l'AvgLoss UP de l'annee 1 (3.895% vs AvgWin 2.455% — payoff 0.63).

Deux leviers testes (miroir du succes WR-DOWN) :
  A) palier de cliquet UP supplementaire hors bear installe (arm -> lock 0.10,
     net-positif ; le palier actuel 0.80->0.05 est NET-NEGATIF apres frais) ;
  B) extension du SL serre (1.2/0.5, actuellement chop-only) au macro-bull
     (le pire trade -7.55 = SL 1.5 pct non-chop en mbull) ;
  C) croisement des meilleurs.

Usage : cd modele_team && python optimize/exp_upwr.py
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
        team_bets, _ = filters.apply_conf_filter(team_bets, regime_map)
        uf = up_full[up_full["take"] == 1]
        up_meta = {int(r["timestamp"]):
                   (int(r["volatility_state"]) if pd.notna(r["volatility_state"]) else -1,
                    int(r["is_weekend"]) if pd.notna(r["is_weekend"]) else -1)
                   for _, r in uf.iterrows()}
        _TEAM = (team_bets, up_meta)
    return _TEAM


def run(up_arm=None, up_lock=None, mbull_sl=None):
    """up_arm/up_lock (pct) : palier UP supplementaire hors bear installe.
    mbull_sl = (start,floor) pct : SL UP en macro-bull NON-chop (chop garde 1.2/0.5)."""
    H.reset_params()
    H.set_params(**PROD)
    tb, up_meta = team_and_meta()
    orig_up = M._up_managed_exit

    def up_wrap(prices_intrabar, traded_ts, widen_cands=None, take_profit=None,
                sl_override=None):
        dts = traded_ts - C.INTERVAL_1H_MS
        e = TREND.get(int(dts))
        in_chop = filters.chop_mode(TREND, dts)
        in_mbull = e is not None and e[2] and not in_chop
        slo = sl_override
        if mbull_sl is not None and in_mbull:
            slo = (-(mbull_sl[0] / 100.0), -(mbull_sl[1] / 100.0))
        if up_arm is not None and (in_chop or (e is not None and e[2])):
            old = M.UP_RATCHET
            M.UP_RATCHET = tuple(sorted(old + ((up_arm / 100.0, up_lock / 100.0),)))
            try:
                return orig_up(prices_intrabar, traded_ts, widen_cands, take_profit, slo)
            finally:
                M.UP_RATCHET = old
        return orig_up(prices_intrabar, traded_ts, widen_cands, take_profit, slo)

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
    n2, wr2, net2, aw2, al2 = up_stats(tr, False)
    print(f"  {tag:28s} a1={m['h1_net']:+7.2f}%  a1_up: wr={wr1:5.2f}% net={net1:+7.2f}% aw={aw1:.2f} al={al1:.2f} "
          f"| a2_up net={net2:+7.2f}%  net={m['net']:+9.2f}%  dd={m['dd']:+6.2f}%", flush=True)
    rows.append(dict(tag=tag, a1=m["h1_net"], a1_up_wr=wr1, a1_up_net=net1, a1_up_aw=aw1,
                     a1_up_al=al1, a2_up_net=net2, a2=m["h2_net"], net=m["net"], dd=m["dd"]))


rows = []
m0, tr0 = run()
rec(rows, "production", m0, tr0)

# anatomie des pertes UP a1 actuelles
tr0 = tr0.copy()
up1 = tr0[(tr0["traded_ts"] < Y1_END) & (tr0.direction == "up")].copy()
up1["pts"] = up1["net_ret"] * 500


def ctx(t):
    t = int(t)
    if filters.chop_mode(TREND, t):
        return "chop"
    e = TREND.get(t)
    return "mbull" if (e is not None and e[2]) else "mbear"


up1["ctx"] = [ctx(t) for t in up1["timestamp"]]
print("\nanatomie UP annee 1 (ctx x exit_reason) :", flush=True)
print(up1.groupby(["ctx", "exit_reason"]).agg(n=("pts", "size"), tot=("pts", "sum"),
                                              avg=("pts", "mean")).round(2).to_string(), flush=True)

print("\n=== A) palier UP hors bear installe ===", flush=True)
for arm in (0.3, 0.4, 0.5, 0.6):
    for lock in (0.10, 0.15):
        m, tr = run(up_arm=arm, up_lock=lock)
        rec(rows, f"upStep[{arm}->{lock}]", m, tr)

print("\n=== B) SL serre etendu au macro-bull non-chop ===", flush=True)
for sl in ((1.2, 0.5), (1.2, 0.6), (1.0, 0.5)):
    m, tr = run(mbull_sl=sl)
    rec(rows, f"mbullSL[{sl[0]}/{sl[1]}]", m, tr)

pd.DataFrame(rows).to_csv(H.ROOT / "logs" / "exp_upwr.csv", index=False)
print("\n-> logs/exp_upwr.csv", flush=True)
