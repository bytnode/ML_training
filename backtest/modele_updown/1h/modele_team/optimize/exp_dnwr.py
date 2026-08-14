# -*- coding: utf-8 -*-
"""exp_dnwr.py — campagne WR-DOWN (2026-08-11, 4e goal du jour) : remonter le
WR DOWN de l'annee 1 (42.7%) SANS sacrifier les gros gains (AvgWin 5.37%).

Levier : palier de cliquet supplementaire ACTIF EN CHOP SEULEMENT (variante :
chop+mbull) — un short qui a atteint arm% ne peut plus finir sous lock%
(lock >= 0.10 = net-positif apres frais). Le TP 1.10 et le trailing act0.4/gap0.7
restent intacts. La campagne cliquet du matin a prouve que verrouiller tot est
destructeur EN TENDANCE ; jamais teste en chop, ou les retours en arriere sont
la norme. Le palier de securite production (1.10 -> 0.25) est conserve.

Usage : cd modele_team && python optimize/exp_dnwr.py
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
    """Chaine de filtres PRODUCTION (avec le bump de delta chop)."""
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


def run(arm=None, lock=None, scope="chop"):
    """arm/lock (pct) : palier plat supplementaire du cliquet DOWN, applique en
    chop (scope='chop') ou en chop OU macro-bull (scope='both'). None = prod."""
    H.reset_params()
    H.set_params(**PROD)
    tb, up_meta = team_and_meta()
    orig_down = M._down_managed_exit

    def down_wrap(prices_intrabar, traded_ts, max_steps=None, trail_gap=None,
                  widen_candidates=None, regime="neutral", swing_pivots=None,
                  take_profit=None, trail_activate=None):
        dts = traded_ts - C.INTERVAL_1H_MS
        e = TREND.get(int(dts))
        in_chop = filters.chop_mode(TREND, dts)
        in_scope = in_chop or (scope == "both" and e is not None and e[2])
        if arm is None or not in_scope:
            return orig_down(prices_intrabar, traded_ts, max_steps, trail_gap,
                             widen_candidates, regime, swing_pivots,
                             take_profit, trail_activate)
        old = M.DOWN_RATCHET
        M.DOWN_RATCHET = tuple(sorted(old + ((arm / 100.0, lock / 100.0),)))
        try:
            return orig_down(prices_intrabar, traded_ts, max_steps, trail_gap,
                             widen_candidates, regime, swing_pivots,
                             take_profit, trail_activate)
        finally:
            M.DOWN_RATCHET = old

    M._down_managed_exit = down_wrap
    try:
        trades, _ = M.simulate_trades(tb, prepared["body_map"], prepared["prices_intrabar"],
                                      up_meta, prepared["extremes"], regime_map, None,
                                      TREND)
    finally:
        M._down_managed_exit = orig_down
    if C.GOVERNOR_ENABLED and not trades.empty:
        trades, _ = filters.apply_governor(trades)
    return H.metrics(trades, Y1_END), trades


def side_stats(tr, y1=True):
    m = (tr["traded_ts"] < Y1_END) if y1 else (tr["traded_ts"] >= Y1_END)
    s = tr[m & (tr.direction == "down")]
    net = s["net_ret"].to_numpy(float) * C.EXPOSURE
    if not len(net):
        return 0, 0.0, 0.0, 0.0
    w = net[net > 0]
    return len(net), (net > 0).mean() * 100, (np.prod(1 + net) - 1) * 100, \
        (w.mean() * 100 if len(w) else 0.0)


def rec(rows, tag, m, tr):
    n1, wr1, net1, aw1 = side_stats(tr, True)
    n2, wr2, net2, aw2 = side_stats(tr, False)
    print(f"  {tag:26s} a1={m['h1_net']:+7.2f}%  a1_dn: wr={wr1:5.2f}% net={net1:+7.2f}% aw={aw1:.2f} "
          f"| a2_dn: wr={wr2:5.2f}% net={net2:+8.2f}%  net={m['net']:+9.2f}%  dd={m['dd']:+6.2f}%", flush=True)
    rows.append(dict(tag=tag, a1=m["h1_net"], a1_dn_wr=wr1, a1_dn_net=net1, a1_dn_aw=aw1,
                     a2_dn_wr=wr2, a2_dn_net=net2, net=m["net"], dd=m["dd"], n=m["n"]))


rows = []
m0, tr0 = run()
rec(rows, "production", m0, tr0)

print("\n=== palier chop seulement ===", flush=True)
for arm in (0.4, 0.5, 0.6, 0.7):
    for lock in (0.10, 0.15, 0.25):
        m, tr = run(arm=arm, lock=lock)
        rec(rows, f"chop[{arm}->{lock}]", m, tr)

print("\n=== palier chop + macro-bull ===", flush=True)
for arm in (0.5, 0.6, 0.7):
    for lock in (0.10, 0.25):
        m, tr = run(arm=arm, lock=lock, scope="both")
        rec(rows, f"both[{arm}->{lock}]", m, tr)

pd.DataFrame(rows).to_csv(H.ROOT / "logs" / "exp_dnwr.csv", index=False)
print("\n-> logs/exp_dnwr.csv", flush=True)
