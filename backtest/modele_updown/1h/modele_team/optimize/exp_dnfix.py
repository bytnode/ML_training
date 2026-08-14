# -*- coding: utf-8 -*-
"""exp_dnfix.py — campagne DOWN-ANNEE-1 (2026-08-11, 3e goal du jour) :
reduire les 34 stop-loss DOWN en chop (-153 pts, le plus gros poste de perte
residuel de l'annee 1).

Contexte : le SL DOWN chop avait ete teste et rejete AVANT le changement du
trailing (act 0.0->0.4) — a l'epoque le trailing sortait les perdants tot ;
maintenant ils vont au SL profond (14 -> 34 sorties SL). Re-test sous les
parametres ACTUELS + 2 axes jamais testes : selectivite d'entree DOWN en chop
(bump de DOWN_CONF_DELTA) et TP scalp re-regle. GOUVERNEUR INCLUS dans chaque
mesure (c'est le systeme de production complet qui est evalue).

Usage : cd modele_team && python optimize/exp_dnfix.py
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


def team_and_meta(chop_delta_bump=0.0):
    """Chaine de filtres du harnais + bump OPTIONNEL de DOWN_CONF_DELTA sur les
    barres en chop (selectivite d'entree renforcee en chop uniquement)."""
    up_full = prepared["up_full"]
    down_full = prepared["down_full"]
    ts_arr = down_full["timestamp"].to_numpy()
    eff_delta = filters.effective_down_delta(ts_arr, regime_map)
    if chop_delta_bump:
        bump = np.array([chop_delta_bump if filters.chop_mode(TREND, int(t)) else 0.0
                         for t in ts_arr])
        eff_delta = eff_delta + bump
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
    return team_bets, up_meta


def run(dn_sl=None, dn_tp=None, chop_delta_bump=0.0):
    """dn_sl = (start,end,floor) pct du SL DOWN en chop ; dn_tp = TP scalp pct ;
    chop_delta_bump = ajout a DOWN_CONF_DELTA en chop. Gouverneur applique."""
    H.reset_params()
    H.set_params(**PROD)
    tb, up_meta = team_and_meta(chop_delta_bump)
    orig_down = M._down_managed_exit

    def down_wrap(prices_intrabar, traded_ts, max_steps=None, trail_gap=None,
                  widen_candidates=None, regime="neutral", swing_pivots=None,
                  take_profit=None, trail_activate=None):
        dts = traded_ts - C.INTERVAL_1H_MS
        if not filters.chop_mode(TREND, dts):
            return orig_down(prices_intrabar, traded_ts, max_steps, trail_gap,
                             widen_candidates, regime, swing_pivots,
                             take_profit, trail_activate)
        old = (M.DOWN_SL_START_DEC, M.DOWN_SL_END_DEC, M.DOWN_SL_FLOOR_DEC)
        if dn_sl is not None:
            M.DOWN_SL_START_DEC = -(dn_sl[0] / 100.0)
            M.DOWN_SL_END_DEC = -(dn_sl[1] / 100.0)
            M.DOWN_SL_FLOOR_DEC = -(dn_sl[2] / 100.0)
        tp = (dn_tp / 100.0) if dn_tp is not None else take_profit
        try:
            return orig_down(prices_intrabar, traded_ts, max_steps, trail_gap,
                             widen_candidates, regime, swing_pivots,
                             tp, trail_activate)
        finally:
            M.DOWN_SL_START_DEC, M.DOWN_SL_END_DEC, M.DOWN_SL_FLOOR_DEC = old

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


def rec(rows, tag, m):
    print(f"  {tag:44s} a1={m['h1_net']:+8.2f}%  a1_dn={m['h1_down_net']:+7.2f}%  "
          f"a2={m['h2_net']:+8.2f}%  net={m['net']:+9.2f}%  dd={m['dd']:+6.2f}%  n={m['n']}", flush=True)
    rows.append(dict(tag=tag, a1=m["h1_net"], a1_dn=m["h1_down_net"], a2=m["h2_net"],
                     net=m["net"], dd=m["dd"], n=m["n"]))


rows = []
m0, _ = run()
rec(rows, "production (verif reproduction)", m0)

print("\n=== A) SL DOWN chop re-regle (sous act0.4/gap0.7) ===", flush=True)
for sl in ((0.9, 0.4, 0.15), (0.8, 0.4, 0.15), (0.8, 0.3, 0.10), (0.7, 0.3, 0.10),
           (0.6, 0.3, 0.10), (1.0, 0.3, 0.10)):
    m, _ = run(dn_sl=sl)
    rec(rows, f"dnSL[{sl[0]}/{sl[1]}/{sl[2]}]", m)

print("\n=== B) TP scalp re-regle ===", flush=True)
for tp in (0.8, 1.0, 1.1):
    m, _ = run(dn_tp=tp)
    rec(rows, f"dnTP[{tp}]", m)

print("\n=== C) selectivite d'entree DOWN en chop (delta bump) ===", flush=True)
for b in (0.01, 0.02, 0.03, 0.04):
    m, _ = run(chop_delta_bump=b)
    rec(rows, f"deltaBump[+{b}]", m)

pd.DataFrame(rows).to_csv(H.ROOT / "logs" / "exp_dnfix.csv", index=False)
print("\n-> logs/exp_dnfix.csv", flush=True)
