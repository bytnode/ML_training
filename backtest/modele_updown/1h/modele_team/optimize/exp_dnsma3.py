# -*- coding: utf-8 -*-
"""exp_dnsma3.py — suite du balayage SMA du contexte de tendance (demande user
2026-08-14) : SMA 120j et 180j, config "SMA aussi cote DOWN"
(CHOP_DOWN_ENABLED=True). References deja mesurees (exp_dnsma*.py) :
  B200 : total +2271.76% dd -31.87% a1 +101.79% a2 +1075.34%
  B365 : total +2202.99% dd -31.87% a1 +115.72% a2  +967.60%

Une SMA plus courte colle plus au prix -> le prix la croise plus souvent ->
l'age du bear se remet a zero plus souvent (moins de "bear installe") et le
flag macro-bull clignote plus. Cache contexte reconstruit par variante.

Usage : cd modele_team && python optimize/exp_dnsma3.py
"""
from __future__ import annotations
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main as M
import config_main as C
import filters

ROOT = Path(__file__).resolve().parent.parent
# Cache pickle ecrit par optimize/test_720.py (nos propres DataFrames) -> sur.
with open(ROOT / "optimize" / ".cache_prepared_720.pkl", "rb") as f:
    prepared, _ = pickle.load(f)
M._SWING_PIVOTS = None
REGIME_MAP = filters.load_regime_map()
Y1_END = prepared["start_ms"] + 365 * C.DAY_MS
WIN_TS = prepared["down_full"]["timestamp"].astype("int64").to_numpy()


def trend_ctx_for(sma_days: int):
    filters.CHOP_SMA_DAYS = sma_days
    filters._TREND_CTX_CACHE = None
    ctx = filters.load_trend_context()
    cov = np.mean([int(t) in ctx for t in WIN_TS]) * 100
    print(f"[SMA {sma_days}j] couverture du contexte sur la fenetre : {cov:.1f}%")
    return ctx


def build_and_run(trend):
    M.CHOP_DOWN_ENABLED = True
    filters.CHOP_DOWN_ENABLED = True
    down_full = prepared["down_full"].copy()
    eff = filters.effective_down_delta(down_full["timestamp"].to_numpy(), REGIME_MAP, trend)
    down_full["take"] = (down_full["confidence"] >= down_full["threshold"] + eff).astype(int)
    up_bets = filters.apply_xmodel_filter(M._taken_bets(prepared["up_full"], "up"), down_full)
    team, _ = M.build_team_bets(up_bets, M._taken_bets(down_full, "down"))
    team, _, _ = filters.apply_dmi_filter(team, prepared.get("dmi_map"))
    team, _ = filters.apply_conf_filter(team, REGIME_MAP, trend)
    trades, _ = M.simulate_trades(team, prepared["body_map"], prepared["prices_intrabar"],
                                  None, prepared["extremes"], REGIME_MAP, None, trend)
    return trades


def side_stats(tr, direction, y1):
    m = (tr.traded_ts < Y1_END) if y1 else (tr.traded_ts >= Y1_END)
    s = tr[m & (tr.direction == direction)]
    net = s["net_ret"].to_numpy(float) * C.EXPOSURE
    if not len(net):
        return "n=0"
    return (f"n={len(net):3d} WR={(net > 0).mean()*100:5.2f}% "
            f"net={(np.prod(1+net)-1)*100:+8.2f}%")


def show(tag, tr):
    net = tr.sort_values("traded_ts")["net_ret"].to_numpy(float) * C.EXPOSURE
    eq = np.cumprod(1 + net)
    dd = float((eq / np.maximum.accumulate(eq) - 1).min()) * 100
    m1 = tr.traded_ts < Y1_END
    a1 = (np.prod(1 + tr[m1].sort_values("traded_ts")["net_ret"].to_numpy(float) * C.EXPOSURE) - 1) * 100
    a2 = (np.prod(1 + tr[~m1].sort_values("traded_ts")["net_ret"].to_numpy(float) * C.EXPOSURE) - 1) * 100
    print(f"\n=== {tag} ===")
    print(f"  TOTAL net={(eq[-1]-1)*100:+9.2f}%  dd={dd:+6.2f}%  n={len(tr)}  |  annee1={a1:+8.2f}%  annee2={a2:+9.2f}%")
    for y1, lab in ((True, "annee 1"), (False, "annee 2")):
        print(f"  {lab}  UP   : {side_stats(tr, 'up', y1)}")
        print(f"  {lab}  DOWN : {side_stats(tr, 'down', y1)}")


for days in (180, 120):
    show(f"B{days}. SMA{days}j, UP+DOWN", build_and_run(trend_ctx_for(days)))

# restauration etat module (config production inchangee)
filters.CHOP_SMA_DAYS = C.CHOP_SMA_DAYS
filters._TREND_CTX_CACHE = None
M.CHOP_DOWN_ENABLED = filters.CHOP_DOWN_ENABLED = C.CHOP_DOWN_ENABLED
