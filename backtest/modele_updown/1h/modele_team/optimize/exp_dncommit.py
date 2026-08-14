# -*- coding: utf-8 -*-
"""exp_dncommit.py — decomposition de l'ecart DOWN commit ecb04f8 (+372.01%,
140 trades, WR 55.71% sur 365j) vs run actuel (annee 2 +143.74%, 114 trades).

Le moteur DOWN est bit-identique au commit depuis le retour du 2026-08-14 ;
les modeles n'ont pas ete re-entraines depuis le 2026-07-03 (memes signaux).
Candidats restants : (a) caps de concurrence (2026-07-10, 68 DOWN sautes),
(b) gouverneur d'equity (2026-08-11), (c) fenetre (annee 2 = 2025-07-13 ->
2026-07-03 vs commit = ~2025-07-04 -> 2026-07-04).

Deux variantes, memes entrees : PROD (caps+gouverneur, = run du jour) et
NOCAP (caps 9999 + gouverneur off = cadre exact du commit). Stats DOWN sur
l'annee 2 ET sur les 365 derniers jours (fenetre du commit).

Usage : cd modele_team && python optimize/exp_dncommit.py
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
    prepared, _regime_map_cache = pickle.load(f)
M._SWING_PIVOTS = None
REGIME_MAP = None          # production actuelle : wrapper regime desactive
TREND = filters.load_trend_context()   # n'affecte plus que l'UP

# --- Chaine d'entrees PRODUCTION (identique aux deux variantes) ---
up_full = prepared["up_full"]
down_full = prepared["down_full"].copy()
ts_arr = down_full["timestamp"].to_numpy()
eff_delta = filters.effective_down_delta(ts_arr, REGIME_MAP, TREND)  # plat 0.03
down_full["take"] = (down_full["confidence"] >= down_full["threshold"] + eff_delta).astype(int)
up_bets = M._taken_bets(up_full, "up")
up_bets = filters.apply_xmodel_filter(up_bets, down_full)
down_bets = M._taken_bets(down_full, "down")
team_bets, _ = M.build_team_bets(up_bets, down_bets)
team_bets, _, _ = filters.apply_dmi_filter(team_bets, prepared.get("dmi_map"))
team_bets, _ = filters.apply_conf_filter(team_bets, REGIME_MAP, TREND)

DAY_MS = C.DAY_MS
Y1_END = prepared["start_ms"] + 365 * DAY_MS


def dn_stats(tr, lo=None, hi=None):
    s = tr[tr.direction == "down"]
    if lo is not None:
        s = s[s.traded_ts >= lo]
    if hi is not None:
        s = s[s.traded_ts < hi]
    net = s["net_ret"].to_numpy(float) * C.EXPOSURE
    if not len(net):
        return "n=0"
    w, l = net[net > 0], net[net < 0]
    comp = (np.prod(1 + net) - 1) * 100
    return (f"n={len(net):3d}  WRnet={(net > 0).mean()*100:5.2f}%  net={comp:+8.2f}%  "
            f"aw={w.mean()*100 if len(w) else 0:.3f}%  al={-l.mean()*100 if len(l) else 0:.3f}%  "
            f"pire={net.min()*100:+.2f}%")


def run(cap: bool, gov: bool):
    old = (M.MAX_CONCURRENT_POSITIONS, M.MAX_CONCURRENT_POSITIONS_PER_SIDE)
    if not cap:
        M.MAX_CONCURRENT_POSITIONS = 9999
        M.MAX_CONCURRENT_POSITIONS_PER_SIDE = 9999
    try:
        trades, info = M.simulate_trades(team_bets, prepared["body_map"],
                                         prepared["prices_intrabar"], None,
                                         prepared["extremes"], REGIME_MAP, None, TREND)
    finally:
        M.MAX_CONCURRENT_POSITIONS, M.MAX_CONCURRENT_POSITIONS_PER_SIDE = old
    n_gov = 0
    # Gouverneur supprime du code le 2026-08-14 (decision user) : le garde-fou
    # getattr laisse ce script historique tourner sans lui.
    if gov and getattr(C, "GOVERNOR_ENABLED", False) and not trades.empty:
        trades, n_gov = filters.apply_governor(trades)
    return trades, info, n_gov


END = None
for tag, cap, gov in (("PROD (caps+gouverneur)", True, True),
                      ("CAPS seuls (gouverneur off)", True, False),
                      ("NOCAP (cadre commit)", False, False)):
    tr, info, n_gov = run(cap, gov)
    if END is None:
        END = int(tr["traded_ts"].max())
    last365 = END - 365 * DAY_MS
    print(f"\n=== {tag} ===")
    print(f"  concurrence sautee : {info['down_concurrency_skipped']} DOWN "
          f"/ {info['up_concurrency_skipped']} UP ; gouverneur : {n_gov} trades")
    print(f"  DOWN annee 2        : {dn_stats(tr, lo=Y1_END)}")
    print(f"  DOWN 365 derniers j : {dn_stats(tr, lo=last365)}")
    print(f"  DOWN annee 1        : {dn_stats(tr, hi=Y1_END)}")
print("\nReference commit ecb04f8 (365j, sans caps ni gouverneur) : "
      "n=140  WRnet=55.71%  net=+372.01%  aw=4.996%  al=3.493%  pire=-8.93%")
