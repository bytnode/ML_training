# -*- coding: utf-8 -*-
"""exp_upgov.py — pourquoi l'UP annee 1 est tombe a +11% apres l'alignement
prod DOWN (2026-08-14) alors que la strategie UP n'a pas change.

⚠️ OBSOLETE (2026-08-14, quelques heures apres) : ce diagnostic a conduit a la
SUPPRESSION complete du gouverneur du code (decision user). Ce script ne tourne
plus (filters.apply_governor n'existe plus) — garde comme trace de la decision.
Verdict qu'il avait etabli : l'UP seul ne declenche JAMAIS le disjoncteur ;
les suspensions venaient a 100%% du saignement DOWN annee 1.

Hypothese : seul canal de couplage = le GOUVERNEUR. Son livre fantome compte
les resultats des DEUX sens ; le DOWN prod saigne en annee 1 -> suspensions ;
et GOVERNOR_SIDES=("up",) fait payer les suspensions a l'UP seul.

3 variantes, entrees identiques (config alignee prod) :
  A) gouverneur actuel (livre 2 sens, gate UP)  -> doit reproduire +11.01%
  B) gouverneur OFF (= prod exacte, elle n'en a pas)
  C) gouverneur UP-autonome (livre UP seul, gate UP seul)

Usage : cd modele_team && python optimize/exp_upgov.py
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
REGIME_MAP = filters.load_regime_map()      # prod : wrapper actif (sorties DOWN)
TREND = filters.load_trend_context()

up_full = prepared["up_full"]
down_full = prepared["down_full"].copy()
eff = filters.effective_down_delta(down_full["timestamp"].to_numpy(), REGIME_MAP, TREND)
down_full["take"] = (down_full["confidence"] >= down_full["threshold"] + eff).astype(int)
up_bets = filters.apply_xmodel_filter(M._taken_bets(up_full, "up"), down_full)
team, _ = M.build_team_bets(up_bets, M._taken_bets(down_full, "down"))
team, _, _ = filters.apply_dmi_filter(team, prepared.get("dmi_map"))
team, _ = filters.apply_conf_filter(team, REGIME_MAP, TREND)

Y1_END = prepared["start_ms"] + 365 * C.DAY_MS

trades_raw, info = M.simulate_trades(team, prepared["body_map"],
                                     prepared["prices_intrabar"], None,
                                     prepared["extremes"], REGIME_MAP, None, TREND)


def stats(tr, direction, y1=True):
    m = (tr.traded_ts < Y1_END) if y1 else (tr.traded_ts >= Y1_END)
    s = tr[m & (tr.direction == direction)]
    net = s["net_ret"].to_numpy(float) * C.EXPOSURE
    if not len(net):
        return "n=0"
    return (f"n={len(net):3d}  WR={(net > 0).mean()*100:5.2f}%  "
            f"net={(np.prod(1+net)-1)*100:+8.2f}%")


def show(tag, tr, n_skip):
    print(f"\n=== {tag} (gouverneur : {n_skip} trades sautes) ===")
    print(f"  UP   annee 1 : {stats(tr, 'up', True)}")
    print(f"  UP   annee 2 : {stats(tr, 'up', False)}")
    print(f"  DOWN annee 1 : {stats(tr, 'down', True)}")


# A) actuel : livre 2 sens, gate UP seul (filters.apply_governor + GOVERNOR_SIDES)
tr_a, n_a = filters.apply_governor(trades_raw)
show("A. livre 2 sens, gate UP (actuel)", tr_a, n_a)

# B) gouverneur OFF (= prod exacte)
show("B. gouverneur OFF (prod exacte)", trades_raw, 0)

# C) livre UP seul, gate UP seul (l'UP se protege de SES propres series)
up_only = trades_raw[trades_raw.direction == "up"].reset_index(drop=True)
dn_only = trades_raw[trades_raw.direction == "down"]
up_gov, n_c = filters.apply_governor(up_only)
tr_c = pd.concat([up_gov, dn_only]).sort_values("traded_ts").reset_index(drop=True)
show("C. livre UP seul, gate UP seul", tr_c, n_c)
