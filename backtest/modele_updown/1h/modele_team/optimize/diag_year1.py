# -*- coding: utf-8 -*-
"""diag_year1.py — PHASE 1 du goal "rentable sur tout type de marche" (2026-08-11).

Question : d'ou vient EXACTEMENT la perte de l'annee 1 (2024-07 -> 2025-07,
-46.87% net) alors que l'annee 2 fait +1652.97% avec la MEME config ?

Decompose les 720j en annee 1 / annee 2 et croise : sens x raison de sortie,
sens x regime (barre de decision), mois, frais, precision du label 1h des
modeles (verifie "les modeles vont bien, c'est la gestion qui perd").

Usage : cd modele_team && python optimize/diag_year1.py
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

# Cache pickle ecrit par optimize/test_720.py (nos propres DataFrames, aucune
# source externe) -> surface de deserialisation sure, cf note dans test_720.py.
CACHE720 = H.ROOT / "optimize" / ".cache_prepared_720.pkl"
with open(CACHE720, "rb") as f:
    prepared, regime_map = pickle.load(f)
M._SWING_PIVOTS = None

# Config PRODUCTION exacte (BASE du harness remet les cliquets a zero -> les repasser)
PROD = dict(up_ratchet=C.UP_RATCHET_STEPS, down_ratchet=C.DOWN_RATCHET_STEPS,
            up_prop=C.UP_RATCHET_PROP, down_prop=C.DOWN_RATCHET_PROP)

Y1_END = prepared["start_ms"] + 365 * C.DAY_MS
m, tr = H.evaluate(prepared, regime_map, Y1_END, **PROD)
print(f"baseline prod 720j : n={m['n']}  net={m['net']:.2f}%  "
      f"annee1={m['h1_net']:.2f}%  annee2={m['h2_net']:.2f}%")

tr = tr.copy()
tr["year"] = np.where(tr["traded_ts"] < Y1_END, "annee1", "annee2")
tr["net_pts"] = tr["net_ret"] * C.EXPOSURE * 100          # points d'equity (additifs)
tr["gross_pts"] = tr["gross_ret"] * C.EXPOSURE * 100
tr["regime"] = [regime_map.get(int(t), "?") if regime_map else "?" for t in tr["timestamp"]]
tr["month"] = pd.to_datetime(tr["traded_ts"], unit="ms").dt.strftime("%Y-%m")


def agg(g):
    net = g["net_ret"].to_numpy(float) * C.EXPOSURE
    w, l = net[net > 0], net[net < 0]
    return pd.Series({
        "n": len(net),
        "wr": (net > 0).mean() * 100 if len(net) else np.nan,
        "tot_pts": net.sum() * 100,
        "avg_win": w.mean() * 100 if len(w) else 0.0,
        "avg_loss": -l.mean() * 100 if len(l) else 0.0,
        "ev": (len(w) / len(net) * (w.mean() if len(w) else 0)
               - len(l) / len(net) * (-l.mean() if len(l) else 0)) * 100 if len(net) else 0.0,
        "label_acc": g["label"].mean() * 100 if g["label"].notna().any() else np.nan,
    })


pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 300)

print("\n=== 1) annee x sens ===")
print(tr.groupby(["year", "direction"]).apply(agg, include_groups=False).round(2).to_string())

print("\n=== 2) annee x sens x raison de sortie ===")
print(tr.groupby(["year", "direction", "exit_reason"]).apply(agg, include_groups=False).round(2).to_string())

print("\n=== 3) annee x regime x sens ===")
print(tr.groupby(["year", "regime", "direction"]).apply(agg, include_groups=False).round(2).to_string())

print("\n=== 4) distribution des regimes (barres de decision 1h de la fenetre) ===")
bars = prepared["up_full"]["timestamp"].astype("int64")
for tag, lo, hi in (("annee1", prepared["start_ms"], Y1_END),
                    ("annee2", Y1_END, prepared["end_ms"])):
    sel = bars[(bars >= lo) & (bars < hi)]
    regs = pd.Series([regime_map.get(int(t), "?") for t in sel]) if regime_map else pd.Series()
    dist = (regs.value_counts(normalize=True) * 100).round(1).to_dict()
    print(f"  {tag}: {len(sel)} barres  {dist}")

print("\n=== 5) par mois (net pts, n, WR) ===")
mo = tr.groupby(["month"]).apply(agg, include_groups=False).round(2)
print(mo.to_string())

print("\n=== 6) frais par annee (pts) ===")
fee = (tr["gross_ret"] - tr["net_ret"]) * C.EXPOSURE * 100
print(tr.assign(fee_pts=fee).groupby("year")["fee_pts"].sum().round(2).to_string())

print("\n=== 7) brut vs net par annee (le label du modele avait-il raison ?) ===")
for y in ("annee1", "annee2"):
    s = tr[tr["year"] == y]
    for d in ("up", "down"):
        sd = s[s["direction"] == d]
        if not len(sd):
            continue
        print(f"  {y} {d}: label_acc(1h)={sd['label'].mean()*100:.1f}%  "
              f"WR_gross={(sd['gross_ret']>0).mean()*100:.1f}%  WR_net={(sd['net_ret']>0).mean()*100:.1f}%  "
              f"gross_tot={sd['gross_pts'].sum():+.1f}pts  net_tot={sd['net_pts'].sum():+.1f}pts")

tr.to_csv(H.ROOT / "logs" / "diag_year1_trades.csv", index=False)
print("\ntrades annotes -> logs/diag_year1_trades.csv")
