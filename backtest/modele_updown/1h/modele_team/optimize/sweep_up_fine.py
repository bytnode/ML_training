"""sweep_up_fine.py — PHASE D : maille FINE autour de l'optimum UP de la phase C
(tp=1.6 / gap=1.0 / act=0.3 / cliquet prop(arm=1.0, frac=0.3)).

But : verifier que ce point est le CENTRE d'un plateau et non un pic isole. Le
protocole du repo interdit d'adopter une valeur au bord d'un plateau ; la phase C
montre que les configs valides se concentrent sur tp in {1.4, 1.6} et s'effondrent
a 1.8, donc 1.6 doit etre re-examine a maille fine avant adoption.

Usage : cd modele_team && python optimize/sweep_up_fine.py
"""
from __future__ import annotations
import itertools, sys, time
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _harness as H

prepared, regime_map = H.load_prepared()
split = H.split_point(prepared)
base, _ = H.evaluate(prepared, regime_map, split)
b_net, b_h1, b_h2 = base["up_net"], base["h1_up_net"], base["h2_up_net"]
b_h1e, b_h2e = base["h1_up_ev"], base["h2_up_ev"]
print(f"BASELINE UP net={b_net:.2f}% h1={b_h1:+.2f} h2={b_h2:+.2f}")

tps   = (1.30, 1.40, 1.50, 1.55, 1.60, 1.65, 1.70, 1.80, 1.90)
gaps  = (0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20)
acts  = (0.25, 0.30, 0.35)
arms  = (0.85, 0.90, 1.00, 1.10, 1.20, 1.30)
fracs = (0.20, 0.25, 0.30, 0.35, 0.40)

rows = []
grid = list(itertools.product(tps, gaps, acts, arms, fracs))
print(f"  {len(grid)} configs...")
t0 = time.time()
for tp, gap, act, arm, frac in grid:
    m, _ = H.evaluate(prepared, regime_map, split, up_tp=tp/100, up_gap=gap/100,
                      up_activate=act/100, up_prop=(arm/100, frac))
    rows.append({"tp": tp, "gap": gap, "act": act, "arm": arm, "frac": frac,
                 "n": m["up_n"], "net": m["up_net"], "ev": m["up_ev"], "wr": m["up_wr"],
                 "h1_net": m["h1_up_net"], "h2_net": m["h2_up_net"],
                 "h1_ev": m["h1_up_ev"], "h2_ev": m["h2_up_ev"],
                 "total_n": m["n"], "total_net": m["net"],
                 "total_worst": m["worst"], "dd": m["dd"]})
df = pd.DataFrame(rows)
df["valid"] = ((df.h1_net > b_h1) & (df.h2_net > b_h2) & (df.net > b_net)
               & (df.h1_ev > b_h1e) & (df.h2_ev > b_h2e)
               & (df.total_n >= 240) & (df.total_worst >= -15) & (df.dd >= -20))
df.to_csv(H.ROOT/"logs"/"sweep_up_fine.csv", index=False)
print(f"  fait en {time.time()-t0:.0f}s — {df.valid.sum()}/{len(df)} valides")
ok = df[df.valid]
c = ["tp","gap","act","arm","frac","n","net","ev","h1_net","h2_net","total_n","total_net","total_worst","dd"]
print("\nTOP 15 :"); print(ok.sort_values("net",ascending=False).head(15)[c].to_string(index=False))
print("\nROBUSTESSE — net moyen des VALIDES par valeur (plateau si voisins nombreux et proches) :")
for p in ("tp","gap","act","arm","frac"):
    print(f"\n  {p}:"); print(ok.groupby(p)["net"].agg(["size","mean","max"]).round(2).to_string())
