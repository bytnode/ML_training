"""sweep_down_negfloor.py — PHASE E : reprise du fil laisse pendant en phase A.

La phase A (paliers PLATS) avait trouve DEUX configs DOWN battant la baseline sur
les deux moities : pic>=1.1% -> plancher -0.1% (+539.27%) et pic>=1.2% -> -0.1%
(+522.98%). Ecartees a l'epoque car -0.1% etait la BORNE BASSE de la grille (le
protocole interdit d'adopter une valeur au bord). Le sweep conjoint ne pouvait pas
les retrouver : il ne testait que des cliquets PROPORTIONNELS, dont le plancher
(frac x pic) est toujours >= 0.

Ici on etend la grille FRANCHEMENT vers le bas (jusqu'a -1.0%) pour savoir si
-0.1% est le centre d'un plateau ou un pic isole, et on evalue le total avec la
config UP retenue (TP 1.55 / gap 1.00 / act 0.30 / cliquet prop(1.00, 0.30)).

Usage : cd modele_team && python optimize/sweep_down_negfloor.py
"""
from __future__ import annotations
import itertools, sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _harness as H

UP_RETENU = dict(up_tp=0.0155, up_gap=0.0100, up_activate=0.0030, up_prop=(0.0100, 0.30))

prepared, regime_map = H.load_prepared()
split = H.split_point(prepared)
base, _ = H.evaluate(prepared, regime_map, split)
b_net, b_h1, b_h2 = base["down_net"], base["h1_down_net"], base["h2_down_net"]
b_h1e, b_h2e = base["h1_down_ev"], base["h2_down_ev"]
print(f"BASELINE DOWN net={b_net:.4f}% h1={b_h1:+.2f} h2={b_h2:+.2f} ev={base['down_ev']:.4f}")

# grille etendue VERS LE BAS + maille fine autour de l'armement 1.1
peaks  = [round(x,2) for x in np.arange(0.70, 1.75, 0.05)]
locked = [round(x,2) for x in np.arange(-1.00, 0.45, 0.05)]
rows = []
grid = [(p,l) for p,l in itertools.product(peaks, locked) if l < p]
print(f"  {len(grid)} configs (paliers plats, plancher jusqu'a -1.00%)...")
t0=time.time()
for pk, lk in grid:
    m,_ = H.evaluate(prepared, regime_map, split, down_ratchet=((pk/100, lk/100),), **UP_RETENU)
    rows.append({"peak":pk,"locked":lk,"n":m["down_n"],"net":m["down_net"],"ev":m["down_ev"],
                 "wr":m["down_wr"],"h1_net":m["h1_down_net"],"h2_net":m["h2_down_net"],
                 "h1_ev":m["h1_down_ev"],"h2_ev":m["h2_down_ev"],
                 "total_n":m["n"],"total_net":m["net"],"total_worst":m["worst"],"dd":m["dd"]})
df = pd.DataFrame(rows)
df["valid"] = ((df.net>b_net)&(df.h1_net>b_h1)&(df.h2_net>b_h2)
               &(df.h1_ev>b_h1e)&(df.h2_ev>b_h2e)
               &(df.total_n>=240)&(df.total_worst>=-15)&(df.dd>=-20))
df.to_csv(H.ROOT/"logs"/"sweep_down_negfloor.csv", index=False)
print(f"  fait en {time.time()-t0:.0f}s — {df.valid.sum()}/{len(df)} valides")
c=["peak","locked","n","net","ev","wr","h1_net","h2_net","total_n","total_net","total_worst","dd"]
ok=df[df.valid]
if len(ok):
    print("\nTOP 15 valides :"); print(ok.sort_values("net",ascending=False).head(15)[c].to_string(index=False))
    print("\nPLATEAU — net moyen des valides par valeur :")
    for p in ("peak","locked"):
        print(f"\n  {p}:"); print(ok.groupby(p)["net"].agg(["size","mean","max"]).round(2).to_string())
else:
    print("\nAUCUNE config valide.")
    print("TOP 10 par net brut (sans filtre) :")
    print(df.sort_values("net",ascending=False).head(10)[c].to_string(index=False))
