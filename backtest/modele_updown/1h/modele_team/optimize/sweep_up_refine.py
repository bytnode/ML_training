"""sweep_up_refine.py — PHASE C : raffinement UP, grilles ETENDUES.

Le sweep conjoint (phase B) place l'optimum UP a tp=1.3 / gap=1.0 / act=0.40 /
cliquet prop(arm=1.0, frac=0.3) -- mais tp, act ET arm etaient tous au BORD de
leur grille, ce que le protocole interdit d'adopter tel quel. On re-balaye donc
large des deux cotes de ce point pour trouver le PLATEAU et s'y placer au milieu.

DOWN n'est pas rebalaye : la phase B a teste 3150 configs (tp 1.0-3.5, gap
0.6-9.0) et AUCUNE ne bat la baseline sur les deux moities ; son optimum
(tp=1.6, gap=1.2) est INTERIEUR a la grille. DOWN reste donc strictement
inchange.

Usage : cd modele_team && python optimize/sweep_up_refine.py
"""
from __future__ import annotations

import itertools
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _harness as H                                       # noqa: E402

OUT = H.ROOT / "logs"


def main():
    prepared, regime_map = H.load_prepared()
    split = H.split_point(prepared)
    base, _ = H.evaluate(prepared, regime_map, split)
    b_net, b_h1, b_h2 = base["up_net"], base["h1_up_net"], base["h2_up_net"]
    print(f"BASELINE UP net={b_net:.2f}% h1={b_h1:+.2f} h2={b_h2:+.2f} "
          f"ev={base['up_ev']:.4f}% n={base['up_n']}")

    tps = (1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.4, 2.8, 3.2)
    gaps = (0.6, 0.8, 1.0, 1.2, 1.5, 1.8, 2.2)
    acts = (0.2, 0.3, 0.4, 0.5, 0.7, 0.9)
    ratchets = [("none", None)]
    for arm in (0.5, 0.8, 1.0, 1.3, 1.6, 2.0, 2.5):
        for frac in (0.0, 0.2, 0.3, 0.4, 0.5, 0.6):
            ratchets.append((f"prop{arm}/{frac}", (arm / 100.0, frac)))

    rows = []
    grid = [g for g in itertools.product(tps, gaps, acts, ratchets) if g[2] < g[0]]
    print(f"  {len(grid)} configs...")
    t0 = time.time()
    for tp, gap, act, (rname, prop) in grid:
        m, _ = H.evaluate(prepared, regime_map, split,
                          up_tp=tp / 100.0, up_gap=gap / 100.0,
                          up_activate=act / 100.0, up_prop=prop)
        rows.append({"tp": tp, "gap": gap, "act": act, "ratchet": rname,
                     "arm": None if prop is None else prop[0] * 100,
                     "frac": None if prop is None else prop[1],
                     "n": m["up_n"], "net": m["up_net"], "ev": m["up_ev"],
                     "wr": m["up_wr"], "worst": m["up_worst"],
                     "h1_net": m["h1_up_net"], "h2_net": m["h2_up_net"],
                     "h1_ev": m["h1_up_ev"], "h2_ev": m["h2_up_ev"],
                     "total_n": m["n"], "total_net": m["net"],
                     "total_worst": m["worst"], "dd": m["dd"]})
    df = pd.DataFrame(rows)
    df["both_net"] = (df["h1_net"] > b_h1) & (df["h2_net"] > b_h2)
    df["both_ev"] = (df["h1_ev"] > base["h1_up_ev"]) & (df["h2_ev"] > base["h2_up_ev"])
    df["valid"] = (df["both_net"] & (df["net"] > b_net) & (df["total_n"] >= 240)
                   & (df["total_worst"] >= -15.0) & (df["dd"] >= -20.0))
    print(f"  fait en {time.time()-t0:.0f}s")
    OUT.mkdir(exist_ok=True)
    df.to_csv(OUT / "sweep_up_refine.csv", index=False)

    cols = ["tp", "gap", "act", "ratchet", "n", "net", "ev", "wr", "worst",
            "h1_net", "h2_net", "total_n", "total_net", "dd", "both_ev"]
    ok = df[df["valid"]]
    print(f"\n{len(ok)}/{len(df)} configs VALIDES (2 moities + n>=240 + worst>=-15 + dd>=-20)")
    print("\nTOP 20 valides :")
    print(ok.sort_values("net", ascending=False).head(20)[cols].to_string(index=False))

    # --- Structure du plateau : moyenne du net par valeur de chaque parametre,
    #     restreinte aux configs valides -> revele si l'optimum est un pic isole. ---
    print("\nPLATEAU (net moyen des configs valides par valeur de parametre) :")
    for p in ("tp", "gap", "act", "arm", "frac"):
        g = ok.groupby(p)["net"].agg(["size", "mean", "max"]).round(2)
        print(f"\n  {p}:\n{g.to_string()}")


if __name__ == "__main__":
    main()
