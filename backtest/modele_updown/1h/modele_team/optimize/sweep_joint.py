"""sweep_joint.py — PHASE B : re-sweep CONJOINT (take-profit x trailing gap x
activation x cliquet), un cote a la fois (decomposabilite verifiee en phase A).

Justification du perimetre elargi : les phases A/A2 montrent que le cliquet SEUL
est monotonement destructeur (plus il mord, plus le net baisse : DOWN +502% ->
+4% a arm=0.2%, UP +125% -> +27% a arm=0.10%) -- les trades CLIPPES coutent bien
plus que les trades SAUVES. Le trou de verrouillage est donc le prix du
trend-ride A GAP/TP CONSTANTS. La question qui reste ouverte, et que le user a
explicitement autorisee, est : une fois le cliquet disponible, le couple
(TP, gap) optimal est-il le meme ? Le SL temporel et l'horizon (hold) restent
STRICTEMENT figes -- seule la partie "laisser courir / verrouiller" bouge.

Usage : cd modele_team && python optimize/sweep_joint.py
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

# Cliquets candidats : None (aucun) + proportionnels + paliers plats.
# arm en % de mouvement de prix, frac = fraction du pic conservee.
def ratchet_grid(max_arm):
    g = [("none", None, None)]
    for arm in [a for a in (0.3, 0.5, 0.8, 1.0, 1.5, 2.0) if a <= max_arm]:
        for frac in (0.0, 0.3, 0.5, 0.7):
            g.append((f"prop{arm}/{frac}", arm, frac))
    return g


def sweep(prepared, regime_map, split, side, tps, gaps, acts, base, max_arm):
    tkey, gkey, akey, pkey = (("up_tp", "up_gap", "up_activate", "up_prop") if side == "up"
                              else ("down_tp", "down_gap", "down_activate", "down_prop"))
    rows = []
    grid = list(itertools.product(tps, gaps, acts, ratchet_grid(max_arm)))
    t0 = time.time()
    for tp, gap, act, (rname, arm, frac) in grid:
        if act >= tp:                       # activation au-dela du TP = trailing mort
            continue
        prop = None if arm is None else (arm / 100.0, frac)
        m, _ = H.evaluate(prepared, regime_map, split, **{
            tkey: tp / 100.0, gkey: gap / 100.0, akey: act / 100.0, pkey: prop})
        rows.append({
            "side": side, "tp": tp, "gap": gap, "act": act, "ratchet": rname,
            "arm": arm, "frac": frac,
            "n": m[f"{side}_n"], "net": m[f"{side}_net"], "ev": m[f"{side}_ev"],
            "wr": m[f"{side}_wr"], "worst": m[f"{side}_worst"],
            "h1_net": m[f"h1_{side}_net"], "h2_net": m[f"h2_{side}_net"],
            "h1_ev": m[f"h1_{side}_ev"], "h2_ev": m[f"h2_{side}_ev"],
            "total_n": m["n"], "total_worst": m["worst"], "dd": m["dd"],
        })
    df = pd.DataFrame(rows)
    df["d_net"] = df["net"] - base[f"{side}_net"]
    df["both_net"] = ((df["h1_net"] > base[f"h1_{side}_net"])
                      & (df["h2_net"] > base[f"h2_{side}_net"]))
    df["both_ev"] = ((df["h1_ev"] > base[f"h1_{side}_ev"])
                     & (df["h2_ev"] > base[f"h2_{side}_ev"]))
    print(f"  {side}: {len(df)} configs en {time.time()-t0:.0f}s")
    return df


def main():
    prepared, regime_map = H.load_prepared()
    split = H.split_point(prepared)
    base, _ = H.evaluate(prepared, regime_map, split)
    print(f"BASELINE n={base['n']} net={base['net']:.2f}% ev={base['ev']:.4f}% "
          f"worst={base['worst']:.2f}% dd={base['dd']:.2f}%")
    print(f"  UP   net={base['up_net']:.2f}% h1={base['h1_up_net']:+.2f} h2={base['h2_up_net']:+.2f}")
    print(f"  DOWN net={base['down_net']:.2f}% h1={base['h1_down_net']:+.2f} h2={base['h2_down_net']:+.2f}\n")

    # gap 9.0% = trailing desactive de fait (jamais atteint) -> isole l'effet TP seul
    up = sweep(prepared, regime_map, split, "up",
               tps=(0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.3),
               gaps=(0.3, 0.5, 0.7, 1.0, 1.5, 2.5, 9.0),
               acts=(0.05, 0.15, 0.25, 0.40), base=base, max_arm=1.0)
    dn = sweep(prepared, regime_map, split, "down",
               tps=(1.0, 1.3, 1.6, 2.0, 2.5, 3.5),
               gaps=(0.6, 0.9, 1.2, 1.6, 2.2, 3.0, 9.0),
               acts=(0.0, 0.2, 0.5), base=base, max_arm=2.0)

    out = pd.concat([up, dn], ignore_index=True)
    OUT.mkdir(exist_ok=True)
    out.to_csv(OUT / "sweep_joint.csv", index=False)
    print(f"\n  written {OUT / 'sweep_joint.csv'} ({len(out)} lignes)\n")

    cols = ["tp", "gap", "act", "ratchet", "n", "net", "ev", "wr", "worst",
            "h1_net", "h2_net", "both_net", "both_ev"]
    for side, df in (("up", up), ("down", dn)):
        b = base[f"{side}_net"]
        print(f"=== {side.upper()} — baseline net={b:.2f}% ===")
        print("  TOP net (toutes configs) :")
        print(df.sort_values("net", ascending=False).head(10)[cols].to_string(index=False))
        ok = df[df["both_net"] & (df["d_net"] > 0)]
        print(f"  -> {len(ok)}/{len(df)} configs battent la baseline sur LES DEUX MOITIES")
        if len(ok):
            print(ok.sort_values("net", ascending=False).head(12)[cols].to_string(index=False))
        print()


if __name__ == "__main__":
    main()
