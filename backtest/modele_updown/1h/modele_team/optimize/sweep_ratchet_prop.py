"""sweep_ratchet_prop.py — PHASE A2 : cliquet PROPORTIONNEL (garder une fraction
du pic) balaye par cote, seul puis combine a un palier plat.

Motivation (resultat de la phase A) : les paliers PLATS seuls ne battent pas la
baseline cote UP (meilleur +1.5pt, et il echoue le test des 2 moities) et, cote
DOWN, les seules configs gagnantes verrouillent a -0.10% -- elles ne securisent
donc AUCUN gain, elles ne font que caper le giveback en regime bear (gap x1.3 =
1.56%). Le parametre qui mord n'est pas un niveau absolu mais la FRACTION du pic
rendue : c'est exactement ce qu'un plancher proportionnel exprime.

Usage : cd modele_team && python optimize/sweep_ratchet_prop.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _harness as H                                       # noqa: E402

OUT = H.ROOT / "logs"


def run(prepared, regime_map, split, side, arms, fracs, base):
    pkey = "up_prop" if side == "up" else "down_prop"
    rows = []
    for arm in arms:
        for fr in fracs:
            m, _ = H.evaluate(prepared, regime_map, split, **{pkey: (arm / 100.0, fr)})
            rows.append({
                "side": side, "arm_pct": arm, "frac": fr,
                "n": m[f"{side}_n"], "net": m[f"{side}_net"], "ev": m[f"{side}_ev"],
                "wr": m[f"{side}_wr"], "worst": m[f"{side}_worst"],
                "h1_net": m[f"h1_{side}_net"], "h2_net": m[f"h2_{side}_net"],
                "h1_ev": m[f"h1_{side}_ev"], "h2_ev": m[f"h2_{side}_ev"],
            })
    df = pd.DataFrame(rows)
    df["d_net"] = df["net"] - base[f"{side}_net"]
    df["both_net"] = ((df["h1_net"] > base[f"h1_{side}_net"])
                      & (df["h2_net"] > base[f"h2_{side}_net"]))
    df["both_ev"] = ((df["h1_ev"] > base[f"h1_{side}_ev"])
                     & (df["h2_ev"] > base[f"h2_{side}_ev"]))
    return df


def show(df, side, base):
    print(f"\n=== {side.upper()} proportionnel — {len(df)} configs "
          f"(baseline net={base[f'{side}_net']:.2f}%, "
          f"h1={base[f'h1_{side}_net']:+.2f} h2={base[f'h2_{side}_net']:+.2f}) ===")
    top = df.sort_values("net", ascending=False).head(15)
    print(top[["arm_pct", "frac", "n", "net", "ev", "wr", "worst",
               "h1_net", "h2_net", "both_net", "both_ev"]].to_string(index=False))
    ok = df[df["both_net"] & (df["d_net"] > 0)]
    print(f"  -> {len(ok)} configs battent la baseline SUR LES DEUX MOITIES (net)")
    if len(ok):
        print(ok.sort_values("net", ascending=False).head(10)[
            ["arm_pct", "frac", "n", "net", "ev", "worst", "h1_net", "h2_net", "both_ev"]
        ].to_string(index=False))
    return ok


def main():
    prepared, regime_map = H.load_prepared()
    split = H.split_point(prepared)
    base, _ = H.evaluate(prepared, regime_map, split)
    print(f"BASELINE total n={base['n']} net={base['net']:.2f}% ev={base['ev']:.4f}% "
          f"worst={base['worst']:.2f}% dd={base['dd']:.2f}%")

    fracs = [round(x, 2) for x in np.arange(0.0, 0.85, 0.05)]
    up_arms = [round(x, 2) for x in np.arange(0.10, 0.62, 0.05)]      # TP UP = 0.60%
    dn_arms = [round(x, 2) for x in np.arange(0.20, 1.65, 0.10)]      # TP DOWN = 1.60%

    frames = {}
    for side, arms in (("up", up_arms), ("down", dn_arms)):
        df = run(prepared, regime_map, split, side, arms, fracs, base)
        show(df, side, base)
        frames[side] = df

    out = pd.concat(frames.values(), ignore_index=True)
    OUT.mkdir(exist_ok=True)
    out.to_csv(OUT / "sweep_ratchet_prop.csv", index=False)
    print(f"\n  written {OUT / 'sweep_ratchet_prop.csv'}  ({len(out)} lignes)")

    # --- Combinaison du meilleur de chaque cote (decomposable -> produit exact) ---
    best = {}
    for side in ("up", "down"):
        ok = frames[side][frames[side]["both_net"] & (frames[side]["d_net"] > 0)]
        pool = ok if len(ok) else frames[side]
        r = pool.sort_values("net", ascending=False).iloc[0]
        best[side] = (r["arm_pct"], r["frac"], r["net"], bool(r["both_net"]))
        print(f"  meilleur {side.upper()} : arm={r['arm_pct']}% frac={r['frac']} "
              f"net={r['net']:.2f}% (2 moities OK={bool(r['both_net'])})")
    m, tr = H.evaluate(prepared, regime_map, split,
                       up_prop=(best["up"][0] / 100.0, best["up"][1]),
                       down_prop=(best["down"][0] / 100.0, best["down"][1]))
    print(f"\nCOMBINE  n={m['n']} net={m['net']:.2f}% ev={m['ev']:.4f}% "
          f"worst={m['worst']:.2f}% dd={m['dd']:.2f}%  "
          f"(baseline net={base['net']:.2f}% ev={base['ev']:.4f}% "
          f"worst={base['worst']:.2f}% dd={base['dd']:.2f}%)")


if __name__ == "__main__":
    main()
