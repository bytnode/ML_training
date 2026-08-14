"""sweep_ratchet.py — PHASE A : effet ISOLE du cliquet de gain, un cote a la fois.

Balaye (seuil d'armement x niveau verrouille) pour UP puis pour DOWN, cliquet
a 1 palier, TOUS les autres parametres (TP, gap, activation, SL, hold) figes aux
valeurs de config_main.py.

DECOMPOSABILITE : avec MAX_CONCURRENT_POSITIONS_PER_SIDE=1, le plafond total de 2
ne peut jamais mordre seul (1 UP + 1 DOWN = 2 au plus), donc les trades UP ne
dependent QUE des parametres UP et reciproquement ; le net compose total est le
PRODUIT des deux nets par sens. Verifie explicitement au demarrage (assert), puis
chaque cote est balaye seul.

Usage : cd modele_team && python optimize/sweep_ratchet.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _harness as H                                       # noqa: E402

OUT = H.ROOT / "logs"


def _steps(peak_pct, locked_pct):
    return ((peak_pct / 100.0, locked_pct / 100.0),)


def check_decomposability(prepared, regime_map, split):
    """Change un parametre DOWN et verifie que les trades UP sont INCHANGES."""
    base, tb = H.evaluate(prepared, regime_map, split)
    alt, ta = H.evaluate(prepared, regime_map, split, down_ratchet=_steps(0.6, 0.0))
    up_b = tb[tb["direction"] == "up"]["net_ret"].to_numpy()
    up_a = ta[ta["direction"] == "up"]["net_ret"].to_numpy()
    assert len(up_b) == len(up_a) and np.allclose(up_b, up_a), "UP a bouge en changeant DOWN !"
    prod = (1 + base["up_net"] / 100) * (1 + base["down_net"] / 100) - 1
    assert abs(prod * 100 - base["net"]) < 1e-6, "net total != produit des nets par sens"
    print(f"  OK decomposable : UP inchange ({len(up_b)} trades), "
          f"net total = produit des sens ({prod*100:.4f}%)")
    return base


def sweep_side(prepared, regime_map, split, side, peaks, lockeds, base):
    rows = []
    key = "up_ratchet" if side == "up" else "down_ratchet"
    for pk in peaks:
        for lk in lockeds:
            if lk >= pk:          # un plancher >= au pic d'armement n'a pas de sens
                continue
            m, tr = H.evaluate(prepared, regime_map, split, **{key: _steps(pk, lk)})
            rows.append({
                "side": side, "peak_pct": pk, "locked_pct": lk,
                "n": m[f"{side}_n"], "net": m[f"{side}_net"], "ev": m[f"{side}_ev"],
                "wr": m[f"{side}_wr"], "worst": m[f"{side}_worst"],
                "h1_net": m[f"h1_{side}_net"], "h2_net": m[f"h2_{side}_net"],
                "h1_ev": m[f"h1_{side}_ev"], "h2_ev": m[f"h2_{side}_ev"],
                "total_n": m["n"], "total_net": m["net"], "dd": m["dd"], "total_worst": m["worst"],
            })
    df = pd.DataFrame(rows)
    b = {"net": base[f"{side}_net"], "ev": base[f"{side}_ev"],
         "h1_net": base[f"h1_{side}_net"], "h2_net": base[f"h2_{side}_net"],
         "h1_ev": base[f"h1_{side}_ev"], "h2_ev": base[f"h2_{side}_ev"]}
    df["d_net"] = df["net"] - b["net"]
    df["d_ev"] = df["ev"] - b["ev"]
    df["both_halves"] = (df["h1_ev"] > b["h1_ev"]) & (df["h2_ev"] > b["h2_ev"])
    df["both_halves_net"] = (df["h1_net"] > b["h1_net"]) & (df["h2_net"] > b["h2_net"])
    return df, b


def main():
    prepared, regime_map = H.load_prepared()
    split = H.split_point(prepared)
    print(f"fenetre : {pd.Timestamp(prepared['start_ms'], unit='ms')} -> "
          f"{pd.Timestamp(prepared['end_ms'], unit='ms')}  | split h1/h2 : "
          f"{pd.Timestamp(split, unit='ms')}")
    print("\n[0] verification de la decomposabilite UP/DOWN")
    base = check_decomposability(prepared, regime_map, split)
    print(f"\nBASELINE  total n={base['n']} net={base['net']:.2f}% ev={base['ev']:.4f}% "
          f"worst={base['worst']:.2f}% dd={base['dd']:.2f}%")
    for s in ("up", "down"):
        print(f"  {s.upper():5s} n={base[f'{s}_n']:3d} net={base[f'{s}_net']:8.2f}% "
              f"ev={base[f'{s}_ev']:.4f}% wr={base[f'{s}_wr']:.2f}% "
              f"| h1_ev={base[f'h1_{s}_ev']:+.4f} h2_ev={base[f'h2_{s}_ev']:+.4f} "
              f"h1_net={base[f'h1_{s}_net']:+.2f} h2_net={base[f'h2_{s}_net']:+.2f}")

    # UP : TP=0.60% -> un pic d'armement au-dela n'est jamais atteint (le TP sort avant)
    up_peaks = [round(x, 2) for x in np.arange(0.10, 0.60, 0.05)]
    up_locked = [round(x, 2) for x in np.arange(-0.10, 0.55, 0.05)]
    # DOWN : TP=1.60% (x4 si elargi) -> on balaye jusqu'au TP
    dn_peaks = [round(x, 2) for x in np.arange(0.20, 1.65, 0.10)]
    dn_locked = [round(x, 2) for x in np.arange(-0.10, 1.55, 0.10)]

    frames = []
    for side, pk, lk in (("up", up_peaks, up_locked), ("down", dn_peaks, dn_locked)):
        df, b = sweep_side(prepared, regime_map, split, side, pk, lk, base)
        frames.append(df)
        print(f"\n=== {side.upper()} — {len(df)} configs, baseline net={b['net']:.2f}% ev={b['ev']:.4f}% ===")
        top = df.sort_values("net", ascending=False).head(12)
        print(top[["peak_pct", "locked_pct", "n", "net", "ev", "wr", "worst",
                   "h1_net", "h2_net", "both_halves", "both_halves_net"]].to_string(index=False))
        ok = df[df["both_halves_net"] & (df["net"] > b["net"])]
        print(f"  -> {len(ok)} configs battent la baseline SUR LES DEUX MOITIES (net)")
        if len(ok):
            print(ok.sort_values("net", ascending=False).head(8)[
                ["peak_pct", "locked_pct", "n", "net", "ev", "worst", "h1_net", "h2_net"]
            ].to_string(index=False))

    out = pd.concat(frames, ignore_index=True)
    OUT.mkdir(exist_ok=True)
    out.to_csv(OUT / "sweep_ratchet_phaseA.csv", index=False)
    print(f"\n  written {OUT / 'sweep_ratchet_phaseA.csv'}  ({len(out)} lignes)")


if __name__ == "__main__":
    main()
