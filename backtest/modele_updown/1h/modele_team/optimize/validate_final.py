"""validate_final.py — protocole de validation d'une config candidate.

Verifie, contre la baseline, TOUTES les contraintes fixees par le user :
  1. net compose total  > baseline (+1253.85%)
  2. nombre de trades   >= 240 sur l'annee
  3. pire trade         >= -15%   (net, LEVIER INCLUS)
  4. max drawdown       >= -20%
  5. amelioration sur LES DEUX MOITIES de la fenetre (par cote et au total)
et produit la comptabilite SAUVES / CLIPPES demandee : trade par trade, ce que
la nouvelle config change par rapport a la baseline.

Usage : cd modele_team && python optimize/validate_final.py
        (edite CANDIDATS ci-dessous)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _harness as H                                       # noqa: E402
import config_main as C                                    # noqa: E402

BASE_NET = 1253.8459          # baseline de reference (verifiee bit-a-bit)
MIN_TRADES = 240
MIN_WORST = -15.0
MIN_DD = -20.0


def compare(base_tr: pd.DataFrame, new_tr: pd.DataFrame) -> pd.DataFrame:
    """Jointure par timestamp : ce que la config change trade par trade."""
    b = base_tr.set_index("timestamp")[["direction", "net_ret", "exit_reason"]]
    n = new_tr.set_index("timestamp")[["direction", "net_ret", "exit_reason"]]
    j = b.join(n, lsuffix="_b", rsuffix="_n", how="outer")
    j["d_net_lev"] = (j["net_ret_n"].fillna(0) - j["net_ret_b"].fillna(0)) * C.EXPOSURE * 100
    return j


def accounting(j: pd.DataFrame) -> None:
    ch = j[j["d_net_lev"].abs() > 1e-9]
    saved = ch[ch["d_net_lev"] > 0]
    clipped = ch[ch["d_net_lev"] < 0]
    print(f"  trades touches : {len(ch)}  (sauves {len(saved)}, clippes {len(clipped)})")
    print(f"    gain total des SAUVES  : {saved['d_net_lev'].sum():+8.2f} pts "
          f"(moyenne {saved['d_net_lev'].mean() if len(saved) else 0:+.2f})")
    print(f"    cout total des CLIPPES : {clipped['d_net_lev'].sum():+8.2f} pts "
          f"(moyenne {clipped['d_net_lev'].mean() if len(clipped) else 0:+.2f})")
    print(f"    SOLDE                  : {ch['d_net_lev'].sum():+8.2f} pts (somme arithmetique)")


def check(name: str, params: dict, prepared, regime_map, split, base, base_tr) -> dict:
    m, tr = H.evaluate(prepared, regime_map, split, **params)
    ok = {
        "net":    m["net"] > BASE_NET,
        "n":      m["n"] >= MIN_TRADES,
        "worst":  m["worst"] >= MIN_WORST,
        "dd":     m["dd"] >= MIN_DD,
        "h1":     m["h1_net"] > base["h1_net"],
        "h2":     m["h2_net"] > base["h2_net"],
    }
    print(f"\n{'='*78}\n{name}\n{'='*78}")
    print(f"  net={m['net']:9.2f}%  (baseline {BASE_NET:.2f}, delta {m['net']-BASE_NET:+.2f})")
    print(f"  n={m['n']:4d}   ev={m['ev']:.4f}%   wr={m['wr']:.2f}%   "
          f"worst={m['worst']:.2f}%   dd={m['dd']:.2f}%")
    print(f"  UP   n={m['up_n']:3d} net={m['up_net']:8.2f}% ev={m['up_ev']:.4f}%  "
          f"| DOWN n={m['down_n']:3d} net={m['down_net']:8.2f}% ev={m['down_ev']:.4f}%")
    print(f"  moities : h1_net={m['h1_net']:+8.2f} (base {base['h1_net']:+.2f})   "
          f"h2_net={m['h2_net']:+8.2f} (base {base['h2_net']:+.2f})")
    print("  CONTRAINTES : " + "  ".join(
        f"{k}={'OK' if v else 'ECHEC'}" for k, v in ok.items()))
    print(f"  -> {'VALIDE' if all(ok.values()) else 'REJETE'}")
    accounting(compare(base_tr, tr))
    return {"name": name, "all_ok": all(ok.values()), **m}


# Configs a valider -- remplies par l'analyse du sweep conjoint.
CANDIDATS: dict[str, dict] = {
    "RETENU  UP tp=1.55 gap=1.00 act=0.30 cliquet prop(1.00%, 0.30) | DOWN inchange": dict(
        up_tp=0.0155, up_gap=0.0100, up_activate=0.0030, up_prop=(0.0100, 0.30)),
    "voisin  tp=1.60 (plateau +1 cran)": dict(
        up_tp=0.0160, up_gap=0.0100, up_activate=0.0030, up_prop=(0.0100, 0.30)),
    "voisin  tp=1.50 (plateau -1 cran)": dict(
        up_tp=0.0150, up_gap=0.0100, up_activate=0.0030, up_prop=(0.0100, 0.30)),
    "voisin  gap=1.10": dict(
        up_tp=0.0155, up_gap=0.0110, up_activate=0.0030, up_prop=(0.0100, 0.30)),
    "voisin  arm=0.85 frac=0.25": dict(
        up_tp=0.0155, up_gap=0.0100, up_activate=0.0030, up_prop=(0.0085, 0.25)),
    "CONTROLE meme TP/gap SANS cliquet (isole l'apport du cliquet)": dict(
        up_tp=0.0155, up_gap=0.0100, up_activate=0.0030, up_prop=None),
}


def main():
    prepared, regime_map = H.load_prepared()
    split = H.split_point(prepared)
    base, base_tr = H.evaluate(prepared, regime_map, split)
    print(f"BASELINE n={base['n']} net={base['net']:.4f}% ev={base['ev']:.4f}% "
          f"worst={base['worst']:.2f}% dd={base['dd']:.2f}% "
          f"| h1={base['h1_net']:+.2f} h2={base['h2_net']:+.2f}")
    if not CANDIDATS:
        print("\n(aucun candidat defini -- edite CANDIDATS)")
        return
    res = [check(n, p, prepared, regime_map, split, base, base_tr)
           for n, p in CANDIDATS.items()]
    print(f"\n{'='*78}\nRECAP\n{'='*78}")
    df = pd.DataFrame(res)[["name", "n", "net", "ev", "wr", "worst", "dd", "all_ok"]]
    print(df.sort_values("net", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
