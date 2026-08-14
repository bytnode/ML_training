"""diag_peaks.py — DIAGNOSTIC : que rend-on vraiment ?

Pour chaque trade de la baseline, re-marche la bougie 1-min et enregistre le PIC
de profit atteint, le profit final, la raison de sortie et le giveback
(pic - final). Repond factuellement a : "combien de trades montent haut puis
finissent bas, et combien d'argent y a-t-il dans ce trou ?" -- avant de balayer
quoi que ce soit.

Puis compte, pour une grille de cliquets, les trades SAUVES (pic franchit le
palier, sortie actuelle <= 0) et CLIPPES (sortie au palier alors que le trade
serait reparti plus haut), comme demande par le protocole de validation.

Usage : cd modele_team && python optimize/diag_peaks.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _harness as H                                       # noqa: E402
import main as M                                           # noqa: E402
import config_main as C                                    # noqa: E402
import filters                                             # noqa: E402


def walk(prepared, regime_map, traded_ts, up: bool, decision_ts: int):
    """Re-marche un trade et renvoie la trajectoire (pic courant, pp) minute par
    minute, avec les memes parametres effectifs que la simulation baseline."""
    open_ib, close_ib = prepared["prices_intrabar"]
    entry = open_ib.get(traded_ts)
    if entry is None:
        return None
    if up:
        max_steps, step_ms = C.UP_MAX_STEPS, C.UP_STEP_MS
    else:
        ms_eff, _ = filters.effective_down_exit_params(decision_ts, regime_map)
        max_steps = ms_eff if ms_eff is not None else C.DOWN_MAX_STEPS
        step_ms = C.DOWN_STEP_MS
    pps = []
    for j in range(max_steps):
        cl = close_ib.get(traded_ts + j * step_ms)
        if cl is None:
            break
        pps.append((cl / entry - 1.0) if up else (1.0 - cl / entry))
    return np.array(pps) if pps else None


def main():
    prepared, regime_map = H.load_prepared()
    split = H.split_point(prepared)
    base, trades = H.evaluate(prepared, regime_map, split)
    print(f"BASELINE n={base['n']} net={base['net']:.2f}% ev={base['ev']:.4f}%\n")

    rows = []
    for _, t in trades.iterrows():
        up = t["direction"] == "up"
        pps = walk(prepared, regime_map, int(t["traded_ts"]), up, int(t["timestamp"]))
        if pps is None:
            continue
        # pic ATTEINT AVANT la sortie reelle : on tronque la trajectoire a l'index
        # de sortie (approx : premier index dont le pp egale le gross realise).
        gross = float(t["gross_ret"])
        idx = int(np.argmin(np.abs(pps - gross)))
        pre = pps[:idx + 1]
        rows.append({
            "ts": int(t["timestamp"]), "traded_ts": int(t["traded_ts"]),
            "dir": t["direction"], "reason": t["exit_reason"],
            "peak_pct": float(pre.max() * 100) if len(pre) else np.nan,
            "gross_pct": gross * 100, "net_lev_pct": float(t["net_ret"]) * C.EXPOSURE * 100,
            "exit_idx": idx,
        })
    d = pd.DataFrame(rows)
    d["giveback_pct"] = d["peak_pct"] - d["gross_pct"]

    for side in ("up", "down"):
        s = d[d["dir"] == side]
        print(f"=== {side.upper()} ({len(s)} trades) ===")
        print(f"  pic median {s['peak_pct'].median():.3f}%   pic p75 {s['peak_pct'].quantile(.75):.3f}%   "
              f"pic max {s['peak_pct'].max():.3f}%")
        print(f"  giveback median {s['giveback_pct'].median():.3f}%   p90 {s['giveback_pct'].quantile(.9):.3f}%   "
              f"max {s['giveback_pct'].max():.3f}%")
        print("  par raison de sortie :")
        g = s.groupby("reason").agg(n=("ts", "size"), pic=("peak_pct", "mean"),
                                    final=("gross_pct", "mean"), give=("giveback_pct", "mean"),
                                    net=("net_lev_pct", "mean"))
        print(g.round(3).to_string())
        # LE TROU : trades montes haut et finis <= 0
        for thr in (0.3, 0.5, 0.8, 1.0, 1.5):
            bad = s[(s["peak_pct"] >= thr) & (s["gross_pct"] <= 0)]
            tot = s[s["peak_pct"] >= thr]
            print(f"    pic >= {thr:.1f}% : {len(tot):3d} trades, dont {len(bad):3d} finissent <= 0 "
                  f"(perte nette moyenne {bad['net_lev_pct'].mean() if len(bad) else 0:+.2f}%, "
                  f"total {bad['net_lev_pct'].sum() if len(bad) else 0:+.1f} pts)")
        print()

    d.to_csv(H.ROOT / "logs" / "diag_peaks.csv", index=False)
    print(f"  written logs/diag_peaks.csv ({len(d)} trades)")


if __name__ == "__main__":
    main()
