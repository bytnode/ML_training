# -*- coding: utf-8 -*-
"""exp_governor2.py — gouverneur d'equity EXACT (2026-08-11, campagne POINTS-FAIBLES).

Version corrigee de exp_governor.py : un resultat de trade n'entre dans le
livre-fantome QU'A SA CLOTURE (exit_ts <= entree du trade courant) — la v1
comptait des trades encore ouverts (micro-lookahead). Fantome = TOUS les
candidats du systeme (pris ou sautes par le gouverneur) ; le gouverneur ne
decide que l'execution REELLE. Applique aux trades du meilleur combo
(logs/exp_combo_best_trades.csv, colonne exit_ts).

Usage : cd modele_team && python optimize/exp_governor2.py
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
EXPOSURE = 5.0
DAY = 86400_000

tr = pd.read_csv(ROOT / "logs" / "exp_combo_best_trades.csv").sort_values("traded_ts").reset_index(drop=True)
net = tr["net_ret"].to_numpy(float) * EXPOSURE
ts = tr["traded_ts"].to_numpy("int64")
xts = tr["exit_ts"].to_numpy("int64")
Y1_END = ts.min() + 365 * DAY
N = len(net)
print(f"combo : n={N}  a1={(np.prod(1+net[ts<Y1_END])-1)*100:+.2f}%  "
      f"a2={(np.prod(1+net[ts>=Y1_END])-1)*100:+.2f}%  net={(np.prod(1+net)-1)*100:+.2f}%", flush=True)


def stats(mask_taken):
    eff = np.where(mask_taken, net, 0.0)
    eq = np.cumprod(1 + eff)
    peak = np.maximum.accumulate(eq)
    dd = float(((eq - peak) / peak).min() * 100)
    a1 = (np.prod(1 + eff[ts < Y1_END]) - 1) * 100
    a2 = (np.prod(1 + eff[ts >= Y1_END]) - 1) * 100
    worst_w = 1e9
    t0 = ts.min()
    while t0 + 365 * DAY <= ts.max() + DAY:
        m = (ts >= t0) & (ts < t0 + 365 * DAY)
        worst_w = min(worst_w, (np.prod(1 + eff[m]) - 1) * 100)
        t0 += 30 * DAY
    return a1, a2, (eq[-1] - 1) * 100, dd, worst_w, int(mask_taken.sum())


rows = []
for K in (20, 30, 40):
    for skip_thr in (-5.0, -8.0, -10.0, -15.0):
        for resume_thr in (skip_thr, skip_thr / 2, 0.0):
            taken = np.ones(N, dtype=bool)
            skipping = False
            for i in range(N):
                # fantome : les K derniers candidats DEJA CLOS a l'entree du trade i
                closed = np.where(xts[:i] <= ts[i])[0]
                roll = net[closed[-K:]].sum() * 100 if len(closed) else 0.0
                if not skipping and roll < skip_thr:
                    skipping = True
                elif skipping and roll >= resume_thr:
                    skipping = False
                if skipping:
                    taken[i] = False
            a1, a2, tot, dd, ww, n_taken = stats(taken)
            rows.append(dict(K=K, skip=skip_thr, resume=resume_thr, a1=a1, a2=a2,
                             tot=tot, dd=dd, min365=ww, n=n_taken))

df = pd.DataFrame(rows)
df.to_csv(ROOT / "logs" / "exp_governor2.csv", index=False)
print(df.round(2).to_string(index=False), flush=True)
print("\nTOP 10 par dd (dd le moins profond d'abord) :", flush=True)
print(df.sort_values(["dd", "tot"], ascending=False).head(10).round(2).to_string(index=False), flush=True)
print("-> logs/exp_governor2.csv", flush=True)
