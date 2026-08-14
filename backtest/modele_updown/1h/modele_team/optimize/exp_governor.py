# -*- coding: utf-8 -*-
"""exp_governor.py — campagne POINTS-FAIBLES (2026-08-11) : gouverneur d'equity.

Diagnostic : TOUT le max DD (-57.9%) est UN segment (2024-11-27 -> 2025-06-17).
Mecanisme teste : LIVRE-FANTOME sans lookahead — on suit la somme glissante des
K derniers resultats de TOUS les signaux (pris ou sautes, le fantome continue de
tout simuler) ; quand elle passe sous skip_thr, on SAUTE les trades reels ;
on reprend quand elle repasse au-dessus de resume_thr. Aucune information
future : la decision au trade i n'utilise que les resultats des trades < i.

APPROXIMATION documentee : les resultats par trade sont ceux du run champion
(logs/trades.csv) — sauter un trade libere en realite un slot de concurrence
(trades en plus) et change les widen ; ignore ici. Si gagnant -> implementation
exacte dans simulate_trades.

Usage : cd modele_team && python optimize/exp_governor.py
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
EXPOSURE = 5.0
DAY = 86400_000

tr = pd.read_csv(ROOT / "logs" / "trades.csv").sort_values("traded_ts").reset_index(drop=True)
net = tr["net_ret"].to_numpy(float) * EXPOSURE
ts = tr["traded_ts"].to_numpy("int64")
Y1_END = ts.min() + 365 * DAY
N = len(net)
print(f"champion : n={N}  a1={(np.prod(1+net[ts<Y1_END])-1)*100:+.2f}%  "
      f"a2={(np.prod(1+net[ts>=Y1_END])-1)*100:+.2f}%  net={(np.prod(1+net)-1)*100:+.2f}%", flush=True)


def stats(mask_taken):
    n_taken = int(mask_taken.sum())
    eff = np.where(mask_taken, net, 0.0)
    eq = np.cumprod(1 + eff)
    peak = np.maximum.accumulate(eq)
    dd = float(((eq - peak) / peak).min() * 100)
    a1 = (np.prod(1 + eff[ts < Y1_END]) - 1) * 100
    a2 = (np.prod(1 + eff[ts >= Y1_END]) - 1) * 100
    tot = (eq[-1] - 1) * 100
    # min fenetre 365j glissante (pas 30j)
    worst_w = 1e9
    t0 = ts.min()
    while t0 + 365 * DAY <= ts.max() + DAY:
        m = (ts >= t0) & (ts < t0 + 365 * DAY)
        worst_w = min(worst_w, (np.prod(1 + eff[m]) - 1) * 100)
        t0 += 30 * DAY
    return a1, a2, tot, dd, worst_w, n_taken


rows = []
print("\n=== grille K x skip_thr x resume_thr (pts d'equity, levier inclus) ===", flush=True)
for K in (10, 15, 20, 30):
    for skip_thr in (-5.0, -8.0, -10.0, -15.0):
        for resume_thr in (skip_thr, skip_thr / 2, 0.0):
            taken = np.ones(N, dtype=bool)
            skipping = False
            for i in range(N):
                lo = max(0, i - K)
                roll = net[lo:i].sum() * 100     # fantome : TOUS les signaux passes
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
df.to_csv(ROOT / "logs" / "exp_governor.csv", index=False)
print(df.round(2).to_string(index=False), flush=True)
print("\nTOP 8 par dd (puis a1) :", flush=True)
print(df.sort_values(["dd", "a1"], ascending=False).head(8).round(2).to_string(index=False), flush=True)
print("\nTOP 8 par a1 (dd > -45) :", flush=True)
print(df[df.dd > -45].sort_values("a1", ascending=False).head(8).round(2).to_string(index=False), flush=True)
print("\n-> logs/exp_governor.csv", flush=True)
