# -*- coding: utf-8 -*-
"""test_720.py — test des configs de paliers sur 720 jours (demande user 2026-08-11).

MEMES modeles qu'en production (aucun re-entrainement) : l'api 1h predit a la
volee depuis le parquet features. ⚠️ CONTAMINATION ASSUMEE (decision user) : le
segment 2024-07 -> 2025-07 etait DANS la fenetre d'entrainement des modeles ->
WR optimiste sur ce segment. La coupure IS/OOS est rapportee pour chaque config :
  h1 = in-sample  (2024-07-14 -> 2025-07-03, ~355j, optimiste)
  h2 = out-of-sample (2025-07-03 -> 2026-07-03, = la fenetre 365j de la session)

Usage : cd modele_team && python optimize/test_720.py
"""
from __future__ import annotations
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _harness as H
import main as M
import config_main as C

UP = dict(up_tp=0.0155, up_gap=0.0100, up_activate=0.0030, up_prop=(0.0100, 0.30))
# Cache pickle : ecrit/relu par CE script uniquement, contenu = DataFrames issus
# de nos propres modeles (aucune source externe) -> surface de deserialisation sure.
CACHE720 = H.ROOT / "optimize" / ".cache_prepared_720.pkl"

# --- prepared 720j (cache dedie, ne touche pas celui du 365j) ---
if CACHE720.exists():
    with open(CACHE720, "rb") as f:
        prepared, regime_map = pickle.load(f)
    print("cache 720j charge")
else:
    t0 = time.time()
    M.TEAM_TEST_DAYS = 720
    prepared = M._prepare_predictions()
    import filters
    regime_map = filters.load_regime_map() if M.cfg_regime.REGIME_WRAPPER_ENABLED else None
    with open(CACHE720, "wb") as f:
        pickle.dump((prepared, regime_map), f)
    print(f"prepare 720j : {time.time()-t0:.0f}s")
print(f"fenetre : {pd.Timestamp(prepared['start_ms'], unit='ms')} -> "
      f"{pd.Timestamp(prepared['end_ms'], unit='ms')}")

M._SWING_PIVOTS = None          # force la reconstruction des pivots sur le 1-min etendu
SPLIT = prepared["end_ms"] - 365 * 86400_000   # frontiere IS / OOS (= debut fenetre session)

# --- arrays 1-min pour le comptage des violations ---
df1 = pd.read_parquet(H.ROOT / "DB_1m" / "btc_1m.parquet")
ts1 = df1.timestamp_ms.to_numpy("int64")
o1 = df1.open.to_numpy(float); c1 = df1.close.to_numpy(float)
IDX = {int(t): i for i, t in enumerate(ts1)}


def viols(tr, thr=1.0):
    n, loss = 0, 0.0
    for _, t in tr[tr.direction == "down"].iterrows():
        tts = int(t.traded_ts); i_ent = IDX.get(tts)
        if i_ent is None: continue
        entry = o1[i_ent]; gross = float(t.gross_ret)
        pps = []
        for j in range(4000):
            i = i_ent + j
            if i >= len(c1) or ts1[i] != tts + j * 60_000: break
            pps.append(1.0 - c1[i] / entry)
        if not pps: continue
        pps = np.array(pps)
        e = int(np.argmin(np.abs(pps - gross)))
        if float(pps[:e + 1].max()) * 100 >= thr and float(t.net_ret) < 0:
            n += 1; loss += float(t.net_ret) * C.EXPOSURE * 100
    return n, loss


CONFIGS = [
    ("reference (sans verrou)",            None),
    ("1 marche (1.0 -> 0.25)",             ((0.010, 0.0025),)),
    ("1 marche (1.1 -> 0.25)  [retenue]",  ((0.011, 0.0025),)),
    ("2 marches (1.0->0.10 + 1.3->0.40)",  ((0.010, 0.0010), (0.013, 0.0040))),
    ("2 marches (0.9->0.10 + 1.2->0.30)",  ((0.009, 0.0010), (0.012, 0.0030))),
]

rows = []
for name, steps in CONFIGS:
    kw = dict(UP)
    if steps:
        kw["down_ratchet"] = steps
    t0 = time.time()
    m, tr = H.evaluate(prepared, regime_map, SPLIT, **kw)
    v, vl = viols(tr, 1.0)
    rows.append(dict(config=name, n=m["n"], net=m["net"], dn=m["down_net"],
                     up=m["up_net"], viol=v, viol_pts=round(vl, 1),
                     is_net=m["h1_net"], oos_net=m["h2_net"],
                     worst=m["worst"], dd=m["dd"]))
    print(f"  {name}: {time.time()-t0:.0f}s  net={m['net']:.0f}%  viol={v}")

df = pd.DataFrame(rows)
df.to_csv(H.ROOT / "logs" / "test_720.csv", index=False)
print("\n=== RESULTATS 720 JOURS (h1 = IN-SAMPLE 355j optimiste / h2 = OOS 365j) ===")
print(df.round(2).to_string(index=False))
