"""sweep_security.py — CAMPAGNE SECURITE (2026-08-11, 3e goal de la journee).

Objectif user : une position montee a ~1% de profit prix ne doit plus JAMAIS
finir perdante ("pic 1.13% -> perte finale : interdit"). Tolerance des pullbacks
normaux du 1h en-dessous du seuil d'armement. Budget : net total >= 1800%
(base production : +2179.62%).

Mecanisme : le CLIQUET (filters.ratchet_floor) deja porte en production —
paliers plats (arm_pct -> lock_pct) et/ou proportionnels. Le verrou doit couvrir
les frais (0.07% A/R) : lock >= 0.10% garantit un NET positif.
Variante ADAPTATIVE AU REGIME (demande explicite) : verrou different en bear
(trend-ride a proteger -> verrou minimal, pullbacks toleres au-dessus du seuil)
vs neutral/bull (pas de tendance a rider -> verrouille plus haut).

Garantie par construction (granularite 1-min) : une fois pic >= arm, le plancher
>= lock ne redescend jamais -> la sortie se fait au-dessus de lock, MOINS le
depassement intra-minute (une bougie 1-min traversant lock ET le SL d'un coup
sort au prix de sa cloture -- cas borne par la taille des bougies 1-min).

Usage : cd modele_team && python optimize/sweep_security.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _harness as H
import main as M
import config_main as C

UP = dict(up_tp=0.0155, up_gap=0.0100, up_activate=0.0030, up_prop=(0.0100, 0.30))

df1 = pd.read_parquet(H.ROOT / "DB_1m" / "btc_1m.parquet")
ts1 = df1.timestamp_ms.to_numpy("int64")
o1 = df1.open.to_numpy(float); c1 = df1.close.to_numpy(float)
IDX = {int(t): i for i, t in enumerate(ts1)}

# --- wrapper regime-adaptatif : table {regime: steps} appliquee par trade ---
REGIME_TABLE: dict | None = None
_orig_down = M._down_managed_exit


def down_regime_ratchet(prices_intrabar, traded_ts, max_steps=None, trail_gap=None,
                        widen_candidates=None, regime="neutral", swing_pivots=None):
    if REGIME_TABLE is not None:
        M.DOWN_RATCHET = REGIME_TABLE.get(regime, ())
    return _orig_down(prices_intrabar, traded_ts, max_steps, trail_gap,
                      widen_candidates, regime, swing_pivots)


M._down_managed_exit = down_regime_ratchet


def count_violations(tr, thr):
    out = {}
    for side in ("up", "down"):
        s = tr[tr.direction == side]
        n = 0; loss = 0.0
        for _, t in s.iterrows():
            tts = int(t.traded_ts); i_ent = IDX.get(tts)
            if i_ent is None: continue
            entry = o1[i_ent]
            gross = float(t.gross_ret)
            pps = []
            for j in range(4000):
                i = i_ent + j
                if i >= len(c1) or ts1[i] != tts + j * 60_000: break
                pps.append((c1[i] / entry - 1.0) if side == "up" else (1.0 - c1[i] / entry))
            if not pps: continue
            pps = np.array(pps)
            exit_j = int(np.argmin(np.abs(pps - gross)))
            if float(pps[:exit_j + 1].max()) * 100 >= thr and float(t.net_ret) < 0:
                n += 1; loss += float(t.net_ret) * C.EXPOSURE * 100
        out[side] = (n, loss)
    return out


def main():
    prepared, regime_map = H.load_prepared()
    split = H.split_point(prepared)
    global REGIME_TABLE
    REGIME_TABLE = None
    b, tr = H.evaluate(prepared, regime_map, split, **UP)
    v = count_violations(tr, 1.0)
    print(f"BASE net={b['net']:.2f}% (UP {b['up_net']:.2f} / DOWN {b['down_net']:.2f}) "
          f"viol>=1.0 : up {v['up'][0]}, down {v['down'][0]} ({v['down'][1]:+.2f} pts)")

    rows = []

    def run(tag, up_steps=(), down_steps=(), table=None):
        global REGIME_TABLE
        REGIME_TABLE = table
        extra = dict(UP)
        if up_steps:
            extra["up_ratchet"] = tuple((a / 100, l / 100) for a, l in up_steps)
        if down_steps and table is None:
            extra["down_ratchet"] = tuple((a / 100, l / 100) for a, l in down_steps)
        m, tr = H.evaluate(prepared, regime_map, split, **extra)
        v10 = count_violations(tr, 1.0); v08 = count_violations(tr, 0.8)
        rows.append(dict(tag=tag,
                         up_v10=v10["up"][0], dn_v10=v10["down"][0],
                         up_v08=v08["up"][0], dn_v08=v08["down"][0],
                         net=m["net"], up_net=m["up_net"], dn_net=m["down_net"],
                         h1=m["h1_net"], h2=m["h2_net"], worst=m["worst"],
                         dd=m["dd"], n=m["n"]))
        REGIME_TABLE = None

    # --- DOWN : paliers plats (arm x lock) ---
    for arm in (0.8, 0.9, 1.0, 1.1):
        for lock in (0.10, 0.15, 0.25, 0.40):
            run(f"dn_flat_{arm}/{lock}", down_steps=((arm, lock),))
    # --- DOWN : deux paliers (securisation progressive) ---
    run("dn_2steps_1.0/0.10+1.3/0.40", down_steps=((1.0, 0.10), (1.3, 0.40)))
    run("dn_2steps_0.9/0.10+1.2/0.30", down_steps=((0.9, 0.10), (1.2, 0.30)))
    # --- DOWN : adaptatif au regime (bear tolere, neutral/bull verrouille haut) ---
    for bl in (0.10, 0.15):
        for nl in (0.25, 0.40):
            for na in (0.8, 1.0):
                tbl = {"bear": ((1.0 / 100, bl / 100),),
                       "neutral": ((na / 100, nl / 100),),
                       "bull": ((na / 100, nl / 100),)}
                run(f"dn_reg_b1.0/{bl}_nb{na}/{nl}", table=tbl)
    # --- UP : palier supplementaire sous le cliquet prop ---
    for arm in (0.7, 0.8):
        for lock in (0.05, 0.10):
            run(f"up_flat_{arm}/{lock}", up_steps=((arm, lock),),
                down_steps=((1.0, 0.10),))

    df = pd.DataFrame(rows)
    df.to_csv(H.ROOT / "logs" / "sweep_security.csv", index=False)
    df["secure"] = (df.dn_v10 == 0) & (df.up_v10 == 0)
    df["okbudget"] = df.net >= 1800.0
    print(f"\n{len(df)} configs — secure (0 violation >=1.0) : {int(df.secure.sum())}, "
          f"dans le budget >=1800 : {int((df.secure & df.okbudget).sum())}")
    cc = ["tag", "up_v10", "dn_v10", "dn_v08", "net", "up_net", "dn_net", "h1", "h2", "worst", "dd"]
    print("\nTOP 20 par net (avec drapeaux) :")
    print(df.sort_values("net", ascending=False).head(20)[cc + ["secure"]].round(2).to_string(index=False))
    ok = df[df.secure & df.okbudget]
    if len(ok):
        print("\nSECURE + BUDGET :")
        print(ok.sort_values("net", ascending=False)[cc].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
