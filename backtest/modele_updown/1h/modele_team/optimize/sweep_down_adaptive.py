"""sweep_down_adaptive.py — PHASE 2 (campagne ADAPTATIVITE DOWN 2026-08-11).

Hypotheses falsifiables issues du diagnostic (diag_down_adaptive.py) :
  H2a. vol-trigger    : pic>=ARM et vol_ratio>=V        -> gap resserre a TIGHT
  H2b. volume-trigger : pic>=ARM et adv_volshare>=A     -> idem
  H2c. combine OR/AND des deux
  H1.  gap normalise ATR : gap = clip(m * atr60_entry, lo, hi) (adaptatif
       inter-trade, pas intra-trade -- le controle "pro" de base)
  H3.  stall : pic>=ARM, aucun nouveau pic depuis M min, pp>0 -> sortie

Le plancher resserre est MONOTONE (max avec le plancher precedent) : une fois
resserre, jamais re-elargi. SL temporel, TP, hold, widen : STRICTEMENT inchanges.
Implemente par MONKEYPATCH de main._down_managed_exit -- main.py n'est pas touche
tant qu'aucune hypothese n'a gagne (convention optimize/ = read-only).

Neutralite : mode "vol" avec arm inatteignable (9.9) doit reproduire la baseline
bit-a-bit -- verifie par assert avant tout sweep (le walk reecrit est exerce).

Usage : cd modele_team && python optimize/sweep_down_adaptive.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _harness as H
import main as M
import config_main as C

UP_RETENU = dict(up_tp=0.0155, up_gap=0.0100, up_activate=0.0030, up_prop=(0.0100, 0.30))

# ---------- features 1-min precalculees (vectorise, zero lookahead) ----------
# A la minute j on decide sur le CLOSE de j (minute terminee) -> les fenetres
# rolling INCLUENT la minute courante, rien au-dela.
df1 = pd.read_parquet(H.ROOT / "DB_1m" / "btc_1m.parquet")
ts1 = df1.timestamp_ms.to_numpy("int64")
o1 = df1.open.to_numpy(float); h1 = df1.high.to_numpy(float)
l1 = df1.low.to_numpy(float); c1 = df1.close.to_numpy(float); v1 = df1.volume.to_numpy(float)
IDX = {int(t): i for i, t in enumerate(ts1)}
lr = pd.Series(np.concatenate(([0.0], np.diff(np.log(np.maximum(c1, 1e-9))))))
VOL15 = lr.rolling(15).std().to_numpy()
VOL60 = lr.rolling(60).std().to_numpy()
adv = (c1 > o1).astype(float) * v1
ADV15 = pd.Series(adv).rolling(15).sum().to_numpy() / np.maximum(
    pd.Series(v1).rolling(15).sum().to_numpy(), 1e-12)
ATR60 = pd.Series((h1 - l1) / np.maximum(c1, 1e-9)).rolling(60).mean().to_numpy()

# ---------- parametres adaptatifs (module-level, balayes) ----------
P = dict(mode="off", arm=0.008, tight=0.0035, vthr=1.3, athr=0.55, stall_min=60,
         atr_mult=1.0, atr_lo=0.006, atr_hi=0.025)

_orig = M._down_managed_exit


def adaptive_down_exit(prices_intrabar, traded_ts, max_steps=None, trail_gap=None,
                       widen_candidates=None):
    if P["mode"] == "off":
        return _orig(prices_intrabar, traded_ts, max_steps, trail_gap, widen_candidates)
    sl_max_steps = max_steps if max_steps is not None else C.DOWN_MAX_STEPS
    if max_steps is None:
        max_steps = C.DOWN_MAX_STEPS
    if trail_gap is None:
        trail_gap = C.DOWN_TRAIL_GAP
    take_profit = C.DOWN_TAKE_PROFIT
    i_ent = IDX.get(traded_ts)
    if i_ent is None:
        return None, "close", traded_ts + M.INTERVAL_1H_MS, None
    entry = o1[i_ent]
    if P["mode"] == "atr":                     # H1 : gap fixe par trade, normalise ATR
        trail_gap = float(np.clip(P["atr_mult"] * ATR60[i_ent], P["atr_lo"], P["atr_hi"]))
    vol_ent = VOL60[i_ent]
    peak = -9.0; floor = None; last = None; last_j = 0
    j_peak = 0; widened_at = None; j = 0
    while j < max_steps:
        i = i_ent + j
        if i >= len(c1) or ts1[i] != traded_ts + j * 60_000:
            break
        pp = 1.0 - c1[i] / entry
        last = pp; last_j = j
        t = traded_ts + j * 60_000
        if (C.WIDEN_ON_PROFIT_ENABLED and widened_at is None and widen_candidates
                and pp > 0.0 and t in widen_candidates):
            take_profit = C.DOWN_TAKE_PROFIT * C.DOWN_WIDEN_TP_MULT
            trail_gap = trail_gap * C.DOWN_WIDEN_TRAIL_GAP_MULT
            max_steps = int(max_steps * C.DOWN_WIDEN_MAX_HOLD_MULT)
            widened_at = t
        sl_j = min(j, sl_max_steps - 1)
        sl_th = M._sl_threshold(sl_j, sl_max_steps, C.DOWN_SL_START_DEC,
                                C.DOWN_SL_END_DEC, C.DOWN_SL_FLOOR_DEC)
        if pp <= sl_th:
            return pp, "stop_loss", t, widened_at
        if pp >= take_profit:
            return pp, "take_profit", t, widened_at
        if pp > peak:
            peak = pp; j_peak = j
        gap_now = trail_gap
        if P["mode"] == "atr_live":            # H1b : gap re-normalise CHAQUE minute
            gap_now = float(np.clip(P["atr_mult"] * ATR60[i], P["atr_lo"], P["atr_hi"]))
        if P["mode"] in ("vol", "volu", "or", "and") and peak >= P["arm"]:
            vr = VOL15[i] / vol_ent if (vol_ent and vol_ent > 0) else 0.0
            va = ADV15[i]
            fire = {"vol": vr >= P["vthr"], "volu": va >= P["athr"],
                    "or": vr >= P["vthr"] or va >= P["athr"],
                    "and": vr >= P["vthr"] and va >= P["athr"]}[P["mode"]]
            if fire:
                gap_now = min(gap_now, P["tight"])
        if P["mode"] == "stall" and peak >= P["arm"] and (j - j_peak) >= P["stall_min"] and pp > 0:
            return pp, "trailing", t, widened_at
        cand = peak - gap_now
        if floor is None or cand > floor:      # plancher MONOTONE
            floor = cand
        if peak >= C.DOWN_TRAIL_ACTIVATE and floor is not None and pp < floor:
            return pp, "trailing", t, widened_at
        j += 1
    if last is None:
        return None, "close", traded_ts + M.INTERVAL_1H_MS, widened_at
    return last, "timestop", traded_ts + last_j * 60_000, widened_at


M._down_managed_exit = adaptive_down_exit

# ---------- run (partie sweep, executee seulement en script) ----------
def _sweep_main():
    prepared, regime_map = H.load_prepared()
    split = H.split_point(prepared)
    P["mode"] = "off"
    base, _ = H.evaluate(prepared, regime_map, split, **UP_RETENU)
    b = dict(net=base["down_net"], h1n=base["h1_down_net"], h2n=base["h2_down_net"],
             h1e=base["h1_down_ev"], h2e=base["h2_down_ev"])
    print(f"BASELINE DOWN net={b['net']:.4f}% h1={b['h1n']:+.2f} h2={b['h2n']:+.2f}")
    P["mode"] = "vol"; P["arm"] = 9.9
    m, _ = H.evaluate(prepared, regime_map, split, **UP_RETENU)
    assert abs(m["down_net"] - b["net"]) < 1e-6, "NEUTRALITE ECHOUEE"
    print("  neutralite du walk adaptatif : OK")

    rows = []

    def run(tag, **kw):
        for k, v in kw.items():
            P[k] = v
        m, _ = H.evaluate(prepared, regime_map, split, **UP_RETENU)
        rows.append(dict(tag=tag, **{k: P[k] for k in ("mode", "arm", "tight", "vthr",
                                                       "athr", "stall_min", "atr_mult")},
                         n=m["down_n"], net=m["down_net"], ev=m["down_ev"], wr=m["down_wr"],
                         h1=m["h1_down_net"], h2=m["h2_down_net"],
                         h1e=m["h1_down_ev"], h2e=m["h2_down_ev"],
                         tot_n=m["n"], tot_net=m["net"], worst=m["worst"], dd=m["dd"]))

    for arm in (0.006, 0.008, 0.010):
        for tight in (0.002, 0.0035, 0.005):
            for v in (1.2, 1.5, 2.0):
                run("H2a_vol", mode="vol", arm=arm, tight=tight, vthr=v)
            for a in (0.55, 0.65):
                run("H2b_volu", mode="volu", arm=arm, tight=tight, athr=a)
            for v, a in ((1.2, 0.55), (1.5, 0.55), (1.5, 0.65), (2.0, 0.65)):
                run("H2c_and", mode="and", arm=arm, tight=tight, vthr=v, athr=a)
                run("H2c_or", mode="or", arm=arm, tight=tight, vthr=v, athr=a)
    for mmul in (0.5, 0.75, 1.0, 1.25, 1.5):
        run("H1_atr", mode="atr", atr_mult=mmul)
    for arm in (0.006, 0.008, 0.010):
        for sm in (30, 60, 120):
            run("H3_stall", mode="stall", arm=arm, stall_min=sm)

    df = pd.DataFrame(rows)
    df["ok"] = ((df.net > b["net"]) & (df.h1 > b["h1n"]) & (df.h2 > b["h2n"])
                & (df.h1e > b["h1e"]) & (df.h2e > b["h2e"])
                & (df.tot_n >= 240) & (df.worst >= -15) & (df.dd >= -20))
    df.to_csv(H.ROOT / "logs" / "sweep_down_adaptive.csv", index=False)
    print(f"{len(df)} configs, {int(df.ok.sum())} VALIDES")
    cc = ["tag", "arm", "tight", "vthr", "athr", "stall_min", "atr_mult",
          "n", "net", "ev", "wr", "h1", "h2", "tot_net", "dd"]
    print(df.sort_values("net", ascending=False).head(12)[cc].round(3).to_string(index=False))
    if df.ok.sum():
        print(df[df.ok].sort_values("net", ascending=False)[cc].round(3).to_string(index=False))


if __name__ == "__main__":
    _sweep_main()
