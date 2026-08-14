"""sweep_down_tpext.py — EXTENSION CONDITIONNELLE DU TP EN BEAR (campagne
ADAPTATIVITE DOWN, iteration 4, 2026-08-11).

Diagnostic fondateur (diag_down_posttp.csv) : sur les 39 sorties take_profit a
+1.60%, le pic post-TP moyen est +2.98% ; en regime BEAR (28/39) il est +3.27%
(p75 +4.41%). Le systeme encaisse 1.6 et laisse la queue. Toutes les tentatives
de RESSERRAGE ont echoue (les sorties actuelles ne sont pas prematurees, cf
diag_down_postexit.csv) -- le gisement est de l'autre cote : LAISSER COURIR
AU-DELA DU TP quand le contexte porte le short.

Mecanisme (mode "tpext") :
  - regime a l'entree (bear par defaut) dans EXT_REGIMES et position pas widened :
    quand pp atteint TP0 (1.60%), on NE sort PAS ;
    - plancher de protection : floor >= TP0 - GUARD (le gain acquis est verrouille
      a GUARD pres) -- s'ajoute au trailing classique (max des deux), MONOTONE ;
    - nouvelle cible : TP_EXT (sortie take_profit) ;
    - le trailing classique (peak - gap) continue de courir en parallele.
  - autres regimes / trade widened : comportement EXACTEMENT baseline.
  - SL temporel, hold, widen-on-profit : STRICTEMENT inchanges.

Anti-lookahead : le regime est lu a la barre de DECISION (comme le wrapper
existant) ; aucune donnee future.

Usage : cd modele_team && python optimize/sweep_down_tpext.py
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

df1 = pd.read_parquet(H.ROOT / "DB_1m" / "btc_1m.parquet")
ts1 = df1.timestamp_ms.to_numpy("int64")
o1 = df1.open.to_numpy(float); c1 = df1.close.to_numpy(float)
h1_ = df1.high.to_numpy(float); l1_ = df1.low.to_numpy(float)
IDX = {int(t): i for i, t in enumerate(ts1)}
ATR60 = pd.Series((h1_ - l1_) / np.maximum(c1, 1e-9)).rolling(60).mean().to_numpy()
# pivots hauts confirmes (p barres de chaque cote, confirmes p barres plus tard)
_PIV = {}
for _p in (10, 20, 30):
    lvl = np.full(len(c1), np.nan); born = np.full(len(c1), -1, dtype=np.int64)
    hh = pd.Series(h1_)
    left = hh.rolling(_p).max().shift(1).to_numpy()
    right = hh.shift(-_p).rolling(_p).max().to_numpy()
    isp = (h1_ > np.nan_to_num(left, nan=-1)) & (h1_ >= np.nan_to_num(right, nan=-1))
    cur, curb = np.nan, -1
    for i in range(len(c1)):
        b = i - _p
        if b >= 0 and isp[b]:
            cur, curb = h1_[b], b
        lvl[i] = cur; born[i] = curb
    _PIV[_p] = (lvl, born)

P = dict(on=False, ext_regimes=("bear",), tp_ext=0.028, guard=0.003,
         max_mins_to_tp=None,   # (inverse: autopsie -> les grinders gagnent) TP en >= X min
         min_er=None,           # (inverse) ER <= E : chemins peu efficients = grind persistant
         flip_exit=False,       # sortie immediate si le regime quitte bear pendant l'extension
         flip_grace=0,          # minutes de grace apres l'armement avant d'ecouter le flip
         ext_mode="guard",      # sortie d'extension : guard | chand (k xATR60) | swing (pivot p) | sar
         chand_k=3.0, swing_p=20, sar_af=0.02, sar_max=0.2,
         hard_guard=0.008)      # backstop absolu : floor >= TP0 - hard_guard quel que soit le mode
REG = {}          # traded_ts -> regime (rempli au run)
REG_H = {}        # ts de barre 1h -> regime (pour le flip-exit intra-extension)

_orig = M._down_managed_exit


def tpext_down_exit(prices_intrabar, traded_ts, max_steps=None, trail_gap=None,
                    widen_candidates=None):
    if not P["on"] or REG.get(traded_ts, "neutral") not in P["ext_regimes"]:
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
    peak = -9.0; last = None; last_j = 0
    widened_at = None; extended = False; guard_floor = None
    path_len = 0.0; prev_pp = 0.0; ext_j = 0; ll = None; sar = None; ep = None; af = None
    j = 0
    while j < max_steps:
        i = i_ent + j
        if i >= len(c1) or ts1[i] != traded_ts + j * 60_000:
            break
        pp = 1.0 - c1[i] / entry
        path_len = (path_len + abs(pp - prev_pp)) if j > 0 else 0.0
        prev_pp = pp
        last = pp; last_j = j
        t = traded_ts + j * 60_000
        if (C.WIDEN_ON_PROFIT_ENABLED and widened_at is None and widen_candidates
                and pp > 0.0 and t in widen_candidates):
            take_profit = C.DOWN_TAKE_PROFIT * C.DOWN_WIDEN_TP_MULT
            trail_gap = trail_gap * C.DOWN_WIDEN_TRAIL_GAP_MULT
            max_steps = int(max_steps * C.DOWN_WIDEN_MAX_HOLD_MULT)
            widened_at = t
            extended = False; guard_floor = None   # widen prend le relais -> baseline
        sl_j = min(j, sl_max_steps - 1)
        sl_th = M._sl_threshold(sl_j, sl_max_steps, C.DOWN_SL_START_DEC,
                                C.DOWN_SL_END_DEC, C.DOWN_SL_FLOOR_DEC)
        if pp <= sl_th:
            return pp, "stop_loss", t, widened_at
        # --- extension du TP : au franchissement de TP0 (non-widened), on arme ---
        if (not extended and widened_at is None and pp >= C.DOWN_TAKE_PROFIT):
            gate = True
            if P["max_mins_to_tp"] is not None and j < P["max_mins_to_tp"]:
                gate = False                       # grinders only : TP atteint LENTEMENT
            if gate and P["min_er"] is not None:
                er = pp / max(path_len, 1e-9)
                if er > P["min_er"]:
                    gate = False                   # grinders only : ER FAIBLE
            if gate:
                extended = True
                ext_j = j
                guard_floor = C.DOWN_TAKE_PROFIT - P["guard"]
                take_profit = P["tp_ext"]
            else:
                return pp, "take_profit", t, widened_at   # gate ferme -> baseline
        # --- sortie d'extension par structure/volatilite (modes chand/swing/sar) ---
        if extended and P["ext_mode"] != "guard":
            m_floor = None
            if P["ext_mode"] == "chand":
                ll = min(ll, l1_[i]) if ll is not None else l1_[i]
                stop_px = ll + P["chand_k"] * ATR60[i] * c1[i]
                m_floor = 1.0 - stop_px / entry
            elif P["ext_mode"] == "swing":
                lvl, born = _PIV[P["swing_p"]]
                if born[i] >= i_ent + ext_j:      # pivot forme APRES l'armement
                    m_floor = 1.0 - lvl[i] / entry
            elif P["ext_mode"] == "sar":
                if sar is None:
                    sar, ep, af = entry * (1.0 - (pp - 0.002)), c1[i], P["sar_af"]
                if c1[i] < ep:
                    ep = c1[i]; af = min(af + P["sar_af"], P["sar_max"])
                sar = sar + af * (ep - sar)
                m_floor = 1.0 - sar / entry
            hard = C.DOWN_TAKE_PROFIT - P["hard_guard"]
            guard_floor = max(m_floor, hard) if m_floor is not None else hard
        # --- flip de regime pendant l'extension : le contexte qui justifiait le
        #     ride a disparu -> sortie immediate (label de la DERNIERE barre 1h
        #     CLOSE, jamais la barre en cours -> zero lookahead) ---
        if (extended and P["flip_exit"] and (j - ext_j) >= P["flip_grace"]):
            hb = (t // 3600_000) * 3600_000 - 3600_000
            if REG_H.get(hb, "neutral") != "bear":
                return pp, "trailing", t, widened_at
        if pp >= take_profit:
            return pp, "take_profit", t, widened_at
        if pp > peak:
            peak = pp
        floor = None
        if peak >= C.DOWN_TRAIL_ACTIVATE:
            floor = peak - trail_gap
        if guard_floor is not None and (floor is None or guard_floor > floor):
            floor = guard_floor
        if floor is not None and pp < floor:
            return pp, "trailing", t, widened_at
        j += 1
    if last is None:
        return None, "close", traded_ts + M.INTERVAL_1H_MS, widened_at
    return last, "timestop", traded_ts + last_j * 60_000, widened_at


M._down_managed_exit = tpext_down_exit


def _sweep_main():
    prepared, regime_map = H.load_prepared()
    split = H.split_point(prepared)
    REG.clear(); REG_H.clear()
    if regime_map:
        for ts, r in regime_map.items():
            REG[int(ts) + 3600_000] = r
            REG_H[int(ts)] = r
    P["on"] = False
    base, _ = H.evaluate(prepared, regime_map, split, **UP_RETENU)
    b = dict(net=base["down_net"], h1n=base["h1_down_net"], h2n=base["h2_down_net"],
             h1e=base["h1_down_ev"], h2e=base["h2_down_ev"])
    print(f"BASELINE DOWN net={b['net']:.4f}% h1={b['h1n']:+.2f} h2={b['h2n']:+.2f} "
          f"ev={base['down_ev']:.4f} | total {base['net']:.2f}%")
    # neutralite : on=True mais ext_regimes vide -> baseline bit-a-bit
    P["on"] = True; P["ext_regimes"] = ()
    m, _ = H.evaluate(prepared, regime_map, split, **UP_RETENU)
    assert abs(m["down_net"] - b["net"]) < 1e-6, "NEUTRALITE ECHOUEE"
    # neutralite 2 : le walk reecrit lui-meme (regimes bear mais tp_ext=TP0)
    P["ext_regimes"] = ("bear",); P["tp_ext"] = C.DOWN_TAKE_PROFIT; P["guard"] = 0.0
    m, _ = H.evaluate(prepared, regime_map, split, **UP_RETENU)
    assert abs(m["down_net"] - b["net"]) < 1e-3, f"NEUTRALITE2: {m['down_net']} vs {b['net']}"
    print("  neutralite : OK (2 tests)")

    rows = []
    for tp_ext, guard, mtt, mer, fe, fg in [
        (tp_ext, guard, mtt, mer, fe, fg)
        for tp_ext in (9.9,)
        for guard in (0.0035, 0.0050)
        for mtt in (None, 400, 550)
        for mer in (None, 0.05, 0.065, 0.08)
        for fe in (False, True)
        for fg in ((0,) if not fe else (0, 120, 240))]:
        P.update(on=True, ext_regimes=("bear",), tp_ext=tp_ext, guard=guard,
                 max_mins_to_tp=mtt, min_er=mer, flip_exit=fe, flip_grace=fg)
        if True:
                m, _ = H.evaluate(prepared, regime_map, split, **UP_RETENU)
                rows.append(dict(tp_ext=tp_ext * 100, guard=guard * 100,
                                 mtt=mtt if mtt is not None else -1, mer=mer if mer is not None else -1,
                                 fe=fe, fg=fg,
                                 n=m["down_n"], net=m["down_net"], ev=m["down_ev"], wr=m["down_wr"],
                                 h1=m["h1_down_net"], h2=m["h2_down_net"],
                                 h1e=m["h1_down_ev"], h2e=m["h2_down_ev"],
                                 tot_n=m["n"], tot_net=m["net"], worst=m["worst"], dd=m["dd"]))
    df = pd.DataFrame(rows)
    df["ok"] = ((df.net > b["net"]) & (df.h1 > b["h1n"]) & (df.h2 > b["h2n"])
                & (df.h1e > b["h1e"]) & (df.h2e > b["h2e"])
                & (df.tot_n >= 240) & (df.worst >= -15) & (df.dd >= -20))
    df.to_csv(H.ROOT / "logs" / "sweep_down_tpext.csv", index=False)
    print(f"\n{len(df)} configs, {int(df.ok.sum())} VALIDES (2 moities net+EV + contraintes)")
    cc = ["guard", "mtt", "mer", "fe", "fg", "n", "net", "ev", "wr", "h1", "h2", "h1e", "h2e", "tot_net", "dd"]
    print("\nTOP 15 par net DOWN :")
    print(df.sort_values("net", ascending=False).head(15)[cc].round(2).to_string(index=False))
    if df.ok.sum():
        print("\nVALIDES :")
        print(df[df.ok].sort_values("net", ascending=False)[cc].round(2).to_string(index=False))


if __name__ == "__main__":
    _sweep_main()
