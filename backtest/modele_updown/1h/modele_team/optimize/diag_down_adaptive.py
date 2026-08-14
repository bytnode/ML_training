"""diag_down_adaptive.py — PHASE 1 (campagne ADAPTATIVITE DOWN 2026-08-11).

Pour chaque trade DOWN de la config retenue :
  - trajectoire minute par minute (pp, pic courant) avec les params EFFECTIFS
    (regime-ajustes : bear -> hold x2, gap x1.3)
  - MFE / MAE / P&L realise / MFE capture ratio
  - EVENT STUDY : au premier instant ou (pic >= PEAK_MIN et retrace >= RETRACE_FRAC
    du pic), snapshot de signaux SANS LOOKAHEAD :
      vol_ratio   = vol realisee 15min / vol realisee 60min pre-entree
      adv_volshare= part du volume des 15 dernieres minutes portee par les minutes
                    ADVERSES (close>open pour un short)
      retrace_spd = ampleur du retrace / minutes ecoulees depuis le pic
      mins_since_peak, peak_at_event, pp_at_event
    Label : recovered = le trade refait un NOUVEAU pic apres l'event ; died sinon.
Usage : cd modele_team && python optimize/diag_down_adaptive.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _harness as H
import config_main as C
import filters

PEAK_MIN = 0.006        # event arme seulement si pic >= 0.6% (zone "gain concret")
RETRACE_FRAC = 0.40     # event = retrace de 40% du pic

UP_RETENU = dict(up_tp=0.0155, up_gap=0.0100, up_activate=0.0030, up_prop=(0.0100, 0.30))

prepared, regime_map = H.load_prepared()
split = H.split_point(prepared)
base, trades = H.evaluate(prepared, regime_map, split, **UP_RETENU)
dn = trades[trades.direction == "down"].reset_index(drop=True)
print(f"config retenue: total net={base['net']:.2f}% | DOWN n={len(dn)} net={base['down_net']:.2f}%")

# --- donnees 1-min completes (volume inclus) ---
df1 = pd.read_parquet(H.ROOT / "DB_1m" / "btc_1m.parquet")
ts1 = df1.timestamp_ms.to_numpy("int64")
o1, c1, v1 = (df1[k].to_numpy(float) for k in ("open", "close", "volume"))
idx_of = {int(t): i for i, t in enumerate(ts1)}
logret = np.zeros(len(c1)); logret[1:] = np.diff(np.log(np.maximum(c1, 1e-9)))

def rvol(i0, i1):
    """std des log-rendements 1-min sur [i0, i1) — vol realisee locale."""
    if i1 - i0 < 5: return np.nan
    return float(np.std(logret[i0:i1]))

rows, events = [], []
for _, t in dn.iterrows():
    tts = int(t.traded_ts)
    i_ent = idx_of.get(tts)
    if i_ent is None: continue
    entry = o1[i_ent]
    ms_eff, gap_eff = filters.effective_down_exit_params(int(t.timestamp), regime_map)
    max_steps = ms_eff if ms_eff is not None else C.DOWN_MAX_STEPS
    vol_entry = rvol(max(0, i_ent - 60), i_ent)
    # trajectoire jusqu'a la sortie REELLE (approx = gross realise atteint)
    gross = float(t.gross_ret)
    pps = []
    for j in range(max_steps):
        i = i_ent + j
        if i >= len(c1) or ts1[i] != tts + j * 60_000: break
        pps.append(1.0 - c1[i] / entry)
    if not pps: continue
    pps = np.array(pps)
    exit_j = int(np.argmin(np.abs(pps - gross)))
    traj = pps[:exit_j + 1]
    peaks = np.maximum.accumulate(traj)
    mfe, mae = float(traj.max()), float(traj.min())
    capture = gross / mfe if mfe > 1e-9 else np.nan
    rows.append(dict(ts=int(t.timestamp), reason=t.exit_reason, gross=gross * 100,
                     mfe=mfe * 100, mae=mae * 100, capture=capture,
                     hold_min=exit_j, regime=(regime_map or {}).get(int(t.timestamp), "neutral")))
    # --- event study ---
    armed = (peaks >= PEAK_MIN) & (traj <= peaks * (1 - RETRACE_FRAC))
    if armed.any():
        e = int(np.argmax(armed))          # premier event
        i_e = i_ent + e
        pk = float(peaks[e])
        j_pk = int(np.argmax(traj[:e + 1] >= pk - 1e-12))
        mins_since = e - j_pk
        vol_now = rvol(max(0, i_e - 15), i_e)
        seg = slice(max(0, i_e - 15), i_e)
        adv = c1[seg] > o1[seg]            # minute adverse pour un short
        vtot = v1[seg].sum()
        adv_share = float(v1[seg][adv].sum() / vtot) if vtot > 0 else np.nan
        future_new_peak = bool((pps[e + 1:] > pk + 1e-6).any()) if e + 1 < len(pps) else False
        events.append(dict(ts=int(t.timestamp), peak=pk * 100, pp=float(traj[e]) * 100,
                           mins_since_peak=mins_since,
                           retrace_spd=(pk - traj[e]) * 100 / max(mins_since, 1),
                           vol_ratio=vol_now / vol_entry if vol_entry and vol_entry > 0 else np.nan,
                           adv_volshare=adv_share,
                           recovered=future_new_peak, final=gross * 100,
                           reason=t.exit_reason))

d = pd.DataFrame(rows); ev = pd.DataFrame(events)
d.to_csv(H.ROOT / "logs" / "diag_down_mfe.csv", index=False)
ev.to_csv(H.ROOT / "logs" / "diag_down_events.csv", index=False)

print(f"\n--- MFE / capture ({len(d)} trades DOWN) ---")
print(f"MFE median {d.mfe.median():.2f}%  p75 {d.mfe.quantile(.75):.2f}%")
print(f"capture ratio (trades MFE>=0.6%) : median {d[d.mfe>=0.6].capture.median():.2f}  "
      f"mean {d[d.mfe>=0.6].capture.mean():.2f}  (1.0 = tout capture)")
g = d.groupby("reason").agg(n=("ts","size"), mfe=("mfe","mean"), gross=("gross","mean"),
                            capture=("capture","mean"), hold=("hold_min","mean"))
print(g.round(2).to_string())
print(f"\nprofit RESTITUE total (MFE - realise, trades MFE>=0.6) : "
      f"{(d[d.mfe>=0.6].mfe - d[d.mfe>=0.6].gross).sum():.1f} pts de prix "
      f"(x{C.EXPOSURE:.0f} levier = {(d[d.mfe>=0.6].mfe - d[d.mfe>=0.6].gross).sum()*C.EXPOSURE:.0f} pts equity)")

print(f"\n--- EVENT STUDY (pic>={PEAK_MIN*100:.1f}%, retrace>={RETRACE_FRAC*100:.0f}% du pic) : {len(ev)} events ---")
if len(ev):
    for lab, sub in ev.groupby("recovered"):
        print(f"\n  {'RECOVERED (repart vers nouveau pic)' if lab else 'DIED (jamais de nouveau pic)'} : n={len(sub)}")
        print(f"    final moyen {sub.final.mean():+.2f}%  | pic a l'event {sub.peak.mean():.2f}%")
        for f in ("vol_ratio","adv_volshare","retrace_spd","mins_since_peak"):
            print(f"    {f:15s} mean {sub[f].mean():.3f}  median {sub[f].median():.3f}")
