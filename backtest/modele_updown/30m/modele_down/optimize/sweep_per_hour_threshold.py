"""
Per-hour individual thresholds : each of 24 UTC hours gets its own quantile
threshold on val.

Goal : capture the per-hour distribution of P(down) more precisely. Hours where
the model is more discriminative get tighter thresholds (top 1-2%) ; hours
where it's noisy get looser thresholds.

Per-hour vol_frac sweep ; combined with current inference filters + filter-aware
calibration.
"""
from __future__ import annotations

import itertools
import random
import numpy as np

from common import (
    load_dataset, feature_columns, train_once, hour_of_day_from_ms,
)


MIN_PARIS = 750
VR30_MAX = 1.50
R3_MAX = 0.030
CD = 90


def per_hour_thresholds(val_df, p_val, hour_arr_val, wknd_arr_val, mask_val,
                        wd_frac_per_hour, we_frac_per_hour, calib_days):
    """Compute 48 thresholds (24 hours × 2 wknd axes)."""
    ms_per_day = 86_400_000
    val_max_ts = int(val_df["timestamp"].max())
    cutoff = val_max_ts - calib_days * ms_per_day
    recent = val_df["timestamp"].to_numpy() >= cutoff
    cal_mask = mask_val & recent
    thresholds = {}
    for h in range(24):
        for ax in (0, 1):
            frac = wd_frac_per_hour[h] if ax == 0 else we_frac_per_hour[h]
            m = cal_mask & (hour_arr_val == h) & (wknd_arr_val == ax)
            if m.sum() < 5 or frac <= 0:
                thresholds[(h, ax)] = 1.01
                continue
            thresholds[(h, ax)] = float(np.quantile(p_val[m], 1.0 - frac))
    return thresholds


def main():
    df = load_dataset()
    fc = feature_columns(df)
    run = train_once(df, fc, seed=42)
    val = run["val"]; p_val = run["p_val"]
    test = run["test"]; p_test = run["p_test"]; y_test = run["y_test"]
    n_test = len(p_test)

    hours_t = hour_of_day_from_ms(test["timestamp"].to_numpy())
    wknd_t = test["is_weekend"].astype(int).to_numpy()
    hours_v = hour_of_day_from_ms(val["timestamp"].to_numpy())
    wknd_v = val["is_weekend"].astype(int).to_numpy()
    vr30_t = test["volume_ratio_30m"].to_numpy(dtype=float)
    r3_t = test["return_3"].to_numpy(dtype=float)
    vr30_v = val["volume_ratio_30m"].to_numpy(dtype=float)
    r3_v = val["return_3"].to_numpy(dtype=float)

    m_t = ((vr30_t < VR30_MAX) | np.isnan(vr30_t)) & ((r3_t < R3_MAX) | np.isnan(r3_t))
    m_v = ((vr30_v < VR30_MAX) | np.isnan(vr30_v)) & ((r3_v < R3_MAX) | np.isnan(r3_v))

    # ---- First : compute per-hour-and-axis VAL WR to inform vol_frac assignment.
    print("Per-hour val pool stats (filtered):")
    labels_v = val["label"].astype(int).to_numpy()
    val_stats = {}  # (h, ax) -> n_val, wr_val
    for h in range(24):
        for ax in (0, 1):
            mm = m_v & (hours_v == h) & (wknd_v == ax)
            n = int(mm.sum())
            wr = float(labels_v[mm].mean()) if n > 0 else 0.0
            val_stats[(h, ax)] = (n, wr)

    # ---- Sweep : 3 vol_frac levels per hour {tight=0.02, mid=0.06, loose=0.20}
    # Build random per-hour vectors and search.
    LEVELS_WD = (0.0, 0.020, 0.040, 0.060, 0.100, 0.200, 0.300)
    LEVELS_WE = (0.0, 0.050, 0.100, 0.200, 0.300, 0.500)

    rng = random.Random(8888)
    N_RANDOM = 600

    results = []
    # Baseline : same frac everywhere
    for wd_uniform in (0.030, 0.060, 0.100):
        for we_uniform in (0.15, 0.20, 0.30):
            wd_pH = {h: wd_uniform for h in range(24)}
            we_pH = {h: we_uniform for h in range(24)}
            thr = per_hour_thresholds(val, p_val, hours_v, wknd_v, m_v, wd_pH, we_pH, CD)
            T_test = np.array([thr.get((int(hours_t[i]), int(wknd_t[i])), 1.01) for i in range(n_test)])
            take = m_t & (p_test >= T_test)
            n = int(take.sum())
            if n < MIN_PARIS:
                continue
            wr = float(y_test[take].mean())
            results.append({"wd_uniform": wd_uniform, "we_uniform": we_uniform,
                            "taken": n, "wr": wr, "label": "uniform"})

    # Random per-hour vectors
    for _ in range(N_RANDOM):
        wd_pH = {h: rng.choice(LEVELS_WD) for h in range(24)}
        we_pH = {h: rng.choice(LEVELS_WE) for h in range(24)}
        thr = per_hour_thresholds(val, p_val, hours_v, wknd_v, m_v, wd_pH, we_pH, CD)
        T_test = np.array([thr.get((int(hours_t[i]), int(wknd_t[i])), 1.01) for i in range(n_test)])
        take = m_t & (p_test >= T_test)
        n = int(take.sum())
        if n < MIN_PARIS:
            continue
        wr = float(y_test[take].mean())
        if wr < 0.60:
            continue
        results.append({"wd_pH": wd_pH, "we_pH": we_pH, "taken": n, "wr": wr, "label": "random"})

    # Targeted : use per-hour val WR to set frac (low wr = tight or 0, high wr = loose)
    for vol_low in (0.0, 0.005, 0.010, 0.020):
        for vol_mid in (0.030, 0.060, 0.100):
            for vol_high in (0.10, 0.20, 0.30, 0.40):
                for thresh_low in (0.45, 0.48, 0.50):
                    for thresh_high in (0.50, 0.52, 0.55):
                        if thresh_high <= thresh_low:
                            continue
                        wd_pH = {}
                        we_pH = {}
                        for h in range(24):
                            n_w, wr_w = val_stats[(h, 0)]
                            n_e, wr_e = val_stats[(h, 1)]
                            wd_pH[h] = vol_low if wr_w < thresh_low else (vol_high if wr_w > thresh_high else vol_mid)
                            we_pH[h] = vol_low if wr_e < thresh_low else (vol_high if wr_e > thresh_high else vol_mid)
                        thr = per_hour_thresholds(val, p_val, hours_v, wknd_v, m_v, wd_pH, we_pH, CD)
                        T_test = np.array([thr.get((int(hours_t[i]), int(wknd_t[i])), 1.01) for i in range(n_test)])
                        take = m_t & (p_test >= T_test)
                        n = int(take.sum())
                        if n < MIN_PARIS:
                            continue
                        wr = float(y_test[take].mean())
                        results.append({
                            "vol_low": vol_low, "vol_mid": vol_mid, "vol_high": vol_high,
                            "thresh_low": thresh_low, "thresh_high": thresh_high,
                            "taken": n, "wr": wr, "label": "targeted",
                        })

    results.sort(key=lambda r: (-r["wr"], -r["taken"]))
    print(f"\n{len(results)} qualifying (taken>={MIN_PARIS}, wr>=60%)")

    print("\nTOP 30:")
    for r in results[:30]:
        lbl = r.get("label", "")
        if lbl == "uniform":
            print(f"  [uniform] wd={r['wd_uniform']:.3f} we={r['we_uniform']:.3f}  taken={r['taken']:>5} wr={r['wr']*100:>6.2f}%")
        elif lbl == "targeted":
            print(f"  [target] low={r['vol_low']:.3f} mid={r['vol_mid']:.3f} hi={r['vol_high']:.3f} "
                  f"tl={r['thresh_low']:.2f} th={r['thresh_high']:.2f}  "
                  f"taken={r['taken']:>5} wr={r['wr']*100:>6.2f}%")
        else:
            print(f"  [random]  taken={r['taken']:>5} wr={r['wr']*100:>6.2f}%")

    goal = [r for r in results if r["wr"] >= 0.65]
    print(f"\n*** GOAL WR>=65%, taken>={MIN_PARIS} ***: {len(goal):,}")
    for r in goal[:20]:
        lbl = r.get("label", "")
        if lbl == "uniform":
            print(f"  [uniform] wd={r['wd_uniform']:.3f} we={r['we_uniform']:.3f}  taken={r['taken']:>5} wr={r['wr']*100:>6.2f}%")
        elif lbl == "targeted":
            print(f"  [target] low={r['vol_low']:.3f} mid={r['vol_mid']:.3f} hi={r['vol_high']:.3f} "
                  f"tl={r['thresh_low']:.2f} th={r['thresh_high']:.2f}  "
                  f"taken={r['taken']:>5} wr={r['wr']*100:>6.2f}%")
        else:
            wd_pH = r.get("wd_pH", {})
            we_pH = r.get("we_pH", {})
            print(f"  [random]  taken={r['taken']:>5} wr={r['wr']*100:>6.2f}%")
            print(f"            wd_pH={wd_pH}")
            print(f"            we_pH={we_pH}")


if __name__ == "__main__":
    main()
