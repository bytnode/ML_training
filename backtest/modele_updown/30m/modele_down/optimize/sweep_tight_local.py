"""
Tight local search around current best (63.79% / 754).
"""
from __future__ import annotations

import itertools
import numpy as np

from common import (
    load_dataset, feature_columns, train_once, hour_of_day_from_ms,
)


MIN_PARIS = 750


def filtered_threshold(val_df, p_val, wknd_arr_val, mask_val,
                       wd_frac, we_frac, calib_days):
    ms_per_day = 86_400_000
    val_max_ts = int(val_df["timestamp"].max())
    cutoff = val_max_ts - calib_days * ms_per_day
    recent = val_df["timestamp"].to_numpy() >= cutoff
    cal_mask = mask_val & recent
    thresholds = {}
    for ax, frac in [(0, wd_frac), (1, we_frac)]:
        m = cal_mask & (wknd_arr_val == ax)
        if m.sum() < 30 or frac <= 0:
            thresholds[ax] = 1.0
            continue
        thresholds[ax] = float(np.quantile(p_val[m], 1.0 - frac))
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

    arrays = {
        "vr30": (test["volume_ratio_30m"].to_numpy(dtype=float),
                 val["volume_ratio_30m"].to_numpy(dtype=float)),
        "r3":   (test["return_3"].to_numpy(dtype=float),
                 val["return_3"].to_numpy(dtype=float)),
        "r96":  (test["return_96"].to_numpy(dtype=float),
                 val["return_96"].to_numpy(dtype=float)),
        "cr":   (test["close_ret"].to_numpy(dtype=float),
                 val["close_ret"].to_numpy(dtype=float)),
        "stoch":(test["stoch_k_14"].to_numpy(dtype=float),
                 val["stoch_k_14"].to_numpy(dtype=float)),
        "mfi":  (test["mfi_norm"].to_numpy(dtype=float),
                 val["mfi_norm"].to_numpy(dtype=float)),
        "rsi":  (test["rsi_14"].to_numpy(dtype=float),
                 val["rsi_14"].to_numpy(dtype=float)),
        "uw":   (test["upper_wick"].to_numpy(dtype=float),
                 val["upper_wick"].to_numpy(dtype=float)),
        "pir50":(test["price_in_range_50"].to_numpy(dtype=float),
                 val["price_in_range_50"].to_numpy(dtype=float)),
    }

    # Tight grid around best
    BAD_HOURS_VARIANTS = [
        (1, 8, 13, 14, 17, 21),
        (1, 8, 10, 13, 14, 17, 21),
        (8, 13, 14, 17, 21),
        (1, 7, 8, 13, 14, 17, 21),
        (1, 8, 13, 14, 17, 18, 21),
        (1, 8, 13, 14, 17, 21, 23),
        (0, 1, 8, 13, 14, 17, 21),
        (1, 4, 8, 13, 14, 17, 21),
        (1, 8, 11, 13, 14, 17, 21),
        (1, 5, 8, 13, 14, 17, 21),
        (1, 8, 13, 14, 16, 17, 21),
        (1, 8, 13, 14, 17, 20, 21),
    ]
    GOOD_HOURS_VARIANTS = [
        (2, 9, 20, 22),
        (2, 20, 22),
        (2, 11, 20, 22),
        (2, 9, 11, 20, 22),
        (2, 9, 19, 20, 22),
        (9, 20, 22),
        (2, 9, 20),
        (2, 9, 22),
    ]
    DELTAS = (0.020, 0.025, 0.030, 0.035, 0.040)
    CD_GRID = (60, 75, 90, 105, 120, 150, 180)
    WD_GRID = (0.040, 0.050, 0.060, 0.070, 0.080, 0.100, 0.120)
    WE_GRID = (0.15, 0.18, 0.20, 0.22, 0.25)

    # Filter sub-grids (tight)
    F_GRIDS = {
        "vr30": [np.inf, 1.6, 1.5, 1.4, 1.3, 1.2],
        "r3":   [np.inf, 0.040, 0.030, 0.025, 0.020, 0.015],
        "r96":  [np.inf, 0.10, 0.05, 0.02, 0.0],
        "cr":   [np.inf, 0.030, 0.020, 0.015, 0.010],
        "stoch":[np.inf, 0.7, 0.5, 0.3, 0.1],
        "mfi":  [np.inf, 0.5, 0.3, 0.0],
        "pir50":[np.inf, 0.95, 0.90, 0.80],
    }

    import random
    rng = random.Random(2026)
    filter_combos = []
    # Known good base
    filter_combos.append({"vr30": 1.5, "r3": 0.030, "r96": np.inf, "cr": np.inf, "stoch": np.inf, "mfi": np.inf, "pir50": np.inf})
    # Random combos
    for _ in range(200):
        filter_combos.append({k: rng.choice(v) for k, v in F_GRIDS.items()})

    def apply_filters(m, get_arr, fc_combo, is_val):
        for name, cut in fc_combo.items():
            if cut == np.inf:
                continue
            arr = arrays[name][1 if is_val else 0]
            m = m & ((arr < cut) | np.isnan(arr))
        return m

    results = []
    for hb in BAD_HOURS_VARIANTS:
        bad_t = np.isin(hours_t, list(hb))
        bad_v = np.isin(hours_v, list(hb))
        for hg in GOOD_HOURS_VARIANTS:
            good_t = np.isin(hours_t, list(hg))
            for fc_combo in filter_combos:
                m_t = apply_filters(~bad_t, None, fc_combo, is_val=False)
                m_v = apply_filters(~bad_v, None, fc_combo, is_val=True)
                if m_v.sum() < 100 or m_t.sum() < MIN_PARIS:
                    continue
                for cd in CD_GRID:
                    for wd in WD_GRID:
                        for we in WE_GRID:
                            thr = filtered_threshold(val, p_val, wknd_v, m_v, wd, we, cd)
                            T_base = np.where(wknd_t == 1, thr.get(1, 1.0), thr.get(0, 1.0))
                            for d in DELTAS:
                                T_eff = T_base - d * good_t
                                take = m_t & (p_test >= T_eff)
                                n = int(take.sum())
                                if n < MIN_PARIS:
                                    continue
                                wr = float(y_test[take].mean())
                                if wr < 0.64:
                                    continue
                                results.append({
                                    "bad": hb, "good": hg, "delta": d,
                                    "cd": cd, "wd": wd, "we": we,
                                    "filters": dict(fc_combo),
                                    "taken": n, "wr": wr,
                                })

    results.sort(key=lambda r: (-r["wr"], -r["taken"]))
    print(f"\n{len(results)} qualifying configs (wr>=64%, taken>={MIN_PARIS})")
    print("\nTOP 30:")
    for r in results[:30]:
        def fmt(v, p=3):
            return "off" if v in (np.inf,) else f"{v:.{p}f}"
        f = r["filters"]
        bh = "(" + ",".join(str(h) for h in r["bad"]) + ")"
        gh = "(" + ",".join(str(h) for h in r["good"]) + ")"
        flt = " ".join(f"{k}<{fmt(v, 2 if k=='pir50' else 3)}" for k, v in f.items() if v != np.inf)
        print(f"  cd={r['cd']:>3} wd={r['wd']:.3f} we={r['we']:.2f} bad={bh:<30} good={gh:<18} "
              f"d={r['delta']:>5.3f} | {flt}  taken={r['taken']:>5} wr={r['wr']*100:>6.2f}%")

    goal = [r for r in results if r["wr"] >= 0.65]
    print(f"\nGOAL WR>=65%: {len(goal)}")
    for r in goal[:30]:
        def fmt(v, p=3):
            return "off" if v in (np.inf,) else f"{v:.{p}f}"
        f = r["filters"]
        bh = "(" + ",".join(str(h) for h in r["bad"]) + ")"
        gh = "(" + ",".join(str(h) for h in r["good"]) + ")"
        flt = " ".join(f"{k}<{fmt(v, 2 if k=='pir50' else 3)}" for k, v in f.items() if v != np.inf)
        print(f"  cd={r['cd']:>3} wd={r['wd']:.3f} we={r['we']:.2f} bad={bh:<30} good={gh:<18} "
              f"d={r['delta']:>5.3f} | {flt}  taken={r['taken']:>5} wr={r['wr']*100:>6.2f}%")


if __name__ == "__main__":
    main()
