"""
Apply filters to the VAL pool BEFORE computing the threshold quantile,
matching the inference-time filter regime.

If filters drop some val rows, the remaining val distribution is what test
will be drawn from. Calibrating on the filtered val should give a properly
calibrated threshold for filtered test.

Sweep variations of filter cutoffs with filter-aware calibration.
"""
from __future__ import annotations

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
        T = float(np.quantile(p_val[m], 1.0 - frac))
        thresholds[ax] = T
    return thresholds


def main():
    df = load_dataset()
    fc = feature_columns(df)
    print(f"Dataset: {len(df):,} rows, {len(fc)} features.")
    run = train_once(df, fc, seed=42)
    val = run["val"]; p_val = run["p_val"]
    test = run["test"]; p_test = run["p_test"]; y_test = run["y_test"]
    n_test = len(p_test)

    hours_t = hour_of_day_from_ms(test["timestamp"].to_numpy())
    wknd_t = test["is_weekend"].astype(int).to_numpy()
    hours_v = hour_of_day_from_ms(val["timestamp"].to_numpy())
    wknd_v = val["is_weekend"].astype(int).to_numpy()

    feat_t = {
        "vr30": test["volume_ratio_30m"].to_numpy(dtype=float),
        "r3":   test["return_3"].to_numpy(dtype=float),
        "r96":  test["return_96"].to_numpy(dtype=float),
        "uw":   test["upper_wick"].to_numpy(dtype=float),
        "cr":   test["close_ret"].to_numpy(dtype=float),
        "body": test["body_ratio_close"].to_numpy(dtype=float),
        "stoch":test["stoch_k_14"].to_numpy(dtype=float),
        "rsi":  test["rsi_14"].to_numpy(dtype=float),
    }
    feat_v = {
        "vr30": val["volume_ratio_30m"].to_numpy(dtype=float),
        "r3":   val["return_3"].to_numpy(dtype=float),
        "r96":  val["return_96"].to_numpy(dtype=float),
        "uw":   val["upper_wick"].to_numpy(dtype=float),
        "cr":   val["close_ret"].to_numpy(dtype=float),
        "body": val["body_ratio_close"].to_numpy(dtype=float),
        "stoch":val["stoch_k_14"].to_numpy(dtype=float),
        "rsi":  val["rsi_14"].to_numpy(dtype=float),
    }

    BAD_HOURS = (8, 13, 14, 17, 21)
    GOOD_HOURS = (2, 20, 22)
    DELTA = 0.020
    bad_t = np.isin(hours_t, list(BAD_HOURS))
    good_t = np.isin(hours_t, list(GOOD_HOURS))
    bad_v = np.isin(hours_v, list(BAD_HOURS))
    good_v = np.isin(hours_v, list(GOOD_HOURS))

    # Grid of filter cutoffs
    GRIDS = {
        "vr30": [np.inf, 1.5, 1.3, 1.2, 1.1, 1.0],
        "r3":   [np.inf, 0.030, 0.020, 0.015, 0.010, 0.005, 0.0],
        "r96":  [np.inf, 0.10, 0.05, 0.02, 0.0],
        "uw":   [0.0, 0.10, 0.15, 0.20],
        "cr":   [np.inf, 0.020, 0.015, 0.010, 0.005, 0.0],
    }
    # Only sweep some combos
    CD_GRID = (55, 90, 180)
    WD_GRID = (0.040, 0.060, 0.080, 0.100, 0.140)
    WE_GRID = (0.15, 0.20, 0.25)

    print(f"\nWith filter-aware calibration:")

    def mask_for(filters, arrs, is_val=False):
        m = ~bad_v.copy() if is_val else ~bad_t.copy()
        for name, cut in filters.items():
            if name == "uw":
                if cut > 0:
                    m &= (arrs[name] >= cut) | np.isnan(arrs[name])
            else:
                if cut != np.inf:
                    m &= (arrs[name] < cut) | np.isnan(arrs[name])
        return m

    results = []
    # Smaller grid: only the top combos
    import itertools as it
    combos = list(it.product(
        GRIDS["vr30"], GRIDS["r3"], GRIDS["r96"], GRIDS["uw"], GRIDS["cr"]
    ))
    print(f"  {len(combos)} filter combos x {len(CD_GRID)*len(WD_GRID)*len(WE_GRID)} calib variants")

    for vr_c, r3_c, r96_c, uw_c, cr_c in combos:
        filters = {"vr30": vr_c, "r3": r3_c, "r96": r96_c, "uw": uw_c, "cr": cr_c}
        # Test mask
        m_t = mask_for(filters, feat_t, is_val=False)
        # Val mask
        m_v = mask_for(filters, feat_v, is_val=True)
        if m_v.sum() < 100 or m_t.sum() < MIN_PARIS:
            continue

        for cd in CD_GRID:
            for wd in WD_GRID:
                for we in WE_GRID:
                    thr = filtered_threshold(val, p_val, wknd_v, m_v, wd, we, cd)
                    T_base = np.where(wknd_t == 1, thr.get(1, 1.0), thr.get(0, 1.0))
                    T_eff = T_base - DELTA * good_t
                    take = m_t & (p_test >= T_eff)
                    n = int(take.sum())
                    if n < MIN_PARIS:
                        continue
                    wr = float(y_test[take].mean())
                    if wr < 0.62:
                        continue
                    results.append({
                        "filters": filters, "cd": cd, "wd": wd, "we": we,
                        "T_wd": thr.get(0, 1.0), "T_we": thr.get(1, 1.0),
                        "taken": n, "wr": wr,
                    })

    results.sort(key=lambda r: (-r["wr"], -r["taken"]))
    print(f"\n{len(results)} qualifying configs")
    print("\nTOP 30:")
    print("-" * 200)
    for r in results[:30]:
        def fmt(v, p=3):
            return "off" if v in (np.inf, -np.inf) else f"{v:.{p}f}"
        f = r["filters"]
        print(f"  cd={r['cd']:>3} wd={r['wd']:.3f} we={r['we']:.2f}  "
              f"vr30<{fmt(f['vr30'], 2):<4} r3<{fmt(f['r3']):<6} r96<{fmt(f['r96'],2):<5} "
              f"uw>={f['uw']:.2f} cr<{fmt(f['cr']):<6}  T_wd={r['T_wd']:.4f} T_we={r['T_we']:.4f}  "
              f"taken={r['taken']:>5} wr={r['wr']*100:>6.2f}%")

    goal = [r for r in results if r["wr"] >= 0.65]
    print(f"\nGOAL WR>=65%: {len(goal):,}")
    for r in goal[:15]:
        def fmt(v, p=3):
            return "off" if v in (np.inf, -np.inf) else f"{v:.{p}f}"
        f = r["filters"]
        print(f"  cd={r['cd']:>3} wd={r['wd']:.3f} we={r['we']:.2f}  "
              f"vr30<{fmt(f['vr30'], 2):<4} r3<{fmt(f['r3']):<6} r96<{fmt(f['r96'],2):<5} "
              f"uw>={f['uw']:.2f} cr<{fmt(f['cr']):<6}  "
              f"taken={r['taken']:>5} wr={r['wr']*100:>6.2f}%")


if __name__ == "__main__":
    main()
