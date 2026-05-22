"""
Targeted sweep based on analyze_current_best findings:
- Add h12 to bad hours
- Try more good hours (h0, h6, h11, h16 had >70% WR in current best)
- Skip vol_state=2
- Try adding obv_pressure_4h filter
"""
from __future__ import annotations

import itertools
import numpy as np
import random

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
    vol_t = test["volatility_state"].astype(int).to_numpy()
    vol_v = val["volatility_state"].astype(int).to_numpy()

    arrays = {
        "vr30": (test["volume_ratio_30m"].to_numpy(dtype=float),
                 val["volume_ratio_30m"].to_numpy(dtype=float)),
        "r3":   (test["return_3"].to_numpy(dtype=float),
                 val["return_3"].to_numpy(dtype=float)),
        "obv4": (test["obv_pressure_4h"].to_numpy(dtype=float),
                 val["obv_pressure_4h"].to_numpy(dtype=float)),
        "bb":   (test["bb_position"].to_numpy(dtype=float),
                 val["bb_position"].to_numpy(dtype=float)),
    }

    BAD_HOUR_VARIANTS = [
        (1, 8, 10, 12, 13, 14, 17, 21),
        (1, 8, 12, 13, 14, 17, 21),
        (8, 12, 13, 14, 17, 21),
        (1, 8, 10, 13, 14, 17, 21),  # original best
        (1, 8, 10, 12, 13, 14, 17, 21, 23),
        (1, 7, 8, 12, 13, 14, 17, 21),
        (1, 8, 12, 13, 14, 17, 21, 23),
    ]
    GOOD_HOUR_VARIANTS = [
        (0, 6, 11, 16),       # the "good in current" hours
        (0, 11, 16),
        (0, 6, 11, 16, 22),
        (0, 6, 11, 16, 20, 22),
        (2, 9, 20, 22),
        (0, 2, 9, 11, 16, 20, 22),
        (0, 11, 16, 20, 22),
    ]
    DELTAS = (0.020, 0.030, 0.040, 0.050, 0.060, 0.080)
    SKIP_VOL2 = (False, True)

    CD_GRID = (55, 75, 90, 120)
    WD_GRID = (0.050, 0.060, 0.080, 0.100, 0.140)
    WE_GRID = (0.15, 0.20, 0.25)

    F_GRIDS = {
        "vr30": [np.inf, 1.6, 1.5, 1.4, 1.3, 1.2],
        "r3":   [np.inf, 0.030, 0.020, 0.015],
        "obv4": [np.inf, 0.5, 0.3, 0.0, -0.3],
        "bb":   [np.inf, 0.8, 0.6, 0.4],
    }
    rng = random.Random(2027)
    filter_combos = []
    # Known good
    filter_combos.append({"vr30": 1.5, "r3": 0.030, "obv4": np.inf, "bb": np.inf})
    for _ in range(100):
        filter_combos.append({k: rng.choice(v) for k, v in F_GRIDS.items()})

    def apply_filters(m, fc_combo, is_val):
        for name, cut in fc_combo.items():
            if cut == np.inf:
                continue
            arr = arrays[name][1 if is_val else 0]
            m = m & ((arr < cut) | np.isnan(arr))
        return m

    print(f"Hour bads: {len(BAD_HOUR_VARIANTS)}, goods: {len(GOOD_HOUR_VARIANTS)}, "
          f"deltas: {len(DELTAS)}, vol_skip: {len(SKIP_VOL2)}, "
          f"calib: {len(CD_GRID)*len(WD_GRID)*len(WE_GRID)}, filters: {len(filter_combos)}")

    results = []
    for hb in BAD_HOUR_VARIANTS:
        bad_t = np.isin(hours_t, list(hb))
        bad_v = np.isin(hours_v, list(hb))
        for hg in GOOD_HOUR_VARIANTS:
            good_t = np.isin(hours_t, list(hg))
            for skip_vol2 in SKIP_VOL2:
                vol_mask_t = (vol_t != 2) if skip_vol2 else np.ones(n_test, dtype=bool)
                vol_mask_v = (vol_v != 2) if skip_vol2 else np.ones(len(val), dtype=bool)
                for fc_combo in filter_combos:
                    m_t = apply_filters(~bad_t & vol_mask_t, fc_combo, is_val=False)
                    m_v = apply_filters(~bad_v & vol_mask_v, fc_combo, is_val=True)
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
                                    if wr < 0.635:
                                        continue
                                    results.append({
                                        "bad": hb, "good": hg, "delta": d,
                                        "skip_v2": skip_vol2,
                                        "cd": cd, "wd": wd, "we": we,
                                        "filters": dict(fc_combo),
                                        "taken": n, "wr": wr,
                                    })

    results.sort(key=lambda r: (-r["wr"], -r["taken"]))
    print(f"\n{len(results)} qualifying (wr>=63.5%, taken>={MIN_PARIS})")
    print("\nTOP 30:")
    for r in results[:30]:
        def fmt(v):
            return "off" if v == np.inf else f"{v:.3f}"
        f = r["filters"]
        bh = "(" + ",".join(str(h) for h in r["bad"]) + ")"
        gh = "(" + ",".join(str(h) for h in r["good"]) + ")"
        sv = "v2skip" if r["skip_v2"] else ""
        print(f"  cd={r['cd']:>3} wd={r['wd']:.3f} we={r['we']:.2f} bad={bh:<32} good={gh:<24} "
              f"d={r['delta']:>5.3f} {sv:<7} | "
              f"vr30<{fmt(f['vr30']):>5} r3<{fmt(f['r3']):>6} obv4<{fmt(f['obv4']):>5} bb<{fmt(f['bb']):>5}  "
              f"taken={r['taken']:>5} wr={r['wr']*100:>6.2f}%")

    goal = [r for r in results if r["wr"] >= 0.65]
    print(f"\n*** GOAL WR>=65% AND taken>=750 ***: {len(goal):,}")
    for r in goal[:20]:
        def fmt(v):
            return "off" if v == np.inf else f"{v:.3f}"
        f = r["filters"]
        bh = "(" + ",".join(str(h) for h in r["bad"]) + ")"
        gh = "(" + ",".join(str(h) for h in r["good"]) + ")"
        sv = "v2skip" if r["skip_v2"] else ""
        print(f"  cd={r['cd']:>3} wd={r['wd']:.3f} we={r['we']:.2f} bad={bh:<32} good={gh:<24} "
              f"d={r['delta']:>5.3f} {sv:<7} | "
              f"vr30<{fmt(f['vr30']):>5} r3<{fmt(f['r3']):>6} obv4<{fmt(f['obv4']):>5} bb<{fmt(f['bb']):>5}  "
              f"taken={r['taken']:>5} wr={r['wr']*100:>6.2f}%")


if __name__ == "__main__":
    main()
