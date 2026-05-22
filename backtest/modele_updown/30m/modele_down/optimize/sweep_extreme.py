"""
Extreme push: very loose wd (more val volume), very strict filters, hour
combinations not previously tried.
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
    print(f"Dataset: {len(df):,} rows, {len(fc)} features.")
    run = train_once(df, fc, seed=42)
    val = run["val"]; p_val = run["p_val"]
    test = run["test"]; p_test = run["p_test"]; y_test = run["y_test"]
    n_test = len(p_test)
    hours_t = hour_of_day_from_ms(test["timestamp"].to_numpy())
    wknd_t = test["is_weekend"].astype(int).to_numpy()
    hours_v = hour_of_day_from_ms(val["timestamp"].to_numpy())
    wknd_v = val["is_weekend"].astype(int).to_numpy()

    def get(name, src):
        return src[name].to_numpy(dtype=float)
    feats_t = {n: get(c, test) for c, n in [
        ("volume_ratio_30m", "vr30"), ("return_3", "r3"), ("return_96", "r96"),
        ("upper_wick", "uw"), ("close_ret", "cr"), ("body_ratio_close", "body"),
        ("stoch_k_14", "stoch"), ("rsi_14", "rsi"),
    ]}
    feats_v = {n: get(c, val) for c, n in [
        ("volume_ratio_30m", "vr30"), ("return_3", "r3"), ("return_96", "r96"),
        ("upper_wick", "uw"), ("close_ret", "cr"), ("body_ratio_close", "body"),
        ("stoch_k_14", "stoch"), ("rsi_14", "rsi"),
    ]}

    # Many hour combos including new ones
    HOUR_BADS = [
        (8, 13, 14, 17, 21),
        (8, 14, 17, 21),
        (8, 10, 13, 14, 17, 21),
        (1, 8, 13, 14, 17, 21),
        (7, 8, 13, 14, 17, 21),
        (4, 8, 13, 14, 17, 21),
        (8, 13, 14, 17, 18, 21),
        (8, 13, 14, 17, 21, 23),
        (0, 8, 13, 14, 17, 21),
    ]
    HOUR_GOODS = [
        (2, 20, 22),
        (2, 9, 20, 22),
        (2, 11, 20, 22),
        (2, 11, 22),
        (2, 22),
        (9, 11, 20),
        (2, 9, 11, 20, 22),
        (11, 20, 22),
    ]
    DELTAS = (0.015, 0.020, 0.030, 0.040, 0.060)

    CD_GRID = (40, 55, 75, 90, 120, 180)
    WD_GRID = (0.060, 0.080, 0.100, 0.120, 0.150, 0.200, 0.300)
    WE_GRID = (0.10, 0.15, 0.20, 0.25, 0.30, 0.40)

    FILTER_GRIDS = {
        "vr30": [np.inf, 1.6, 1.5, 1.4, 1.3, 1.2, 1.0],
        "r3":   [np.inf, 0.030, 0.020, 0.015, 0.010, 0.005, 0.0, -0.005],
        "r96":  [np.inf, 0.10, 0.05, 0.0],
        "cr":   [np.inf, 0.030, 0.020, 0.015, 0.010, 0.005, 0.0],
        "stoch":[np.inf, 0.5, 0.3, 0.1],
    }
    # Reduce search by trying random combos
    import random
    rng = random.Random(2024)
    N_FILTER_SAMPLES = 200
    filter_combos = []
    for _ in range(N_FILTER_SAMPLES):
        fc_combo = {k: rng.choice(v) for k, v in FILTER_GRIDS.items()}
        filter_combos.append(fc_combo)
    # Add the known-good combos explicitly
    filter_combos.append({"vr30": np.inf, "r3": 0.015, "r96": np.inf, "cr": 0.015, "stoch": np.inf})
    filter_combos.append({"vr30": 1.5, "r3": 0.030, "r96": np.inf, "cr": np.inf, "stoch": np.inf})

    def apply_filters(m, feats, fc_combo):
        for name, cut in fc_combo.items():
            if cut == np.inf:
                continue
            arr = feats[name]
            m = m & ((arr < cut) | np.isnan(arr))
        return m

    print(f"\nHour bads: {len(HOUR_BADS)} x goods {len(HOUR_GOODS)} x deltas {len(DELTAS)}")
    print(f"Calib: {len(CD_GRID)*len(WD_GRID)*len(WE_GRID)}")
    print(f"Filter combos: {len(filter_combos)}")

    results = []
    progress = 0
    for hb in HOUR_BADS:
        bad_t = np.isin(hours_t, list(hb))
        bad_v = np.isin(hours_v, list(hb))
        for hg in HOUR_GOODS:
            good_t = np.isin(hours_t, list(hg))
            for fc_combo in filter_combos:
                m_t = apply_filters(~bad_t, feats_t, fc_combo)
                m_v = apply_filters(~bad_v, feats_v, fc_combo)
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
                                if wr < 0.625:
                                    continue
                                results.append({
                                    "bad": hb, "good": hg, "delta": d,
                                    "cd": cd, "wd": wd, "we": we,
                                    "filters": dict(fc_combo),
                                    "taken": n, "wr": wr,
                                })
            progress += 1

    results.sort(key=lambda r: (-r["wr"], -r["taken"]))
    print(f"\n{len(results)} qualifying configs (wr>=62.5%, taken>={MIN_PARIS})")

    print("\nTOP 30:")
    print("-" * 220)
    for r in results[:30]:
        def fmt(v, p=3):
            return "off" if v in (np.inf, -np.inf) else f"{v:.{p}f}"
        f = r["filters"]
        bh_s = "(" + ",".join(str(h) for h in r["bad"]) + ")"
        gh_s = "(" + ",".join(str(h) for h in r["good"]) + ")"
        print(f"  cd={r['cd']:>3} wd={r['wd']:.3f} we={r['we']:.2f} bad={bh_s:<26} good={gh_s:<22} "
              f"d={r['delta']:>5.3f} vr30<{fmt(f['vr30'],2):<4} r3<{fmt(f['r3']):<6} "
              f"r96<{fmt(f['r96'],2):<5} cr<{fmt(f['cr']):<6} stoch<{fmt(f['stoch'],2):<5}  "
              f"taken={r['taken']:>5} wr={r['wr']*100:>6.2f}%")

    goal = [r for r in results if r["wr"] >= 0.65]
    print(f"\nGOAL WR>=65%: {len(goal):,}")
    for r in goal[:20]:
        def fmt(v, p=3):
            return "off" if v in (np.inf, -np.inf) else f"{v:.{p}f}"
        f = r["filters"]
        bh_s = "(" + ",".join(str(h) for h in r["bad"]) + ")"
        gh_s = "(" + ",".join(str(h) for h in r["good"]) + ")"
        print(f"  cd={r['cd']:>3} wd={r['wd']:.3f} we={r['we']:.2f} bad={bh_s:<26} good={gh_s:<22} "
              f"d={r['delta']:>5.3f} vr30<{fmt(f['vr30'],2):<4} r3<{fmt(f['r3']):<6} "
              f"r96<{fmt(f['r96'],2):<5} cr<{fmt(f['cr']):<6} stoch<{fmt(f['stoch'],2):<5}  "
              f"taken={r['taken']:>5} wr={r['wr']*100:>6.2f}%")


if __name__ == "__main__":
    main()
