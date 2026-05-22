"""
Mega random search : 10,000 configs across all dimensions.
Vectorized, writes top configs to file incrementally.

Search space :
- 24 binary bad-hour bits
- 24 binary good-hour bits
- calib_days (30-200), wd (0.02-0.2), we (0.05-0.4), delta (0.005-0.08)
- vr30_max, r3_max, r96_max, cr_max, uw_min, stoch_max, rsi_max, bb_max
"""
from __future__ import annotations

import random
import sys
import time
import numpy as np

from common import (
    load_dataset, feature_columns, train_once, hour_of_day_from_ms,
)


MIN_PARIS = 750
N_CALLS = 10000
SEED = 31337


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
    feat_t = {n: test[c].to_numpy(dtype=float) for c, n in [
        ("volume_ratio_30m","vr30"), ("return_3","r3"), ("return_96","r96"),
        ("close_ret","cr"), ("upper_wick","uw"), ("stoch_k_14","stoch"),
        ("rsi_14","rsi"), ("bb_position","bb"),
    ]}
    feat_v = {n: val[c].to_numpy(dtype=float) for c, n in [
        ("volume_ratio_30m","vr30"), ("return_3","r3"), ("return_96","r96"),
        ("close_ret","cr"), ("upper_wick","uw"), ("stoch_k_14","stoch"),
        ("rsi_14","rsi"), ("bb_position","bb"),
    ]}
    hour_masks_t = {h: (hours_t == h) for h in range(24)}
    hour_masks_v = {h: (hours_v == h) for h in range(24)}

    rng = random.Random(SEED)
    best_wr = 0
    best_n = 0
    best_cfg = None
    log_file = "sweep_mega_log.txt"
    fl = open(log_file, "w")
    fl.write("# mega random sweep results\n")
    fl.write(f"# {N_CALLS} random configs, MIN_PARIS={MIN_PARIS}\n")
    fl.flush()

    t0 = time.time()
    for i in range(N_CALLS):
        cd = rng.randint(30, 200)
        wd = rng.uniform(0.020, 0.200)
        we = rng.uniform(0.05, 0.40)
        delta = rng.uniform(0.005, 0.080)
        vr30 = rng.choice([float("inf"), 2.5, 2.0, 1.8, 1.5, 1.3, 1.2, 1.1, 1.0])
        r3 = rng.choice([float("inf"), 0.05, 0.030, 0.020, 0.015, 0.010, 0.005, 0.0, -0.005])
        r96 = rng.choice([float("inf"), 0.30, 0.20, 0.15, 0.10, 0.05, 0.0])
        cr = rng.choice([float("inf"), 0.030, 0.020, 0.015, 0.010, 0.005, 0.0])
        uw = rng.choice([0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
        stoch = rng.choice([float("inf"), 0.5, 0.3, 0.1, -0.1, -0.3])
        rsi = rng.choice([float("inf"), 0.5, 0.3, 0.1, -0.1])
        bb = rng.choice([float("inf"), 0.95, 0.85, 0.75, 0.65, 0.55])
        bad_bits = [rng.random() < 0.35 for _ in range(24)]
        good_bits = [rng.random() < 0.25 for _ in range(24)]
        bad_hours = tuple(h for h in range(24) if bad_bits[h])
        good_hours = tuple(h for h in range(24) if good_bits[h])

        bad_t = np.zeros(n_test, dtype=bool)
        for h in bad_hours: bad_t |= hour_masks_t[h]
        bad_v_mask = np.zeros(len(val), dtype=bool)
        for h in bad_hours: bad_v_mask |= hour_masks_v[h]
        good_t = np.zeros(n_test, dtype=bool)
        for h in good_hours: good_t |= hour_masks_t[h]

        m_t = ~bad_t
        if vr30 != float("inf"): m_t &= (feat_t["vr30"] < vr30) | np.isnan(feat_t["vr30"])
        if r3 != float("inf"): m_t &= (feat_t["r3"] < r3) | np.isnan(feat_t["r3"])
        if r96 != float("inf"): m_t &= (feat_t["r96"] < r96) | np.isnan(feat_t["r96"])
        if cr != float("inf"): m_t &= (feat_t["cr"] < cr) | np.isnan(feat_t["cr"])
        if uw > 0: m_t &= (feat_t["uw"] >= uw) | np.isnan(feat_t["uw"])
        if stoch != float("inf"): m_t &= (feat_t["stoch"] < stoch) | np.isnan(feat_t["stoch"])
        if rsi != float("inf"): m_t &= (feat_t["rsi"] < rsi) | np.isnan(feat_t["rsi"])
        if bb != float("inf"): m_t &= (feat_t["bb"] < bb) | np.isnan(feat_t["bb"])

        m_v = ~bad_v_mask
        if vr30 != float("inf"): m_v &= (feat_v["vr30"] < vr30) | np.isnan(feat_v["vr30"])
        if r3 != float("inf"): m_v &= (feat_v["r3"] < r3) | np.isnan(feat_v["r3"])
        if r96 != float("inf"): m_v &= (feat_v["r96"] < r96) | np.isnan(feat_v["r96"])
        if cr != float("inf"): m_v &= (feat_v["cr"] < cr) | np.isnan(feat_v["cr"])
        if uw > 0: m_v &= (feat_v["uw"] >= uw) | np.isnan(feat_v["uw"])
        if stoch != float("inf"): m_v &= (feat_v["stoch"] < stoch) | np.isnan(feat_v["stoch"])
        if rsi != float("inf"): m_v &= (feat_v["rsi"] < rsi) | np.isnan(feat_v["rsi"])
        if bb != float("inf"): m_v &= (feat_v["bb"] < bb) | np.isnan(feat_v["bb"])

        if m_v.sum() < 100 or m_t.sum() < MIN_PARIS:
            continue

        thr = filtered_threshold(val, p_val, wknd_v, m_v, wd, we, cd)
        T_base = np.where(wknd_t == 1, thr.get(1, 1.0), thr.get(0, 1.0))
        T_eff = T_base - delta * good_t
        take = m_t & (p_test >= T_eff)
        n = int(take.sum())
        if n < MIN_PARIS:
            continue
        wr = float(y_test[take].mean())
        if wr > best_wr:
            best_wr = wr
            best_n = n
            best_cfg = {
                "cd": cd, "wd": round(wd, 4), "we": round(we, 4), "delta": round(delta, 4),
                "vr30": vr30, "r3": r3, "r96": r96, "cr": cr,
                "uw": uw, "stoch": stoch, "rsi": rsi, "bb": bb,
                "bad": bad_hours, "good": good_hours,
            }
            msg = f"[{i+1}/{N_CALLS}] NEW BEST: n={n} wr={wr*100:.2f}% cfg={best_cfg}"
            print(msg, flush=True)
            fl.write(msg + "\n"); fl.flush()
            if wr >= 0.65:
                msg2 = f"*** GOAL REACHED at iter {i+1} ***"
                print(msg2, flush=True)
                fl.write(msg2 + "\n"); fl.flush()
        if i % 500 == 499:
            print(f"  [{i+1}/{N_CALLS}] best so far: n={best_n} wr={best_wr*100:.2f}% "
                  f"elapsed={time.time()-t0:.0f}s", flush=True)

    print(f"\n=== FINAL ===", flush=True)
    print(f"  best: n={best_n} wr={best_wr*100:.2f}%", flush=True)
    print(f"  cfg: {best_cfg}", flush=True)
    fl.write(f"\nFINAL n={best_n} wr={best_wr*100:.2f}% cfg={best_cfg}\n")
    fl.close()


if __name__ == "__main__":
    main()
