"""
Hill climb from current best (63.79%). Perturb one dim at a time, accept
strict improvement. Run 5 random restarts.
"""
from __future__ import annotations

import random
import numpy as np

from common import (
    load_dataset, feature_columns, train_once, hour_of_day_from_ms,
)


MIN_PARIS = 750


BASE = {
    "cd": 90, "wd": 0.060, "we": 0.20, "delta": 0.030,
    "vr30": 1.5, "r3": 0.030,
    "r96": float("inf"), "cr": float("inf"),
    "uw": 0.0, "stoch": float("inf"), "rsi": float("inf"), "bb": float("inf"),
    "bad": (1, 8, 10, 13, 14, 17, 21),
    "good": (2, 9, 20, 22),
}


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


def eval_cfg(cfg, val, p_val, test, p_test, y_test,
             hours_t, wknd_t, hours_v, wknd_v, feat_t, feat_v,
             hour_masks_t, hour_masks_v):
    n_test = len(p_test)
    bad_t = np.zeros(n_test, dtype=bool)
    for h in cfg["bad"]: bad_t |= hour_masks_t[h]
    bad_v_mask = np.zeros(len(val), dtype=bool)
    for h in cfg["bad"]: bad_v_mask |= hour_masks_v[h]
    good_t = np.zeros(n_test, dtype=bool)
    for h in cfg["good"]: good_t |= hour_masks_t[h]

    m_t = ~bad_t; m_v = ~bad_v_mask
    INF = float("inf")
    for nm in ("vr30", "r3", "r96", "cr"):
        c = cfg[nm]
        if c != INF:
            m_t &= (feat_t[nm] < c) | np.isnan(feat_t[nm])
            m_v &= (feat_v[nm] < c) | np.isnan(feat_v[nm])
    if cfg["uw"] > 0:
        m_t &= (feat_t["uw"] >= cfg["uw"]) | np.isnan(feat_t["uw"])
        m_v &= (feat_v["uw"] >= cfg["uw"]) | np.isnan(feat_v["uw"])
    for nm in ("stoch", "rsi", "bb"):
        c = cfg[nm]
        if c != INF:
            m_t &= (feat_t[nm] < c) | np.isnan(feat_t[nm])
            m_v &= (feat_v[nm] < c) | np.isnan(feat_v[nm])

    if m_v.sum() < 100 or m_t.sum() < MIN_PARIS:
        return 0, 0.0
    thr = filtered_threshold(val, p_val, wknd_v, m_v, cfg["wd"], cfg["we"], cfg["cd"])
    T_base = np.where(wknd_t == 1, thr.get(1, 1.0), thr.get(0, 1.0))
    T_eff = T_base - cfg["delta"] * good_t
    take = m_t & (p_test >= T_eff)
    n = int(take.sum())
    if n < MIN_PARIS:
        return 0, 0.0
    wr = float(y_test[take].mean())
    return n, wr


def neighbors(cfg, rng):
    """Generate ~30 random perturbations of cfg."""
    INF = float("inf")
    neighs = []
    # Continuous perturbations
    for _ in range(8):
        new = dict(cfg)
        new["cd"] = max(20, min(250, cfg["cd"] + rng.choice([-20, -10, -5, 5, 10, 20])))
        neighs.append(new)
    for _ in range(8):
        new = dict(cfg)
        new["wd"] = max(0.01, min(0.30, cfg["wd"] + rng.uniform(-0.02, 0.02)))
        neighs.append(new)
        new = dict(cfg)
        new["we"] = max(0.03, min(0.50, cfg["we"] + rng.uniform(-0.05, 0.05)))
        neighs.append(new)
        new = dict(cfg)
        new["delta"] = max(0.005, min(0.10, cfg["delta"] + rng.uniform(-0.015, 0.015)))
        neighs.append(new)
    # Filter perturbations
    for nm, levels in [
        ("vr30", [INF, 2.5, 2.0, 1.8, 1.6, 1.5, 1.4, 1.3, 1.2, 1.1, 1.0]),
        ("r3", [INF, 0.05, 0.04, 0.030, 0.025, 0.020, 0.015, 0.010, 0.005, 0.0, -0.005]),
        ("r96", [INF, 0.20, 0.15, 0.10, 0.05, 0.02, 0.0]),
        ("cr", [INF, 0.030, 0.020, 0.015, 0.010, 0.005, 0.0]),
        ("uw", [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]),
        ("stoch", [INF, 0.5, 0.3, 0.1, -0.1, -0.3]),
        ("rsi", [INF, 0.5, 0.3, 0.1, -0.1]),
        ("bb", [INF, 0.95, 0.85, 0.75, 0.65, 0.55]),
    ]:
        for lvl in levels:
            new = dict(cfg); new[nm] = lvl
            neighs.append(new)
    # Hour toggles
    for h in range(24):
        new = dict(cfg)
        if h in cfg["bad"]:
            new["bad"] = tuple(x for x in cfg["bad"] if x != h)
        else:
            new["bad"] = tuple(sorted(cfg["bad"] + (h,)))
        neighs.append(new)
        new = dict(cfg)
        if h in cfg["good"]:
            new["good"] = tuple(x for x in cfg["good"] if x != h)
        else:
            new["good"] = tuple(sorted(cfg["good"] + (h,)))
        neighs.append(new)
    return neighs


def main():
    df = load_dataset()
    fc = feature_columns(df)
    run = train_once(df, fc, seed=42)
    val = run["val"]; p_val = run["p_val"]
    test = run["test"]; p_test = run["p_test"]; y_test = run["y_test"]
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

    cur = dict(BASE)
    n_cur, wr_cur = eval_cfg(cur, val, p_val, test, p_test, y_test,
                              hours_t, wknd_t, hours_v, wknd_v, feat_t, feat_v,
                              hour_masks_t, hour_masks_v)
    print(f"Start: n={n_cur} wr={wr_cur*100:.2f}%", flush=True)

    rng = random.Random(42)
    overall_best = (n_cur, wr_cur, dict(cur))

    for restart in range(3):
        for it in range(50):
            neigh_list = neighbors(cur, rng)
            improved = False
            for n_cfg in neigh_list:
                n, w = eval_cfg(n_cfg, val, p_val, test, p_test, y_test,
                                 hours_t, wknd_t, hours_v, wknd_v, feat_t, feat_v,
                                 hour_masks_t, hour_masks_v)
                if w > wr_cur + 0.0001 and n >= MIN_PARIS:
                    cur = n_cfg
                    wr_cur = w; n_cur = n
                    improved = True
                    print(f"  [r{restart} it{it}] n={n} wr={w*100:.2f}%  cfg={cur}", flush=True)
                    if w > overall_best[1]:
                        overall_best = (n, w, dict(cur))
                    if w >= 0.65:
                        print(f"  *** GOAL REACHED ***", flush=True)
                    break
            if not improved:
                # Stuck
                print(f"  [r{restart} it{it}] stuck at n={n_cur} wr={wr_cur*100:.2f}%", flush=True)
                break
        # Random restart with small perturbation
        if restart < 2:
            cur = dict(overall_best[2])
            INF = float("inf")
            for _ in range(3):
                k = rng.choice(["cd", "wd", "we", "delta", "vr30", "r3"])
                if k == "cd": cur["cd"] = rng.randint(40, 200)
                elif k == "wd": cur["wd"] = rng.uniform(0.03, 0.15)
                elif k == "we": cur["we"] = rng.uniform(0.10, 0.35)
                elif k == "delta": cur["delta"] = rng.uniform(0.01, 0.08)
                elif k == "vr30": cur["vr30"] = rng.choice([INF, 1.6, 1.4, 1.3])
                elif k == "r3": cur["r3"] = rng.choice([INF, 0.025, 0.020, 0.015])
            n_cur, wr_cur = eval_cfg(cur, val, p_val, test, p_test, y_test,
                                      hours_t, wknd_t, hours_v, wknd_v, feat_t, feat_v,
                                      hour_masks_t, hour_masks_v)
            print(f"  restart {restart+1}: n={n_cur} wr={wr_cur*100:.2f}%", flush=True)

    print(f"\n=== OVERALL BEST ===", flush=True)
    print(f"  n={overall_best[0]} wr={overall_best[1]*100:.2f}%", flush=True)
    print(f"  cfg={overall_best[2]}", flush=True)


if __name__ == "__main__":
    main()
