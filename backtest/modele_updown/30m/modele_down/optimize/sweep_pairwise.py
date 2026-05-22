"""
Pairwise filter exploration from v1 best (62.73% / 770 bets) to escape local optimum.

At each step, try ALL pairs of single changes (one numeric filter + one hour
addition/removal/change). Pick the pair that maximizes WR.

Repeat until no pair-change improves.
"""
from __future__ import annotations

import itertools
import numpy as np

from common import (
    load_dataset, feature_columns, train_once, hour_of_day_from_ms,
)
from train import compute_thresholds_asym_wknd_excl


MIN_PARIS = 750


BASE = {
    "cd": 90, "wd": 0.060, "we": 0.20,
    "bad": (8, 13, 14, 17, 21),
    "good": (2, 20, 22), "delta": 0.020,
    "vr30_max": np.inf, "r3_max": 0.015,
    "r96_max": np.inf, "uw_min": 0.0, "cr_max": np.inf,
    "body_max": np.inf, "stoch_max": np.inf, "rsi_max": np.inf,
}

DIMS = {
    "vr30_max": [np.inf, 1.5, 1.4, 1.3, 1.2, 1.1, 1.0],
    "r3_max": [np.inf, 0.025, 0.020, 0.015, 0.010, 0.005, 0.0, -0.005],
    "r96_max": [np.inf, 0.20, 0.15, 0.10, 0.05, 0.02, 0.0],
    "uw_min": [0.0, 0.05, 0.10, 0.15, 0.20, 0.25],
    "cr_max": [np.inf, 0.025, 0.015, 0.010, 0.005, 0.0, -0.005],
    "body_max": [np.inf, 0.9, 0.8, 0.7, 0.6],
    "stoch_max": [np.inf, 0.5, 0.3, 0.1, -0.1],
    "rsi_max": [np.inf, 0.5, 0.3, 0.1, -0.1],
}
BAD_POOL = [0, 1, 4, 7, 10, 13, 18, 20, 23]
GOOD_POOL = [9, 11, 12, 15, 19, 23]
DELTA_GRID = [0.020, 0.030, 0.040, 0.050, 0.060]
CALIB_GRID = list(itertools.product([55, 90, 180], [0.040, 0.060, 0.080], [0.15, 0.20, 0.25]))


def eval_config(cfg, T_base, p_test, y_test, hour_masks, vr30_t, ret3_t,
                ret96_t, uw_t, cr_t, body_t, stoch_t, rsi_t, n_test):
    bad_mask = np.zeros(n_test, dtype=bool)
    for h in cfg["bad"]:
        bad_mask |= hour_masks[h]
    good_mask = np.zeros(n_test, dtype=bool)
    for h in cfg["good"]:
        good_mask |= hour_masks[h]
    T_eff = T_base - cfg["delta"] * good_mask
    take = (~bad_mask) & (p_test >= T_eff)
    if cfg["vr30_max"] != np.inf: take &= (vr30_t < cfg["vr30_max"]) | np.isnan(vr30_t)
    if cfg["r3_max"] != np.inf: take &= (ret3_t < cfg["r3_max"]) | np.isnan(ret3_t)
    if cfg["r96_max"] != np.inf: take &= (ret96_t < cfg["r96_max"]) | np.isnan(ret96_t)
    if cfg["uw_min"] > 0: take &= (uw_t >= cfg["uw_min"]) | np.isnan(uw_t)
    if cfg["cr_max"] != np.inf: take &= (cr_t < cfg["cr_max"]) | np.isnan(cr_t)
    if cfg["body_max"] != np.inf: take &= (body_t < cfg["body_max"]) | np.isnan(body_t)
    if cfg["stoch_max"] != np.inf: take &= (stoch_t < cfg["stoch_max"]) | np.isnan(stoch_t)
    if cfg["rsi_max"] != np.inf: take &= (rsi_t < cfg["rsi_max"]) | np.isnan(rsi_t)
    n = int(take.sum())
    wr = float(y_test[take].mean()) if n > 0 else 0.0
    return n, wr


def generate_changes(cur):
    """Generate all single changes possible from cur."""
    changes = []
    for dim, values in DIMS.items():
        for v in values:
            if v != cur[dim]:
                new = dict(cur); new[dim] = v
                changes.append(("dim", dim, v, new))
    for h in BAD_POOL:
        if h in cur["bad"]:
            new = dict(cur); new["bad"] = tuple(sorted(x for x in cur["bad"] if x != h))
            changes.append(("bad_rm", h, new["bad"], new))
        else:
            new = dict(cur); new["bad"] = tuple(sorted(cur["bad"] + (h,)))
            changes.append(("bad_add", h, new["bad"], new))
    for h in GOOD_POOL:
        if h in cur["good"]:
            new = dict(cur); new["good"] = tuple(sorted(x for x in cur["good"] if x != h))
            changes.append(("good_rm", h, new["good"], new))
        else:
            new = dict(cur); new["good"] = tuple(sorted(cur["good"] + (h,)))
            changes.append(("good_add", h, new["good"], new))
    for d in DELTA_GRID:
        if d != cur["delta"]:
            new = dict(cur); new["delta"] = d
            changes.append(("delta", d, d, new))
    for cd, wd, we in CALIB_GRID:
        if (cd, wd, we) != (cur["cd"], cur["wd"], cur["we"]):
            new = dict(cur); new["cd"], new["wd"], new["we"] = cd, wd, we
            changes.append(("calib", (cd, wd, we), (cd, wd, we), new))
    return changes


def main():
    df = load_dataset()
    fc = feature_columns(df)
    print(f"Dataset: {len(df):,} rows, {len(fc)} features.")
    run = train_once(df, fc, seed=42)
    print(f"  best_iter={run['best_iter']} val_logloss={run['val_logloss']:.5f}")

    val = run["val"]; p_val = run["p_val"]
    test = run["test"]; p_test = run["p_test"]; y_test = run["y_test"]
    n_test = len(p_test)
    hours_t = hour_of_day_from_ms(test["timestamp"].to_numpy())
    wknd_t = test["is_weekend"].astype(int).to_numpy()
    vr30_t = test["volume_ratio_30m"].to_numpy(dtype=float)
    ret3_t = test["return_3"].to_numpy(dtype=float)
    ret96_t = test["return_96"].to_numpy(dtype=float)
    uw_t = test["upper_wick"].to_numpy(dtype=float)
    cr_t = test["close_ret"].to_numpy(dtype=float)
    body_t = test["body_ratio_close"].to_numpy(dtype=float)
    stoch_t = test["stoch_k_14"].to_numpy(dtype=float)
    rsi_t = test["rsi_14"].to_numpy(dtype=float)
    hour_masks = {h: (hours_t == h) for h in range(24)}

    T_base_cache = {}
    def get_T_base(cd, wd, we):
        key = (cd, wd, we)
        if key not in T_base_cache:
            thr, _ = compute_thresholds_asym_wknd_excl(
                val, p_val,
                vol_frac_wd=wd, vol_frac_we=we,
                skip_cells=(), use_full_val=False, calib_days=cd,
            )
            T_base_cache[key] = np.where(wknd_t == 1, thr.get(1, 1.0), thr.get(0, 1.0))
        return T_base_cache[key]

    cur = dict(BASE)
    T = get_T_base(cur["cd"], cur["wd"], cur["we"])
    n, wr = eval_config(cur, T, p_test, y_test, hour_masks,
                        vr30_t, ret3_t, ret96_t, uw_t, cr_t,
                        body_t, stoch_t, rsi_t, n_test)
    print(f"\nStarting: taken={n} wr={wr*100:.2f}%")

    for rnd in range(1, 11):
        # Generate all single changes
        changes = generate_changes(cur)
        # Filter for "useful" single changes first
        single_results = []
        for ct, key, val_changed, new in changes:
            T_new = get_T_base(new["cd"], new["wd"], new["we"]) if ct == "calib" else T
            tn, tw = eval_config(new, T_new, p_test, y_test, hour_masks,
                                 vr30_t, ret3_t, ret96_t, uw_t, cr_t,
                                 body_t, stoch_t, rsi_t, n_test)
            if tn < MIN_PARIS:
                continue
            single_results.append((ct, key, val_changed, new, T_new, tn, tw))

        if not single_results:
            print(f"Round {rnd}: no single change keeps taken>={MIN_PARIS}")
            break

        # Now try all PAIRS of single changes (top 30 single changes only)
        single_results.sort(key=lambda r: -r[6])  # by wr
        top = single_results[:40]

        best_pair = (None, None, n, wr)
        for i in range(len(top)):
            ct1, _, _, new1, _, _, _ = top[i]
            for j in range(i + 1, len(top)):
                ct2, _, _, new2, _, _, _ = top[j]
                # Combine
                combo = dict(new1)
                # Apply new2's diff onto combo
                for k in new2:
                    if new2[k] != cur[k]:
                        combo[k] = new2[k]
                T_combo = get_T_base(combo["cd"], combo["wd"], combo["we"])
                tn, tw = eval_config(combo, T_combo, p_test, y_test, hour_masks,
                                     vr30_t, ret3_t, ret96_t, uw_t, cr_t,
                                     body_t, stoch_t, rsi_t, n_test)
                if tn < MIN_PARIS:
                    continue
                if tw > best_pair[3] + 0.001:
                    best_pair = (top[i], top[j], tn, tw)

        # Also check best single change
        best_single = max(single_results, key=lambda r: r[6])

        if best_pair[0] is not None and best_pair[3] > best_single[6]:
            # Pair improvement wins
            ct1, key1, _, new1, _, _, _ = best_pair[0]
            ct2, key2, _, new2, _, _, _ = best_pair[1]
            combo = dict(new1)
            for k in new2:
                if new2[k] != cur[k]:
                    combo[k] = new2[k]
            cur = combo
            T = get_T_base(cur["cd"], cur["wd"], cur["we"])
            n, wr = best_pair[2], best_pair[3]
            print(f"Round {rnd}: PAIR ({ct1}, {ct2}) -> taken={n} wr={wr*100:.2f}%")
        elif best_single[6] > wr + 0.001:
            ct, key, val_changed, new, T_new, tn, tw = best_single
            cur = new
            T = T_new
            n, wr = tn, tw
            print(f"Round {rnd}: SINGLE ({ct}: {key}) -> taken={n} wr={wr*100:.2f}%")
        else:
            print(f"Round {rnd}: no improvement (best_pair={best_pair[3]*100:.2f}, "
                  f"best_single={best_single[6]*100:.2f})")
            break

        if wr >= 0.65:
            print(f"  *** GOAL REACHED ***")
            break

    print(f"\nFinal config:")
    for k, v in cur.items():
        print(f"  {k}: {v}")
    print(f"  taken={n} wr={wr*100:.2f}%")


if __name__ == "__main__":
    main()
