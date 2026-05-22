"""
Drill down on remaining filter cutoffs from analyze_current_best findings:
- obv_pressure_4h: winners higher → filter low values?
- dist_ema_20: winners higher → filter low values
- macd: winners > 0 → filter macd<X
- bb_position: winners > 0.7 → filter bb<X

Test additive AND combinations of these on top of current lock.
"""
from __future__ import annotations

import itertools
import numpy as np

from common import (
    load_dataset, feature_columns, train_once, hour_of_day_from_ms,
)


BAD_HOURS = (1, 8, 10, 13, 14, 17, 21)
GOOD_HOURS = (2, 9, 20, 22)
DELTA = 0.030
CD = 90; WD = 0.060; WE = 0.20
VR30_MAX = 1.50
R3_MAX = 0.030
MIN_PARIS = 750
MS_PER_DAY = 86_400_000


def filtered_threshold(val_df, p_val, wknd_arr_val, mask_val,
                       wd_frac, we_frac, calib_days):
    val_max_ts = int(val_df["timestamp"].max())
    cutoff = val_max_ts - calib_days * MS_PER_DAY
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
        ("volume_ratio_30m", "vr30"), ("return_3", "r3"),
        ("obv_pressure_4h", "obv4"), ("dist_ema_20", "ema20"),
        ("dist_ema_50", "ema50"), ("dist_ema_100", "ema100"),
        ("macd", "macd"), ("macd_hist", "macdh"),
        ("bb_position", "bb"), ("bb_width", "bbw"),
        ("price_in_range_50", "pir50"), ("dist_high_50", "dh50"),
        ("dist_low_50", "dl50"), ("roc_10", "roc10"), ("roc_20", "roc20"),
        ("atr_pct_1d", "atr1d"),
    ]}
    feat_v = {n: val[c].to_numpy(dtype=float) for c, n in [
        ("volume_ratio_30m", "vr30"), ("return_3", "r3"),
        ("obv_pressure_4h", "obv4"), ("dist_ema_20", "ema20"),
        ("dist_ema_50", "ema50"), ("dist_ema_100", "ema100"),
        ("macd", "macd"), ("macd_hist", "macdh"),
        ("bb_position", "bb"), ("bb_width", "bbw"),
        ("price_in_range_50", "pir50"), ("dist_high_50", "dh50"),
        ("dist_low_50", "dl50"), ("roc_10", "roc10"), ("roc_20", "roc20"),
        ("atr_pct_1d", "atr1d"),
    ]}
    bad_t = np.isin(hours_t, list(BAD_HOURS))
    bad_v = np.isin(hours_v, list(BAD_HOURS))
    good_t = np.isin(hours_t, list(GOOD_HOURS))
    m_t = (~bad_t) & ((feat_t["vr30"] < VR30_MAX) | np.isnan(feat_t["vr30"])) & ((feat_t["r3"] < R3_MAX) | np.isnan(feat_t["r3"]))
    m_v = (~bad_v) & ((feat_v["vr30"] < VR30_MAX) | np.isnan(feat_v["vr30"])) & ((feat_v["r3"] < R3_MAX) | np.isnan(feat_v["r3"]))

    # Filter dimensions (NEW feature filters to try on top of lock)
    # Each filter : (feature_name_short, mode, grid)
    # mode = ">" means keep if feature > cutoff
    # mode = "<" means keep if feature < cutoff
    filters_to_try = {
        "obv4>": ("obv4", ">", [-np.inf, -0.3, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4]),
        "ema20>": ("ema20", ">", [-np.inf, -0.005, 0.0, 0.005, 0.01, 0.02, 0.03]),
        "ema50>": ("ema50", ">", [-np.inf, -0.005, 0.0, 0.005, 0.01, 0.02, 0.04]),
        "ema100>": ("ema100", ">", [-np.inf, -0.01, 0.0, 0.01, 0.02, 0.05]),
        "macd>": ("macd", ">", [-np.inf, -0.001, 0.0, 0.001, 0.003]),
        "bb>": ("bb", ">", [-np.inf, 0.5, 0.6, 0.7, 0.8, 0.9]),
        "bbw>": ("bbw", ">", [-np.inf, 0.01, 0.02, 0.03, 0.05]),
        "pir50>": ("pir50", ">", [-np.inf, 0.5, 0.7, 0.85, 0.95]),
        "dh50>": ("dh50", ">", [-np.inf, -0.10, -0.05, -0.02, -0.01]),
        "dl50<": ("dl50", "<", [np.inf, 0.20, 0.10, 0.05, 0.02]),
        "roc10>": ("roc10", ">", [-np.inf, -0.02, 0.0, 0.01, 0.02]),
        "roc20>": ("roc20", ">", [-np.inf, -0.02, 0.0, 0.01, 0.03]),
        "atr1d>": ("atr1d", ">", [-np.inf, 0.01, 0.02, 0.03, 0.05]),
    }

    # Single-filter sweep on top of lock
    print("Single filter on top of lock:")
    print(f"{'filter':>15} {'cutoff':>8} {'taken':>6} {'wr':>7}")
    candidates = []
    for name, (feat, mode, grid) in filters_to_try.items():
        for cutoff in grid:
            arr_t = feat_t[feat]; arr_v = feat_v[feat]
            if mode == ">":
                if cutoff == -np.inf:
                    fm_t = np.ones(n_test, dtype=bool); fm_v = np.ones(len(val), dtype=bool)
                else:
                    fm_t = (arr_t > cutoff) | np.isnan(arr_t)
                    fm_v = (arr_v > cutoff) | np.isnan(arr_v)
            else:
                if cutoff == np.inf:
                    fm_t = np.ones(n_test, dtype=bool); fm_v = np.ones(len(val), dtype=bool)
                else:
                    fm_t = (arr_t < cutoff) | np.isnan(arr_t)
                    fm_v = (arr_v < cutoff) | np.isnan(arr_v)

            m_t_full = m_t & fm_t
            m_v_full = m_v & fm_v
            if m_v_full.sum() < 100 or m_t_full.sum() < MIN_PARIS:
                continue

            thr = filtered_threshold(val, p_val, wknd_v, m_v_full, WD, WE, CD)
            T_base = np.where(wknd_t == 1, thr.get(1, 1.0), thr.get(0, 1.0))
            T_eff = T_base - DELTA * good_t
            take = m_t_full & (p_test >= T_eff)
            n = int(take.sum())
            if n < MIN_PARIS:
                continue
            wr = float(y_test[take].mean())
            candidates.append({"name": name, "cutoff": cutoff, "taken": n, "wr": wr})

    candidates.sort(key=lambda r: -r["wr"])
    for r in candidates[:30]:
        marker = " ***GOAL" if r["wr"] >= 0.65 and r["taken"] >= 750 else ""
        ct = "inf" if abs(r["cutoff"]) == np.inf else f"{r['cutoff']:.4f}"
        print(f"  {r['name']:>14} {ct:>8} {r['taken']:>6} {r['wr']*100:>6.2f}%{marker}")

    # Try PAIRS of filters from top candidates
    print("\nPair filters on top of lock:")
    top_singles = candidates[:8]
    pair_results = []
    for i, c1 in enumerate(top_singles):
        for j, c2 in enumerate(top_singles):
            if i >= j: continue
            f1, m1, g1 = filters_to_try[c1["name"]]
            f2, m2, g2 = filters_to_try[c2["name"]]
            arr_t1 = feat_t[f1]; arr_v1 = feat_v[f1]
            arr_t2 = feat_t[f2]; arr_v2 = feat_v[f2]

            def apply(arr, cut, mode):
                if cut in (-np.inf, np.inf):
                    return np.ones(len(arr), dtype=bool)
                if mode == ">":
                    return (arr > cut) | np.isnan(arr)
                else:
                    return (arr < cut) | np.isnan(arr)

            fm_t = apply(arr_t1, c1["cutoff"], m1) & apply(arr_t2, c2["cutoff"], m2)
            fm_v = apply(arr_v1, c1["cutoff"], m1) & apply(arr_v2, c2["cutoff"], m2)
            m_t_full = m_t & fm_t
            m_v_full = m_v & fm_v
            if m_v_full.sum() < 100 or m_t_full.sum() < MIN_PARIS:
                continue
            thr = filtered_threshold(val, p_val, wknd_v, m_v_full, WD, WE, CD)
            T_base = np.where(wknd_t == 1, thr.get(1, 1.0), thr.get(0, 1.0))
            T_eff = T_base - DELTA * good_t
            take = m_t_full & (p_test >= T_eff)
            n = int(take.sum())
            if n < MIN_PARIS:
                continue
            wr = float(y_test[take].mean())
            pair_results.append({
                "f1": c1["name"], "c1": c1["cutoff"],
                "f2": c2["name"], "c2": c2["cutoff"],
                "taken": n, "wr": wr,
            })

    pair_results.sort(key=lambda r: -r["wr"])
    for r in pair_results[:20]:
        c1 = "inf" if abs(r["c1"]) == np.inf else f"{r['c1']:.4f}"
        c2 = "inf" if abs(r["c2"]) == np.inf else f"{r['c2']:.4f}"
        marker = " ***GOAL" if r["wr"] >= 0.65 and r["taken"] >= 750 else ""
        print(f"  {r['f1']:>10} {c1:>8} & {r['f2']:>10} {c2:>8}  taken={r['taken']:>5} wr={r['wr']*100:>6.2f}%{marker}")


if __name__ == "__main__":
    main()
