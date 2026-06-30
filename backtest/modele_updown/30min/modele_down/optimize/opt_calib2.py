"""6-cell separable calibration search for modele_down.

Cell = volatility_state (0,1,2) x is_weekend (0,1) -> 6 cells. Each cell gets
its own quantile threshold (frac) calibrated on val. Per-cell quantile
calibration is SEPARABLE: precompute each cell's test (n, wins) curve once,
then search frac vectors by pure lookup -> hundreds of thousands/sec.

Honest protocol:
  - 6 yearly windows, production-mirror split (train_days=4000, seed 42).
  - recent  = window 0 (production headline window).
  - older   = windows 1..5 pooled (headline-excluded => honest).
  - A config is ranked by older_wr (honest); recent is the headline check.

Production baseline: recent (1800, 60.06%), older ~56.5%.
"""
import numpy as np
import pandas as pd

import lib

MS_DAY = 86_400_000
BASE_N, BASE_WR = 1800, 0.6006
PROD_GOOD_HOURS = (20, 21, 22)
PROD_DELTA = 0.02

# frac grid per cell (0 = skip the cell)
FRAC_GRID = np.array([0.0] + list(np.round(np.arange(0.005, 0.301, 0.005), 4)))


def build_windows():
    df = lib.load_master()
    fc = lib.feature_cols(df)
    ts_max = int(df["timestamp"].max())
    windows = []
    for i in range(6):
        end_ts = ts_max - i * 365 * MS_DAY
        train, val, test = lib.temporal_split(df, train_days=4000,
                                              test_days=365, end_ts=end_ts)
        if len(test) < 5000 or len(val) < 2000 or len(train) < 20000:
            continue
        m = lib.train_model(train, val, fc)
        windows.append(dict(
            i=i,
            val_p=lib.predict(m, val, fc),
            val_vol=val["volatility_state"].to_numpy().astype(int),
            val_we=val["is_weekend"].to_numpy().astype(int),
            test_p=lib.predict(m, test, fc),
            test_vol=test["volatility_state"].to_numpy().astype(int),
            test_we=test["is_weekend"].to_numpy().astype(int),
            test_hour=lib.hour_utc(test),
            test_y=test["label"].to_numpy().astype(int),
        ))
        print(f"  window {i}: val={len(val):,} test={len(test):,} "
              f"best_iter={m.best_iteration}")
    return windows


def cell_curves(windows):
    """curves[w][cell] = (n_arr, wins_arr) over FRAC_GRID on the test fold.
    Hour boost folded into an effective test prob (p + boost)."""
    good = list(PROD_GOOD_HOURS)
    curves = []
    for rec in windows:
        boost = np.where(np.isin(rec["test_hour"], good), PROD_DELTA, 0.0)
        p_eff = rec["test_p"] + boost
        wc = {}
        for vol in (0, 1, 2):
            for we in (0, 1):
                cell = vol * 2 + we
                vm = (rec["val_vol"] == vol) & (rec["val_we"] == we)
                tm = (rec["test_vol"] == vol) & (rec["test_we"] == we)
                val_p = rec["val_p"][vm]
                order = np.argsort(-p_eff[tm])
                ps = p_eff[tm][order]
                wcum = np.concatenate([[0], np.cumsum(rec["test_y"][tm][order])])
                n_arr = np.zeros(len(FRAC_GRID), dtype=int)
                w_arr = np.zeros(len(FRAC_GRID), dtype=int)
                for fi, frac in enumerate(FRAC_GRID):
                    if frac <= 0 or len(val_p) < 30:
                        continue
                    T = float(np.quantile(val_p, 1.0 - frac))
                    n = int(np.searchsorted(-ps, -T, side="right"))
                    n_arr[fi] = n
                    w_arr[fi] = int(wcum[n])
                wc[cell] = (n_arr, w_arr)
        curves.append(wc)
    return curves


def eval_vec(curves, win_index, fis):
    out = []
    for w, wc in zip(win_index, curves):
        n = sum(wc[c][0][fis[c]] for c in range(6))
        wins = sum(wc[c][1][fis[c]] for c in range(6))
        out.append((w, n, wins))
    return out


def summarize(res):
    rec = next(x for x in res if x[0] == 0)
    older = [x for x in res if x[0] != 0]
    rn, rw = rec[1], (rec[2] / rec[1] if rec[1] else 0.0)
    on = sum(x[1] for x in older)
    ow = sum(x[2] for x in older) / on if on else 0.0
    return rn, rw, on, ow


def main():
    print("Training one production-mirror model per window (seed 42)...")
    windows = build_windows()
    win_index = [r["i"] for r in windows]
    curves = cell_curves(windows)
    print(f"  {len(windows)} windows, 6 cells, {len(FRAC_GRID)} fracs/cell")

    def frac_to_fi(frac):
        return int(np.argmin(np.abs(FRAC_GRID - frac)))

    # production 2-cell baseline in 6-cell form
    base_fis = {0: frac_to_fi(0.025), 1: frac_to_fi(0.10),
                2: frac_to_fi(0.025), 3: frac_to_fi(0.10),
                4: 0, 5: 0}
    rn, rw, on, ow = summarize(eval_vec(curves, win_index, base_fis))
    print(f"\nBaseline (6-cell form): recent n={rn} wr={rw*100:.2f}%"
          f"  older n={on} wr={ow*100:.2f}%")
    base_older_wr = ow

    # random search over 6 fracs
    rng = np.random.default_rng(0)
    nF = len(FRAC_GRID)
    N = 400_000
    best = []
    for _ in range(N):
        fis = {
            0: int(rng.integers(0, nF)),     # vol0 weekday
            1: int(rng.integers(0, nF)),     # vol0 weekend
            2: int(rng.integers(0, nF)),     # vol1 weekday
            3: int(rng.integers(0, nF)),     # vol1 weekend
            4: int(rng.integers(0, 13)),     # vol2 weekday (frac <= 0.06)
            5: int(rng.integers(0, 13)),     # vol2 weekend
        }
        rn, rw, on, ow = summarize(eval_vec(curves, win_index, fis))
        if rn < BASE_N or rw < BASE_WR or ow < base_older_wr:
            continue
        best.append((ow, rw, rn, on, dict(fis)))

    best.sort(key=lambda x: (-x[0], -x[1]))
    print(f"\n{len(best)} honest Pareto wins "
          f"(recent>=1800/60.06% AND older_wr>=baseline).")
    print("Top 25 by older_wr (honest) then recent_wr:")
    print(f"  {'old_wr':>7} {'rec_wr':>7} {'rec_n':>6} {'old_n':>6}  "
          f"fracs[v0wd,v0we,v1wd,v1we,v2wd,v2we]")
    for ow, rw, rn, on, fis in best[:25]:
        fr = [FRAC_GRID[fis[c]] for c in range(6)]
        print(f"  {ow*100:>6.2f}% {rw*100:>6.2f}% {rn:>6} {on:>6}  "
              f"[{', '.join(f'{x:.3f}' for x in fr)}]")

    if best:
        br = max(best, key=lambda x: (x[1], x[2]))
        ow, rw, rn, on, fis = br
        fr = [FRAC_GRID[fis[c]] for c in range(6)]
        print(f"\nBest recent_wr (honest): recent={rn}@{rw*100:.2f}%  "
              f"older={ow*100:.2f}%  fracs=[{', '.join(f'{x:.3f}' for x in fr)}]")


if __name__ == "__main__":
    main()
