"""Test ensemble: train N models with subsample<1, colsample<1, different
seeds, average predictions, then apply calibration. Compare to single model."""
from __future__ import annotations
import numpy as np

from common import (
    load_dataset, feature_columns, train_once,
)
from train import (
    compute_thresholds_asym_wknd_excl, apply_thresholds_asym_hour_boost,
)
from config import (
    ASYM_VOL_FRAC_WEEKDAY, ASYM_VOL_FRAC_WEEKEND, ASYM_CALIB_DAYS,
    ASYM_HOUR_BOOST_GOOD_HOURS, ASYM_HOUR_BOOST_DELTA, ASYM_BAD_HOURS,
)


def evaluate_ensemble(runs):
    """Average p_val and p_test across runs, then apply current calibration."""
    p_val_avg = np.mean([r["p_val"] for r in runs], axis=0)
    p_test_avg = np.mean([r["p_test"] for r in runs], axis=0)
    val = runs[0]["val"]
    test = runs[0]["test"]
    y_test = runs[0]["y_test"]

    th, _ = compute_thresholds_asym_wknd_excl(
        val, p_val_avg,
        vol_frac_wd=ASYM_VOL_FRAC_WEEKDAY, vol_frac_we=ASYM_VOL_FRAC_WEEKEND,
        skip_cells=(), use_full_val=False, calib_days=ASYM_CALIB_DAYS,
    )
    take = apply_thresholds_asym_hour_boost(
        test, p_test_avg, th, skip_cells=(),
        good_hours=ASYM_HOUR_BOOST_GOOD_HOURS,
        hour_boost_delta=ASYM_HOUR_BOOST_DELTA,
        bad_hours=ASYM_BAD_HOURS,
    )
    n = int(take.sum())
    if n == 0:
        return None
    wr = float(y_test[take].mean())
    return {"taken": n, "wr": wr, "ev": n * (wr - 0.5)}


def main():
    df = load_dataset()
    fcols = feature_columns(df)

    # First : single model (current)
    print("Single model (current config, deterministic) :")
    run0 = train_once(df, fcols, seed=42)
    res = evaluate_ensemble([run0])
    print(f"  taken={res['taken']} WR={res['wr']*100:.2f}%")

    # Ensemble : try different subsample/colsample combos
    for subsample in (0.7, 0.8, 0.9):
        for colsample in (0.7, 0.8, 0.9):
            hp = {"subsample": subsample, "colsample_bytree": colsample}
            for n_models in (3, 5, 7):
                runs = []
                for s in range(n_models):
                    r = train_once(df, fcols, seed=42 + s, hp_overrides=hp)
                    runs.append(r)
                res = evaluate_ensemble(runs)
                if res is None:
                    continue
                print(f"  ensemble n={n_models} ss={subsample} cs={colsample}: "
                      f"taken={res['taken']} WR={res['wr']*100:.2f}% EV={res['ev']:+.2f}")


if __name__ == "__main__":
    main()
