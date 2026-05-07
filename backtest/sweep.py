"""
Sweep d'hyperparametres : fait varier UN seul hyperparametre a la fois autour
des valeurs courantes de config.py, et logue chaque run dans logs/runs.csv.

Usage :
    python sweep.py                      # tous les sweeps definis ci-dessous
    python sweep.py --only max_depth     # un seul sweep
    python sweep.py --only max_depth,learning_rate
    python sweep.py --quick              # 3 valeurs par sweep au lieu de 5

Le modele btc_reversal_model.json N'est PAS ecrase pendant les sweeps.
Chaque run est ecrit dans runs.csv avec run_label = "sweep:<hp>=<value>".
"""

import argparse
import time

from train import train_and_evaluate, RUNS_CSV

# ============================================================
#  Definition des sweeps
# ============================================================
# Pour chaque hyperparametre, on liste les valeurs a tester. La valeur courante
# de config.py est incluse pour comparaison directe (ce sera la "reference").
#
# Note sur reg_lambda / reg_alpha / gamma : la valeur False signifie
# "ne pas passer cet argument a XGBoost = utiliser le defaut natif".

SWEEPS_FULL = {
    "max_depth": [3, 4, 5, 6, 8],
    "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
    "min_child_weight": [1, 3, 5, 10, 20],
    "subsample": [0.5, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.5, 0.7, 0.8, 0.9, 1.0],
    "reg_lambda": [False, 0.0, 1.0, 5.0, 10.0],
    "reg_alpha": [False, 0.0, 0.1, 1.0, 5.0],
    "gamma": [False, 0.0, 0.5, 1.0, 5.0],
    "n_estimators": [200, 500, 1000, 2000],  # combine avec early_stopping_rounds
}

# Variante "quick" : 3 valeurs par hyperparametre (la valeur courante + 1 plus
# basse + 1 plus haute), pour avoir un signal grossier en peu de runs.
SWEEPS_QUICK = {
    "max_depth": [3, 4, 6],
    "learning_rate": [0.03, 0.05, 0.1],
    "min_child_weight": [1, 5, 20],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "reg_lambda": [False, 1.0, 10.0],
    "reg_alpha": [False, 0.1, 1.0],
    "gamma": [False, 0.5, 5.0],
    "n_estimators": [500, 1000, 2000],
}


def parse_args():
    parser = argparse.ArgumentParser(description="Sweep one-at-a-time des hyperparametres XGBoost.")
    parser.add_argument(
        "--only",
        default=None,
        help="Liste d'hyperparametres a sweeper, separes par virgules. Ex: --only max_depth,learning_rate. Defaut : tous.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="3 valeurs par sweep au lieu de 5 (~40%% moins de runs).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    sweeps = SWEEPS_QUICK if args.quick else SWEEPS_FULL

    if args.only:
        wanted = {s.strip() for s in args.only.split(",")}
        unknown = wanted - sweeps.keys()
        if unknown:
            raise SystemExit(f"Inconnu : {unknown}. Choix : {sorted(sweeps.keys())}")
        sweeps = {k: v for k, v in sweeps.items() if k in wanted}

    total_runs = sum(len(v) for v in sweeps.values())
    print(f"Sweep config : {len(sweeps)} hyperparametres, {total_runs} runs au total.")
    print(f"Mode : {'quick' if args.quick else 'full'}")
    print(f"Log : {RUNS_CSV}\n")

    t0 = time.time()
    done = 0

    for hp_name, values in sweeps.items():
        print(f"\n{'#' * 60}")
        print(f"# Sweep {hp_name}  ({len(values)} valeurs)")
        print(f"{'#' * 60}")

        for v in values:
            done += 1
            label = f"sweep:{hp_name}={v}"
            print(f"\n[{done}/{total_runs}] {label}")
            try:
                m = train_and_evaluate(
                    overrides={hp_name: v},
                    save_model=False,    # ne pas ecraser le modele de prod
                    verbose=False,       # silencieux pendant le sweep
                    log_run=True,
                    run_label=label,
                )
                wr058 = m.get("wr_at_0.58")
                wr058_str = f"{wr058:.4f}" if wr058 is not None else "N/A"
                print(f"    duration={m['duration_sec']}s  best_iter={m['best_iteration']}  "
                      f"val_logloss={m['val_logloss_best']}  "
                      f"wr_at_0.58={wr058_str}  "
                      f"breakdown_total_wr={m['breakdown_total_wr']}")
            except Exception as e:
                print(f"    FAILED : {type(e).__name__}: {e}")

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"Sweep termine : {done}/{total_runs} runs en {elapsed:.1f}s")
    print(f"Resultats dans {RUNS_CSV}")
    print(f"Pour analyser : python analyze_runs.py")


if __name__ == "__main__":
    main()
