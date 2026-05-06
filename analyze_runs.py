"""
Lit logs/runs.csv et imprime un resume de l'impact de chaque hyperparametre
sur les metriques cles (issu des runs labellises 'sweep:*').

Usage :
    python analyze_runs.py
    python analyze_runs.py --metric breakdown_total_wr
    python analyze_runs.py --only max_depth,learning_rate
"""

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
RUNS_CSV = ROOT / "logs" / "runs.csv"

# Metriques que l'on resume autour de chaque sweep
DEFAULT_METRICS = [
    "wr_at_0.58",
    "n_at_0.58",
    "breakdown_total_wr",
    "breakdown_total_taken",
    "best_iteration",
    "val_logloss_best",
    "duration_sec",
]

# Metrique "score principal" pour le ranking — on l'affichera en tete de tableau
PRIMARY_METRIC = "wr_at_0.58"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--only", default=None,
                   help="Liste d'hyperparametres a afficher, ex: --only max_depth,learning_rate")
    p.add_argument("--metric", default=PRIMARY_METRIC,
                   help=f"Metrique principale pour le ranking (defaut : {PRIMARY_METRIC})")
    p.add_argument("--top", type=int, default=10,
                   help="Nombre de meilleurs runs a afficher dans le top global (defaut : 10)")
    return p.parse_args()


def parse_sweep_label(label: str) -> tuple[str, str] | None:
    """'sweep:max_depth=5' -> ('max_depth', '5'). None si pas un sweep."""
    if not isinstance(label, str) or not label.startswith("sweep:"):
        return None
    rest = label[len("sweep:"):]
    if "=" not in rest:
        return None
    hp, val = rest.split("=", 1)
    return hp, val


def main():
    args = parse_args()

    if not RUNS_CSV.exists():
        raise SystemExit(f"{RUNS_CSV} n'existe pas. Lance d'abord `python sweep.py`.")

    df = pd.read_csv(RUNS_CSV)
    print(f"Loaded {len(df)} runs from {RUNS_CSV}\n")

    # Identifier les sweeps
    parsed = df["run_label"].apply(parse_sweep_label)
    df["sweep_hp"] = parsed.apply(lambda x: x[0] if x else None)
    df["sweep_val"] = parsed.apply(lambda x: x[1] if x else None)

    sweep_hps = sorted(df["sweep_hp"].dropna().unique().tolist())
    if not sweep_hps:
        print("Aucun run de type 'sweep:*' trouve. Lance d'abord `python sweep.py`.")
        return

    if args.only:
        wanted = {s.strip() for s in args.only.split(",")}
        sweep_hps = [h for h in sweep_hps if h in wanted]

    primary = args.metric
    if primary not in df.columns:
        raise SystemExit(f"Metrique inconnue : {primary}. Disponibles : {list(df.columns)}")

    metrics_to_show = [m for m in DEFAULT_METRICS if m in df.columns]
    if primary not in metrics_to_show:
        metrics_to_show.insert(0, primary)

    # ---- Reference : le run "manual" le plus recent (= valeurs config.py courantes) ----
    ref = df[df["run_label"] == "manual"].sort_values("timestamp_iso").tail(1)
    if not ref.empty:
        ref_metric = ref.iloc[0].get(primary)
        ref_str = f"{float(ref_metric):.4f}" if pd.notna(ref_metric) else "N/A"
        print(f"Reference (run 'manual' le plus recent) : {primary} = {ref_str}\n")

    # ---- Tableau par hyperparametre ----
    print("=" * 80)
    print(f"  IMPACT PAR HYPERPARAMETRE  (metrique principale : {primary})")
    print("=" * 80)

    for hp in sweep_hps:
        sub = df[df["sweep_hp"] == hp].copy()
        if sub.empty:
            continue

        # Tri intelligent : numerique si possible, sinon string
        def _try_float(s):
            try:
                return float(s)
            except (TypeError, ValueError):
                return None

        sub["sweep_val_num"] = sub["sweep_val"].apply(_try_float)
        if sub["sweep_val_num"].notna().all():
            sub = sub.sort_values("sweep_val_num")
        else:
            sub = sub.sort_values("sweep_val")

        print(f"\n--- {hp}  ({len(sub)} runs) ---")
        cols = ["sweep_val"] + metrics_to_show
        # On affiche avec un index propre
        view = sub[cols].rename(columns={"sweep_val": hp})
        # Marquer la meilleure ligne sur la metrique principale
        if primary in view.columns and view[primary].notna().any():
            best_idx = view[primary].astype(float).idxmax()
            marker = pd.Series("", index=view.index)
            marker.loc[best_idx] = "*"
            view.insert(0, "best", marker)
        print(view.to_string(index=False))

    # ---- Top N global, tous sweeps confondus ----
    sweep_only = df[df["sweep_hp"].notna()].copy()
    if not sweep_only.empty and primary in sweep_only.columns:
        sweep_only = sweep_only.dropna(subset=[primary])
        if not sweep_only.empty:
            print("\n" + "=" * 80)
            print(f"  TOP {args.top} runs (tous sweeps confondus) selon {primary}")
            print("=" * 80)
            top = sweep_only.nlargest(args.top, primary)[
                ["run_label", primary, "n_at_0.58", "breakdown_total_wr",
                 "best_iteration", "val_logloss_best", "duration_sec"]
            ]
            print(top.to_string(index=False))

    # ---- Hint actionnable ----
    print("\n" + "=" * 80)
    print("  CONSEILS")
    print("=" * 80)
    print("- L'asterisque (*) marque la valeur de l'hyperparametre qui a maximise la metrique principale.")
    print("- Si le 'best' tombe AUX BORDS de la plage testee, etends la plage dans sweep.py et relance.")
    print("- Compare wr_at_0.58 ET breakdown_total_wr : un sweep peut booster l'un sans l'autre.")
    print("- val_logloss_best plus bas = meilleur modele en val. Si wr_at_0.58 monte mais val_logloss aussi, suspect d'overfit du test.")
    print(f"- Pour changer la metrique de ranking : python analyze_runs.py --metric breakdown_total_wr")


if __name__ == "__main__":
    main()
