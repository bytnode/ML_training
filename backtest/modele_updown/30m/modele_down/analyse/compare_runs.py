"""
compare_runs.py
===============

Compare les dernieres runs de logs/runs.csv pour evaluer l'impact reel
d'une feature ajoutee/dropped (et pas seulement son delta_logloss isole).

Affiche les colonnes-clefs :
    run_label             - libelle de la run
    adaptive_total_taken  - volume de paris pris sur le test
    adaptive_total_wr     - WR realisee sur les paris pris
    baseline_wr_down      - WR si on prenait TOUTES les bougies (P(down) sur le test)
    val_logloss_best      - logloss best sur val (proxy qualite globale)
    n_features            - nb de features utilisees au train
    duration_sec          - temps de train

Diagnostic :
    taken ↑↑ + wr ↓ legerement   -> trade-off volume/WR, calcule l'EV
    taken ≈  + wr ↓              -> vraie regression, drop la feature
    val_logloss ↓ + test_wr ↓   -> drift val/test, elargir la fenetre

Usage :
    cd backtest/modele_updown/modele_down
    python compare_runs.py            # 2 dernieres runs
    python compare_runs.py 5          # 5 dernieres
    python compare_runs.py --all      # toutes
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
RUNS_CSV = ROOT / "logs" / "runs.csv"

DEFAULT_N = 2

KEY_COLS = [
    "run_label",
    "n_features",
    "adaptive_total_taken",
    "adaptive_total_wr",
    "baseline_wr_down",
    "val_logloss_best",
    "duration_sec",
]


def main(n: int | None) -> int:
    if not RUNS_CSV.exists():
        print(f"ERROR : {RUNS_CSV} introuvable.", file=sys.stderr)
        return 1

    df = pd.read_csv(RUNS_CSV)
    if len(df) == 0:
        print("Aucune run dans logs/runs.csv.", file=sys.stderr)
        return 0

    if n is None or n >= len(df):
        sub = df
    else:
        sub = df.tail(n)

    cols_present = [c for c in KEY_COLS if c in sub.columns]
    if not cols_present:
        print(f"ERROR : aucune colonne clef trouvee dans runs.csv.", file=sys.stderr)
        return 1

    print(f"Source : {RUNS_CSV}  ({len(df):,} runs total, affichant {len(sub)})")
    print()

    # Format compact
    view = sub[cols_present].copy()

    # Arrondi des floats
    for c in ("adaptive_total_wr", "baseline_wr_down", "val_logloss_best"):
        if c in view.columns:
            view[c] = view[c].apply(lambda x: f"{x:.5f}" if pd.notna(x) else "N/A")
    if "duration_sec" in view.columns:
        view["duration_sec"] = view["duration_sec"].apply(
            lambda x: f"{x:.1f}s" if pd.notna(x) else "N/A"
        )

    print(view.to_string(index=False))

    # Diff entre les 2 dernieres runs (si possible)
    if len(sub) >= 2:
        a = sub.iloc[-2]
        b = sub.iloc[-1]
        print()
        print("=" * 60)
        print(f"Diff : {a.get('run_label', '?')}  ->  {b.get('run_label', '?')}")
        print("=" * 60)

        for c in ("n_features", "adaptive_total_taken", "adaptive_total_wr",
                  "baseline_wr_down", "val_logloss_best"):
            if c not in sub.columns:
                continue
            va = a[c]
            vb = b[c]
            if pd.isna(va) or pd.isna(vb):
                continue
            try:
                diff = vb - va
                if c in ("adaptive_total_wr", "baseline_wr_down", "val_logloss_best"):
                    print(f"  {c:<22} {va:.5f}  ->  {vb:.5f}   ({diff:+.5f})")
                else:
                    pct = (diff / va * 100) if va != 0 else 0.0
                    print(f"  {c:<22} {va:>10}  ->  {vb:>10}   ({diff:+}, {pct:+.1f}%)")
            except (TypeError, ValueError):
                print(f"  {c:<22} {va}  ->  {vb}")

        # Verdict EV simple : (taken_b * (wr_b - 0.5)) vs (taken_a * (wr_a - 0.5))
        try:
            ev_a = a["adaptive_total_taken"] * (a["adaptive_total_wr"] - 0.5)
            ev_b = b["adaptive_total_taken"] * (b["adaptive_total_wr"] - 0.5)
            print()
            print(f"  EV (= taken * (wr - 0.5)) :")
            print(f"    {a.get('run_label', '?'):<20} : {ev_a:+.2f}")
            print(f"    {b.get('run_label', '?'):<20} : {ev_b:+.2f}")
            delta = ev_b - ev_a
            verdict = "GAIN" if delta > 0 else ("EGAL" if abs(delta) < 1 else "PERTE")
            print(f"    Diff EV               : {delta:+.2f}   -> {verdict}")
        except (KeyError, TypeError):
            pass

    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--all":
            n = None
        else:
            try:
                n = int(sys.argv[1])
            except ValueError:
                print(f"ERROR : argument doit etre un int ou --all, "
                      f"recu {sys.argv[1]!r}", file=sys.stderr)
                sys.exit(1)
    else:
        n = DEFAULT_N
    sys.exit(main(n))
