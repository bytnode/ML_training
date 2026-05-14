"""
features.py
===========

Lit le master 1h (parquet ou CSV) produit par data.py et renvoie le DataFrame
d'entrainement pret pour train.py.

Toutes les colonnes derivees (label, volatility_state, is_weekend) sont
PRECALCULEES par data.py et lues directement depuis le parquet. Plus de
calcul on-the-fly ici, plus de relecture SQLite, plus de boucle Python.

API publique :
    build_training_dataframe(csv_path, verbose=True) -> pd.DataFrame

Etapes :
  1. Read parquet/CSV
  2. Drop les WARMUP premieres rows (NaN dans les rolling features)
  3. Drop les rows sans label valide (flat candle, derniere row, NaN warmup)
  4. Cast label en int
  5. Renomme timestamp_ms -> timestamp pour compat train.py
"""

from pathlib import Path

import pandas as pd

from config import WARMUP


# Colonnes du parquet qui ne sont PAS des features d'entraînement.
# Tout le reste (= features Tier 0/1/2 + is_weekend + volatility_state) est
# automatiquement utilise comme feature par train.py.
NON_FEATURE_COLS = {"timestamp_ms", "timestamp", "close", "volume"}


def build_training_dataframe(csv_path, verbose: bool = True) -> pd.DataFrame:
    """
    Lit le master 1h (parquet ou CSV selon l'extension) et renvoie le
    DataFrame d'entrainement pret a passer a train.py.

    Le master DOIT contenir les colonnes pre-calculees par data.py :
        - label              (int 0|1, NaN pour les rows a skip)
        - volatility_state   (int 0|1|2, bucket key adaptive threshold)
        - is_weekend         (int 0|1, bucket key adaptive threshold)

    Si une de ces colonnes manque, l'erreur explique qu'il faut re-runner
    'python data.py' (la version actuelle precalcule tout).

    Raises:
        FileNotFoundError si le fichier n'existe pas.
        ValueError       si l'extension n'est ni .csv ni .parquet.
        RuntimeError     si une colonne pre-calculee manque.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path.resolve()} introuvable. Lance d'abord 'python data.py' "
            f"pour generer le master 1h."
        )

    suffix = csv_path.suffix.lower()
    if verbose:
        print(f"Loading {csv_path}...")
    if suffix == ".parquet":
        df = pd.read_parquet(csv_path)
    elif suffix == ".csv":
        df = pd.read_csv(csv_path)
    else:
        raise ValueError(
            f"Extension non supportee pour {csv_path} : '{suffix}'. "
            f"Attendu : .csv ou .parquet."
        )

    if verbose:
        print(f"  {len(df):,} rows.")

    # Verifie que les colonnes pre-calculees sont presentes
    required = ["timestamp_ms", "label", "volatility_state", "is_weekend"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"Colonnes pre-calculees manquantes dans le parquet : {missing}. "
            f"Re-run 'python data.py' (la version actuelle precalcule label, "
            f"volatility_state et is_weekend dans le parquet)."
        )

    # WARMUP : drop les premieres rows (rolling features ont NaN au debut)
    n_total = len(df)
    df = df.iloc[WARMUP:].reset_index(drop=True)
    n_after_warmup = len(df)

    # Drop les rows sans label valide (flat candle, derniere row, ou NaN warmup)
    n_label_nan = int(df["label"].isna().sum())
    df = df.dropna(subset=["label"]).reset_index(drop=True)
    df["label"] = df["label"].astype(int)

    # Standardise le nom de la colonne timestamp pour train.py
    if "timestamp" in df.columns:
        df = df.drop(columns=["timestamp"])
    df = df.rename(columns={"timestamp_ms": "timestamp"})

    if verbose:
        print(f"\nWARMUP={WARMUP} : {n_total:,} -> {n_after_warmup:,} rows")
        if n_label_nan > 0:
            print(f"  (skipped {n_label_nan:,} rows : flat candle ou derniere row)")
        n_down = int((df["label"] == 1).sum())
        n_up = len(df) - n_down
        if len(df) > 0:
            p_down = n_down / len(df)
            print(f"  Baseline down/up : {n_down:,} down / {n_up:,} up "
                  f"(P(down) = {p_down:.4f})")
        n_features = len([c for c in df.columns if c not in {"timestamp", "label"}])
        print(f"Generated {len(df):,} training examples with {n_features} features.")

    return df
