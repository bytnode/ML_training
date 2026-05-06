"""
ETAPE 3 : Correction des points faibles detectes par analyze.py

Lit training_data.csv (racine du repo), produit training_data_clean.csv
dans le meme dossier (a cote de l'original) + un journal cleaning_log.md
dans clean_data/.

Aucune ligne n'est supprimee. Toutes les corrections sont :
  - winsorization (clip aux quantiles extremes)
  - ajout de features derivees
  - ajout de sample_weight et era_year pour gerer le concept drift

Le CSV original n'est PAS ecrase. train.py n'est pas modifie : c'est a
l'utilisateur de pointer CSV_PATH vers training_data_clean.csv quand
il veut entrainer sur les donnees nettoyees.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CSV_IN = ROOT / "training_data.csv"
CSV_OUT = ROOT / "training_data_clean.csv"
LOG_PATH = Path(__file__).resolve().parent / "cleaning_log.md"

# ----- features a winsorizer (decisions tirees du rapport d'analyse) -----
# Bornes a p0.5 / p99.5 (= 1% des plus extremes capees, symetrique).
# On EXCLUT les features deja bornees (body_ratio_*, *_wick, rsi, atr_ratio dont
# les queues sont moins extremes) — sauf atr_ratio qui a un max |z|=18.89.
WINSOR_COLS = [
    "volume_ratio_last",
    "volume_ratio_avg",
    "obv_slope_10",
    "move_pct",
    "move_atr",
    "rsi_change_5",
    "atr_ratio",
    "dist_ema20",
    "dist_ema50",
    "dist_ema200",
]
WINSOR_LO_Q = 0.005
WINSOR_HI_Q = 0.995

# ----- sample weights : decroissance exponentielle vers le passe -----
# w(t) = exp(-alpha * age_years), normalise pour que mean=1
# alpha=0.15 : une row d'il y a 5 ans pese exp(-0.75)=0.47 ; il y a 10 ans pese 0.22
WEIGHT_ALPHA_PER_YEAR = 0.15


def winsorize(df: pd.DataFrame, cols: list[str], q_lo: float, q_hi: float) -> dict:
    """Clip les colonnes de df aux quantiles q_lo / q_hi. Retourne un dict des bornes appliquees."""
    bounds = {}
    for c in cols:
        if c not in df.columns:
            continue
        lo = df[c].quantile(q_lo)
        hi = df[c].quantile(q_hi)
        n_clipped_lo = int((df[c] < lo).sum())
        n_clipped_hi = int((df[c] > hi).sum())
        df[c] = df[c].clip(lower=lo, upper=hi)
        bounds[c] = {
            "lo": float(lo), "hi": float(hi),
            "n_clipped_low": n_clipped_lo, "n_clipped_high": n_clipped_hi,
        }
    return bounds


def add_era_year(df: pd.DataFrame) -> None:
    df["era_year"] = pd.to_datetime(df["timestamp"], unit="ms").dt.year.astype(int)


def add_sample_weight(df: pd.DataFrame, alpha_per_year: float) -> dict:
    """Poids exponentiel decroissant vers le passe, normalise a mean=1."""
    ts_max = df["timestamp"].max()
    age_years = (ts_max - df["timestamp"]) / (86_400_000 * 365.25)
    raw = np.exp(-alpha_per_year * age_years.values)
    raw_mean = raw.mean()
    df["sample_weight"] = raw / raw_mean  # mean=1 sur le dataset
    return {
        "alpha_per_year": alpha_per_year,
        "weight_min": float(df["sample_weight"].min()),
        "weight_max": float(df["sample_weight"].max()),
        "weight_mean": float(df["sample_weight"].mean()),
        "weight_p10": float(df["sample_weight"].quantile(0.10)),
        "weight_p90": float(df["sample_weight"].quantile(0.90)),
    }


def add_engineered_features(df: pd.DataFrame) -> list[str]:
    """Ajoute des features derivees qui capturent du signal non lineaire ou non capture."""
    new_cols = []

    # 1. Log des volume ratios : reduit le skew brutal sans perdre de signal
    if "volume_ratio_last" in df.columns:
        df["volume_ratio_last_log"] = np.log1p(df["volume_ratio_last"].clip(lower=0))
        new_cols.append("volume_ratio_last_log")
    if "volume_ratio_avg" in df.columns:
        df["volume_ratio_avg_log"] = np.log1p(df["volume_ratio_avg"].clip(lower=0))
        new_cols.append("volume_ratio_avg_log")

    # 2. Asymetrie de meche : signal de rejet directionnel
    if {"upper_wick", "lower_wick"}.issubset(df.columns):
        df["wick_imbalance"] = df["upper_wick"] - df["lower_wick"]
        new_cols.append("wick_imbalance")

    # 3. Alignement des EMAs : signe identique sur 20/50/200 = tendance forte
    if {"dist_ema20", "dist_ema50", "dist_ema200"}.issubset(df.columns):
        s20 = np.sign(df["dist_ema20"])
        s50 = np.sign(df["dist_ema50"])
        s200 = np.sign(df["dist_ema200"])
        df["dist_ema_aligned"] = ((s20 == s50) & (s50 == s200)).astype(int)
        new_cols.append("dist_ema_aligned")

    # 4. RSI zone extreme : overbought/oversold explicite
    if "rsi" in df.columns:
        df["rsi_extreme"] = ((df["rsi"] > 70) | (df["rsi"] < 30)).astype(int)
        new_cols.append("rsi_extreme")

    # 5. Move efficiency : combien de pct par unite d'ATR (deja move_atr, mais
    # avec signe pour distinguer un gros move stable d'un faux mouvement)
    if {"move_pct", "move_atr"}.issubset(df.columns):
        # move_atr est deja en valeur absolue / atr ; on ajoute le signe du move
        df["move_signed_atr"] = np.sign(df["move_pct"]) * df["move_atr"]
        new_cols.append("move_signed_atr")

    # 6. Distance EMA divergence : EMA20 contre EMA200 (microstructure vs macro)
    if {"dist_ema20", "dist_ema200"}.issubset(df.columns):
        df["dist_ema20_vs_ema200"] = df["dist_ema20"] - df["dist_ema200"]
        new_cols.append("dist_ema20_vs_ema200")

    return new_cols


def write_log(stats: dict):
    L = []
    L.append("# Cleaning log\n")
    L.append(f"- Input  : `{CSV_IN}`")
    L.append(f"- Output : `{CSV_OUT}`")
    L.append(f"- Rows   : {stats['n_rows']:,} (inchange — aucune ligne supprimee)")
    L.append(f"- Cols avant : {stats['n_cols_in']}, Cols apres : {stats['n_cols_out']}\n")

    L.append("## Winsorization\n")
    L.append(f"Quantiles utilises : p{WINSOR_LO_Q*100:.1f} / p{WINSOR_HI_Q*100:.1f}\n")
    L.append("| feature | borne basse | borne haute | clipped bas | clipped haut |")
    L.append("|---|---|---|---|---|")
    for c, b in stats["winsor_bounds"].items():
        L.append(f"| {c} | {b['lo']:.6f} | {b['hi']:.6f} | {b['n_clipped_low']:,} | {b['n_clipped_high']:,} |")
    L.append("")

    L.append("## Sample weights\n")
    sw = stats["sample_weight"]
    L.append(f"- alpha = {sw['alpha_per_year']} (decroissance exponentielle par annee de recul)")
    L.append(f"- min / mean / max : {sw['weight_min']:.4f} / {sw['weight_mean']:.4f} / {sw['weight_max']:.4f}")
    L.append(f"- p10 / p90 : {sw['weight_p10']:.4f} / {sw['weight_p90']:.4f}")
    L.append(f"- Une row de 2026 pese ~{sw['weight_max']:.2f} ; une row de 2012 pese ~{sw['weight_min']:.2f}")
    L.append("- Pour utiliser : `model.fit(X, y, sample_weight=df['sample_weight'])`\n")

    L.append("## Features ajoutees\n")
    for c in stats["new_features"]:
        L.append(f"- `{c}`")
    L.append("")

    L.append("## Colonnes meta ajoutees (a exclure des features X dans train.py)\n")
    L.append("- `era_year` : annee de la row, utile comme feature OU comme groupe pour CV temporelle")
    L.append("- `sample_weight` : NE PAS inclure dans X — passer au .fit() via sample_weight=...\n")

    L.append("## Comment l'utiliser dans train.py\n")
    L.append("```python")
    L.append("# remplacer CSV_PATH par 'training_data_clean.csv' dans config.py")
    L.append("df = pd.read_csv(CSV_PATH)")
    L.append("# exclure timestamp, label, ET sample_weight des features")
    L.append("feature_cols = [c for c in df.columns")
    L.append("                if c not in ('timestamp', 'label', 'sample_weight')]")
    L.append("# era_year reste DANS X — c'est une feature volontaire")
    L.append("model.fit(X_train, y_train,")
    L.append("          sample_weight=train['sample_weight'],")
    L.append("          eval_set=[(X_val, y_val)],")
    L.append("          sample_weight_eval_set=[val['sample_weight']])")
    L.append("```")

    LOG_PATH.write_text("\n".join(L), encoding="utf-8")


def main():
    print(f"Loading {CSV_IN}...")
    df = pd.read_csv(CSV_IN)
    df = df.sort_values("timestamp").reset_index(drop=True)
    n_cols_in = df.shape[1]
    print(f"Loaded {len(df):,} rows, {n_cols_in} cols.")

    print("\n[1/4] Winsorization...")
    bounds = winsorize(df, WINSOR_COLS, WINSOR_LO_Q, WINSOR_HI_Q)
    total_clipped = sum(b["n_clipped_low"] + b["n_clipped_high"] for b in bounds.values())
    print(f"  Clipped {total_clipped:,} valeurs au total sur {len(WINSOR_COLS)} colonnes.")

    print("\n[2/4] Feature engineering...")
    new_features = add_engineered_features(df)
    print(f"  Ajoute {len(new_features)} features : {new_features}")

    print("\n[3/4] Era year + sample weights...")
    add_era_year(df)
    sw_stats = add_sample_weight(df, WEIGHT_ALPHA_PER_YEAR)
    print(f"  era_year range : {df['era_year'].min()} -> {df['era_year'].max()}")
    print(f"  sample_weight : min={sw_stats['weight_min']:.4f}, mean={sw_stats['weight_mean']:.4f}, max={sw_stats['weight_max']:.4f}")

    # On reordonne : timestamp, era_year, [features + nouvelles + label], sample_weight a la fin
    label_col = df.pop("label")
    sw_col = df.pop("sample_weight")
    era_col = df.pop("era_year")
    # remettre dans l'ordre : timestamp, era_year, ...features..., label, sample_weight
    df.insert(1, "era_year", era_col)
    df["label"] = label_col
    df["sample_weight"] = sw_col

    n_cols_out = df.shape[1]

    print(f"\n[4/4] Writing {CSV_OUT}...")
    df.to_csv(CSV_OUT, index=False)
    print(f"  Written {len(df):,} rows, {n_cols_out} cols.")

    stats = {
        "n_rows": len(df),
        "n_cols_in": n_cols_in,
        "n_cols_out": n_cols_out,
        "winsor_bounds": bounds,
        "sample_weight": sw_stats,
        "new_features": new_features,
    }
    write_log(stats)
    print(f"  Wrote log to {LOG_PATH}")
    print("\nDone. Le CSV original n'a pas ete touche.")


if __name__ == "__main__":
    main()
