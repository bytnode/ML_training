"""
analyze_features.py
===================

Audit complet de l'impact de chaque feature sur le modèle DOWN-only entraîné.
Combine 6 angles d'analyse en un seul outil et émet un verdict KEEP/DROP par
feature, classé du plus utile au moins utile.

Pré-requis :
    - btc_updown_model.json           (sortie de train.py)
    - btc_updown_model.thresholds.json
    - logs/test_predictions.csv       (sortie de train.py)

Sections :
    1. Distribution       — NaN%, mean, std, n_unique, near-constant warning
    2. Class-conditional  — mean_down vs mean_up, Cohen's d (taille d'effet)
    3. Univarié           — AUC, Mutual Information (signal univarié vs label)
    4. Corrélation        — paires |r| > 0.85 (redondance)
    5. XGBoost            — total_gain, n_splits (importance interne)
    6. Permutation        — delta_logloss quand la feature est shufflée

Synthèse :
    Score composite normalisé pour chaque feature (gain + permutation + |AUC-0.5|
    + |Cohen's d| + MI), penalisé si redondante (paire avec |r|>0.85 où l'autre
    a un score plus haut).
    Verdict :
        KEEP_STRONG   (top 30 % des scores, non-redondante)
        KEEP_MEDIUM   (milieu)
        WEAK_REVIEW   (bottom 30 %, signal univarié faible)
        REDUNDANT     (corrélée fortement à une feature plus utile)

Pour un modèle DOWN-only :
    label = 1 => prochaine bougie ROUGE (down)
    label = 0 => prochaine bougie VERTE (up)
    Donc Cohen's d > 0 => feature plus haute en moyenne sur les rows DOWN.

Sorties :
    - stdout                          : sections + tableau de synthèse
    - logs/feature_analysis.md        : résumé human-readable
    - logs/feature_analysis.csv       : tableau complet pour spreadsheet

Usage :
    cd backtest/modele_updown/modele_down
    python analyze_features.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import log_loss, roc_auc_score

from config import MODEL_OUTPUT


ROOT = Path(__file__).resolve().parent
TEST_PRED_CSV = ROOT / "logs" / "test_predictions.csv"
SIDECAR_PATH = Path(MODEL_OUTPUT).with_suffix(".thresholds.json")

OUT_MD  = ROOT / "logs" / "feature_analysis.md"
OUT_CSV = ROOT / "logs" / "feature_analysis.csv"

# Sub-sample taille pour mutual_info (kNN coûteux). Les autres analyses utilisent
# tout le test set.
MI_SAMPLE = 8000

# Seuils de verdict / drapeaux
NEAR_CONSTANT_STD_RATIO = 1e-3   # std / |mean| en dessous = constante effective
HIGH_NAN_PCT            = 5.0    # % de NaN au delà duquel on flag
HIGH_CORR_THRESHOLD     = 0.85   # |r| au-delà = paire considérée redondante
COHENS_D_NEGLIGIBLE     = 0.05   # |d| en dessous = pas de séparation
AUC_FLAT_BAND           = 0.005  # AUC in [0.5 ± 0.005] = pas de signal univarié

# Quantiles pour les buckets de verdict
KEEP_STRONG_QUANTILE = 0.70  # top 30 % du score composite
WEAK_QUANTILE        = 0.30  # bottom 30 % du score composite


# ============================================================================
# Loading
# ============================================================================

def load_model_and_data() -> tuple[xgb.XGBClassifier, pd.DataFrame, list[str]]:
    """Charge le modèle, le sidecar (info) et le test_predictions.csv.
    Retourne (model, test_df, feature_cols).

    feature_cols = liste exacte des features que le modèle a vues à l'entraînement,
    obtenue via model.get_booster().feature_names (source de vérité).
    """
    if not Path(MODEL_OUTPUT).exists():
        raise FileNotFoundError(f"{MODEL_OUTPUT} introuvable. Lance d'abord 'python train.py'.")
    if not TEST_PRED_CSV.exists():
        raise FileNotFoundError(f"{TEST_PRED_CSV} introuvable. Lance d'abord 'python train.py'.")

    model = xgb.XGBClassifier()
    model.load_model(MODEL_OUTPUT)

    df = pd.read_csv(TEST_PRED_CSV)
    feature_cols = list(model.get_booster().feature_names)

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"Le test_predictions.csv n'a pas {len(missing)} colonne(s) attendue(s) "
            f"par le modèle : {missing[:5]}{'...' if len(missing) > 5 else ''}. "
            "Le CSV est probablement out-of-sync — relance 'python train.py'."
        )

    return model, df, feature_cols


# ============================================================================
# Section 1 — Distribution
# ============================================================================

def section_1_distribution(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    n_total = len(df)
    for f in features:
        s = df[f]
        n_nan = int(s.isna().sum())
        nan_pct = 100.0 * n_nan / n_total if n_total else 0.0
        mean = float(s.mean())
        std = float(s.std(ddof=0))
        n_unique = int(s.nunique(dropna=True))
        denom = abs(mean) if abs(mean) > 1e-12 else 1.0
        near_const = (std / denom) < NEAR_CONSTANT_STD_RATIO and n_unique < 50
        rows.append({
            "feature": f,
            "n_nan": n_nan,
            "nan_pct": round(nan_pct, 2),
            "mean": mean,
            "std": std,
            "min": float(s.min()),
            "max": float(s.max()),
            "n_unique": n_unique,
            "near_constant": near_const,
            "high_nan": nan_pct > HIGH_NAN_PCT,
        })
    return pd.DataFrame(rows)


# ============================================================================
# Section 2 — Class-conditional (Cohen's d)
# ============================================================================

def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if a.size < 2 or b.size < 2:
        return 0.0
    sa = a.std(ddof=1)
    sb = b.std(ddof=1)
    pooled_var = ((a.size - 1) * sa**2 + (b.size - 1) * sb**2) / (a.size + b.size - 2)
    pooled_std = pooled_var**0.5
    if pooled_std < 1e-12:
        return 0.0
    return float((a.mean() - b.mean()) / pooled_std)


def section_2_class_conditional(df: pd.DataFrame, features: list[str],
                                label_col: str) -> pd.DataFrame:
    y = df[label_col].to_numpy().astype(int)
    is_down = y == 1
    is_up = y == 0
    rows = []
    for f in features:
        v = df[f].to_numpy(dtype=float)
        d = _cohens_d(v[is_down], v[is_up])
        rows.append({
            "feature": f,
            "mean_down": float(np.nanmean(v[is_down])) if is_down.any() else float("nan"),
            "mean_up":   float(np.nanmean(v[is_up]))   if is_up.any()   else float("nan"),
            "cohens_d":  round(d, 4),
            "abs_d":     round(abs(d), 4),
        })
    return pd.DataFrame(rows)


# ============================================================================
# Section 3 — Univariate predictive power (AUC + MI)
# ============================================================================

def _auc_one(values: np.ndarray, y: np.ndarray) -> float:
    """AUC d'une feature seule. Si AUC < 0.5, la feature prédit en sens
    inverse — on renvoie max(auc, 1 - auc) pour mesurer le pouvoir absolu.
    Renvoie 0.5 si la feature est constante ou full NaN."""
    mask = ~np.isnan(values)
    if mask.sum() < 10:
        return 0.5
    try:
        auc = roc_auc_score(y[mask], values[mask])
    except ValueError:
        return 0.5
    return float(max(auc, 1.0 - auc))


def section_3_univariate(df: pd.DataFrame, features: list[str],
                         label_col: str) -> pd.DataFrame:
    y = df[label_col].to_numpy().astype(int)

    print("    Calcul AUC...", end="", flush=True)
    t0 = time.time()
    auc_rows = []
    for f in features:
        v = df[f].to_numpy(dtype=float)
        auc_rows.append((f, _auc_one(v, y)))
    print(f" {time.time() - t0:.1f}s")

    print(f"    Calcul Mutual Information (sample {MI_SAMPLE} rows)...", end="", flush=True)
    t0 = time.time()
    n = len(df)
    if n > MI_SAMPLE:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, size=MI_SAMPLE, replace=False)
    else:
        idx = np.arange(n)
    X_sample = df[features].iloc[idx].fillna(0.0).to_numpy()
    y_sample = y[idx]
    mi = mutual_info_classif(X_sample, y_sample, random_state=42, n_jobs=-1)
    print(f" {time.time() - t0:.1f}s")

    rows = []
    for (f, auc), m in zip(auc_rows, mi):
        rows.append({
            "feature": f,
            "auc_abs": round(auc, 4),
            "auc_signal": round(2 * (auc - 0.5), 4),  # 0 = no signal, 1 = perfect
            "mutual_info": round(float(m), 6),
        })
    return pd.DataFrame(rows)


# ============================================================================
# Section 4 — Correlations
# ============================================================================

def section_4_correlation(df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Renvoie (corr_pairs_high, max_abs_corr_per_feature).

    corr_pairs_high : toutes les paires (f1, f2) avec |r| > HIGH_CORR_THRESHOLD,
                      triées par |r| descendant.
    max_abs_corr_per_feature : pour chaque feature, son |r| max avec une autre
                               feature (utilisé dans la synthèse pour pénaliser
                               les redondances).
    """
    corr = df[features].corr().abs()
    np.fill_diagonal(corr.values, 0.0)

    pairs = []
    cols = list(corr.columns)
    n = len(cols)
    for i in range(n):
        for j in range(i + 1, n):
            r = float(corr.iat[i, j])
            if r > HIGH_CORR_THRESHOLD:
                pairs.append({"f1": cols[i], "f2": cols[j], "abs_corr": round(r, 4)})

    pairs_df = pd.DataFrame(pairs).sort_values("abs_corr", ascending=False).reset_index(drop=True) \
               if pairs else pd.DataFrame(columns=["f1", "f2", "abs_corr"])

    max_per = pd.DataFrame({
        "feature": features,
        "max_abs_corr": [round(float(corr[f].max()), 4) for f in features],
        "best_corr_partner": [str(corr[f].idxmax()) for f in features],
    })

    return pairs_df, max_per


# ============================================================================
# Section 5 — XGBoost importance
# ============================================================================

def section_5_xgb_importance(model: xgb.XGBClassifier,
                             features: list[str]) -> pd.DataFrame:
    booster = model.get_booster()
    gain = booster.get_score(importance_type="total_gain")
    weight = booster.get_score(importance_type="weight")  # n_splits

    rows = []
    for f in features:
        rows.append({
            "feature": f,
            "total_gain": round(float(gain.get(f, 0.0)), 4),
            "n_splits":   int(weight.get(f, 0)),
        })
    return pd.DataFrame(rows)


# ============================================================================
# Section 6 — Permutation importance
# ============================================================================

def section_6_permutation(model: xgb.XGBClassifier, df: pd.DataFrame,
                          features: list[str], label_col: str) -> pd.DataFrame:
    """Pour chaque feature : shuffle la colonne, re-prédit, mesure delta_logloss.
    Plus le delta est grand, plus la feature est importante."""
    X = df[features].copy()
    y = df[label_col].to_numpy().astype(int)

    base_proba = model.predict_proba(X)[:, 1]
    base_logloss = log_loss(y, base_proba, labels=[0, 1])

    rng = np.random.default_rng(42)
    rows = []
    print(f"    Permutation sur {len(features)} features × {len(df):,} rows... ",
          end="", flush=True)
    t0 = time.time()
    for f in features:
        original = X[f].to_numpy(dtype=float).copy()
        shuffled = original.copy()
        rng.shuffle(shuffled)
        X[f] = shuffled
        try:
            p = model.predict_proba(X)[:, 1]
            ll = log_loss(y, p, labels=[0, 1])
            delta = float(ll - base_logloss)
        finally:
            X[f] = original
        rows.append({"feature": f, "delta_logloss": round(delta, 6)})
    print(f"{time.time() - t0:.1f}s (base_logloss={base_logloss:.4f})")
    return pd.DataFrame(rows)


# ============================================================================
# Synthesis — verdict per feature
# ============================================================================

def _normalize(s: pd.Series) -> pd.Series:
    """Rescale linéairement à [0, 1]. Si toutes les valeurs sont identiques, renvoie 0."""
    s = s.astype(float)
    lo, hi = s.min(), s.max()
    if hi - lo < 1e-12:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - lo) / (hi - lo)


def synthesize(s1: pd.DataFrame, s2: pd.DataFrame, s3: pd.DataFrame,
               s4_max: pd.DataFrame, s5: pd.DataFrame,
               s6: pd.DataFrame) -> pd.DataFrame:
    """Combine les 6 sections en un score composite + verdict par feature."""
    df = s1[["feature", "n_nan", "nan_pct", "mean", "std", "n_unique",
             "near_constant", "high_nan"]].copy()
    df = df.merge(s2[["feature", "cohens_d", "abs_d"]], on="feature")
    df = df.merge(s3[["feature", "auc_abs", "auc_signal", "mutual_info"]], on="feature")
    df = df.merge(s4_max, on="feature")
    df = df.merge(s5[["feature", "total_gain", "n_splits"]], on="feature")
    df = df.merge(s6[["feature", "delta_logloss"]], on="feature")

    # Score composite : moyenne pondérée des 5 axes (chacun normalisé [0,1]).
    n_gain  = _normalize(df["total_gain"])
    n_perm  = _normalize(df["delta_logloss"].clip(lower=0))   # un delta négatif n'est pas utile
    n_auc   = _normalize(df["auc_signal"].clip(lower=0))
    n_d     = _normalize(df["abs_d"])
    n_mi    = _normalize(df["mutual_info"].clip(lower=0))

    df["score_raw"] = (
        0.30 * n_gain +
        0.25 * n_perm +
        0.20 * n_auc +
        0.15 * n_mi +
        0.10 * n_d
    ).round(4)

    # Pénalité de redondance : si max_abs_corr > seuil ET que le partenaire a un
    # score brut PLUS HAUT, on dégrade la feature -> REDUNDANT.
    score_lookup = dict(zip(df["feature"], df["score_raw"]))
    df["redundant_with_better"] = [
        bool(
            df.loc[df["feature"] == f, "max_abs_corr"].iloc[0] > HIGH_CORR_THRESHOLD
            and score_lookup.get(
                df.loc[df["feature"] == f, "best_corr_partner"].iloc[0], 0
            ) > score_lookup[f]
        )
        for f in df["feature"]
    ]

    # Verdict par quantiles (sur le score brut, sans la pénalité de redondance)
    q_strong = df["score_raw"].quantile(KEEP_STRONG_QUANTILE)
    q_weak   = df["score_raw"].quantile(WEAK_QUANTILE)

    def _verdict(row) -> str:
        if row["near_constant"]:
            return "DROP_CONSTANT"
        if row["redundant_with_better"]:
            return "REDUNDANT"
        if row["score_raw"] >= q_strong:
            return "KEEP_STRONG"
        if row["score_raw"] <= q_weak:
            return "WEAK_REVIEW"
        return "KEEP_MEDIUM"

    df["verdict"] = df.apply(_verdict, axis=1)

    # Recommandation actionnable (free-text court)
    def _recommendation(row) -> str:
        if row["verdict"] == "DROP_CONSTANT":
            return "Feature quasi constante — drop direct."
        if row["verdict"] == "REDUNDANT":
            return f"Redondante avec {row['best_corr_partner']} (|r|={row['max_abs_corr']:.2f}) — drop."
        if row["verdict"] == "WEAK_REVIEW":
            extras = []
            if row["auc_signal"] < AUC_FLAT_BAND:
                extras.append(f"AUC={row['auc_abs']:.3f}")
            if row["abs_d"] < COHENS_D_NEGLIGIBLE:
                extras.append(f"|d|={row['abs_d']:.3f}")
            if row["total_gain"] < 1.0:
                extras.append(f"gain={row['total_gain']:.2f}")
            tail = " ; ".join(extras) if extras else "score bas"
            return f"Faible sur tous les axes ({tail}) — candidat au drop."
        if row["verdict"] == "KEEP_STRONG":
            return "Top usefulness — garder."
        return "Signal moyen — garder pour l'instant."

    df["recommendation"] = df.apply(_recommendation, axis=1)

    df = df.sort_values("score_raw", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", df.index + 1)
    return df


# ============================================================================
# Pretty printing
# ============================================================================

def _print_section_header(title: str) -> None:
    line = "=" * 78
    print(f"\n{line}\n  {title}\n{line}")


def _print_table(df: pd.DataFrame, cols: list[str], n: int | None = None,
                 sort_by: str | None = None, ascending: bool = False) -> None:
    view = df.copy()
    if sort_by is not None and sort_by in view.columns:
        view = view.sort_values(sort_by, ascending=ascending)
    if n is not None:
        view = view.head(n)
    print(view[cols].to_string(index=False))


def print_synthesis(synth: pd.DataFrame) -> None:
    _print_section_header("SYNTHÈSE — verdict par feature (trié par score)")

    summary_cols = ["rank", "feature", "verdict", "score_raw",
                    "total_gain", "delta_logloss", "auc_abs", "abs_d",
                    "max_abs_corr", "recommendation"]

    print(synth[summary_cols].to_string(index=False))

    print()
    counts = synth["verdict"].value_counts().to_dict()
    print(f"Compteurs verdicts : "
          f"KEEP_STRONG={counts.get('KEEP_STRONG', 0)}  "
          f"KEEP_MEDIUM={counts.get('KEEP_MEDIUM', 0)}  "
          f"WEAK_REVIEW={counts.get('WEAK_REVIEW', 0)}  "
          f"REDUNDANT={counts.get('REDUNDANT', 0)}  "
          f"DROP_CONSTANT={counts.get('DROP_CONSTANT', 0)}")

    drop_cands = synth[synth["verdict"].isin(
        ["WEAK_REVIEW", "REDUNDANT", "DROP_CONSTANT"]
    )]
    if not drop_cands.empty:
        print(f"\nCandidats au drop ({len(drop_cands)}) :")
        for _, r in drop_cands.iterrows():
            print(f"  - {r['feature']:<30s}  ({r['verdict']:<14s})  {r['recommendation']}")
    else:
        print("\nAucun candidat au drop — toutes les features ont un signal correct.")


# ============================================================================
# Output writing
# ============================================================================

def save_outputs(synth: pd.DataFrame, corr_pairs: pd.DataFrame) -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    # CSV : tableau complet
    synth.to_csv(OUT_CSV, index=False)
    print(f"\nTableau complet écrit : {OUT_CSV.relative_to(ROOT)}")

    # Markdown : résumé human-readable
    lines: list[str] = []
    lines.append(f"# Feature analysis — modele_updown\n")
    lines.append(f"_Genere le {pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M UTC')} sur "
                 f"{len(synth)} features._\n")

    counts = synth["verdict"].value_counts().to_dict()
    lines.append("## Compteurs\n")
    lines.append(f"- **KEEP_STRONG**   : {counts.get('KEEP_STRONG', 0)}")
    lines.append(f"- **KEEP_MEDIUM**   : {counts.get('KEEP_MEDIUM', 0)}")
    lines.append(f"- **WEAK_REVIEW**   : {counts.get('WEAK_REVIEW', 0)}")
    lines.append(f"- **REDUNDANT**     : {counts.get('REDUNDANT', 0)}")
    lines.append(f"- **DROP_CONSTANT** : {counts.get('DROP_CONSTANT', 0)}\n")

    lines.append("## Top 15 features (par score composite)\n")
    lines.append("| rank | feature | verdict | score | gain | dlogloss | |AUC-0.5| | |d| | max_corr |")
    lines.append("|---:|:---|:---|---:|---:|---:|---:|---:|---:|")
    for _, r in synth.head(15).iterrows():
        lines.append(
            f"| {r['rank']} | `{r['feature']}` | {r['verdict']} | "
            f"{r['score_raw']:.3f} | {r['total_gain']:.2f} | "
            f"{r['delta_logloss']:.4f} | {abs(r['auc_signal'])/2:.4f} | "
            f"{r['abs_d']:.3f} | {r['max_abs_corr']:.2f} |"
        )

    lines.append("\n## Candidats au drop\n")
    drop_cands = synth[synth["verdict"].isin(["WEAK_REVIEW", "REDUNDANT", "DROP_CONSTANT"])]
    if drop_cands.empty:
        lines.append("_Aucun._")
    else:
        lines.append("| feature | verdict | recommandation |")
        lines.append("|:---|:---|:---|")
        for _, r in drop_cands.iterrows():
            lines.append(f"| `{r['feature']}` | {r['verdict']} | {r['recommendation']} |")

    if not corr_pairs.empty:
        lines.append("\n## Paires fortement corrélées (|r| > 0.85)\n")
        lines.append("| f1 | f2 | |r| |")
        lines.append("|:---|:---|---:|")
        for _, r in corr_pairs.iterrows():
            lines.append(f"| `{r['f1']}` | `{r['f2']}` | {r['abs_corr']:.3f} |")

    lines.append("\n## Comment dropper une feature\n")
    lines.append("Édite `data.py` :")
    lines.append("- **MTF (suffixée `_30m`/`_1h`/`_4h`/`_1d`/`_5m`)** -> retire la colonne "
                 "de `MTF_KEPT_COLS[<tf>]`. Le calcul + l'écriture sont skippés "
                 "automatiquement.")
    lines.append("- **Tier 1 dérivée** (`dist_ema_20`, `return_3`, `return_96`, "
                 "`hour_sin/cos`, `upper_wick/lower_wick`) -> retire le bloc "
                 "correspondant dans `_add_tier1_derived`.")
    lines.append("- **Tier 2** (`macd*`, `bb_*`, `mfi_norm`, `rv_*`) -> retire le bloc "
                 "correspondant dans `_add_tier2`.")
    lines.append("- **Base 15m** (`open_ret`, `close_ret`, etc.) -> ces colonnes sont "
                 "produites par `compute_normalized_features`. Ne pas les supprimer "
                 "sans réfléchir : elles servent à plusieurs autres features (Tier 1, "
                 "Tier 2, MTF).\n")
    lines.append("Puis relance `python data.py && python train.py` et reviens ici "
                 "pour vérifier le nouveau verdict.")

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Résumé markdown écrit : {OUT_MD.relative_to(ROOT)}")


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    print("Chargement du modèle + test_predictions.csv...")
    model, df, features = load_model_and_data()
    print(f"  modèle : {MODEL_OUTPUT}")
    print(f"  test set : {TEST_PRED_CSV.relative_to(ROOT)}")
    print(f"  features : {len(features)}")
    print(f"  rows    : {len(df):,}")

    if SIDECAR_PATH.exists():
        with open(SIDECAR_PATH) as f:
            meta = json.load(f).get("metadata", {})
        if meta:
            print(f"  trained_at : {meta.get('trained_at', '?')} | "
                  f"target_wr={meta.get('target_wr', '?')} | "
                  f"test_days={meta.get('test_days', '?')}")

    print(f"\n  baseline P(label=1) sur le test : {df['label'].mean():.4f}")

    # ---- Section 1 ----
    _print_section_header("1. DISTRIBUTION (NaN%, mean, std, n_unique)")
    s1 = section_1_distribution(df, features)
    flagged = s1[s1["near_constant"] | s1["high_nan"]]
    if not flagged.empty:
        print("Features avec drapeaux (near_constant ou high_nan) :")
        _print_table(flagged, ["feature", "n_unique", "std", "nan_pct",
                               "near_constant", "high_nan"], sort_by="nan_pct")
    else:
        print("Aucune feature near-constant ou high-nan détectée.")
    print(f"\n(Tableau complet — {len(s1)} features dispo dans {OUT_CSV.name})")

    # ---- Section 2 ----
    _print_section_header("2. CLASS-CONDITIONAL (Cohen's d : taille de séparation)")
    s2 = section_2_class_conditional(df, features, "label")
    print("Top 10 |Cohen's d| :")
    _print_table(s2, ["feature", "mean_down", "mean_up", "cohens_d", "abs_d"],
                 n=10, sort_by="abs_d")
    print(f"\nFeatures avec |d| < {COHENS_D_NEGLIGIBLE} (séparation négligeable) : "
          f"{(s2['abs_d'] < COHENS_D_NEGLIGIBLE).sum()}")

    # ---- Section 3 ----
    _print_section_header("3. UNIVARIÉ (AUC + Mutual Information)")
    s3 = section_3_univariate(df, features, "label")
    print("\nTop 10 par AUC absolu :")
    _print_table(s3, ["feature", "auc_abs", "auc_signal", "mutual_info"],
                 n=10, sort_by="auc_abs")
    print(f"\nFeatures dans la bande plate AUC in [0.5 ± {AUC_FLAT_BAND}] : "
          f"{((s3['auc_abs'] - 0.5).abs() < AUC_FLAT_BAND).sum()}")

    # ---- Section 4 ----
    _print_section_header("4. CORRÉLATION (paires |r| > 0.85)")
    corr_pairs, corr_max = section_4_correlation(df, features)
    if corr_pairs.empty:
        print("Aucune paire avec |r| > 0.85 — pas de redondance forte.")
    else:
        print(f"{len(corr_pairs)} paire(s) hautement corrélée(s) :")
        _print_table(corr_pairs, ["f1", "f2", "abs_corr"])

    # ---- Section 5 ----
    _print_section_header("5. XGBOOST INTERNE (total_gain, n_splits)")
    s5 = section_5_xgb_importance(model, features)
    print("Top 15 par total_gain :")
    _print_table(s5, ["feature", "total_gain", "n_splits"], n=15, sort_by="total_gain")
    n_zero = (s5["total_gain"] == 0).sum()
    print(f"\nFeatures avec total_gain == 0 (jamais utilisées par le modèle) : {n_zero}")

    # ---- Section 6 ----
    _print_section_header("6. PERMUTATION IMPORTANCE (dlogloss quand shufflée)")
    s6 = section_6_permutation(model, df, features, "label")
    print("\nTop 15 par delta_logloss :")
    _print_table(s6, ["feature", "delta_logloss"], n=15, sort_by="delta_logloss")
    n_neg = (s6["delta_logloss"] < 0).sum()
    if n_neg > 0:
        print(f"\nNote : {n_neg} feature(s) avec delta_logloss < 0 (permuter améliore "
              "le logloss = bruit pur, à drop en priorité).")

    # ---- Synthèse ----
    synth = synthesize(s1, s2, s3, corr_max, s5, s6)
    print_synthesis(synth)

    # ---- Sauvegarde ----
    save_outputs(synth, corr_pairs)

    print(f"\nFait. Relis le tableau de synthèse ci-dessus, ouvre "
          f"`{OUT_MD.relative_to(ROOT)}` pour la version markdown, "
          f"ou `{OUT_CSV.relative_to(ROOT)}` pour le tableau complet.")


if __name__ == "__main__":
    main()
