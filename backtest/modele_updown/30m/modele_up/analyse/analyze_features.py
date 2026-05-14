"""
analyze_features.py
===================

Analyse detaillee de l'impact de chaque feature sur l'entrainement du modele.
Pour chaque feature, mesure 4 dimensions independantes et reporte les
chiffres bruts (pas de verdict, juste les faits).

Methodologie (4 dimensions) :

  1. UNIVARIATE         (sans modele, feature seule vs label)
       - Distribution stats : mean, std, skew, range, %NaN
       - Cohen's d : separation des distributions class 0 / class 1
       - Mutual information (binnee) avec le label

  2. CORRELATION        (vs autres features)
       - Top 3 features les plus correlees
       - Max |r| dans le panier (flag d'info si > 0.85)

  3. PERMUTATION        (avec modele baseline, sur test set)
       - Permute la feature sur le test, mesure delta logloss global
       - Permute la feature sur le subset des paris pris
         -> isole l'impact sur la queue de distribution

  4. ABLATION FULL      (production-parity)
       - Retrain sans la feature (1000 trees + adaptive threshold)
       - Mesure delta_val_logloss, delta_test_logloss
       - Mesure delta_taken, delta_WR, delta_EV (ce que ca change en prod)

Output :
  - stdout : tableau classe par |delta_EV| decroissant (= impact mesurable)
  - logs/feature_analysis.md : rapport detaille feature par feature avec
    TOUTES les metriques

Cout :
  ~ N+1 trainings (N features + 1 baseline). Avec N=26 et 1000 trees + early
  stopping, compter 4-7 minutes. Multi-seed multiplie le cout.

Usage :
    cd backtest/modele_updown/1h/modele_up
    python analyse/analyze_features.py                       # toutes les features
    python analyse/analyze_features.py macd bb_position      # subset
    python analyse/analyze_features.py --multi-seed 3        # moyenne sur 3 seeds
    python analyse/analyze_features.py --no-ablation         # rapide (pas de retrain)

La calibration adaptative utilisee pour l'ablation suit `config.ADAPTIVE_STRATEGY`
(V1 = `asym_wknd_excl` ou V2 = `asym_wknd_excl_hour_boost`) afin que les delta_EV
mesures refletent la decision de production.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# Le fichier vit dans `modele_up/analyse/` mais importe `config`, `features`,
# `train` depuis `modele_up/`. On ajoute le parent au sys.path pour que
# `python analyse/analyze_features.py` fonctionne depuis `modele_up/`.
_PKG_ROOT = Path(__file__).resolve().parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, mutual_info_score
from xgboost import XGBClassifier

from config import (
    DB_PATH, TRAIN_DAYS, TEST_DAYS, VAL_FRACTION, XGB_RANDOM_STATE,
    ADAPTIVE_STRATEGY,
)
from features import build_training_dataframe
from train import (
    build_hyperparams,
    _resolve_scale_pos_weight,
    compute_thresholds_asym_wknd_excl,
    apply_thresholds_asym_wknd_excl,
    apply_thresholds_asym_hour_boost,
)


ROOT = Path(__file__).resolve().parent
# Rapport ecrit dans `modele_up/logs/` (= meme dossier que runs.csv et
# test_predictions.csv), pas dans `analyse/logs/`.
LOGS_DIR = _PKG_ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)
REPORT_PATH = LOGS_DIR / "feature_analysis.md"

EXCLUDED = {"timestamp", "label", "volatility_state", "is_weekend"}


def _apply_decision(test_df, p_test, thresholds_per_wknd):
    """Dispatch sur ADAPTIVE_STRATEGY pour mirroir `train.train_and_evaluate`."""
    if ADAPTIVE_STRATEGY == "asym_wknd_excl_hour_boost":
        return apply_thresholds_asym_hour_boost(test_df, p_test, thresholds_per_wknd)
    if ADAPTIVE_STRATEGY == "asym_wknd_excl":
        return apply_thresholds_asym_wknd_excl(test_df, p_test, thresholds_per_wknd)
    raise ValueError(
        f"ADAPTIVE_STRATEGY={ADAPTIVE_STRATEGY!r} non supporte par analyze_features."
    )


# ============================================================================
#  Data structures
# ============================================================================

@dataclass
class FeatureMetrics:
    name: str
    # Univariate
    mean: float = np.nan
    std: float = np.nan
    skew: float = np.nan
    range_min: float = np.nan
    range_max: float = np.nan
    nan_pct: float = 0.0
    cohens_d: float = np.nan
    mutual_info: float = np.nan
    # Correlation
    top_corr_features: list = field(default_factory=list)
    max_abs_corr: float = np.nan
    # Native + permutation (vs baseline)
    gain_importance: float = np.nan
    rank_in_baseline: int = -1
    perm_delta_logloss_global: float = np.nan
    perm_delta_logloss_taken_subset: float = np.nan
    # Ablation (full retrain)
    abl_val_logloss: float = np.nan
    abl_test_logloss: float = np.nan
    abl_taken: int = -1
    abl_wr: float = np.nan
    abl_ev: float = np.nan
    delta_val_logloss: float = np.nan
    delta_test_logloss: float = np.nan
    delta_taken: int = 0
    delta_wr: float = np.nan
    delta_ev: float = np.nan


# ============================================================================
#  Helpers : split + train production-parity
# ============================================================================

def _split(df: pd.DataFrame):
    df = df.sort_values("timestamp").reset_index(drop=True)
    ms_per_day = 86_400_000
    ts_max = df["timestamp"].max()
    test_cutoff = ts_max - TEST_DAYS * ms_per_day
    test_start = int(df["timestamp"].searchsorted(test_cutoff, side="left"))
    if TRAIN_DAYS is None:
        tv_start = 0
    else:
        tv_cutoff = test_cutoff - TRAIN_DAYS * ms_per_day
        tv_start = int(df["timestamp"].searchsorted(tv_cutoff, side="left"))
    n_tv = test_start - tv_start
    val_size = int(n_tv * VAL_FRACTION)
    train_end = test_start - val_size
    return (df.iloc[tv_start:train_end],
            df.iloc[train_end:test_start],
            df.iloc[test_start:])


def _build_model(y_train, seed):
    hp = build_hyperparams()
    optional_kwargs = {}
    if hp["reg_lambda"] is not False: optional_kwargs["reg_lambda"] = hp["reg_lambda"]
    if hp["reg_alpha"]  is not False: optional_kwargs["reg_alpha"]  = hp["reg_alpha"]
    if hp["gamma"]      is not False: optional_kwargs["gamma"]      = hp["gamma"]
    spw = _resolve_scale_pos_weight(hp["scale_pos_weight"], y_train)
    if spw is not None:
        optional_kwargs["scale_pos_weight"] = spw
    return XGBClassifier(
        n_estimators=hp["n_estimators"],
        max_depth=hp["max_depth"],
        learning_rate=hp["learning_rate"],
        min_child_weight=hp["min_child_weight"],
        subsample=hp["subsample"],
        colsample_bytree=hp["colsample_bytree"],
        objective="binary:logistic",
        eval_metric="logloss",
        early_stopping_rounds=hp["early_stopping_rounds"],
        random_state=seed,
        n_jobs=-1,
        **optional_kwargs,
    )


def _train_and_measure(df, feature_cols, seed):
    train, val, test = _split(df)
    X_train, y_train = train[feature_cols], train["label"]
    X_val,   y_val   = val[feature_cols],   val["label"]
    X_test,  y_test  = test[feature_cols],  test["label"]

    model = _build_model(y_train, seed)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    best = int(model.best_iteration)
    val_ll = float(model.evals_result()["validation_0"]["logloss"][best])

    p_val  = model.predict_proba(X_val)[:, 1]
    p_test = model.predict_proba(X_test)[:, 1]
    test_ll = float(log_loss(y_test, p_test))

    thresholds_per_wknd, _ = compute_thresholds_asym_wknd_excl(val, p_val)
    take = _apply_decision(test, p_test, thresholds_per_wknd)
    n_taken = int(take.sum())
    wr = float(y_test.to_numpy()[take].mean()) if n_taken > 0 else 0.0
    ev = n_taken * (wr - 0.5)

    return {
        "model":        model,
        "best_iter":    best,
        "val_logloss":  val_ll,
        "test_logloss": test_ll,
        "taken":        n_taken,
        "wr":           wr,
        "ev":           ev,
        "p_test":       p_test,
        "take":         take,
        "X_test":       X_test,
        "y_test":       y_test,
    }


def _train_multiseed(df, feature_cols, n_seeds):
    if n_seeds == 1:
        return _train_and_measure(df, feature_cols, seed=XGB_RANDOM_STATE)
    seeds = [XGB_RANDOM_STATE + i for i in range(n_seeds)]
    runs = [_train_and_measure(df, feature_cols, seed=s) for s in seeds]
    return {
        "model":        runs[0]["model"],
        "best_iter":    np.mean([r["best_iter"]    for r in runs]),
        "val_logloss":  np.mean([r["val_logloss"]  for r in runs]),
        "test_logloss": np.mean([r["test_logloss"] for r in runs]),
        "taken":        np.mean([r["taken"]        for r in runs]),
        "wr":           np.mean([r["wr"]           for r in runs]),
        "ev":           np.mean([r["ev"]           for r in runs]),
        "ev_std":       np.std([r["ev"]            for r in runs], ddof=0),
        "p_test":       runs[0]["p_test"],
        "take":         runs[0]["take"],
        "X_test":       runs[0]["X_test"],
        "y_test":       runs[0]["y_test"],
    }


# ============================================================================
#  Univariate metrics
# ============================================================================

def _cohens_d(values: pd.Series, labels: pd.Series) -> float:
    a = values[labels == 1].dropna()
    b = values[labels == 0].dropna()
    if len(a) < 30 or len(b) < 30:
        return np.nan
    n1, n2 = len(a), len(b)
    s1, s2 = a.std(ddof=1), b.std(ddof=1)
    pooled = np.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
    if pooled == 0 or np.isnan(pooled):
        return np.nan
    return float((a.mean() - b.mean()) / pooled)


def _mutual_info_binned(values: pd.Series, labels: pd.Series, n_bins: int = 20) -> float:
    s = values.dropna()
    if len(s) < 100:
        return np.nan
    try:
        bins = pd.qcut(s, q=n_bins, labels=False, duplicates="drop")
    except ValueError:
        return np.nan
    mask = ~values.isna()
    return float(mutual_info_score(bins, labels[mask].astype(int)))


def _univariate_stats(values: pd.Series, labels: pd.Series) -> dict:
    s = values.dropna()
    if len(s) == 0:
        return {
            "mean": np.nan, "std": np.nan, "skew": np.nan,
            "range_min": np.nan, "range_max": np.nan,
            "nan_pct": 100.0, "cohens_d": np.nan, "mutual_info": np.nan,
        }
    return {
        "mean":        float(s.mean()),
        "std":         float(s.std()),
        "skew":        float(s.skew()) if len(s) > 3 else np.nan,
        "range_min":   float(s.min()),
        "range_max":   float(s.max()),
        "nan_pct":     float(values.isna().mean() * 100),
        "cohens_d":    _cohens_d(values, labels),
        "mutual_info": _mutual_info_binned(values, labels),
    }


# ============================================================================
#  Correlation
# ============================================================================

def _top_correlations(df, target_feature, fcols, k=3):
    others = [c for c in fcols if c != target_feature]
    sub = df[fcols].dropna()
    if len(sub) < 100:
        return [], np.nan
    corrs = sub[others].corrwith(sub[target_feature])
    sorted_corrs = corrs.abs().sort_values(ascending=False).head(k)
    top = [(c, float(corrs[c])) for c in sorted_corrs.index]
    max_abs = float(sorted_corrs.iloc[0]) if len(sorted_corrs) > 0 else np.nan
    return top, max_abs


# ============================================================================
#  Permutation impact (vs baseline model)
# ============================================================================

def _permutation_impact(baseline_metrics, target_col, n_repeats=3):
    model  = baseline_metrics["model"]
    X_test = baseline_metrics["X_test"]
    y_test = baseline_metrics["y_test"]
    take   = baseline_metrics["take"]
    p_base = baseline_metrics["p_test"]

    ll_base_global = float(log_loss(y_test, p_base))
    subset_mask = np.asarray(take, dtype=bool)
    if subset_mask.sum() > 1:
        ll_base_taken = float(log_loss(y_test[subset_mask], p_base[subset_mask]))
    else:
        ll_base_taken = np.nan

    rng = np.random.default_rng(0)
    deltas_global, deltas_taken = [], []
    for _ in range(n_repeats):
        X_perm = X_test.copy()
        X_perm[target_col] = rng.permutation(X_perm[target_col].values)
        p_perm = model.predict_proba(X_perm)[:, 1]
        deltas_global.append(float(log_loss(y_test, p_perm)) - ll_base_global)
        if not np.isnan(ll_base_taken) and subset_mask.sum() > 1:
            deltas_taken.append(
                float(log_loss(y_test[subset_mask], p_perm[subset_mask])) - ll_base_taken
            )

    return {
        "perm_global": float(np.mean(deltas_global)),
        "perm_taken":  float(np.mean(deltas_taken)) if deltas_taken else np.nan,
    }


# ============================================================================
#  Output : stdout + markdown
# ============================================================================

def _print_summary(metrics_list, baseline, n_total):
    print()
    print("=" * 115)
    print(f"  FEATURE ANALYSIS - {len(metrics_list)} features measured")
    print("=" * 115)
    print(f"  Baseline (toutes les features) : taken={int(baseline['taken'])}  "
          f"WR={baseline['wr']*100:.2f}%  EV={baseline['ev']:+.2f}  "
          f"val_ll={baseline['val_logloss']:.6f}  test_ll={baseline['test_logloss']:.6f}")
    print()
    # Trie par |delta_EV| decroissant (impact en valeur absolue, peu importe signe)
    sorted_m = sorted(
        metrics_list,
        key=lambda x: -(abs(x.delta_ev) if not np.isnan(x.delta_ev) else -1),
    )

    print(f"  {'feature':<24}{'dEV':>9}{'dWR':>9}{'dTaken':>9}"
          f"{'dVal_ll':>14}{'perm_global':>14}{'perm_taken':>14}"
          f"{'gain_rank':>11}{'max_corr':>10}{'cohen_d':>9}{'MI':>10}")
    print("  " + "-" * 130)
    for m in sorted_m:
        rank_str = f"{m.rank_in_baseline}/{n_total}" if m.rank_in_baseline >= 0 else "N/A"
        d_ev   = f"{m.delta_ev:+.2f}"   if not np.isnan(m.delta_ev) else "N/A"
        d_wr   = f"{m.delta_wr * 100:+.2f}pt" if not np.isnan(m.delta_wr) else "N/A"
        d_tk   = f"{m.delta_taken:+,}"
        d_val  = f"{m.delta_val_logloss:+.6f}" if not np.isnan(m.delta_val_logloss) else "N/A"
        p_glb  = f"{m.perm_delta_logloss_global:+.6f}" if not np.isnan(m.perm_delta_logloss_global) else "N/A"
        p_tkn  = f"{m.perm_delta_logloss_taken_subset:+.6f}" if not np.isnan(m.perm_delta_logloss_taken_subset) else "N/A"
        corr   = f"{m.max_abs_corr:.3f}" if not np.isnan(m.max_abs_corr) else "N/A"
        cd     = f"{m.cohens_d:+.3f}"    if not np.isnan(m.cohens_d) else "N/A"
        mi     = f"{m.mutual_info:.5f}"  if not np.isnan(m.mutual_info) else "N/A"
        print(f"  {m.name:<24}{d_ev:>9}{d_wr:>9}{d_tk:>9}{d_val:>14}{p_glb:>14}{p_tkn:>14}"
              f"{rank_str:>11}{corr:>10}{cd:>9}{mi:>10}")

    print()
    print(f"  Legende :")
    print(f"    dEV         = baseline_EV - ablated_EV   (positif = la feature contribue a l'EV)")
    print(f"    dWR         = baseline_WR - ablated_WR")
    print(f"    dTaken      = baseline_taken - ablated_taken")
    print(f"    dVal_ll     = baseline_val_logloss - ablated_val_logloss   (negatif = la feature aide val)")
    print(f"    perm_global = delta logloss en permutant cette feature sur tout le test")
    print(f"    perm_taken  = delta logloss en permutant sur le subset des paris pris")
    print(f"    gain_rank   = rang de l'importance native XGBoost (gain) parmi {n_total}")
    print(f"    max_corr    = max |corr| avec une autre feature  (> 0.85 = candidate redundante)")
    print(f"    cohen_d     = effect size feature vs label  (>0.5 = large)")
    print(f"    MI          = mutual info binnee feature vs label")
    print(f"\n  Rapport detaille : {REPORT_PATH}")


def _write_markdown(metrics_list, baseline, args, n_features_total, t_total):
    # Trie par |delta_EV| decroissant
    sorted_m = sorted(
        metrics_list,
        key=lambda x: -(abs(x.delta_ev) if not np.isnan(x.delta_ev) else -1),
    )
    lines = []
    lines.append(f"# Feature analysis - modele_up (30m)\n")
    lines.append(f"**Date :** {pd.Timestamp.utcnow().isoformat(timespec='seconds')}  ")
    lines.append(f"**Multi-seed :** {args.multi_seed}  ")
    lines.append(f"**Ablation :** {'enabled' if not args.no_ablation else 'disabled'}  ")
    lines.append(f"**Calibration :** `{ADAPTIVE_STRATEGY}`  ")
    lines.append(f"**Duration :** {t_total:.1f}s  ")
    lines.append(f"**Features evaluated :** {len(metrics_list)} / {n_features_total}\n")

    lines.append(f"## Baseline (avec toutes les features)\n")
    lines.append(f"- best_iter    : {int(baseline['best_iter'])}")
    lines.append(f"- val_logloss  : {baseline['val_logloss']:.6f}")
    lines.append(f"- test_logloss : {baseline['test_logloss']:.6f}")
    lines.append(f"- taken        : {int(baseline['taken'])}")
    lines.append(f"- WR           : {baseline['wr']*100:.2f}%")
    lines.append(f"- EV           : {baseline['ev']:+.2f}")
    if "ev_std" in baseline:
        lines.append(f"- EV std (multi-seed) : {baseline['ev_std']:.2f}")
    lines.append("")

    lines.append(f"## Tableau recapitulatif (trie par |delta_EV| decroissant)\n")
    lines.append("| Rang | Feature | dEV | dTaken | dWR | dVal_ll | dTest_ll | perm_global | perm_taken | gain_rank | max_corr | cohens_d | MI | NaN% |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for i, m in enumerate(sorted_m, 1):
        rank_str = f"{m.rank_in_baseline}/{n_features_total}" if m.rank_in_baseline >= 0 else "N/A"
        d_ev   = f"{m.delta_ev:+.2f}" if not np.isnan(m.delta_ev) else "N/A"
        d_tk   = f"{m.delta_taken:+,}"
        d_wr   = f"{m.delta_wr * 100:+.2f}pt" if not np.isnan(m.delta_wr) else "N/A"
        d_val  = f"{m.delta_val_logloss:+.6f}" if not np.isnan(m.delta_val_logloss) else "N/A"
        d_test = f"{m.delta_test_logloss:+.6f}" if not np.isnan(m.delta_test_logloss) else "N/A"
        p_glb  = f"{m.perm_delta_logloss_global:+.6f}" if not np.isnan(m.perm_delta_logloss_global) else "N/A"
        p_tkn  = f"{m.perm_delta_logloss_taken_subset:+.6f}" if not np.isnan(m.perm_delta_logloss_taken_subset) else "N/A"
        corr   = f"{m.max_abs_corr:.3f}" if not np.isnan(m.max_abs_corr) else "N/A"
        cd     = f"{m.cohens_d:+.3f}" if not np.isnan(m.cohens_d) else "N/A"
        mi     = f"{m.mutual_info:.5f}" if not np.isnan(m.mutual_info) else "N/A"
        nan    = f"{m.nan_pct:.2f}%"
        lines.append(f"| {i} | `{m.name}` | {d_ev} | {d_tk} | {d_wr} | {d_val} | {d_test} | {p_glb} | {p_tkn} | {rank_str} | {corr} | {cd} | {mi} | {nan} |")
    lines.append("")

    lines.append(f"## Definitions des metriques\n")
    lines.append(f"- **dEV** = `baseline_EV - ablated_EV`. Mesure en points combien la feature contribue a l'EV reel quand on retrain sans elle.")
    lines.append(f"- **dTaken** = `baseline_taken - ablated_taken`. Variation du nombre de paris pris.")
    lines.append(f"- **dWR** = `baseline_WR - ablated_WR`. Variation de la WR sur les paris pris.")
    lines.append(f"- **dVal_ll / dTest_ll** = variation du logloss val/test entre baseline et ablated. Negatif = la feature aide.")
    lines.append(f"- **perm_global** = delta logloss en permutant la feature sur tout le test set. Mesure d'impact global.")
    lines.append(f"- **perm_taken** = delta logloss en permutant uniquement sur le subset des paris pris. Mesure l'impact dans la queue de distribution (ou se trouvent les bets).")
    lines.append(f"- **gain_rank** = rang de l'importance native XGBoost (gain) parmi toutes les features.")
    lines.append(f"- **max_corr** = correlation absolue maximale avec une autre feature. Au-dela de 0.85 indique une forte redondance.")
    lines.append(f"- **cohens_d** = separation des distributions class 0 vs class 1 (effect size). Convention : >0.2 small, >0.5 medium, >0.8 large.")
    lines.append(f"- **MI** = mutual information binnee entre la feature et le label.")
    lines.append(f"- **NaN%** = proportion de valeurs manquantes (warmup + edge cases).\n")

    lines.append(f"## Detail par feature\n")
    for i, m in enumerate(sorted_m, 1):
        lines.append(f"### {i}. `{m.name}`\n")
        lines.append(f"**Distribution**")
        lines.append(f"- mean = {m.mean:.5f}, std = {m.std:.5f}, skew = {m.skew:+.3f}")
        lines.append(f"- range = [{m.range_min:+.4f}, {m.range_max:+.4f}], NaN = {m.nan_pct:.2f}%\n")
        lines.append(f"**Relation avec le label (univariate)**")
        cd_size = "large" if abs(m.cohens_d) > 0.5 else "medium" if abs(m.cohens_d) > 0.2 else "small"
        lines.append(f"- Cohen's d   = {m.cohens_d:+.4f}  ({cd_size})")
        lines.append(f"- Mutual info = {m.mutual_info:.5f}\n")
        if m.top_corr_features:
            lines.append(f"**Top 3 correlations avec autres features**")
            for fname, r in m.top_corr_features:
                marker = " (>0.85 = forte redondance)" if abs(r) > 0.85 else ""
                lines.append(f"- `{fname}` : r = {r:+.3f}{marker}")
            lines.append("")
        lines.append(f"**Importance dans le baseline**")
        lines.append(f"- gain XGBoost = {m.gain_importance:.5f}  (rang {m.rank_in_baseline}/{n_features_total})\n")
        lines.append(f"**Permutation (modele baseline, feature melangee aleatoirement)**")
        lines.append(f"- delta logloss sur tout le test = {m.perm_delta_logloss_global:+.6f}")
        lines.append(f"- delta logloss sur le subset des paris pris = {m.perm_delta_logloss_taken_subset:+.6f}\n")
        if not args.no_ablation:
            lines.append(f"**Ablation : retrain SANS cette feature**")
            lines.append(f"- val_logloss   : {m.abl_val_logloss:.6f}   (baseline = {baseline['val_logloss']:.6f}, delta = {m.delta_val_logloss:+.6f})")
            lines.append(f"- test_logloss  : {m.abl_test_logloss:.6f}   (baseline = {baseline['test_logloss']:.6f}, delta = {m.delta_test_logloss:+.6f})")
            lines.append(f"- taken         : {m.abl_taken}  (baseline = {int(baseline['taken'])}, delta = {m.delta_taken:+,})")
            lines.append(f"- WR            : {m.abl_wr*100:.2f}%  (baseline = {baseline['wr']*100:.2f}%, delta = {m.delta_wr*100:+.2f}pt)")
            lines.append(f"- EV            : {m.abl_ev:+.2f}  (baseline = {baseline['ev']:+.2f}, **delta = {m.delta_ev:+.2f}**)\n")
        lines.append("---\n")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


# ============================================================================
#  Main
# ============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("features", nargs="*",
                        help="Features cibles. Vide = toutes.")
    parser.add_argument("--multi-seed", type=int, default=1, metavar="N",
                        help="Moyenne sur N seeds (default 1).")
    parser.add_argument("--no-ablation", action="store_true",
                        help="Skip le retrain ablation (rapide mais verdict moins fiable).")
    parser.add_argument("--perm-repeats", type=int, default=3,
                        help="Nb de repetitions pour la permutation (default 3).")
    args = parser.parse_args()

    t_start = time.time()

    print(f"Loading parquet...")
    df = build_training_dataframe(DB_PATH, verbose=False)
    fcols = [c for c in df.columns if c not in EXCLUDED]
    print(f"  {len(df):,} rows x {df.shape[1]} cols  ({len(fcols)} features training)")

    targets = args.features if args.features else fcols
    missing = [f for f in targets if f not in fcols]
    if missing:
        print(f"\nERROR : feature(s) absente(s) : {missing}", file=sys.stderr)
        return 1

    print(f"\nMode : {'ABLATION FULL' if not args.no_ablation else 'PERMUTATION ONLY'} "
          f"({args.multi_seed} seed{'s' if args.multi_seed > 1 else ''})")
    strat_label = "asym_wknd_excl_hour_boost V2" if ADAPTIVE_STRATEGY == "asym_wknd_excl_hour_boost" else "asym_wknd_excl V1"
    print(f"Hyperparams : 1000 trees + early stopping + {strat_label} calibration")
    train, val, test = _split(df)
    print(f"Split : train={len(train):,}  val={len(val):,}  test={len(test):,}")

    print(f"\n[1/3] Training baseline ({len(fcols)} features)... ", end="", flush=True)
    t0 = time.time()
    baseline = _train_multiseed(df, fcols, args.multi_seed)
    print(f"{time.time() - t0:.1f}s")
    print(f"  baseline : taken={int(baseline['taken'])}  WR={baseline['wr']*100:.2f}%  "
          f"EV={baseline['ev']:+.2f}  val_ll={baseline['val_logloss']:.6f}")

    imp = dict(zip(fcols, baseline["model"].feature_importances_))
    ranking = sorted(fcols, key=lambda c: -imp[c])

    print(f"\n[2/3] Univariate + correlation + permutation ({len(targets)} features)...")
    metrics_list: list[FeatureMetrics] = []
    full_train = pd.concat([train, val], ignore_index=True)

    for i, f in enumerate(targets, 1):
        m = FeatureMetrics(name=f)

        stats = _univariate_stats(full_train[f], full_train["label"])
        for k, v in stats.items():
            setattr(m, k, v)

        top_corr, max_corr = _top_correlations(full_train, f, fcols, k=3)
        m.top_corr_features = top_corr
        m.max_abs_corr = max_corr

        m.gain_importance = float(imp[f])
        m.rank_in_baseline = ranking.index(f) + 1

        perm = _permutation_impact(baseline, f, n_repeats=args.perm_repeats)
        m.perm_delta_logloss_global = perm["perm_global"]
        m.perm_delta_logloss_taken_subset = perm["perm_taken"]

        metrics_list.append(m)
        if i % 5 == 0 or i == len(targets):
            print(f"  ({i}/{len(targets)}) processed")

    if not args.no_ablation:
        n_total = len(targets)
        print(f"\n[3/3] Full ablation ({n_total} retrain)... ETA ~{n_total * 5}-{n_total * 30}s")
        for i, m in enumerate(metrics_list, 1):
            ablated_cols = [c for c in fcols if c != m.name]
            t0 = time.time()
            res = _train_multiseed(df, ablated_cols, args.multi_seed)
            elapsed = time.time() - t0
            m.abl_val_logloss  = res["val_logloss"]
            m.abl_test_logloss = res["test_logloss"]
            m.abl_taken        = int(res["taken"])
            m.abl_wr           = res["wr"]
            m.abl_ev           = res["ev"]
            m.delta_val_logloss  = baseline["val_logloss"]  - m.abl_val_logloss
            m.delta_test_logloss = baseline["test_logloss"] - m.abl_test_logloss
            m.delta_taken        = int(baseline["taken"])   - m.abl_taken
            m.delta_wr           = baseline["wr"]           - m.abl_wr
            m.delta_ev           = baseline["ev"]           - m.abl_ev
            print(f"  ({i}/{n_total}) {m.name:<24} dEV={m.delta_ev:+8.2f}   "
                  f"dTaken={m.delta_taken:+,}   dWR={m.delta_wr*100:+.2f}pt   "
                  f"({elapsed:.1f}s)")
    else:
        print(f"\n[3/3] Ablation skippee (--no-ablation)")

    t_total = time.time() - t_start
    _print_summary(metrics_list, baseline, len(fcols))
    _write_markdown(metrics_list, baseline, args, len(fcols), t_total)
    print(f"\nTotal duration : {t_total:.1f}s  (~{t_total/60:.1f} min)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
