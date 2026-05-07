"""
ETAPE 1 + 2 : Comprehension + detection des points faibles de training_data.csv

Produit deux artefacts dans clean_data/ :
  - analysis_report.md   : rapport lisible
  - analysis_report.json : memes findings en JSON, consommable par clean.py

Aucune modification du CSV original.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_classif

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "training_data.csv"
OUT_DIR = ROOT / "clean_data"
OUT_DIR.mkdir(exist_ok=True)

# ----- colonnes par role -----
ID_COLS = ["timestamp"]
LABEL_COL = "label"
CATEGORICAL_COLS = ["seq_len", "seq_dir", "obv_vs_ema", "volatility_state",
                    "hour_utc", "day_of_week", "is_weekend"]
# tout le reste est continu

# Regle metier connue (config.py: BODY_RATIO_MIN = 0.6)
BODY_RATIO_MIN_EXPECTED = 0.6


def load() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def section_overview(df: pd.DataFrame) -> dict:
    n = len(df)
    ts_min = pd.to_datetime(df["timestamp"].min(), unit="ms")
    ts_max = pd.to_datetime(df["timestamp"].max(), unit="ms")
    span_days = (df["timestamp"].max() - df["timestamp"].min()) / 86_400_000

    return {
        "n_rows": int(n),
        "n_cols": int(df.shape[1]),
        "ts_min": str(ts_min),
        "ts_max": str(ts_max),
        "span_days": round(span_days, 1),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
    }


def section_missing(df: pd.DataFrame) -> dict:
    nan_count = df.isna().sum()
    inf_count = {c: int(np.isinf(df[c]).sum())
                 for c in df.columns if np.issubdtype(df[c].dtype, np.number)}
    return {
        "total_nan": int(nan_count.sum()),
        "per_column_nan": {c: int(v) for c, v in nan_count.items() if v > 0},
        "total_inf": int(sum(inf_count.values())),
        "per_column_inf": {c: v for c, v in inf_count.items() if v > 0},
    }


def section_duplicates(df: pd.DataFrame) -> dict:
    feat_cols = [c for c in df.columns if c not in ID_COLS + [LABEL_COL]]

    # Dupliques sur (features+label) : exactement la meme ligne
    full_dup = df.duplicated(subset=feat_cols + [LABEL_COL]).sum()

    # Dupliques sur features mais labels differents = "conflits"
    grouped = df.groupby(feat_cols, as_index=False)[LABEL_COL].nunique()
    conflict_keys = grouped[grouped[LABEL_COL] > 1]
    n_conflicts = len(conflict_keys)
    n_rows_in_conflicts = (df.duplicated(subset=feat_cols, keep=False).sum()
                           - df.duplicated(subset=feat_cols + [LABEL_COL], keep=False).sum())

    # Dupliques (timestamp+seq_len) : grain naturel du dataset
    grain_dup = df.duplicated(subset=["timestamp", "seq_len"]).sum()

    return {
        "exact_duplicates": int(full_dup),
        "feature_only_duplicates": int(df.duplicated(subset=feat_cols).sum()),
        "feature_conflicts_groups": int(n_conflicts),
        "feature_conflicts_rows": int(n_rows_in_conflicts),
        "grain_duplicates_timestamp_seq_len": int(grain_dup),
    }


def section_label_distribution(df: pd.DataFrame) -> dict:
    overall_wr = df[LABEL_COL].mean() * 100
    by_seq_len = df.groupby("seq_len")[LABEL_COL].agg(["count", "mean"])
    by_seq_dir = df.groupby("seq_dir")[LABEL_COL].agg(["count", "mean"])
    by_seq_combo = df.groupby(["seq_len", "seq_dir"])[LABEL_COL].agg(["count", "mean"])

    return {
        "overall_baseline_wr_pct": round(overall_wr, 3),
        "label_balance": df[LABEL_COL].value_counts(normalize=True).to_dict(),
        "by_seq_len": {int(k): {"count": int(v["count"]), "wr_pct": round(float(v["mean"]) * 100, 3)}
                       for k, v in by_seq_len.iterrows()},
        "by_seq_dir": {int(k): {"count": int(v["count"]), "wr_pct": round(float(v["mean"]) * 100, 3)}
                       for k, v in by_seq_dir.iterrows()},
        "by_seq_len_dir": {f"{int(k[0])}-{int(k[1])}": {"count": int(v["count"]), "wr_pct": round(float(v["mean"]) * 100, 3)}
                           for k, v in by_seq_combo.iterrows()},
    }


def section_descriptive(df: pd.DataFrame) -> dict:
    feat_cols = [c for c in df.columns if c not in ID_COLS + [LABEL_COL]]
    out = {}
    for c in feat_cols:
        s = df[c]
        if not np.issubdtype(s.dtype, np.number):
            continue
        s_clean = s.replace([np.inf, -np.inf], np.nan).dropna()
        if len(s_clean) == 0:
            continue
        out[c] = {
            "count": int(s_clean.count()),
            "mean": round(float(s_clean.mean()), 6),
            "std": round(float(s_clean.std()), 6),
            "min": round(float(s_clean.min()), 6),
            "p1": round(float(s_clean.quantile(0.01)), 6),
            "p25": round(float(s_clean.quantile(0.25)), 6),
            "p50": round(float(s_clean.quantile(0.50)), 6),
            "p75": round(float(s_clean.quantile(0.75)), 6),
            "p99": round(float(s_clean.quantile(0.99)), 6),
            "max": round(float(s_clean.max()), 6),
            "skewness": round(float(stats.skew(s_clean)), 4),
            "kurtosis": round(float(stats.kurtosis(s_clean)), 4),
            "n_unique": int(s_clean.nunique()),
        }
    return out


def section_outliers(df: pd.DataFrame) -> dict:
    """Pour chaque feature continue : % de points hors bornes IQR (k=1.5 et k=3)."""
    cont_cols = [c for c in df.columns
                 if c not in ID_COLS + [LABEL_COL] + CATEGORICAL_COLS]
    out = {}
    for c in cont_cols:
        s = df[c].replace([np.inf, -np.inf], np.nan).dropna()
        if len(s) == 0:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lo15, hi15 = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        lo30, hi30 = q1 - 3.0 * iqr, q3 + 3.0 * iqr
        # bornes percentiles extremes (winsorization candidates)
        p_lo, p_hi = s.quantile(0.001), s.quantile(0.999)

        out[c] = {
            "iqr_outliers_pct_k1.5": round(((s < lo15) | (s > hi15)).mean() * 100, 3),
            "iqr_outliers_pct_k3.0": round(((s < lo30) | (s > hi30)).mean() * 100, 3),
            "extreme_pct_outside_p001_p999": round(((s < p_lo) | (s > p_hi)).mean() * 100, 3),
            "p001": round(float(p_lo), 6),
            "p999": round(float(p_hi), 6),
            "abs_max_z": round(float(((s - s.mean()).abs() / (s.std() + 1e-12)).max()), 2),
        }
    return out


def section_low_variance(df: pd.DataFrame) -> dict:
    feat_cols = [c for c in df.columns if c not in ID_COLS + [LABEL_COL]]
    out = {}
    for c in feat_cols:
        s = df[c]
        if not np.issubdtype(s.dtype, np.number):
            continue
        vc = s.value_counts(normalize=True)
        top_share = float(vc.iloc[0]) if len(vc) > 0 else 1.0
        if top_share > 0.95 or s.nunique() < 5:
            out[c] = {
                "top_value": (float(vc.index[0]) if len(vc) > 0 else None),
                "top_share_pct": round(top_share * 100, 3),
                "n_unique": int(s.nunique()),
            }
    return out


def section_correlation(df: pd.DataFrame) -> dict:
    feat_cols = [c for c in df.columns if c not in ID_COLS + [LABEL_COL]]
    num = df[feat_cols].select_dtypes(include=[np.number])
    corr = num.corr().abs()
    pairs = []
    cols = corr.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            v = corr.iloc[i, j]
            if pd.notna(v) and v >= 0.85:
                pairs.append({"a": cols[i], "b": cols[j], "abs_corr": round(float(v), 4)})
    pairs.sort(key=lambda d: -d["abs_corr"])

    # correlation avec le label (Spearman pour capter le monotone non-lineaire)
    label_corr = {}
    y = df[LABEL_COL]
    for c in feat_cols:
        if not np.issubdtype(df[c].dtype, np.number):
            continue
        try:
            r, _ = stats.spearmanr(df[c], y)
            if pd.notna(r):
                label_corr[c] = round(float(r), 4)
        except Exception:
            pass
    label_corr = dict(sorted(label_corr.items(), key=lambda kv: -abs(kv[1])))

    return {
        "high_corr_pairs_abs_ge_0.85": pairs,
        "label_spearman_corr_top": dict(list(label_corr.items())[:15]),
    }


def section_consistency(df: pd.DataFrame) -> dict:
    """Verifie les invariants metiers attendus."""
    issues = {}

    # 1. body_ratio_last doit etre >= BODY_RATIO_MIN (filtre du dataset)
    if "body_ratio_last" in df.columns:
        viol = (df["body_ratio_last"] < BODY_RATIO_MIN_EXPECTED).sum()
        issues["body_ratio_last_below_min"] = int(viol)
        issues["body_ratio_last_min_seen"] = round(float(df["body_ratio_last"].min()), 6)

    # 2. body_ratio + upper_wick + lower_wick doit etre ~1 sur la derniere bougie
    if {"body_ratio_last", "upper_wick", "lower_wick"}.issubset(df.columns):
        s = df["body_ratio_last"] + df["upper_wick"] + df["lower_wick"]
        bad = ((s - 1.0).abs() > 0.01).sum()
        issues["candle_anatomy_violations_gt_1pct"] = int(bad)
        issues["candle_anatomy_max_abs_dev"] = round(float((s - 1.0).abs().max()), 6)

    # 3. seq_len in {1,2,3,4}
    if "seq_len" in df.columns:
        bad_seq = (~df["seq_len"].isin([1, 2, 3, 4])).sum()
        issues["seq_len_out_of_range"] = int(bad_seq)
        issues["seq_len_unique"] = sorted(df["seq_len"].unique().tolist())

    # 4. seq_dir in {0,1}, is_weekend in {0,1}, obv_vs_ema in {0,1}, volatility_state in {0,1,2}
    for col, allowed in [("seq_dir", {0, 1}), ("is_weekend", {0, 1}),
                         ("obv_vs_ema", {0, 1}), ("volatility_state", {0, 1, 2})]:
        if col in df.columns:
            bad = (~df[col].isin(list(allowed))).sum()
            issues[f"{col}_out_of_range"] = int(bad)

    # 5. hour_utc in [0,23], day_of_week in [0,6]
    if "hour_utc" in df.columns:
        issues["hour_utc_out_of_range"] = int(((df["hour_utc"] < 0) | (df["hour_utc"] > 23)).sum())
    if "day_of_week" in df.columns:
        issues["day_of_week_out_of_range"] = int(((df["day_of_week"] < 0) | (df["day_of_week"] > 6)).sum())

    # 6. is_weekend doit etre coherent avec day_of_week
    if {"day_of_week", "is_weekend"}.issubset(df.columns):
        expected = (df["day_of_week"] >= 5).astype(int)
        issues["is_weekend_inconsistent"] = int((df["is_weekend"] != expected).sum())

    return issues


def section_temporal(df: pd.DataFrame) -> dict:
    """Densite temporelle du dataset."""
    df_sorted = df.sort_values("timestamp")
    ts = df_sorted["timestamp"].values

    # gaps en minutes entre lignes consecutives (pour la meme seq_len = 1 typiquement)
    df1 = df_sorted[df_sorted["seq_len"] == 1].sort_values("timestamp")
    gaps_15m = np.diff(df1["timestamp"].values) / 60_000  # en minutes
    big_gaps = (gaps_15m > 60).sum()  # > 1h sans aucune ligne seq_len=1

    # densite par annee
    years = pd.to_datetime(df["timestamp"], unit="ms").dt.year
    by_year = years.value_counts().sort_index().to_dict()

    return {
        "rows_per_year": {int(k): int(v) for k, v in by_year.items()},
        "median_gap_seq_len_1_minutes": round(float(np.median(gaps_15m)), 2) if len(gaps_15m) else None,
        "max_gap_seq_len_1_minutes": round(float(np.max(gaps_15m)), 2) if len(gaps_15m) else None,
        "n_gaps_seq_len_1_over_1h": int(big_gaps),
    }


def section_concept_drift(df: pd.DataFrame) -> dict:
    """Compare distribution des features entre 1ere moitie et derniere moitie (KS test)."""
    feat_cols = [c for c in df.columns
                 if c not in ID_COLS + [LABEL_COL] + CATEGORICAL_COLS]
    df_sorted = df.sort_values("timestamp").reset_index(drop=True)
    half = len(df_sorted) // 2
    a = df_sorted.iloc[:half]
    b = df_sorted.iloc[half:]

    drift = {}
    for c in feat_cols:
        sa = a[c].replace([np.inf, -np.inf], np.nan).dropna()
        sb = b[c].replace([np.inf, -np.inf], np.nan).dropna()
        if len(sa) < 100 or len(sb) < 100:
            continue
        # echantillon pour rester rapide
        sa_s = sa.sample(min(20000, len(sa)), random_state=42)
        sb_s = sb.sample(min(20000, len(sb)), random_state=42)
        ks_stat, p = stats.ks_2samp(sa_s, sb_s)
        drift[c] = {
            "ks_stat": round(float(ks_stat), 4),
            "p_value": round(float(p), 6),
            "mean_first_half": round(float(sa.mean()), 4),
            "mean_second_half": round(float(sb.mean()), 4),
        }
    # tri par KS decroissant
    drift = dict(sorted(drift.items(), key=lambda kv: -kv[1]["ks_stat"]))

    # WR par moitie
    wr_first = a[LABEL_COL].mean() * 100
    wr_second = b[LABEL_COL].mean() * 100

    return {
        "label_wr_first_half_pct": round(wr_first, 3),
        "label_wr_second_half_pct": round(wr_second, 3),
        "feature_drift_ks_top": dict(list(drift.items())[:10]),
    }


def section_mutual_info(df: pd.DataFrame) -> dict:
    feat_cols = [c for c in df.columns if c not in ID_COLS + [LABEL_COL]]
    X = df[feat_cols].select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).fillna(0)
    y = df[LABEL_COL].values
    # echantillon : 50000 lignes max pour vitesse
    if len(X) > 50000:
        idx = np.random.RandomState(42).choice(len(X), 50000, replace=False)
        X_s, y_s = X.iloc[idx], y[idx]
    else:
        X_s, y_s = X, y
    mi = mutual_info_classif(X_s, y_s, random_state=42)
    out = dict(zip(X_s.columns, [round(float(v), 5) for v in mi]))
    out = dict(sorted(out.items(), key=lambda kv: -kv[1]))
    return out


def write_markdown(report: dict, path: Path):
    """Convertit le dict de report en rapport markdown lisible."""
    lines = []
    lines.append("# Rapport d'analyse - training_data.csv\n")

    o = report["overview"]
    lines.append("## 1. Vue d'ensemble\n")
    lines.append(f"- **Lignes** : {o['n_rows']:,}")
    lines.append(f"- **Colonnes** : {o['n_cols']}")
    lines.append(f"- **Plage temporelle** : {o['ts_min']} → {o['ts_max']} ({o['span_days']} jours)")
    lines.append(f"- **Memoire** : {o['memory_mb']} MB\n")

    m = report["missing"]
    lines.append("## 2. Valeurs manquantes / infinies\n")
    lines.append(f"- NaN totaux : **{m['total_nan']}**")
    lines.append(f"- Inf totaux : **{m['total_inf']}**")
    if m["per_column_nan"]:
        lines.append(f"- NaN par colonne : `{m['per_column_nan']}`")
    if m["per_column_inf"]:
        lines.append(f"- Inf par colonne : `{m['per_column_inf']}`")
    lines.append("")

    d = report["duplicates"]
    lines.append("## 3. Doublons / conflits\n")
    lines.append(f"- Doublons exacts (features+label identiques) : **{d['exact_duplicates']:,}**")
    lines.append(f"- Doublons sur features (label peut differer) : **{d['feature_only_duplicates']:,}**")
    lines.append(f"- **Conflits** (memes features → labels differents) : **{d['feature_conflicts_groups']:,} groupes**, **{d['feature_conflicts_rows']:,} lignes**")
    lines.append(f"- Doublons sur grain (timestamp+seq_len) : **{d['grain_duplicates_timestamp_seq_len']:,}**")
    lines.append("")

    lab = report["label_distribution"]
    lines.append("## 4. Distribution du label\n")
    lines.append(f"- WR baseline global : **{lab['overall_baseline_wr_pct']}%**")
    lines.append(f"- Balance : `{lab['label_balance']}`\n")
    lines.append("**WR par seq_len**:")
    lines.append("| seq_len | n | WR |")
    lines.append("|---|---|---|")
    for k, v in lab["by_seq_len"].items():
        lines.append(f"| {k} | {v['count']:,} | {v['wr_pct']}% |")
    lines.append("\n**WR par seq_dir** (1=up, 0=down):")
    lines.append("| seq_dir | n | WR |")
    lines.append("|---|---|---|")
    for k, v in lab["by_seq_dir"].items():
        lines.append(f"| {k} | {v['count']:,} | {v['wr_pct']}% |")
    lines.append("")

    cons = report["consistency"]
    lines.append("## 5. Coherence metier\n")
    for k, v in cons.items():
        flag = " ⚠️" if (isinstance(v, int) and v > 0 and "out_of_range" in k) or \
                       (isinstance(v, int) and v > 0 and "inconsistent" in k) or \
                       (isinstance(v, int) and v > 0 and "violations" in k) or \
                       (isinstance(v, int) and v > 0 and "below_min" in k) else ""
        lines.append(f"- `{k}` : {v}{flag}")
    lines.append("")

    out = report["outliers"]
    lines.append("## 6. Outliers (par feature continue)\n")
    lines.append("| feature | %hors IQR k=1.5 | %hors IQR k=3 | %hors [p0.1, p99.9] | max |z| | p0.1 | p99.9 |")
    lines.append("|---|---|---|---|---|---|---|")
    # tri par % outliers extremes decroissant
    for c, v in sorted(out.items(), key=lambda kv: -kv[1]["extreme_pct_outside_p001_p999"]):
        lines.append(f"| {c} | {v['iqr_outliers_pct_k1.5']}% | {v['iqr_outliers_pct_k3.0']}% | {v['extreme_pct_outside_p001_p999']}% | {v['abs_max_z']} | {v['p001']} | {v['p999']} |")
    lines.append("")

    lv = report["low_variance"]
    lines.append("## 7. Features quasi-constantes / faible variance\n")
    if not lv:
        lines.append("- Aucune detectee.")
    else:
        for c, v in lv.items():
            lines.append(f"- `{c}` : {v['n_unique']} valeurs uniques, top={v['top_value']} ({v['top_share_pct']}%)")
    lines.append("")

    cor = report["correlation"]
    lines.append("## 8. Correlations\n")
    lines.append(f"**Paires |r| ≥ 0.85** ({len(cor['high_corr_pairs_abs_ge_0.85'])} paires) :")
    if not cor["high_corr_pairs_abs_ge_0.85"]:
        lines.append("- Aucune.")
    for p in cor["high_corr_pairs_abs_ge_0.85"]:
        lines.append(f"- `{p['a']}` ↔ `{p['b']}` : |r|={p['abs_corr']}")
    lines.append("\n**Top features Spearman vs label** :")
    for c, v in cor["label_spearman_corr_top"].items():
        lines.append(f"- `{c}` : {v}")
    lines.append("")

    mi = report["mutual_info"]
    lines.append("## 9. Mutual information avec label (top 15)\n")
    for i, (c, v) in enumerate(list(mi.items())[:15]):
        lines.append(f"- `{c}` : {v}")
    lines.append("")

    drift = report["concept_drift"]
    lines.append("## 10. Concept drift (1ere moitie vs 2e moitie)\n")
    lines.append(f"- WR 1ere moitie : {drift['label_wr_first_half_pct']}%")
    lines.append(f"- WR 2e moitie : {drift['label_wr_second_half_pct']}%")
    lines.append("\n**Top 10 features qui drift le plus (KS test)** :")
    lines.append("| feature | KS stat | p-value | mean 1ere moitie | mean 2e moitie |")
    lines.append("|---|---|---|---|---|")
    for c, v in drift["feature_drift_ks_top"].items():
        lines.append(f"| {c} | {v['ks_stat']} | {v['p_value']} | {v['mean_first_half']} | {v['mean_second_half']} |")
    lines.append("")

    t = report["temporal"]
    lines.append("## 11. Densite temporelle\n")
    lines.append(f"- Median gap entre lignes seq_len=1 : {t['median_gap_seq_len_1_minutes']} min")
    lines.append(f"- Max gap : {t['max_gap_seq_len_1_minutes']} min")
    lines.append(f"- Nb gaps > 1h : {t['n_gaps_seq_len_1_over_1h']}")
    lines.append("\n**Lignes par annee** :")
    for y, n in t["rows_per_year"].items():
        lines.append(f"- {y} : {n:,}")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    print(f"Loading {CSV_PATH}...")
    df = load()
    print(f"Loaded {len(df):,} rows.\n")

    report = {}
    print("[1/10] Overview..."); report["overview"] = section_overview(df)
    print("[2/10] Missing/inf..."); report["missing"] = section_missing(df)
    print("[3/10] Duplicates..."); report["duplicates"] = section_duplicates(df)
    print("[4/10] Label distribution..."); report["label_distribution"] = section_label_distribution(df)
    print("[5/10] Descriptive stats..."); report["descriptive"] = section_descriptive(df)
    print("[6/10] Outliers..."); report["outliers"] = section_outliers(df)
    print("[7/10] Low variance..."); report["low_variance"] = section_low_variance(df)
    print("[8/10] Correlations..."); report["correlation"] = section_correlation(df)
    print("[9/10] Consistency..."); report["consistency"] = section_consistency(df)
    print("[10/10] Temporal & drift & MI...")
    report["temporal"] = section_temporal(df)
    report["concept_drift"] = section_concept_drift(df)
    report["mutual_info"] = section_mutual_info(df)

    json_path = OUT_DIR / "analysis_report.json"
    md_path = OUT_DIR / "analysis_report.md"

    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    write_markdown(report, md_path)

    print(f"\nWrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
