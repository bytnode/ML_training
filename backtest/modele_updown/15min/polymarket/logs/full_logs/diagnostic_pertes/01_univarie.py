"""
01_univarie.py — Balayage univarie des sources de pertes.

Pour chaque predicteur connu a la decision :
  - variables categorielles  -> taux de perte par categorie,
  - variables continues      -> taux de perte par decile.
Chaque sous-groupe est teste (binomial vs baseline), corrige en multi-tests
(Benjamini-Hochberg), et juge ROBUSTE seulement si la sur-perte tient sur les
deux moities temporelles du backtest.

Analyse menee 3 fois : GLOBAL, puis paris UP seuls, puis paris DOWN seuls
(les deux modeles ont des features et des comportements distincts).

Sorties : resultats/01_*.csv  +  synthese console.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

import lib

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 120)

CAT_FEATURES = ["dir", "volatility_state", "is_weekend", "hour", "dow", "bar_green"]


def scan_categoriel(df: pd.DataFrame, label: str) -> pd.DataFrame:
    base = lib.baseline_loss(df)
    print(f"\n{'#' * 78}\n# CATEGORIEL — {label}  (baseline perte {base:.4f}, "
          f"n={len(df)})\n{'#' * 78}")
    allrows = []
    for feat in CAT_FEATURES:
        if feat not in df.columns or df[feat].nunique() < 2:
            continue
        t = lib.loss_table(df, df[feat], baseline=base, min_n=25)
        if t.empty:
            continue
        t.insert(0, "feature", feat)
        allrows.append(t)
        print(f"\n--- {feat} ---")
        show = t[["cat", "n", "loss_rate", "wr", "lift", "ecart_pts",
                  "pval", "robuste"]].copy()
        show["loss_rate"] = show["loss_rate"].map(lambda x: f"{x:.3f}")
        show["wr"] = show["wr"].map(lambda x: f"{x:.3f}")
        show["lift"] = show["lift"].map(lambda x: f"{x:.2f}")
        show["ecart_pts"] = show["ecart_pts"].map(lambda x: f"{x:+.1f}")
        show["pval"] = show["pval"].map(lambda x: f"{x:.1e}")
        print(show.to_string(index=False))
    return pd.concat(allrows, ignore_index=True) if allrows else pd.DataFrame()


def scan_continu(df: pd.DataFrame, label: str, feats: list[str]) -> tuple:
    base = lib.baseline_loss(df)
    print(f"\n{'#' * 78}\n# CONTINU — {label}  ({len(feats)} features, "
          f"deciles)\n{'#' * 78}")

    # mutual information feature <-> loss
    X = df[feats].fillna(df[feats].median())
    mi = mutual_info_classif(X, df["loss"], random_state=0,
                             discrete_features=False)
    mi = dict(zip(feats, mi))

    bin_rows, feat_rows = [], []
    for feat in feats:
        s = df[feat]
        if s.nunique() < 5:
            continue
        bins = lib.qbins(s, 10)
        t = lib.loss_table(df, bins, baseline=base, min_n=25, sort=False)
        if t.empty:
            continue
        t.insert(0, "feature", feat)
        bin_rows.append(t)

        # correlation rang-perte (Spearman) sur les valeurs brutes
        rho = df[[feat]].assign(loss=df["loss"]).corr("spearman").iloc[0, 1]
        worst = t.loc[t["loss_rate"].idxmax()]
        best = t.loc[t["loss_rate"].idxmin()]
        # monotonie : correlation entre rang du decile et taux de perte
        t_ord = t.reset_index(drop=True)
        mono = np.corrcoef(np.arange(len(t_ord)), t_ord["loss_rate"])[0, 1] \
            if len(t_ord) > 2 else np.nan
        feat_rows.append(dict(
            feature=feat, spearman=rho, mi=mi.get(feat, np.nan),
            monotonie=mono,
            pire_bin=str(worst["cat"]), pire_lr=worst["loss_rate"],
            pire_n=int(worst["n"]), pire_lift=worst["lift"],
            pire_pval=worst["pval"], pire_robuste=bool(worst["robuste"]),
            meilleur_lr=best["loss_rate"],
            spread=worst["loss_rate"] - best["loss_rate"],
        ))

    bins_df = pd.concat(bin_rows, ignore_index=True) if bin_rows else pd.DataFrame()
    feat_df = pd.DataFrame(feat_rows)

    # FDR sur l'ensemble des bins testes
    if not bins_df.empty:
        bins_df["fdr_sig"] = lib.bh_fdr(bins_df["pval"].values, q=0.05)

    if not feat_df.empty:
        feat_df = feat_df.sort_values("pire_lr", ascending=False).reset_index(drop=True)
        print("\n>>> Classement des features par pire decile (taux de perte) :")
        show = feat_df.copy()
        for c in ["spearman", "mi", "monotonie", "pire_lr", "pire_lift",
                  "meilleur_lr", "spread"]:
            show[c] = show[c].map(lambda x: f"{x:.3f}" if pd.notna(x) else "n/a")
        show["pire_pval"] = show["pire_pval"].map(lambda x: f"{x:.1e}")
        print(show[["feature", "spearman", "mi", "monotonie", "pire_bin",
                    "pire_lr", "pire_n", "pire_lift", "pire_pval",
                    "pire_robuste", "spread"]].head(25).to_string(index=False))

    return bins_df, feat_df


def main() -> None:
    df = lib.load()
    feats = lib.feature_cols(df)
    cont = [f for f in feats if f not in CAT_FEATURES]

    segments = {
        "GLOBAL": df,
        "UP": df[df["dir"] == "UP"].reset_index(drop=True),
        "DOWN": df[df["dir"] == "DOWN"].reset_index(drop=True),
    }

    for name, seg in segments.items():
        cat = scan_categoriel(seg, name)
        if not cat.empty:
            cat.to_csv(lib.OUT_DIR / f"01_categoriel_{name}.csv", index=False)
        bins, franking = scan_continu(seg, name, cont)
        if not bins.empty:
            bins.to_csv(lib.OUT_DIR / f"01_continu_bins_{name}.csv", index=False)
        if not franking.empty:
            franking.to_csv(lib.OUT_DIR / f"01_ranking_{name}.csv", index=False)

        # synthese : patterns robustes + significatifs FDR a fort lift
        if not bins.empty:
            strong = bins[(bins["fdr_sig"]) & (bins["robuste"]) &
                          (bins["lift"] >= 1.15) & (bins["n"] >= 40)]
            strong = strong.sort_values("loss_rate", ascending=False)
            print(f"\n>>> [{name}] Bins ROBUSTES + significatifs FDR + lift>=1.15 "
                  f": {len(strong)} trouves")
            if len(strong):
                s = strong[["feature", "cat", "n", "loss_rate", "wr",
                            "lift", "lr_h1", "lr_h2"]].copy()
                for c in ["loss_rate", "wr", "lift", "lr_h1", "lr_h2"]:
                    s[c] = s[c].map(lambda x: f"{x:.3f}")
                print(s.head(30).to_string(index=False))

    print("\n[OK] Tables ecrites dans resultats/01_*.csv")


if __name__ == "__main__":
    main()
