"""
03_interactions_regles.py — Patterns multivaries et regles de perte.

Le balayage univarie (script 01) n'a trouve aucun predicteur simple. Les pertes,
si elles sont structurees, le sont par des COMBINAISONS. Ce script teste :

  A. Hypotheses structurelles explicites (mean-reversion, modele oppose "chaud",
     desaccord des deux modeles) — avec des seuils metier, pas des deciles.
  B. Tableaux croises 2D (taux de perte par paire de variables).
  C. Minage de regles par arbre de decision, AVEC validation hors-echantillon :
     l'arbre est appris sur la 1re moitie temporelle (H1), les regles a forte
     perte sont extraites, puis APPLIQUEES telles quelles sur la 2e moitie (H2).
     Une regle n'est reelle que si la sur-perte tient sur H2 (et reciproquement).

Sorties : resultats/03_*.csv  +  synthese console.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, _tree

import lib

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 50)
pd.set_option("display.max_rows", 200)


def section(txt: str) -> None:
    print(f"\n{'=' * 78}\n{txt}\n{'=' * 78}")


def bucket_table(df: pd.DataFrame, name: str, mask_def: dict,
                 baseline: float) -> pd.DataFrame:
    """Taux de perte pour une liste de masques metier nommes."""
    rows = []
    half = df["t_frac"] >= 0.5
    for lbl, mask in mask_def.items():
        sub = df.loc[mask]
        n = len(sub)
        if n < 20:
            rows.append(dict(bucket=lbl, n=n, loss_rate=np.nan, wr=np.nan,
                             lift=np.nan, pval=np.nan, lr_h1=np.nan,
                             lr_h2=np.nan, robuste=False))
            continue
        nl = int(sub["loss"].sum())
        lr1 = df.loc[mask & ~half, "loss"].mean()
        lr2 = df.loc[mask & half, "loss"].mean()
        rows.append(dict(
            bucket=lbl, n=n, loss_rate=nl / n, wr=1 - nl / n,
            lift=(nl / n) / baseline, pval=lib.binom_p(nl, n, baseline),
            lr_h1=lr1, lr_h2=lr2,
            robuste=(lr1 > baseline) and (lr2 > baseline)))
    t = pd.DataFrame(rows)
    print(f"\n--- {name} (baseline perte {baseline:.4f}) ---")
    show = t.copy()
    for c in ["loss_rate", "wr", "lift", "lr_h1", "lr_h2"]:
        show[c] = show[c].map(lambda x: f"{x:.3f}" if pd.notna(x) else "n/a")
    show["pval"] = show["pval"].map(lambda x: f"{x:.1e}" if pd.notna(x) else "n/a")
    print(show.to_string(index=False))
    return t


# --------------------------------------------------------------------------
# A. Hypotheses structurelles
# --------------------------------------------------------------------------
def hypotheses_structurelles(df: pd.DataFrame) -> None:
    section("A. HYPOTHESES STRUCTURELLES")

    up = df[df.dir == "UP"]
    dn = df[df.dir == "DOWN"]
    b_up, b_dn = up["loss"].mean(), dn["loss"].mean()

    # H3 mean-reversion : parier UP en haut de bande / apres forte hausse
    bucket_table(up, "H3a UP — position dans les bandes de Bollinger", {
        "bb > 1.0 (au-dessus bande haute)": up.up_bb_position > 1.0,
        "bb 0.5..1.0 (haut)": up.up_bb_position.between(0.5, 1.0),
        "bb 0.0..0.5 (milieu)": up.up_bb_position.between(0.0, 0.5),
        "bb < 0.0 (sous bande basse)": up.up_bb_position < 0.0,
    }, b_up)
    bucket_table(up, "H3b UP — momentum recent (up_return_3)", {
        "return_3 > +1%": up.up_return_3 > 0.01,
        "return_3 0..+1%": up.up_return_3.between(0, 0.01),
        "return_3 -1%..0": up.up_return_3.between(-0.01, 0),
        "return_3 < -1%": up.up_return_3 < -0.01,
    }, b_up)
    bucket_table(dn, "H3c DOWN — position dans les bandes de Bollinger", {
        "bb > 1.0 (au-dessus bande haute)": dn.dn_bb_position > 1.0,
        "bb 0.5..1.0 (haut)": dn.dn_bb_position.between(0.5, 1.0),
        "bb 0.0..0.5 (milieu)": dn.dn_bb_position.between(0.0, 0.5),
        "bb < 0.0 (sous bande basse)": dn.dn_bb_position < 0.0,
    }, b_dn)
    bucket_table(dn, "H3d DOWN — momentum recent (dn_return_3)", {
        "return_3 > +1%": dn.dn_return_3 > 0.01,
        "return_3 0..+1%": dn.dn_return_3.between(0, 0.01),
        "return_3 -1%..0": dn.dn_return_3.between(-0.01, 0),
        "return_3 < -1% (forte baisse)": dn.dn_return_3 < -0.01,
    }, b_dn)

    # H4 modele oppose "chaud" : opp_edge proche de 0 = l'autre modele a failli tirer
    base = df["loss"].mean()
    bucket_table(df, "H4 — chaleur du modele oppose (opp_edge)", {
        "oppose tres froid  (edge < -0.08)": df.opp_edge < -0.08,
        "oppose froid       (-0.08..-0.04)": df.opp_edge.between(-0.08, -0.04),
        "oppose tiede       (-0.04..-0.01)": df.opp_edge.between(-0.04, -0.01),
        "oppose CHAUD       (edge > -0.01)": df.opp_edge > -0.01,
    }, base)

    # H_disagree : coherence des deux modeles (p_sum proche de 1 = coherents)
    bucket_table(df, "H5 — coherence des modeles (p_sum = p_up + p_down)", {
        "p_sum < 1.00 (incoherents)": df.p_sum < 1.00,
        "p_sum 1.00..1.02": df.p_sum.between(1.00, 1.02),
        "p_sum 1.02..1.04": df.p_sum.between(1.02, 1.04),
        "p_sum > 1.04 (tres alignes)": df.p_sum > 1.04,
    }, base)

    # H_conf : edge marginal -> pari ~ pile ou face ?
    bucket_table(df, "H6 — edge du pari (marge au-dessus du seuil)", {
        "edge < 0.003 (a la limite)": df.bet_edge < 0.003,
        "edge 0.003..0.010": df.bet_edge.between(0.003, 0.010),
        "edge 0.010..0.025": df.bet_edge.between(0.010, 0.025),
        "edge > 0.025 (confortable)": df.bet_edge > 0.025,
    }, base)


# --------------------------------------------------------------------------
# B. Croisements 2D
# --------------------------------------------------------------------------
def croisements_2d(df: pd.DataFrame) -> None:
    section("B. TABLEAUX CROISES 2D — taux de perte (n entre parentheses)")
    df = df.copy()
    df["hour_blk"] = pd.cut(df["hour"], [-1, 5, 11, 17, 23],
                            labels=["00-05", "06-11", "12-17", "18-23"])
    df["edge_t"] = pd.qcut(df["bet_edge"], 3, labels=["edge-bas", "edge-moy", "edge-haut"])
    df["opp_t"] = pd.qcut(df["opp_edge"], 3, labels=["opp-froid", "opp-moy", "opp-chaud"])

    pairs = [("dir", "is_weekend"), ("dir", "volatility_state"),
             ("hour_blk", "volatility_state"), ("edge_t", "opp_t"),
             ("dir", "hour_blk"), ("edge_t", "volatility_state")]
    for a, b in pairs:
        lr = df.pivot_table("loss", a, b, aggfunc="mean")
        cnt = df.pivot_table("loss", a, b, aggfunc="count")
        print(f"\n--- {a} x {b} ---")
        combo = lr.round(3).astype(str) + " (" + cnt.astype("Int64").astype(str) + ")"
        print(combo.to_string())


# --------------------------------------------------------------------------
# C. Minage de regles par arbre + validation hors-echantillon
# --------------------------------------------------------------------------
def _leaf_rules(tree: DecisionTreeClassifier, feats: list[str]) -> dict:
    """Extrait, pour chaque feuille, la conjonction de conditions du chemin."""
    t = tree.tree_
    rules = {}

    def recurse(node, conds):
        if t.feature[node] == _tree.TREE_UNDEFINED:        # feuille
            rules[node] = list(conds)
            return
        f = feats[t.feature[node]]
        thr = t.threshold[node]
        recurse(t.children_left[node], conds + [f"{f} <= {thr:.5g}"])
        recurse(t.children_right[node], conds + [f"{f} > {thr:.5g}"])

    recurse(0, [])
    return rules


def minage_regles(df: pd.DataFrame, label: str, feats: list[str],
                  depth: int = 4, min_leaf: int = 70) -> pd.DataFrame:
    """Apprend un arbre sur H1, extrait les feuilles a forte perte, les valide
    sur H2 (et reciproquement H2 -> H1). Renvoie les regles validees."""
    print(f"\n{'-' * 70}\nMINAGE DE REGLES — {label} "
          f"(profondeur {depth}, min feuille {min_leaf})\n{'-' * 70}")
    base = df["loss"].mean()
    X = df[feats].fillna(df[feats].median())
    y = df["loss"].values
    half = (df["t_frac"] >= 0.5).values

    found = []
    for train_lbl, tr_mask, te_mask in [("H1->H2", ~half, half),
                                        ("H2->H1", half, ~half)]:
        clf = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=min_leaf,
                                     random_state=0)
        clf.fit(X[tr_mask], y[tr_mask])
        rules = _leaf_rules(clf, feats)
        leaf_tr = clf.apply(X[tr_mask].values)
        leaf_te = clf.apply(X[te_mask].values)
        y_tr, y_te = y[tr_mask], y[te_mask]
        for leaf in np.unique(leaf_tr):
            in_tr = leaf_tr == leaf
            in_te = leaf_te == leaf
            n_tr, n_te = in_tr.sum(), in_te.sum()
            if n_tr < min_leaf or n_te < 30:
                continue
            lr_tr = y_tr[in_tr].mean()
            lr_te = y_te[in_te].mean()
            if lr_tr <= base + 0.04:        # feuille interessante = sur-perte sur train
                continue
            found.append(dict(
                segment=label, sens=train_lbl, n_train=int(n_tr),
                lr_train=lr_tr, n_test=int(n_te), lr_test=lr_te,
                valide=(lr_te > base),
                regle=" ET ".join(rules[leaf])))
    res = pd.DataFrame(found)
    if res.empty:
        print("  Aucune feuille a sur-perte > baseline+4pts sur le train.")
        return res
    res = res.sort_values("lr_train", ascending=False).reset_index(drop=True)
    show = res.copy()
    for c in ["lr_train", "lr_test"]:
        show[c] = show[c].map(lambda x: f"{x:.3f}")
    print(show[["sens", "n_train", "lr_train", "n_test", "lr_test",
                "valide", "regle"]].to_string(index=False))
    n_val = res["valide"].sum()
    print(f"\n  => {n_val}/{len(res)} feuilles a sur-perte se confirment "
          f"hors-echantillon (lr_test > {base:.3f}).")
    return res


def main() -> None:
    df = lib.load()
    feats = lib.feature_cols(df)

    hypotheses_structurelles(df)
    croisements_2d(df)

    section("C. MINAGE DE REGLES PAR ARBRE (validation hors-echantillon)")
    allrules = []
    for label, seg in [("GLOBAL", df),
                       ("UP", df[df.dir == "UP"].reset_index(drop=True)),
                       ("DOWN", df[df.dir == "DOWN"].reset_index(drop=True))]:
        r = minage_regles(seg, label, feats)
        if not r.empty:
            allrules.append(r)
    if allrules:
        out = pd.concat(allrules, ignore_index=True)
        out.to_csv(lib.OUT_DIR / "03_regles_arbre.csv", index=False)
        print(f"\n[OK] Regles -> resultats/03_regles_arbre.csv")
        nv = out["valide"].sum()
        print(f"\nBILAN GLOBAL : {nv}/{len(out)} regles a forte perte tiennent "
              f"hors-echantillon.")


if __name__ == "__main__":
    main()
