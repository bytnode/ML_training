"""
07_signature_pertes.py — Une donnee ressort-elle cote pertes vs cote gains ?

Question directe : pour CHAQUE feature, sa valeur est-elle systematiquement
differente sur les paris perdus vs gagnes ? (ex : ATR plus eleve sur les pertes)

Pour chaque feature, on compare les deux distributions win vs loss :
  - moyenne / mediane / ecart-type cote gagne et cote perdu,
  - taille d'effet de Cohen (d) : >0 => feature plus elevee sur les pertes,
  - test de Mann-Whitney (la distribution est-elle decalee ?) + correction FDR,
  - AUC d'un seul feature pour separer perte/gain : 0.50 = ne ressort pas,
    >0.55 = ressort un peu, >0.60 = ressort nettement,
  - robustesse : l'effet garde-t-il le meme sens sur les 2 moities du backtest ?

Verifie aussi :
  - les QUEUES extremes (top/bottom 5 % et 2.5 %) — un effet de queue que les
    deciles dilueraient,
  - la VARIANCE (les pertes arrivent-elles sur des features plus dispersees ?).

Focus explicite sur ATR / volume comme demande.

Sorties : resultats/07_*.csv  +  resultats/07_signature.png
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import roc_auc_score

import lib

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 100)


def section(txt: str) -> None:
    print(f"\n{'=' * 80}\n{txt}\n{'=' * 80}")


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    """Taille d'effet : (moyenne_a - moyenne_b) / ecart-type combine."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    sp = np.sqrt(((na - 1) * a.std(ddof=1) ** 2 +
                  (nb - 1) * b.std(ddof=1) ** 2) / (na + nb - 2))
    return (a.mean() - b.mean()) / sp if sp > 0 else np.nan


def signature(df: pd.DataFrame, label: str, feats: list[str]) -> pd.DataFrame:
    """Compare la distribution de chaque feature entre paris perdus et gagnes."""
    section(f"SIGNATURE WIN vs LOSS — {label}  (n={len(df)}, "
            f"perdus={int(df.loss.sum())}, gagnes={int(df.win.sum())})")
    los_m = df["loss"] == 1
    win_m = df["win"] == 1
    half = df["t_frac"] >= 0.5
    rows = []
    for f in feats:
        s = df[f]
        los = s[los_m].dropna()
        win = s[win_m].dropna()
        if len(los) < 20 or len(win) < 20:
            continue
        # AUC : proba qu'un pari perdu ait une valeur > a un pari gagne
        auc = roc_auc_score(df.loc[s.notna(), "loss"], s[s.notna()])
        # robustesse : meme sens d'AUC sur les 2 moities ?
        a1 = a2 = np.nan
        for seg, store in [(~half, "1"), (half, "2")]:
            sub = df.loc[seg & s.notna()]
            if sub["loss"].nunique() == 2 and len(sub) > 40:
                v = roc_auc_score(sub["loss"], sub[f])
                if store == "1":
                    a1 = v
                else:
                    a2 = v
        meme_sens = (np.sign(a1 - 0.5) == np.sign(a2 - 0.5)) \
            if pd.notna(a1) and pd.notna(a2) else False
        u, p = stats.mannwhitneyu(los, win)
        rows.append(dict(
            feature=f,
            moy_perte=los.mean(), moy_gain=win.mean(),
            med_perte=los.median(), med_gain=win.median(),
            cohen_d=cohen_d(los.values, win.values),
            auc=auc, ecart_auc=abs(auc - 0.5),
            mw_p=p, auc_h1=a1, auc_h2=a2, meme_sens=meme_sens))
    t = pd.DataFrame(rows)
    if t.empty:
        return t
    t["fdr_sig"] = lib.bh_fdr(t["mw_p"].values, q=0.05)
    t["sens"] = np.where(t["auc"] > 0.5, "eleve->PERTE", "eleve->gain")
    t = t.sort_values("ecart_auc", ascending=False).reset_index(drop=True)

    show = t.copy()
    for c in ["moy_perte", "moy_gain", "med_perte", "med_gain"]:
        show[c] = show[c].map(lambda x: f"{x:.4g}")
    for c in ["cohen_d", "auc", "ecart_auc"]:
        show[c] = show[c].map(lambda x: f"{x:+.4f}" if c == "cohen_d"
                              else f"{x:.4f}")
    show["mw_p"] = show["mw_p"].map(lambda x: f"{x:.2e}")
    print("Classe par ecart d'AUC (= a quel point la feature ressort) :\n")
    print(show[["feature", "moy_perte", "moy_gain", "cohen_d", "auc",
                "sens", "mw_p", "fdr_sig", "meme_sens"]].head(25).to_string(index=False))

    strong = t[(t["fdr_sig"]) & (t["meme_sens"]) & (t["ecart_auc"] >= 0.03)]
    print(f"\n>>> Features qui ressortent VRAIMENT "
          f"(FDR-significatif + meme sens H1/H2 + AUC ecartee de >=0.03) : "
          f"{len(strong)}")
    if len(strong):
        print(strong[["feature", "auc", "cohen_d", "sens"]].to_string(index=False))
    else:
        print("    AUCUNE. Aucune donnee ne ressort de facon nette et robuste.")
    return t


def focus_atr(df: pd.DataFrame) -> None:
    section("FOCUS DEMANDE — ATR & volume : perte vs gain")
    for col in ["atr_pct", "volume", "volume_ratio", "up_volume_rel_30d",
                "bar_range_pct"]:
        los = df.loc[df.loss == 1, col]
        win = df.loc[df.win == 1, col]
        d = cohen_d(los.values, win.values)
        u, p = stats.mannwhitneyu(los, win)
        verdict = ("plus ELEVE sur les pertes" if los.median() > win.median()
                   else "plus eleve sur les gains")
        print(f"\n[{col}]")
        print(f"  paris PERDUS : mediane {los.median():.5g}  moyenne {los.mean():.5g}")
        print(f"  paris GAGNES : mediane {win.median():.5g}  moyenne {win.mean():.5g}")
        print(f"  Cohen d = {d:+.4f}  |  Mann-Whitney p = {p:.3f}  |  {verdict}")
        signif = "OUI" if p < 0.05 and abs(d) > 0.1 else "NON (effet negligeable)"
        print(f"  => difference exploitable ? {signif}")

    # par sens
    print("\n--- atr_pct detaille par sens de pari ---")
    for d in ["UP", "DOWN"]:
        seg = df[df.dir == d]
        los = seg.loc[seg.loss == 1, "atr_pct"]
        win = seg.loc[seg.win == 1, "atr_pct"]
        print(f"  {d:>5} : ATR median perte {los.median():.5g} vs "
              f"gain {win.median():.5g}  (d={cohen_d(los.values, win.values):+.3f})")


def queues(df: pd.DataFrame, feats: list[str]) -> None:
    section("QUEUES EXTREMES — taux de perte dans les 5 % / 2.5 % extremes")
    base = df["loss"].mean()
    print(f"baseline perte {base:.4f}\n")
    print(f"{'feature':<24}{'bas 2.5%':>10}{'bas 5%':>9}"
          f"{'haut 5%':>10}{'haut 2.5%':>11}")
    rows = []
    for f in feats:
        s = df[f]
        qs = s.quantile([0.025, 0.05, 0.95, 0.975])
        lr_b2 = df.loc[s <= qs[0.025], "loss"].mean()
        lr_b5 = df.loc[s <= qs[0.05], "loss"].mean()
        lr_h5 = df.loc[s >= qs[0.95], "loss"].mean()
        lr_h2 = df.loc[s >= qs[0.975], "loss"].mean()
        rows.append(dict(feature=f, bas2=lr_b2, bas5=lr_b5,
                         haut5=lr_h5, haut2=lr_h2,
                         ecart_max=max(abs(lr_b2 - base), abs(lr_h2 - base))))
    t = pd.DataFrame(rows).sort_values("ecart_max", ascending=False)
    for r in t.head(15).itertuples():
        print(f"{r.feature:<24}{r.bas2:>10.3f}{r.bas5:>9.3f}"
              f"{r.haut5:>10.3f}{r.haut2:>11.3f}")
    print(f"\n(baseline {base:.3f} — une queue 'coupable' afficherait >> 0.50 "
          f"sur n~100. Ici n par queue 2.5% ~ 103.)")
    t.to_csv(lib.OUT_DIR / "07_queues.csv", index=False)


def variance_check(df: pd.DataFrame, feats: list[str]) -> None:
    section("DISPERSION — les pertes ont-elles des features plus dispersees ?")
    rows = []
    for f in feats:
        los = df.loc[df.loss == 1, f].dropna()
        win = df.loc[df.win == 1, f].dropna()
        if len(los) < 20 or len(win) < 20:
            continue
        stat, p = stats.levene(los, win)
        rows.append(dict(feature=f, std_perte=los.std(), std_gain=win.std(),
                         ratio_std=los.std() / win.std() if win.std() else np.nan,
                         levene_p=p))
    t = pd.DataFrame(rows)
    t["fdr_sig"] = lib.bh_fdr(t["levene_p"].values, q=0.05)
    t = t.reindex(t["ratio_std"].sub(1).abs().sort_values(ascending=False).index)
    show = t.head(10).copy()
    for c in ["std_perte", "std_gain", "ratio_std"]:
        show[c] = show[c].map(lambda x: f"{x:.4g}")
    show["levene_p"] = show["levene_p"].map(lambda x: f"{x:.3f}")
    print(show.to_string(index=False))
    n_sig = int(t["fdr_sig"].sum())
    print(f"\n>>> Features a dispersion significativement differente (FDR) : "
          f"{n_sig} / {len(t)}")


def main() -> None:
    df = lib.load()
    feats = lib.feature_cols(df)

    tg = signature(df, "GLOBAL", feats)
    tu = signature(df[df.dir == "UP"].reset_index(drop=True), "UP", feats)
    td = signature(df[df.dir == "DOWN"].reset_index(drop=True), "DOWN", feats)
    for name, t in [("GLOBAL", tg), ("UP", tu), ("DOWN", td)]:
        if not t.empty:
            t.to_csv(lib.OUT_DIR / f"07_signature_{name}.csv", index=False)

    focus_atr(df)
    queues(df, feats)
    variance_check(df, feats)

    # ---- figure ----
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    top = tg.reindex(tg["cohen_d"].abs().sort_values(ascending=False).index).head(22)
    top = top.iloc[::-1]
    colors = ["firebrick" if d > 0 else "steelblue" for d in top["cohen_d"]]
    ax[0].barh(top["feature"], top["cohen_d"], color=colors)
    ax[0].axvspan(-0.2, 0.2, color="grey", alpha=.15, label="effet negligeable |d|<0.2")
    ax[0].axvline(0, color="k", lw=.8)
    ax[0].set_title("Ecart standardise perte vs gain (Cohen d)\n"
                    "rouge = plus eleve sur les pertes")
    ax[0].set_xlabel("Cohen d")
    ax[0].set_xlim(-0.35, 0.35)
    ax[0].legend(fontsize=8, loc="lower right")

    los = df.loc[df.loss == 1, "atr_pct"] * 100
    win = df.loc[df.win == 1, "atr_pct"] * 100
    bins = np.linspace(0, 1.0, 45)
    ax[1].hist(win, bins=bins, density=True, alpha=.55, color="green",
               label=f"gagnes (med {win.median():.3f}%)")
    ax[1].hist(los, bins=bins, density=True, alpha=.55, color="red",
               label=f"perdus (med {los.median():.3f}%)")
    ax[1].set_title("Distribution de l'ATR % — gagnes vs perdus")
    ax[1].set_xlabel("atr_pct (%)")
    ax[1].set_ylabel("densite")
    ax[1].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(lib.OUT_DIR / "07_signature.png", dpi=110)
    print(f"\n[OK] Figure -> resultats/07_signature.png")


if __name__ == "__main__":
    main()
