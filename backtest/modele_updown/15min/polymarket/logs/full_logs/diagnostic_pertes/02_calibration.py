"""
02_calibration.py — Le modele sait-il quand il a raison ?

Question centrale : la confiance (bet_confidence) et l'edge (bet_edge) du modele
classent-ils correctement les paris gagnants vs perdants ?
  - Si OUI -> on peut couper le bas du classement (filtre exploitable).
  - Si NON -> la confiance est du bruit ; tous les paris se valent ; le 40 % de
    perte est irreductible par la confiance du modele lui-meme.

Mesures :
  - AUC de bet_confidence / bet_edge pour separer win vs loss (+ IC bootstrap),
  - courbe de fiabilite (WR reel par decile de confiance),
  - WR par decile d'edge,
  - calibration brute de p_up / p_down sur les 4123 paris,
  - calibration par bucket (volatility_state x is_weekend),
  - courbe "top-K %" : WR si on ne gardait que les K % paris les plus confiants.

Sorties : resultats/02_*.csv  +  resultats/02_calibration.png
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

import lib

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)
RNG = np.random.default_rng(0)


def auc_ci(score: np.ndarray, win: np.ndarray, n_boot: int = 2000) -> tuple:
    """AUC de `score` pour predire `win`, avec IC95 bootstrap."""
    score = np.asarray(score, float)
    win = np.asarray(win, int)
    if len(np.unique(win)) < 2:
        return np.nan, np.nan, np.nan
    auc = roc_auc_score(win, score)
    boots = []
    n = len(win)
    for _ in range(n_boot):
        idx = RNG.integers(0, n, n)
        if len(np.unique(win[idx])) < 2:
            continue
        boots.append(roc_auc_score(win[idx], score[idx]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return auc, lo, hi


def reliability(df: pd.DataFrame, score_col: str, q: int = 10) -> pd.DataFrame:
    """WR reel par decile du score. monotone croissant => score informatif."""
    b = lib.qbins(df[score_col], q)
    rows = []
    for c, idx in df.groupby(b, observed=True).groups.items():
        sub = df.loc[idx]
        rows.append(dict(bin=str(c), n=len(sub),
                         score_moy=sub[score_col].mean(),
                         wr=sub["win"].mean(),
                         loss_rate=sub["loss"].mean()))
    return pd.DataFrame(rows)


def topk_curve(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    """WR cumule des K % paris au score le plus eleve."""
    s = df.sort_values(score_col, ascending=False).reset_index(drop=True)
    rows = []
    for k in [5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 90, 100]:
        m = max(1, int(len(s) * k / 100))
        rows.append(dict(top_pct=k, n=m, wr=s["win"].iloc[:m].mean()))
    return pd.DataFrame(rows)


def section(txt: str) -> None:
    print(f"\n{'=' * 78}\n{txt}\n{'=' * 78}")


def main() -> None:
    df = lib.load()
    base_wr = df["win"].mean()

    section("1. POUVOIR DISCRIMINANT — AUC win/loss (0.50 = aucun)")
    print(f"{'segment':<10}{'score':<18}{'AUC':>8}{'IC95':>22}")
    for seg_name, seg in [("GLOBAL", df), ("UP", df[df.dir == "UP"]),
                          ("DOWN", df[df.dir == "DOWN"])]:
        for sc in ["bet_confidence", "bet_edge", "conf_minus_half"]:
            a, lo, hi = auc_ci(seg[sc].values, seg["win"].values)
            flag = "" if (lo <= 0.5 <= hi) else "  <-- significatif"
            print(f"{seg_name:<10}{sc:<18}{a:>8.4f}   [{lo:.4f}, {hi:.4f}]{flag}")
    print("\nLecture : si l'IC95 contient 0.50, le score ne distingue PAS "
          "gagnants et perdants.")

    section("2. COURBE DE FIABILITE — WR reel par decile de bet_confidence")
    for seg_name, seg in [("UP", df[df.dir == "UP"]), ("DOWN", df[df.dir == "DOWN"])]:
        rel = reliability(seg, "bet_confidence")
        rel.to_csv(lib.OUT_DIR / f"02_reliability_conf_{seg_name}.csv", index=False)
        print(f"\n--- {seg_name} (baseline WR {seg['win'].mean():.4f}) ---")
        show = rel.copy()
        show["ecart_calib"] = show["wr"] - show["score_moy"]
        for c in ["score_moy", "wr", "loss_rate", "ecart_calib"]:
            show[c] = show[c].map(lambda x: f"{x:.4f}")
        print(show.to_string(index=False))
        rho = seg[["bet_confidence"]].assign(w=seg["win"]).corr("spearman").iloc[0, 1]
        print(f"Spearman(confidence, win) = {rho:+.4f}")

    section("3. WR par decile de bet_edge (edge = confidence - seuil)")
    for seg_name, seg in [("GLOBAL", df), ("UP", df[df.dir == "UP"]),
                          ("DOWN", df[df.dir == "DOWN"])]:
        rel = reliability(seg, "bet_edge")
        rel.to_csv(lib.OUT_DIR / f"02_reliability_edge_{seg_name}.csv", index=False)
        print(f"\n--- {seg_name} (baseline WR {seg['win'].mean():.4f}) ---")
        show = rel[["bin", "n", "score_moy", "wr", "loss_rate"]].copy()
        for c in ["score_moy", "wr", "loss_rate"]:
            show[c] = show[c].map(lambda x: f"{x:.4f}")
        print(show.to_string(index=False))

    section("4. CALIBRATION BRUTE de p_up / p_down sur les 4123 paris")
    # p_up doit valoir P(bougie verte) ; p_down doit valoir P(bougie rouge)
    df["_green"] = (df["next_candle"] == "green").astype(int)
    df["_red"] = (df["next_candle"] == "red").astype(int)
    for pcol, ycol, lbl in [("p_up", "_green", "p_up vs P(verte)"),
                            ("p_down", "_red", "p_down vs P(rouge)")]:
        print(f"\n--- {lbl} ---")
        b = lib.qbins(df[pcol], 10)
        rows = []
        for c, idx in df.groupby(b, observed=True).groups.items():
            sub = df.loc[idx]
            rows.append(dict(bin=str(c), n=len(sub),
                             p_moy=sub[pcol].mean(), reel=sub[ycol].mean()))
        t = pd.DataFrame(rows)
        t["ecart"] = t["reel"] - t["p_moy"]
        a = roc_auc_score(df[ycol], df[pcol])
        for c in ["p_moy", "reel", "ecart"]:
            t[c] = t[c].map(lambda x: f"{x:.4f}")
        print(t.to_string(index=False))
        print(f"AUC({pcol} -> {ycol}) = {a:.4f}")

    section("5. CALIBRATION PAR BUCKET volatility_state x is_weekend")
    rows = []
    for (v, w), sub in df.groupby(["volatility_state", "is_weekend"]):
        a, lo, hi = auc_ci(sub["bet_edge"].values, sub["win"].values, n_boot=800)
        rows.append(dict(vol=v, wknd=w, n=len(sub), wr=sub["win"].mean(),
                         conf_moy=sub["bet_confidence"].mean(),
                         edge_moy=sub["bet_edge"].mean(),
                         auc_edge=a, auc_lo=lo, auc_hi=hi))
    t = pd.DataFrame(rows)
    for c in ["wr", "conf_moy", "edge_moy", "auc_edge", "auc_lo", "auc_hi"]:
        t[c] = t[c].map(lambda x: f"{x:.4f}")
    print(t.to_string(index=False))

    section("6. COURBE TOP-K % (ne garder que les K % paris les plus confiants)")
    for seg_name, seg, sc in [("UP/confidence", df[df.dir == "UP"], "bet_confidence"),
                              ("DOWN/confidence", df[df.dir == "DOWN"], "bet_confidence"),
                              ("GLOBAL/edge", df, "bet_edge")]:
        t = topk_curve(seg, sc)
        print(f"\n--- {seg_name} (baseline WR {seg['win'].mean():.4f}) ---")
        show = t.copy()
        show["wr"] = show["wr"].map(lambda x: f"{x:.4f}")
        print(show.to_string(index=False))

    # ---- figure ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for ax, (seg_name, seg) in zip(axes[:2], [("UP", df[df.dir == "UP"]),
                                              ("DOWN", df[df.dir == "DOWN"])]):
        rel = reliability(seg, "bet_confidence")
        ax.plot(rel["score_moy"], rel["wr"], "o-", label="WR reel")
        ax.plot([rel["score_moy"].min(), rel["score_moy"].max()],
                [rel["score_moy"].min(), rel["score_moy"].max()],
                "k--", alpha=.5, label="calibration parfaite")
        ax.axhline(seg["win"].mean(), color="r", ls=":", label="WR moyen")
        ax.set_title(f"Fiabilite confiance — {seg_name}")
        ax.set_xlabel("bet_confidence (decile moyen)")
        ax.set_ylabel("WR reel")
        ax.legend(fontsize=8)
        ax.grid(alpha=.3)
    t = topk_curve(df, "bet_edge")
    axes[2].plot(t["top_pct"], t["wr"], "o-", color="purple")
    axes[2].axhline(base_wr, color="r", ls=":", label="WR moyen")
    axes[2].set_title("WR des top-K% par edge")
    axes[2].set_xlabel("top K % paris (tries par edge)")
    axes[2].set_ylabel("WR cumule")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(lib.OUT_DIR / "02_calibration.png", dpi=110)
    print(f"\n[OK] Figure -> resultats/02_calibration.png")


if __name__ == "__main__":
    main()
