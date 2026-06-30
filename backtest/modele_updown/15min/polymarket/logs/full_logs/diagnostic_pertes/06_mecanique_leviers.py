"""
06_mecanique_leviers.py — Mecanique des pertes et leviers de configuration.

Apres avoir etabli (01-05) qu'il n'existe pas de pattern de perte filtrable,
ce script repond a deux questions concretes :

  A. MECANIQUE — quand un pari est perdu, l'echoue-t-il de peu (bougie minuscule,
     pile ou face) ou nettement ? Quelle part des pertes est du bruit pur ?

  B. LEVIERS — trois reglages actionnables sont chiffres avec leur compromis
     WR / volume, et valides sur les deux moities temporelles :
       1. Plancher d'edge : sauter les paris pris trop pres du seuil.
       2. hour_boost : le rabais de seuil de -0.02 sur 20h-22h admet-il du dechet ?
       3. Segment semaine : les jours ouvres sont-ils le maillon faible ?

Sorties : resultats/06_*.csv  +  resultats/06_leviers.png
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

import lib

pd.set_option("display.width", 200)
BOOST_HOURS = {20, 21, 22}
BOOST_DELTA = 0.02


def section(txt: str) -> None:
    print(f"\n{'=' * 78}\n{txt}\n{'=' * 78}")


def mecanique(df: pd.DataFrame) -> None:
    section("A. MECANIQUE DES PERTES — amplitude de la bougie resultat")
    win = df[df.win == 1]["outcome_abs_move"]
    los = df[df.loss == 1]["outcome_abs_move"]
    print(f"Amplitude |next_close-next_open| / next_open :")
    print(f"  paris GAGNES : mediane {win.median() * 100:.3f}%  "
          f"moyenne {win.mean() * 100:.3f}%")
    print(f"  paris PERDUS : mediane {los.median() * 100:.3f}%  "
          f"moyenne {los.mean() * 100:.3f}%")
    u, p = stats.mannwhitneyu(win, los)
    print(f"  Mann-Whitney gagnes vs perdus : p = {p:.2e}")
    print(f"  => les bougies des paris perdus sont "
          f"{'PLUS PETITES' if los.median() < win.median() else 'plus grandes'} "
          f"(ratio mediane {los.median() / win.median():.2f}).")

    print("\nPart des PERTES qui sont des quasi-egalites (bougie minuscule) :")
    for thr in [0.0005, 0.001, 0.0015, 0.002, 0.003]:
        share = (los < thr).mean()
        print(f"  perte avec bougie adverse < {thr * 100:.2f}% : "
              f"{share * 100:.1f}% des pertes")

    # une bougie minuscule = direction quasi aleatoire : peut-on la prevoir ?
    rho = df[["atr_pct"]].assign(m=df["outcome_abs_move"]).corr("spearman").iloc[0, 1]
    print(f"\nCorrelation Spearman(atr_pct pre-pari, amplitude bougie resultat) "
          f"= {rho:+.3f}")
    # WR sur le tiers de paris a plus petite bougie-resultat (a posteriori)
    df = df.copy()
    df["amp_t"] = pd.qcut(df["outcome_abs_move"], 3,
                          labels=["petite", "moyenne", "grande"])
    wr_amp = df.groupby("amp_t", observed=True)["win"].agg(["mean", "size"])
    print("\nWR par taille de la bougie-resultat (constat a posteriori) :")
    print(wr_amp.to_string())
    print("Lecture : si le WR s'effondre sur les petites bougies, une partie "
          "des pertes est du pile-ou-face mecanique.")


def levier_edge(df: pd.DataFrame) -> pd.DataFrame:
    section("B1. LEVIER — plancher d'edge (sauter les paris pres du seuil)")
    base_wr = df["win"].mean()
    print(f"WR actuel {base_wr:.4f} sur {len(df)} paris\n")
    print(f"{'plancher':>10}{'n sautes':>10}{'n gardes':>10}{'WR garde':>11}"
          f"{'gain pts':>10}{'WR H1':>9}{'WR H2':>9}{'robuste':>9}")
    rows = []
    for floor in [0.002, 0.003, 0.004, 0.005, 0.007, 0.010, 0.013, 0.016]:
        skip = df["bet_edge"] < floor
        r = lib.eval_filter(df, skip)
        robuste = (r["wr_apres_h1"] > r["wr_avant_h1"]) and \
                  (r["wr_apres_h2"] > r["wr_avant_h2"])
        rows.append(dict(plancher=floor, **r, robuste=robuste))
        print(f"{floor:>10.3f}{r['n_skip']:>10}{r['n_kept']:>10}"
              f"{r['wr_apres']:>11.4f}{r['gain_pts']:>+10.2f}"
              f"{r['wr_apres_h1']:>9.4f}{r['wr_apres_h2']:>9.4f}"
              f"{str(robuste):>9}")
    t = pd.DataFrame(rows)
    t.to_csv(lib.OUT_DIR / "06_levier_edge.csv", index=False)
    return t


def levier_hourboost(df: pd.DataFrame) -> None:
    section("B2. LEVIER — hour_boost (rabais de seuil -0.02 sur 20h-22h)")
    base = df["loss"].mean()
    in_boost = df["hour"].isin(BOOST_HOURS)
    # un pari "boost-only" n'aurait pas passe le seuil sans le rabais
    boost_only = in_boost & (df["bet_edge"] < BOOST_DELTA)
    print(f"Paris sur 20h-22h          : {in_boost.sum():>5}  "
          f"WR {df.loc[in_boost, 'win'].mean():.4f}")
    print(f"  dont 'boost-only' (n'auraient pas passe sans le rabais) : "
          f"{boost_only.sum():>5}  WR {df.loc[boost_only, 'win'].mean():.4f}")
    print(f"  20h-22h hors boost-only  : {(in_boost & ~boost_only).sum():>5}  "
          f"WR {df.loc[in_boost & ~boost_only, 'win'].mean():.4f}")
    print(f"Paris hors 20h-22h         : {(~in_boost).sum():>5}  "
          f"WR {df.loc[~in_boost, 'win'].mean():.4f}")
    # robustesse boost-only
    half = df["t_frac"] >= 0.5
    for lbl, m in [("H1", ~half), ("H2", half)]:
        seg = boost_only & m
        if seg.sum() > 20:
            print(f"  boost-only {lbl} : n={seg.sum():>4}  "
                  f"WR {df.loc[seg, 'win'].mean():.4f}")
    nl = int(df.loc[boost_only, "loss"].sum())
    print(f"  test binomial boost-only vs baseline : "
          f"p = {lib.binom_p(nl, int(boost_only.sum()), base):.3e}")
    print("Lecture : si les 'boost-only' perdent nettement plus, le rabais "
          "hour_boost admet des paris faibles.")


def levier_semaine(df: pd.DataFrame) -> None:
    section("B3. LEVIER — segment jours ouvres vs week-end")
    for lbl, seg in [("Jours ouvres (is_weekend=0)", df[df.is_weekend == 0]),
                     ("Week-end    (is_weekend=1)", df[df.is_weekend == 1])]:
        print(f"\n{lbl} : n={len(seg)}  WR {seg['win'].mean():.4f}  "
              f"edge moyen {seg['bet_edge'].mean():.4f}")
        # plancher d'edge applique au seul segment semaine
        if "ouvres" in lbl:
            base_wr = seg["win"].mean()
            for floor in [0.003, 0.005, 0.008, 0.012]:
                skip = seg["bet_edge"] < floor
                kept = seg.loc[~skip]
                print(f"   plancher edge {floor:.3f} -> garde {len(kept):>4} "
                      f"({100 * len(kept) / len(seg):.0f}%)  "
                      f"WR {kept['win'].mean():.4f} "
                      f"({100 * (kept['win'].mean() - base_wr):+.2f} pts)")


def main() -> None:
    df = lib.load()
    mecanique(df)
    t = levier_edge(df)
    levier_hourboost(df)
    levier_semaine(df)

    # figure : courbe de compromis WR/volume du plancher d'edge
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()
    ax1.plot(t["plancher"], t["wr_apres"], "o-", color="navy", label="WR garde")
    ax1.axhline(df["win"].mean(), color="r", ls=":", label="WR actuel")
    ax2.plot(t["plancher"], 100 * t["vol_garde"], "s--", color="orange",
             label="volume garde %")
    ax1.set_xlabel("plancher d'edge (paris sous ce seuil = sautes)")
    ax1.set_ylabel("WR des paris gardes", color="navy")
    ax2.set_ylabel("% du volume garde", color="orange")
    ax1.set_title("Levier plancher d'edge : compromis WR / volume")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(lib.OUT_DIR / "06_leviers.png", dpi=110)
    print(f"\n[OK] Figure -> resultats/06_leviers.png")


if __name__ == "__main__":
    main()
