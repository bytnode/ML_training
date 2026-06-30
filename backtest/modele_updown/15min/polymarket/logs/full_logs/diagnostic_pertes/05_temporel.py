"""
05_temporel.py — Les pertes sont-elles structurees dans le TEMPS ?

L'analyse par features (scripts 01-04) ne peut pas voir un pattern temporel :
des pertes qui s'enchainent (regime persistant) ou une derive du modele.
Ce script teste :

  1. WR par mois / semaine  -> derive, degradation recente ?
  2. WR glissant (fenetre 250 paris) + courbe d'equity / drawdown.
  3. Test des suites (runs test) + autocorrelation de la serie de pertes
     -> les pertes s'enchainent-elles plus que le hasard ne le voudrait ?
  4. Taux de perte CONDITIONNEL au passe recent
     -> P(perte | le pari precedent etait perdu), P(perte | 3 derniers perdus)...
     Si oui : un simple "cooldown" apres une serie de pertes serait exploitable.
  5. Plus longue serie perdante observee vs attendue sous independance.

Sorties : resultats/05_*.csv  +  resultats/05_temporel.png
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import lib

pd.set_option("display.width", 200)
pd.set_option("display.max_rows", 80)


def section(txt: str) -> None:
    print(f"\n{'=' * 78}\n{txt}\n{'=' * 78}")


def runs_test(seq: np.ndarray) -> dict:
    """Test des suites de Wald-Wolfowitz sur une sequence binaire.
    z<<0 : moins de suites qu'attendu => les valeurs se regroupent (clustering)."""
    seq = np.asarray(seq, int)
    n = len(seq)
    n1 = int(seq.sum())
    n0 = n - n1
    runs = 1 + int((seq[1:] != seq[:-1]).sum())
    mu = 2 * n1 * n0 / n + 1
    var = 2 * n1 * n0 * (2 * n1 * n0 - n) / (n ** 2 * (n - 1))
    z = (runs - mu) / np.sqrt(var) if var > 0 else np.nan
    return dict(runs=runs, attendu=mu, z=z)


def longest_streak(seq: np.ndarray, val: int = 1) -> int:
    best = cur = 0
    for x in seq:
        cur = cur + 1 if x == val else 0
        best = max(best, cur)
    return best


def main() -> None:
    df = lib.load().sort_values("timestamp_ms").reset_index(drop=True)
    base_wr = df["win"].mean()
    base_loss = df["loss"].mean()
    loss = df["loss"].values

    # ----- 1. WR par mois -----
    section("1. WR PAR MOIS")
    m = df.groupby("month").agg(n=("loss", "size"),
                                wr=("win", "mean"),
                                loss_rate=("loss", "mean"))
    m["ecart_pts"] = 100 * (m["loss_rate"] - base_loss)
    m["pval"] = [lib.binom_p(int(r.loss_rate * r.n), int(r.n), base_loss)
                 for r in m.itertuples()]
    show = m.copy()
    for c in ["wr", "loss_rate"]:
        show[c] = show[c].map(lambda x: f"{x:.3f}")
    show["ecart_pts"] = show["ecart_pts"].map(lambda x: f"{x:+.1f}")
    show["pval"] = show["pval"].map(lambda x: f"{x:.2e}")
    print(show.to_string())
    m.to_csv(lib.OUT_DIR / "05_wr_mensuel.csv")
    # derive lineaire
    rho = np.corrcoef(df["t_frac"], df["loss"])[0, 1]
    print(f"\nCorrelation (temps, perte) = {rho:+.4f}  "
          f"(~0 => pas de derive monotone)")

    # ----- 2. WR glissant + drawdown -----
    section("2. WR GLISSANT (250 paris) ET DRAWDOWN")
    roll = df["win"].rolling(250).mean()
    print(f"WR glissant : min {roll.min():.3f}  max {roll.max():.3f}  "
          f"(amplitude {roll.max() - roll.min():.3f})")
    pnl = (2 * df["win"] - 1).cumsum()             # +1 gain, -1 perte
    peak = pnl.cummax()
    dd = pnl - peak
    print(f"PnL final (gains - pertes) : {int(pnl.iloc[-1])}")
    print(f"Drawdown max : {int(dd.min())} paris nets")

    # ----- 3. Runs test + autocorrelation -----
    section("3. TEST DES SUITES & AUTOCORRELATION DE LA SERIE DE PERTES")
    rt = runs_test(loss)
    print(f"Suites observees : {rt['runs']}  |  attendu sous independance : "
          f"{rt['attendu']:.1f}  |  z = {rt['z']:+.3f}")
    if rt["z"] < -1.96:
        print("  => z < -1.96 : les pertes se REGROUPENT (clustering significatif).")
    elif rt["z"] > 1.96:
        print("  => z > +1.96 : les resultats ALTERNENT plus que le hasard.")
    else:
        print("  => |z| < 1.96 : sequence compatible avec l'independance.")
    print("\nAutocorrelation de la serie de pertes :")
    s = pd.Series(loss)
    for lag in range(1, 11):
        ac = s.autocorr(lag)
        print(f"  lag {lag:>2} : {ac:+.4f}")

    # ----- 4. Taux de perte conditionnel au passe -----
    section("4. TAUX DE PERTE CONDITIONNEL AU PASSE RECENT")
    df["prev1"] = df["loss"].shift(1)
    df["prev2_sum"] = df["loss"].shift(1).rolling(2).sum()
    df["prev3_sum"] = df["loss"].shift(1).rolling(3).sum()
    df["prev5_sum"] = df["loss"].shift(1).rolling(5).sum()

    print(f"Taux de perte de base : {base_loss:.4f}\n")
    print("Conditionnement sur le pari PRECEDENT :")
    for v, lbl in [(0.0, "precedent GAGNE"), (1.0, "precedent PERDU")]:
        sub = df[df["prev1"] == v]
        lr = sub["loss"].mean()
        p = lib.binom_p(int(sub["loss"].sum()), len(sub), base_loss)
        print(f"  {lbl:<18} n={len(sub):>5}  perte={lr:.4f}  "
              f"lift={lr / base_loss:.3f}  p={p:.2e}")

    print("\nConditionnement sur les 3 paris precedents (nb de pertes) :")
    for k in [0, 1, 2, 3]:
        sub = df[df["prev3_sum"] == k]
        if len(sub) < 25:
            continue
        lr = sub["loss"].mean()
        p = lib.binom_p(int(sub["loss"].sum()), len(sub), base_loss)
        print(f"  {k}/3 perdus  n={len(sub):>5}  perte={lr:.4f}  "
              f"lift={lr / base_loss:.3f}  p={p:.2e}")

    print("\nConditionnement sur les 5 paris precedents (nb de pertes) :")
    cond5 = []
    for k in range(0, 6):
        sub = df[df["prev5_sum"] == k]
        if len(sub) < 25:
            continue
        lr = sub["loss"].mean()
        p = lib.binom_p(int(sub["loss"].sum()), len(sub), base_loss)
        cond5.append(dict(k_perdus=k, n=len(sub), loss_rate=lr,
                          lift=lr / base_loss, pval=p))
        print(f"  {k}/5 perdus  n={len(sub):>5}  perte={lr:.4f}  "
              f"lift={lr / base_loss:.3f}  p={p:.2e}")
    pd.DataFrame(cond5).to_csv(lib.OUT_DIR / "05_perte_conditionnelle.csv",
                               index=False)

    # ----- 5. Series perdantes -----
    section("5. SERIES PERDANTES — observe vs attendu sous independance")
    obs = longest_streak(loss, 1)
    rng = np.random.default_rng(0)
    sims = [longest_streak(rng.permutation(loss), 1) for _ in range(3000)]
    sims = np.array(sims)
    print(f"Plus longue serie perdante observee : {obs}")
    print(f"Sous independance (memes pertes melangees) : "
          f"moyenne {sims.mean():.1f}, p95 {np.percentile(sims, 95):.0f}, "
          f"max {sims.max()}")
    print(f"P(serie >= {obs} | independance) = {(sims >= obs).mean():.3f}")

    # ----- figure -----
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    mb = m.reset_index()
    ax[0].bar(range(len(mb)), mb["wr"], color="steelblue")
    ax[0].axhline(base_wr, color="r", ls=":", label=f"WR moyen {base_wr:.3f}")
    ax[0].axhline(0.5, color="k", ls="--", alpha=.5, label="pile ou face")
    ax[0].set_xticks(range(len(mb)))
    ax[0].set_xticklabels(mb["month"], rotation=70, fontsize=7)
    ax[0].set_title("WR par mois")
    ax[0].set_ylim(0.45, 0.72)
    ax[0].legend(fontsize=8)
    ax[1].plot(roll.values, color="darkgreen")
    ax[1].axhline(base_wr, color="r", ls=":")
    ax[1].set_title("WR glissant (250 paris)")
    ax[1].set_xlabel("indice du pari")
    ax[1].grid(alpha=.3)
    ax[2].fill_between(range(len(dd)), dd.values, 0, color="firebrick", alpha=.6)
    ax[2].set_title("Drawdown (paris nets sous le pic)")
    ax[2].set_xlabel("indice du pari")
    ax[2].grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(lib.OUT_DIR / "05_temporel.png", dpi=110)
    print(f"\n[OK] Figure -> resultats/05_temporel.png")


if __name__ == "__main__":
    main()
