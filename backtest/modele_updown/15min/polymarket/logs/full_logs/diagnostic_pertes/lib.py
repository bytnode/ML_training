"""
lib.py — Bibliotheque partagee du diagnostic des pertes.

Source unique de verite pour :
  - le chargement et le nettoyage de all_bets.csv,
  - les groupes de colonnes (ids / outcomes / decision / contexte / features),
  - les features derivees calculables A L'INSTANT DE LA DECISION (no-lookahead),
  - les outils statistiques (taux de perte + test binomial + robustesse temporelle),
  - l'evaluation de filtres (combien de WR gagne-t-on en sautant un sous-ensemble).

Regle d'or anti-fuite : un "predicteur" est une colonne connue AVANT le resultat.
Les colonnes next_* / result / next_candle sont des OUTCOMES — jamais des predicteurs.
Les prix bruts open/high/low/close/volume sont connus mais leur NIVEAU absolu est
un proxy temporel (BTC 63k -> 125k sur l'annee) ; on ne les utilise pas comme
features, seulement leurs derivees relatives.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

DATA_DIR = Path(__file__).resolve().parent.parent      # .../full_logs/
OUT_DIR = Path(__file__).resolve().parent / "resultats"
OUT_DIR.mkdir(exist_ok=True)

# --------------------------------------------------------------------------
# Groupes de colonnes
# --------------------------------------------------------------------------
IDS = ["timestamp_ms", "datetime_utc", "target_timestamp_ms"]
OUTCOMES = ["result", "next_open", "next_close", "next_candle"]
RAW_PRICE = ["open", "high", "low", "close", "volume"]   # proxy temporel -> exclus
# Sorties des modeles : connues a la decision, predicteurs legitimes
DECISION = ["bet_confidence", "bet_threshold", "bet_edge",
            "p_up", "th_up", "edge_up", "p_down", "th_down", "edge_down"]
CONTEXT = ["volatility_state", "is_weekend", "hour"]


def _const_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if df[c].nunique(dropna=False) <= 1]


def load() -> pd.DataFrame:
    """all_bets.csv nettoye + enrichi de features derivees no-lookahead."""
    df = pd.read_csv(DATA_DIR / "all_bets.csv")
    df = df.sort_values("timestamp_ms").reset_index(drop=True)

    # --- label ---
    df["loss"] = (df["result"].astype(str).str.lower() == "loss").astype(int)
    df["win"] = 1 - df["loss"]
    df["dir"] = df["team_decision"].str.upper()

    # --- temps ---
    df["dt"] = pd.to_datetime(df["datetime_utc"])
    df["dow"] = df["dt"].dt.dayofweek                 # 0=lundi .. 6=dimanche
    df["month"] = df["dt"].dt.to_period("M").astype(str)
    df["week"] = df["dt"].dt.to_period("W").astype(str)
    # index temporel 0..1 pour les splits walk-forward
    t = df["timestamp_ms"].astype(float)
    df["t_frac"] = (t - t.min()) / (t.max() - t.min())

    # --- bougie analysee (connue a la decision) ---
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    df["bar_body_pct"] = (df["close"] - df["open"]) / df["open"]
    df["bar_range_pct"] = (df["high"] - df["low"]) / df["open"]
    df["bar_green"] = (df["close"] > df["open"]).astype(int)
    df["bar_close_loc"] = ((df["close"] - df["low"]) / rng).clip(0, 1)
    df["bar_upper_wick"] = (df["high"] - df[["open", "close"]].max(axis=1)) / df["open"]
    df["bar_lower_wick"] = (df[["open", "close"]].min(axis=1) - df["low"]) / df["open"]

    # --- metriques de decision derivees ---
    df["conf_minus_half"] = df["bet_confidence"] - 0.5
    df["p_sum"] = df["p_up"] + df["p_down"]              # 1.0 = modeles coherents
    df["consensus_green"] = (df["p_up"] + (1.0 - df["p_down"])) / 2.0
    df["disagree"] = df["p_up"] + df["p_down"] - 1.0     # >0 : up plus haussier que down

    # Modele OPPOSE (celui qui n'a pas tire). opp_edge proche de 0 = oppose "chaud".
    up = df["dir"].eq("UP")
    df["opp_p"] = np.where(up, df["p_down"], df["p_up"])
    df["opp_edge"] = np.where(up, df["edge_down"], df["edge_up"])
    df["opp_th"] = np.where(up, df["th_down"], df["th_up"])
    # le consensus pointe-t-il dans le sens du pari ?
    df["consensus_aligned"] = np.where(
        up, df["consensus_green"] - 0.5, 0.5 - df["consensus_green"])

    # --- severite de la perte (OUTCOME — usage descriptif uniquement) ---
    df["outcome_move_pct"] = (df["next_close"] - df["next_open"]) / df["next_open"]
    df["outcome_abs_move"] = df["outcome_move_pct"].abs()

    return df


def feature_cols(df: pd.DataFrame) -> list[str]:
    """Predicteurs numeriques connus a la decision, dedupliques up_/dn_.

    Exclut : ids, outcomes, prix bruts, constantes, colonnes derivees d'outcome.
    Deduplique les paires up_X / dn_X identiques (memes valeurs)."""
    drop = set(IDS + OUTCOMES + RAW_PRICE + _const_cols(df))
    drop |= {"result", "loss", "win", "team_decision", "dir", "dt", "dow",
             "month", "week", "t_frac", "outcome_move_pct", "outcome_abs_move",
             "take_up", "take_down"}
    cands = [c for c in df.columns
             if c not in drop and pd.api.types.is_numeric_dtype(df[c])]
    # dedup up_X / dn_X identiques
    keep, seen = [], []
    for c in cands:
        dup = False
        for k in keep:
            if df[c].equals(df[k]):
                dup = True
                break
        if not dup:
            keep.append(c)
    return keep


# --------------------------------------------------------------------------
# Outils statistiques
# --------------------------------------------------------------------------
def baseline_loss(df: pd.DataFrame) -> float:
    return df["loss"].mean()


def binom_p(n_loss: int, n: int, p0: float) -> float:
    """p-value bilaterale : ce sous-groupe perd-il plus/moins que p0 ?"""
    if n == 0:
        return 1.0
    return stats.binomtest(int(n_loss), int(n), p0, alternative="two-sided").pvalue


def bh_fdr(pvals: np.ndarray, q: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg : renvoie un masque booleen des decouvertes significatives."""
    p = np.asarray(pvals, float)
    n = len(p)
    if n == 0:
        return np.array([], bool)
    order = np.argsort(p)
    thresh = q * (np.arange(1, n + 1) / n)
    passed = p[order] <= thresh
    sig = np.zeros(n, bool)
    if passed.any():
        kmax = np.where(passed)[0].max()
        sig[order[: kmax + 1]] = True
    return sig


def loss_table(df: pd.DataFrame, cat: pd.Series, baseline: float | None = None,
               min_n: int = 25, sort: bool = True) -> pd.DataFrame:
    """Taux de perte par categorie + lift + test binomial + robustesse H1/H2.

    cat : Series categorielle alignee sur df.
    Robustesse : la moitie 1 (1er semestre) et la moitie 2 doivent toutes deux
    depasser la baseline pour qu'un pattern de perte soit juge robuste."""
    if baseline is None:
        baseline = baseline_loss(df)
    half = df["t_frac"] >= 0.5
    rows = []
    for c, idx in df.groupby(cat, observed=True).groups.items():
        sub = df.loc[idx]
        n = len(sub)
        if n < min_n:
            continue
        nl = int(sub["loss"].sum())
        lr = nl / n
        h1 = sub.loc[~half.loc[idx]]
        h2 = sub.loc[half.loc[idx]]
        lr1 = h1["loss"].mean() if len(h1) else np.nan
        lr2 = h2["loss"].mean() if len(h2) else np.nan
        rows.append(dict(
            cat=c, n=n, n_loss=nl, loss_rate=lr, wr=1 - lr,
            lift=lr / baseline, ecart_pts=100 * (lr - baseline),
            pval=binom_p(nl, n, baseline),
            lr_h1=lr1, lr_h2=lr2,
            robuste=(lr1 > baseline) and (lr2 > baseline),
        ))
    out = pd.DataFrame(rows)
    if sort and len(out):
        out = out.sort_values("loss_rate", ascending=False).reset_index(drop=True)
    return out


def qbins(s: pd.Series, q: int = 10) -> pd.Series:
    """Bins par quantiles, etiquetes par l'intervalle, robustes aux ex-aequo."""
    try:
        return pd.qcut(s, q, duplicates="drop")
    except Exception:
        return pd.cut(s, q)


def eval_filter(df: pd.DataFrame, skip_mask: pd.Series,
                baseline: float | None = None) -> dict:
    """Impact d'un filtre qui SAUTE les paris ou skip_mask est vrai.

    Renvoie WR avant/apres, volume garde, WR des paris sautes, et la
    robustesse temporelle (le filtre tient-il sur les 2 moities ?)."""
    skip_mask = skip_mask.astype(bool)
    kept = df.loc[~skip_mask]
    skipped = df.loc[skip_mask]
    n = len(df)
    half = df["t_frac"] >= 0.5
    res = dict(
        n_total=n, n_skip=int(skip_mask.sum()), n_kept=len(kept),
        wr_avant=df["win"].mean(),
        wr_apres=kept["win"].mean() if len(kept) else np.nan,
        wr_sautes=skipped["win"].mean() if len(skipped) else np.nan,
        vol_garde=len(kept) / n if n else np.nan,
    )
    for lbl, m in [("h1", ~half), ("h2", half)]:
        seg = df.loc[m]
        segk = seg.loc[~skip_mask.loc[seg.index]]
        res[f"wr_apres_{lbl}"] = segk["win"].mean() if len(segk) else np.nan
        res[f"n_skip_{lbl}"] = int(skip_mask.loc[seg.index].sum())
        res[f"wr_avant_{lbl}"] = seg["win"].mean()
    res["gain_pts"] = 100 * (res["wr_apres"] - res["wr_avant"])
    return res


def fmt_pct(x: float) -> str:
    return "n/a" if pd.isna(x) else f"{100 * x:.2f}%"


if __name__ == "__main__":
    d = load()
    print(f"{len(d)} paris | baseline perte {baseline_loss(d):.4f}")
    fc = feature_cols(d)
    print(f"{len(fc)} predicteurs numeriques retenus :")
    for c in fc:
        print(f"  - {c}")
