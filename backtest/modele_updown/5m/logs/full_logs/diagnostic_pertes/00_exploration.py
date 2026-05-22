"""
00_exploration.py — Probe d'exploration des logs de backtest.

Objectif : comprendre la structure des fichiers all_bets / lost_bets / won_bets
avant toute analyse. Verifie l'integrite des donnees, identifie les colonnes
constantes (hyperparams/config/sidecar), valide la coherence resultat/direction.

Lecture seule. Aucune modification des CSV sources.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 60)

DATA_DIR = Path(__file__).resolve().parent.parent   # .../full_logs/


def banner(txt: str) -> None:
    print("\n" + "=" * 78)
    print(txt)
    print("=" * 78)


def main() -> None:
    banner("CHARGEMENT")
    allb = pd.read_csv(DATA_DIR / "all_bets.csv")
    lost = pd.read_csv(DATA_DIR / "lost_bets.csv")
    won = pd.read_csv(DATA_DIR / "won_bets.csv")
    print(f"all_bets  : {allb.shape[0]:>6} lignes x {allb.shape[1]} colonnes")
    print(f"lost_bets : {lost.shape[0]:>6} lignes")
    print(f"won_bets  : {won.shape[0]:>6} lignes")
    print(f"won + lost = {won.shape[0] + lost.shape[0]}  (attendu = {allb.shape[0]})")

    banner("PARTITION lost + won == all ?")
    # Cle d'identite = timestamp_ms
    ts_all = set(allb["timestamp_ms"])
    ts_lost = set(lost["timestamp_ms"])
    ts_won = set(won["timestamp_ms"])
    print(f"timestamps uniques all  : {len(ts_all)} / {len(allb)}")
    print(f"timestamps uniques lost : {len(ts_lost)} / {len(lost)}")
    print(f"timestamps uniques won  : {len(ts_won)} / {len(won)}")
    print(f"lost INTER won (doit etre vide) : {len(ts_lost & ts_won)}")
    print(f"(lost UNION won) symdiff all     : {len((ts_lost | ts_won) ^ ts_all)}")

    banner("COLONNE result / team_decision / next_candle")
    for c in ["result", "team_decision", "next_candle", "volatility_state",
              "is_weekend", "take_up", "take_down"]:
        if c in allb.columns:
            print(f"\n[{c}]")
            print(allb[c].value_counts(dropna=False).to_string())

    banner("COHERENCE result <-> direction <-> candle reelle")
    # candle reelle a partir de next_open / next_close
    allb["_real_green"] = allb["next_close"] > allb["next_open"]
    allb["_real_red"] = allb["next_close"] < allb["next_open"]
    allb["_real_doji"] = allb["next_close"] == allb["next_open"]
    print(f"vraies bougies vertes : {allb['_real_green'].sum()}")
    print(f"vraies bougies rouges : {allb['_real_red'].sum()}")
    print(f"vrais dojis           : {allb['_real_doji'].sum()}")
    # un pari UP gagne si bougie verte ; DOWN gagne si rouge
    exp_win = ((allb["team_decision"].str.upper() == "UP") & allb["_real_green"]) | \
              ((allb["team_decision"].str.upper() == "DOWN") & allb["_real_red"])
    res_win = allb["result"].astype(str).str.lower().isin(["win", "won", "gagne", "1", "true"])
    mism = (exp_win != res_win).sum()
    print(f"incoherences result vs (direction,candle) : {mism}")
    ct = pd.crosstab(allb["team_decision"], allb["result"])
    print("\ncrosstab team_decision x result :")
    print(ct.to_string())
    ct2 = pd.crosstab(allb["team_decision"], allb["next_candle"])
    print("\ncrosstab team_decision x next_candle :")
    print(ct2.to_string())

    banner("BASELINE")
    n = len(allb)
    n_loss = res_win.eq(False).sum()
    print(f"taux de perte global : {n_loss}/{n} = {n_loss / n:.4f}")
    print(f"win rate global      : {res_win.mean():.4f}")
    for d in allb["team_decision"].unique():
        sub = allb[allb["team_decision"] == d]
        subwin = res_win[allb["team_decision"] == d]
        print(f"  {d:>5} : {len(sub):>5} paris | WR {subwin.mean():.4f} | "
              f"loss {1 - subwin.mean():.4f}")

    banner("COUVERTURE TEMPORELLE")
    dt = pd.to_datetime(allb["datetime_utc"])
    print(f"debut : {dt.min()}")
    print(f"fin   : {dt.max()}")
    print(f"duree : {(dt.max() - dt.min()).days} jours")
    print(f"timestamps strictement croissants : {allb['timestamp_ms'].is_monotonic_increasing}")
    print(f"doublons de timestamp : {allb['timestamp_ms'].duplicated().sum()}")

    banner("COLONNES CONSTANTES (variance nulle)")
    const_cols, var_cols = [], []
    for c in allb.columns:
        if c.startswith("_"):
            continue
        nun = allb[c].nunique(dropna=False)
        (const_cols if nun <= 1 else var_cols).append((c, nun))
    print(f"{len(const_cols)} colonnes constantes / {len(var_cols)} variables\n")
    print("--- CONSTANTES (contexte modele, inutiles pour discriminer) ---")
    for c, _ in const_cols:
        val = allb[c].iloc[0]
        print(f"  {c:<34} = {val}")

    banner("COLONNES VARIABLES — resume")
    for c, nun in var_cols:
        s = allb[c]
        if pd.api.types.is_numeric_dtype(s) and nun > 12:
            q = s.quantile([0, .01, .25, .5, .75, .99, 1]).values
            print(f"  {c:<26} nun={nun:<5} miss={s.isna().sum():<4} "
                  f"min={q[0]:.4g} p1={q[1]:.4g} med={q[3]:.4g} "
                  f"p99={q[5]:.4g} max={q[6]:.4g}")
        else:
            vc = s.value_counts(dropna=False)
            head = ", ".join(f"{k}:{v}" for k, v in vc.head(8).items())
            print(f"  {c:<26} nun={nun:<5} miss={s.isna().sum():<4} [{head}]")

    banner("RELATION p_up / p_down / bet_confidence")
    print("bet_confidence == p_up sur paris UP   :",
          np.allclose(allb.loc[allb.team_decision.str.upper() == "UP", "bet_confidence"],
                      allb.loc[allb.team_decision.str.upper() == "UP", "p_up"], atol=1e-6))
    print("bet_confidence == p_down sur paris DOWN :",
          np.allclose(allb.loc[allb.team_decision.str.upper() == "DOWN", "bet_confidence"],
                      allb.loc[allb.team_decision.str.upper() == "DOWN", "p_down"], atol=1e-6))
    print("bet_edge == bet_confidence - bet_threshold :",
          np.allclose(allb["bet_edge"], allb["bet_confidence"] - allb["bet_threshold"], atol=1e-6))
    print(f"\np_up   : min={allb.p_up.min():.4f} max={allb.p_up.max():.4f}")
    print(f"p_down : min={allb.p_down.min():.4f} max={allb.p_down.max():.4f}")
    print(f"p_up + p_down : min={(allb.p_up + allb.p_down).min():.4f} "
          f"med={(allb.p_up + allb.p_down).median():.4f} "
          f"max={(allb.p_up + allb.p_down).max():.4f}")
    print(f"take_up + take_down par ligne — value counts :")
    print((allb["take_up"] + allb["take_down"]).value_counts().to_string())


if __name__ == "__main__":
    main()
