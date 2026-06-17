"""
classify_features3.py — classification d'impact des features en 3 CLASSES.

Refonte de classify_features.py, plus JUSTE :
  - L'ancien script utilisait une bande de bruit = 2*sigma. Sous la config de
    prod (subsample/colsample=0.8), sigma ~= 0.0008 => bande ~0.0016, ce qui
    ecrasait presque toutes les features en "inutile" (artefact du bruit
    stochastique de XGBoost, PAS une vraie absence d'impact).
  - Ici on MESURE de facon DETERMINISTE : subsample=colsample=1.0 => sigma~0,
    donc le delta d'ablation est stable et on classe par SIGNE avec une petite
    zone morte fixe (DEAD), pas par 2*sigma.

DEUX mesures complementaires (une feature peut etre "nulle" par l'une et utile
par l'autre) :

  A) IMPACT MARGINAL (ablation leave-one-out sur le modele COMPLET) :
       delta = AUC(sans la feature) - AUC(toutes)
         delta < -DEAD  -> IMPACT POSITIF   (la retirer fait BAISSER l'AUC -> utile)
         |delta| <= DEAD -> IMPACT NUL       (ne change rien -> redondante ou inerte)
         delta > +DEAD  -> IMPACT NEGATIF   (la retirer fait MONTER l'AUC -> nuisible)
     NB : l'ablation est SEVERE avec la redondance : une feature tres
     predictive mais dont un clone existe ailleurs montre delta~0 (NULLE), car
     le modele a un substitut. C'est pourquoi on ajoute la mesure B.

  B) POUVOIR STANDALONE (la feature SEULE vs 0.5) :
       power = AUC(feature seule) - 0.5
     Mesure le signal BRUT de la feature, NON penalise par la redondance.
     power > STANDALONE_EPS -> la feature porte un signal exploitable seule.

Lecture croisee :
  - POSITIF (A)                  : utile ET unique (le modele en a besoin).
  - NUL (A) + power eleve (B)    : utile mais REDONDANTE (substitut presente).
  - NUL (A) + power faible (B)   : vraiment INERTE (aucun signal).
  - NEGATIF (A)                  : NUISIBLE (la retirer ameliore le modele).

Usage (depuis la racine du package) :
    cd backtest/modele_updown/30min/modele_down
    python analyse/classify_features3.py
    python analyse/classify_features3.py --dead 0.0005    # zone morte plus large
    python analyse/classify_features3.py --features 27    # n'analyse que la whitelist du modele
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

_PKG_ROOT = Path(__file__).resolve().parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from config import (DB_PATH, TRAIN_DAYS, TEST_DAYS, VAL_FRACTION, XGB_RANDOM_STATE,
                    FEATURE_WHITELIST)
from features import build_training_dataframe
from train import build_hyperparams, _resolve_scale_pos_weight

EXCLUDED = {"timestamp", "label", "volatility_state", "is_weekend"}

# Config de MESURE deterministe (annule le bruit stochastique -> sigma~0).
STABLE_OVERRIDES = {"subsample": 1.0, "colsample_bytree": 1.0}
DEAD = 0.0003            # zone morte : |delta| <= DEAD -> impact nul
STANDALONE_EPS = 0.002   # power > EPS -> la feature a un signal brut exploitable


def _split(df):
    df = df.sort_values("timestamp").reset_index(drop=True)
    ms = 86_400_000
    ts_max = df["timestamp"].max()
    test_cut = ts_max - TEST_DAYS * ms
    test_start = int(df["timestamp"].searchsorted(test_cut, side="left"))
    tv_start = 0 if TRAIN_DAYS is None else int(df["timestamp"].searchsorted(test_cut - TRAIN_DAYS * ms, side="left"))
    val_size = int((test_start - tv_start) * VAL_FRACTION)
    return df.iloc[tv_start:test_start - val_size], df.iloc[test_start - val_size:test_start], df.iloc[test_start:]


def _train_auc(train, val, test, feature_cols, seed) -> float:
    if not feature_cols:
        return 0.5
    hp = build_hyperparams(STABLE_OVERRIDES)
    optional = {}
    if hp["reg_lambda"] is not False: optional["reg_lambda"] = hp["reg_lambda"]
    if hp["reg_alpha"] is not False: optional["reg_alpha"] = hp["reg_alpha"]
    if hp["gamma"] is not False: optional["gamma"] = hp["gamma"]
    spw = _resolve_scale_pos_weight(hp["scale_pos_weight"], train["label"])
    if spw is not None:
        optional["scale_pos_weight"] = spw
    model = XGBClassifier(
        n_estimators=hp["n_estimators"], max_depth=hp["max_depth"],
        learning_rate=hp["learning_rate"], min_child_weight=hp["min_child_weight"],
        subsample=hp["subsample"], colsample_bytree=hp["colsample_bytree"],
        objective="binary:logistic", eval_metric="logloss",
        early_stopping_rounds=hp["early_stopping_rounds"], random_state=seed,
        n_jobs=-1, **optional,
    )
    model.fit(train[feature_cols], train["label"],
              eval_set=[(val[feature_cols], val["label"])], verbose=False)
    return float(roc_auc_score(test["label"], model.predict_proba(test[feature_cols])[:, 1]))


def impact_class(delta, dead):
    if delta < -dead:
        return "POSITIF"
    if delta > dead:
        return "NEGATIF"
    return "NUL"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dead", type=float, default=DEAD, help="zone morte |delta|<=dead -> nul")
    ap.add_argument("--features", choices=["all", "27"], default="all",
                    help="'all' = toutes les colonnes du parquet ; '27' = la FEATURE_WHITELIST")
    args = ap.parse_args()

    t0 = time.time()
    df = build_training_dataframe(DB_PATH)
    df = df.sort_values("timestamp").reset_index(drop=True)
    train, val, test = _split(df)
    print(f"Split : {len(train):,} train / {len(val):,} val / {len(test):,} test "
          f"| label_test={test['label'].mean():.4f}")
    print(f"Config de mesure : DETERMINISTE {STABLE_OVERRIDES} (sigma~0, pas de bande de bruit)")

    if args.features == "27" and FEATURE_WHITELIST:
        feats = [c for c in FEATURE_WHITELIST if c in df.columns]
    else:
        feats = [c for c in df.columns if c not in EXCLUDED]
    print(f"Features analysees : {len(feats)}\n")

    # Determinisme : verifie sigma sur 2 seeds.
    auc_all = _train_auc(train, val, test, feats, XGB_RANDOM_STATE)
    auc_all2 = _train_auc(train, val, test, feats, XGB_RANDOM_STATE + 7)
    sigma = abs(auc_all - auc_all2)
    print(f"AUC modele complet = {auc_all:.5f} | sigma(2 seeds) = {sigma:.6f} "
          f"| zone morte DEAD = {args.dead}\n")

    rows = []
    for k, f in enumerate(feats, 1):
        cols = [c for c in feats if c != f]
        delta = _train_auc(train, val, test, cols, XGB_RANDOM_STATE) - auc_all      # ablation
        standalone = _train_auc(train, val, test, [f], XGB_RANDOM_STATE)            # seule
        power = standalone - 0.5
        rows.append({"feature": f, "delta": delta, "standalone": standalone,
                     "power": power, "classe": impact_class(delta, args.dead)})
        print(f"  [{k:>3}/{len(feats)}] {f:<22} delta={delta:+.5f} standalone={standalone:.4f} "
              f"-> {rows[-1]['classe']}")

    res = pd.DataFrame(rows).sort_values("delta").reset_index(drop=True)

    n_pos = int((res["classe"] == "POSITIF").sum())
    n_nul = int((res["classe"] == "NUL").sum())
    n_neg = int((res["classe"] == "NEGATIF").sum())
    # sous-decomposition du NUL via le pouvoir standalone
    nul = res[res["classe"] == "NUL"]
    n_nul_redondant = int((nul["power"] > STANDALONE_EPS).sum())   # utile mais redondante
    n_nul_inerte = n_nul - n_nul_redondant                         # vraiment sans signal

    print("\n" + "=" * 78)
    print("  CLASSEMENT 3 CLASSES (impact marginal d'ablation, modele deterministe)")
    print("=" * 78)
    print(f"  {'feature':<24}{'delta':>11}{'standalone':>12}{'power':>10}   classe")
    print(f"  {'-' * 74}")
    for _, r in res.iterrows():
        print(f"  {r['feature']:<24}{r['delta']:>+11.5f}{r['standalone']:>12.4f}"
              f"{r['power']:>+10.4f}   {r['classe']}")

    print("\n" + "=" * 78)
    print("  RESUME")
    print("=" * 78)
    print(f"  Total features analysees : {len(feats)}")
    print(f"  IMPACT POSITIF (utiles, uniques)   : {n_pos}")
    print(f"  IMPACT NUL (sans impact marginal)  : {n_nul}")
    print(f"      dont REDONDANTES (signal brut, mais substitut existe) : {n_nul_redondant}")
    print(f"      dont INERTES (aucun signal brut)                       : {n_nul_inerte}")
    print(f"  IMPACT NEGATIF (nuisibles)         : {n_neg}")
    print(f"\n  >> UTILES (positif OU redondant-avec-signal) : {n_pos + n_nul_redondant}")
    print(f"  >> INUTILES (inertes)                        : {n_nul_inerte}")
    print(f"  >> MAUVAISES (nuisibles)                     : {n_neg}")

    # sensibilite a la zone morte
    print(f"\n  Sensibilite a la zone morte (DEAD) :")
    for d in (0.0001, 0.0003, 0.0005, 0.0010):
        p = int((res["delta"] < -d).sum())
        ng = int((res["delta"] > d).sum())
        nu = len(res) - p - ng
        print(f"    DEAD={d:.4f} : POSITIF={p:>3}  NUL={nu:>3}  NEGATIF={ng:>3}")

    # markdown
    out = _PKG_ROOT / "analyse" / "feature_classes3.md"
    lines = [
        "# Classement d'impact des features (3 classes, modele deterministe)", "",
        f"- Features analysees : **{len(feats)}**",
        f"- AUC modele complet : **{auc_all:.5f}** (sigma 2 seeds = {sigma:.6f}, deterministe)",
        f"- Config de mesure : `{STABLE_OVERRIDES}` | zone morte DEAD = {args.dead}",
        f"- Split : {len(train):,} train / {len(val):,} val / {len(test):,} test", "",
        "## Resume", "",
        f"- **IMPACT POSITIF (utiles)** : {n_pos}",
        f"- **IMPACT NUL** : {n_nul} (dont {n_nul_redondant} redondantes-avec-signal, {n_nul_inerte} inertes)",
        f"- **IMPACT NEGATIF (nuisibles)** : {n_neg}", "",
        f"- **Utiles** (positif + redondant-avec-signal) : **{n_pos + n_nul_redondant}**",
        f"- **Inutiles** (inertes) : **{n_nul_inerte}**",
        f"- **Mauvaises** (nuisibles) : **{n_neg}**", "",
        "| feature | delta_ablation | standalone_auc | power | classe |",
        "|---|---:|---:|---:|---|",
    ]
    for _, r in res.iterrows():
        lines.append(f"| {r['feature']} | {r['delta']:+.5f} | {r['standalone']:.4f} "
                     f"| {r['power']:+.4f} | {r['classe']} |")
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  -> ecrit {out.relative_to(_PKG_ROOT)} | {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
