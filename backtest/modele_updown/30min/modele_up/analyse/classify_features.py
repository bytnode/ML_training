"""
classify_features.py
=====================

Classe TOUTES les features du modele DOWN courant en 4 categories, par ABLATION
leave-one-out (la seule methode qui detecte les features NUISIBLES : les retirer
ameliore le modele) :

    - TRES UTILE              : la retirer fait chuter nettement l'AUC test
    - MOYENNEMENT UTILE       : la retirer fait baisser un peu l'AUC test
    - INUTILE                 : la retirer ne change rien (dans le bruit)
    - IMPACTANTE NEGATIVEMENT : la retirer AMELIORE l'AUC test (= feature nuisible)

Methode
-------
Metric = AUC sur le test set (pouvoir discriminant, independant du seuil/calibration,
beaucoup plus stable que le WR pour comparer des features).

    delta = AUC(sans la feature) - AUC(baseline avec toutes)   [au meme seed]
      delta << 0  -> la feature aide beaucoup   (tres utile)
      delta <  0  -> la feature aide un peu      (moyennement utile)
      delta ~  0  -> la feature n'apporte rien   (inutile)
      delta >  0  -> la feature degrade           (impactante negativement)

La BANDE DE BRUIT est estimee en re-entrainant le baseline sur plusieurs seeds :
band = max(2*sigma_AUC, ABS_FLOOR). On classe `delta` par rapport a cette bande,
ce qui evite de sur-interpreter du bruit XGBoost (n_jobs=-1 n'est pas 100% deterministe).

Entrainement = miroir EXACT de train.py (memes hyperparams, early stopping sur val,
scale_pos_weight, meme split temporel). Les features sont celles du modele entraine
courant (booster.feature_names), donc le classement porte sur le modele reellement
deploye.

Usage (depuis la racine du package) :
    cd backtest/modele_updown/30min/modele_down
    python analyse/classify_features.py
    python analyse/classify_features.py --noise-seeds 5   # bande de bruit plus fine
    python analyse/classify_features.py --top 0            # n'affiche que le tableau complet
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from xgboost import XGBClassifier

# Racine du package sur sys.path (le script vit dans analyse/).
_PKG_ROOT = Path(__file__).resolve().parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from config import DB_PATH, TRAIN_DAYS, TEST_DAYS, VAL_FRACTION, MODEL_OUTPUT, XGB_RANDOM_STATE
from features import build_training_dataframe
from train import build_hyperparams, _resolve_scale_pos_weight

EXCLUDED = {"timestamp", "label", "volatility_state"}

# Bornes de classification (en points d'AUC), relatives a la bande de bruit.
ABS_FLOOR = 0.0005          # plancher de la bande "inutile" si le bruit est minuscule
USELESS_K = 2.0             # |delta| <= USELESS_K * band -> inutile
VERY_K = 3.0                # delta <= -VERY_K * band -> tres utile


def _split(df: pd.DataFrame):
    """Split temporel identique a train.py (sort -> test = derniers TEST_DAYS,
    val = derniers VAL_FRACTION de train+val)."""
    df = df.sort_values("timestamp").reset_index(drop=True)
    ms_per_day = 86_400_000
    ts_max = df["timestamp"].max()
    test_cutoff = ts_max - TEST_DAYS * ms_per_day
    test_start = int(df["timestamp"].searchsorted(test_cutoff, side="left"))
    if TRAIN_DAYS is None:
        tv_start = 0
    else:
        tv_cutoff = test_cutoff - TRAIN_DAYS * ms_per_day
        tv_start = int(df["timestamp"].searchsorted(tv_cutoff, side="left"))
    n_tv = test_start - tv_start
    val_size = int(n_tv * VAL_FRACTION)
    train_end = test_start - val_size
    return df.iloc[tv_start:train_end], df.iloc[train_end:test_start], df.iloc[test_start:]


def _train_auc(train, val, test, feature_cols: list[str], seed: int) -> tuple[float, float]:
    """Entraine (production-parity) et renvoie (auc_test, logloss_test)."""
    X_train, y_train = train[feature_cols], train["label"]
    X_val, y_val = val[feature_cols], val["label"]
    X_test, y_test = test[feature_cols], test["label"]

    hp = build_hyperparams()
    optional = {}
    if hp["reg_lambda"] is not False: optional["reg_lambda"] = hp["reg_lambda"]
    if hp["reg_alpha"] is not False: optional["reg_alpha"] = hp["reg_alpha"]
    if hp["gamma"] is not False: optional["gamma"] = hp["gamma"]
    spw = _resolve_scale_pos_weight(hp["scale_pos_weight"], y_train)
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
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    p = model.predict_proba(X_test)[:, 1]
    return float(roc_auc_score(y_test, p)), float(log_loss(y_test, p))


def _feature_cols_from_model() -> list[str] | None:
    """Features du modele entraine courant (la verite sur ce qui est deploye)."""
    mp = _PKG_ROOT / MODEL_OUTPUT
    if not mp.exists():
        return None
    m = XGBClassifier()
    m.load_model(str(mp))
    return list(m.get_booster().feature_names)


def _gain_importance(train, val, feature_cols, seed) -> dict:
    """Gain importance du baseline (contexte secondaire)."""
    X_train, y_train = train[feature_cols], train["label"]
    X_val, y_val = val[feature_cols], val["label"]
    hp = build_hyperparams()
    optional = {}
    if hp["reg_lambda"] is not False: optional["reg_lambda"] = hp["reg_lambda"]
    if hp["reg_alpha"] is not False: optional["reg_alpha"] = hp["reg_alpha"]
    if hp["gamma"] is not False: optional["gamma"] = hp["gamma"]
    spw = _resolve_scale_pos_weight(hp["scale_pos_weight"], y_train)
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
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    booster = model.get_booster()
    score = booster.get_score(importance_type="gain")
    return {f: float(score.get(f, 0.0)) for f in feature_cols}


def classify(delta: float, band: float) -> str:
    if delta > band:
        return "IMPACTANTE NEGATIVEMENT"
    if abs(delta) <= band:
        return "INUTILE"
    if delta <= -VERY_K * band:
        return "TRES UTILE"
    return "MOYENNEMENT UTILE"


def main() -> None:
    ap = argparse.ArgumentParser(description="Classe les features du modele DOWN en 4 categories (ablation AUC).")
    ap.add_argument("--noise-seeds", type=int, default=3,
                    help="Nb de seeds supplementaires pour estimer la bande de bruit (defaut 3).")
    ap.add_argument("--top", type=int, default=12, help="Nb de features extremes a mettre en avant (0 = off).")
    args = ap.parse_args()

    t0 = time.time()
    df = build_training_dataframe(DB_PATH)
    df = df.sort_values("timestamp").reset_index(drop=True)

    feature_cols = _feature_cols_from_model()
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c not in EXCLUDED]
        print(f"[warn] modele introuvable -> features deduites du parquet ({len(feature_cols)}).")
    else:
        print(f"Features du modele courant : {len(feature_cols)}")

    train, val, test = _split(df)
    print(f"Split : {len(train):,} train / {len(val):,} val / {len(test):,} test "
          f"| label_test={test['label'].mean():.4f}")

    # Baseline au seed de prod + seeds de bruit.
    seeds = [XGB_RANDOM_STATE] + [XGB_RANDOM_STATE + 100 + i for i in range(args.noise_seeds)]
    base_aucs = []
    auc_base, ll_base = None, None
    for i, s in enumerate(seeds):
        auc, ll = _train_auc(train, val, test, feature_cols, s)
        base_aucs.append(auc)
        if i == 0:
            auc_base, ll_base = auc, ll
        print(f"  baseline seed={s}: AUC={auc:.5f} logloss={ll:.5f}")
    sigma = float(np.std(base_aucs)) if len(base_aucs) > 1 else 0.0
    band = max(USELESS_K * sigma, ABS_FLOOR)
    print(f"\nBaseline AUC (seed prod) = {auc_base:.5f} | sigma_bruit = {sigma:.5f} "
          f"| bande = {band:.5f}\n")

    gain = _gain_importance(train, val, feature_cols, XGB_RANDOM_STATE)

    # Ablation leave-one-out.
    rows = []
    for k, feat in enumerate(feature_cols, 1):
        cols = [c for c in feature_cols if c != feat]
        auc_abl, ll_abl = _train_auc(train, val, test, cols, XGB_RANDOM_STATE)
        delta_auc = auc_abl - auc_base          # >0 = retirer aide = nuisible
        delta_ll = ll_abl - ll_base             # >0 = retirer degrade logloss = utile
        rows.append({
            "feature": feat,
            "delta_auc": delta_auc,
            "delta_logloss": delta_ll,
            "gain": gain.get(feat, 0.0),
            "classe": classify(delta_auc, band),
        })
        print(f"  [{k:>2}/{len(feature_cols)}] {feat:<22} dAUC={delta_auc:+.5f}  -> {rows[-1]['classe']}")

    res = pd.DataFrame(rows).sort_values("delta_auc").reset_index(drop=True)

    order = ["TRES UTILE", "MOYENNEMENT UTILE", "INUTILE", "IMPACTANTE NEGATIVEMENT"]
    print("\n" + "=" * 78)
    print("  CLASSEMENT DES FEATURES (trie : plus utile -> plus nuisible)")
    print(f"  bande de bruit = {band:.5f} AUC  (|dAUC| <= bande => inutile)")
    print("=" * 78)
    print(f"  {'Feature':<24}{'dAUC':>11}{'dLogloss':>11}{'gain':>10}   Classe")
    print(f"  {'-' * 74}")
    for _, r in res.iterrows():
        print(f"  {r['feature']:<24}{r['delta_auc']:>+11.5f}{r['delta_logloss']:>+11.5f}"
              f"{r['gain']:>10.1f}   {r['classe']}")

    print("\n  --- Repartition ---")
    for cls in order:
        feats = res[res["classe"] == cls]["feature"].tolist()
        print(f"  {cls:<26} {len(feats):>2} : {', '.join(feats) if feats else '-'}")

    # Markdown
    out = _PKG_ROOT / "analyse" / "feature_classes.md"
    lines = [
        "# Classement des features (modele DOWN) — ablation AUC",
        "",
        f"- Features analysees : **{len(feature_cols)}**",
        f"- Baseline AUC test (seed prod) : **{auc_base:.5f}**",
        f"- Sigma bruit ({len(seeds)} seeds) : {sigma:.5f} — bande = **{band:.5f}** AUC",
        f"- Split : {len(train):,} train / {len(val):,} val / {len(test):,} test",
        "",
        "`delta_auc = AUC(sans la feature) - AUC(baseline)`. "
        "delta<0 = la feature aide ; delta>0 = elle nuit.",
        "",
        "| Feature | dAUC | dLogloss | gain | Classe |",
        "|---|---:|---:|---:|---|",
    ]
    for _, r in res.iterrows():
        lines.append(f"| {r['feature']} | {r['delta_auc']:+.5f} | {r['delta_logloss']:+.5f} "
                     f"| {r['gain']:.1f} | {r['classe']} |")
    lines += ["", "## Repartition", ""]
    for cls in order:
        feats = res[res["classe"] == cls]["feature"].tolist()
        lines.append(f"- **{cls}** ({len(feats)}) : {', '.join(feats) if feats else '-'}")
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  -> ecrit {out.relative_to(_PKG_ROOT)}  ({len(feature_cols)} features)")
    print(f"  duree : {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
