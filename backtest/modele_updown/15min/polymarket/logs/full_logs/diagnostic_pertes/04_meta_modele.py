"""
04_meta_modele.py — Les pertes sont-elles predictibles ? (test decisif)

On entraine un META-MODELE a predire `loss` (1 = pari perdu) a partir des 56
predicteurs connus a la decision. Si un meta-modele puissant, valide en
cross-validation TEMPORELLE (jamais de fuite du futur), atteint une AUC
nettement > 0.50, alors les pertes sont structurees et un meta-filtre est
exploitable. Si l'AUC reste collee a 0.50, les pertes sont du bruit
irreductible a partir de l'information disponible.

Etapes :
  1. CV temporelle (TimeSeriesSplit) — XGBoost, LightGBM, Regression logistique.
  2. Bande de bruit : meme CV sur des labels permutes (a quoi ressemble 0.50 ?).
  3. Walk-forward : predictions OOS pour ~2700 paris, AUC OOS honnete.
  4. TreeSHAP natif XGBoost — sur quoi le meta-modele s'appuie-t-il ?
  5. Simulation de meta-filtre : si on saute les X % paris les plus a risque
     (selon les predictions OOS), quel WR et quel volume restant ?

Sorties : resultats/04_*.csv  +  resultats/04_meta_modele.png
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import lib

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
RNG = np.random.default_rng(0)


def xgb() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=300, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=20,
        reg_lambda=1.0, random_state=0, eval_metric="logloss",
        n_jobs=-1, verbosity=0)


def lgbm() -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=300, max_depth=3, num_leaves=8, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=30,
        reg_lambda=1.0, random_state=0, n_jobs=-1, verbose=-1)


def logreg():
    return make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=2000, C=0.3))


def cv_auc(model_fn, X: np.ndarray, y: np.ndarray, n_splits: int = 6) -> list:
    """AUC en validation croisee temporelle (train = passe, test = futur)."""
    tss = TimeSeriesSplit(n_splits=n_splits)
    aucs = []
    for tr, te in tss.split(X):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        m = model_fn()
        m.fit(X[tr], y[tr])
        p = m.predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y[te], p))
    return aucs


def section(txt: str) -> None:
    print(f"\n{'=' * 78}\n{txt}\n{'=' * 78}")


def main() -> None:
    df = lib.load()
    feats = lib.feature_cols(df)
    df["dir_up"] = (df["dir"] == "UP").astype(int)
    cols = feats + ["dir_up"]
    X = df[cols].fillna(df[cols].median()).values.astype(float)
    y = df["loss"].values.astype(int)
    base_loss = y.mean()
    base_wr = 1 - base_loss
    print(f"{len(df)} paris | {len(cols)} predicteurs | baseline perte {base_loss:.4f}")

    # ----- 1. CV temporelle, 3 familles de modeles -----
    section("1. AUC EN VALIDATION CROISEE TEMPORELLE (predire la perte)")
    print(f"{'modele':<22}{'AUC moy':>10}{'ecart-type':>12}   folds")
    results = {}
    for name, fn in [("XGBoost", xgb), ("LightGBM", lgbm),
                     ("Reg. logistique", logreg)]:
        aucs = cv_auc(fn, X, y)
        results[name] = aucs
        folds = " ".join(f"{a:.3f}" for a in aucs)
        print(f"{name:<22}{np.mean(aucs):>10.4f}{np.std(aucs):>12.4f}   {folds}")

    # ----- 2. Bande de bruit : labels permutes -----
    section("2. BANDE DE BRUIT — meme CV sur labels permutes (XGBoost)")
    null_aucs = []
    for _ in range(40):
        yp = RNG.permutation(y)
        null_aucs.append(np.mean(cv_auc(xgb, X, yp, n_splits=4)))
    null_aucs = np.array(null_aucs)
    lo, hi = np.percentile(null_aucs, [2.5, 97.5])
    real_xgb = np.mean(results["XGBoost"])
    print(f"AUC sous H0 (aucun signal) : moyenne {null_aucs.mean():.4f}, "
          f"IC95 [{lo:.4f}, {hi:.4f}]")
    print(f"AUC reelle XGBoost         : {real_xgb:.4f}")
    verdict = ("AU-DESSUS du bruit -> signal exploitable"
               if real_xgb > hi else
               "DANS la bande de bruit -> pertes ~ irreductibles")
    print(f"VERDICT : {verdict}")

    # ----- 3. Walk-forward : predictions OOS -----
    section("3. WALK-FORWARD — predictions hors-echantillon")
    min_train, step = 1400, 270
    oos_idx, oos_p = [], []
    k = min_train
    while k < len(df):
        end = min(k + step, len(df))
        m = xgb()
        m.fit(X[:k], y[:k])
        oos_p.extend(m.predict_proba(X[k:end])[:, 1])
        oos_idx.extend(range(k, end))
        k = end
    oos_idx = np.array(oos_idx)
    oos_p = np.array(oos_p)
    y_oos = y[oos_idx]
    auc_oos = roc_auc_score(y_oos, oos_p)
    print(f"{len(oos_idx)} paris predits OOS | AUC walk-forward = {auc_oos:.4f}")
    oos = df.iloc[oos_idx].copy()
    oos["p_loss"] = oos_p

    # ----- 4. TreeSHAP natif (importance descriptive) -----
    section("4. TreeSHAP — features sur lesquelles le meta-modele s'appuie")
    m_full = xgb()
    m_full.fit(X, y)
    booster = m_full.get_booster()
    import xgboost as xgbmod
    contribs = booster.predict(xgbmod.DMatrix(X, feature_names=cols),
                               pred_contribs=True)
    shap_imp = np.abs(contribs[:, :-1]).mean(axis=0)
    shap_df = pd.DataFrame({"feature": cols, "shap_abs_moy": shap_imp}) \
        .sort_values("shap_abs_moy", ascending=False).reset_index(drop=True)
    shap_df.to_csv(lib.OUT_DIR / "04_shap_importance.csv", index=False)
    print(shap_df.head(15).to_string(index=False))

    # ----- 5. Simulation de meta-filtre -----
    section("5. META-FILTRE — sauter les X % paris les plus a risque (OOS)")
    print(f"Reference OOS : {len(oos)} paris, WR {oos['win'].mean():.4f}\n")
    print(f"{'skip %':>8}{'n sautes':>10}{'n gardes':>10}"
          f"{'WR gardes':>12}{'WR sautes':>12}{'gain pts':>10}")
    rows = []
    for pct in [5, 10, 15, 20, 25, 30, 40, 50]:
        thr = np.percentile(oos["p_loss"], 100 - pct)
        skip = oos["p_loss"] >= thr
        kept = oos.loc[~skip]
        wr_k = kept["win"].mean()
        wr_s = oos.loc[skip, "win"].mean()
        gain = 100 * (wr_k - oos["win"].mean())
        rows.append(dict(skip_pct=pct, n_skip=int(skip.sum()),
                         n_kept=len(kept), wr_kept=wr_k, wr_skip=wr_s,
                         gain_pts=gain))
        print(f"{pct:>7}%{int(skip.sum()):>10}{len(kept):>10}"
              f"{wr_k:>12.4f}{wr_s:>12.4f}{gain:>+10.2f}")
    filt = pd.DataFrame(rows)
    filt.to_csv(lib.OUT_DIR / "04_metafiltre.csv", index=False)

    # robustesse : le filtre tient-il sur les 2 moities de la periode OOS ?
    print("\nRobustesse du meta-filtre (skip 25%) sur les 2 moities OOS :")
    half = oos["t_frac"] >= oos["t_frac"].median()
    thr25 = np.percentile(oos["p_loss"], 75)
    for lbl, m in [("OOS moitie 1", ~half), ("OOS moitie 2", half)]:
        seg = oos.loc[m]
        kept = seg.loc[seg["p_loss"] < thr25]
        print(f"  {lbl}: WR avant {seg['win'].mean():.4f} -> "
              f"apres {kept['win'].mean():.4f} "
              f"({100 * (kept['win'].mean() - seg['win'].mean()):+.2f} pts)")

    # ----- figure -----
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].axhline(base_wr, color="r", ls=":", label=f"WR actuel {base_wr:.3f}")
    ax[0].plot(filt["skip_pct"], filt["wr_kept"], "o-", label="WR apres filtre")
    ax[0].set_xlabel("% de paris sautes (les plus a risque)")
    ax[0].set_ylabel("WR des paris gardes (OOS)")
    ax[0].set_title(f"Meta-filtre walk-forward (AUC OOS {auc_oos:.3f})")
    ax[0].legend()
    ax[0].grid(alpha=.3)
    top = shap_df.head(15).iloc[::-1]
    ax[1].barh(top["feature"], top["shap_abs_moy"], color="teal")
    ax[1].set_title("Importance TreeSHAP (|SHAP| moyen)")
    ax[1].set_xlabel("|SHAP| moyen")
    fig.tight_layout()
    fig.savefig(lib.OUT_DIR / "04_meta_modele.png", dpi=110)
    print(f"\n[OK] Figure -> resultats/04_meta_modele.png")

    # synthese chiffree pour le rapport
    pd.DataFrame({
        "metrique": ["auc_cv_xgb", "auc_cv_lgbm", "auc_cv_logreg",
                     "auc_null_moy", "auc_null_hi", "auc_walkforward"],
        "valeur": [np.mean(results["XGBoost"]), np.mean(results["LightGBM"]),
                   np.mean(results["Reg. logistique"]), null_aucs.mean(),
                   hi, auc_oos],
    }).to_csv(lib.OUT_DIR / "04_synthese_auc.csv", index=False)


if __name__ == "__main__":
    main()
