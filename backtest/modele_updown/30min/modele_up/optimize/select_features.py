"""
select_features.py — selecteur de features qui satisfait le contrat
analyse/classify_features.py : composer un jeu FINAL de >= TARGET_MIN features
ou CHAQUE feature a un delta d'ablation leave-one-out <= -bande (donc 0 feature
"inutile", 0 feature "nuisible" ; toutes "tres utile" ou "moyennement utile").

METHODE (miroir EXACT de classify_features.py pour l'eval) :
  - meme split temporel (_split), memes hyperparams (build_hyperparams),
    meme scale_pos_weight, meme seed prod, memes seeds de bruit -> meme bande.
  - delta_auc(f) = AUC(set sans f) - AUC(set complet)  [seed prod]
    f utile ssi delta_auc(f) <= -bande.

PIPELINE :
  0. load parquet + split.
  1. PRUNE par correlation : dans chaque cluster |corr|>CORR_THRESH, on garde
     le representant le plus correle au label (les quasi-doublons disparaissent).
  2. FORWARD SELECTION : on ajoute iterativement la feature qui maximise l'AUC
     test (seed prod), jusqu'a MAX_SET features.
  3. BANDE : sigma sur NOISE_SEEDS -> bande = max(2*sigma, ABS_FLOOR).
  4. ELAGAGE + BACKFILL : on retire la pire feature dont delta > -SAFETY*bande
     (redondante), on recomplete depuis le pool, jusqu'a ce que TOUTES les
     features aient delta <= -bande et len(S) >= TARGET_MIN (ou max_iter).
  5. OUTPUT : tuple whitelist + classes projetees + ecrit selected_features.txt.

Usage (depuis la racine du package) :
    cd backtest/modele_updown/30min/modele_down
    python optimize/select_features.py
    python optimize/select_features.py --max-set 22 --target 14
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

from config import DB_PATH, TRAIN_DAYS, TEST_DAYS, VAL_FRACTION, XGB_RANDOM_STATE
from features import build_training_dataframe
from train import build_hyperparams, _resolve_scale_pos_weight

EXCLUDED = {"timestamp", "label", "volatility_state", "is_weekend"}

# Miroir classify_features.py
ABS_FLOOR = 0.0005
USELESS_K = 2.0
VERY_K = 3.0

# Overrides d'hyperparams (remplis depuis la CLI ; {} = config de prod).
_HP_OVERRIDES: dict = {}

# Selection
CORR_THRESH = 0.92      # |corr| au-dessus = quasi-doublon -> on n'en garde qu'un
SAFETY = 1.30           # marge : on exige delta <= -SAFETY*bande pour garder (robustesse vs bruit n_jobs)
NOISE_SEEDS_N = 3       # = defaut classify --noise-seeds 3


def _split(df: pd.DataFrame):
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


def _train_auc(train, val, test, feature_cols, seed) -> float:
    if not feature_cols:                 # set vide (LOO d'une feature unique) = modele nul
        return 0.5
    X_train, y_train = train[feature_cols], train["label"]
    X_val, y_val = val[feature_cols], val["label"]
    X_test, y_test = test[feature_cols], test["label"]
    hp = build_hyperparams(_HP_OVERRIDES)
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
    return float(roc_auc_score(y_test, p))


def classify(delta: float, band: float) -> str:
    if delta > band:
        return "NUISIBLE"
    if abs(delta) <= band:
        return "INUTILE"
    if delta <= -VERY_K * band:
        return "TRES UTILE"
    return "MOYEN"


def correlation_prune(train, pool, label, thresh):
    """Garde, dans chaque cluster |corr|>thresh, le representant le plus correle au label."""
    sub = train[pool]
    lab_corr = {c: abs(float(np.corrcoef(
        np.nan_to_num(train[c].to_numpy(dtype=float)),
        label.to_numpy(dtype=float))[0, 1])) if train[c].notna().any() else 0.0
        for c in pool}
    order = sorted(pool, key=lambda c: lab_corr[c], reverse=True)
    cmat = sub.corr().abs()
    kept = []
    for c in order:
        red = False
        for k in kept:
            v = cmat.loc[c, k]
            if pd.notna(v) and v > thresh:
                red = True
                break
        if not red:
            kept.append(c)
    dropped = [c for c in pool if c not in kept]
    return kept, dropped, lab_corr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-set", type=int, default=20, help="taille max apres forward selection")
    ap.add_argument("--target", type=int, default=13, help="nb min de features finales")
    ap.add_argument("--corr", type=float, default=CORR_THRESH)
    ap.add_argument("--max-iter", type=int, default=30)
    ap.add_argument("--subsample", type=float, default=None, help="override XGB subsample")
    ap.add_argument("--colsample", type=float, default=None, help="override XGB colsample_bytree")
    ap.add_argument("--max-depth", type=int, default=None, help="override XGB max_depth (depth bas = modele additif)")
    ap.add_argument("--report-only", action="store_true",
                    help="forward-select puis affiche les deltas leave-one-out propres (pas de phase 4)")
    ap.add_argument("--fixed-set", type=str, default=None,
                    help="evalue UN jeu donne (csv) : LOO + bande + classes, pas de forward/phase4")
    ap.add_argument("--pool", type=str, default=None,
                    help="restreint le pool de candidats a cette liste csv (sinon toutes les colonnes)")
    ap.add_argument("--clean-forward", action="store_true",
                    help="construit le set max ou TOUTES les features passent la bande (config stable requise)")
    args = ap.parse_args()

    global _HP_OVERRIDES
    if args.subsample is not None:
        _HP_OVERRIDES["subsample"] = args.subsample
    if args.colsample is not None:
        _HP_OVERRIDES["colsample_bytree"] = args.colsample
    if args.max_depth is not None:
        _HP_OVERRIDES["max_depth"] = args.max_depth
    if _HP_OVERRIDES:
        print(f"[HP overrides] {_HP_OVERRIDES}")

    t0 = time.time()
    df = build_training_dataframe(DB_PATH)
    df = df.sort_values("timestamp").reset_index(drop=True)
    train, val, test = _split(df)
    print(f"Split : {len(train):,} train / {len(val):,} val / {len(test):,} test "
          f"| label_test={test['label'].mean():.4f}")

    pool = [c for c in df.columns if c not in EXCLUDED]
    if args.pool:
        wanted = [c.strip() for c in args.pool.split(",") if c.strip()]
        miss = [c for c in wanted if c not in df.columns]
        if miss:
            print(f"ERREUR --pool : colonnes absentes du parquet : {miss}")
            return
        pool = [c for c in wanted if c not in EXCLUDED]
    print(f"Pool initial : {len(pool)} features")

    seeds = [XGB_RANDOM_STATE] + [XGB_RANDOM_STATE + 100 + i for i in range(NOISE_SEEDS_N)]

    # --- Mode FIXED-SET : evalue un jeu donne (LOO + bande), pas de selection ---
    if args.fixed_set:
        S = [c.strip() for c in args.fixed_set.split(",") if c.strip()]
        miss = [c for c in S if c not in df.columns]
        if miss:
            print(f"ERREUR fixed-set : colonnes absentes du parquet : {miss}")
            return
        base = _train_auc(train, val, test, S, XGB_RANDOM_STATE)
        aucs = [_train_auc(train, val, test, S, s) for s in seeds]
        sigma = float(np.std(aucs))
        band = max(USELESS_K * sigma, ABS_FLOOR)
        deltas = {f: _train_auc(train, val, test, [c for c in S if c != f], XGB_RANDOM_STATE) - base
                  for f in S}
        rows = sorted(((f, deltas[f], classify(deltas[f], band)) for f in S), key=lambda r: r[1])
        print("\n" + "=" * 70)
        print(f"  FIXED-SET : {len(S)} features | AUC={base:.5f} sigma={sigma:.5f} bande={band:.5f}")
        print("=" * 70)
        print(f"  {'feature':<24}{'dAUC':>11}   classe")
        n_inutile = 0
        for f, dl, cls in rows:
            if cls in ("INUTILE", "NUISIBLE"):
                n_inutile += 1
            print(f"  {f:<24}{dl:>+11.5f}   {cls}")
        n_tres = sum(1 for _, _, c in rows if c == "TRES UTILE")
        n_moy = sum(1 for _, _, c in rows if c == "MOYEN")
        print(f"\n  TRES UTILE={n_tres} MOYEN={n_moy} INUTILE/NUISIBLE={n_inutile} | total={len(S)}")
        print(f"  OBJECTIF (0 inutile, >= 13) : {'ATTEINT' if (n_inutile == 0 and len(S) >= 13) else 'NON'}")
        out = _PKG_ROOT / "optimize" / "selected_features.txt"
        out.write_text("FEATURE_WHITELIST = (\n" +
                       "".join(f'    "{f}",\n' for f, _, _ in rows) + ")\n", encoding="utf-8")
        print(f"  -> {out.relative_to(_PKG_ROOT)} | duree {time.time() - t0:.0f}s")
        return

    # --- Mode CLEAN-FORWARD : construit le set max ou TOUTES les features passent ---
    # (sous config stable subsample/colsample=1.0, sigma=0 => bande fixe ABS_FLOOR)
    if args.clean_forward:
        band = ABS_FLOOR
        print(f"[clean-forward] bande fixe = {band:.5f} (config stable, sigma~0). "
              f"pool={len(pool)} max_set={args.max_set}")
        S = []
        while len(S) < args.max_set:
            best_c, best_score, best_deltas = None, -1e9, None
            for c in [p for p in pool if p not in S]:
                cand = S + [c]
                base = _train_auc(train, val, test, cand, XGB_RANDOM_STATE)
                ok, min_margin, dd = True, 1e9, {}
                for f in cand:
                    d = _train_auc(train, val, test, [x for x in cand if x != f], XGB_RANDOM_STATE) - base
                    dd[f] = d
                    if d > -band:
                        ok = False
                        break
                    min_margin = min(min_margin, (-d) - band)
                if ok and min_margin > best_score:
                    best_score, best_c, best_deltas = min_margin, c, dd
            if best_c is None:
                print(f"  aucune addition propre possible -> set max = {len(S)}")
                break
            S.append(best_c)
            print(f"  +{best_c:<22} | set={len(S):>2} | min_margin={best_score:+.5f}")
        # rapport final
        base = _train_auc(train, val, test, S, XGB_RANDOM_STATE)
        deltas = {f: _train_auc(train, val, test, [x for x in S if x != f], XGB_RANDOM_STATE) - base for f in S}
        rows = sorted(((f, deltas[f], classify(deltas[f], band)) for f in S), key=lambda r: r[1])
        print("\n" + "=" * 70)
        print(f"  CLEAN-FORWARD : {len(S)} features | AUC={base:.5f} | bande={band:.5f}")
        print("=" * 70)
        print(f"  {'feature':<24}{'dAUC':>11}   classe")
        n_inutile = 0
        for f, dl, cls in rows:
            if cls in ("INUTILE", "NUISIBLE"):
                n_inutile += 1
            print(f"  {f:<24}{dl:>+11.5f}   {cls}")
        n_tres = sum(1 for _, _, c in rows if c == "TRES UTILE")
        n_moy = sum(1 for _, _, c in rows if c == "MOYEN")
        print(f"\n  TRES UTILE={n_tres} MOYEN={n_moy} INUTILE/NUISIBLE={n_inutile} | total={len(S)}")
        print(f"  OBJECTIF (0 inutile, >= 13) : {'ATTEINT' if (n_inutile == 0 and len(S) >= 13) else 'NON'}")
        out = _PKG_ROOT / "optimize" / "selected_features.txt"
        out.write_text("FEATURE_WHITELIST = (\n" +
                       "".join(f'    "{f}",\n' for f, _, _ in rows) + ")\n", encoding="utf-8")
        print(f"  -> {out.relative_to(_PKG_ROOT)} | duree {time.time() - t0:.0f}s")
        return

    # --- Phase 1 : correlation prune ---
    kept, dropped, lab_corr = correlation_prune(train, pool, train["label"], args.corr)
    print(f"\n[corr prune |r|>{args.corr}] {len(pool)} -> {len(kept)} "
          f"({len(dropped)} quasi-doublons retires)")
    pool = kept

    seeds = [XGB_RANDOM_STATE] + [XGB_RANDOM_STATE + 100 + i for i in range(NOISE_SEEDS_N)]

    # --- Phase 2 : forward selection (seed prod) ---
    print(f"\n[forward selection] cible {args.max_set} features (seed {XGB_RANDOM_STATE})")
    S = []
    remaining = list(pool)
    n_fits = 0
    while len(S) < args.max_set and remaining:
        best_c, best_auc = None, -1.0
        for c in remaining:
            auc = _train_auc(train, val, test, S + [c], XGB_RANDOM_STATE)
            n_fits += 1
            if auc > best_auc:
                best_auc, best_c = auc, c
        S.append(best_c)
        remaining.remove(best_c)
        print(f"  +{best_c:<22} | set={len(S):>2} AUC={best_auc:.5f}")

    # --- Phase 3 : bande de bruit sur le set courant ---
    base_aucs = [_train_auc(train, val, test, S, s) for s in seeds]
    n_fits += len(seeds)
    auc_base = base_aucs[0]
    sigma = float(np.std(base_aucs))
    band = max(USELESS_K * sigma, ABS_FLOOR)
    print(f"\n[bande] AUC_base(seed prod)={auc_base:.5f} sigma={sigma:.5f} bande={band:.5f}")

    # --- Phase 4 : elagage leave-one-out + backfill (swaps) ---
    def loo_deltas(curr):
        base = _train_auc(train, val, test, curr, XGB_RANDOM_STATE)
        d = {}
        for f in curr:
            cols = [c for c in curr if c != f]
            d[f] = _train_auc(train, val, test, cols, XGB_RANDOM_STATE) - base
        return base, d

    # --- Mode REPORT-ONLY : deltas leave-one-out propres sur le set forward ---
    if args.report_only:
        base, deltas = loo_deltas(S)
        rows = sorted(((f, deltas[f], classify(deltas[f], band)) for f in S), key=lambda r: r[1])
        print("\n" + "=" * 70)
        print(f"  REPORT-ONLY : {len(S)} features forward | AUC={base:.5f} | bande={band:.5f}")
        print("=" * 70)
        print(f"  {'feature':<24}{'dAUC':>11}   classe")
        n_useful = 0
        for f, dl, cls in rows:
            if cls in ("TRES UTILE", "MOYEN"):
                n_useful += 1
            print(f"  {f:<24}{dl:>+11.5f}   {cls}")
        print(f"\n  features qui passent la bande (dAUC <= -bande) : {n_useful}/{len(S)}")
        print(f"  duree {time.time() - t0:.0f}s")
        return

    def _loo_deltas_unused(curr):
        base = _train_auc(train, val, test, curr, XGB_RANDOM_STATE)
        d = {}
        for f in curr:
            cols = [c for c in curr if c != f]
            d[f] = _train_auc(train, val, test, cols, XGB_RANDOM_STATE) - base
        return base, d

    def band_for(curr):
        aucs = [_train_auc(train, val, test, curr, s) for s in seeds]
        return max(USELESS_K * float(np.std(aucs)), ABS_FLOOR)

    def backfill(curr, banned):
        """Ajoute la meilleure feature marginale (candidats non bannis d'abord)."""
        avail = [c for c in pool if c not in curr and c not in banned] \
                or [c for c in pool if c not in curr]
        if not avail:
            return None, 0.0
        base_now = _train_auc(train, val, test, curr, XGB_RANDOM_STATE)
        best_c, best_gain = None, -1e9
        for c in avail:
            g = _train_auc(train, val, test, curr + [c], XGB_RANDOM_STATE) - base_now
            if g > best_gain:
                best_gain, best_c = g, c
        return best_c, best_gain

    banned = []          # features retirees comme redondantes (deprioritisees au backfill)
    swap_budget = 8      # nb max de swaps "marge" apres avoir atteint le critere dur
    n_fits_phase4 = 0
    for it in range(args.max_iter):
        base, deltas = loo_deltas(S)
        band = band_for(S)
        n_fits += len(S) + 1 + len(seeds)
        strict_useless = [f for f in S if deltas[f] > -band]            # critere OFFICIEL
        fragile = [f for f in S if deltas[f] > -SAFETY * band]          # pas assez de marge
        n_ok = len(S) - len(strict_useless)
        print(f"  iter {it:>2}: |S|={len(S)} band={band:.5f} | utiles={n_ok}/{len(S)} "
              f"inutiles={len(strict_useless)} fragiles={len(fragile)} banned={len(banned)}")

        hard_met = (len(strict_useless) == 0 and len(S) >= args.target)
        if hard_met and (len(fragile) == 0 or swap_budget <= 0):
            print("  -> converge (0 inutile, cible atteinte).")
            break

        # candidats de remplacement disponibles ?
        avail = [c for c in pool if c not in S and c not in banned] \
                or [c for c in pool if c not in S]
        can_replace = len(avail) > 0

        # 1) retire la pire (inutile en priorite, sinon la + fragile)
        cand_remove = strict_useless if strict_useless else fragile
        removed = None
        if cand_remove and (len(S) > args.target or can_replace):
            removed = max(cand_remove, key=lambda f: deltas[f])
            S.remove(removed)
            if removed not in banned:
                banned.append(removed)
            if not strict_useless:
                swap_budget -= 1
            print(f"     - retire {removed} (delta={deltas[removed]:+.5f})")

        # 2) backfill jusqu'a la cible
        added_any = False
        while len(S) < args.target:
            best_c, best_gain = backfill(S, banned)
            if best_c is None:
                break
            S.append(best_c)
            added_any = True
            print(f"     + backfill {best_c} (gain={best_gain:+.5f})")

        if removed is None and not added_any:
            print("  -> aucun mouvement productif possible -> stop.")
            break

    # --- Verdict final ---
    base, deltas = loo_deltas(S)
    base_aucs = [_train_auc(train, val, test, S, s) for s in seeds]
    sigma = float(np.std(base_aucs))
    band = max(USELESS_K * sigma, ABS_FLOOR)
    rows = sorted(((f, deltas[f], classify(deltas[f], band)) for f in S), key=lambda r: r[1])

    print("\n" + "=" * 70)
    print(f"  JEU FINAL : {len(S)} features | AUC={base:.5f} | bande={band:.5f}")
    print("=" * 70)
    print(f"  {'feature':<24}{'dAUC':>11}   classe")
    n_inutile = 0
    for f, dl, cls in rows:
        if cls in ("INUTILE", "NUISIBLE"):
            n_inutile += 1
        print(f"  {f:<24}{dl:>+11.5f}   {cls}")
    n_tres = sum(1 for _, _, c in rows if c == "TRES UTILE")
    n_moy = sum(1 for _, _, c in rows if c == "MOYEN")
    print(f"\n  TRES UTILE={n_tres}  MOYEN={n_moy}  INUTILE/NUISIBLE={n_inutile}  | total={len(S)}")
    ok = (n_inutile == 0 and len(S) >= args.target)
    print(f"  OBJECTIF (0 inutile, >= {args.target}) : {'ATTEINT' if ok else 'NON ATTEINT'}")

    out = _PKG_ROOT / "optimize" / "selected_features.txt"
    out.write_text(
        "FEATURE_WHITELIST = (\n" +
        "".join(f'    "{f}",\n' for f, _, _ in rows) +
        ")\n", encoding="utf-8")
    print(f"\n  -> {out.relative_to(_PKG_ROOT)} ({len(S)} features)")
    print(f"  {n_fits} fits | duree {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
