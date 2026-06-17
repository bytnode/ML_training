"""
best_combo.py — recherche de la MEILLEURE COMBINAISON de features (parmi les 68
classees POSITIF par analyse/classify_features3.py) qui maximise l'AUC de
modele_up.

Probleme : selection de sous-ensemble de features. L'espace est 2^68 (~3e20),
intraitable en force brute. On combine 3 heuristiques complementaires, puis on
garde le meilleur sous-ensemble trouve :

  1. FORWARD SELECTION (greedy)
        Part du vide, ajoute a chaque pas la feature qui maximise l'AUC val.
        S'arrete quand plus aucun ajout n'ameliore (patience) ou a MAX_SIZE.
        Donne un bon point de depart + un ordre d'importance conditionnel.

  2. ALGORITHME GENETIQUE
        Population de sous-ensembles (chromosome = masque binaire sur le pool).
        Fitness = AUC_val - PARSIMONY * n_features (penalise legerement la
        taille pour casser les egalites vers le plus petit set). Selection par
        tournoi, croisement uniforme (chaque feature heritee d'un parent),
        mutation = flip de quelques bits, elitisme. Seede avec le resultat
        forward + le set deploye + des sous-ensembles aleatoires. Explore des
        combinaisons que le greedy rate (optima locaux).

  3. POLISSAGE LOCAL (SFFS-like)
        Depuis le meilleur sous-ensemble courant, tente tout add/remove unitaire
        tant que ca ameliore -> optimum local strict.

METHODO ANTI-OVERFIT :
  - La SELECTION se fait sur l'AUC de VALIDATION (jamais le test). Le test reste
    un hold-out : on ne rapporte son AUC qu'a la fin (chiffre honnete).
  - Mesure deterministe (subsample=colsample=1.0 => sigma~0) pour comparer les
    sous-ensembles sans bruit stochastique.
  - Le set gagnant est RE-EVALUE sous la config de PROD (subsample=0.8) moyennee
    sur plusieurs seeds, et compare au set deploye (FEATURE_WHITELIST actuel).
  - Cache (frozenset -> AUC) + budget d'evaluations borne pour le cout.

Le script NE MODIFIE RIEN (ni config, ni modele, ni parquet) : il ecrit un
rapport optimize/best_combo_report.md et imprime le verdict.

Usage (depuis la racine du package) :
    cd backtest/modele_updown/30min/modele_up
    python optimize/best_combo.py
    python optimize/best_combo.py --budget 800 --max-size 22 --ga-gens 15
    python optimize/best_combo.py --select-on test   # max AUC test (overfit, deconseille)
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

# Config de MESURE deterministe pour la recherche (annule le bruit -> sigma~0,
# comparaisons de sous-ensembles fiables). Identique a classify_features3.
STABLE_OVERRIDES = {"subsample": 1.0, "colsample_bytree": 1.0}
# Config de PROD (ce qui est reellement deploye) pour la confirmation finale.
PROD_OVERRIDES = {}  # = build_hyperparams() par defaut (subsample=0.8, colsample=0.8)

PARSIMONY = 2e-5        # penalite d'AUC par feature (casse les egalites vers + petit)
SELECT_SEEDS = (XGB_RANDOM_STATE,)         # seeds pour la selection (deterministe -> 1 suffit)
CONFIRM_SEEDS = (42, 7, 123)               # seeds pour la confirmation prod (moyenne)

SELECT_ON = "val"  # surcharge par CLI


# ----------------------------------------------------------------------------
#  Donnees / split (identique a classify_features3 / train)
# ----------------------------------------------------------------------------
def _split(df):
    df = df.sort_values("timestamp").reset_index(drop=True)
    ms = 86_400_000
    ts_max = df["timestamp"].max()
    test_cut = ts_max - TEST_DAYS * ms
    test_start = int(df["timestamp"].searchsorted(test_cut, side="left"))
    tv_start = 0 if TRAIN_DAYS is None else int(
        df["timestamp"].searchsorted(test_cut - TRAIN_DAYS * ms, side="left"))
    val_size = int((test_start - tv_start) * VAL_FRACTION)
    return (df.iloc[tv_start:test_start - val_size],
            df.iloc[test_start - val_size:test_start],
            df.iloc[test_start:])


def _read_positif_pool(md_path: Path) -> list[str]:
    """Lit analyse/feature_classes3.md et renvoie les features de classe POSITIF.
    Table markdown : | feature | delta_ablation | standalone_auc | power | classe |"""
    if not md_path.exists():
        raise FileNotFoundError(
            f"{md_path} introuvable. Lance d'abord "
            f"'python analyse/classify_features3.py --features all'.")
    pool = []
    for line in md_path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| "):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != 5 or cells[0] in ("feature", "---", ":---:"):
            continue
        feat, classe = cells[0], cells[4]
        if classe == "POSITIF":
            pool.append(feat)
    return pool


# ----------------------------------------------------------------------------
#  Evaluateur (avec cache + compteur de budget)
# ----------------------------------------------------------------------------
class Evaluator:
    def __init__(self, train, val, test, overrides, tag):
        self.train, self.val, self.test = train, val, test
        self.overrides, self.tag = overrides, tag
        self.cache: dict = {}
        self.n_evals = 0

    def _train_once(self, cols, seed):
        hp = build_hyperparams(self.overrides)
        optional = {}
        for k in ("reg_lambda", "reg_alpha", "gamma"):
            if hp[k] is not False:
                optional[k] = hp[k]
        spw = _resolve_scale_pos_weight(hp["scale_pos_weight"], self.train["label"])
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
        model.fit(self.train[cols], self.train["label"],
                  eval_set=[(self.val[cols], self.val["label"])], verbose=False)
        va = roc_auc_score(self.val["label"], model.predict_proba(self.val[cols])[:, 1])
        te = roc_auc_score(self.test["label"], model.predict_proba(self.test[cols])[:, 1])
        return float(va), float(te)

    def auc(self, cols, seeds=SELECT_SEEDS):
        """Renvoie (val_auc, test_auc) moyennes sur `seeds`. Cache par frozenset+tag."""
        key = (frozenset(cols), self.tag, seeds)
        if key in self.cache:
            return self.cache[key]
        if not cols:
            res = (0.5, 0.5)
        else:
            vs, ts = [], []
            for s in seeds:
                va, te = self._train_once(list(cols), s)
                vs.append(va); ts.append(te)
                self.n_evals += 1
            res = (float(np.mean(vs)), float(np.mean(ts)))
        self.cache[key] = res
        return res

    def fitness(self, cols):
        """Score de selection = AUC_val - parsimonie (ou AUC_test si --select-on test)."""
        va, te = self.auc(cols)
        base = te if SELECT_ON == "test" else va
        return base - PARSIMONY * len(cols), va, te


# ----------------------------------------------------------------------------
#  1) Forward selection greedy
# ----------------------------------------------------------------------------
def forward_selection(ev: Evaluator, pool, max_size, patience, budget):
    chosen, best_fit = [], -1.0
    history = []
    stale = 0
    while len(chosen) < max_size and ev.n_evals < budget:
        cand_best, cand_feat = None, None
        for f in pool:
            if f in chosen:
                continue
            fit, va, te = ev.fitness(chosen + [f])
            if cand_best is None or fit > cand_best[0]:
                cand_best, cand_feat = (fit, va, te), f
            if ev.n_evals >= budget:
                break
        if cand_feat is None:
            break
        if cand_best[0] > best_fit + 1e-9:
            best_fit = cand_best[0]
            chosen.append(cand_feat)
            history.append((len(chosen), cand_feat, cand_best[1], cand_best[2]))
            stale = 0
            print(f"  [forward] +{cand_feat:<22} n={len(chosen):>2} "
                  f"val={cand_best[1]:.5f} test={cand_best[2]:.5f}", flush=True)
        else:
            stale += 1
            print(f"  [forward] plateau ({stale}/{patience}) -> stop a n={len(chosen)}",
                  flush=True)
            if stale >= patience:
                break
            break
    return chosen, history


# ----------------------------------------------------------------------------
#  2) Algorithme genetique
# ----------------------------------------------------------------------------
def _rand_subset(rng, pool, lo, hi):
    k = int(rng.integers(lo, hi + 1))
    return set(rng.choice(pool, size=min(k, len(pool)), replace=False).tolist())


def genetic(ev: Evaluator, pool, seeds_init, budget, pop_size, gens, max_size, rng):
    pool = list(pool)
    pop = [set(s) for s in seeds_init if s]
    while len(pop) < pop_size:
        pop.append(_rand_subset(rng, pool, 4, max_size))

    def score(ind):
        fit, va, te = ev.fitness(sorted(ind))
        return fit, va, te

    scored = [(score(ind), ind) for ind in pop]
    best = max(scored, key=lambda x: x[0][0])
    print(f"  [GA] init best val={best[0][1]:.5f} test={best[0][2]:.5f} "
          f"n={len(best[1])} (evals={ev.n_evals})", flush=True)

    for g in range(gens):
        if ev.n_evals >= budget:
            print(f"  [GA] budget atteint, stop gen {g}", flush=True)
            break
        scored.sort(key=lambda x: x[0][0], reverse=True)
        elite = [ind for _, ind in scored[:max(2, pop_size // 5)]]
        new_pop = [set(e) for e in elite]
        while len(new_pop) < pop_size and ev.n_evals < budget:
            def tourn():
                cands = [scored[int(rng.integers(0, len(scored)))] for _ in range(3)]
                return max(cands, key=lambda x: x[0][0])[1]
            p1, p2 = tourn(), tourn()
            union = list(p1 | p2)
            child = set()
            for f in union:
                in1, in2 = f in p1, f in p2
                if in1 and in2:
                    keep = True
                elif in1 or in2:
                    keep = rng.random() < 0.5
                else:
                    keep = False
                if keep:
                    child.add(f)
            for f in pool:
                if rng.random() < 1.5 / len(pool):
                    child.symmetric_difference_update({f})
            if len(child) < 3:
                child |= _rand_subset(rng, pool, 3, 5)
            if len(child) > max_size:
                child = set(rng.choice(sorted(child), size=max_size, replace=False).tolist())
            new_pop.append(child)
        scored = [(score(ind), ind) for ind in new_pop]
        gbest = max(scored, key=lambda x: x[0][0])
        if gbest[0][0] > best[0][0]:
            best = gbest
        print(f"  [GA] gen {g + 1}/{gens} best val={best[0][1]:.5f} "
              f"test={best[0][2]:.5f} n={len(best[1])} (evals={ev.n_evals})", flush=True)
    return sorted(best[1]), best[0]


# ----------------------------------------------------------------------------
#  3) Polissage local (add/remove unitaire = SFFS strict)
# ----------------------------------------------------------------------------
def local_polish(ev: Evaluator, subset, pool, budget):
    cur = set(subset)
    cur_fit, cur_va, cur_te = ev.fitness(sorted(cur))
    improved = True
    while improved and ev.n_evals < budget:
        improved = False
        for f in sorted(cur):
            fit, va, te = ev.fitness(sorted(cur - {f}))
            if fit > cur_fit + 1e-9:
                cur.discard(f); cur_fit, cur_va, cur_te = fit, va, te
                improved = True
                print(f"  [polish] -{f:<22} val={va:.5f} test={te:.5f} n={len(cur)}",
                      flush=True)
                break
            if ev.n_evals >= budget:
                break
        if improved:
            continue
        for f in pool:
            if f in cur:
                continue
            fit, va, te = ev.fitness(sorted(cur | {f}))
            if fit > cur_fit + 1e-9:
                cur.add(f); cur_fit, cur_va, cur_te = fit, va, te
                improved = True
                print(f"  [polish] +{f:<22} val={va:.5f} test={te:.5f} n={len(cur)}",
                      flush=True)
                break
            if ev.n_evals >= budget:
                break
    return sorted(cur), (cur_fit, cur_va, cur_te)


# ----------------------------------------------------------------------------
#  Main
# ----------------------------------------------------------------------------
def main():
    global SELECT_ON
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=700, help="budget total d'entrainements")
    ap.add_argument("--max-size", type=int, default=20, help="taille max du sous-ensemble")
    ap.add_argument("--forward-patience", type=int, default=2)
    ap.add_argument("--ga-pop", type=int, default=24)
    ap.add_argument("--ga-gens", type=int, default=12)
    ap.add_argument("--select-on", choices=["val", "test"], default="val",
                    help="metrique de selection (val = anti-overfit ; test = overfit)")
    ap.add_argument("--seed", type=int, default=12345, help="seed du GA (stochastique)")
    args = ap.parse_args()
    SELECT_ON = args.select_on

    t0 = time.time()
    print("=" * 78)
    print("  best_combo.py — meilleure combinaison de features (modele_up)")
    print("=" * 78, flush=True)

    df = build_training_dataframe(DB_PATH).sort_values("timestamp").reset_index(drop=True)
    train, val, test = _split(df)
    print(f"Split : {len(train):,} train / {len(val):,} val / {len(test):,} test "
          f"| label_test={test['label'].mean():.4f}")

    pool = _read_positif_pool(_PKG_ROOT / "analyse" / "feature_classes3.md")
    pool = [f for f in pool if f in df.columns]
    print(f"Pool POSITIF : {len(pool)} features "
          f"| selection sur AUC_{'TEST' if SELECT_ON == 'test' else 'VAL'} "
          f"| mesure deterministe {STABLE_OVERRIDES}")
    print(f"Budget = {args.budget} entrainements | max_size = {args.max_size}\n", flush=True)

    ev = Evaluator(train, val, test, STABLE_OVERRIDES, tag="stable")

    base11 = [f for f in FEATURE_WHITELIST if f in df.columns]
    full_fit, full_va, full_te = ev.fitness(pool)
    b11_fit, b11_va, b11_te = ev.fitness(base11)
    print(f"[ref] pool complet ({len(pool)})  val={full_va:.5f} test={full_te:.5f}")
    print(f"[ref] set deploye ({len(base11)})  val={b11_va:.5f} test={b11_te:.5f}\n", flush=True)

    print("--- Phase 1 : forward selection ---", flush=True)
    fwd, _ = forward_selection(ev, pool, args.max_size, args.forward_patience,
                               budget=int(args.budget * 0.5))
    fwd_fit, fwd_va, fwd_te = ev.fitness(fwd)
    print(f"  forward final : n={len(fwd)} val={fwd_va:.5f} test={fwd_te:.5f}\n", flush=True)

    print("--- Phase 2 : algorithme genetique ---", flush=True)
    rng = np.random.default_rng(args.seed)
    seeds_init = [set(fwd), set(base11), set(fwd) | set(base11)]
    ga_best, (ga_fit, ga_va, ga_te) = genetic(
        ev, pool, seeds_init, budget=int(args.budget * 0.85),
        pop_size=args.ga_pop, gens=args.ga_gens, max_size=args.max_size, rng=rng)
    print(f"  GA final : n={len(ga_best)} val={ga_va:.5f} test={ga_te:.5f}\n", flush=True)

    print("--- Phase 3 : polissage local ---", flush=True)
    start = ga_best if ga_fit >= fwd_fit else fwd
    polished, (pol_fit, pol_va, pol_te) = local_polish(ev, start, pool, budget=args.budget)
    print(f"  polish final : n={len(polished)} val={pol_va:.5f} test={pol_te:.5f}\n", flush=True)

    candidates = {
        "forward": (fwd, fwd_va, fwd_te, fwd_fit),
        "genetic": (ga_best, ga_va, ga_te, ga_fit),
        "polish":  (polished, pol_va, pol_te, pol_fit),
    }
    best_name = max(candidates, key=lambda k: candidates[k][3])
    best_cols, best_va, best_te, _ = candidates[best_name]

    print("--- Confirmation config PROD (subsample=0.8, moyenne 3 seeds) ---", flush=True)
    ev_prod = Evaluator(train, val, test, PROD_OVERRIDES, tag="prod")
    prod_best_va, prod_best_te = ev_prod.auc(best_cols, seeds=CONFIRM_SEEDS)
    prod_b11_va, prod_b11_te = ev_prod.auc(base11, seeds=CONFIRM_SEEDS)
    print(f"  best ({best_name}, n={len(best_cols)}) : "
          f"PROD val={prod_best_va:.5f} test={prod_best_te:.5f}")
    print(f"  set deploye (n={len(base11)})           : "
          f"PROD val={prod_b11_va:.5f} test={prod_b11_te:.5f}", flush=True)

    dt = time.time() - t0
    delta_test = prod_best_te - prod_b11_te

    print("\n" + "=" * 78)
    print("  VERDICT")
    print("=" * 78)
    print(f"  Meilleur sous-ensemble ({best_name}) : {len(best_cols)} features")
    print(f"    {best_cols}")
    print(f"  AUC (mesure stable)  : val={best_va:.5f}  test={best_te:.5f}")
    print(f"  AUC (config PROD)    : val={prod_best_va:.5f}  test={prod_best_te:.5f}")
    print(f"  vs set deploye (11)  : test PROD {prod_b11_te:.5f} -> delta = {delta_test:+.5f}")
    print(f"  Total entrainements  : {ev.n_evals + ev_prod.n_evals} | duree {dt:.0f}s")
    verdict = ("AMELIORATION" if delta_test > 1e-4 else
               "EGAL" if delta_test > -1e-4 else "PAS MIEUX")
    print(f"  >> {verdict} vs le set deploye actuel.", flush=True)

    out = _PKG_ROOT / "optimize" / "best_combo_report.md"
    lines = [
        "# best_combo — meilleure combinaison de features (modele_up)", "",
        f"- Pool analyse : **{len(pool)}** features POSITIF",
        f"- Selection sur : **AUC_{'TEST' if SELECT_ON == 'test' else 'VAL'}** "
        f"| mesure deterministe {STABLE_OVERRIDES}",
        f"- Budget : {ev.n_evals} entrainements de recherche + "
        f"{ev_prod.n_evals} de confirmation | duree {dt:.0f}s",
        f"- Split : {len(train):,} train / {len(val):,} val / {len(test):,} test", "",
        "## Verdict", "",
        f"- **Meilleur set ({best_name})** : **{len(best_cols)} features**",
        f"- AUC stable : val **{best_va:.5f}** / test **{best_te:.5f}**",
        f"- AUC config PROD (moy 3 seeds) : val **{prod_best_va:.5f}** / test **{prod_best_te:.5f}**",
        f"- Set deploye (11) en PROD : test **{prod_b11_te:.5f}** "
        f"-> **delta = {delta_test:+.5f}** ({verdict})", "",
        "### Features retenues", "",
        "```python",
        "FEATURE_WHITELIST = (",
    ]
    for i in range(0, len(best_cols), 5):
        lines.append("    " + ", ".join(f'"{c}"' for c in best_cols[i:i + 5]) + ",")
    lines += [")", "```", "",
              "## Candidats par strategie", "",
              "| strategie | n | AUC val (stable) | AUC test (stable) |",
              "|---|---:|---:|---:|"]
    for name, (cols, va, te, _) in candidates.items():
        mark = " **(retenu)**" if name == best_name else ""
        lines.append(f"| {name}{mark} | {len(cols)} | {va:.5f} | {te:.5f} |")
    lines += ["", f"| pool complet | {len(pool)} | {full_va:.5f} | {full_te:.5f} |",
              f"| set deploye | {len(base11)} | {b11_va:.5f} | {b11_te:.5f} |", ""]
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  -> rapport ecrit : {out.relative_to(_PKG_ROOT)}")
    print(f"  (le script NE MODIFIE PAS config.py — a toi de deployer le set si OK)", flush=True)


if __name__ == "__main__":
    main()
