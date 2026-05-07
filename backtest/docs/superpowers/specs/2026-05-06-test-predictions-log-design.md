# Spec : log des prédictions sur le test set

**Date** : 2026-05-06
**Statut** : approuvé

## Problème

Le script `train.py` affiche des résumés agrégés (WR par seuil, breakdown par `seq_len`, top-10 importance), mais ne permet pas d'inspecter le détail des paris individuels. Pour chaque setup du test set, on ne sait pas quelle confiance le modèle lui a attribuée, ni si ce setup s'est avéré gagnant ou perdant.

L'objectif est de produire un fichier listant **chaque setup évalué pendant la phase de test, avec son score de confiance**, pour permettre une analyse manuelle (filtrage, tri, croisement avec d'autres données).

## Décisions

1. **Périmètre** : tous les rows du test set (pas seulement les paris "pris" au seuil) — l'utilisateur pourra filtrer par `confidence` après coup.
2. **Stratégie de fichier** : fichier unique écrasé à chaque run. Pas d'historique multi-run (c'est `logs/runs.csv` qui sert à ça via les métriques agrégées).
3. **Active uniquement via `train.py` CLI** : `sweep.py` ne doit pas générer ce fichier (sinon on aurait juste le dernier run du sweep, ce qui n'a pas de sens). Implémenté via un paramètre `log_predictions: bool = False` à `train_and_evaluate(...)`, mis à `True` uniquement dans le CLI `main()` de `train.py`.

## Spécification

### Fichier produit

`logs/test_predictions.csv`, écrasé à chaque appel de `python train.py`.

### Contenu

Une row = un setup évalué dans la phase de test (= une row du dataframe `test`, déjà filtré par `SEQ_TEST` si actif).

Colonnes :

- **Toutes les colonnes du test set telles quelles** : `timestamp` + 22 features 15m + 20 features MTF + `label` (44 colonnes).
- **Colonne ajoutée** : `confidence` = `model.predict_proba(X_test)[:, 1]` (proba de classe 1 = retournement).

Total : **45 colonnes**. Ordre : colonnes du test_df dans leur ordre original, puis `confidence` à la fin.

### API

Signature étendue de `train_and_evaluate` :

```python
def train_and_evaluate(
    overrides: dict | None = None,
    save_model: bool = True,
    verbose: bool = True,
    log_run: bool = True,
    run_label: str = "manual",
    log_predictions: bool = False,   # <- NOUVEAU
) -> dict:
```

Défaut `False` pour ne pas casser les appelants existants (`sweep.py` notamment).

### CLI

`train.py` CLI passe `log_predictions=True` à `train_and_evaluate`. Aucun nouveau flag CLI n'est nécessaire (le comportement est implicite : lancer `train.py` génère le fichier).

### Implémentation

Nouvelle fonction privée dans `train.py` :

```python
def _write_test_predictions(test_df: pd.DataFrame, y_pred: np.ndarray, output_path: Path) -> None:
    """Ecrit le test set + colonne confidence dans output_path (ecrase)."""
    out = test_df.copy()
    out["confidence"] = y_pred
    out.to_csv(output_path, index=False)
```

Appelée dans `train_and_evaluate` juste après le calcul de `y_pred = model.predict_proba(X_test)[:, 1]` (ligne 188 actuelle), conditionnée par `if log_predictions:`.

Le chemin est constant : `LOGS_DIR / "test_predictions.csv"`.

### Edge cases

- **Test set vide** (cas `SEQ_TEST` filtre tout) : `test_df.copy()` produit un dataframe vide avec les colonnes correctes ; ajouter une colonne `confidence` à un dataframe vide fonctionne ; `to_csv` écrit le header seul. Pas de plantage.
- **Verbose** : le CLI imprime un message confirmant l'écriture, comme pour `MODEL_OUTPUT` et `RUNS_CSV` (`f"Test predictions written to {path}"`).

## Hors scope

- Pas de filtrage par seuil dans le fichier — `confidence` est brute, l'utilisateur filtre après.
- Pas de timestamp ou run_label dans le nom du fichier (pas d'historique).
- Pas d'activation pour `sweep.py`.
- Pas de format JSON ou autre — CSV uniquement, pour cohérence avec `runs.csv` et `training_data.csv`.

## Fichiers touchés

- `train.py` : ajout de `_write_test_predictions`, signature de `train_and_evaluate`, appel dans le CLI.
- `CLAUDE.md` (racine ET `ML_training-main/`) : ajouter une mention du fichier produit dans la section "Pipeline" / "Commands".
