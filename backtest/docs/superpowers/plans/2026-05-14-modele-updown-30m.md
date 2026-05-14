# modele_updown 30m — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cloner `backtest/modele_updown/15min/` en `backtest/modele_updown/30m/` pour produire une équipe modele_updown qui prédit la prochaine bougie 30 minutes (au lieu de 15). Périmètre : modèles + team backtest. Pas de live.

**Architecture:** Deux paquets sœurs (`30m/modele_up/` + `30m/modele_down/`) qui exposent le contrat `evaluate_on_slice` standardisé, orchestrés par `30m/main.py` qui applique la coherence rule. Le seul vrai pipeline data vit dans `30m/modele_down/data.py` ; le parquet de `30m/modele_up/` est dérivé du sien avec le label flippé (mécanisme identique au 15min).

**Tech Stack:** Python 3.10+ (testé 3.14), pandas, numpy, xgboost, scikit-learn, scipy, rich. Pas de framework de test ni de linter — vérification = inspecter `logs/runs.csv` et `logs/team_runs.csv`.

**Spec:** `backtest/docs/superpowers/specs/2026-05-14-modele-updown-30m-design.md`.

---

## File structure

**Created files (32 fichiers Python/MD + leurs sidecar+model JSON + parquet/DB générés à l'exécution)** :

```
backtest/modele_updown/30m/
  modele_up/
    CLAUDE.md
    api.py
    config.py
    features.py
    predict.py
    train.py
    analyse/
      analyze_features.py
      check_new_features.py
    DB/                                     (créé par train.py au runtime)
    logs/                                   (créé par train.py au runtime)
  modele_down/
    CLAUDE.md
    api.py
    config.py
    data.py
    features.py
    predict.py
    train.py
    analyse/
      00_synthese_finale.md
      01_features_diff.py + .md
      02_train_variants.py + .md
      03_feature_ablation.py + .md
      04_hyperparam_sweep.py + .md
      05_final_combo.py + .md
      06_leave_one_out_wr.py + .md
      07_pourquoi_compression.py + .md
      08_cascade_check.py + .md
      analyze_features.py
      check_new_features.py
      compare_runs.py
    DB/                                     (créé par data.py au runtime)
    logs/                                   (créé par train.py au runtime)
  main.py
  logs/                                     (créé par main.py au runtime)
```

**Untouched files** : `backtest/modele_updown/15min/**` (zéro modification), `backtest/modele_updown/main.py`, `backtest/modele_updown/live/**`.

---

### Task 1: Bootstrap — directory tree + verbatim copy of `30m/modele_down/`

**Files:**
- Create: `backtest/modele_updown/30m/modele_down/` (full tree, copy from `15min/modele_down/`)

- [ ] **Step 1: Create the empty directory tree**

```powershell
cd C:\Users\fremo\Desktop\ML_agent_poly\backtest\modele_updown\30m
New-Item -ItemType Directory -Force -Path modele_down\analyse | Out-Null
New-Item -ItemType Directory -Force -Path modele_up\analyse  | Out-Null
```

- [ ] **Step 2: Copy `modele_down/` tracked files verbatim from `15min/modele_down/`**

Use git ls-tree to list exactly what's tracked, then copy each file. (Avoid `Copy-Item -Recurse` on the whole folder so we don't pick up gitignored artifacts like `DB/`, `logs/strategy_lab/`, `__pycache__/`.)

```powershell
cd C:\Users\fremo\Desktop\ML_agent_poly
$src = "backtest\modele_updown\15min\modele_down"
$dst = "backtest\modele_updown\30m\modele_down"
$files = git ls-tree -r --name-only HEAD -- $src
foreach ($f in $files) {
    $rel = $f.Substring($src.Length + 1)  # strip "backtest/.../modele_down/"
    $target = Join-Path $dst $rel
    New-Item -ItemType Directory -Force -Path (Split-Path $target -Parent) | Out-Null
    Copy-Item -Path $f -Destination $target -Force
}
```

- [ ] **Step 3: Verify the file count**

Run: `git -C C:\Users\fremo\Desktop\ML_agent_poly status backtest/modele_updown/30m/modele_down/`

Expected: 25 untracked files (1 CLAUDE.md + 6 `.py` à la racine + 18 fichiers dans `analyse/`). Quick check :
```powershell
(Get-ChildItem -Recurse -File backtest\modele_updown\30m\modele_down).Count
```
Expected output: `25`.

- [ ] **Step 4: Commit the bootstrap**

```powershell
git -C C:\Users\fremo\Desktop\ML_agent_poly add backtest/modele_updown/30m/modele_down/
git -C C:\Users\fremo\Desktop\ML_agent_poly commit -m "feat(30m): bootstrap modele_down clone (verbatim copy of 15min/)"
```

---

### Task 2: Adapter `30m/modele_down/config.py`

**Files:**
- Modify: `backtest/modele_updown/30m/modele_down/config.py` (lines 8 et 14 du clone)

- [ ] **Step 1: Open the file and locate the substitutions**

Lines to change (you copied the file verbatim, so these match the 15min content) :

```python
# Line 8 of the clone (was: INTERVAL = "15m")
INTERVAL = "15m"
# Line 9 of the clone (was: DB_PATH = "db/btc_multi_tf_15m.parquet")
DB_PATH = "db/btc_multi_tf_15m.parquet"
# Line 14 of the clone
MTF_INTERVALS = ["30m", "1h", "4h", "1d"]
```

- [ ] **Step 2: Apply the four substitutions**

Final state of the relevant block (exact code) :

```python
SYMBOL = "btcusd"
INTERVAL = "30m"
DAYS_BACK = 5000
DB_PATH = "db/btc_multi_tf_30m.parquet"  # parquet 30m lu par features.py + train.py

# --- Higher timeframes ajoutés à chaque ligne du dataset (no-lookahead via
# last_closed_index : OHLCV de la dernière bougie closed avant le timestamp
# de décision T = times[i] + 30m).
MTF_INTERVALS = ["1h", "4h", "1d"]

INTERVAL_MS = {
    "15m": 15 * 60 * 1000,
    "30m": 30 * 60 * 1000,
    "1h":  60 * 60 * 1000,
    "4h":  4 * 60 * 60 * 1000,
    "1d":  24 * 60 * 60 * 1000,
}
```

Note : on garde `15m` dans `INTERVAL_MS` parce qu'il sert maintenant de contexte fin (analogue au rôle du `5m` côté 15min).

- [ ] **Step 3: Verify the file parses**

Run: `python -c "import ast; ast.parse(open(r'backtest/modele_updown/30m/modele_down/config.py').read())"`
Expected: no output (parse OK).

- [ ] **Step 4: Commit**

```powershell
git -C C:\Users\fremo\Desktop\ML_agent_poly add backtest/modele_updown/30m/modele_down/config.py
git -C C:\Users\fremo\Desktop\ML_agent_poly commit -m "feat(30m/modele_down): config substitutions (INTERVAL=30m, DB=btc_multi_tf_30m, MTF=[1h,4h,1d])"
```

---

### Task 3: Adapter `30m/modele_down/data.py` (la pièce centrale)

**Files:**
- Modify: `backtest/modele_updown/30m/modele_down/data.py`

C'est le seul fichier où le clone demande plus qu'une substitution mécanique. Faire les changements dans l'ordre listé pour éviter les passes oubliées.

- [ ] **Step 1: Header docstring (lignes 1-50)**

Remplacer les phrases "modele_updown" → "modele_updown 30m" et toute mention "6 timeframes (5m, 15m, ..." → "5 timeframes (15m, 30m, 1h, 4h, 1d)". Mettre à jour la liste des features attendues dans la docstring :

```python
"""
data.py — pipeline complet pour modele_updown 30m.

Un seul fichier qui :
  1. Fetch OHLCV BTC/USD depuis Bitstamp sur 5 timeframes (15m, 30m, 1h, 4h, 1d)
  2. Calcule les 9 features normalisées par TF + persiste close & volume bruts
  3. Construit le master 30m (.parquet) avec UNIQUEMENT des features normalisées :
       - 9  features de base 30m
       - 9  features 15m (lookup no-lookahead, 15m close-aligned uniquement)
       - 18 features MTF (1h/4h/1d, déjà filtrées : on n'écrit que les colonnes
            qui survivent au drop — pas de calcul inutile)
       -  7 features Tier 1 dérivées : dist_ema_20, return_3, return_96,
            hour_sin, hour_cos, upper_wick, lower_wick
       -  9 features Tier 2  (NEW)   : macd, macd_signal, macd_hist,
            bb_position, bb_width, mfi_norm, rv_20, rv_100, rv_ratio
  ...
```

- [ ] **Step 2: `TIMEFRAMES` dict (lignes ~81-88) — drop 5m**

```python
TIMEFRAMES: dict[str, tuple[int, int]] = {
    "15m": (15 * 60 * 1000,          15 * 60),
    "30m": (30 * 60 * 1000,          30 * 60),
    "1h":  (60 * 60 * 1000,          60 * 60),
    "4h":  ( 4 * 60 * 60 * 1000,  4 * 60 * 60),
    "1d":  (24 * 60 * 60 * 1000, 24 * 60 * 60),
}
```

- [ ] **Step 3: `MTF_KEPT_COLS` (lignes ~134-152) — réorganiser**

Le `5m` devient `15m` (fine context), le `30m` disparaît (devient la base). Tableau final :

```python
MTF_KEPT_COLS: dict[str, list[str]] = {
    # 15m : 9 features (fine context, rôle joué par "5m" côté 15min)
    "15m": ["open_ret", "high_ret", "low_ret", "close_ret",
            "volume_ratio", "body_ratio_close", "body_ratio_avg",
            "obv_pressure", "atr_pct"],
    # 1h : 5 (héritage 15min audit)
    "1h":  ["open_ret", "low_ret", "close_ret",
            "volume_ratio", "body_ratio_close"],
    # 4h : 5 (héritage 15min audit)
    "4h":  ["close_ret", "volume_ratio",
            "body_ratio_avg", "obv_pressure", "atr_pct"],
    # 1d : 8 (héritage 15min audit)
    "1d":  ["open_ret", "high_ret", "low_ret",
            "volume_ratio", "body_ratio_close", "body_ratio_avg",
            "obv_pressure", "atr_pct"],
}
```

- [ ] **Step 4: Constantes d'intervalle (ligne ~116-117) — ajouter 30m, garder 15m**

```python
INTERVAL_15M_MS = 15 * 60 * 1000
INTERVAL_30M_MS = 30 * 60 * 1000
```

Supprimer `INTERVAL_5M_MS = 5 * 60 * 1000` (plus utilisé).

- [ ] **Step 5: `process_timeframe_to_candles` — cas spécial 5m devient cas 15m (lignes ~391-462)**

Le cas spécial filtrait les 5m close-aligned. Pour 30m base, on filtre les 15m close-aligned :

```python
def process_timeframe_to_candles(db_path: Path, timeframe: str) -> int:
    """Lit candles_raw du TF, calcule features, INSERT dans candles.

    Cas spécial 15m : on calcule les features sur la série COMPLÈTE (les rolling
    SMA20/EWMA14/etc. ont besoin de toutes les bougies contiguës), puis on FILTRE
    pour ne garder que la 15m close-aligned avec chaque 30m (open_time = T + 15min).
    On passe ainsi de ~2N rows à ~N rows en DB & CSV, sans dégrader les
    features (qui ont vu la série complète au moment du calcul).
    """
    # ... fetch/normalize block unchanged ...

    if timeframe == "15m":
        # Garde les 15m dont open_time tombe à T + 15min pour un T multiple de 30min.
        # Ce sont exactement les 15m qui ferment à T + 30min = close du 30m.
        offset_ms = feat["timestamp_ms"].to_numpy(dtype=np.int64) % INTERVAL_30M_MS
        keep_mask = offset_ms == (15 * 60 * 1000)
        n_before = len(feat)
        feat = feat.loc[keep_mask].reset_index(drop=True)
        print(f"    [15m] filtre close-align : {n_before:,} -> {len(feat):,} rows "
              f"(garde 1/2, les autres ne sont pas utilisées par le 30m)")
```

Tout le bloc `if timeframe == "5m"` est remplacé par ce `if timeframe == "15m"`.

- [ ] **Step 6: `build_master_15m` → `build_master_30m` (rename + corps adapté, lignes ~700-800)**

Renommer la fonction et adapter le corps :

```python
def build_master_30m(db_path: Path, out_path: Path) -> tuple[int, int]:
    """
    Construit le DataFrame final 30m (Tier 0 + 15m + MTF + Tier 1 + Tier 2)
    ...
    """
    print("\n--- Phase 4/5 : construction du master 30m DataFrame ---")
    t0 = time.time()

    # 4.1 — base 30m
    df_30m = _load_tf(db_path, "30m")
    print(f"  [base 30m] {len(df_30m):,} rows × {len(df_30m.columns)} cols (avec close+volume)")

    # 4.2 — 15m lookups (no-lookahead, fine context)
    df_15m = _load_tf(db_path, "15m")
    if not df_15m.empty:
        n_15m = _add_lookup_features(df_30m, df_15m, "15m",
                                     INTERVAL_15M_MS, MTF_KEPT_COLS["15m"])
        print(f"  [15m lookup] +{n_15m} cols")
    else:
        print(f"  [15m lookup] skip (table 15m vide)")

    # 4.3 — MTF lookups (1h, 4h, 1d) — colonnes filtrées post-drop
    for tf in ["1h", "4h", "1d"]:
        df_oth = _load_tf(db_path, tf)
        if df_oth.empty:
            print(f"  [{tf} lookup] skip (table vide)")
            continue
        step_ms, _ = TIMEFRAMES[tf]
        n = _add_lookup_features(df_30m, df_oth, tf, step_ms, MTF_KEPT_COLS[tf])
        print(f"  [{tf} lookup] +{n} cols (post-drop)")

    # 4.4 — Tier 1 derived (7 cols)  -- unchanged, opère sur df_30m
    n_t1 = _add_tier1_derived(df_30m)
    print(f"  [Tier 1 derived] +{n_t1} cols (dist_ema_20, return_3/96, hour_sin/cos, wicks)")

    # 4.5 — Tier 2 (9 cols)  -- unchanged
    n_t2 = _add_tier2(df_30m)
    print(f"  [Tier 2] +{n_t2} cols (MACD ×3, BB ×2, MFI ×1, rv ×3)")

    # 4.6 — derived columns is_weekend / volatility_state / label  -- unchanged but on df_30m
    ts_dt = pd.to_datetime(df_30m["timestamp_ms"], unit="ms", utc=True)
    df_30m["is_weekend"] = (ts_dt.dt.dayofweek >= 5).astype(int)

    atr_long = df_30m["atr_pct"].rolling(window=ATR_LONG_PERIOD, min_periods=20).mean()
    atr_ratio = df_30m["atr_pct"] / atr_long
    vol_state = np.where(atr_ratio > VOLATILITY_HIGH, 2,
                np.where(atr_ratio < VOLATILITY_LOW, 0, 1))
    vol_state = np.where(np.isnan(atr_ratio.to_numpy()), 1, vol_state)
    df_30m["volatility_state"] = vol_state.astype(int)

    cr = df_30m["close_ret"].to_numpy()
    or_ = df_30m["open_ret"].to_numpy()
    cr_next = np.concatenate([cr[1:], [np.nan]])
    or_next = np.concatenate([or_[1:], [np.nan]])
    invalid = (
        (cr == or_) | (cr_next == or_next)
        | np.isnan(cr) | np.isnan(or_)
        | np.isnan(cr_next) | np.isnan(or_next)
    )
    label = np.where(invalid, np.nan, (cr_next < or_next).astype(float))
    df_30m["label"] = label
    n_valid_label = int(np.sum(~np.isnan(label)))
    print(f"  [derived] +3 cols : is_weekend, volatility_state, label "
          f"({n_valid_label:,} labels valides / {len(df_30m):,})")

    df_30m = df_30m.drop(columns=["close", "volume", "atr_pct", "open_ret"])

    suffix = out_path.suffix.lower()
    if suffix == ".parquet":
        df_30m.to_parquet(out_path, index=False)
    elif suffix == ".csv":
        df_30m.to_csv(out_path, index=False)
    else:
        raise ValueError(f"Extension non supportée pour {out_path} : '{suffix}'. "
                         f"Attendu : .parquet ou .csv.")
    elapsed = time.time() - t0
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"  -> {out_path.name} : {len(df_30m):,} rows x {len(df_30m.columns)} cols, "
          f"{size_mb:.1f} MB ({elapsed:.1f}s)")

    return len(df_30m), len(df_30m.columns)
```

- [ ] **Step 7: Caller de `build_master_15m` (ligne ~858) — rename + master path**

Dans `run()` :

```python
    # ---- Phase 4 — master 30m parquet (= dataset d'entrée de train.py) ----
    master_path = csv_dir / f"{db_stem}_30m.parquet"
    n_rows, n_cols = build_master_30m(db_path, master_path)
```

- [ ] **Step 8: Recap text (ligne ~810 et ~865-869) — adapter les chiffres**

Dans `run()` :

```python
    print(f"=== modele_updown 30m — pipeline data.py (Tier 0 + Tier 1 + Tier 2) ===")
    ...
    print(f"Master 30m (parquet)        : {master_path.resolve()}")
    print(f"  {n_rows:,} rows x {n_cols} cols (= 2 metadata + 9 base + 9 15m + 18 MTF "
          f"+ 7 Tier1 + 9 Tier2 ; close/volume bruts gardes en SQLite uniquement)")
```

Et dans le bloc `summary = pd.read_sql_query(...)` (ligne ~872-882), retirer le `5m` du `CASE` :

```python
        ORDER BY CASE timeframe
            WHEN '15m' THEN 1 WHEN '30m' THEN 2
            WHEN '1h'  THEN 3 WHEN '4h'  THEN 4 WHEN '1d'  THEN 5 END
```

- [ ] **Step 9: argparse description (ligne ~897)**

```python
        description="modele_updown 30m — fetch + features (Tier 0 + 1 + 2) en un seul script."
```

- [ ] **Step 10: Verify the file parses**

Run: `python -c "import ast; ast.parse(open(r'backtest/modele_updown/30m/modele_down/data.py').read())"`
Expected: no output.

- [ ] **Step 11: Commit**

```powershell
git -C C:\Users\fremo\Desktop\ML_agent_poly add backtest/modele_updown/30m/modele_down/data.py
git -C C:\Users\fremo\Desktop\ML_agent_poly commit -m "feat(30m/modele_down): data.py adapt for 30m base + 15m fine context + MTF [1h,4h,1d]"
```

---

### Task 4: Adapter le `CLAUDE.md` du paquet `30m/modele_down/`

**Files:**
- Modify: `backtest/modele_updown/30m/modele_down/CLAUDE.md`

Le CLAUDE.md cloné décrit la pipeline legacy (data.py + patch_body_signed.py + fetch_and_add_5m.py + add_tier1_features.py + drop_redundant_features.py) qui n'existe plus dans le repo. Profiter du clone pour la nettoyer.

- [ ] **Step 1: Replace the "Build pipeline — order matters" section**

Remplacer tout le bloc entre `## Build pipeline — order matters` et le `## What \`train.py\` writes` qui suit, par :

```markdown
## Build pipeline

Le pipeline data est consolidé dans un seul script :

```powershell
cd backtest\modele_updown\30m\modele_down

# 1. Cold fetch + features (Tier 0 + 1 + 2) en un seul script.
#    Fetch 5 TFs (15m, 30m, 1h, 4h, 1d) depuis Bitstamp, calcule les 9 features
#    normalisées par TF, joint les MTF no-lookahead, ajoute les features Tier 1
#    et Tier 2, calcule label + is_weekend + volatility_state, écrit
#    DB/btc_multi_tf.db + DB/btc_multi_tf_30m.parquet. Takes ~3-5 min.
python data.py            # --pair btcusd --days 5000 --db DB/btc_multi_tf.db

# 2. Train + evaluate. Writes btc_updown_model.json + sidecar + logs/runs.csv
#    + logs/test_predictions.csv. Direct CLI runs always log per-row test
#    predictions (log_predictions=True is hardcoded in main()).
python train.py [--label name] [--no-save] [--no-log]

# Inspection tools
python predict.py             # smoke-test on the last test_predictions row
python api.py                 # exercise evaluate_on_slice on the full window
python analyse/analyze_features.py    # 7-section feature audit (writes to stdout/log)
```

`features.py` est le runtime feature builder utilisé par `train.py` — il lit le
parquet, drop le warmup, drop les rows sans label valide. Toutes les colonnes
dérivées (`label`, `volatility_state`, `is_weekend`) sont pré-calculées par
`data.py`. Pas de re-calcul on-the-fly côté training.
```

- [ ] **Step 2: Update the substitutions throughout the rest of the file**

Search & replace dans tout le fichier :
- `15m` → `30m` (sauf les références explicites au 15min en contexte fin, ex : "le 15m est ajouté comme contexte fin")
- `15min` → `30m`
- `modele_15m` → (référence historique, peut rester telle quelle dans les phrases qui regardent vers le passé)

Vérifier que les noms de fichiers cités (`db/btc_multi_tf_15m.csv` → `DB/btc_multi_tf_30m.parquet`, `db/btc_multi_tf.db` → `DB/btc_multi_tf.db`) sont cohérents avec le `config.py` adapté.

- [ ] **Step 3: Update the "Working-directory contract" section**

Le chemin `cd backtest/modele_updown/modele_down/` devient
`cd backtest/modele_updown/30m/modele_down/`.

- [ ] **Step 4: Commit**

```powershell
git -C C:\Users\fremo\Desktop\ML_agent_poly add backtest/modele_updown/30m/modele_down/CLAUDE.md
git -C C:\Users\fremo\Desktop\ML_agent_poly commit -m "feat(30m/modele_down): adapt CLAUDE.md (drop legacy patch chain, 30m paths)"
```

---

### Task 5: Run `data.py` end-to-end and verify the parquet

**Files:**
- Generated: `backtest/modele_updown/30m/modele_down/DB/btc_multi_tf.db`
- Generated: `backtest/modele_updown/30m/modele_down/DB/btc_multi_tf_30m.parquet`

- [ ] **Step 1: Run `python data.py` from the package root**

```powershell
cd C:\Users\fremo\Desktop\ML_agent_poly\backtest\modele_updown\30m\modele_down
python data.py
```

Expected: ~3-5 min runtime. Last printed line should look like :
```
=== Terminé. Lance maintenant : python train.py ===
```

If it fails on a Bitstamp HTTP error : retry the script (it's idempotent at the DB level — `init_db` drops existing tables and refetches).

- [ ] **Step 2: Verify the parquet schema**

```powershell
python -c "import pandas as pd; df = pd.read_parquet(r'DB/btc_multi_tf_30m.parquet'); print(df.shape); print(sorted(df.columns))"
```

Expected output:
- `df.shape` : something like `(240000, ~52)` (5000 days × 48 bars/day, minus warmup ~= 240k rows; ~52 cols = 1 timestamp_ms + 1 timestamp + 9 base + 9 `_15m` + 18 MTF (5+5+8) + 7 Tier1 + 9 Tier2 + 3 derived).
- Columns include `_15m` suffixes but **no** `_5m` or `_30m` suffixes (the 30m is the base, no suffix; the 5m was dropped).
- `label`, `volatility_state`, `is_weekend` present.

- [ ] **Step 3: Quick sanity check on the label distribution**

```powershell
python -c "import pandas as pd; df = pd.read_parquet(r'DB/btc_multi_tf_30m.parquet'); print('label mean (DOWN baseline):', df['label'].dropna().mean())"
```

Expected: ~0.49-0.50 (DOWN baseline on BTC 30m). If it's wildly different (e.g. 0.0 or 1.0), the label computation is broken — investigate before going further.

- [ ] **Step 4: Don't commit** — parquet and DB are gitignored.

---

### Task 6: Train `modele_down` 30m

**Files:**
- Generated: `backtest/modele_updown/30m/modele_down/btc_updown_model.json`
- Generated: `backtest/modele_updown/30m/modele_down/btc_updown_model.thresholds.json`
- Generated: `backtest/modele_updown/30m/modele_down/logs/runs.csv`
- Generated: `backtest/modele_updown/30m/modele_down/logs/test_predictions.csv`

- [ ] **Step 1: Run `python train.py` from the package root**

```powershell
cd C:\Users\fremo\Desktop\ML_agent_poly\backtest\modele_updown\30m\modele_down
python train.py
```

Expected: ~30 s à 2 min depending on n_estimators. Last lines should show the calibration summary (per-bucket deltas, adaptive_total_taken, adaptive_total_wr).

- [ ] **Step 2: Verify the artifacts exist**

```powershell
Test-Path .\btc_updown_model.json
Test-Path .\btc_updown_model.thresholds.json
Test-Path .\logs\runs.csv
Test-Path .\logs\test_predictions.csv
```

All four expected: `True`.

- [ ] **Step 3: Verify the run beats baseline**

```powershell
python -c "import pandas as pd; r = pd.read_csv(r'logs/runs.csv').iloc[-1]; print('baseline_wr_down:', r.get('baseline_wr_down')); print('adaptive_total_wr:', r.get('adaptive_total_wr')); print('adaptive_total_taken:', r.get('adaptive_total_taken'))"
```

Expected: `adaptive_total_wr > baseline_wr_down` AND `adaptive_total_taken > 0`. If the WR is below baseline or `taken == 0`, that's the bar — it could mean the model didn't learn anything useful on 30m. Note the values; don't block on this — the user wants to see the actual number.

- [ ] **Step 4: Don't commit** — artifacts are gitignored.

---

### Task 7: Bootstrap — verbatim copy of `30m/modele_up/`

**Files:**
- Create: `backtest/modele_updown/30m/modele_up/` (full tree from `15min/modele_up/`)

- [ ] **Step 1: Copy tracked files via git ls-tree**

```powershell
cd C:\Users\fremo\Desktop\ML_agent_poly
$src = "backtest\modele_updown\15min\modele_up"
$dst = "backtest\modele_updown\30m\modele_up"
$files = git ls-tree -r --name-only HEAD -- $src
foreach ($f in $files) {
    $rel = $f.Substring($src.Length + 1)
    $target = Join-Path $dst $rel
    New-Item -ItemType Directory -Force -Path (Split-Path $target -Parent) | Out-Null
    Copy-Item -Path $f -Destination $target -Force
}
```

- [ ] **Step 2: Verify file count**

```powershell
(Get-ChildItem -Recurse -File backtest\modele_updown\30m\modele_up).Count
```

Expected: depending on what `git ls-tree` returns — ~6-12 files. Run :
```powershell
git -C C:\Users\fremo\Desktop\ML_agent_poly ls-tree -r --name-only HEAD -- backtest/modele_updown/15min/modele_up/ | Measure-Object -Line
```
and compare the count to the count of files actually copied. They must match.

- [ ] **Step 3: Commit**

```powershell
git -C C:\Users\fremo\Desktop\ML_agent_poly add backtest/modele_updown/30m/modele_up/
git -C C:\Users\fremo\Desktop\ML_agent_poly commit -m "feat(30m): bootstrap modele_up clone (verbatim copy of 15min/)"
```

---

### Task 8: Adapter `30m/modele_up/config.py`

**Files:**
- Modify: `backtest/modele_updown/30m/modele_up/config.py`

- [ ] **Step 1: Apply the four substitutions**

Identical to Task 2 but in the modele_up directory. Replace :

```python
SYMBOL = "btcusd"
INTERVAL = "30m"
DAYS_BACK = 5000
DB_PATH = "DB/btc_multi_tf_30m.parquet"  # parquet 30m lu par features.py + train.py

MTF_INTERVALS = ["1h", "4h", "1d"]
```

Note la capitalisation : côté `modele_up`, le 15min utilisait `"db/..."` minuscule pour `DB_PATH` mais le code lit `Path(...)` qui est case-insensitive sur Windows. Aligner sur `"DB/..."` pour cohérence avec `modele_down` (cf. `api.py` qui pointe vers `DB/btc_multi_tf_15m.parquet`).

- [ ] **Step 2: Verify parse**

```powershell
python -c "import ast; ast.parse(open(r'backtest/modele_updown/30m/modele_up/config.py').read())"
```

- [ ] **Step 3: Commit**

```powershell
git -C C:\Users\fremo\Desktop\ML_agent_poly add backtest/modele_updown/30m/modele_up/config.py
git -C C:\Users\fremo\Desktop\ML_agent_poly commit -m "feat(30m/modele_up): config substitutions (INTERVAL=30m, DB=btc_multi_tf_30m, MTF=[1h,4h,1d])"
```

---

### Task 9: Adapter `30m/modele_up/api.py` — fix data.py reference

**Files:**
- Modify: `backtest/modele_updown/30m/modele_up/api.py`

- [ ] **Step 1: Update the parquet path constant**

Line ~28 of the clone :

```python
_PARQUET_PATH = _MODEL_DIR / "DB" / "btc_multi_tf_30m.parquet"
```

- [ ] **Step 2: Fix error message references to `data.py`**

Côté 15min, l'erreur disait `"Lance 'cd 15min/modele_up && python data.py'"`. Mais `data.py` n'existe pas dans modele_up. Remplacer par :

```python
def _load_parquet() -> pd.DataFrame:
    global _parquet
    if _parquet is None:
        if not _PARQUET_PATH.exists():
            raise FileNotFoundError(
                f"{_PARQUET_PATH} introuvable. Lance d'abord "
                f"'cd ../modele_down && python data.py' puis copie le parquet "
                f"vers modele_up/DB/ et inverse la colonne label "
                f"(df['label'] = 1 - df['label'])."
            )
        _parquet = pd.read_parquet(_PARQUET_PATH).sort_values("timestamp_ms").reset_index(drop=True)
    return _parquet
```

Faire la même chose dans `_load_model()` et `_load_sidecar()` — remplacer `"cd 15min/modele_up && python train.py"` par `"cd 30m/modele_up && python train.py"`.

- [ ] **Step 3: Verify parse**

```powershell
python -c "import ast; ast.parse(open(r'backtest/modele_updown/30m/modele_up/api.py').read())"
```

- [ ] **Step 4: Commit**

```powershell
git -C C:\Users\fremo\Desktop\ML_agent_poly add backtest/modele_updown/30m/modele_up/api.py
git -C C:\Users\fremo\Desktop\ML_agent_poly commit -m "feat(30m/modele_up): api.py points to 30m parquet + correct error messages"
```

---

### Task 10: Adapter le `CLAUDE.md` du paquet `30m/modele_up/`

**Files:**
- Modify: `backtest/modele_updown/30m/modele_up/CLAUDE.md`

- [ ] **Step 1: Search & replace 15m → 30m**

Substitutions textuelles dans tout le fichier :
- `15m` → `30m`
- `15min` → `30m` (uniformiser sur "30m" comme désignation du paquet)
- `5m` → `15m` (le fine context)
- `30m` (qui apparaissait comme MTF) → supprimer ou remplacer par "1h" suivant le contexte

- [ ] **Step 2: Remove the legacy patch chain references**

Le CLAUDE.md mentionne `python data.py` et `python add_*.py` qui n'existent pas dans modele_up. Remplacer la section "Build pipeline" par :

```markdown
## Build pipeline

`modele_up` n'a pas son propre `data.py`. Son parquet est dérivé de celui de
`modele_down` (mécanisme documenté en bas de ce fichier dans les "Naming traps").

```powershell
# 1. Construire le parquet via modele_down
cd backtest\modele_updown\30m\modele_down
python data.py          # produit DB/btc_multi_tf_30m.parquet

# 2. Copier vers modele_up/DB/ et inverser le label
cd ..\modele_up
New-Item -ItemType Directory -Force -Path DB | Out-Null
Copy-Item ..\modele_down\DB\btc_multi_tf_30m.parquet .\DB\
python -c "import pandas as pd; p = 'DB/btc_multi_tf_30m.parquet'; df = pd.read_parquet(p); df['label'] = 1 - df['label']; df.to_parquet(p, index=False); print('label flipped')"

# 3. Train + evaluate
python train.py [--label name] [--no-save] [--no-log]

# Inspection
python predict.py
python api.py
```
```

- [ ] **Step 3: Update the working-directory contract paths**

Toute occurrence de `cd backtest/modele_updown/modele_up/` →
`cd backtest/modele_updown/30m/modele_up/`.

- [ ] **Step 4: Commit**

```powershell
git -C C:\Users\fremo\Desktop\ML_agent_poly add backtest/modele_updown/30m/modele_up/CLAUDE.md
git -C C:\Users\fremo\Desktop\ML_agent_poly commit -m "feat(30m/modele_up): adapt CLAUDE.md (drop legacy patches, point to modele_down data.py)"
```

---

### Task 11: Copy parquet from modele_down to modele_up + flip label

**Files:**
- Create: `backtest/modele_updown/30m/modele_up/DB/btc_multi_tf_30m.parquet` (gitignored)

- [ ] **Step 1: Create the DB/ directory**

```powershell
cd C:\Users\fremo\Desktop\ML_agent_poly\backtest\modele_updown\30m\modele_up
New-Item -ItemType Directory -Force -Path DB | Out-Null
```

- [ ] **Step 2: Copy the parquet and flip the label**

```powershell
Copy-Item ..\modele_down\DB\btc_multi_tf_30m.parquet .\DB\
python -c "import pandas as pd; p = r'DB/btc_multi_tf_30m.parquet'; df = pd.read_parquet(p); df['label'] = 1 - df['label']; df.to_parquet(p, index=False); print('label flipped, new mean =', df['label'].dropna().mean())"
```

Expected printed mean: `~0.50-0.51` (the UP baseline). If the original was `~0.49`, the flipped should be `~1 - 0.49 = ~0.51`.

- [ ] **Step 3: Don't commit** — parquet is gitignored.

---

### Task 12: Train `modele_up` 30m

**Files:**
- Generated: `backtest/modele_updown/30m/modele_up/btc_updown_model.json`
- Generated: `backtest/modele_updown/30m/modele_up/btc_updown_model.thresholds.json`
- Generated: `backtest/modele_updown/30m/modele_up/logs/runs.csv`
- Generated: `backtest/modele_updown/30m/modele_up/logs/test_predictions.csv`

- [ ] **Step 1: Run train.py**

```powershell
cd C:\Users\fremo\Desktop\ML_agent_poly\backtest\modele_updown\30m\modele_up
python train.py
```

Expected: ~30 s à 2 min runtime. Last lines show calibration summary.

- [ ] **Step 2: Verify artifacts**

```powershell
Test-Path .\btc_updown_model.json
Test-Path .\btc_updown_model.thresholds.json
Test-Path .\logs\runs.csv
Test-Path .\logs\test_predictions.csv
```

All expected: `True`.

- [ ] **Step 3: Verify the run beats baseline**

```powershell
python -c "import pandas as pd; r = pd.read_csv(r'logs/runs.csv').iloc[-1]; print('baseline_wr_up:', r.get('baseline_wr_up')); print('adaptive_total_wr:', r.get('adaptive_total_wr')); print('adaptive_total_taken:', r.get('adaptive_total_taken'))"
```

Expected: `adaptive_total_wr > baseline_wr_up` AND `adaptive_total_taken > 0`. Note the values.

- [ ] **Step 4: Smoke-test `api.py`**

```powershell
python api.py
```

Expected: prints `name=modele_up setups_taken=<int>` and a small DataFrame head. Should not crash.

- [ ] **Step 5: Don't commit** — artifacts are gitignored.

---

### Task 13: Create `30m/main.py` from clone of parent `main.py`

**Files:**
- Create: `backtest/modele_updown/30m/main.py`

- [ ] **Step 1: Copy parent `main.py` to 30m/**

```powershell
Copy-Item C:\Users\fremo\Desktop\ML_agent_poly\backtest\modele_updown\main.py C:\Users\fremo\Desktop\ML_agent_poly\backtest\modele_updown\30m\main.py
```

- [ ] **Step 2: Adapter le `sys.path` trick**

Open `backtest/modele_updown/30m/main.py`. Lines around 38-46 should look like :

```python
# 15min/ contains modele_up and modele_down. The directory name starts with a
# digit so we can't write `from 15min.modele_up import ...` directly -- inject
# 15min/ into sys.path and import the packages as top-level instead.
_15MIN_DIR = str(Path(__file__).resolve().parent / "15min")
if _15MIN_DIR not in sys.path:
    sys.path.insert(0, _15MIN_DIR)

from modele_up import api as api_up
from modele_down import api as api_down
```

Replace with :

```python
# 30m/ is where this main lives. modele_up and modele_down are direct children.
# But "30m" starts with a digit, so we can't write `from 30m.modele_up import ...`
# directly even from the parent — we always import as top-level via sys.path.
# Injecting our own parent into sys.path is enough since main.py lives in 30m/.
_PKG_DIR = str(Path(__file__).resolve().parent)
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)

from modele_up import api as api_up
from modele_down import api as api_down
```

- [ ] **Step 3: Adapter le panel Rich (cosmétique)**

Around line 167-171 the panel header reads :

```python
    header = Text.assemble(
        ("TEAM ", "bold cyan"),
        ("modele_updown", "bold magenta"),
        ("  -  backtest report", "dim"),
    )
```

Change to :

```python
    header = Text.assemble(
        ("TEAM ", "bold cyan"),
        ("modele_updown 30m", "bold magenta"),
        ("  -  backtest report", "dim"),
    )
```

- [ ] **Step 4: Verify parse**

```powershell
python -c "import ast; ast.parse(open(r'backtest/modele_updown/30m/main.py').read())"
```

- [ ] **Step 5: Commit**

```powershell
git -C C:\Users\fremo\Desktop\ML_agent_poly add backtest/modele_updown/30m/main.py
git -C C:\Users\fremo\Desktop\ML_agent_poly commit -m "feat(30m): add main.py team backtest orchestrator (clone of modele_updown/main.py)"
```

---

### Task 14: End-to-end smoke test — run `30m/main.py`

**Files:**
- Generated: `backtest/modele_updown/30m/logs/team_bets.csv`
- Generated: `backtest/modele_updown/30m/logs/team_runs.csv`

- [ ] **Step 1: Run main.py from 30m/**

```powershell
cd C:\Users\fremo\Desktop\ML_agent_poly\backtest\modele_updown\30m
python main.py
```

Expected: Rich panel "TEAM modele_updown 30m" with solo + team stats. No exceptions.

- [ ] **Step 2: Verify logs exist and have one row**

```powershell
Test-Path .\logs\team_bets.csv
Test-Path .\logs\team_runs.csv
python -c "import pandas as pd; print('team_runs rows:', len(pd.read_csv(r'logs/team_runs.csv'))); print(pd.read_csv(r'logs/team_runs.csv').tail(1).T)"
```

Expected: `team_runs rows: 1` (at least), and the printed row contains `team_wr`, `team_bets`, `up_solo_wr`, `down_solo_wr` etc.

- [ ] **Step 3: Verify `team_wr` is sensible**

`team_wr` should be > 0.50. If the team backtest underperforms the solo models or shows `nan`, investigate before committing — it could indicate a mismatched test window between the two packages, or a parquet/sidecar mismatch.

- [ ] **Step 4: Don't commit** — logs are gitignored.

---

### Task 15: Update root `CLAUDE.md` to document the 30m equipe

**Files:**
- Modify: `C:\Users\fremo\Desktop\ML_agent_poly\CLAUDE.md`

- [ ] **Step 1: Update the layer-1 description**

Current section "There are three layers" lists only the 15min packages. Add a note under layer 1 :

```markdown
1. **`backtest/modele_updown/15min/modele_up/`** and **`backtest/modele_updown/15min/modele_down/`** — two sister XGBoost binary classifiers. Each predicts the direction of the next 15m BTC/USD candle from its own side.
   ...
   **A parallel team `backtest/modele_updown/30m/`** (with the same `modele_up` / `modele_down` layout + a self-contained `main.py`) predicts the direction of the next 30m candle. Independent of the 15m team. Same conventions, same sidecar shape, just a different base TF and a different MTF set (`[1h, 4h, 1d]` + 15m fine context instead of `[30m, 1h, 4h, 1d]` + 5m).
```

- [ ] **Step 2: Update the "import trick" section**

Add a note that the trick now applies to `30m/` too — its child packages (`modele_up`, `modele_down`) are imported by inserting `30m/` itself on sys.path.

- [ ] **Step 3: Update the "Working-directory contract" examples block**

Add a sibling block under the existing 15min examples :

```markdown
# Team backtest 30m
cd backtest\modele_updown\30m
python main.py
```

- [ ] **Step 4: Commit**

```powershell
git -C C:\Users\fremo\Desktop\ML_agent_poly add CLAUDE.md
git -C C:\Users\fremo\Desktop\ML_agent_poly commit -m "docs(claude): document the parallel 30m team in root CLAUDE.md"
```

---

## Verification — full state check

After all 15 tasks, run :

```powershell
cd C:\Users\fremo\Desktop\ML_agent_poly
git status
```

Expected uncommitted changes : **none** related to this work. (Other unrelated modifications from the user's working tree may still appear, but nothing under `backtest/modele_updown/30m/` should be untracked except the gitignored runtime artifacts `DB/`, `logs/`, `*.parquet`, `*.json`).

```powershell
git -C C:\Users\fremo\Desktop\ML_agent_poly log --oneline -15
```

Expected to see ~6-8 new commits authored by this plan : bootstrap × 2, config × 2, data.py, CLAUDE.md × 2, api.py, main.py, root CLAUDE.md.

```powershell
Test-Path C:\Users\fremo\Desktop\ML_agent_poly\backtest\modele_updown\30m\modele_down\btc_updown_model.json
Test-Path C:\Users\fremo\Desktop\ML_agent_poly\backtest\modele_updown\30m\modele_up\btc_updown_model.json
Test-Path C:\Users\fremo\Desktop\ML_agent_poly\backtest\modele_updown\30m\logs\team_runs.csv
```

All three expected: `True`.

```powershell
git -C C:\Users\fremo\Desktop\ML_agent_poly status backtest/modele_updown/15min/
```

Expected: **no modifications** under `15min/`. The 15m team is strictly untouched.
