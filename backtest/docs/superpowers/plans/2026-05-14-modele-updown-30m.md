# modele_updown 30m — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cloner `backtest/modele_updown/15min/` en `backtest/modele_updown/30m/` pour produire une équipe modele_updown qui prédit la prochaine bougie 30 minutes. Périmètre : modèles + team backtest. Pas de live.

**Architecture:** Deux paquets sœurs (`30m/modele_up/` + `30m/modele_down/`) qui exposent le contrat `evaluate_on_slice` standardisé, orchestrés par `30m/main.py`. Le seul vrai pipeline data vit dans `30m/modele_down/data.py` ; le parquet de `30m/modele_up/` est dérivé du sien avec le label flippé.

**Tech Stack:** Python 3.10+, pandas, numpy, xgboost, scikit-learn, rich. Pas de framework de test — vérification = inspecter `logs/runs.csv` et `logs/team_runs.csv`.

**Spec:** `backtest/docs/superpowers/specs/2026-05-14-modele-updown-30m-design.md`.

**Les seules vraies différences vs 15min** (rappel) :
| # | Quoi | 15min | 30m |
|---|---|---|---|
| 1 | INTERVAL | `"15m"` | `"30m"` |
| 2 | MTF | `[30m, 1h, 4h, 1d]` | `[1h, 4h, 1d]` |
| 3 | Contexte fin | `5m` | `15m` |
| 4 | Nom parquet | `btc_multi_tf_15m.parquet` | `btc_multi_tf_30m.parquet` |

---

### Task 1: Bootstrap — verbatim copy of both packages

**Files:**
- Create: `backtest/modele_updown/30m/modele_down/` (full tree from `15min/modele_down/`)
- Create: `backtest/modele_updown/30m/modele_up/` (full tree from `15min/modele_up/`)

- [ ] **Step 1: Copy both packages via `git ls-tree` (so we skip gitignored artifacts)**

```powershell
cd C:\Users\fremo\Desktop\ML_agent_poly
foreach ($pkg in @("modele_down", "modele_up")) {
    $src = "backtest\modele_updown\15min\$pkg"
    $dst = "backtest\modele_updown\30m\$pkg"
    $files = git ls-tree -r --name-only HEAD -- $src
    foreach ($f in $files) {
        $rel = $f.Substring($src.Length + 1)
        $target = Join-Path $dst $rel
        New-Item -ItemType Directory -Force -Path (Split-Path $target -Parent) | Out-Null
        Copy-Item -Path $f -Destination $target -Force
    }
}
```

- [ ] **Step 2: Verify file counts match the source**

```powershell
"modele_down", "modele_up" | ForEach-Object {
    $src_count = (git -C C:\Users\fremo\Desktop\ML_agent_poly ls-tree -r --name-only HEAD -- "backtest/modele_updown/15min/$_/" | Measure-Object -Line).Lines
    $dst_count = (Get-ChildItem -Recurse -File "C:\Users\fremo\Desktop\ML_agent_poly\backtest\modele_updown\30m\$_").Count
    Write-Host "$_ : source=$src_count, copy=$dst_count"
}
```

Expected: les deux counts égaux pour chaque paquet.

- [ ] **Step 3: Commit**

```powershell
git -C C:\Users\fremo\Desktop\ML_agent_poly add backtest/modele_updown/30m/
git -C C:\Users\fremo\Desktop\ML_agent_poly commit -m "feat(30m): bootstrap modele_up + modele_down (verbatim copy of 15min/)"
```

---

### Task 2: Adapt `30m/modele_down/` for 30m base TF

**Files:**
- Modify: `backtest/modele_updown/30m/modele_down/config.py`
- Modify: `backtest/modele_updown/30m/modele_down/data.py`
- Modify: `backtest/modele_updown/30m/modele_down/CLAUDE.md`

- [ ] **Step 1: `config.py` — apply the four substitutions**

```python
SYMBOL = "btcusd"
INTERVAL = "30m"
DAYS_BACK = 5000
DB_PATH = "db/btc_multi_tf_30m.parquet"

MTF_INTERVALS = ["1h", "4h", "1d"]

INTERVAL_MS = {
    "15m": 15 * 60 * 1000,
    "30m": 30 * 60 * 1000,
    "1h":  60 * 60 * 1000,
    "4h":  4 * 60 * 60 * 1000,
    "1d":  24 * 60 * 60 * 1000,
}
```

(On garde `15m` dans `INTERVAL_MS` parce qu'il sert maintenant de contexte fin, analogue au rôle du `5m` côté 15min.)

- [ ] **Step 2: `data.py` — substitutions structurelles**

Trois zones à modifier :

**(a) `TIMEFRAMES` dict (lignes ~81-88)** — drop 5m :

```python
TIMEFRAMES: dict[str, tuple[int, int]] = {
    "15m": (15 * 60 * 1000,          15 * 60),
    "30m": (30 * 60 * 1000,          30 * 60),
    "1h":  (60 * 60 * 1000,          60 * 60),
    "4h":  ( 4 * 60 * 60 * 1000,  4 * 60 * 60),
    "1d":  (24 * 60 * 60 * 1000, 24 * 60 * 60),
}
```

**(b) `MTF_KEPT_COLS` (lignes ~134-152)** — le `5m` devient `15m` (fine context), le `30m` disparaît (devient la base) :

```python
MTF_KEPT_COLS: dict[str, list[str]] = {
    "15m": ["open_ret", "high_ret", "low_ret", "close_ret",
            "volume_ratio", "body_ratio_close", "body_ratio_avg",
            "obv_pressure", "atr_pct"],
    "1h":  ["open_ret", "low_ret", "close_ret",
            "volume_ratio", "body_ratio_close"],
    "4h":  ["close_ret", "volume_ratio",
            "body_ratio_avg", "obv_pressure", "atr_pct"],
    "1d":  ["open_ret", "high_ret", "low_ret",
            "volume_ratio", "body_ratio_close", "body_ratio_avg",
            "obv_pressure", "atr_pct"],
}
```

**(c) Constantes d'intervalle (lignes ~116-117)** — supprimer `INTERVAL_5M_MS`, ajouter `INTERVAL_30M_MS` :

```python
INTERVAL_15M_MS = 15 * 60 * 1000
INTERVAL_30M_MS = 30 * 60 * 1000
```

**(d) `process_timeframe_to_candles` (~lignes 391-462)** — le cas spécial `5m` devient `15m` :

```python
    if timeframe == "15m":
        # Garde les 15m dont open_time tombe à T + 15min pour un T multiple de 30min.
        offset_ms = feat["timestamp_ms"].to_numpy(dtype=np.int64) % INTERVAL_30M_MS
        keep_mask = offset_ms == (15 * 60 * 1000)
        n_before = len(feat)
        feat = feat.loc[keep_mask].reset_index(drop=True)
        print(f"    [15m] filtre close-align : {n_before:,} -> {len(feat):,} rows "
              f"(garde 1/2, les autres ne sont pas utilisées par le 30m)")
```

**(e) `build_master_15m` → `build_master_30m`** — renommer la fonction et son contenu :

```python
def build_master_30m(db_path: Path, out_path: Path) -> tuple[int, int]:
    """Construit le DataFrame final 30m (Tier 0 + 15m + MTF + Tier 1 + Tier 2)."""
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

    # 4.3 — MTF lookups (1h, 4h, 1d)
    for tf in ["1h", "4h", "1d"]:
        df_oth = _load_tf(db_path, tf)
        if df_oth.empty:
            print(f"  [{tf} lookup] skip (table vide)")
            continue
        step_ms, _ = TIMEFRAMES[tf]
        n = _add_lookup_features(df_30m, df_oth, tf, step_ms, MTF_KEPT_COLS[tf])
        print(f"  [{tf} lookup] +{n} cols (post-drop)")

    # 4.4 — Tier 1 derived (7 cols)
    n_t1 = _add_tier1_derived(df_30m)
    print(f"  [Tier 1 derived] +{n_t1} cols (dist_ema_20, return_3/96, hour_sin/cos, wicks)")

    # 4.5 — Tier 2 (9 cols)
    n_t2 = _add_tier2(df_30m)
    print(f"  [Tier 2] +{n_t2} cols (MACD ×3, BB ×2, MFI ×1, rv ×3)")

    # 4.6 — derived columns (unchanged logic, just runs on 30m bars now)
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
    invalid = ((cr == or_) | (cr_next == or_next)
               | np.isnan(cr) | np.isnan(or_)
               | np.isnan(cr_next) | np.isnan(or_next))
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
        raise ValueError(f"Extension non supportée pour {out_path} : '{suffix}'.")
    elapsed = time.time() - t0
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"  -> {out_path.name} : {len(df_30m):,} rows x {len(df_30m.columns)} cols, "
          f"{size_mb:.1f} MB ({elapsed:.1f}s)")

    return len(df_30m), len(df_30m.columns)
```

**(f) Caller de `build_master_15m`** (~ligne 858) — rename + master path :

```python
    master_path = csv_dir / f"{db_stem}_30m.parquet"
    n_rows, n_cols = build_master_30m(db_path, master_path)
```

**(g) Cosmétique : recap text, docstrings, argparse description, SQL `CASE` du summary** — substitutions textuelles `15m` → `30m`, et dans le `ORDER BY CASE` du summary :

```python
        ORDER BY CASE timeframe
            WHEN '15m' THEN 1 WHEN '30m' THEN 2
            WHEN '1h'  THEN 3 WHEN '4h'  THEN 4 WHEN '1d'  THEN 5 END
```

- [ ] **Step 3: `CLAUDE.md` — substitutions textuelles + nettoyage**

Substitutions globales dans le fichier :
- `15m` → `30m` (sauf les phrases qui parlent du 15m en tant que contexte fin)
- `5m` → `15m` (le nouveau contexte fin)
- `cd backtest/modele_updown/modele_down/` → `cd backtest/modele_updown/30m/modele_down/`

Remplacer aussi la section "Build pipeline — order matters" (qui décrit la chaîne legacy de patches qui n'existe plus dans le repo) par la pipeline mono-script actuelle :

```markdown
## Build pipeline

Le pipeline data est consolidé dans un seul script :

```powershell
cd backtest\modele_updown\30m\modele_down
python data.py            # cold fetch + features Tier 0+1+2 en un seul script
python train.py           # entraîne, écrit btc_updown_model.json + sidecar + logs/
python predict.py         # smoke-test
python api.py             # exercice evaluate_on_slice
```
```

- [ ] **Step 4: Verify all three files parse**

```powershell
python -c "import ast; [ast.parse(open(f).read()) for f in ['backtest/modele_updown/30m/modele_down/config.py', 'backtest/modele_updown/30m/modele_down/data.py']]"
```

Expected: no output.

- [ ] **Step 5: Commit**

```powershell
git -C C:\Users\fremo\Desktop\ML_agent_poly add backtest/modele_updown/30m/modele_down/
git -C C:\Users\fremo\Desktop\ML_agent_poly commit -m "feat(30m/modele_down): adapt config + data.py + CLAUDE.md for 30m base TF"
```

---

### Task 3: Adapt `30m/modele_up/` for 30m base TF

**Files:**
- Modify: `backtest/modele_updown/30m/modele_up/config.py`
- Modify: `backtest/modele_updown/30m/modele_up/api.py`
- Modify: `backtest/modele_updown/30m/modele_up/CLAUDE.md`

- [ ] **Step 1: `config.py` — apply the four substitutions**

Identique à Task 2 Step 1, mais dans `30m/modele_up/config.py` :

```python
SYMBOL = "btcusd"
INTERVAL = "30m"
DAYS_BACK = 5000
DB_PATH = "DB/btc_multi_tf_30m.parquet"

MTF_INTERVALS = ["1h", "4h", "1d"]

INTERVAL_MS = {
    "15m": 15 * 60 * 1000,
    "30m": 30 * 60 * 1000,
    "1h":  60 * 60 * 1000,
    "4h":  4 * 60 * 60 * 1000,
    "1d":  24 * 60 * 60 * 1000,
}
```

- [ ] **Step 2: `api.py` — update parquet path + fix error messages**

Ligne ~28 :

```python
_PARQUET_PATH = _MODEL_DIR / "DB" / "btc_multi_tf_30m.parquet"
```

Dans `_load_parquet()` (~ligne 67-69), remplacer le message d'erreur (qui référence un `data.py` qui n'existe pas dans modele_up) :

```python
        if not _PARQUET_PATH.exists():
            raise FileNotFoundError(
                f"{_PARQUET_PATH} introuvable. Lance d'abord "
                f"'cd ../modele_down && python data.py' puis copie le parquet "
                f"vers modele_up/DB/ et inverse la colonne label "
                f"(df['label'] = 1 - df['label'])."
            )
```

Dans `_load_model()` et `_load_sidecar()`, remplacer `"cd 15min/modele_up && python train.py"` par `"cd 30m/modele_up && python train.py"`.

- [ ] **Step 3: `CLAUDE.md` — substitutions + nettoyage de la pipeline legacy**

Substitutions globales `15m` → `30m`, `5m` → `15m`. Remplacer la section "Build pipeline" par la procédure réelle pour modele_up (qui n'a pas de data.py) :

```markdown
## Build pipeline

`modele_up` n'a pas son propre `data.py`. Son parquet est dérivé de celui de `modele_down` :

```powershell
# 1. Construire le parquet via modele_down
cd backtest\modele_updown\30m\modele_down
python data.py            # produit DB/btc_multi_tf_30m.parquet

# 2. Copier vers modele_up/DB/ et inverser le label
cd ..\modele_up
New-Item -ItemType Directory -Force -Path DB | Out-Null
Copy-Item ..\modele_down\DB\btc_multi_tf_30m.parquet .\DB\
python -c "import pandas as pd; p='DB/btc_multi_tf_30m.parquet'; df=pd.read_parquet(p); df['label']=1-df['label']; df.to_parquet(p, index=False)"

# 3. Train + evaluate
python train.py
python api.py
```
```

- [ ] **Step 4: Verify parse**

```powershell
python -c "import ast; [ast.parse(open(f).read()) for f in ['backtest/modele_updown/30m/modele_up/config.py', 'backtest/modele_updown/30m/modele_up/api.py']]"
```

- [ ] **Step 5: Commit**

```powershell
git -C C:\Users\fremo\Desktop\ML_agent_poly add backtest/modele_updown/30m/modele_up/
git -C C:\Users\fremo\Desktop\ML_agent_poly commit -m "feat(30m/modele_up): adapt config + api.py + CLAUDE.md for 30m base TF"
```

---

### Task 4: Create `30m/main.py` + update root `CLAUDE.md`

**Files:**
- Create: `backtest/modele_updown/30m/main.py`
- Modify: `C:\Users\fremo\Desktop\ML_agent_poly\CLAUDE.md`

- [ ] **Step 1: Copy `15min/main.py` to `30m/main.py`**

```powershell
Copy-Item C:\Users\fremo\Desktop\ML_agent_poly\backtest\modele_updown\15min\main.py `
          C:\Users\fremo\Desktop\ML_agent_poly\backtest\modele_updown\30m\main.py
```

(Note : récemment `main.py` et `live/` ont été déplacés sous `15min/` pour rendre la structure symétrique. Le 30m miroitera cette structure exactement.)

- [ ] **Step 2: Adapter le `sys.path` trick (lignes 38-46) et le header Rich (ligne ~167)**

Remplacer le bloc `_15MIN_DIR = ...` par :

```python
# 30m/ is where this main lives. modele_up and modele_down are direct children.
# But "30m" starts with a digit so we always import as top-level via sys.path.
_PKG_DIR = str(Path(__file__).resolve().parent)
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)

from modele_up import api as api_up
from modele_down import api as api_down
```

Et adapter le header du panel Rich (line ~167-171) :

```python
    header = Text.assemble(
        ("TEAM ", "bold cyan"),
        ("modele_updown 30m", "bold magenta"),
        ("  -  backtest report", "dim"),
    )
```

- [ ] **Step 3: Update root `CLAUDE.md`**

Sous la section "There are three layers", ajouter un sous-paragraphe à la fin du point 1 (les deux paquets sœurs 15min) :

```markdown
**A parallel team `backtest/modele_updown/30m/`** (with the same `modele_up` / `modele_down` layout + a self-contained `main.py`) predicts the direction of the next 30m candle. Independent of the 15m team. Same conventions, same sidecar shape, just a different base TF and a different MTF set (`[1h, 4h, 1d]` + 15m fine context instead of `[30m, 1h, 4h, 1d]` + 5m). The `sys.path` import trick applies the same way (`30m/` also starts with a digit). To run the team backtest: `cd backtest\modele_updown\30m && python main.py`.
```

- [ ] **Step 4: Verify parse**

```powershell
python -c "import ast; ast.parse(open(r'backtest/modele_updown/30m/main.py').read())"
```

- [ ] **Step 5: Commit**

```powershell
git -C C:\Users\fremo\Desktop\ML_agent_poly add backtest/modele_updown/30m/main.py CLAUDE.md
git -C C:\Users\fremo\Desktop\ML_agent_poly commit -m "feat(30m): add team backtest main.py + document in root CLAUDE.md"
```

---

### Task 5: Run end-to-end and verify

This task generates only gitignored artifacts (parquet, model JSON, logs). No commit at the end — the verification IS the test.

- [ ] **Step 1: Run `data.py` for `modele_down`**

```powershell
cd C:\Users\fremo\Desktop\ML_agent_poly\backtest\modele_updown\30m\modele_down
python data.py
```

Expected: ~3-5 min runtime. Last line : `=== Terminé. Lance maintenant : python train.py ===`.

Verify the parquet :
```powershell
python -c "import pandas as pd; df = pd.read_parquet(r'DB/btc_multi_tf_30m.parquet'); print('shape:', df.shape); print('label baseline (DOWN):', df['label'].dropna().mean())"
```

Expected: shape ~(240000, ~52), label baseline ~0.49-0.50.

- [ ] **Step 2: Train `modele_down`**

```powershell
python train.py
python -c "import pandas as pd; r = pd.read_csv(r'logs/runs.csv').iloc[-1]; print('baseline:', r.get('baseline_wr_down'), 'adaptive_wr:', r.get('adaptive_total_wr'), 'taken:', r.get('adaptive_total_taken'))"
```

Expected: `adaptive_total_wr > baseline_wr_down` AND `taken > 0`. Note the values.

- [ ] **Step 3: Build `modele_up` parquet (copy + flip label) and train**

```powershell
cd ..\modele_up
New-Item -ItemType Directory -Force -Path DB | Out-Null
Copy-Item ..\modele_down\DB\btc_multi_tf_30m.parquet .\DB\
python -c "import pandas as pd; p = r'DB/btc_multi_tf_30m.parquet'; df = pd.read_parquet(p); df['label'] = 1 - df['label']; df.to_parquet(p, index=False); print('label flipped, new mean (UP baseline):', df['label'].dropna().mean())"

python train.py
python -c "import pandas as pd; r = pd.read_csv(r'logs/runs.csv').iloc[-1]; print('baseline:', r.get('baseline_wr_up'), 'adaptive_wr:', r.get('adaptive_total_wr'), 'taken:', r.get('adaptive_total_taken'))"
```

Expected: `adaptive_total_wr > baseline_wr_up` AND `taken > 0`. Note the values.

- [ ] **Step 4: Run the team backtest**

```powershell
cd ..
python main.py
```

Expected: Rich panel "TEAM modele_updown 30m" with both solo + team stats. `logs/team_runs.csv` and `logs/team_bets.csv` created.

```powershell
python -c "import pandas as pd; print(pd.read_csv(r'logs/team_runs.csv').tail(1).T)"
```

Expected: row with `team_wr` > 0.50 (sensible value).

- [ ] **Step 5: Final cleanup check — 15min must be untouched**

```powershell
git -C C:\Users\fremo\Desktop\ML_agent_poly status backtest/modele_updown/15min/
```

Expected: no modifications. The 15m team must be strictly intact.

```powershell
git -C C:\Users\fremo\Desktop\ML_agent_poly log --oneline -8
```

Expected: 4 new commits from Tasks 1-4 visible.

---

## Done.

Si les WR sont décevants sur 30m (≤ baseline), c'est attendu : les hyperparamètres ont été tunés pour le 15m. Tuning pour 30m = passe suivante hors périmètre.
