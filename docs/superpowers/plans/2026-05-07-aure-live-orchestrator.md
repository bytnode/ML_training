# Aure Live Orchestrator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Mettre Aure (modèle dans `aure.json`) en conditions réelles via un daemon qui fetch Bitstamp toutes les 15min, prédit, archive (CSVs) et notifie sur Telegram.

**Architecture:** Daemon Python long-running aligné sur les frontières 15min UTC, dans `live/`, sibling de `backtest/`. Réutilise les indicateurs de `backtest/data.py` via sys.path hack. Stocke deux CSVs (`training_data_live` + `decisions`) et un log. Notifie via Telegram HTTP API. Cf. spec `docs/superpowers/specs/2026-05-07-aure-live-orchestrator-design.md`.

**Tech Stack:** Python 3.10+, requests, pandas, numpy, xgboost (toutes déjà dispos pour backtest).

---

## Convention de vérification (pas de pytest)

Le repo n'a pas de pytest, ruff ou Makefile (cf. `backtest/CLAUDE.md` : "Don't add pytest, ruff, or a Makefile without asking"). Chaque task se vérifie via :

- **Smoke-test stand-alone** : section `if __name__ == "__main__":` dans le module, exécutée à la main, output à comparer contre l'attendu spécifié dans la step.
- **Inspection manuelle** : `Get-Content` sur le CSV/log, `python -c "..."` ad-hoc.
- **Run end-to-end final** via `python orchestrator.py --dry-run` (Task 8).

Pas d'ajout de framework de test. Si une vérification est impossible sans framework, c'est un signal de problème de design, pas une excuse pour introduire pytest.

## Note d'architecture critique : nommage du module de config local

Le module de config de `live/` s'appelle **`settings.py`** (et non `config.py`). Raison technique :

- `live/fetch.py` (et `features.py`) fait `sys.path.insert(0, "../backtest")` au top du fichier pour pouvoir faire `from data import ...`.
- `backtest/data.py` exécute `from config import ...` au load (ses imports module-level).
- À ce moment-là, `sys.path = [backtest_path, live_path, ...]`. Donc Python trouve `backtest/config.py` en premier — bon pour `data.py`.
- **MAIS** ensuite, si `live/orchestrator.py` fait `from config import MODEL_PATH`, Python trouve **encore** `backtest/config.py` (qui n'a pas `MODEL_PATH`) → `ImportError`.
- En nommant le module local `settings.py`, plus de collision : `from settings import MODEL_PATH` ne peut résoudre qu'au fichier local.

---

## File Structure

| Path | Purpose | Created in |
|---|---|---|
| `live/__init__.py` | Marqueur de dossier (optionnel, mais utile pour autodiscovery d'IDE) | Task 1 |
| `live/settings.py` | Tous les chemins, constantes de cycle, env vars Telegram | Task 1 |
| `live/fetch.py` | Wrappers fetch Bitstamp (warmup + recent), réutilise `backtest/data.py` | Task 2 |
| `live/features.py` | Détection des setups au dernier index + extraction des 42 features (mirror live de `extract_features`) | Task 3 |
| `live/store.py` | I/O CSV : append rows, update label rétroactif, update outcome | Task 4 |
| `live/notify.py` | Client Telegram (POST sendMessage) | Task 5 |
| `live/orchestrator.py` | Daemon principal : sleep alignment, model load, sidecar copy, run_cycle | Task 6 |
| `live/logs/.gitkeep` | Garde le dossier en repo | Task 1 |
| `aure.thresholds.json` | Sidecar (créé au 1er run de l'orchestrateur) | Task 6 (runtime) |
| `.gitignore` | Ignore `live/logs/*.csv`, `*.log`, `aure.thresholds.json` | Task 1 |
| `CLAUDE.md` (racine) | Mention du subproject `live/` | Task 7 |
| `live/CLAUDE.md` | Doc spécifique au subproject (commandes, contrats internes) | Task 7 |

Total : 7 nouveaux fichiers Python + 4 fichiers de doc/config.

---

### Task 1: Scaffold du dossier `live/` et `settings.py`

**Files:**
- Create: `live/__init__.py`
- Create: `live/settings.py`
- Create: `live/logs/.gitkeep`
- Modify: `.gitignore`

- [ ] **Step 1: Créer la structure de dossier**

Run (PowerShell):
```powershell
New-Item -ItemType Directory -Path "live\logs" -Force | Out-Null
New-Item -ItemType File -Path "live\__init__.py" -Force | Out-Null
New-Item -ItemType File -Path "live\logs\.gitkeep" -Force | Out-Null
```

Expected: les fichiers existent. Vérifier avec `Get-ChildItem live -Recurse`.

- [ ] **Step 2: Écrire `live/settings.py`**

Create `live/settings.py` with:

```python
"""Configuration locale de l'orchestrateur Aure (live mode).

Important : ce module s'appelle settings.py et non config.py pour eviter
la collision de namespace avec backtest/config.py quand sys.path inclut
backtest/. Voir docs/superpowers/plans/2026-05-07-aure-live-orchestrator.md
section 'Note d'architecture critique'.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
LOGS_DIR = ROOT / "logs"

TRAINING_DATA_LIVE = LOGS_DIR / "training_data_live.csv"
DECISIONS_CSV = LOGS_DIR / "decisions.csv"
ORCHESTRATOR_LOG = LOGS_DIR / "orchestrator.log"

MODEL_PATH = REPO_ROOT / "aure.json"
THRESHOLDS_PATH = REPO_ROOT / "aure.thresholds.json"
THRESHOLDS_SOURCE = REPO_ROOT / "backtest" / "btc_reversal_model.thresholds.json"

# Cycle parameters
WARMUP_CANDLES_15M = 200
WARMUP_CANDLES_MTF = 50
RECENT_FETCH_15M = 5
RECENT_FETCH_MTF = 3
FETCH_RETRIES = 3
FETCH_BACKOFF_BASE = 2  # seconds, exponential

# Telegram
TELEGRAM_BOT_TOKEN_ENV = "AURE_TELEGRAM_BOT_TOKEN"
TELEGRAM_CHAT_ID_ENV = "AURE_TELEGRAM_CHAT_ID"
TELEGRAM_TIMEOUT = 10  # seconds

# 15min en millisecondes (utilise pour calculer next boundary et target_open_time)
INTERVAL_15M_MS = 15 * 60 * 1000

# Marge apres frontiere 15min avant de fetch (laisse le temps a Bitstamp de publier)
CYCLE_FUDGE_SECONDS = 2
```

- [ ] **Step 3: Mettre à jour `.gitignore`**

Append to `.gitignore`:

```
# Live orchestrator artifacts
live/logs/*.csv
live/logs/*.log
aure.thresholds.json
```

- [ ] **Step 4: Smoke-vérifier les chemins**

Run:
```powershell
python -c "from live.settings import MODEL_PATH, LOGS_DIR; print(MODEL_PATH); print(LOGS_DIR.exists())"
```

Expected output (path absolu de votre machine) :
```
C:\Users\fremo\Desktop\ML_agent_poly\aure.json
True
```

- [ ] **Step 5: Commit**

```bash
git add live/__init__.py live/settings.py live/logs/.gitkeep .gitignore
git commit -m "feat(live): scaffold directory structure and settings"
```

---

### Task 2: `live/fetch.py` — wrapper Bitstamp

Réutilise `fetch_all_klines` et `fetch_all_timeframes` de `backtest/data.py`. Fournit deux fonctions :
- `fetch_warmup_window(n_15m, n_mtf)` : appel au démarrage du daemon.
- `fetch_recent_window(n_15m, n_mtf)` : appel à chaque cycle (typiquement n_15m=5, n_mtf=3).

Les deux ont la même implémentation (sémantique différente seulement). Les fetches Bitstamp sont parallélisés en interne par `fetch_all_klines` (cf. `backtest/data.py`).

**Files:**
- Create: `live/fetch.py`

- [ ] **Step 1: Écrire `live/fetch.py`**

Create `live/fetch.py` with:

```python
"""Fetch des bougies Bitstamp pour le mode live.

Reutilise les fonctions de backtest/data.py via un sys.path hack.
"""
import os
import sys
import time

# Ajoute backtest/ a sys.path AVANT d'importer ses modules. Cette ligne doit
# rester avant les imports backtest et avant tout import qui ferait
# 'from config import ...' (cf. note critique du plan).
_BACKTEST_DIR = os.path.join(os.path.dirname(__file__), "..", "backtest")
sys.path.insert(0, os.path.abspath(_BACKTEST_DIR))

from data import fetch_all_klines, fetch_all_timeframes  # noqa: E402
from config import SYMBOL, MTF_INTERVALS, INTERVAL_MS  # noqa: E402

INTERVAL_15M = "15m"


def fetch_warmup_window(n_15m: int, n_mtf: int) -> dict:
    """Fetche assez de bougies pour warmup les indicateurs au demarrage.

    Returns: dict avec keys '15m', '30m', '1h', '4h', '1d'. Chaque value est
             une liste triee de [open_time_ms, o, h, l, c, v].
    """
    return _fetch_all(n_15m, n_mtf)


def fetch_recent_window(n_15m: int, n_mtf: int) -> dict:
    """Fetche les dernieres bougies a chaque cycle (typiquement 5 / 3).

    Same implementation que fetch_warmup_window ; semantique distincte pour
    clarifier l'intention au call site.
    """
    return _fetch_all(n_15m, n_mtf)


def _fetch_all(n_15m: int, n_mtf: int) -> dict:
    now_ms = int(time.time() * 1000)
    out = {}

    # 15m
    interval_15m_ms = INTERVAL_MS[INTERVAL_15M]
    start_15m = now_ms - (n_15m + 2) * interval_15m_ms  # +2 marge pour overlap
    klines_15m = fetch_all_klines(start_15m, now_ms, INTERVAL_15M)
    out[INTERVAL_15M] = klines_15m[-n_15m:] if len(klines_15m) > n_15m else klines_15m

    # MTF
    for tf in MTF_INTERVALS:
        tf_ms = INTERVAL_MS[tf]
        start_tf = now_ms - (n_mtf + 2) * tf_ms
        klines_tf = fetch_all_klines(start_tf, now_ms, tf)
        out[tf] = klines_tf[-n_mtf:] if len(klines_tf) > n_mtf else klines_tf

    return out


def merge_klines(existing: list, fresh: list) -> list:
    """Merge fresh dans existing en deduplicant par open_time_ms.

    fresh ecrase existing pour les open_time_ms en doublon (Bitstamp peut
    re-publier les dernieres bougies avec des valeurs revisees).
    """
    by_ts = {k[0]: k for k in existing}
    for k in fresh:
        by_ts[k[0]] = k
    return sorted(by_ts.values(), key=lambda x: x[0])


if __name__ == "__main__":
    # Smoke test : fetche 10 bougies recentes sur tous les TF
    print("Fetching warmup window (10 15m + 5 MTF each)...")
    result = fetch_warmup_window(10, 5)
    for tf, candles in result.items():
        latest_ts = candles[-1][0] if candles else None
        print(f"  {tf}: {len(candles)} candles, latest open_time_ms = {latest_ts}")
```

- [ ] **Step 2: Smoke-test**

Run:
```powershell
cd live
python fetch.py
cd ..
```

Expected (les open_time_ms doivent être récents : un timestamp ms d'aujourd'hui ou hier) :
```
Fetching warmup window (10 15m + 5 MTF each)...
  15m: 10 candles, latest open_time_ms = 1746...
  30m: 5 candles, latest open_time_ms = 1746...
  1h: 5 candles, latest open_time_ms = 1746...
  4h: 5 candles, latest open_time_ms = 1746...
  1d: 5 candles, latest open_time_ms = 1746...
```

Si le fetch échoue (réseau / Bitstamp down), retry avant de débugger plus profond.

- [ ] **Step 3: Commit**

```bash
git add live/fetch.py
git commit -m "feat(live): bitstamp fetch wrapper for warmup and recent windows"
```

---

### Task 3: `live/features.py` — détection setups + extraction features (mirror live de `extract_features`)

C'est la task la plus dense. Cette fonction reproduit la logique de `backtest/data.py:extract_features` (lignes 211-371) **mais uniquement pour les positions où le streak SE TERMINE à la dernière bougie 15m fermée** (`last_idx == n - 1`). Pas de `label` (la bougie cible n'est pas encore close).

**Convention** : si `extract_features` change dans `backtest/data.py` (ajout d'une feature, etc.), cette fonction DOIT être mise à jour en miroir. Un commentaire en tête le rappelle.

**Files:**
- Create: `live/features.py`

- [ ] **Step 1: Écrire `live/features.py`**

Create `live/features.py` with:

```python
"""Detection des setups qualifiants au dernier index 15m + extraction des 42
features attendues par Aure.

MIRROR de backtest/data.py:extract_features (lignes 211-371). Si tu ajoutes
une feature dans extract_features, ajoute-la ici aussi. Pas de label calcule
ici (la bougie cible n'est pas encore fermee au moment de la decision).
"""
import os
import sys
import time

_BACKTEST_DIR = os.path.join(os.path.dirname(__file__), "..", "backtest")
sys.path.insert(0, os.path.abspath(_BACKTEST_DIR))

from data import (  # noqa: E402
    compute_obv, compute_ema, compute_rsi, compute_atr, compute_volume_avg,
    body_ratio, upper_wick_ratio, lower_wick_ratio, get_direction,
    last_closed_index, _index_mtf,
)
from config import (  # noqa: E402
    INTERVAL_MS, MTF_INTERVALS, MAX_SEQ_LEN, BODY_RATIO_MIN, WARMUP,
    EMA_CLOSE_PERIODS, RSI_PERIOD, ATR_PERIOD, ATR_LONG_PERIOD,
    VOL_AVG_PERIOD, OBV_SLOPE_WINDOW, RSI_CHANGE_WINDOW, EMA_OBV_PERIOD,
    VOLATILITY_HIGH, VOLATILITY_LOW,
)


def find_setups_at_latest(klines_15m: list, klines_mtf: dict) -> list:
    """Trouve tous les setups qualifiants dont le streak se termine a la
    derniere bougie 15m fermee (klines_15m[-1]).

    Args:
        klines_15m : liste triee de [open_time_ms, o, h, l, c, v]. Doit
                     contenir au moins WARMUP + MAX_SEQ_LEN bougies pour
                     stabiliser les indicateurs.
        klines_mtf : dict {iv: liste klines} pour iv in MTF_INTERVALS.

    Returns:
        list de tuples (seq_len, features_dict). features_dict contient les
        42 features attendues par Aure (sans timestamp ni label). Liste vide
        si aucun setup ne qualifie au dernier index.
    """
    n = len(klines_15m)
    if n < WARMUP + MAX_SEQ_LEN:
        return []

    # Decompose klines en arrays (mirror de extract_features)
    times = [k[0] for k in klines_15m]
    opens = [k[1] for k in klines_15m]
    highs = [k[2] for k in klines_15m]
    lows = [k[3] for k in klines_15m]
    closes = [k[4] for k in klines_15m]
    volumes = [k[5] for k in klines_15m]
    directions = [get_direction(o, c) for o, c in zip(opens, closes)]

    ema_p_short, ema_p_mid, ema_p_long = EMA_CLOSE_PERIODS

    # Indicateurs (mirror de extract_features)
    obv = compute_obv(closes, volumes)
    ema_obv_20 = compute_ema(obv, EMA_OBV_PERIOD)
    ema_close_20 = compute_ema(closes, ema_p_short)
    ema_close_50 = compute_ema(closes, ema_p_mid)
    ema_close_200 = compute_ema(closes, ema_p_long)
    rsi_14 = compute_rsi(closes, RSI_PERIOD)
    atr_14 = compute_atr(highs, lows, closes, ATR_PERIOD)
    atr_100_avg = compute_atr(highs, lows, closes, ATR_LONG_PERIOD)
    vol_avg_20 = compute_volume_avg(volumes, VOL_AVG_PERIOD)

    mtf = {iv: _index_mtf(kl) for iv, kl in klines_mtf.items()}

    interval_15m_ms = INTERVAL_MS["15m"]
    last_idx = n - 1
    setups = []

    for seq_len in range(1, MAX_SEQ_LEN + 1):
        i = n - seq_len  # streak de i a last_idx
        if i < WARMUP:
            continue

        window = directions[i:i + seq_len]
        if "flat" in window:
            continue
        if len(set(window)) != 1:
            continue

        seq_dir = window[0]

        br_last = body_ratio(opens[last_idx], highs[last_idx], lows[last_idx], closes[last_idx])
        if br_last < BODY_RATIO_MIN:
            continue

        # Decision time T (mirror de extract_features)
        T = times[last_idx] + interval_15m_ms

        # MTF features (no-lookahead via last_closed_index)
        mtf_features = {}
        mtf_ok = True
        for iv in MTF_INTERVALS:
            d = mtf[iv]
            idx = last_closed_index(d["times"], T, INTERVAL_MS[iv])
            if idx < 0:
                mtf_ok = False
                break
            mtf_features[f"open_{iv}"] = d["opens"][idx]
            mtf_features[f"high_{iv}"] = d["highs"][idx]
            mtf_features[f"low_{iv}"] = d["lows"][idx]
            mtf_features[f"close_{iv}"] = d["closes"][idx]
            mtf_features[f"vol_{iv}"] = d["volumes"][idx]

        if not mtf_ok:
            continue

        # Features 15m (mirror exact de extract_features)
        f_seq_dir = 1 if seq_dir == "up" else 0
        f_body_ratio_last = br_last
        f_body_ratio_avg = sum(
            body_ratio(opens[j], highs[j], lows[j], closes[j])
            for j in range(i, i + seq_len)
        ) / seq_len
        f_upper_wick = upper_wick_ratio(opens[last_idx], highs[last_idx], lows[last_idx], closes[last_idx])
        f_lower_wick = lower_wick_ratio(opens[last_idx], highs[last_idx], lows[last_idx], closes[last_idx])
        f_volume_ratio_last = volumes[last_idx] / vol_avg_20[last_idx] if vol_avg_20[last_idx] > 0 else 1
        f_volume_ratio_avg = (
            sum(volumes[j] for j in range(i, i + seq_len)) / seq_len / vol_avg_20[last_idx]
            if vol_avg_20[last_idx] > 0 else 1
        )
        f_move_pct = (closes[last_idx] - opens[i]) / opens[i] * 100
        f_move_atr = abs(closes[last_idx] - opens[i]) / atr_14[last_idx] if atr_14[last_idx] > 0 else 0
        f_rsi = rsi_14[last_idx]
        f_rsi_change_5 = (
            rsi_14[last_idx] - rsi_14[last_idx - RSI_CHANGE_WINDOW]
            if last_idx >= RSI_CHANGE_WINDOW else 0
        )
        f_obv_slope_10 = (
            (obv[last_idx] - obv[last_idx - OBV_SLOPE_WINDOW]) / abs(obv[last_idx - OBV_SLOPE_WINDOW])
            if abs(obv[last_idx - OBV_SLOPE_WINDOW]) > 1e-9 else 0
        )
        f_obv_vs_ema = 1 if obv[last_idx] > ema_obv_20[last_idx] else 0
        f_atr_ratio = atr_14[last_idx] / atr_100_avg[last_idx] if atr_100_avg[last_idx] > 0 else 1
        if f_atr_ratio > VOLATILITY_HIGH:
            f_volatility_state = 2
        elif f_atr_ratio < VOLATILITY_LOW:
            f_volatility_state = 0
        else:
            f_volatility_state = 1

        price = closes[last_idx]
        f_dist_ema20 = (price - ema_close_20[last_idx]) / ema_close_20[last_idx] * 100
        f_dist_ema50 = (price - ema_close_50[last_idx]) / ema_close_50[last_idx] * 100
        f_dist_ema200 = (price - ema_close_200[last_idx]) / ema_close_200[last_idx] * 100

        ts_seconds = times[last_idx] // 1000
        tm = time.gmtime(ts_seconds)
        f_hour_utc = tm.tm_hour
        f_day_of_week = tm.tm_wday
        f_is_weekend = 1 if f_day_of_week >= 5 else 0

        features = {
            "seq_len": seq_len,
            "seq_dir": f_seq_dir,
            "body_ratio_last": f_body_ratio_last,
            "body_ratio_avg": f_body_ratio_avg,
            "upper_wick": f_upper_wick,
            "lower_wick": f_lower_wick,
            "volume_ratio_last": f_volume_ratio_last,
            "volume_ratio_avg": f_volume_ratio_avg,
            "move_pct": f_move_pct,
            "move_atr": f_move_atr,
            "rsi": f_rsi,
            "rsi_change_5": f_rsi_change_5,
            "obv_slope_10": f_obv_slope_10,
            "obv_vs_ema": f_obv_vs_ema,
            "atr_ratio": f_atr_ratio,
            "volatility_state": f_volatility_state,
            "dist_ema20": f_dist_ema20,
            "dist_ema50": f_dist_ema50,
            "dist_ema200": f_dist_ema200,
            "hour_utc": f_hour_utc,
            "day_of_week": f_day_of_week,
            "is_weekend": f_is_weekend,
        }
        features.update(mtf_features)
        # Note: timestamp = times[last_idx] (open_time_ms de la derniere bougie
        # du streak), coherent avec extract_features. target_open_time = T.
        features["timestamp"] = times[last_idx]
        features["target_open_time"] = T

        setups.append((seq_len, features))

    return setups


if __name__ == "__main__":
    # Smoke-test : fetche les 250 dernieres bougies 15m + 50 MTF, appelle
    # find_setups_at_latest, et imprime le nombre de setups trouves.
    from fetch import fetch_warmup_window
    print("Fetching warmup window (250 15m + 50 MTF)...")
    klines = fetch_warmup_window(250, 50)
    print(f"  fetched: {sum(len(v) for v in klines.values())} candles total")
    klines_15m = klines.pop("15m")
    setups = find_setups_at_latest(klines_15m, klines)
    print(f"Setups qualifiants au dernier index : {len(setups)}")
    for seq_len, feat in setups:
        print(f"  seq_len={seq_len}, seq_dir={feat['seq_dir']}, "
              f"body_ratio_last={feat['body_ratio_last']:.3f}, "
              f"timestamp={feat['timestamp']}, target_open_time={feat['target_open_time']}")
```

- [ ] **Step 2: Smoke-test**

Run:
```powershell
cd live
python features.py
cd ..
```

Expected output (le nombre de setups varie 0-4 selon le marché à l'instant T) :
```
Fetching warmup window (250 15m + 50 MTF)...
  fetched: ~450 candles total
Setups qualifiants au dernier index : 0
```
ou (si streak qualifiant en cours) :
```
Setups qualifiants au dernier index : 2
  seq_len=2, seq_dir=1, body_ratio_last=0.745, timestamp=174..., target_open_time=174...
  seq_len=3, seq_dir=1, body_ratio_last=0.745, timestamp=174..., target_open_time=174...
```

Vérifier visuellement : `target_open_time == timestamp + 900_000` (15min en ms) pour chaque setup retourné.

- [ ] **Step 3: Commit**

```bash
git add live/features.py
git commit -m "feat(live): live setup detection mirroring extract_features"
```

---

### Task 4: `live/store.py` — I/O CSV (training_data_live + decisions)

Trois responsabilités :
1. **Append** une ligne dans `training_data_live.csv` (shape 44 cols, label=NaN au moment du write).
2. **Append** une ligne dans `decisions.csv` (shape 53 cols).
3. **Update rétroactif** : `update_pending_labels(klines_15m)` et `update_pending_outcomes(klines_15m)` qui réécrivent les CSVs en remplaçant les NaN/`pending` quand la bougie cible est connue.

Toutes les writes utilisent l'idiome `to_csv(tmp); os.replace(tmp, real)` pour atomicité. Header écrit seulement la première fois.

**Files:**
- Create: `live/store.py`

- [ ] **Step 1: Écrire `live/store.py`**

Create `live/store.py` with:

```python
"""I/O CSV pour l'orchestrateur live.

Deux fichiers geres :
- training_data_live.csv : 44 cols, mirror du schema de backtest/training_data.csv
- decisions.csv : 53 cols, audit self-contained des decisions Aure

Updates retroactifs (label, outcome) atomiques via tmp + os.replace.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from settings import TRAINING_DATA_LIVE, DECISIONS_CSV, INTERVAL_15M_MS

# Ordre des colonnes (figer pour rester compatible avec backtest/training_data.csv).
TRAINING_DATA_COLUMNS = [
    "timestamp", "seq_len", "seq_dir", "body_ratio_last", "body_ratio_avg",
    "upper_wick", "lower_wick", "volume_ratio_last", "volume_ratio_avg",
    "move_pct", "move_atr", "rsi", "rsi_change_5", "obv_slope_10",
    "obv_vs_ema", "atr_ratio", "volatility_state", "dist_ema20", "dist_ema50",
    "dist_ema200", "hour_utc", "day_of_week", "is_weekend",
    "open_30m", "high_30m", "low_30m", "close_30m", "vol_30m",
    "open_1h", "high_1h", "low_1h", "close_1h", "vol_1h",
    "open_4h", "high_4h", "low_4h", "close_4h", "vol_4h",
    "open_1d", "high_1d", "low_1d", "close_1d", "vol_1d",
    "label",
]

# 42 features = TRAINING_DATA_COLUMNS sans timestamp et sans label.
FEATURE_COLUMNS = [c for c in TRAINING_DATA_COLUMNS if c not in ("timestamp", "label")]

DECISION_COLUMNS = (
    ["timestamp"]
    + FEATURE_COLUMNS
    + [
        "confidence", "effective_threshold", "bucket_v", "bucket_w",
        "bet_direction", "taken", "notified_at", "target_open_time",
        "outcome", "outcome_resolved_at",
    ]
)


def _utc_iso_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _atomic_write(df: pd.DataFrame, path: Path) -> None:
    """Write df to path atomiquement via tmp + os.replace.

    Necessaire pour les updates retroactifs (label, outcome) qui rewrite
    le fichier entier ; un crash a mi-write laisserait sinon un CSV tronque.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def append_training_row(features: dict) -> None:
    """Append une ligne dans training_data_live.csv. label=NaN au depart.

    features doit contenir 'timestamp' + les 42 features. La fonction
    rejette les autres clefs et reordonne selon TRAINING_DATA_COLUMNS.
    """
    row = {col: features.get(col) for col in TRAINING_DATA_COLUMNS}
    row["label"] = pd.NA  # marquera label comme nullable
    df = pd.DataFrame([row], columns=TRAINING_DATA_COLUMNS)
    _append_with_header_if_missing(df, TRAINING_DATA_LIVE)


def append_decision_row(features: dict, decision: dict) -> None:
    """Append une ligne dans decisions.csv.

    decision doit contenir : confidence, effective_threshold, bucket_v,
    bucket_w, bet_direction, taken, notified_at, target_open_time. outcome
    est mis a 'pending' et outcome_resolved_at a NaN par cette fonction.
    """
    row = {col: features.get(col) for col in FEATURE_COLUMNS}
    row["timestamp"] = features["timestamp"]
    row["target_open_time"] = features["target_open_time"]
    row.update({
        "confidence": decision["confidence"],
        "effective_threshold": decision["effective_threshold"],
        "bucket_v": decision["bucket_v"],
        "bucket_w": decision["bucket_w"],
        "bet_direction": decision["bet_direction"],
        "taken": decision["taken"],
        "notified_at": decision.get("notified_at"),
        "outcome": "pending",
        "outcome_resolved_at": None,
    })
    df = pd.DataFrame([row], columns=DECISION_COLUMNS)
    _append_with_header_if_missing(df, DECISIONS_CSV)


def _append_with_header_if_missing(df: pd.DataFrame, path: Path) -> None:
    """Append df a path. Ecrit le header seulement si le fichier n'existe pas."""
    if path.exists():
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        df.to_csv(path, mode="w", header=True, index=False)


def update_pending_labels(klines_15m: list) -> int:
    """Resoud les rows de training_data_live.csv ou label est NaN.

    Pour chaque row pendante :
      - target_open_time = timestamp + 15min
      - cherche la bougie 15m d'open_time == target_open_time dans klines_15m
      - si trouvee : label = 1 si direction inverse seq_dir, 0 sinon
      - si pas trouvee (target pas encore close) : laisse NaN

    Returns: nombre de rows mises a jour.
    """
    if not TRAINING_DATA_LIVE.exists():
        return 0
    df = pd.read_csv(TRAINING_DATA_LIVE)
    if df.empty or "label" not in df.columns:
        return 0

    pending_mask = df["label"].isna()
    if not pending_mask.any():
        return 0

    candle_by_ts = {k[0]: k for k in klines_15m}
    updated = 0

    for idx in df.index[pending_mask]:
        ts = int(df.at[idx, "timestamp"])
        target_ts = ts + INTERVAL_15M_MS
        candle = candle_by_ts.get(target_ts)
        if candle is None:
            continue
        # candle = [open_time, o, h, l, c, v]
        next_dir_up = candle[4] > candle[1]   # close > open
        next_dir_down = candle[4] < candle[1]  # close < open
        seq_dir_up = bool(df.at[idx, "seq_dir"])

        if seq_dir_up and next_dir_down:
            label = 1
        elif (not seq_dir_up) and next_dir_up:
            label = 1
        else:
            label = 0  # flat (close==open) inclus dans loss

        df.at[idx, "label"] = label
        updated += 1

    if updated:
        # Restaure le type int sur label (pandas peut avoir promu en float a cause des NaN restants).
        # On garde nullable Int64 pour pouvoir retenir des NaN sur les rows non encore resolues.
        df["label"] = df["label"].astype("Int64")
        _atomic_write(df, TRAINING_DATA_LIVE)
    return updated


def update_pending_outcomes(klines_15m: list) -> int:
    """Resoud les rows de decisions.csv ou outcome=='pending'.

    Pour chaque row pendante :
      - cherche la bougie 15m d'open_time == target_open_time
      - si trouvee : outcome = 'win' si direction == bet_direction, 'loss' sinon
                     outcome_resolved_at = utc iso now
      - sinon : laisse 'pending'

    Returns: nombre de rows mises a jour.
    """
    if not DECISIONS_CSV.exists():
        return 0
    df = pd.read_csv(DECISIONS_CSV)
    if df.empty or "outcome" not in df.columns:
        return 0

    pending_mask = df["outcome"] == "pending"
    if not pending_mask.any():
        return 0

    candle_by_ts = {k[0]: k for k in klines_15m}
    now_iso = _utc_iso_now()
    updated = 0

    for idx in df.index[pending_mask]:
        target_ts = int(df.at[idx, "target_open_time"])
        candle = candle_by_ts.get(target_ts)
        if candle is None:
            continue

        bet_direction = df.at[idx, "bet_direction"]
        candle_dir = "up" if candle[4] > candle[1] else ("down" if candle[4] < candle[1] else "flat")

        if candle_dir == bet_direction:
            outcome = "win"
        else:
            outcome = "loss"  # flat inclus dans loss (coherent avec extract_features)

        df.at[idx, "outcome"] = outcome
        df.at[idx, "outcome_resolved_at"] = now_iso
        updated += 1

    if updated:
        _atomic_write(df, DECISIONS_CSV)
    return updated


if __name__ == "__main__":
    # Smoke test : append une row factice dans chaque CSV puis affiche
    # les CSV pour visuel.
    print("Smoke test store.py")
    fake_features = {col: 0.0 for col in FEATURE_COLUMNS}
    fake_features.update({"timestamp": 1746000000000, "seq_len": 2, "seq_dir": 1,
                          "body_ratio_last": 0.7, "is_weekend": 0, "volatility_state": 1,
                          "target_open_time": 1746000000000 + INTERVAL_15M_MS})
    print("  Appending training row...")
    append_training_row(fake_features)
    print("  Appending decision row...")
    append_decision_row(fake_features, {
        "confidence": 0.65, "effective_threshold": 0.58,
        "bucket_v": 1, "bucket_w": 0, "bet_direction": "down",
        "taken": True, "notified_at": _utc_iso_now(),
    })
    print(f"  training_data_live.csv : {TRAINING_DATA_LIVE.stat().st_size} bytes")
    print(f"  decisions.csv : {DECISIONS_CSV.stat().st_size} bytes")
    print("  Done. Inspect CSVs manually if needed.")
```

- [ ] **Step 2: Smoke-test**

Run:
```powershell
cd live
python store.py
cd ..
```

Expected:
```
Smoke test store.py
  Appending training row...
  Appending decision row...
  training_data_live.csv : ~XXX bytes
  decisions.csv : ~XXX bytes
  Done. Inspect CSVs manually if needed.
```

Inspect:
```powershell
Get-Content live\logs\training_data_live.csv | Select-Object -First 3
Get-Content live\logs\decisions.csv | Select-Object -First 3
```

Vérifier visuellement :
- `training_data_live.csv` a 1 row de header (44 cols) + 1 row de data avec label vide.
- `decisions.csv` a 1 row de header (53 cols) + 1 row de data avec `outcome=pending`, `outcome_resolved_at` vide.

- [ ] **Step 3: Cleanup les CSVs de smoke-test**

Run:
```powershell
Remove-Item live\logs\training_data_live.csv, live\logs\decisions.csv -ErrorAction SilentlyContinue
```

(On laisse les fichiers vides pour le run réel.)

- [ ] **Step 4: Commit**

```bash
git add live/store.py
git commit -m "feat(live): csv store with append + retroactive label/outcome updates"
```

---

### Task 5: `live/notify.py` — Telegram client

Client Telegram minimal : un seul appel `send_message(text: str) -> bool`. Lit token + chat_id depuis env vars au load. Best-effort : ne raise jamais, retourne `False` en cas d'échec.

**Files:**
- Create: `live/notify.py`

- [ ] **Step 1: Écrire `live/notify.py`**

Create `live/notify.py` with:

```python
"""Client Telegram pour les notifs Aure.

Best-effort : ne raise jamais. Retourne True si la notif est partie, False sinon.
"""
import logging
import os
import time

import requests

from settings import (
    TELEGRAM_BOT_TOKEN_ENV,
    TELEGRAM_CHAT_ID_ENV,
    TELEGRAM_TIMEOUT,
)

_log = logging.getLogger(__name__)

# Etat global :
# - _bot_token, _chat_id : lus a chaque appel pour permettre des changements
#   d'env vars sans relancer le daemon (rare mais utile en debug).
# - _disabled : passe a True si une 4xx d'auth survient, pour eviter de
#   spammer Telegram qui rejettera tout. Reset au prochain restart.
_disabled = False


def _read_env() -> tuple[str | None, str | None]:
    return os.environ.get(TELEGRAM_BOT_TOKEN_ENV), os.environ.get(TELEGRAM_CHAT_ID_ENV)


def is_configured() -> bool:
    token, chat_id = _read_env()
    return bool(token and chat_id) and not _disabled


def send_message(text: str) -> bool:
    """Envoie text via sendMessage (parse_mode=HTML). Retourne True si OK.

    Si bot token / chat_id sont missing : log WARN une fois, retourne False
    sans faire d'appel reseau.
    Si 4xx d'auth : log ERROR, set _disabled=True, retourne False.
    Si timeout / 5xx : retry 1x, puis log WARN et retourne False.
    """
    global _disabled
    if _disabled:
        return False

    token, chat_id = _read_env()
    if not token or not chat_id:
        _log.warning("telegram_disabled (missing %s or %s env var)",
                     TELEGRAM_BOT_TOKEN_ENV, TELEGRAM_CHAT_ID_ENV)
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text, "parse_mode": "HTML"}

    for attempt in (1, 2):
        try:
            resp = requests.post(url, json=payload, timeout=TELEGRAM_TIMEOUT)
        except (requests.Timeout, requests.ConnectionError) as e:
            if attempt == 1:
                time.sleep(2)
                continue
            _log.warning("telegram_failed network attempt=%d err=%s", attempt, e)
            return False

        if resp.status_code == 200:
            return True
        if 400 <= resp.status_code < 500:
            _log.error("telegram_misconfigured status=%d body=%s — disabling notif until restart",
                       resp.status_code, resp.text[:200])
            _disabled = True
            return False
        # 5xx : retry une fois
        if attempt == 1:
            time.sleep(2)
            continue
        _log.warning("telegram_failed status=%d", resp.status_code)
        return False

    return False


def format_decision_message(features: dict, decision: dict, price: float) -> str:
    """Genere le HTML d'une notif Aure.

    features : dict avec seq_len, seq_dir (1=up, 0=down), volatility_state, is_weekend.
    decision : dict avec bet_direction ('up'|'down'), confidence, effective_threshold.
    price : prix BTC au moment du setup (= close de la derniere bougie du streak).
    """
    bet = decision["bet_direction"]
    emoji = "🟢" if bet == "up" else "🔴"
    action = "BUY" if bet == "up" else "SELL"

    streak_dir_word = "up" if features["seq_dir"] == 1 else "down"
    vol_word = {0: "low", 1: "normal", 2: "high"}.get(features["volatility_state"], "?")
    weekend_word = "yes" if features["is_weekend"] == 1 else "no"

    timestamp_ms = features["timestamp"]
    from datetime import datetime, timezone
    ts_str = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    return (
        f"{emoji} <b>{action} BTC</b>\n"
        f"<i>Aure setup</i>\n"
        f"Price : <b>${price:,.0f}</b>\n"
        f"Confidence : {decision['confidence']*100:.1f}% (th {decision['effective_threshold']*100:.1f}%)\n"
        f"Streak : {features['seq_len']} candles {streak_dir_word}\n"
        f"Bucket : vol={vol_word} weekend={weekend_word}\n"
        f"Time : {ts_str}"
    )


if __name__ == "__main__":
    # Smoke test : verifie format_decision_message sur un setup factice,
    # ET tente un send_message si les env vars sont presentes.
    fake_features = {
        "timestamp": 1746000000000, "seq_len": 3, "seq_dir": 0,
        "volatility_state": 1, "is_weekend": 0,
    }
    fake_decision = {
        "bet_direction": "up", "confidence": 0.624, "effective_threshold": 0.580,
    }
    msg = format_decision_message(fake_features, fake_decision, price=67234.0)
    print("--- formatted message ---")
    print(msg)
    print("-------------------------")
    if is_configured():
        print("Telegram configured. Sending...")
        ok = send_message("(smoke test) " + msg)
        print(f"  send_message returned: {ok}")
    else:
        print("Telegram not configured (env vars manquantes). Skipping send.")
```

- [ ] **Step 2: Smoke-test sans env vars (format only)**

Run:
```powershell
cd live
python notify.py
cd ..
```

Expected (env vars Telegram pas set) :
```
--- formatted message ---
🟢 BUY BTC
Aure setup
Price : $67,234
Confidence : 62.4% (th 58.0%)
Streak : 3 candles down
Bucket : vol=normal weekend=no
Time : 2026-04-30 06:40 UTC
-------------------------
Telegram not configured (env vars manquantes). Skipping send.
```

(le `<b>...</b>` apparaît brut puisque c'est juste un print Python ; c'est normal.)

- [ ] **Step 3: Smoke-test AVEC env vars (optionnel, à faire si tu as un bot)**

Run (PowerShell, remplacer par tes vraies valeurs) :
```powershell
$env:AURE_TELEGRAM_BOT_TOKEN = "123:abc..."
$env:AURE_TELEGRAM_CHAT_ID = "-100..."
cd live
python notify.py
cd ..
```

Expected: `send_message returned: True` + un message reçu côté Telegram. Si retourne `False`, vérifier le log pour la raison (bad token / chat_id, network).

- [ ] **Step 4: Commit**

```bash
git add live/notify.py
git commit -m "feat(live): telegram client with HTML message formatter"
```

---

### Task 6: `live/orchestrator.py` — daemon principal

C'est l'orchestrateur. Responsabilités :
1. Charger `aure.json` + sidecar (copier de backtest si manquant).
2. Fetch warmup au démarrage.
3. Résoudre les pendings au démarrage (label + outcome).
4. Boucle principale : sleep until next 15m boundary → run cycle → repeat.
5. Cycle : fetch recent → resolve pendings → find setups → predict → archive → notify.

**Files:**
- Create: `live/orchestrator.py`

- [ ] **Step 1: Écrire `live/orchestrator.py`**

Create `live/orchestrator.py` with:

```python
"""Daemon principal de l'orchestrateur Aure live."""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import xgboost as xgb

from settings import (
    MODEL_PATH, THRESHOLDS_PATH, THRESHOLDS_SOURCE,
    LOGS_DIR, ORCHESTRATOR_LOG,
    WARMUP_CANDLES_15M, WARMUP_CANDLES_MTF,
    RECENT_FETCH_15M, RECENT_FETCH_MTF,
    INTERVAL_15M_MS, CYCLE_FUDGE_SECONDS,
)
from fetch import fetch_warmup_window, fetch_recent_window, merge_klines
from features import find_setups_at_latest
from store import (
    append_training_row, append_decision_row,
    update_pending_labels, update_pending_outcomes,
)
from notify import send_message, format_decision_message

_log = logging.getLogger("orchestrator")


# ---------------------------------------------------------------------------
# Setup logging
# ---------------------------------------------------------------------------
def _setup_logging() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)sZ %(levelname)s  %(message)s"
    datefmt = "%Y-%m-%dT%H:%M:%S"
    file_handler = logging.FileHandler(ORCHESTRATOR_LOG, encoding="utf-8")
    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter(fmt, datefmt=datefmt)
    formatter.converter = time.gmtime
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers = [file_handler, stream_handler]


# ---------------------------------------------------------------------------
# Model + sidecar loading
# ---------------------------------------------------------------------------
def _ensure_sidecar() -> None:
    """Copie backtest/btc_reversal_model.thresholds.json -> aure.thresholds.json
    si ce dernier n'existe pas. Idempotent."""
    if THRESHOLDS_PATH.exists():
        return
    if not THRESHOLDS_SOURCE.exists():
        _log.warning("sidecar_missing source=%s — Aure utilisera un seuil fixe (fallback)",
                     THRESHOLDS_SOURCE)
        return
    shutil.copy2(THRESHOLDS_SOURCE, THRESHOLDS_PATH)
    _log.info("sidecar_copied %s -> %s", THRESHOLDS_SOURCE.name, THRESHOLDS_PATH.name)


def _load_model() -> xgb.XGBClassifier:
    model = xgb.XGBClassifier()
    model.load_model(str(MODEL_PATH))
    _log.info("model_loaded %s features=%d", MODEL_PATH.name,
              len(model.get_booster().feature_names))
    return model


def _load_sidecar() -> dict:
    """Returns {'base_threshold': float, 'deltas': dict[str, float]}.

    Si pas de sidecar : fallback {'base_threshold': 0.58, 'deltas': {}}.
    """
    if not THRESHOLDS_PATH.exists():
        return {"base_threshold": 0.58, "deltas": {}}
    with open(THRESHOLDS_PATH) as f:
        data = json.load(f)
    return {
        "base_threshold": float(data["base_threshold"]),
        "deltas": {k: float(v) for k, v in data.get("deltas", {}).items()},
    }


# ---------------------------------------------------------------------------
# Sleep alignment
# ---------------------------------------------------------------------------
def next_boundary_ms(now_ms: int) -> int:
    """Retourne le prochain open_time_ms 15m strictement apres now_ms."""
    return ((now_ms // INTERVAL_15M_MS) + 1) * INTERVAL_15M_MS


def sleep_until(target_ms: int) -> None:
    while True:
        delta = target_ms / 1000 - time.time()
        if delta <= 0:
            return
        time.sleep(min(delta, 30))  # capping a 30s pour eviter sleep trop long


# ---------------------------------------------------------------------------
# Predict + decide
# ---------------------------------------------------------------------------
def _predict_and_decide(model: xgb.XGBClassifier, sidecar: dict, features: dict) -> dict:
    """Calcule confidence, effective_threshold, taken, bet_direction.

    Returns dict utilise par store.append_decision_row + notify.format_decision_message.
    """
    expected_cols = list(model.get_booster().feature_names)
    df = pd.DataFrame([{col: features[col] for col in expected_cols}])
    confidence = float(model.predict_proba(df)[:, 1][0])

    bucket_v = int(features["volatility_state"])
    bucket_w = int(features["is_weekend"])
    bucket_key = f"{bucket_v}_{bucket_w}"
    delta = sidecar["deltas"].get(bucket_key, 0.0)
    effective_threshold = sidecar["base_threshold"] + delta

    bet_direction = "down" if features["seq_dir"] == 1 else "up"
    taken = confidence >= effective_threshold

    return {
        "confidence": confidence,
        "effective_threshold": effective_threshold,
        "bucket_v": bucket_v,
        "bucket_w": bucket_w,
        "bet_direction": bet_direction,
        "taken": taken,
    }


# ---------------------------------------------------------------------------
# Run cycle
# ---------------------------------------------------------------------------
def run_cycle(model, sidecar, klines_state: dict, no_notify: bool) -> dict:
    """Execute un cycle complet. Mute klines_state in place avec les nouvelles bougies.

    Returns: stats du cycle (dict) pour log.
    """
    cycle_start = time.time()

    # 1. Fetch recent
    fresh = fetch_recent_window(RECENT_FETCH_15M, RECENT_FETCH_MTF)
    for tf, candles in fresh.items():
        klines_state[tf] = merge_klines(klines_state.get(tf, []), candles)

    counts = {tf: len(klines_state[tf]) for tf in klines_state}
    _log.info("fetched candles_15m=%d candles_30m=%d candles_1h=%d candles_4h=%d candles_1d=%d",
              len(fresh.get("15m", [])), len(fresh.get("30m", [])),
              len(fresh.get("1h", [])), len(fresh.get("4h", [])),
              len(fresh.get("1d", [])))

    # 2. Resolve pendings (avec les klines_15m mises a jour)
    n_labels = update_pending_labels(klines_state["15m"])
    n_outcomes = update_pending_outcomes(klines_state["15m"])
    if n_labels or n_outcomes:
        _log.info("pendings_resolved labels=%d outcomes=%d", n_labels, n_outcomes)

    # 3. Detect setups
    klines_15m = klines_state["15m"]
    klines_mtf = {tf: klines_state[tf] for tf in klines_state if tf != "15m"}
    setups = find_setups_at_latest(klines_15m, klines_mtf)
    _log.info("setups_qualifying=%d", len(setups))

    # 4. Pour chaque setup : append + predict + notif
    notifs_sent = 0
    for seq_len, features in setups:
        append_training_row(features)
        decision = _predict_and_decide(model, sidecar, features)

        notified_at = None
        if decision["taken"] and not no_notify:
            price = klines_15m[-1][4]  # close de la derniere bougie
            msg = format_decision_message(features, decision, price=float(price))
            ok = send_message(msg)
            if ok:
                notified_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                notifs_sent += 1
        decision["notified_at"] = notified_at

        append_decision_row(features, decision)
        _log.info(
            "decision seq_len=%d confidence=%.3f threshold=%.3f bucket=%d_%d taken=%s direction=%s",
            seq_len, decision["confidence"], decision["effective_threshold"],
            decision["bucket_v"], decision["bucket_w"], decision["taken"],
            decision["bet_direction"],
        )

    duration_ms = int((time.time() - cycle_start) * 1000)
    _log.info("cycle_end duration_ms=%d notifs_sent=%d", duration_ms, notifs_sent)
    return {
        "setups": len(setups),
        "notifs_sent": notifs_sent,
        "duration_ms": duration_ms,
        "labels_resolved": n_labels,
        "outcomes_resolved": n_outcomes,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Daemon Aure live orchestrator")
    parser.add_argument("--dry-run", action="store_true",
                        help="Execute UN seul cycle puis quitte (test rapide).")
    parser.add_argument("--no-notify", action="store_true",
                        help="Ne pas envoyer de notif Telegram (decisions seulement loguees).")
    args = parser.parse_args(argv)

    _setup_logging()
    _log.info("orchestrator_start dry_run=%s no_notify=%s", args.dry_run, args.no_notify)

    _ensure_sidecar()
    model = _load_model()
    sidecar = _load_sidecar()

    # Warmup
    _log.info("fetching_warmup candles_15m=%d candles_mtf=%d", WARMUP_CANDLES_15M, WARMUP_CANDLES_MTF)
    klines_state = fetch_warmup_window(WARMUP_CANDLES_15M, WARMUP_CANDLES_MTF)

    # Resolve startup pendings (avec les klines warmup)
    n_labels = update_pending_labels(klines_state["15m"])
    n_outcomes = update_pending_outcomes(klines_state["15m"])
    if n_labels or n_outcomes:
        _log.info("startup_pendings_resolved labels=%d outcomes=%d", n_labels, n_outcomes)

    if args.dry_run:
        _log.info("dry_run executing single cycle")
        run_cycle(model, sidecar, klines_state, no_notify=args.no_notify)
        _log.info("dry_run done")
        return 0

    # Daemon loop
    while True:
        now_ms = int(time.time() * 1000)
        target_ms = next_boundary_ms(now_ms) + CYCLE_FUDGE_SECONDS * 1000
        target_iso = datetime.fromtimestamp(target_ms / 1000, tz=timezone.utc).strftime("%H:%M:%S UTC")
        _log.info("sleeping_until %s", target_iso)
        try:
            sleep_until(target_ms)
        except KeyboardInterrupt:
            _log.info("orchestrator_stopping (keyboard interrupt)")
            return 0

        _log.info("cycle_start")
        try:
            run_cycle(model, sidecar, klines_state, no_notify=args.no_notify)
        except KeyboardInterrupt:
            _log.info("orchestrator_stopping (keyboard interrupt mid-cycle)")
            return 0
        except Exception as e:
            _log.exception("cycle_failed err=%s", e)
            # On continue : prochain cycle dans 15min.


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Smoke-test --dry-run sans notif**

Run:
```powershell
cd live
python orchestrator.py --dry-run --no-notify
cd ..
```

Expected output (le nombre de setups varie 0-4) :
```
2026-05-07T...Z INFO  orchestrator_start dry_run=True no_notify=True
2026-05-07T...Z INFO  sidecar_copied btc_reversal_model.thresholds.json -> aure.thresholds.json
2026-05-07T...Z INFO  model_loaded aure.json features=42
2026-05-07T...Z INFO  fetching_warmup candles_15m=200 candles_mtf=50
2026-05-07T...Z INFO  dry_run executing single cycle
2026-05-07T...Z INFO  fetched candles_15m=5 candles_30m=3 candles_1h=3 candles_4h=3 candles_1d=3
2026-05-07T...Z INFO  setups_qualifying=N
[ pour chaque setup : 1 ligne 'decision' ]
2026-05-07T...Z INFO  cycle_end duration_ms=XXX notifs_sent=0
2026-05-07T...Z INFO  dry_run done
```

Vérifier post-run :
- `aure.thresholds.json` existe à la racine du repo (`Test-Path ..\aure.thresholds.json` from `live/`).
- Si setups détectés : `live/logs/training_data_live.csv` et `live/logs/decisions.csv` existent et contiennent N rows + header.
- `live/logs/orchestrator.log` contient le même output que stdout.

- [ ] **Step 3: Commit**

```bash
git add live/orchestrator.py
git commit -m "feat(live): main daemon orchestrator with cycle loop"
```

---

### Task 7: Documentation (root CLAUDE.md + live/CLAUDE.md)

**Files:**
- Modify: `CLAUDE.md` (racine)
- Create: `live/CLAUDE.md`

- [ ] **Step 1: Mettre à jour `CLAUDE.md` (racine)**

Replace the `## Repo layout` section to mention `live/`:

```markdown
## Repo layout

This is a monorepo. Code lives in subprojects, **not at the root**.

- `backtest/` — XGBoost binary classifier for "next-candle reversal after a body_ratio-filtered streak" on BTC/USD 15m bars (Bitstamp). The full pipeline (`data.py` → `train.py` → `predict.py`), hyperparameter sweep tooling, and a data-cleaning module. **Read `backtest/CLAUDE.md` before touching anything in there**.
- `live/` — Production orchestrator that runs the trained Aure model (`aure.json` at the repo root) in real conditions: fetches Bitstamp every 15min, predicts, archives to CSVs, sends Telegram notifications. **Read `live/CLAUDE.md` before touching anything in there**.

The root contains repo-wide config (`.gitignore`, this file) and the model artifact (`aure.json` + `aure.thresholds.json` sidecar — created by `live/orchestrator.py` at first run).
```

(remplacer le bullet `backtest/` existant et l'ancienne mention `btc_reversal_model.json` stale, qui n'est plus vraie depuis qu'on a renommé en `aure.json`.)

Aussi mettre à jour le bloc "## Working-directory contract" pour ajouter une note sur `live/` :

```markdown
## Working-directory contract

`backtest/config.py` uses **relative paths** (`CSV_PATH = "training_data.csv"`, `MODEL_OUTPUT = "btc_reversal_model.json"`), so all scripts in `backtest/` must be run with `backtest/` as the cwd:

```powershell
cd backtest
python data.py
python train.py
```

Same convention for `live/`: scripts must run with `live/` as cwd. `live/settings.py` uses `Path(__file__)` for its own paths (so `live/logs/` always lands in `live/`), but it imports `backtest/data.py` via a sys.path hack — to keep that working without subtle bugs, run from `live/`:

```powershell
cd live
python orchestrator.py
```
```

- [ ] **Step 2: Créer `live/CLAUDE.md`**

Create `live/CLAUDE.md` with:

```markdown
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in `live/`.

## Commands

```powershell
# Required env vars for Telegram notifs (sinon le daemon tourne mais ne notifie pas)
$env:AURE_TELEGRAM_BOT_TOKEN = "123:abc..."
$env:AURE_TELEGRAM_CHAT_ID = "-100..."

cd live

# Daemon long-running (mode normal)
python orchestrator.py

# Un seul cycle puis exit (test fonctionnel)
python orchestrator.py --dry-run

# Pas de notif Telegram (test sans spammer le chat)
python orchestrator.py --no-notify

# --dry-run + --no-notify combinables
python orchestrator.py --dry-run --no-notify
```

## Architecture

Daemon Python long-running aligné sur les frontières 15min UTC. Chaque cycle :
1. Sleep until next 15m boundary + 2s fudge.
2. Fetch les 5 dernières bougies 15m + 3 par MTF (30m/1h/4h/1d) Bitstamp.
3. Resolve les rows pendantes (`label=NaN` dans `training_data_live.csv`, `outcome=pending` dans `decisions.csv`) si la bougie cible est maintenant fermée.
4. Détecte les setups qualifiants au dernier index 15m (streak 1-4 same-dir + `body_ratio_last ≥ 0.6`).
5. Pour chaque setup : extrait 42 features, append `training_data_live.csv` (label=NaN), prédit avec Aure, applique le seuil adaptatif du sidecar, append `decisions.csv`, envoie Telegram si `taken=True`.

## Module collision avec backtest/

`live/settings.py` (et **non `config.py`**) parce qu'on insère `backtest/` dans `sys.path` pour réutiliser `backtest/data.py`, ce qui shadowerait `live/config.py` aux imports `from config import ...` dans le reste de `live/`. Voir le commentaire en tête de `live/settings.py`.

## Files produits

- `live/logs/training_data_live.csv` — 44 cols, schema identique à `backtest/training_data.csv`. Réutilisable pour ré-entraîner Aure (drop pas si tu veux).
- `live/logs/decisions.csv` — 53 cols, audit self-contained des décisions (features + confidence + threshold + outcome).
- `live/logs/orchestrator.log` — log texte un événement par ligne. Lisible en `Get-Content -Wait`.

## Sidecar threshold

`aure.thresholds.json` à la racine est créé au 1er run en copiant `backtest/btc_reversal_model.thresholds.json`. Si tu réentraînes Aure, copie le nouveau sidecar à la main au-dessus, ou supprime `aure.thresholds.json` pour forcer la recopie au prochain démarrage. Le daemon lit le sidecar UNE FOIS au démarrage (cache mémoire) ; pas de hot reload.

## Restart / outage

Au démarrage le daemon fetch ~200 dernières bougies 15m + 50 MTF. Pendings dont la bougie cible est dans cette fenêtre sont résolus automatiquement. Au-delà (daemon down >50h) les pendings restent `pending`/`NaN` indéfiniment — relancer `backtest/data.py` pour régénérer si besoin.
```

- [ ] **Step 3: Smoke-vérifier les docs**

Run:
```powershell
Get-Content CLAUDE.md | Select-String "live/"
```

Expected: au moins 2 matches (un dans le layout, un dans working-directory contract).

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md live/CLAUDE.md
git commit -m "docs: add live subproject to CLAUDE.md (root + live/)"
```

---

### Task 8: End-to-end smoke test + correction des bugs émergeants

C'est la task de validation finale. On enchaîne plusieurs scénarios pour vérifier que tout marche ensemble.

**Files:** aucun nouveau fichier ; corrections inline si bugs trouvés.

- [ ] **Step 1: Cleanup état précédent**

Run:
```powershell
Remove-Item live\logs\*.csv, live\logs\*.log, ..\aure.thresholds.json -ErrorAction SilentlyContinue
```

(Les CSVs et le sidecar seront recréés au prochain run.)

- [ ] **Step 2: Run 1 — premier dry-run**

Run:
```powershell
cd live
python orchestrator.py --dry-run --no-notify
cd ..
```

Vérifier :
- `aure.thresholds.json` créé à la racine (`Test-Path aure.thresholds.json` → True).
- `live/logs/orchestrator.log` créé.
- Si setups détectés (`setups_qualifying=N>0`) : `training_data_live.csv` et `decisions.csv` créés avec N rows + header.

- [ ] **Step 3: Run 2 — vérifier idempotence du sidecar copy**

Run:
```powershell
cd live
python orchestrator.py --dry-run --no-notify
cd ..
```

Vérifier dans le log : pas de nouvelle ligne `sidecar_copied` (le sidecar existe déjà, on skip).

- [ ] **Step 4: Vérifier le CSV training_data_live**

```powershell
Get-Content live\logs\training_data_live.csv | Select-Object -First 3
```

Vérifier visuellement :
- Header = 44 cols (timestamp, seq_len, ..., open_30m, ..., vol_1d, label).
- Au moins 1 row de data (si les Run 1 ou 2 ont détecté des setups). `label` est vide à la fin.
- Si pas de setup : seulement le header.

- [ ] **Step 5: Vérifier le CSV decisions**

```powershell
Get-Content live\logs\decisions.csv | Select-Object -First 3
```

Vérifier visuellement :
- Header = 53 cols (timestamp, seq_len, ..., outcome, outcome_resolved_at).
- Si rows : `outcome=pending`, `outcome_resolved_at` vide, `notified_at` vide (puisque `--no-notify`).

- [ ] **Step 6: Run 3 — daemon mode (interrompre après 1 cycle)**

Run (en mode daemon mais on l'interrompt à la main avec Ctrl+C après le premier cycle, qui devrait tomber dans les 15min suivantes) :
```powershell
cd live
python orchestrator.py --no-notify
# Attendre la prochaine frontière 15min UTC (xx:00, xx:15, xx:30, xx:45 + 2s)
# Au cycle execute, voir 'cycle_start ... cycle_end' dans le log
# Ctrl+C pour quitter
cd ..
```

Vérifier :
- Au démarrage : log `sleeping_until HH:MM:SS UTC` avec une heure ≤ 15min dans le futur.
- À la frontière : log `cycle_start ... fetched ... setups_qualifying=N ... cycle_end`.
- Après Ctrl+C : log `orchestrator_stopping (keyboard interrupt)`.

- [ ] **Step 7: Vérifier la résolution rétroactive**

Si Run 1/2/3 ont créé une row dans training_data_live.csv avec `label=NaN`, et qu'au moins 15min se sont écoulées depuis, alors **un cycle suivant** devrait résoudre ce label. Pour vérifier sans attendre, on peut faire un dry-run après 15min :
```powershell
# Attendre 15+ minutes, puis :
cd live
python orchestrator.py --dry-run --no-notify
cd ..
```

Vérifier dans le log : `pendings_resolved labels=N outcomes=N` (avec N≥1 si une row pendante existait).

Inspecter à nouveau le CSV :
```powershell
Get-Content live\logs\training_data_live.csv | Select-Object -First 5
```

Confirmer que `label` est maintenant 0 ou 1 sur les rows anciennes.

- [ ] **Step 8: Si bugs trouvés, corriger inline**

Pour chaque bug trouvé :
1. Reproduire avec un cas minimal (`python -c "..."`).
2. Localiser le module concerné (probablement `features.py`, `store.py`, ou `orchestrator.py`).
3. Corriger.
4. Relancer Step 2-7 jusqu'à clean.

- [ ] **Step 9: Commit final**

Si des fix ont été nécessaires :
```bash
git add live/
git commit -m "fix(live): bugs revealed by end-to-end smoke test"
```

Si tout marchait du premier coup :
```bash
git status
# devrait montrer rien à part des artifacts (CSVs, logs ignorés via .gitignore)
```

---

## Self-Review (à compléter pendant la rédaction)

### 1. Spec coverage

Vérifier que chaque section du spec (`docs/superpowers/specs/2026-05-07-aure-live-orchestrator-design.md`) a au moins une task qui l'implémente :

| Spec section | Task(s) |
|---|---|
| Folder layout (`live/` sibling, fichiers) | Task 1 (scaffold), Task 7 (docs) |
| Réutilisation backtest via sys.path | Task 2 (fetch), Task 3 (features) |
| Cycle d'un tick (sleep, fetch, resolve, detect, predict, notify) | Task 6 (orchestrator) |
| Schéma `training_data_live.csv` 44 cols | Task 4 (store, TRAINING_DATA_COLUMNS) |
| Schéma `decisions.csv` 53 cols | Task 4 (store, DECISION_COLUMNS) |
| `target_open_time = timestamp + 15min` | Task 3 (features.py: `T = times[last_idx] + interval_15m_ms`) |
| `bet_direction` dérivé de `seq_dir` | Task 6 (orchestrator: `_predict_and_decide`) |
| `outcome` win/loss/pending | Task 4 (store: `update_pending_outcomes`) |
| Format Telegram HTML | Task 5 (notify: `format_decision_message`) |
| Config env vars Telegram | Task 1 (settings: ENV consts), Task 5 (notify: `_read_env`) |
| CLI flags `--dry-run`, `--no-notify` | Task 6 (orchestrator: argparse) |
| Erreurs Bitstamp (retry, skip cycle) | Task 6 (orchestrator: try/except autour run_cycle) |
| Erreurs Telegram (best-effort, disable on 4xx) | Task 5 (notify: `_disabled` flag) |
| Atomicité CSV updates | Task 4 (store: `_atomic_write`) |
| Restart : sidecar copy idempotent | Task 6 (orchestrator: `_ensure_sidecar`) |
| Restart : resolve pendings au démarrage | Task 6 (orchestrator: `main()` après warmup) |
| Edge case clock drift | Pas de task dédiée — couvert par `next_boundary_ms` qui se base sur `time.time()` ; le spec mentionne mitigation via lecture de la dernière bougie Bitstamp comme vérité, mais on garde simple ici. **Acceptable selon le spec si tolérance d'1 cycle raté.** |
| Multi-`seq_len` au même tick | Task 3 (features: loop `for seq_len in range(1, MAX_SEQ_LEN+1)`), Task 6 (orchestrator: loop `for seq_len, features in setups`) |

Aucune section du spec n'est non couverte (sauf clock-drift, explicitement noté comme accepté tel-quel).

### 2. Placeholder scan

- Tous les blocs de code Python sont complets (pas de `...`, pas de `# TODO`).
- Toutes les commandes `Run` ont l'expected output.
- Tous les noms de fonctions/types sont définis dans une task antérieure ou importés explicitement.

### 3. Type consistency

- `find_setups_at_latest` retourne `list[tuple[int, dict]]` ; consommé par `_predict_and_decide(features=dict)` et `append_*_row(features=dict)` — cohérent.
- `_predict_and_decide` retourne dict avec keys `confidence, effective_threshold, bucket_v, bucket_w, bet_direction, taken, notified_at` (notified_at ajouté après l'appel). `append_decision_row` consomme exactement ces keys.
- `format_decision_message(features, decision, price)` : `decision["bet_direction"]` correspond à ce que `_predict_and_decide` produit.
- `update_pending_labels` et `update_pending_outcomes` retournent `int` (count) ; consommé par `run_cycle` pour le log.

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-05-07-aure-live-orchestrator.md`. Two execution options:**

**1. Subagent-Driven (recommended)** - Je dispatch un fresh subagent par task, review entre tasks, itération rapide.

**2. Inline Execution** - J'exécute les tasks dans cette session avec executing-plans, batch avec checkpoints.

**Lequel ?**
