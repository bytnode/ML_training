# Spec : Aure Live Orchestrator

**Date** : 2026-05-07
**Statut** : approuvé

## Problème

Le modèle Aure (`aure.json` à la racine du repo) est la sortie du pipeline d'entraînement, actuellement inutilisée hors backtest. Objectif : le mettre en conditions réelles via un orchestrateur qui :

1. Fetche les bougies Bitstamp en continu (15m + MTF 30m/1h/4h/1d).
2. Détecte les setups qualifiants (streak 1-4 same-dir + `body_ratio_last ≥ 0.6`).
3. Calcule les 42 features attendues par Aure.
4. Lance la prédiction et applique le seuil adaptatif par bucket (vol × weekend) du sidecar.
5. Envoie une notification Telegram quand Aure "tire" (BUY ou SELL).
6. Archive l'input (CSV même schéma que `training_data.csv`) ET l'audit des décisions (CSV séparé self-contained avec outcome).

Le tout en mode **paper trading** : aucun ordre réel n'est placé. L'orchestrateur observe et notifie, c'est tout.

## Décisions clés (résumé du brainstorming)

| # | Décision | Choix |
|---|----------|-------|
| 1 | Mode opératoire | Paper trading + notif Telegram (option B du Q1) |
| 2 | Stockage | CSVs simples (pas de SQL) |
| 3 | Schéma | `training_data_live.csv` strictement identique à `training_data.csv` (44 cols) + `decisions.csv` self-contained séparé (53 cols) |
| 4 | Canal de notif | Telegram (bot token + chat_id via env vars) |
| 5 | Modèle d'exécution | Daemon long-running (mono-processus, sleep until next 15m boundary) |
| 6 | État initial | CSV vide au démarrage (pas de copie de `training_data.csv`, pas de refetch full history) |
| 7 | Sidecar threshold | Copié de `backtest/btc_reversal_model.thresholds.json` → `aure.thresholds.json` au 1er run si absent |
| 8 | Emplacement | Nouveau dossier racine `live/`, sibling de `backtest/` |

## Architecture

### Layout repo

```
ML_agent_poly/
├── aure.json                        # modèle (existant)
├── aure.thresholds.json             # sidecar (créé au 1er run, copié de backtest/)
├── backtest/                        # inchangé
└── live/                            # NOUVEAU
    ├── orchestrator.py              # daemon principal (point d'entrée)
    ├── fetch.py                     # wrapper API Bitstamp (réutilise backtest/data.py)
    ├── features.py                  # extraction des 42 features (réutilise backtest/data.py)
    ├── store.py                     # I/O CSV (read tail, append, update outcome/label)
    ├── notify.py                    # client Telegram
    ├── config.py                    # constants live (chemins, intervalles, retry policy)
    └── logs/
        ├── training_data_live.csv   # 44 cols, schema identique à backtest/training_data.csv
        ├── decisions.csv            # 51 cols, audit self-contained
        └── orchestrator.log         # log texte
```

### Réutilisation du code de `backtest/`

`live/fetch.py` et `live/features.py` importent les fonctions de `backtest/data.py` via :

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backtest"))
from data import (
    fetch_all_klines, fetch_all_timeframes,
    compute_obv, compute_ema, compute_rsi, compute_atr, compute_volume_avg,
    last_closed_index,
)
```

Solution moche mais simple. Pas de package install (pas de dette à payer). Si jamais ça devient gênant, transformer `backtest/` en package `pip install -e backtest/` plus tard.

**Attention** : `backtest/predict.py` charge le modèle depuis `MODEL_OUTPUT` (config = `"btc_reversal_model.json"`, relative path). On ne peut pas le réutiliser tel quel. `live/orchestrator.py` charge `aure.json` directement avec son propre code (~5 lignes XGBoost) :

```python
from xgboost import XGBClassifier
model = XGBClassifier()
model.load_model(str(MODEL_PATH))
```

Le sidecar `aure.thresholds.json` est lu à la main (JSON parse) au démarrage, pas via `predict._load_thresholds()`.

### Cycle d'un tick (toutes les 15min)

```
1. SLEEP UNTIL next_15m_boundary_utc
2. RESOLVE PENDING OUTCOMES (decisions.csv where outcome=pending et target now closed)
   + RESOLVE PENDING LABELS (training_data_live.csv where label=NaN et next 15m candle now closed)
3. FETCH NEW CANDLES (15m last 5, MTF last 3 each TF, dedup by open_time)
4. CHECK QUALIFYING SETUPS (pour chaque seq_len ∈ {1,2,3,4})
   ├─ pas de setup → skip cycle
   └─ setup détecté →
        5. EXTRACT 42 FEATURES (22 15m + 20 MTF, no-lookahead via last_closed_index)
        6. APPEND training_data_live.csv (44 cols, label=NaN, sera updaté plus tard)
        7. PREDICT & DECIDE
           - confidence = aure.predict_proba(features)[1]
           - bucket = (volatility_state, is_weekend)
           - effective_threshold = sidecar.base + sidecar.deltas[bucket]
           - taken = (confidence >= effective_threshold)
        8. APPEND decisions.csv (51 cols, outcome=pending)
        9. NOTIFY Telegram (si taken=True)
```

Une même position peut produire 0-4 setups qualifiants (un par seq_len). Chaque setup → 1 row dans `training_data_live.csv` + 1 row dans `decisions.csv` + 1 notif Telegram si `taken=True`. **Pas de dédup** entre seq_len — chaque longueur de streak est un signal indépendant pour Aure.

## Schémas exacts des fichiers

### `live/logs/training_data_live.csv` — 44 colonnes

Strictement identique à `backtest/training_data.csv` pour rester ré-utilisable par `train.py` si on veut ré-entraîner Aure plus tard sur les données live :

```
timestamp, seq_len, seq_dir, body_ratio_last, body_ratio_avg, upper_wick,
lower_wick, volume_ratio_last, volume_ratio_avg, move_pct, move_atr, rsi,
rsi_change_5, obv_slope_10, obv_vs_ema, atr_ratio, volatility_state,
dist_ema20, dist_ema50, dist_ema200, hour_utc, day_of_week, is_weekend,
open_30m, high_30m, low_30m, close_30m, vol_30m,
open_1h,  high_1h,  low_1h,  close_1h,  vol_1h,
open_4h,  high_4h,  low_4h,  close_4h,  vol_4h,
open_1d,  high_1d,  low_1d,  close_1d,  vol_1d,
label
```

`label` reste `NaN` au moment du append. Update rétroactif au cycle suivant quand la bougie cible (`timestamp + 15min`) est connue : `label=1` si elle inverse la direction du streak, `0` sinon.

### `live/logs/decisions.csv` — 53 colonnes

Self-contained : 42 features qui ont servi à la décision + métadonnées de décision + outcome. Total = 1 (timestamp) + 42 (features) + 10 (métadonnées) = 53 colonnes.

```
timestamp,                      # open_time_ms de la dernière bougie du streak (= cohérent avec training_data.csv)
[42 features identiques à training_data_live.csv, sans timestamp/label],
confidence,                     # float ∈ [0, 1]
effective_threshold,            # float = base + sidecar.deltas[bucket]
bucket_v,                       # int ∈ {0,1,2}, copie de volatility_state
bucket_w,                       # int ∈ {0,1}, copie de is_weekend
bet_direction,                  # "up" | "down"
taken,                          # bool
notified_at,                    # ISO 8601 UTC, NULL si taken=False
target_open_time,               # open_time_ms de la bougie cible (= timestamp + 15min)
outcome,                        # "win" | "loss" | "pending"
outcome_resolved_at             # ISO 8601 UTC, NULL si pending
```

**`timestamp` vs `target_open_time`** :
- `timestamp` = `open_time_ms` de la **dernière bougie du streak** (= ce que `backtest/data.py` écrit comme `timestamp` dans `training_data.csv`, ligne `row["timestamp"] = times[last_idx]`).
- `target_open_time` = `timestamp + 15min` = `open_time_ms` de la **bougie suivante** (= la bougie qui révèle l'outcome). C'est T dans la terminologie de `data.py`.
- Stocker explicitement `target_open_time` permet de résoudre `outcome` en SQL/pandas plus tard sans recalculer, et évite l'ambiguïté que "timestamp" recouvre la bougie du streak, pas la bougie cible.

**`bet_direction`** dérivé directement de `seq_dir` :
- `seq_dir=1` (streak up) + Aure dit "reversal" → bet `down`
- `seq_dir=0` (streak down) + Aure dit "reversal" → bet `up`

**`outcome`** calculé au cycle suivant à partir de la bougie 15m dont `open_time_ms == target_open_time` :
- Bougie haussière (`close > open`) ET `bet_direction=up` → `win`
- Bougie baissière (`close < open`) ET `bet_direction=down` → `win`
- Flat (`close == open`) → `loss` (cohérent avec le filtre dans `data.py` qui exclut les flats du dataset).
- Sinon → `loss`

Cette logique est équivalente à la définition du `label` dans `backtest/data.py` (`label=1` quand `next_dir == expected_reversal`). Donc `outcome=win` ⇔ `label=1`.

### `live/logs/orchestrator.log` — log texte

Un événement par ligne, format clair :

```
2026-05-07T13:30:00Z INFO  cycle_start
2026-05-07T13:30:01Z INFO  fetched candles_15m=5 candles_30m=3 candles_1h=3 candles_4h=3 candles_1d=3
2026-05-07T13:30:01Z INFO  setups_qualifying=2 (seq_len=2,3)
2026-05-07T13:30:01Z INFO  decision seq_len=2 confidence=0.624 threshold=0.580 bucket=1_0 taken=True direction=down
2026-05-07T13:30:01Z INFO  decision seq_len=3 confidence=0.547 threshold=0.580 bucket=1_0 taken=False direction=up
2026-05-07T13:30:02Z INFO  notif_sent telegram seq_len=2
2026-05-07T13:30:02Z INFO  outcomes_resolved=3
2026-05-07T13:30:02Z INFO  cycle_end duration_ms=2103
```

Pas de rotation auto. ~10 lignes/cycle × 96 cycles/jour ≈ 1k lignes/jour ≈ 365k lignes/an. Reste lisible en `tail -f`. Si besoin de rotation plus tard : `RotatingFileHandler` Python stdlib.

## Format Telegram

HTML format, lisible mobile, **un message par décision `taken=True`** :

```html
🟢 <b>BUY BTC</b>
<i>Aure setup</i>
Price : <b>$67,234</b>
Confidence : 62.4% (th 58.0%)
Streak : 3 candles down
Bucket : vol=normal weekend=no
Time : 2026-05-07 13:30 UTC
```

ou en SELL :

```html
🔴 <b>SELL BTC</b>
<i>Aure setup</i>
Price : <b>$67,234</b>
Confidence : 61.8% (th 56.5%)
Streak : 2 candles up
Bucket : vol=high weekend=no
Time : 2026-05-07 13:30 UTC
```

API call :
```
POST https://api.telegram.org/bot<TOKEN>/sendMessage
Content-Type: application/json
{
  "chat_id": "<CHAT_ID>",
  "text": "...",
  "parse_mode": "HTML"
}
```

Si 2 setups qualifient au même tick (seq_len 2 ET 3 par ex), 2 messages séparés. Pas de consolidation.

## Configuration

### `live/config.py`

```python
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOGS_DIR = ROOT / "logs"
TRAINING_DATA_LIVE = LOGS_DIR / "training_data_live.csv"
DECISIONS_CSV = LOGS_DIR / "decisions.csv"
ORCHESTRATOR_LOG = LOGS_DIR / "orchestrator.log"

MODEL_PATH = ROOT.parent / "aure.json"
THRESHOLDS_PATH = ROOT.parent / "aure.thresholds.json"
THRESHOLDS_SOURCE = ROOT.parent / "backtest" / "btc_reversal_model.thresholds.json"

# Cycle
WARMUP_CANDLES_15M = 200
WARMUP_CANDLES_MTF = 50
RECENT_FETCH_15M = 5
RECENT_FETCH_MTF = 3
FETCH_RETRIES = 3
FETCH_BACKOFF_BASE = 2

# Telegram
TELEGRAM_BOT_TOKEN_ENV = "AURE_TELEGRAM_BOT_TOKEN"
TELEGRAM_CHAT_ID_ENV = "AURE_TELEGRAM_CHAT_ID"
TELEGRAM_TIMEOUT = 10

# Symbol / interval / hyperparams sont réutilisés depuis backtest/config.py
# via le sys.path.insert hack au top des modules concernés :
#   import sys, os
#   sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backtest"))
#   from config import SYMBOL, INTERVAL, MTF_INTERVALS, BODY_RATIO_MIN, MAX_SEQ_LEN, WARMUP, ...
```

### Env vars (lus à `os.environ` au démarrage, jamais hardcodés)

```powershell
$env:AURE_TELEGRAM_BOT_TOKEN = "123456789:ABCdef..."
$env:AURE_TELEGRAM_CHAT_ID = "-1001234567890"
```

Si une env var manque, log `WARN telegram_disabled` et le daemon tourne quand même (tu vois les décisions dans `decisions.csv` et `orchestrator.log`).

### Lancement

```powershell
cd live
python orchestrator.py
```

CLI flags optionnels :
- `--no-notify` : log les décisions mais n'envoie pas de Telegram (test sans spammer le chat).
- `--dry-run` : un seul cycle puis exit (test fonctionnel rapide).

## Gestion des erreurs

### Bitstamp (fetch)

- Retry 3× avec backoff exponentiel (`2 ** attempt`s) sur `Timeout` / `ConnectionError`. Cohérent avec `backtest/data.py`.
- Si les 3 retries échouent → log `WARN fetch_failed`, **skip le cycle entier** (pas de prediction, pas de notif). On attend le prochain tick.
- Erreur HTTP non-réseau (4xx/5xx) → raise (= daemon crash). Volontaire : signale un changement d'API Bitstamp qui mérite intervention humaine.

### Telegram (notif)

- Best-effort, jamais faire crasher le cycle.
- Timeout / 5xx → 1 retry après 2s, puis log `WARN telegram_failed setup_timestamp=...` et continue. La décision est déjà loguée dans `decisions.csv` avec `notified_at=NULL`.
- Bot token / chat_id mal configurés (4xx auth) → log `ERROR telegram_misconfigured` une fois et **désactive les notifs jusqu'au prochain restart**. Le daemon continue à logger les décisions.

### Atomicité CSV

- **Append** (`training_data_live.csv` nouvelle row, `decisions.csv` nouvelle row) → `mode='a'` simple. Pire cas en cas de crash mid-write : ligne tronquée à la fin. Acceptable.
- **Update** (`outcome` dans `decisions.csv`, `label` dans `training_data_live.csv`) → read full / modify / `to_csv(tmp_path)` puis `os.replace(tmp_path, real_path)`. Atomique sur Windows depuis Python 3.3.

### Restart du daemon

Au démarrage :
1. Charge `aure.json` (XGBoost).
2. Si `aure.thresholds.json` n'existe pas → copie `backtest/btc_reversal_model.thresholds.json` à cet emplacement, log `INFO sidecar_copied`.
3. Charge le sidecar.
4. Si `training_data_live.csv` ou `decisions.csv` existent → lis-les pour identifier les rows pendantes (label=NaN, outcome=pending).
5. Fetch warmup : 200 dernières bougies 15m + 50 dernières de chaque MTF.
6. Backfill outcomes / labels passés avec ce qu'on peut résoudre depuis le warmup. Si certains pendings sont trop vieux pour être résolus (daemon down >50h) → log `WARN unresolvable_pending count=N`, on les laisse `pending` indéfiniment.
7. Calcule la prochaine frontière 15min UTC, sleep.

### Edge cases

- **Horloge système désynchronisée** : on ne se fie pas à `time.time()` pour décider qu'une bougie est fermée. On regarde le `open_time_ms` de la dernière bougie 15m fermée renvoyée par Bitstamp. Si elle est en retard de plus d'un cycle → on attend 30s et on retry le fetch.
- **Multi-`seq_len` au même tick** : c'est attendu et géré par construction. Aucune dédup. Plusieurs notifs Telegram en rafale possibles.
- **Zone horaire** : tout en UTC. `is_weekend`, `hour_utc`, `day_of_week` calculés depuis `timestamp` ms UTC. Aure a été entraînée en UTC.

## Hors scope

- **Backfill complet** après outage long (>50h). Si ça arrive, relancer `backtest/data.py` pour regenerate les setups historiques manqués.
- **Health check externe** (Prometheus, Datadog, etc.). `orchestrator.log` est la source de vérité.
- **Hot reload** du sidecar threshold ou du modèle. Faut redémarrer le daemon.
- **Recalibration du sidecar** depuis les données live. Pas assez de data au début ; à reconsidérer dans plusieurs mois.
- **Sizing / position management / placement d'ordre réel**. Mode paper trading uniquement.
- **Consolidation des notifs** quand plusieurs `seq_len` qualifient au même tick. Une notif par seq_len, c'est intentionnel (rien de plus).
- **Dashboard / interface visuelle**. Tu lis `decisions.csv` à la main / pandas pour analyses.

## Fichiers touchés / créés

- `aure.thresholds.json` (créé à la racine au 1er run, copie de `backtest/btc_reversal_model.thresholds.json`).
- `live/` (nouveau dossier complet : `orchestrator.py`, `fetch.py`, `features.py`, `store.py`, `notify.py`, `config.py`, `logs/.gitkeep`).
- `.gitignore` (ajouter `live/logs/*.csv`, `live/logs/*.log`, `aure.thresholds.json`).
- `CLAUDE.md` racine (ajouter une mention de `live/` à côté de `backtest/`).
