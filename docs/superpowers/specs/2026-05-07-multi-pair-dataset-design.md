# Spec : Dataset multi-paires (BTC + ETH + SOL + BNB + XRP)

**Date** : 2026-05-07
**Statut** : approuvé

## Problème

Le dataset d'entraînement actuel (`backtest/training_data.csv`, ~290k lignes) ne contient que BTC. Pour augmenter la quantité de signal et permettre à Aure (ou un futur modèle) d'apprendre sur plusieurs cryptos, on veut produire un dataset équivalent contenant **les mêmes features** calculées indépendamment sur 5 paires : BTC, ETH, SOL, BNB, XRP. Toutes les paires sont disponibles sur Bitstamp en quote USD (`btcusd, ethusd, solusd, bnbusd, xrpusd`).

Volume cible : ~1.45M lignes (5×).

## Décisions clés (résumé du brainstorming)

| # | Décision | Choix |
|---|----------|-------|
| 1 | Encodage de la paire | Colonne `symbol_id` int (0=btc, 1=eth, 2=sol, 3=bnb, 4=xrp) |
| 2 | Profondeur d'historique | Max disponible par paire (BTC ~5000j, ETH/XRP ~3000j, SOL ~1500j, BNB ~500j) |
| 3 | Stockage de sortie | Nouveau fichier `training_data_multi.csv` à côté du CSV BTC actuel (pas d'overwrite, pas de backup nécessaire) |
| 4 | Invocation | Nouveau script `backtest/data_multi.py`, sibling de `data.py` |

## Architecture

### Layout des fichiers

```
backtest/
├── config.py              # ajout SYMBOLS_MULTI, SYMBOL_TO_ID, OUTPUT_CSV_MULTI
├── data.py                # parametre fetch_*() pour prendre symbol en arg, sinon inchange
├── data_multi.py          # NOUVEAU : loop sur SYMBOLS_MULTI, concat, sort, ecriture
├── training_data.csv      # inchange (BTC-only, produit par data.py)
└── training_data_multi.csv # NOUVEAU (multi-pair, produit par data_multi.py, gitignore via *.csv)
```

### Constantes ajoutées à `backtest/config.py`

```python
# --- Multi-pair dataset (consume par data_multi.py) ---
SYMBOLS_MULTI = ["btcusd", "ethusd", "solusd", "bnbusd", "xrpusd"]
SYMBOL_TO_ID = {s: i for i, s in enumerate(SYMBOLS_MULTI)}  # btcusd=0, ethusd=1, solusd=2, bnbusd=3, xrpusd=4
OUTPUT_CSV_MULTI = "training_data_multi.csv"
```

`SYMBOL`, `OUTPUT_CSV`, `DAYS_BACK` etc. restent inchangés (consommés par `data.py` legacy).

### Refactor de `backtest/data.py`

Les fonctions de fetch lisent `SYMBOL` via le scope module :

```python
# Avant
def fetch_chunk(start_sec, end_sec, interval):
    url = API_URL.format(pair=SYMBOL)
    ...

def fetch_all_klines(start_time, end_time, interval): ...
def fetch_all_timeframes(start_time, end_time, intervals): ...
```

Devient :

```python
# Apres
def fetch_chunk(symbol, start_sec, end_sec, interval):
    url = API_URL.format(pair=symbol)
    ...

def fetch_all_klines(symbol, start_time, end_time, interval): ...
def fetch_all_timeframes(symbol, start_time, end_time, intervals): ...
```

`data.py:main()` met à jour ses 2 appels pour passer `SYMBOL` explicitement. **Aucun changement de comportement** pour le pipeline BTC actuel.

### Impact sur `live/`

`live/fetch.py` importe `fetch_all_klines` de `backtest/data.py` via le sys.path hack. Avec la nouvelle signature qui prend `symbol`, **`live/fetch.py` doit être mis à jour** pour passer `SYMBOL` (ou hardcoder `"btcusd"` puisque live tourne BTC-only). Modification : ~3 lignes dans `live/fetch.py:_fetch_all`.

### Flux de `backtest/data_multi.py`

```python
import pandas as pd
from data import fetch_all_klines, fetch_all_timeframes, extract_features
from config import (
    SYMBOLS_MULTI, SYMBOL_TO_ID, OUTPUT_CSV_MULTI,
    DAYS_BACK, MTF_INTERVALS, INTERVAL,
)

def main():
    end_time = int(time.time() * 1000)
    start_time = end_time - DAYS_BACK * 86_400_000
    all_rows = []

    for symbol in SYMBOLS_MULTI:
        print(f"\n=== Fetching {symbol} ===", flush=True)
        klines_15m = fetch_all_klines(symbol, start_time, end_time, INTERVAL)
        klines_mtf = fetch_all_timeframes(symbol, start_time, end_time, MTF_INTERVALS)
        rows = extract_features(klines_15m, klines_mtf)
        sym_id = SYMBOL_TO_ID[symbol]
        for r in rows:
            r["symbol_id"] = sym_id
        print(f"  {symbol}: {len(rows):,} feature rows")
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    # symbol_id juste avant label
    cols = [c for c in df.columns if c not in ("symbol_id", "label")] + ["symbol_id", "label"]
    df = df[cols]
    df = df.sort_values("timestamp").reset_index(drop=True)
    df.to_csv(OUTPUT_CSV_MULTI, index=False)
    print(f"\nWrote {len(df):,} rows to {OUTPUT_CSV_MULTI}")

if __name__ == "__main__":
    main()
```

Point critique : **`extract_features` est appelée per-paire**, jamais sur des klines mélangés. Les streaks détectés sont strictement intra-paire — aucun risque de contamination.

## Schéma `training_data_multi.csv` — 45 colonnes

```
timestamp, seq_len, seq_dir, body_ratio_last, body_ratio_avg, upper_wick,
lower_wick, volume_ratio_last, volume_ratio_avg, move_pct, move_atr, rsi,
rsi_change_5, obv_slope_10, obv_vs_ema, atr_ratio, volatility_state,
dist_ema20, dist_ema50, dist_ema200, hour_utc, day_of_week, is_weekend,
open_30m, high_30m, low_30m, close_30m, vol_30m,
open_1h,  high_1h,  low_1h,  close_1h,  vol_1h,
open_4h,  high_4h,  low_4h,  close_4h,  vol_4h,
open_1d,  high_1d,  low_1d,  close_1d,  vol_1d,
symbol_id,
label
```

`symbol_id` placé entre `vol_1d` et `label` (avant-dernière colonne) pour que `train.py:feature_cols = [c for c in df.columns if c not in ("timestamp", "label")]` l'inclue automatiquement comme feature.

## Impact sur le reste du pipeline

### `train.py`

**Aucun changement de code requis.** Pour entraîner sur multi-pair :
1. Changer `CSV_PATH = "training_data_multi.csv"` dans `config.py`.
2. Lancer `python train.py` normalement.

`feature_cols` inclura `symbol_id` automatiquement → 43 features au lieu de 42. Le modèle XGBoost peut splitter dessus pour apprendre des patterns pair-specific.

Note : le split temporel reste cohérent. Test = 365 derniers jours × toutes les paires actives sur cette fenêtre. Pas de fuite cross-pair (chaque paire a ses propres timestamps consecutifs intra-paire ; le sort global les interleave).

### `predict.py`

**Aucun changement de code requis.** `predict.py` lit `model.get_booster().feature_names` au runtime — si le booster a été entraîné avec `symbol_id`, le caller (live/) doit le fournir.

### `live/`

**Modifications nécessaires SI tu réentraînes Aure sur le multi-pair**, sinon aucune. Trois scénarios documentés :

1. **Tu ne réentraînes pas Aure** → live/ continue avec l'Aure BTC-only actuel. Le multi-pair est uniquement pour la recherche backtest. Aucun impact live. **Comportement par défaut.**
2. **Tu réentraînes et déploies un nouveau `aure.json`** → `live/features.py:find_setups_at_latest` doit injecter `features["symbol_id"] = 0` dans chaque dict retourné (puisque live tourne BTC-only, symbol_id est constant). Modification ~2 lignes. La garde défensive de `live/orchestrator.py:_predict_and_decide` (validation des keys) lèvera une `ValueError` claire si tu oublies cette étape.
3. **Tu déploies un orchestrateur multi-paires aussi** → hors scope, design séparé.

Le refactor de signature de `fetch_all_klines` (qui prend désormais `symbol` en argument) impacte `live/fetch.py` indépendamment de la décision sur Aure : à modifier simultanément. Voir section "Impact sur `live/`" plus haut.

### `clean_data/analyze.py`, `clean_data/clean.py`

Fonctionnent déjà sur n'importe quel `training_data*.csv`. `symbol_id` apparaîtra dans leurs analyses (multicollinéarité, drift, mutual information avec label). Aucune modif requise.

### `sweep.py`, `analyze_runs.py`

Pas impactés. Tournent sur le CSV pointé par `CSV_PATH`.

## Gestion d'erreurs

- **Erreur réseau / timeout** → retry 3× avec backoff exponentiel (hérité de `fetch_chunk` actuel). Si les 3 retries échouent → raise et abort tout le run. Pas de fallback "skip cette paire".
- **HTTP 4xx (paire malformée / non listée)** → raise immédiat. Le user voit explicitement quelle paire a échoué.
- **Paire avec peu d'historique** (ex: BNB ~500j) → pas une erreur. `fetch_all_klines` retourne ce qui existe. `extract_features` produit moins de lignes. Le print `{symbol}: {N:,} feature rows` rend la disparité visible.

## Perf & mémoire

| Métrique | Estimation |
|----------|-----------|
| Temps total | ~10 min/paire × 5 paires séquentielles = ~50 min |
| Mémoire peak (DataFrame final) | ~3 GB (avec dtypes optimisés, surtout int8/int16 pour `symbol_id`, `seq_len`, `seq_dir`, `volatility_state`, `hour_utc`, `day_of_week`, `is_weekend`) |
| Disque (CSV final) | ~500 MB |
| Lignes | ~1.45M |

Séquentiel au niveau paire (la parallélisation par fenêtres dans `fetch_all_klines` est déjà là via `FETCH_WORKERS=8`). Pas de chunking — la machine cible peut tenir 3-5 GB en RAM le temps du concat + write.

## Hors scope (explicite)

- **Pas de retraining d'Aure** dans ce spec. C'est une étape suivante triviale (changer `CSV_PATH`, lancer `train.py`).
- **Pas de modification de `aure.json`** ni du sidecar.
- **Pas de modification du orchestrateur live** (sauf le refactor de signature `fetch_all_klines` qui doit être propagé pour ne pas casser live/).
- **Pas de backup du `training_data.csv` BTC** — le user peut le faire à la main si il le veut, ou simplement repointer `CSV_PATH` quand il veut tester l'ancien dataset.
- **Pas de validation cross-pair** (e.g. dédup) — `extract_features` produit des rows uniques par `(timestamp, seq_len)` intra-paire, et les timestamps sont compatibles cross-paire (il peut y avoir 5 lignes au même timestamp, une par paire — c'est attendu et discriminé par `symbol_id`).
- **Pas de parallélisation au niveau paire** — séquentiel pour ne pas saturer Bitstamp et garder le log lisible.

## Fichiers touchés / créés

- `backtest/config.py` — ajout `SYMBOLS_MULTI`, `SYMBOL_TO_ID`, `OUTPUT_CSV_MULTI`.
- `backtest/data.py` — refactor signatures `fetch_chunk`, `fetch_all_klines`, `fetch_all_timeframes` pour prendre `symbol` en argument. Update `data.py:main()` pour passer `SYMBOL`.
- `backtest/data_multi.py` — NOUVEAU script.
- `live/fetch.py` — update les appels à `fetch_all_klines` pour passer `SYMBOL` (importé de backtest/config) ou hardcoder `"btcusd"`.
- `backtest/training_data_multi.csv` — produit au runtime par `data_multi.py`. Auto-ignoré (`.gitignore` contient déjà `*.csv`).
