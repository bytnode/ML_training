# Spec : Polymarket live trader for `modele_15m`

**Date** : 2026-05-08
**Statut** : approuvé

## Problème

Le modèle `backtest/modele_15m/btc_reversal_model.json` est entraîné, calibré, et servable via `predict.should_take_bet`. Il n'est pour l'instant utilisé qu'en backtest et via le contrat `api.evaluate_on_slice`. L'objectif est de le faire tourner en **temps réel** sur les marchés Polymarket BTC up/down 15min, en argent réel, avec une logique strictement identique au backtest.

Workflow visé : à chaque clôture de bougie 15m Bitstamp, détecter si une streak qualifiante existe (2, 3 ou 4 bougies same-direction + `body_ratio_last >= 0.6`), demander la prédiction au modèle, appliquer le seuil adaptatif, et — si la décision est "take" — placer un ordre USDC réel sur le marché Polymarket correspondant à la fenêtre 15min suivante.

## Décisions clés

| # | Sujet | Choix |
|---|---|---|
| 1 | Mode | Argent réel sur Polymarket dès le jour 1 (pas de paper mode pour v1) |
| 2 | Sizing | 10% du USDC libre par fenêtre 15min |
| 3 | Multi-setup | Un seul ordre par fenêtre. Si plusieurs `seq_len` qualifient, on prend le **plus haut confidence** parmi les setups passant `should_take_bet` |
| 4 | `seq_len` actifs en live | `[2, 3, 4]` (séparé de `config.SEQ_TEST` côté backtest) |
| 5 | Type d'ordre | Market FOK uniquement (pas de limit) |
| 6 | Architecture | Resurrection du dossier `live/` à la racine (sibling de `backtest/`), réutilisation maximale de l'ancien design `2026-05-07-aure-live-orchestrator-design.md` |
| 7 | Polymarket API | Encapsulée dans **un seul fichier** `live/polymarket.py`. Aucun autre module ne touche aux endpoints Polymarket / Polygon |
| 8 | Wallet | EOA Polygon (clé privée) + proxy Polymarket (où le USDC vit). Rotation côté utilisateur, hors scope du spec |
| 9 | Notif Telegram | Conservée (visibilité ops sur bets pris, settlements, alertes) |
| 10 | Recovery | Au boot, réconciliation des `outcome=pending` ; pas de catch-up sur signaux ratés |

## Périmètre

### Layout

```
ML_agent_poly/
├── backtest/modele_15m/                  # inchange — source du modele + sidecar
├── live/                                  # NOUVEAU
│   ├── orchestrator.py                    # daemon entry point
│   ├── settings.py                        # config + paths + env loading
│   ├── fetch.py                           # Bitstamp wrapper (reuse backtest/modele_15m/data.py)
│   ├── features.py                        # setup detection + 42-feature build (reuse data.py helpers)
│   ├── decide.py                          # qualifies, calls predict, picks highest-confidence taken setup
│   ├── polymarket.py                      # *** unique entry point Polymarket / Polygon ***
│   ├── store.py                           # CSV append / update for decisions / bets / wallet
│   ├── notify.py                          # Telegram (ops visibility)
│   └── logs/
│       ├── decisions.csv
│       ├── bets.csv
│       ├── wallet_history.csv
│       └── orchestrator.log
├── infra/
│   ├── Dockerfile                          # ajuste : ajoute py-clob-client, web3, eth-account
│   └── docker-compose.yml                  # ajuste : mounts model + sidecar depuis backtest/modele_15m/
└── .env                                    # etend : POLYGON_PRIVATE_KEY, POLYMARKET_PROXY_ADDRESS, etc.
```

### Cross-package imports

`live/fetch.py` et `live/features.py` réutilisent les helpers de `backtest/modele_15m/data.py` via le hack `sys.path` documenté dans le design précédent :

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backtest", "modele_15m"))
from data import (
    fetch_all_klines, fetch_all_timeframes,
    compute_obv, compute_ema, compute_rsi, compute_atr, compute_volume_avg,
    last_closed_index,
)
from predict import predict_proba, should_take_bet
```

Cleanup futur possible : packager `backtest/modele_15m/` en `pip install -e`. Hors scope ici.

## Cycle loop

Une itération toutes les 15 minutes UTC, alignée sur les frontières (`xx:00`, `xx:15`, `xx:30`, `xx:45`).

```
1. SLEEP UNTIL next 15min boundary UTC
2. FRESHNESS RETRY (max FRESHNESS_RETRY_MAX = 5 attempts, 1s apart) :
   refetch Bitstamp until last 15m candle's open_time strictly > previous tick's last open_time.
   If exhausted -> log WARN cycle_skip reason=stale_candle, sleep to next boundary.
3. RESOLVE PENDING bets : pour chaque ligne de bets.csv ou outcome=pending et target_window_end_ms < now,
   appeler polymarket.read_settlement(condition_id). Si resolu, ecrire outcome + realized_pnl_usdc + outcome_resolved_at_iso.
   Si > SETTLEMENT_STUCK_AFTER_CYCLES (= 6) cycles passes sans resolution, marquer outcome=stuck + Telegram alerte.
4. FETCH new candles : 15m last 5 + MTF (30m/1h/4h/1d) last 3 each, dedup par open_time.
5. DETECT QUALIFYING SETUPS au dernier index 15m, pour chaque seq_len in SEQ_LIVE = [2, 3, 4] :
   - meme logique que data.extract_features (streak same-direction + body_ratio_last >= BODY_RATIO_MIN, flat exclus).
   - Renvoie une liste 0..3 de candidats.
6. EXTRACT 42 FEATURES par candidat (no-lookahead via last_closed_index).
7. DECIDE :
   confidences = predict_proba(features per candidate)
   takes      = should_take_bet(features per candidate)   # adaptive sidecar applique
   eligible = [(c, candidate) for c, candidate, t in ... if t]
   chosen = max(eligible, key=lambda x: x[0]) if eligible else None
   Append decisions.csv UNE LIGNE PAR CANDIDAT, avec is_chosen=True uniquement sur la ligne du chosen
   (donc decisions.csv est append-only ; pas de re-ecriture en place).
   bet_id reste NULL pour toutes les lignes a ce stade (sera fill plus bas si l'ordre passe).
   if chosen is None: continue (pas de bet, log no_take)
8. PLACE BET sur chosen :
   a. balance = polymarket.get_usdc_balance()
   b. bet_usdc = floor(balance * BET_FRACTION, 0.01)
   c. Sanity checks :
        bet_usdc < BET_MIN_USDC ($1) -> log ERROR balance_low + Telegram, skip bet.
        bet_usdc > balance * BET_MAX_FRACTION (0.95) -> log ERROR sizing_anomaly, abort cycle.
   d. T_open = times[last_idx_open_time] + 15min  (= debut de la fenetre cible)
   e. market = polymarket.find_btc_15m_market(T_open // 1000)
        Slug : btc-updown-15m-<T_open en seconds>.
        404 -> log WARN market_not_found + Telegram, decisions.csv ligne chosen reste sans bet_id, skip.
   f. token_id = market.down_token_id if chosen.seq_dir == 1 else market.up_token_id
        seq_dir=1 (up streak) -> bet reversal -> "Down". seq_dir=0 -> "Up". Same mapping qu'api.py.
   g. result = polymarket.place_market_order(token_id, bet_usdc)
        retry 1x sur HTTP error.
        si rejected/failed -> append bets.csv avec order_status correspondant, outcome=loss, realized_pnl=0, Telegram.
        si filled -> append bets.csv avec order_status=filled, fill_price_avg, shares_filled, usdc_spent, fee_usdc.
   h. Update decisions.csv ligne chosen avec bet_id.
9. NOTIFY Telegram : "BET PLACED <UP|DOWN> bet=$<x> at <fill_price>, conf=<y>, target window=<T_open>-<T_open+15m>"
10. SNAPSHOT wallet_history.csv : timestamp_iso, cycle_action, usdc_balance, matic_balance,
    pending_bets_n, pending_bets_exposure_usdc.
```

## Setup detection + decision

`live/features.py` reproduit *exactement* la logique de `backtest/modele_15m/data.extract_features`, restreinte au dernier `last_idx` (la bougie qui vient de fermer) et a `seq_len in SEQ_LIVE` :

```python
def detect_qualifying_setups(opens, highs, lows, closes, vols, last_idx) -> list[dict]:
    """Renvoie une liste de candidats {seq_len, seq_dir, last_idx} aux memes conditions
    que data.extract_features : streak same-direction non-flat + body_ratio_last >= BODY_RATIO_MIN."""

def build_features(opens, highs, lows, closes, vols, times, ohlcv_by_tf,
                   last_idx, seq_len, seq_dir) -> dict:
    """Construit le dict de 42 features (22 15m + 20 MTF) avec les memes helpers
    (compute_rsi, compute_atr, compute_ema, compute_obv, compute_volume_avg, last_closed_index)
    importes depuis backtest/modele_15m/data.py."""
```

`live/decide.py` :

```python
def choose_bet(qualifying_with_features: list[dict]) -> tuple[dict | None, list[dict]]:
    """qualifying_with_features : list[{seq_len, seq_dir, features, ...}].
    Retourne (chosen, decision_rows) ou :
      - decision_rows : 1 dict par candidat (pour append decisions.csv).
      - chosen : le candidat retenu (highest-confidence taken) ou None si aucun taken.
    """
```

C'est 1:1 avec la chaine du backtest : `data.extract_features` -> `predict.predict_proba` -> `predict.should_take_bet` -> direction mapping de `api.py`. Meme modele, meme sidecar, memes features -> memes decisions.

## `live/polymarket.py` — fichier unique pour Polymarket / Polygon

**Aucun autre module du projet n'importe `py_clob_client`, `web3`, ni `requests` vers `polymarket.com`/`gamma-api.polymarket.com`.** Cette discipline est volontaire : le jour ou l'API change, un seul fichier a auditer.

```python
class PolymarketClient:
    def __init__(
        self,
        private_key: str,            # POLYGON_PRIVATE_KEY
        eoa_address: str,            # POLYGON_EOA_ADDRESS
        proxy_address: str,          # POLYMARKET_PROXY_ADDRESS (= funder)
        rpc_url: str,                # POLYGON_RPC_URL
        clob_host: str,              # CLOB_HOST
        gamma_host: str,             # GAMMA_HOST
        timeout: float = 10.0,
    ):
        # web3 client pour balanceOf USDC.e directement on-chain
        self.web3 = Web3(HTTPProvider(rpc_url, request_kwargs={"timeout": timeout}))
        self.usdc = self.web3.eth.contract(address=USDC_POLYGON_ADDRESS, abi=ERC20_ABI)
        self.proxy_address = proxy_address
        self.eoa_address = eoa_address
        self.gamma_host = gamma_host
        self.timeout = timeout

        # py-clob-client pour signing + ordering
        self.clob = ClobClient(
            host=clob_host, key=private_key,
            chain_id=137, signature_type=2, funder=proxy_address,
        )
        self.clob.set_api_creds(self.clob.create_or_derive_api_creds())

    # --- USDC balance (lecture directe Polygon, pas via Polymarket API) ---
    def get_usdc_balance(self) -> float:
        wei = self.usdc.functions.balanceOf(self.proxy_address).call()
        return wei / 10**6

    def get_matic_balance(self) -> float:
        return self.web3.eth.get_balance(self.eoa_address) / 10**18

    # --- Market discovery ---
    def find_btc_15m_market(self, window_start_unix_s: int) -> dict:
        """Slug : btc-updown-15m-<unix_s>.
        Hit GET {gamma_host}/events?slug=<slug>.
        Retourne {condition_id, up_token_id, down_token_id, market_slug, end_date_iso, status}.
        Raises MarketNotFoundError si 404 / liste vide."""

    # --- Order placement ---
    def place_market_order(self, token_id: str, usdc_amount: float) -> dict:
        """FOK market BUY de usdc_amount USDC sur token_id.
        Retourne {order_id, status, average_fill_price, shares_filled, usdc_spent, fee_usdc, fills}.
        status in {filled, rejected, failed}.
        rejected = depth/price moved (pas de retry).
        failed = HTTP/network (1 retry 1s, puis levee)."""

    # --- Settlement read ---
    def read_settlement(self, condition_id: str) -> dict | None:
        """Apres cloture du window, hit Gamma pour le marche resolu.
        Retourne {resolved: bool, winning_token: 'up'|'down', resolved_at: iso} ou None si pas encore."""
```

### Auth model

EOA Polygon (clé privée signe les ordres) + proxy Polymarket (où vit le USDC). Les API creds CLOB sont dérivées de la clé privée au démarrage via `create_or_derive_api_creds()` puis cachées en mémoire — pas réécrites sur disque.

### Pourquoi web3 direct pour les balances

Plutôt que passer par l'API Polymarket pour `balanceOf`, on lit USDC.e directement on-chain via Polygon RPC. Plus rapide, pas soumis aux rate limits Polymarket, et c'est la source de vérité de toute façon.

USDC.e contract address sur Polygon : `0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174`.

### Order type

Market FOK uniquement. Limit risque de ne pas filler dans la fenêtre 15min → signal raté. Le slippage sur des marchés liquides BTC 15min à $1-100 USDC est de l'ordre du cent → acceptable. `fill_price_avg` est loggé dans `bets.csv` pour audit du coût de slippage cumulé.

### Bet sizing

```python
bet_usdc = floor(balance * BET_FRACTION, 0.01)  # 0.01 = precision UI Polymarket
# Pre-flight :
assert bet_usdc >= BET_MIN_USDC                  # = $1
assert bet_usdc <= balance * BET_MAX_FRACTION    # = 0.95, garde-fou
```

## Schémas CSV

### `live/logs/decisions.csv` — 1 row par candidat qualifiant

```
timestamp                    # open_time_ms de la derniere bougie de la streak
[42 features]                # 22 15m + 20 MTF, memes noms que training_data.csv
seq_len, seq_dir
bet_direction                # 'up' | 'down'
confidence                   # predict_proba output
effective_threshold          # ADAPTIVE_TARGET_WR + delta(bucket)
bucket_v, bucket_w
taken                        # bool, = should_take_bet result
is_chosen                    # bool, true pour le highest-confidence taken setup au tick (max 1)
bet_id                       # UUID4 link vers bets.csv (NULL si is_chosen=False ou ordre rejete)
created_at_iso
```

~52 colonnes. Si 3 candidats qualifient au meme tick, 3 rows. `is_chosen=True` au plus une fois par tick.

### `live/logs/bets.csv` — 1 row par ordre place reellement

```
bet_id                       # UUID4, primary key
decision_timestamp           # join key vers decisions.csv
created_at_iso
target_window_start_ms       # = decision_timestamp + 15min
target_window_end_ms         # = decision_timestamp + 30min
polymarket_slug              # btc-updown-15m-<unix_s>
polymarket_condition_id
polymarket_token_id          # le side achete
bet_direction                # 'up' | 'down'
bet_size_usdc                # le 10% calcule
polymarket_order_id
order_status                 # 'filled' | 'rejected' | 'failed'
fill_price_avg               # USDC per share
shares_filled
usdc_spent                   # actuel
fee_usdc
outcome                      # 'pending' | 'win' | 'loss' | 'stuck'
outcome_resolved_at_iso
realized_pnl_usdc            # win : shares_filled - usdc_spent ; loss : -usdc_spent
wallet_balance_pre_usdc
wallet_balance_post_usdc
```

### `live/logs/wallet_history.csv` — 1 row par cycle

`timestamp_iso, cycle_action, usdc_balance, matic_balance, pending_bets_n, pending_bets_exposure_usdc`

`cycle_action ∈ {bet_placed, no_setup, no_take, market_not_found, balance_low, settlement_only, error}`.

### `live/logs/orchestrator.log` — texte, 1 ligne par evenement

Pas de rotation auto pour v1.

## Configuration

### `live/settings.py`

```python
ROOT = Path(__file__).resolve().parent
LOGS_DIR = ROOT / "logs"
DECISIONS_CSV = LOGS_DIR / "decisions.csv"
BETS_CSV = LOGS_DIR / "bets.csv"
WALLET_HISTORY_CSV = LOGS_DIR / "wallet_history.csv"
ORCHESTRATOR_LOG = LOGS_DIR / "orchestrator.log"

# Cycle
SEQ_LIVE = [2, 3, 4]
FRESHNESS_RETRY_MAX = 5
FRESHNESS_RETRY_DELAY_S = 1

# Bet sizing
BET_FRACTION = 0.10
BET_MIN_USDC = 1.00
BET_MAX_FRACTION = 0.95

# Settlement reconciliation
SETTLEMENT_STUCK_AFTER_CYCLES = 6   # ~90min apres cloture du window

# Env var names (lus a os.environ au demarrage)
POLYGON_PRIVATE_KEY_ENV      = "POLYGON_PRIVATE_KEY"
POLYGON_EOA_ADDRESS_ENV      = "POLYGON_EOA_ADDRESS"
POLYMARKET_PROXY_ADDRESS_ENV = "POLYMARKET_PROXY_ADDRESS"
POLYGON_RPC_URL_ENV          = "POLYGON_RPC_URL"
CLOB_HOST_ENV                = "CLOB_HOST"
GAMMA_HOST_ENV               = "GAMMA_HOST"
POLYMARKET_API_TIMEOUT       = 10

# Telegram
TELEGRAM_BOT_TOKEN_ENV = "AURE_TELEGRAM_BOT_TOKEN"
TELEGRAM_CHAT_ID_ENV   = "AURE_TELEGRAM_CHAT_ID"
TELEGRAM_TIMEOUT       = 10

# MATIC sanity
MATIC_LOW_THRESHOLD = 0.5
```

### `.env` (extend l'existant)

```
POLYGON_PRIVATE_KEY=0x...
POLYGON_EOA_ADDRESS=0x...
POLYMARKET_PROXY_ADDRESS=0x...
POLYGON_RPC_URL=https://polygon-bor-rpc.publicnode.com
CLOB_HOST=https://clob.polymarket.com
GAMMA_HOST=https://gamma-api.polymarket.com
AURE_TELEGRAM_BOT_TOKEN=...
AURE_TELEGRAM_CHAT_ID=...
```

### `infra/Dockerfile`

```diff
 RUN pip install --no-cache-dir \
     requests \
     pandas \
     numpy \
     xgboost \
-    scikit-learn
+    scikit-learn \
+    py-clob-client \
+    web3 \
+    eth-account \
+    python-dotenv
```

### `infra/docker-compose.yml`

```yaml
volumes:
  # Modele + sidecar montes a /app/live/ pour matcher la convention path-relative de predict.py
  # (predict.py charge MODEL_OUTPUT="btc_reversal_model.json" relatif au cwd, qui est /app/live).
  - ../backtest/modele_15m/btc_reversal_model.json:/app/live/btc_reversal_model.json:ro
  - ../backtest/modele_15m/btc_reversal_model.thresholds.json:/app/live/btc_reversal_model.thresholds.json:ro
  - ../live/logs:/app/live/logs
env_file: ../.env
```

`predict.py` lit `MODEL_OUTPUT` (relative `btc_reversal_model.json`) ; le daemon tourne avec `WORKDIR /app/live`, donc les deux fichiers doivent etre montes a `/app/live/btc_reversal_model.json` et `/app/live/btc_reversal_model.thresholds.json`. Le `predict.py` cote `backtest/modele_15m/` reste inchange ; c'est juste le placement des fichiers dans le container qui s'adapte.

## Comportements de récupération

### Daemon restart

Au boot de `python orchestrator.py` :
1. Lit `bets.csv`. Pour chaque ligne `outcome="pending"` ET `target_window_end_ms < now`, appel `polymarket.read_settlement(condition_id)`. Resolution -> ecrit `outcome` + `realized_pnl_usdc` + `outcome_resolved_at_iso`. Si toujours non resolu et > `SETTLEMENT_STUCK_AFTER_CYCLES` cycles passes -> `outcome="stuck"` + Telegram alerte.
2. Pas de catch-up sur signaux rates : si le daemon a ete down sur des frontieres 15min, ces bougies sont fermees et leurs windows Polymarket potentiellement aussi. On dort jusqu'a la prochaine frontiere et on reprend.
3. Verifie l'auth Polymarket (les creds CLOB derivent proprement de la cle privee). Sinon -> exit avec erreur claire avant la boucle.
4. Log au boot : `INFO wallet_eoa=<...> proxy=<...> usdc_balance=<x> matic_balance=<y>`. Si `matic_balance < MATIC_LOW_THRESHOLD` -> Telegram one-time alerte.

### Matrice de pannes

| Where | Failure | Behavior |
|---|---|---|
| Bitstamp fetch | timeout / 5xx / pas de fresh candle dans 5 retries | log `WARN cycle_skip reason=stale_candle`, no bet, sleep |
| Setup detection | no qualifying setup | log `INFO no_setup`, no decisions.csv row, no bet |
| Setup detection | qualifying mais aucun take | log `INFO no_take`, decisions.csv rows pour chacun (`is_chosen=False`), no bet |
| Polymarket — slug 404 | retry 1x puis skip | `WARN market_not_found ts=<...>`, decisions.csv `is_chosen=True bet_id=NULL`, Telegram |
| Polymarket — order rejected (depth/price moved) | pas de retry | bets.csv `order_status="rejected" outcome="loss" realized_pnl=0` |
| Polymarket — HTTP error | 1 retry 1s, puis fail | bets.csv `order_status="failed" outcome="loss" realized_pnl=0`, Telegram |
| Bet sizing — `bet_usdc < BET_MIN_USDC` | balance trop basse | `ERROR balance_low usdc=<x>`, no bet, Telegram |
| Bet sizing — `bet_usdc > balance * BET_MAX_FRACTION` | sanity guard | `ERROR sizing_anomaly`, abort cycle (ne devrait jamais arriver) |
| Settlement | non resolu apres `SETTLEMENT_STUCK_AFTER_CYCLES` cycles | `outcome="stuck"`, Telegram, manual investigation |
| Telegram | any error | swallow, `WARN telegram_failed`, never blocks |
| Daemon | uncaught exception | log full traceback, re-raise -> Docker `restart: unless-stopped` |

## Vérifications post-implémentation

1. `cd live && python -c "from polymarket import PolymarketClient; ..."` instancie un client en lisant `.env`, fait `get_usdc_balance()` et `get_matic_balance()` -> renvoient des floats coherents avec ce qu'on voit dans MetaMask / Polymarket UI.
2. `python live/polymarket.py` (smoke) : trouve le marche pour la prochaine frontiere 15m via `find_btc_15m_market`, log `condition_id`, `up_token_id`, `down_token_id`, `end_date`.
3. `python live/decide.py` (smoke) : sur les dernieres bougies Bitstamp, detecte les qualifying setups + applique `should_take_bet` ; output identique a celui que `api.evaluate_on_slice` produirait sur la meme fenetre. **Test critique de parite live<->backtest**.
4. `python live/orchestrator.py --dry-run` : un seul cycle complet, sans place_market_order (court-circuite step 8.g) -> append decisions.csv, log les setups, ne touche pas au wallet.
5. `python live/orchestrator.py` : tourne en daemon, place les bets reels.
6. Apres N=10 bets : verifier que la somme `realized_pnl_usdc` cumulee dans `bets.csv` correspond au delta `usdc_balance` observe entre debut et fin de la periode.

## Hors-scope explicite

- Multi-pair (ETH/SOL/...) : v2 si BTC fonctionne.
- Position management post-entry (sortie anticipee, hedge, stop-loss) : Polymarket binary, pas pertinent.
- Backtest re-training trigger : reste manuel `cd backtest/modele_15m && python train.py`.
- Paper-mode toggle persistant : un `--dry-run` flag CLI suffit pour test.
- Bankroll caps / drawdown circuit-breaker : pas de stop automatique. A ajouter si besoin.
- Rotation cle privee : geree cote utilisateur.
- Tests unitaires : aucun (convention projet, pas de pytest).
