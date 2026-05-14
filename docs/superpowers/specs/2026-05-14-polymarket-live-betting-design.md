# Spec: Polymarket live betting for `modele_updown`

**Date** : 2026-05-14
**Statut** : draft (en attente review utilisateur)

## Problème

`backtest/modele_updown/15min/live/live.py` tourne en mode **paper-trading** : à chaque frontière UTC 15min il prédit `UP` / `DOWN` / `PASS` / `CANCEL` via l'équipe `modele_up` + `modele_down`, log la décision dans `logs/live_predictions.csv`, et envoie une notification Telegram à la résolution. Aucun ordre n'est placé.

*Note layout 2026-05-14 (post-spec)*: après la rédaction initiale, `main.py` et le dossier `live/` ont été déplacés à l'intérieur de `backtest/modele_updown/15min/` pour pouvoir cohabiter avec d'autres timeframes (e.g. `5min/`, `30min/`) sans collision. Tous les chemins ci-dessous sont à lire en gardant ce préfixe en tête.

L'objectif : un mode `LIVE_MODE=true` qui place un vrai ordre USDC sur Polymarket à chaque décision `UP` ou `DOWN`, sur le marché binaire 15-min correspondant. Tout le reste (modèle, décision, log, Telegram) reste identique.

Ce design réutilise l'architecture déjà éprouvée dans `docs/superpowers/specs/2026-05-08-polymarket-live-trader-design.md` (qui ciblait `modele_15m` et a été supprimée avec le reste de l'ancien `live/`), retargetée sur la team `modele_updown` et mise à jour suite à la recherche API documentée dans `backtest/docs/superpowers/research/polymarket-api-notes.md`.

## Décisions clés

| # | Sujet | Choix |
|---|---|---|
| 1 | Coexistence avec le paper-trading | Variable d'env **`LIVE_MODE`** dans `.env`. `LIVE_MODE=true` = real-bet mode. `LIVE_MODE=false` ou absent = paper mode (comportement actuel inchangé). Permet de basculer entre les deux modes par simple édition du `.env` + restart du container, sans modifier le code ni le `CMD` du Dockerfile. |
| 2 | Sizing | 10% du USDC libre par fenêtre, floor effectif **$2.50** (minimum 5 shares × ~$0.50 imposé par Polymarket), cap sanity à 95% du balance. |
| 3 | Type d'ordre | **FAK** (Fill-And-Kill) avec limit = `best_ask + slippage_bps/10000` (défaut 200 bps = 2 cents). Pas de FOK : FAK accepte un partial fill et préserve le signal. |
| 4 | Source de vérité pour le PnL | Polymarket d'abord. Si toujours `pending` après une cycle complet post-cloture (~15 min), fallback sur la bougie Bitstamp lue par `live.py`. Après 6 cycles sans résolution → `outcome=stuck` + alerte Telegram. |
| 5 | Recovery au démarrage | Catch-up : au boot, scan de `bets.csv` pour les rows `outcome=pending` dont la window est déjà fermée. Application du même cycle de résolution (Polymarket → fallback Bitstamp). |
| 6 | Redemption des shares gagnantes | **Automatique, immédiate** : dès qu'une résolution `win` est écrite dans `bets.csv`, appel on-chain `redeemPositions(...)` la même seconde. Préserve l'exactitude du `usdc_balance` utilisé pour le sizing. |
| 7 | Architecture | Un seul fichier `live/polymarket.py` encapsule tout (Polymarket CLOB, Gamma, Polygon RPC, redemption on-chain). Aucun autre module ne touche à ces APIs. |
| 8 | Wallet | EOA Polygon (`POLYGON_PRIVATE_KEY` signe les ordres) + Polymarket proxy (`POLYMARKET_PROXY_ADDRESS` détient l'USDC). `signature_type=2` dans `py-clob-client`. |
| 9 | Library Python | **`py-clob-client` v1** (v2 ne supporte pas `signature_type=2`). |
| 10 | Runtime | VPS dans une région **non géo-bloquée** par Polymarket (Hetzner Helsinki / OVH Ireland / DigitalOcean US). La France est bloquée, le paper mode local reste utilisable. |
| 11 | Telegram | Conservé. Nouvelle catégorie d'événement `bet_placed`, en plus du `bet_resolved` actuel. |
| 12 | Hors scope v1 | Bankroll caps, circuit breakers, multi-pair, hot-reload modèle, UI web, dashboard. |

## Layout

```
backtest/modele_updown/15min/
├── modele_up/               # paquet entrainement UP-only (inchange)
├── modele_down/             # paquet entrainement DOWN-only (inchange)
├── main.py                  # team backtest orchestrator (deplace ici depuis modele_updown/)
└── live/
    ├── live.py              # MODIF : read LIVE_MODE from env + add --dry-run flag + hooks Polymarket
    ├── polymarket.py        # NEW : seul fichier qui touche Polymarket / Polygon
    ├── live_helpers/
    │   ├── fetch.py         # inchange
    │   ├── features_up.py   # MODIF : parents[2] hop adapté au nesting 15min/
    │   ├── features_down.py # MODIF : parents[2] hop adapté au nesting 15min/
    │   └── notify.py        # MODIF : add send_bet_placed/resolved_real/stuck + error-bot
    ├── logs/
    │   ├── live_predictions.csv  # inchange — 1 row par cycle, paper-equivalent forever
    │   └── bets.csv              # NEW — 1 row par tentative d'ordre reel
    ├── Dockerfile           # MODIF : add py-clob-client/web3/eth-account ; chemins COPY repointés
    ├── Dockerfile.dockerignore # inchange
    ├── docker-compose.yml   # MODIF : volume mount /app/15min/live/logs (au lieu de /app/live/logs)
    ├── .env                 # MODIF : add 5 vars Polymarket + 2 vars error bot + tunables
    └── .env.example         # NEW — template gitignored-safe
```

## Cycle loop (mode `LIVE_MODE=true`)

Chaque itération aux frontières UTC 15min, structure identique à `live.py` actuel :

```
1. SLEEP UNTIL next 15min boundary UTC.
2. FRESHNESS RETRY (Bitstamp) jusqu'a ce que la bougie qui vient de fermer soit publiee.
3. RESOLVE PENDING BETS (only when LIVE_MODE=true) :
   pour chaque row dans bets.csv ou outcome="pending" ET target_close_ms < now :
     - hit polymarket.read_settlement(slug) :
         resolved -> outcome=win|loss, outcome_source=polymarket
           si win -> polymarket.redeem(condition_id), write redeem_tx_hash
         not resolved AND (now - target_close_ms) >= 15min (= au moins 1 cycle complet) :
           fallback sur live.py's resolution logic (next bar's open/close depuis Bitstamp)
             outcome = bitstamp_win | bitstamp_loss, outcome_source = bitstamp_fallback
             si bitstamp_win : polymarket.redeem() quand meme (UMA peut resoudre plus tard)
         not resolved AND (now - target_close_ms) < 15min :
           leave pending, try next cycle
         pending depuis > 6 cycles (90min) :
           outcome = stuck, Telegram alerte, no further auto-action
4. RESOLVE LAST CYCLE'S PAPER PREDICTION (inchange — comme live.py aujourd'hui).
5. FETCH OHLCV (15m / 5m / 30m / 1h) — inchange.
6. BUILD FEATURES (modele_up + modele_down) — inchange.
7. PREDICT (modele_up + modele_down) — inchange.
8. TEAM DECISION (inchange — UP / DOWN / PASS / CANCEL).
9. APPEND live_predictions.csv (inchange).
10. IF LIVE_MODE=true AND team in (UP, DOWN) :
     a. balance = polymarket.get_usdc_balance()
     b. bet_usdc = floor(balance * 0.10, 0.01)
     c. Sanity :
          bet_usdc < BET_MIN_USDC ($2.50) -> log skipped_balance_low, Telegram, skip step 10.
          bet_usdc > balance * 0.95 -> log sizing_anomaly, abort cycle (ne devrait jamais arriver).
     d. window_start_unix_s = (next_boundary_ms - INTERVAL_15M_MS) // 1000
        Note : la bougie qui vient de fermer ouvre la fenetre Polymarket. Le slug du marche
        Polymarket cible = btc-updown-15m-<window_start_unix_s>.
        WARNING : verifier au smoke-test que c'est bien la convention. Si Polymarket
        utilise window_END_unix_s, ajuster.
     e. market = polymarket.find_btc_15m_market(window_start_unix_s)
          404 -> bets.csv row order_status=skipped_market_not_found, Telegram, skip.
     f. token_id = market.up_token_id if team == "UP" else market.down_token_id
     g. result = polymarket.place_fak_order(token_id, bet_usdc, slippage_bps)
          status=filled / rejected / failed, see polymarket.py docstring.
     h. Append bets.csv UNE row avec status + fill details.
     i. Update Telegram : "BET PLACED <UP|DOWN> $<bet_usdc> at <fill_price> conf=<x>"
                          ou "BET FAILED <reason>".
11. IF LIVE_MODE=true AND --dry-run : steps 10.a / 10.b / 10.c / 10.d / 10.e exécutées normalement,
    mais 10.g remplacé par un stub qui log l'ordre qui aurait été envoyé. bets.csv toujours écrit
    (order_status="dry_run") pour audit.
12. SEND Telegram (inchange pour les resolutions paper, NEW pour les bet_placed/failed/stuck).
```

## `bets.csv` — schéma

Une row par tentative d'ordre réel (PASS / CANCEL / paper-mode cycles n'écrivent rien ici).

| Col | Type | Note |
|---|---|---|
| `bet_id` | uuid4 | primary key |
| `cycle_at` | iso8601 UTC | quand le cycle a tourne |
| `target_open_ms` | int | debut de la fenetre 15min cible. Join key vers `live_predictions.csv:target_open_ms` |
| `target_close_ms` | int | fin de fenetre = target_open_ms + 15min |
| `direction` | `UP` \| `DOWN` | team decision |
| `confidence` | float | `p_up` si UP, `p_dn` si DOWN |
| `polymarket_slug` | str | `btc-updown-15m-<window_start_unix_s>` |
| `polymarket_condition_id` | str | from Gamma `conditionId` |
| `polymarket_token_id` | str | le side achete (up_token_id ou down_token_id) |
| `bet_size_usdc` | float | 10% du free USDC, >= $2.50, ≤ 95% du balance |
| `polymarket_order_id` | str \| null | null si l'ordre n'a jamais atteint le CLOB (skipped_market_not_found, skipped_balance_low) |
| `order_status` | str | `filled` \| `rejected` \| `failed` \| `skipped_market_not_found` \| `skipped_balance_low` \| `dry_run` |
| `fill_price_avg` | float \| null | USDC per share. null si pas fillé |
| `shares_filled` | float | actual quantity received (partial fill possible avec FAK) |
| `usdc_spent` | float | actuel USDC charge. Peut differer de bet_size_usdc par partial fill ou slippage |
| `fee_usdc` | float \| null | Polymarket fee si reported (peut etre null) |
| `outcome` | str | `pending` \| `win` \| `loss` \| `stuck` \| `bitstamp_win` \| `bitstamp_loss` |
| `outcome_source` | str | `polymarket` \| `bitstamp_fallback` \| null (pour pending/stuck) |
| `outcome_resolved_at_iso` | iso8601 \| null | quand on a ecrit l'outcome |
| `realized_pnl_usdc` | float \| null | `shares_filled - usdc_spent` si win, `-usdc_spent` si loss |
| `redeem_tx_hash` | str \| null | tx hash de redeemPositions si win, null si pas encore redeemed ou si loss |
| `wallet_usdc_pre` | float | USDC.e balance lu juste avant `place_fak_order()` (= la base du sizing 10%) |
| `wallet_usdc_post` | float | USDC.e balance lu juste après que `place_fak_order()` retourne. Peut être égal à `pre` si l'ordre n'a pas encore atteint la chain ; sert seulement à mesurer le décalage on-chain vs off-chain. PnL est dérivé de `usdc_spent` / `shares_filled`, pas de cette colonne. |

Append-only. Update en place pour les colonnes `outcome*` / `realized_pnl_usdc` / `redeem_tx_hash` quand on résout.

## `polymarket.py` — API publique

Fichier unique. Aucun autre module n'importe `py_clob_client`, `web3`, `eth_account`, ni ne fait de requests vers `clob.polymarket.com` / `gamma-api.polymarket.com`. Le jour où l'API change, un seul fichier à auditer.

```python
class PolymarketClient:
    def __init__(
        self,
        private_key: str,                 # POLYGON_PRIVATE_KEY
        eoa_address: str,                 # POLYGON_EOA_ADDRESS
        proxy_address: str,               # POLYMARKET_PROXY_ADDRESS = funder
        rpc_url: str,                     # POLYGON_RPC_URL
        clob_host: str = "https://clob.polymarket.com",
        gamma_host: str = "https://gamma-api.polymarket.com",
        slippage_bps: int = 200,
        timeout: float = 10.0,
    ):
        # web3 client pour Polygon RPC (balanceOf USDC, sendTransaction redeemPositions)
        self.web3 = Web3(Web3.HTTPProvider(rpc_url, request_kwargs={"timeout": timeout}))
        self.usdc = self.web3.eth.contract(address=USDC_POLYGON_ADDRESS, abi=ERC20_ABI)
        self.ctf = self.web3.eth.contract(address=CTF_EXCHANGE_ADDRESS, abi=CTF_ABI)
        # py-clob-client : signing + ordering (signature_type=2 = proxy wallet)
        self.clob = ClobClient(
            host=clob_host, key=private_key,
            chain_id=137, signature_type=2, funder=proxy_address,
        )
        self.clob.set_api_creds(self.clob.create_or_derive_api_creds())

    # ---------- Balances ----------
    def get_usdc_balance(self) -> float:
        """USDC free sur la proxy wallet (Polymarket funder)."""

    def get_matic_balance(self) -> float:
        """MATIC sur l'EOA (utilise pour gas redemption)."""

    # ---------- Market discovery ----------
    def find_btc_15m_market(self, window_start_unix_s: int) -> dict:
        """GET {gamma}/markets/slug/btc-updown-15m-<unix_s>.
        Returns {condition_id, up_token_id, down_token_id, end_date_iso, status}.
        Raises MarketNotFoundError on 404.
        Retry 1x on transient HTTP errors."""

    # ---------- Order placement ----------
    def place_fak_order(self, token_id: str, usdc_amount: float) -> dict:
        """FAK BUY de usdc_amount USDC sur token_id.
        1. Read best ask via GET /book?token_id=<...>
        2. limit_price = min(best_ask + self.slippage_bps / 10000, 0.99)
        3. Sign EIP-712 FAK order (12 fields). amount = usdc_amount en base units (10^6).
        4. POST /order via py-clob-client.
        Returns {order_id, status, fill_price_avg, shares_filled, usdc_spent, fee_usdc, error_msg, raw}.
        status in {"filled", "rejected", "failed"}.
        - filled : >0 shares acquired (peut etre partial).
        - rejected : 0 shares filled (book moved, signature invalid, etc.).
        - failed : HTTP / network error after 1 retry.
        """

    # ---------- Settlement ----------
    def read_settlement(self, slug: str) -> dict | None:
        """GET {gamma}/markets/slug/<slug>. Si market.closed=True ET outcomePrices contient "1",
        retourne {resolved: True, winning_outcome: "UP"|"DOWN", resolved_at_iso: str}.
        Sinon (pas closed, ou pas encore de prix "1") retourne None."""

    # ---------- Redemption ----------
    def redeem(self, condition_id: str) -> str | None:
        """Appel on-chain CTF.redeemPositions(pUSD, 0x0...0, condition_id, [1, 2]).
        Cible : CTF_CONTRACT_ADDRESS (= 0x4D97...6045). Collateral : pUSD (= 0xC011...DFB).
        Le proxy wallet doit signer (sig_type=2 implique que l'EOA signe pour le compte du Safe).
        Returns tx_hash si tx soumise et confirmee, None si ECHEC (insufficient gas, revert, etc.).
        Logs full traceback en cas d'erreur. Ne raise pas — la perte est juste le redeem_tx_hash null."""
```

### Constants utilisees

```python
# USDC.e on Polygon — what the user deposits and what we read for balance / sizing.
# Polymarket proxy wallet (signature_type=2) holds USDC.e directly.
USDC_POLYGON_ADDRESS = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"   # 6 decimals
USDC_DECIMALS = 6

# pUSD — Polymarket's 1:1 wrapper of USDC.e used as collateral inside the CTF Exchange.
# redeemPositions() returns pUSD to the funder, not USDC.e directly. The proxy wallet
# (Gnosis Safe) auto-unwraps for the UI but we should verify empirically at smoke test
# that USDC.e balance reflects redemptions within the same cycle.
PUSD_POLYGON_ADDRESS = "0xC011a7E12a19f7B1f670d46F03B03f3342E82DFB"   # 6 decimals

# ConditionalTokens Framework on Polygon — target of redeemPositions() on-chain calls.
CTF_CONTRACT_ADDRESS = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"

# Polymarket CTF Exchange — the EIP-712 verifyingContract used by the order signer.
# We do not call this directly; py-clob-client uses it internally. Documented here for audit.
CTF_EXCHANGE_VERIFYING_CONTRACT = "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"

CHAIN_ID_POLYGON = 137
SIGNATURE_TYPE_PROXY = 2                                              # py-clob-client constant
```

**A vérifier au smoke test** :
1. La convention slug : `btc-updown-15m-<window_START_unix_s>` ou `<window_END_unix_s>` ?
2. Le balance USDC.e du proxy se met-il à jour automatiquement après `redeemPositions()` ? Si non, il faut lire pUSD aussi et faire `total = USDC.e + pUSD` pour le sizing.

### Pourquoi web3 direct pour les balances

Plutôt que `GET /balance` côté Polymarket, on lit `USDC.balanceOf(proxy_address)` directement on-chain. Plus rapide (~50ms), pas de rate limit Polymarket, source de vérité indiscutable.

### Pourquoi FAK et pas FOK

- **FOK** rejette tout si la full size ne s'exécute pas au limit → pour un marché 15min on perd le signal régulièrement.
- **FAK** accepte un partial fill et tue le reste → on récupère ce que le book offre au prix qu'on accepte. Le `shares_filled` réel est tracé dans `bets.csv` pour qu'on puisse mesurer le slippage cumulé.

### Slippage cap

Limit = `min(best_ask + slippage_bps / 10000, 0.99)`. Défaut 200 bps = 2 cents au-dessus du meilleur ask. Le tick size minimum sur Polymarket est 0.01, donc 200 bps ≈ 2 ticks. Override via `--slippage-bps <int>`.

### Pourquoi `py-clob-client` v1 et pas v2

La recherche a confirmé que `py-clob-client` v2 ne supporte pas `signature_type=2` (proxy wallet — c'est notre cas, puisque Polymarket utilise un smart-contract wallet pour le proxy). v2 ne supporte que `signature_type=0` (EOA classique) et `signature_type=3` (POLY_1271 deposit wallet). Tant qu'on garde le pattern EOA + proxy, on reste sur v1. Pin : `py-clob-client==0.21.*`.

## CLI

`live.py` ne prend plus qu'**un seul flag CLI** : `--dry-run`. Tout le reste passe par `.env`.

```
python live.py                  # mode lu depuis LIVE_MODE dans .env (true = real, false / absent = paper)
python live.py --dry-run        # si LIVE_MODE=true : construit + signe l'ordre mais skip le POST /order
                                 # si LIVE_MODE=false : --dry-run est un no-op (paper mode est deja sans ordre)
```

Pourquoi env-only au lieu d'un flag CLI :
- En production le daemon est lancé par Docker (`CMD ["python", "live/live.py"]`). On ne touche pas au `CMD` — on édite `.env` et on fait `docker compose restart live-team`.
- Une VPS dans une région non-bloquée a `LIVE_MODE=true`, la machine locale à la maison reste sur `LIVE_MODE=false`. Même image Docker, deux comportements.
- Évite le risque d'oublier le flag et de placer des ordres réels en pensant être en paper.

`--dry-run` execute tout sauf `clob.post_order()` :
- Hit Gamma pour le market discovery (vraie HTTP call).
- Lit balance via web3 (vraie HTTP call).
- Construit + signe l'ordre (vraie signature EIP-712).
- Skip le POST.
- Write bets.csv avec `order_status="dry_run"`, `polymarket_order_id=null`, `outcome="pending"` qui ne sera jamais résolu (le row ne sera pas catch-uppe au startup parce que dry_run ≠ pending).

C'est le mode pour tester l'environnement complet sur la VPS avant de basculer en real money.

Tunables qui étaient des flags dans la version précédente du spec, déplacés vers `.env` pour cohérence :
- `SLIPPAGE_BPS` (défaut `200`) — override pour ajuster le slippage cap après quelques ordres.
- `BET_FRACTION` (défaut `0.10`) — utile pour le test à 1% décrit en `Vérifications post-implémentation`.
- `BET_MIN_USDC` (défaut `2.50`).

## Env vars

`.env` (gitignored) doit contenir, en plus des 2 vars Telegram existantes :

```
# === MODE SWITCH (the only one that changes day-to-day) ===
LIVE_MODE=false                       # true  -> places real Polymarket orders
                                       # false -> paper mode (current behavior, no Polymarket calls)

# === Polymarket / Polygon (requis seulement si LIVE_MODE=true) ===
POLYGON_PRIVATE_KEY=0x...             # EOA qui signe les orders
POLYGON_EOA_ADDRESS=0x...             # derivee de la cle, pour clarte (et balance MATIC)
POLYMARKET_PROXY_ADDRESS=0x...        # smart-contract wallet qui detient l'USDC
POLYGON_RPC_URL=https://polygon-bor-rpc.publicnode.com    # ou Infura, Alchemy, ...
CLOB_HOST=https://clob.polymarket.com                     # override only for testing
GAMMA_HOST=https://gamma-api.polymarket.com               # idem

# === Sizing & slippage tunables (optionnels, defaults dans le code) ===
SLIPPAGE_BPS=200                      # 200 = 2 cents au-dessus du best ask
BET_FRACTION=0.10                     # fraction du USDC libre par bet
BET_MIN_USDC=2.50                     # plancher (5 shares * ~$0.50 = Polymarket min)

# === Telegram (deja la, inchange) ===
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...
```

`.env.example` est créé en même temps que la feature pour servir de template (memory feedback : scaffold env templates proactively). Il contient tous les noms de variables ci-dessus avec des valeurs factices et des commentaires.

`live.py` au startup :
- Lit `LIVE_MODE` (default = `false` si absent ou unparseable). Parseur tolérant : `true` / `True` / `1` / `yes` → vrai ; tout le reste → faux. Log au boot : `INFO LIVE_MODE=<bool>`.
- **Si `LIVE_MODE=true`** :
  - Lit toutes les vars Polymarket. Raise `RuntimeError` clair si une seule manque.
  - Instancie `PolymarketClient.from_env()`.
  - Logue : `wallet_eoa=<...> proxy=<...> usdc=<x> matic=<y>`.
  - Si `matic < 0.5` → Telegram alerte one-time (le redeem failera sans gas).
  - Catch-up des pending bets dans `bets.csv` (voir `Comportement de récupération`).
- **Si `LIVE_MODE=false`** : aucune des vars Polymarket n'est lue. Le `PolymarketClient` n'est même pas importé (lazy import sous le `if`), donc `py-clob-client` et `web3` peuvent être absents de l'environnement Python sans rien casser. Utile pour le développement local sans les deps lourdes.

## Telegram — deux bots

### Bot opérationnel (existant — TELEGRAM_BOT_TOKEN)
Reçoit : startup, bet placé, bet résolu, stuck, low MATIC.

### Bot d'erreurs dédié (NEW — TELEGRAM_ERROR_BOT_TOKEN)

Variables `.env` :

```
TELEGRAM_ERROR_BOT_TOKEN=<token du deuxieme bot, optionnel>
TELEGRAM_ERROR_CHAT_ID=<chat id ou les erreurs vont>
```

Quand les deux sont set, `notify.install_error_handler()` (appelé tout en haut de `main()` avant tout autre code susceptible d'échouer) attache un `logging.Handler` sur le root logger. Tout `logger.error(...)` ou `logger.exception(...)` n'importe où dans `live.py`, `polymarket.py`, `live_helpers/*` est automatiquement envoyé à ce bot. Pas besoin d'ajouter `notify.send_error()` manuellement à chaque site d'erreur — Python le fait.

**Filtrage** : niveau ERROR par défaut. Les `logger.warning(...)` (retries Bitstamp, retries `find_btc_15m_market`, etc.) restent dans stdout. Si un retry échoue après son budget et propage, l'`except` qui l'attrape log à ERROR et le bot fire.

**Recursion safety** : si l'envoi à l'error bot lui-même échoue (Telegram down, mauvais token), la trace est imprimée sur stderr, **pas** rejouée dans le logger (pour éviter une boucle infinie).

**Ping vivant** : `install_error_handler()` envoie un "🟢 Error bot online" au démarrage si configuré. Si le container crash et redémarre, l'utilisateur voit ce ping plusieurs fois — c'est le signal qu'un crash s'est produit hors du périmètre du logger (segfault, OOM, etc.).

**Endpoints couverts (audit)** :

| Site | Niveau actuel | Va au bot ? |
|---|---|---|
| `live.py` `except Exception as e: logging.exception("Cycle failed: %s", e)` | ERROR | ✓ |
| `live.py` PolymarketClient init failure (RuntimeError ou autre) | ERROR (logging.error / .exception avant `sys.exit`) | ✓ |
| `live.py` `place_real_bet` — `get_usdc_balance` / `find_btc_15m_market` / `place_fak_order` exceptions | ERROR (logging.exception) | ✓ |
| `polymarket.py` `redeem()` exception + on-chain revert + bad condition_id | ERROR (logger.exception / logger.error) | ✓ |
| `polymarket.py` `place_fak_order` HTTP failure après retry | ERROR (logger.error) | ✓ |
| `polymarket.py` `read_settlement` transient | WARNING (caller retries) | non |
| `polymarket.py` `find_btc_15m_market` transient retry | WARNING | non |
| `live.py` "Market not found" | WARNING (déjà notifié via send_bet_placed status=skipped_market_not_found) | non |
| `notify.py` `_send_raw` Telegram échec | WARNING | non (éviter récursion) |

## Telegram — nouveaux événements

`notify.py` ajoute trois fonctions :

```python
def send_bet_placed(direction, usdc_amount, fill_price, confidence,
                    target_open_iso, target_close_iso, order_id, status):
    """Une fois par ordre tente, qu'il soit filled ou rejected."""

def send_bet_resolved_real(bet_id, direction, outcome, source,
                            realized_pnl_usdc, fill_price, close_price):
    """Une fois par bet resolu (en plus du send_resolution paper existant)."""

def send_stuck(slug, target_close_iso, cycles_pending):
    """Alerte ops : un bet est pending depuis >6 cycles."""

def send_low_matic(matic_balance):
    """One-time au boot ou apres un redeem failed pour cause de gas."""
```

Les messages utilisent le même format que `send_resolution` actuel (HTML léger, emoji direction).

## Docker

`live/Dockerfile` — diff minimal :

```diff
 RUN pip install --no-cache-dir --timeout 300 --retries 5 \
         requests==2.32.* \
         pandas==2.2.* \
         numpy==2.1.* \
         xgboost==2.1.* \
         scikit-learn==1.5.* \
         scipy==1.14.* \
-        rich==13.*
+        rich==13.* \
+        py-clob-client==0.21.* \
+        web3==7.* \
+        eth-account==0.13.*
```

`docker-compose.yml` — pas de changement. Le `env_file: - path: .env` lit déjà tous les vars (existantes + nouvelles).

Image grossit de ~50 MB (web3 stack). Toujours largement sous 1 GB.

## Comportement de récupération

### Daemon restart

Au boot, si `LIVE_MODE=true` est actif :
1. Connect Polymarket → derive API creds. Si echec (clé mal formée, etc.) → log `ERROR`, exit avec code non-zero. Docker `restart: unless-stopped` retentera, ce qui spam les logs mais c'est volontaire (pas d'orderpost silencieux).
2. Log `INFO startup : wallet_eoa=<...> proxy=<...> usdc=<x> matic=<y>`.
3. **Catch-up des pending bets** :
   - Lit `bets.csv` en entier.
   - Pour chaque row `outcome=pending` ET `target_close_ms < now()` :
     - Applique la même résolution que dans la cycle loop step 3.
4. Démarre la boucle principale.

Pas de catch-up sur les signaux ratés : si le daemon était down pendant des frontières 15min, ces fenêtres sont fermées, les marchés Polymarket potentiellement aussi. On reprend à la prochaine frontière.

### Matrice de pannes

| Où | Panne | Comportement | Telegram ? |
|---|---|---|---|
| Polymarket — `find_btc_15m_market` | 404 (slug introuvable) | `bets.csv` row `order_status=skipped_market_not_found`, no order. | Oui |
| Polymarket — `find_btc_15m_market` | HTTP 5xx / network | 1 retry après 1s. Si re-échec → même que 404. | Oui |
| Balance check | `usdc_balance < $2.50` | `order_status=skipped_balance_low`. | Oui (max 1x/jour) |
| `place_fak_order` | FAK partial fill | Tracé comme `filled` avec `shares_filled` < bet attendu. PnL math correct sur l'actuel. | Non (c'est un fill) |
| `place_fak_order` | FAK 0 shares fillées (book moved) | `order_status=rejected`, `outcome=loss`, `realized_pnl=0`. Signal gaspillé. | Oui |
| `place_fak_order` | HTTP / signature error | 1 retry. Si re-échec → `order_status=failed`, `outcome=loss`, `realized_pnl=0`. | Oui |
| `read_settlement` | not resolved 1er cycle | Leave pending, retry next cycle. | Non |
| `read_settlement` | pending après 1 cycle complet | Fallback Bitstamp candle. `outcome=bitstamp_win/loss`, `source=bitstamp_fallback`. Tente quand même `redeem()` si `bitstamp_win`. | Non |
| `read_settlement` | pending après 6 cycles (90min) | `outcome=stuck`. No further auto-action. | Oui — investigation manuelle |
| `redeem()` | Tx revert on-chain | `redeem_tx_hash=null`, log revert reason. Les shares restent on-chain, retry au prochain cycle pendant le scan des pending. | Oui |
| `redeem()` | Insufficient MATIC pour gas | Idem, `redeem_tx_hash=null`. Le bet reste comptablement `win`. | Oui avec balance MATIC actuelle |
| Daemon | uncaught exception dans une cycle | Log full traceback, `time.sleep(5)`, continue. Docker `restart: unless-stopped` couvre les crashs process. | Oui |
| Container | killed mid-cycle | Startup catch-up gère les pending bets. Les ordres in-flight déjà confirmés par CLOB sont dans `bets.csv`. | — |

## Vérifications post-implémentation

À exécuter sur la VPS, dans cet ordre :

1. **Module load** :
   `cd backtest/modele_updown/15min/live && python -c "from polymarket import PolymarketClient; PolymarketClient.from_env()"`
   Doit instancier, lire les balances, et logger `wallet_eoa=... proxy=... usdc=... matic=...`.

2. **Market discovery smoke** :
   `python -c "from polymarket import PolymarketClient; c = PolymarketClient.from_env(); print(c.find_btc_15m_market(<unix_s_prochaine_frontiere>))"`.
   Doit retourner un dict avec `condition_id`, `up_token_id`, `down_token_id`, `end_date_iso`.
   **Critique** : confirmer la convention slug (`window_start_unix_s` vs `window_end_unix_s`).

3. **Dry-run end-to-end** :
   Set `LIVE_MODE=true` dans `.env`, puis `python live.py --dry-run` pour un cycle.
   Doit : hit Gamma, lire balance, signer l'ordre, **ne pas** poster, écrire la row `dry_run` dans `bets.csv`.
   Vérifier que les valeurs (slug, token_id, bet_usdc, fill_price simulé) sont cohérentes avec l'UI Polymarket.

4. **Live order — petit montant** :
   Set `LIVE_MODE=true` et `BET_FRACTION=0.01` (1% du balance) dans `.env` le temps du test → ~$1 par ordre. `python live.py`. Laisser tourner 5 cycles. Vérifier :
   - `bets.csv` contient des rows `filled`.
   - `outcome` passe de `pending` à `win/loss` au cycle suivant.
   - `redeem_tx_hash` populé sur les wins, visible sur Polygonscan.
   - `usdc_balance` post-redeem augmente (ou diminue) comme attendu.

5. **Restart recovery** :
   Pendant qu'un bet est `pending`, `docker compose restart live-team`. Au reboot, le bet doit être résolu en 1-2 cycles.

6. **Slippage tuning** :
   Après ~20 ordres, calculer `(usdc_spent / shares_filled) - (best_ask au moment de l'ordre)` pour estimer le slippage moyen. Ajuster `--slippage-bps` si nécessaire.

## Hors-scope explicite

- **Bankroll caps / circuit breakers** : pas de stop automatique sur drawdown. À ajouter si nécessaire après quelques semaines d'observation.
- **Position management post-entry** : impossible sur les binaires Polymarket (no exit before resolution).
- **Multi-pair** (ETH, SOL, etc.) : BTC uniquement, comme aujourd'hui.
- **Hot-reload du modèle** : restart container après chaque retrain.
- **UI web / dashboard** : Telegram + CSVs suffisent.
- **Provisioning VPS automatique** : config manuelle, hors scope du code.
- **Tests unitaires** : aucun (convention projet, pas de pytest).
- **Paper-mode replacement** : `live.py` sans flag reste l'implementation paper actuelle. Pas de migration.
- **Stratégies de sizing alternatives** (Kelly, anti-martingale, etc.) : pas v1.
- **WebSocket subscribe pour fills temps réel** : on poll via le response du `post_order` synchrone, suffisant pour FAK.
