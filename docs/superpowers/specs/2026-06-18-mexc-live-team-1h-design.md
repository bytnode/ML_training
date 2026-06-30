# Spec: Daemon live MEXC futures pour la team 1h (`live_team/`)

**Date** : 2026-06-18
**Statut** : approuvé (design validé par l'utilisateur, prêt pour implémentation)

## Problème

Le backtest `backtest/modele_updown/1h/modele_team/main.py` simule une stratégie de trading sur BTC : à chaque clôture de bougie **1h** il interroge la team `modele_up` + `modele_down`, applique la règle de cohérence (UP / DOWN / PASS / CANCEL), et — pour chaque pari pris — simule une **entrée maker** + une **gestion de sortie intrabar** (SL temporel décroissant, take-profit, trailing, timestop) sur le chemin 1-minute de la bougie suivante. Tout est offline, sur parquet historique.

L'objectif : un **daemon live autonome** qui exécute exactement cette stratégie en argent réel sur **MEXC futures** (`BTC_USDT` perpétuel), avec les **mêmes modèles et les mêmes paramètres** que `config_main.py`.

Différence fondamentale avec le daemon Polymarket existant (`live/`) : Polymarket est un marché **binaire qui se résout tout seul** à la clôture — aucune position vivante, aucune sortie à gérer. Une position MEXC futures est **ouverte, à levier, et vivante** : il faut un **position manager intrabar** complet (SL/TP/trailing/timestop sur prix réel, ordres réels, gestion du levier et du risque de liquidation). Cette couche est **100% neuve** ; le reste (fetch, features, prédiction, team rule, notify, boucle anti-crash, mode dégradé) est réutilisable.

Contexte API critique : l'API **Futures de MEXC a été désactivée de 2022 à mars 2026** (erreur `1002 Contract not allow place order`), puis **rouverte le 31 mars 2026**. Une clé retail **KYC** avec la permission « Futures / Order Placing » peut désormais placer des ordres via `POST /api/v1/private/order/create` (domaine `api.mexc.com`, **pas** l'ancien `/order/submit`). **Pas de testnet API** (le démo MEXC est manuel uniquement).

## Décisions clés

| # | Sujet | Choix |
|---|---|---|
| 1 | Exchange / instrument | **MEXC futures**, `BTC_USDT` perpétuel (USDT-M). Clés API dans `.env`. |
| 2 | Modèles | `1h/modele_up` + `1h/modele_down` (copies runtime embarquées), **identiques à `modele_team`**. xgboost **épinglé `3.2.*`** (skew `base_score` connu sinon). |
| 3 | Deux sources de prix | **Prédiction** : features sur **Bitstamp** (BTC/USD + ETH/USD), comme l'entraînement → zéro skew de distribution. **Exécution & gestion** : prix **MEXC** (entrée, monitoring 1-min, SL/TP/trailing, PnL). Basis Bitstamp↔MEXC négligeable sur BTC. |
| 4 | Granularité du monitoring | **Clôture de bougie 1-minute MEXC** (réplique exacte du backtest, qui évalue les sorties sur le close 1-min, pas sur les mèches). |
| 5 | Boucle | **Tick unique 1-minute.** Chaque minute : gérer positions ouvertes + entrées en attente ; au top de l'heure *en plus* : prédire + décider l'entrée. |
| 6 | Entrée maker | Post-only (`type=2`). LONG = buy-limit **0.01% sous** le marché MEXC ; SHORT = sell-limit 0.01% au-dessus. **3 tentatives de 5 min**, re-pricée au marché courant à chaque tentative ; non rempli après 15 min → **opportunité annulée** (réplique le « missed » du backtest). |
| 7 | Sortie — politique d'exécution | **Sortie en profit** (`pp>0`, ex. TP / trailing gagnant) : **3× post-only reduce-only de 20 s** (0.01% passif), puis **market taker** si non rempli. **SL + toute sortie à perte** (`pp≤0`) : **market reduce-only taker immédiat** (exécution garantie). |
| 8 | Seuils de sortie | Constantes **brutes** de `config_main.py` (TP/trailing en décimal de prix ; SL via `_sl_threshold(j, max_steps, START, END, FLOOR)`). EXPOSURE **n'intervient pas** dans les seuils (uniquement le sizing) → SL/TP/trailing **identiques au backtest**. Horloge `j`/timestop **ancrée à l'open de la bougie 1h** (fidèle backtest). |
| 9 | Concurrence | Jusqu'à **2 positions** simultanées (mathématiquement ≤2 car max_hold <2h), **mode HEDGE** (chaque position son `positionId`/`externalOid` + état trailing/anchor). 3ᵉ signal pendant 2 positions ouvertes → sauté. |
| 10 | Sizing / compte | Levier **10**, marge **ISOLATED**, mode **HEDGE**. Par entrée : `marge = 0.50 × solde_futures_courant` (**composé**), `notionnel = marge×10`, `vol = floor(notionnel / (prix × contractSize))`, `contractSize=0.0001 BTC` lu via `/contract/detail`. EXPOSURE effectif = 5 ⇒ % capital identique au backtest. |
| 11 | Architecture client | Un seul fichier **`mexc.py`** (client REST+WS maison, comme `polymarket.py`). Signature HMAC-SHA256, `/order/create`, WS public (kline) + privé (order/position/asset). Aucun autre module ne touche MEXC. |
| 12 | Validation | **Réel direct, micro-volumes** (1 contrat = 0.0001 BTC ≈ quelques $) puis montée à 50%. Toggle **`DRY_RUN`** (signe/prépare mais ne POST pas) conservé pour tester connexion/signature sans risque. Pas de mode paper simulé (pas de testnet de toute façon). |
| 13 | Frais | Maker **0.01%**, taker **0.05%** (barème API MEXC). À renseigner dans `config_team.py`. |
| 14 | Notifications | Telegram **2-bots** (opérationnel + erreur auto-câblé via `logging.Handler`), réutilisé du daemon existant. Nouveaux events : position ouverte / SL / TP / trailing / timestop / liquidation. |
| 15 | Runtime | VPS **Suisse** (non géo-bloqué). KYC + permission Futures à activer sur la clé API (côté utilisateur). |
| 16 | Hors scope v1 | Funding-rate explicite (négligeable <2h ; capté dans le PnL réalisé réel), circuit breakers, hot-reload modèle, UI/dashboard, multi-paire. |

## Layout

```
backtest/modele_updown/1h/
├── modele_up/                 # paquet entrainement (inchangé)
├── modele_down/               # paquet entrainement (inchangé)
├── modele_team/               # backtest team (inchangé)
└── live_team/                 # ← NOUVEAU daemon, self-contained
    ├── live.py                # boucle 1-min : monitoring + entrées + prédiction horaire
    ├── mexc.py                # client REST+WS maison (seul fichier touchant MEXC)
    ├── position.py            # Position (dataclass) + PositionManager (SL/TP/trailing/timestop, maker→taker)
    ├── config_team.py         # copie des constantes de config_main.py + params MEXC
    ├── live_helpers/
    │   ├── fetch.py           # Bitstamp OHLC (BTC 1h/5m/30m/4h/1d + ETH 15m→1h)
    │   ├── features_up.py     # 6 features UP (stoch_k_14, rsi_6, hour_cos, ema_compression_20, bars_since_low_50, ret_skew_24)
    │   ├── features_down.py   # 6 features DOWN (stoch_k_14, rsi_6, autocorr_12, hour_cos, ef_ret4, ef_upwick)
    │   ├── predict.py         # charge les 2 modèles, _bucket_threshold, filtre cross-model, team rule
    │   └── notify.py          # Telegram 2-bots
    ├── modele_up/             # copies runtime : btc_updown_model.json + .thresholds.json + config.py
    ├── modele_down/           # copies runtime : idem + data.py (helpers importés par features_down)
    ├── logs/  cycles.csv · positions.csv
    ├── .env.example
    ├── Dockerfile · docker-compose.yml · Dockerfile.dockerignore
    └── CLAUDE.md
```

## Boucle (`live.py`) — tick 1-minute

Boucle infinie ; chaque itération s'aligne sur la frontière de minute UTC suivante. Tout le corps dans un `try/except` : `KeyboardInterrupt` → arrêt propre ; toute autre exception → `logging.exception(...)` (déclenche le bot Telegram d'erreur) + `sleep(5)` + on continue. **Le daemon ne meurt jamais sur une erreur de cycle.**

À chaque minute, sur la **clôture de la bougie Min1 MEXC** qui vient de fermer :

1. **Gérer les positions ouvertes** — pour chaque `Position` : récupérer le close Min1 MEXC, calculer `pp` et `j` (= minutes depuis l'open de la bougie 1h d'entrée), évaluer SL > TP > trailing > timestop (ordre exact), exécuter la sortie selon la politique (§ Position manager).
2. **Gérer les entrées en attente** — pour chaque ordre d'entrée maker posté : s'il est rempli → créer la `Position` ; s'il a dépassé 5 min sans fill → annuler, re-pricer, re-poster (jusqu'à 3 tentatives) ; après 3 échecs → abandonner l'opportunité.
3. **Au top de l'heure (`xx:00`) uniquement, en plus** :
   a. `freshness_retry` Bitstamp : confirmer que la bougie 1h qui vient de fermer est publiée.
   b. Rafraîchir les buffers OHLCV (warm tail-only), build features 6 UP + 6 DOWN.
   c. Prédire les 2 modèles → `_bucket_threshold` (skip_cells, bad_hours, deltas config) → **filtre cross-model** (si `p_up ≥ 0.53` → rejet DOWN) → `take` par modèle.
   d. **Team rule** : UP seul → long ; DOWN seul → short ; les deux → CANCEL ; aucun → PASS.
   e. Si UP/DOWN **et** <2 positions ouvertes : lancer une entrée maker (§ Entrée).
   f. Append 1 ligne `cycles.csv`.
4. Notifications Telegram + render Rich + log de timing.
5. Reconciliation : au **démarrage**, lire `/position/open_positions` et reconstruire l'état (reprise du monitoring des positions déjà ouvertes après un redémarrage). Vérifié aussi périodiquement.

## Entrée maker (3 tentatives × 5 min)

- À l'instant de décision (top de l'heure), lire le prix MEXC courant (WS ticker / dernier kline).
- Poster un ordre **post-only `type=2`** : LONG → `price = mkt × (1 − 0.0001)` ; SHORT → `price = mkt × (1 + 0.0001)`. Arrondi au `priceUnit`. `vol` = sizing (§ Sizing). `externalOid` unique (idempotence).
- Tant que non rempli, après **5 min** : `cancel` + re-lire le marché + re-poster au nouveau 0.01% offset. **Max 3 tentatives = 15 min.**
- Rempli (total ou partiel) → créer `Position(entry=prix_fill_moyen, vol=qty_remplie, side, anchor_ts=open_bougie_1h, peak=0, model_dir)`. Partiel : on gère la quantité réellement remplie.
- 3 tentatives sans fill → annuler le dernier ordre, **abandonner** (pas de trade ce cycle).

## Position manager (`position.py`) — cœur neuf

`Position` (persistée pour survivre aux redémarrages) : `id`, `side` (long/short), `entry`, `vol`, `anchor_ts` (open de la bougie 1h), `peak` (high-water-mark du profit), `positionId` MEXC, état (open/closing/closed).

À chaque clôture Min1 MEXC, pour chaque position ouverte :
- `pp = close/entry − 1` (long) ou `pp = 1 − close/entry` (short).
- `j = (now_ms − anchor_ts) // 60000` (minutes depuis l'open de la bougie).
- `max_steps` = `UP_MAX_STEPS` (90) ou `DOWN_MAX_STEPS` (105) selon le côté.
- Tests dans l'**ordre exact** (premier déclenché gagne) :
  1. **SL** : `sl_th = _sl_threshold(j, max_steps, *_SL_START_DEC, *_SL_END_DEC, *_SL_FLOOR_DEC)` (ou `*_STOP_LOSS_DEC` constant si time-decay off) ; si `pp ≤ sl_th` → sortie `stop_loss`.
  2. **TP** : si `pp ≥ *_TAKE_PROFIT` → sortie `take_profit`.
  3. **Trailing** : `peak = max(peak, pp)` ; si `peak ≥ *_TRAIL_ACTIVATE` et `pp < peak − *_TRAIL_GAP` → sortie `trailing`.
  4. **Timestop** : si `j ≥ max_steps` → sortie `timestop`.

**Exécution de la sortie** :
- `pp > 0` (profit) : boucle 3× → poster un **post-only reduce-only** (long : sell-limit 0.01% au-dessus du marché ; short : buy-limit 0.01% en dessous), attendre 20 s, si rempli → fini ; sinon `cancel` + re-pricer. Après 3 échecs → **market reduce-only taker**.
- `pp ≤ 0` (SL ou toute perte) : **market reduce-only taker immédiat**.
- À la fermeture effective : lire le PnL réalisé réel (frais + funding inclus) via l'API, écrire la ligne `positions.csv`, libérer le slot, notifier.

Les constantes `*_TAKE_PROFIT / *_TRAIL_ACTIVATE / *_TRAIL_GAP / *_SL_*_DEC / *_MAX_STEPS` proviennent de `config_team.py` (copie de `config_main.py`). **EXPOSURE n'est jamais appliqué aux seuils** (uniquement au sizing).

## `mexc.py` — client REST+WS maison

- **Signature** HMAC-SHA256 (hex). En-têtes privés : `ApiKey`, `Request-Time` (ms), `Signature`, `Content-Type: application/json`, `Recv-Window` optionnel. POST : signer `accessKey + timestamp + jsonBody` ; GET/DELETE : `accessKey + timestamp + paramString` (params triés alpha). Sync horloge pour éviter code 513.
- **REST** (`https://api.mexc.com`) : `POST /api/v1/private/order/create` (place), `/order/cancel`, `/order/cancel_all`, `GET /order/get/{id}`, `/position/open_positions`, `POST /position/change_leverage`, `GET /account/assets`, `GET /api/v1/contract/detail`, `GET /api/v1/contract/kline/{symbol}`. Retry backoff sur 510 ; idempotence via `externalOid` (≤32 c). Rate limit `/order/create` = 4/2s (large pour du 1h).
- **WS** (`wss://contract.mexc.com/edge`) : public `sub.kline {Min1, Min60}` + `sub.ticker` ; privé `login` (HMAC) → `push.personal.order / position / asset`. **Ping toutes les 10-20 s** ; reconnexion = ré-login + re-subscribe + backoff exponentiel. Ne jamais se fier au seul WS pour l'état critique → reconcilier via REST au boot et périodiquement.
- Champs d'ordre : `symbol, price, vol (contrats), side (1=open long, 3=open short, 2=close short, 4=close long), type (1=limit/2=post-only/5=market), openType (1=isolated), leverage, positionMode (hedge), reduceOnly, externalOid`.

## Logs

- **`cycles.csv`** — 1 ligne par cycle horaire : `cycle_n, ts, bar_close_utc, OHLC bougie 1h, p_up/th_up/take_up, p_dn/th_dn/take_dn, volatility_state/is_weekend/hour, team_decision, n_positions_ouvertes, stats cumulées`.
- **`positions.csv`** — 1 ligne par position (cycle de vie complet) : `pos_id, open_ts, side, entry, vol, marge, leverage, anchor_ts, exit_ts, exit_price, exit_reason (stop_loss/take_profit/trailing/timestop), pp_brut, fees, funding, pnl_usdc_realise, pnl_pct_equity, balance_pre/post, order_status, externalOid`.
- Restauration des stats cumulées depuis la dernière ligne au boot (résistant aux rebuilds container).

## Robustesse & vitesse

- **Anti-crash** : boucle `try/except`, bot Telegram d'erreur câblé en premier (capte même les erreurs de bootstrap).
- **Reconnexion** : WS ping/reconnect ; REST retry/backoff ; `freshness_retry` Bitstamp ; reconciliation des positions au boot.
- **Idempotence** : `externalOid` sur chaque ordre (pas de doublon sur retry).
- **Vitesse** : WS push (pas de polling) pour prix + order/position ; sessions HTTP keep-alive ; buffers de features pré-chauffés (warm tail-only) ; hot-path d'entrée = signer + POST uniquement.
- **Mode dégradé** : `notify` no-op si creds Telegram absentes ; `DRY_RUN` signe sans poster.

## Prérequis (côté utilisateur)

1. **KYC** complété sur MEXC + permission **« Futures / Order Placing »** activée sur la clé API.
2. Compte MEXC réglé en mode **Hedge** + marge **Isolated** (ou réglé automatiquement par le daemon au boot).
3. Runtime sur VPS **Suisse** (non géo-bloqué — confirmé non bloqué).
4. `pip` : `requests pandas numpy xgboost==3.2.* scikit-learn scipy rich websocket-client` (versions exactes épinglées dans le Dockerfile).

## Validation

1. `DRY_RUN=true` : vérifier connexion, signature, lecture balance/positions/klines, build features + prédiction (chemin complet sans poster).
2. Réel **vol minimal** (1 contrat) sur quelques cycles : vérifier entrée maker, fill, monitoring 1-min, sortie maker→taker, écriture `positions.csv`, PnL réalisé.
3. Monter `BET_FRACTION` à 0.50 une fois le chemin validé.

## Notes de portage (anti-skew)

- xgboost **3.2.0** obligatoire (`base_score` embarqué : up=0.5214, down=0.5029).
- Features construites **byte-identiques** aux builders d'entraînement (réutiliser les helpers de `modele_down/data.py` / `add_*.py` via import quand possible, comme `features_down.py` existant). Valider row-by-row contre le parquet avant le live.
- `_bucket_threshold` porté à l'identique ; les deltas (`ADAPTIVE_BAD_DAY_DELTA`, `ADAPTIVE_SOFT_HOURS_DELTA`) sont lus depuis `config.py` au runtime → embarquer les `config.py` des 2 modèles.
- Jointures MTF **no-lookahead** (`searchsorted side='right' − 1`, target = `T − tf_step`, `T = open_1h + 1h`).
