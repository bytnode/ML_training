# CLAUDE.md — live (OKX daemon, 1h team)

Live trading daemon for `modele_team`'s 1h up+down team system, on **OKX**
(`BTC-USDT-SWAP`, HEDGE mode, ISOLATED margin). Lives **inside** `modele_team/`
(not a sibling package) and imports `modele_team/{config_main.py,filters.py,
okx_fees.py}` directly (still shared, unmodified, with the backtest). Design
spec: `docs/superpowers/specs/2026-07-09-okx-live-team-1h-design.md`.

**⚠️ Campagne ALIGNEMENT GRILLE 1-MIN (2026-08-23)** — la campagne du 2026-08-14
avait aligné la **décision** (config + filtres + overrides figés) ; celle-ci
aligne l'**exécution**. Constat de départ : la décision était déjà partagée
(même `config_main.py`, même `filters.py`, mêmes `.json` de modèle — vérifié par
hash), mais le moteur de sortie live lisait un objet différent de celui du
backtest. Ce qui a changé :

- **Le prix de référence des sorties.** `main.py::_up_managed_exit` /
  `_down_managed_exit` déroulent les **clôtures des bougies 1-min** (parquet
  `DB_1m`, clé = open-time) sur la grille `t = traded_ts + j*60s`. Le live
  échantillonnait `markPx` une fois par minute : même cadence, mais un prix pris
  à un instant quelconque de la minute et un index `j` **décalé d'un cran**
  (`elapsed_min=1` au premier tick là où le backtest est encore à `j=0`).
  `position.py` lit désormais les **mêmes objets** chez OKX
  (`api_okx.get_candles_range`, bougies 1-min fermées indexées par open-time) et
  **rejoue la boucle depuis `j=0` à chaque tick** — donc un tick manqué, un
  redémarrage ou une minute de retard sont rattrapés automatiquement au lieu de
  faire dériver l'état. `_up_managed_exit` / `_down_managed_exit` de `position.py`
  sont maintenant un **port ligne à ligne** de leurs homonymes de `main.py`.
- **Extension conditionnelle du TP DOWN (`DOWN_TPEXT_*`) implémentée**, pivots
  1-min compris (`position.swing_pivots_from`, port de `main._load_swing_pivots`).
  C'était le dernier mécanisme absent ; `DOWN_TPEXT_ENABLED=False` au backtest à
  ce jour, donc zéro changement de comportement — mais basculer le flag ne
  désynchronise plus rien.
- **Ancre d'entrée maker = le vrai P0.** Le backtest poste sa limite à
  `open_ib.get(traded_ts)` (l'**open de la bougie 1-min** d'open-time
  `traded_ts`) ; le live postait au bid/ask instantané. Il lit désormais ce même
  open chez OKX. `ord_type` passe aussi de `post_only` à **`limit`** : le
  backtest considère la limite remplie dès qu'un low 1-min descend sous P0, y
  compris quand le marché est **déjà** sous P0 au moment de poster — cas où un
  `post_only` serait rejeté par OKX et ferait perdre un trade que le backtest
  prend.
- **Élargissement sur profit (`widen`) décidé par la marche, plus par le signal.**
  `try_widen` → `register_widen_candidate` : live.py ne fait plus qu'ajouter le
  `traded_ts` du signal bloqué à `entry["widen_candidates"]` (l'exact équivalent
  du `widen_candidates` de `simulate_trades`), et c'est la marche qui élargit au
  **premier candidat atteint en profit**. L'ancienne version tranchait sur le
  prix de l'instant et perdait définitivement un candidat dont le `pp` était
  ≤ 0 à cette seconde-là.
- **Simulation de fill maker en DRY_RUN** : plus de comparaison au ticker
  instantané (qui rate tout passage intra-minute), mais le test exact du
  backtest — rempli ssi un `low` 1-min passe **strictement** sous P0 (long) /
  un `high` **strictement** au-dessus (short).
- **Position adoptée par `reconcile`** : `cTime` est planché sur la minute —
  `traded_ts_ms` est la clé de la grille 1-min, un timestamp non aligné ne
  correspondrait à aucune bougie et la marche ne démarrerait jamais.

**Parité vérifiée** (scripts jetables, non conservés) : `_up_managed_exit` /
`_down_managed_exit` backtest vs live sur **720 séries longues** (plus longues
que le plus grand horizon possible, 5760 pas) → **720/720 identiques** sur
`(reason, pp, step_ts)`, les 4 motifs de sortie représentés, toutes combinaisons
chop / macro-bull / widen / régime / palier DOWN / tpext ; **3600 séries courtes**
→ 3600/3600 conformes à la seule divergence voulue (le backtest sort `timestop`
quand son parquet s'épuise, le live renvoie `None` = « le futur n'est pas encore
arrivé »). Dispatcher `evaluate_exit` vs la sélection de paramètres de
`simulate_trades` : **2400/2400**, dont un rejeu **tick par tick** (buffer tronqué
à k bougies, k croissant) confirmant que le live déclenche au **même pas j**.
Enfin un cycle horaire complet + un tick de surveillance sur bougies OKX réelles
en DRY_RUN.

**Le buffer 1h fini ne fausse aucune décision (vérifié 2026-08-23).** Le
contexte de tendance live est calculé sur 10 300 barres, le backtest sur toute
la SQLite — l'**âge du bear** y est donc structurellement sous-estimé (il se
compte depuis le début de la série fournie). Mesuré en rejouant
`trend_context_from_series` sur la SQLite tronquée à 10 300 barres, à **2 965
ancres** réparties sur tout l'historique : écart d'âge max 8 553 h (saturation),
médian 0 h, mais **0 divergence** sur `chop_mode` et **0** sur
`up_tp_override`. La raison est structurelle : la fenêtre à SMA valide fait
1 540 barres, soit 2,1× le seuil `CHOP_BEAR_AGE_MAX_H` (720 h) — si l'âge réel
est ≤ 1 540 h il est compté exactement (la barre au-dessus de la SMA qui remet
le compteur à zéro est dans la fenêtre), sinon il sature au-dessus de 720 h et
le verdict est le même des deux côtés. ⚠️ Cette garantie **repose sur la marge
`BTC_1H_BARS - CHOP_SMA_DAYS*24 > 2 × CHOP_BEAR_AGE_MAX_H`** : augmenter
`CHOP_SMA_DAYS` ou `CHOP_BEAR_AGE_MAX_DAYS` sans agrandir `BTC_1H_BARS` la
casserait silencieusement.

**Deux écarts documentés se referment du même coup** : (1) le backtest ne lit
QUE des clôtures 1-min pour ses sorties (`low`/`high` ne servent qu'au test de
fill maker) — le live aussi désormais, donc le « wick rapide manqué entre deux
échantillons » n'est plus un écart mais un comportement **partagé** ; (2) DRY_RUN
et mode réel lisent la même source, l'ancienne divergence `ticker.last` vs
`markPx` disparaît. **Écarts restants, assumés** : le funding perpétuel (non
modélisé au backtest) ; l'exécution d'une sortie se fait au marché courant, pas
au prix exact de la clôture qui l'a déclenchée ; et en mode TAKER l'entrée
subit le slippage réel là où le backtest entre à P0.

**⚠️ Campagne ALIGNEMENT COMPLET backtest/live (2026-08-14)** — après un audit
31 agents (25 findings confirmés) : le daemon ne démarrait plus (`position.py`
importait 4 noms supprimés de `config_main.py` le 2026-08-11 — le config est
PARTAGÉ par import, pas copié) et le moteur n'avait reçu aucun mécanisme des
campagnes d'août. Porté ce jour, en parité vérifiée avec `main.py` :
- **Contexte SMA365/ER** : `filters.trend_context_from_series` (math extraite
  de `load_trend_context`, refactor prouvé bit-identique par hash) calculée
  depuis `buf_btc_1h` agrandi 800 → **10 300 barres** (`market_data.trend_ctx()`,
  cache par barre). Bumps d'entrée chop branchés (`effective_*_1(..., trend_ctx)`).
- **`evaluate_exit` reconstruit** : SL toujours time-decay ; cliquet
  `filters.ratchet_floor` (plancher = max(trailing, cliquet), recalculé du pic
  à chaque tick) ; overrides chop/macro-bull **figés à la barre de décision**
  (`_decision_ctx` → `entry["chop"|"up_tp_override"|"dn_ratchet_extra"]`,
  persistés, survivent au redémarrage) ; en chop DOWN l'override régime est
  court-circuité (parité `regime="neutral"`) ; widen multiplie le TP EFFECTIF ;
  gate `DOWN_WIDEN_ENABLED` dans `try_widen`.
- **Cross-model UP pré-cohérence** (`live.py`) : une barre double-pari donne
  DOWN (plus CANCEL) — même ordre que `main.py`.
- **Sémantique maker = backtest** : limite post-only **statique** au prix de
  départ pendant `MAKER_FILL_WINDOW_MIN`, plus AUCUN repricing 60 s
  (`ENTRY_REPRICE_EVERY_S` supprimé), fenêtre expirée = **opportunité ratée**
  (event `entry_missed`, plus jamais de bascule taker). Fill partiel : la
  portion remplie est gardée telle quelle (notional au prorata) ; si la lecture
  du fill échoue après annulation, **rien n'est posté** et on relit au tick
  suivant (remplace le bug de sur-exposition ×1.8).
- **État atomique** : `positions_state.json` écrit via tmp+`os.replace`.
- Écarts restants à l'issue de cette campagne : tpext DOWN non implémenté,
  mark-price échantillonné 60 s vs closes 1-min, funding. ⚠️ Les **deux
  premiers ont été traités** par la campagne du 2026-08-23 ci-dessus ; seul le
  funding reste.

**Full self-containment (2026-07-10, 2nd campaign)**: `live/` now carries its
**own embedded copies** of `modele_up/`, `modele_down/`, `regime/` — code only
(`api.py`, `config.py`, `data.py`, `add_*.py`, the model `.json`/`.thresholds.json`,
`regime_seed.json`), never the heavy `DB/`/`logs/`/training-only scripts. This
replaced the earlier "import the 1h/ siblings directly" design after a VPS
deploy failed: Docker's build context couldn't reach outside `modele_team/`
without shipping the whole `1h/` tree. **Tradeoff, explicitly accepted**: after
a retrain, or any edit to `modele_up`/`modele_down`/`regime`'s source, you must
manually re-copy the changed files into `live/modele_up|down/`, `live/regime/`
— nothing does this automatically, and a stale copy fails silently (same
tradeoff the repo-root `15min/live/` already lives with). `filters.py` (shared,
NOT copied) has **no sys.path bootstrap of its own** — it resolves `modele_down`/
`regime` to whatever the *caller* already put on `sys.path` (main.py -> `1h/`
siblings; live.py/market_data.py/position.py -> `live/`'s own copies). Do not
re-add a self-sufficient `sys.path.insert` to `filters.py` — Python caches
`modele_down`/`regime` in `sys.modules` by name, so whichever copy resolves
*first* wins for the rest of that process; a self-inserting `filters.py` would
silently shadow `live/`'s copies with `1h/`'s (this exact bug was caught and
fixed during the 2026-07-10 restructuring, verified via `python -c` smoke
imports from both `live/` and `modele_team/`).

**Data autonomy (2026-07-10, 1st campaign)**: `live/` never touches modele_down's
SQLite — not at startup, not ever. All market data comes from Bitstamp, fetched
once at launch and maintained incrementally (`market_data.py`); the regime
detector bootstraps from `regime/regime_seed.json` (now `live/regime/regime_seed.json`,
the embedded copy) and advances one bar at a time (`regime_live.py`),
persisting its own state inside `live/logs/`. See *Files* below.

**Bug-fix campaign (2026-07-15)**: a full audit (5 parallel independent reviews
across every file) after DRY_RUN was observed never tracking or closing
positions in prod. Root cause: `check_positions`/`try_widen` called the
ALWAYS-real `get_positions()` to validate DRY_RUN paper state — which OKX can
never show (writes are stubbed) — causing an immediate false "desync" that
wiped tracked state within a minute of every entry. Fixed by making
`position.py` DRY_RUN-aware: paper positions are never checked against
`get_positions()`, price comes from `get_ticker()` instead. The same audit
found and fixed 9 more issues:
- `config_live.is_dry_run()` didn't `.strip()` the env var — a stray trailing
  space/CRLF in `.env` silently flipped paper mode to **real trading**, no
  error, no log. Highest-severity finding of the campaign.
- `main()`'s `md.bootstrap()` was unguarded — a Bitstamp/regime-seed failure at
  startup killed the process with **zero Telegram alert** (crash before any
  logger call runs), crash-looping silently under Docker and leaving an
  already-open position unmonitored. Now wrapped: alerts via
  `notify.send_cycle_error`, then re-raises (still exits — Docker's restart
  policy is the retry mechanism — but now visibly).
- `buf_btc_4h` (feeds the per-trade DMI filter) had **zero gap protection** —
  unlike the 1h/eth buffers, a Bitstamp-omitted candle here silently corrupts
  `filters._wilder_dmi` with no crash and no alert. New
  `calcule_feature.heal_gaps()` scans a buffer for any internal gap (not just
  at one queried timestamp, cf. `backfill_gap_if_needed`) and backfills it —
  applied to all 3 buffers.
- `get_order()` in DRY_RUN always answered `"live"`, never `"filled"` — every
  paper maker entry silently fell back to taker after the fill window
  regardless of price (100% `maker_fallback_taker` in the observed logs).
  `advance_pending_entry` now simulates a realistic fill in DRY_RUN by
  comparing the current ticker to the posted price.
- Taker entries (direct and maker-fallback) recorded the pre-trade ticker
  snapshot as `avg_px`, never the real fill price — real-mode slippage was
  invisible to every downstream SL/TP/trailing check. Now polls `get_order()`
  once after a taker fill (best-effort, falls back to the snapshot).
- The regime's hourly `bootstrap_and_advance` re-fetched ~726 1h-bars + ~142
  4h-bars from Bitstamp **every single hour**, even for the routine
  single-bar advance, instead of reusing `market_data`'s buffers already in
  RAM — contradicted this file's own "never a full re-fetch" principle. Now
  serves from the in-memory buffer when it has enough history, only falls
  back to a real fetch after a long outage.
- `api_okx.OkxClient._request()` called `raise_for_status()` before reading the
  response body — OKX's own `sCode`/`sMsg` diagnostic (e.g. distinguishing a
  bad key from an unwhitelisted IP) was discarded, leaving only a bare
  `401 Client Error`. Now logs the body before re-raising.
- `live.py`'s "widened" `positions.csv` row was missing the `peak_pp`/
  `worst_pp`/`pnl_usdt` columns added by an earlier session — a real
  regression that silently misaligned that row under the fixed CSV header.
- `compute_size_contracts` floored to `min_sz` even when `notional_usd <= 0`
  (zero/negative balance, misconfigured `POSITION_FRACTION`) — would place a
  real minimum-size order instead of skipping the trade; now raises instead.
  Its `sz` formatting (`:g`) could also emit scientific notation for a very
  small `contracts` value — switched to fixed-decimal.
- `entry.get("notional_usd", 0.0)` / `entry.get("sz", "0")` only substitute the
  default when the key is *absent*, not when explicitly `null` — a
  hand-edited/corrupted state file could `TypeError` mid-tick. Switched to
  `entry.get(x) or default`.

**Pre-prod audit (2026-08-03)** — full review of the real-money path before the
first `DRY_RUN=false` switch. Most findings were latent bugs the dry-run mode
*structurally cannot surface* (every write call is stubbed). Fixes landed:
- **#1 Native catastrophe backstop** — all SL/TP is computed locally in
  `evaluate_exit` and executed only while the once-a-minute loop runs; a
  crashed/hung/desynced daemon left a 10x isolated position unprotected until
  ~ −9% liquidation. Now `_finalize_entry` places a **static conditional stop on
  OKX** (`api_okx.place_algo_order`, order-algo) per position, deliberately set
  LOOSER (UP −3% / DOWN −2.5%, `config_live.CATASTROPHE_SL_*`) than the widest
  the dynamic SL ever reaches (`*_SL_START`: −1.5%/−1%) so it NEVER fires before
  the daemon's own logic in normal operation — it only caps catastrophe when the
  daemon is blind. It lives on OKX's matching engine, independent of the process.
  Cancelled on every normal close and on desync; adopted/re-placed by reconcile.
  The dynamic exit logic (time-decay SL, trailing, timestop, regime, widen) is
  **unchanged** — the backstop never mirrors it (one fixed floor per side; the
  SL bound is invariant because widen never touches SL and regime only touches
  hold/trail). Toggle `NATIVE_BACKSTOP_ENABLED`.
- **#2 Stale paper `pending_entry`** — `get_order`'s `DRYRUN-` shortcut wasn't
  gated on `dry_run`, so a paper pending order surviving into real mode fired a
  real market order. Now gated + `PositionManager._sanitize_state_for_mode`
  purges `DRYRUN-` state on load in real mode.
- **#3 Partial maker fills** — the deadline fallback posted a market order for
  the FULL size even if the maker leg had partially filled (over-sizing). Now
  reads `accFillSz` and completes only the remainder, with a size-weighted avg_px.
- **#4 Desync debounce** — a single empty `get_positions()` read wiped a live
  position's tracking. Now requires `DESYNC_CONFIRM_TICKS` (3) consecutive
  absent reads, counter reset when the position reappears.
- **#5 Startup reconciliation** — `reconcile_positions()` adopts an OKX position
  with no local entry (lost/corrupt state, out-of-band open), clears a phantom
  local entry, and guarantees a backstop exists (adopts an existing algo rather
  than duplicating). No-op in DRY_RUN.
- **#6 Sizing on `availEq`** — sizing used total equity, so a 2nd (opposite-side)
  entry re-assumed full capital → ~100% margin, no fee/funding buffer, possible
  rejection. Now sizes on available equity and caps notional to
  `SIZING_SAFETY_BUFFER`×avail.
- **#7 Leverage confirmed at entry** — `set_leverage` failures were swallowed
  but sizing still assumed the configured leverage. Now re-set + read back via
  `get_leverage` right before each entry; notional uses the ACTUAL leverage.
- **#9 Min-size skip** — a balance below OKX's `minSz` notional silently floored
  up to a larger-than-intended order; now `compute_size_contracts` raises
  `SizeTooSmall` (past `MIN_SIZE_OVERSHOOT_TOL`) and the entry is skipped.
- **#11 Clock sync** — `OkxClient.sync_time()` at startup caps signature time to
  OKX server time (offset applied in `_iso_timestamp_ms`); alerts past
  `MAX_CLOCK_SKEW_S`. A drifted host clock would otherwise get every signed
  write rejected (code 50102).
- **#12 Secrets** — added `modele_team/.gitignore` + `live/.env.example`; the
  real keys in `live/.env` were exposed during the audit — **rotate them**,
  scope the key to Trade-only, IP-whitelist.

**Known, accepted limitations** (not fixable without a bigger redesign) —
révisées 2026-08-23, les deux écarts de prix ci-dessous **ont été refermés** par
la campagne ALIGNEMENT GRILLE 1-MIN (DRY_RUN et réel lisent maintenant la même
source, et cette source est celle du backtest) :
- `run_hourly_cycle` runs synchronously in the same single thread as
  `check_positions`/`advance_pending_entry` — a slow Bitstamp fetch
  (`freshness_retry` can block up to ~60s per pair) delays position monitoring
  for that span.
- **Funding perpétuel** (prélevé toutes les 8h sur une position réelle, surtout
  les DOWN tenus jusqu'à 12h) : non modélisé au backtest → dérive de P/L
  réel-vs-backtest. Écart de fond, pas un bug.
- **L'exécution d'une sortie se fait au marché courant**, pas au prix exact de la
  clôture 1-min qui l'a déclenchée : le backtest comptabilise `pp` à cette
  clôture, le live envoie un ordre market quelques secondes plus tard. Le filet
  catastrophe natif (#1) borne le pire cas.
- **Mode TAKER** : le backtest entre à P0 (open de la bougie tradée), le live
  entre au marché et subit le slippage réel. En mode MAKER (le défaut,
  `MAKER=True`) l'ancre est bien P0 des deux côtés.

**Symmetry principle**: every decision (coherence rule, DMI filter, the 3
UP-only filters, sidecar threshold lookup, regime wrapper) runs through
`filters.py` and `modele_up.api`/`modele_down.api` — the exact same code
`main.py` uses for the backtest. If you need to change a threshold or a rule,
change it in `config_main.py`/`filters.py`, never here.

## Files

- `live.py` — the loop. Every minute: `PositionManager.check_positions` (SL/TP/
  trailing/timestop) + `advance_pending_entry` (non-blocking maker-entry state
  machine). At `:00`: fetch Bitstamp → `calcule_feature` → predict → team rule
  → `filters.py` → `PositionManager.start_entry`.
- `api_okx.py` — **the only file that touches the OKX API**. Signed REST
  (HMAC-SHA256, OKX v5 headers). `DRY_RUN=true` stubs every write call (order/
  cancel/amend/leverage/position-mode) and logs what it *would* have done;
  reads (`get_positions`/`get_balance`/`get_ticker`/`get_order`/`get_instrument`)
  always hit the real API, even in dry-run. **`get_candles` / `get_candles_range`**
  (2026-08-23) lisent l'endpoint **public** `/market/candles` : bougies indexées
  par **open-time**, même convention que le parquet `DB_1m` du backtest et que
  Bitstamp. `get_candles_range` pagine (OKX plafonne à 300/requête) et, par
  défaut (`closed_only=True`), ne garde que les bougies **terminées**
  (`confirm=1`) — c'est ce qu'il faut partout où le backtest lit un `close`.
  Passer `closed_only=False` uniquement pour lire l'**open** d'une bougie en
  cours (l'ancre P0 d'une entrée maker : l'open est figé dès la première
  seconde). Ces deux méthodes sont la **source de vérité des prix du moteur de
  sortie**, en DRY_RUN comme en réel.
- `calcule_feature.py` — Bitstamp OHLC fetch + the 6 features of each model,
  reusing the training packages' own helpers (`add_engineered_features.py`,
  `add_alpha_features.py`, `add_eth_features.py`, `data.py`) so there is zero
  train/serve skew. **`modele_up/` and `modele_down/` both have same-named
  modules** (`data.py`, `config.py`, `add_engineered_features.py`,
  `add_eth_features.py`) — `_import_isolated` purges `sys.modules` before/after
  each package's import so one package can never silently borrow the other's
  code. Also computes `volatility_state`/`is_weekend` (the sidecar threshold
  bucket keys — not model features) with the same ATR/rolling formula as
  `data.py`. ETH input is **native 1h** (matches the current `add_eth_features.py`
  — no 15m→1h resample; the old, now-deleted `live_team` package expected a
  resample step that no longer exists).
- `position.py` — `PositionManager` + le moteur de sortie. `_up_managed_exit` /
  `_down_managed_exit` sont un **port ligne à ligne** de leurs homonymes dans
  `main.py` : ils déroulent la **même grille 1-min** (`t = traded_ts + j*60s`,
  clé = open-time), avec les mêmes seuils `config_main.py`, le même
  `filters.sl_threshold`, le même `filters.ratchet_floor` et la même extension
  TP DOWN. La grille vient d'un buffer de bougies 1-min OKX fermées
  (`_refresh_candles` / `_close_by`, alimenté par `api_okx.get_candles_range`),
  **rejoué depuis `j=0` à chaque tick** — d'où le rattrapage automatique après un
  tick manqué ou un redémarrage. `evaluate_exit` n'est plus qu'un dispatcher :
  il lit le contexte **figé à la barre de décision** (`chop`, `up_tp_override`,
  `dn_ratchet_extra`, `regime`, `down_*_override`, `widen_candidates`) et
  choisit exactement les paramètres que `simulate_trades` choisit.
  `logs/positions_state.json` persiste ce qui ne peut pas être recalculé
  (`avg_px`, `traded_ts_ms`, le contexte figé, les candidats de widen, l'entrée
  maker en cours) ; `peak_pp`/`worst_pp`/`floor_pp` y restent pour le **seul
  reporting** et ne pilotent plus aucune décision.
  **Exchange constraint**: OKX HEDGE mode shows one merged position per
  `(instId, posSide)` — a 2nd same-side entry would blend `avgPx` and corrupt
  the older entry's SL/TP/trailing tracking. So this caps at **1 slot per side
  (long, short) = 2 total**. As of 2026-07-10 the backtest enforces the exact
  same cap (`MAX_CONCURRENT_POSITIONS_PER_SIDE=1` in `config_main.py`) — this
  is no longer a live-only deviation, both systems agree.
  **`register_widen_candidate(pos_side, traded_ts_ms)`** (ex-`try_widen`,
  refondu 2026-08-23) : appelé par `live.py` juste avant qu'un signal du même
  sens soit abandonné comme "slot occupé". Il ne fait qu'**enregistrer** le
  `traded_ts` dans `entry["widen_candidates"]` — l'exact équivalent du
  `widen_candidates` de `simulate_trades`. C'est la **marche de sortie** qui
  élargit ensuite, au premier candidat atteint **en profit** (`pp > 0.0 and t in
  widen_candidates`), une seule fois, en multipliant take-profit / trailing-gap /
  max-hold par `UP_WIDEN_*`/`DOWN_WIDEN_*` — le SL et son horizon ne sont
  **jamais** élargis, comme au backtest. L'événement `widened` remonte donc
  depuis `check_positions`, plus depuis le cycle horaire. ⚠️ Ces multiplicateurs
  ont été balayés sur seulement 7 UP / 3 DOWN positions élargies *en backtest* —
  provisoires, pas une réponse validée.
- `market_data.py` — `LiveMarketData`: owns the 3 in-memory Bitstamp buffers
  (BTC 1h — **10 300 bars** depuis 2026-08-14, dimensionné par la SMA
  `CHOP_SMA_DAYS`=365 j du contexte de tendance (8 760 barres) + une fenêtre de
  mesure d'âge du bear de 1 540 barres (> 2× `CHOP_BEAR_AGE_MAX_DAYS`) ; couvre
  aussi les features des modèles et le lookback du régime ;
  BTC 4h — 300 bars, shared by the per-trade DMI filter and the regime's own
  DMI axis; ETH 1h — 400 bars). ⚠️ Ces buffers alimentent la **décision** ; les
  prix du **moteur de sortie** viennent d'ailleurs (bougies 1-min OKX, cf
  `position.py`). `.bootstrap()` fetches all three once at
  startup; `.refresh(target_ts)` merges just the newest few bars each hourly
  cycle (never a full re-fetch), confirms freshness (same `freshness_retry`
  guarantee as before), then advances the regime by one bar. `.dmi_blocks()`
  computes the per-trade DMI filter from the local 4h buffer via
  `filters._wilder_dmi`/`filters.dmi_blocks_direction` **unchanged** — only the
  data source moved (Bitstamp buffer instead of SQLite), never the formula.
  `.current_regime_map()` returns a tiny duck-typed `.get(ts, default)` object
  that always answers the current regime regardless of `ts` — regime changes at
  most once/hour (`confirm_bars`/`cooldown_bars`), so a literal single-key dict
  would break the lookup the moment `advance_pending_entry` (called every
  minute) asks for a slightly different timestamp than the current cycle's.
- `regime_live.py` — incremental regime computation, **reconstructs** what the
  old `live_team/regime_live.py` did (that file was never committed and is
  gone — lost to a working-tree cleanup mid-session, not recoverable via git;
  this is a fresh implementation built from the same documented contract).
  Reuses `regime/state_machine.py::apply_state_machine` **directly, unmodified**
  (it already natively supports resuming from `initial_state=`) and duplicates
  `regime/scorers.py`'s formulas (trend/momentum/DMI raw values, rolling
  z-score) — same convention as `filters._wilder_dmi` elsewhere in this repo.
  `load_or_init_state` resumes from `logs/regime_state.json` if present, else
  bootstraps from `regime/regime_seed.json` (the rolling 730-day tail of raw
  axis values + the hysteresis state machine's terminal state — regenerate via
  `cd 1h/regime && python export_seed.py`, needed every 3-6 months since the
  seed's calibrated thresholds barely drift). `bootstrap_and_advance` replays
  any gap between the loaded state and now, one hour at a time, from
  `market_data`'s buffers — same function handles the very first launch (large
  gap) and a routine hourly tick (gap of exactly one bar) uniformly.
  **Verified 2026-07-10**: both the raw-value computation and the incremental
  normalization+state-machine advancement were cross-checked against
  `regime/scorers.py`'s vectorized computation on identical historical data —
  0 mismatches over 500 bars.
- `notify.py` — Telegram (operational bot + optional separate error bot via
  `TelegramErrorHandler`, attached to the root logger — any `logger.error`/
  `logger.exception` anywhere in `live/` reaches it automatically).
- `config_live.py` — live-only constants (OKX instrument id, tick cadence, log
  paths). Never redefines a threshold that already lives in `config_main.py`.
- `.env` (gitignored, real file — not `.env.example`) — `OKX_API_KEY`/
  `OKX_API_SECRET`/`OKX_API_PASSPHRASE`, `DRY_RUN` (`true`/`false`), Telegram
  tokens. `DRY_RUN=true` by default. **`OKX_LEVERAGE`/`OKX_POSITION_FRACTION`**
  (2026-07-15, explicit operator request — `.env` doubles as the config file
  for money-sensitive knobs, not just credentials): real overridable values in
  `config_live.py` (`_env_int`/`_env_float`, never raise on malformed input,
  log + fall back instead). **Deliberately breaks the symmetry principle**
  above when set — empty/absent (the default) still falls back to exactly
  `config_main.LEVERAGE`/`POSITION_FRACTION`, so live only diverges from what
  the backtest validated if the operator explicitly fills these in.
- `Dockerfile` + `docker-compose.yml` — build context is **`modele_team/`**
  (`..` from `live/`), NOT `1h/` (changed 2026-07-10, 2nd campaign — see *Full
  self-containment* above). A single `COPY live/ /app/modele_team/live/` now
  brings in `live/modele_up/`, `live/modele_down/`, `live/regime/` along with
  everything else — no separate `COPY modele_up/ ...` reaching outside the
  context. `modele_team/.dockerignore` (not `1h/.dockerignore`, now deleted —
  dead since the context moved) keeps `*.parquet`/`*.db`/`*.csv`/`logs/` out of
  the image. **No SQLite volume mount** (removed 2026-07-10 — nothing in
  `live/` reads it anymore, see *Data autonomy* above).
- `logs/` — `cycles.csv` (1 row/hour: predictions, thresholds, decision, which
  filter blocked it if any), `positions.csv` (1 row/position event),
  `positions_state.json` (persisted open-position + pending-entry state),
  `regime_state.json` (persisted regime rolling-window + hysteresis state,
  cf `regime_live.py`), `live.log`.

## Run

```powershell
cd backtest/modele_updown/1h/modele_team/live
# fill live/.env first (OKX keys, Telegram); DRY_RUN=true by default
python live.py
```

Docker:

```powershell
cd backtest/modele_updown/1h/modele_team/live
docker compose build
docker compose up -d
docker compose logs -f live-okx-test
```

⚠️ **Branche `algov2_test` : tous les noms Docker portent le suffixe `-test`**
(projet `live-okx-test` via le `name:` de haut niveau, service `live-okx-test`,
conteneur `live-okx-1h-test`). C'est ce qui permet de cohabiter avec le
déploiement de production sur le même VPS : sans le `name:`, Compose déduit le
nom de projet du dossier (`live`), **identique des deux côtés** — un
`docker compose down` lancé depuis le test aurait arrêté la prod. Aucun `-p`
n'est nécessaire. **À retirer si cette branche est fusionnée vers la prod.**
