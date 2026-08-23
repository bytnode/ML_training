# CLAUDE.md — modele_team

One of four packages in the `1h/` tree, sibling to `modele_up/`, `modele_down/`, and `regime/` (market-regime detector). Holds the **entire team system**: trading-backtest orchestrator, log writers, dashboard/fee-lookup helpers, and (as of 2026-07-09) the **live OKX daemon** in `live/` — see `live/CLAUDE.md`. Only consumes the two directional models via their standardized `api.py` (`evaluate_full_on_slice` → one row per bar, taken or not).

> Successor to the `15min/` tree's WR-only `main.py`: instead of counting win/loss it **simulates the real trade** (entry, intrabar exit mgmt, leverage, fees) and reports compounded equity. Read each sister package's own `CLAUDE.md` before editing inside it.

> ⚠️ **The repo-root `CLAUDE.md` is stale** (verified 2026-08-11) and should not be trusted for tree layout: the `30m/` → `1h/` rename it calls "pending" is **done**; the `5m/` tree it documents **does not exist** (there is a `30min/` tree instead); `15min/` now holds `polymarket/` + `trading_future/` subtrees rather than packages directly; and the repo-root `live/` daemon it describes as "flattened" actually lives at `live/modele_15m/`. This file + `live/CLAUDE.md` + `regime/CLAUDE.md` are the authority for the `1h/` tree.

## Contents

- `main.py` — orchestrator. Pipeline: predict both models → **pre-coherence filters** (UP hour, DOWN selectivity, UP cross-model) → coherence rule → **post-coherence filters** (DMI, UP confidence) → simulate incl. **concurrency cap** (optionally regime-conditional DOWN exits) → Rich report + CSVs. See *Entry filters* for the exact order — it is not "coherence first, filters after".
- `config_main.py` — **all** editable knobs; `main.py` holds none of its own. Inline comments record the sweep result + campaign name/date behind each default — re-tuned often, trust the code over any number written here. ⚠️ partagé avec `live/` par **import direct** (aucune copie n'existe — voir *Backtest/live* ci-dessous) : supprimer/renommer un nom ici peut casser le daemon à l'import.
- `filters.py` — the decision logic extracted out of `main.py` (2026-07-09, OKX live build): coherence rule, DMI filter, all 3 UP filters, regime wrapper, `sl_threshold`. **Never re-implement any of it in `main.py` or in `live/`** — il n'existe AUCUN `live/filters.py` : le daemon importe CE fichier via sys.path. Voir *Backtest/live*.
- `okx_fees.py` — best-effort real OKX maker/taker fee lookup (replaces the old `mexc_fees.py` now that the live exchange is OKX), one signed GET, env vars `OKX_API_KEY`/`OKX_API_SECRET`/`OKX_API_PASSPHRASE`. Returns `None` on any failure (never raises); `main()` falls back to `config_main.py`'s `MAKER_FEE_PER_SIDE`/`FEE_PER_SIDE` and **always prints which one it used**.
- `live/` — the live OKX daemon. Own `CLAUDE.md` — read it before touching anything inside. ⚠️ Son contexte de build Docker est **ce dossier-ci** (`context: ..` dans `live/docker-compose.yml`), pas `live/` : c'est donc bien `modele_team/.dockerignore` qui est actif (`live/.dockerignore` a ete supprime). Le Dockerfile fait `COPY config_main.py filters.py okx_fees.py` PUIS `COPY live/` — d'ou le besoin de remonter d'un cran.
- `g.py` — standalone Plotly dashboard over the last `main.py` run (`logs/trades.csv` → `logs/g_evolution.html`), reusing `trading_stats()`'s exact equity/drawdown math.
- `detailed_log.py` / `full_logs.py` — thin per-bar / wide per-bet CSV writers.
- `DB_1m/btc_1m.parquet` — 1-min intrabar source for SL/TP/trailing/maker-fill, gitignored. **Static artifact**: its builder `fetch_1m.py` was removed from the working tree (2026-07-09) — nothing here rebuilds it now; `main.py` degrades to a plain 1h-close exit if missing entirely.
- `logs/` — all team outputs (see *Logs*).

## The trading engine

Per team bet at decision bar `i`: **(1) Entry** at `open` of bar `i+1` (`traded_ts = ts + 1h`), UP → LONG / DOWN → SHORT. **(2) Body return** from modele_down's SQLite `candles` (`timeframe='1h'`): `body = (1+close_ret)/(1+open_ret) − 1` (scale-invariant; DB missing → run `data.py` first). **(3) Exit** (`_up_managed_exit`/`_down_managed_exit`) walks `DB_1m` minute-by-minute, exits at first of `take_profit` / `trailing` / `stop_loss` / `timestop` (`*_MAX_HOLD_MIN`, currently UP 240min / DOWN 720min, re-swept often — check `config_main.py`). The `trailing` floor is `max(peak − *_TRAIL_GAP  [armed at *_TRAIL_ACTIVATE],  ratchet_floor(peak))` — see *Cliquet de gain* below. SHORT profit `pp = 1 − close/entry`. **(4) P/L**: `net = gross − (entry_fee + exit_fee)`, both ×`EXPOSURE = POSITION_FRACTION · LEVERAGE`, then compounded (`np.cumprod`). Exit fee is **always taker** (a triggered exit needs guaranteed execution); entry fee follows MAKER/TAKER.

**Units (harmonized 2026-08-11):** every editable price threshold in `config_main.py` carries a `_PCT` suffix and is a **percent of the unlevered price move** (0.6 = 0.60%). The engine consumes derived decimal fractions (`UP_TAKE_PROFIT`, `*_SL_*_DEC`, `FEE_PER_SIDE`, … — the "derives, ne pas editer" lines, `/100`). `EXPOSURE` is applied **only once, at the very end**, uniformly to `net_ret`/`gross_ret` in `trading_stats()` for equity/reporting — it never enters the SL/TP/trailing/timestop decision itself. The SL is **always** the time-decay one: tightens linearly `*_SL_START_PCT`→`*_SL_END_PCT`, floored at `*_SL_FLOOR_PCT`. (The constant-SL knobs `*_STOP_LOSS*`, their `*_SL_TIME_DECAY` toggles, the legacy `STOP_LOSS_*` intra-candle block and `UP_SKIP_BUCKETS` were **deleted 2026-08-11** as dead config — in `config_main.py` (partagé) et `live/position.py`.)

**MAKER mode** (`MAKER=True`, entry leg only): posts a limit at `P0` (`i+1` open), fills only if a 1-min low/high touches it within `MAKER_FILL_WINDOW_MIN` — unfilled = missed opportunity (counted, shown as fill-rate). Filled trades pay `MAKER_FEE_PER_SIDE` instead of `FEE_PER_SIDE`; exit path/P&L otherwise identical to taker.

## Backtest/live : CONFIG, FILTRES **ET MOTEUR** alignés (2026-08-23)

⚠️ **Point d'entrée** : `live/` n'est PAS une copie divergente. Il **importe**
`config_main.py` et `filters.py` de ce dossier (sys.path, aucun fichier
dupliqué), et ses copies embarquées de `modele_up`/`modele_down`/`regime` sont
identiques aux originales (vérifié par hash le 2026-08-23 — seul un commentaire
diffère dans `regime/config_regime.py`). La **décision** est donc partagée par
construction. Depuis la campagne **ALIGNEMENT GRILLE 1-MIN (2026-08-23)**,
l'**exécution** l'est aussi :

- `live/position.py::_up_managed_exit` / `_down_managed_exit` sont un **port
  ligne à ligne** de leurs homonymes dans `main.py`, déroulant la **même grille
  1-min** (`t = traded_ts + j*60s`, clé = open-time). Le live lit les bougies
  1-min fermées d'OKX (`api_okx.get_candles_range`) là où le backtest lit
  `DB_1m/btc_1m.parquet` — mêmes objets, même index `j`. La marche est **rejouée
  depuis `j=0` à chaque tick**, donc un tick manqué ou un redémarrage sont
  rattrapés.
- L'**extension conditionnelle du TP DOWN** (`DOWN_TPEXT_*`, pivots 1-min
  compris) est désormais implémentée des deux côtés — elle était le dernier
  mécanisme absent du live. `DOWN_TPEXT_ENABLED=False` aujourd'hui : basculer le
  flag ne désynchronise plus rien.
- L'**ancre d'entrée maker** est le vrai P0 (`open` de la bougie 1-min
  d'open-time `traded_ts`) des deux côtés, et le **widen** est décidé par la
  marche au premier candidat en profit, comme `simulate_trades`.
- Parité prouvée : **720/720** séries longues et **3600/3600** séries courtes sur
  `(reason, pp, step_ts)`, **2400/2400** sur le dispatcher, rejeu tick-par-tick
  sans écart. Détail complet dans `live/CLAUDE.md`.
- **Écart restant assumé** : le funding perpétuel n'est pas modélisé au
  backtest ; l'exécution d'une sortie part au marché quelques secondes après la
  clôture 1-min qui l'a déclenchée.

⚠️ **Conséquence du partage** : supprimer ou renommer un nom dans
`config_main.py`/`filters.py` peut casser le daemon **à l'import** (arrivé le
2026-08-11 : 4 noms supprimés → `ImportError` au démarrage, réparé le
2026-08-14). Et `live/modele_*`/`live/regime` sont des **copies** : à re-copier
après un retrain.

### Historique : alignement de la décision (2026-08-14)

⚠️ L'ancienne version de cette section (« copies indépendantes depuis 2026-08-03 », avec une table de divergences de valeurs) était **fausse** : il n'existe AUCUN `live/config_main.py` ni `live/filters.py`. `live/{live,position,market_data,config_live}.py` importent `config_main.py` et `filters.py` **de ce dossier** via sys.path, et le Dockerfile les COPY dans l'image. Les *valeurs* de config sont donc toujours auto-synchronisées — mais le **code moteur** de `live/position.py` est indépendant, et c'est lui qui avait dérivé. Conséquence directe du partage : supprimer/renommer un nom dans `config_main.py`/`filters.py` peut casser le daemon à l'import (arrivé le 2026-08-11 : 4 noms supprimés → `ImportError` au démarrage, réparé le 2026-08-14). `live/` garde par contre ses **copies embarquées** de `modele_up/`, `modele_down/`, `regime/` (à re-copier après un retrain).

**Campagne ALIGNEMENT LIVE (2026-08-14)** — audit 31 agents (25 findings confirmés adversarialement) puis port complet ; `live/CLAUDE.md` détaille. En bref : crash d'import réparé ; contexte SMA365/ER porté au live (`filters.trend_context_from_series`, math extraite de `load_trend_context` — refactor prouvé bit-identique par hash — calculée depuis le buffer Bitstamp 1h agrandi 800 → **10 300 barres**) ; bumps d'entrée chop branchés (wrappers scalaires `effective_*_1` acceptent `trend_ctx`) ; `evaluate_exit` reconstruit en parité complète (SL time-decay inconditionnel, cliquet `ratchet_floor`, overrides chop/mbull figés à la barre de décision via `_decision_ctx`, palier DOWN chop/mbull, widen sur TP effectif, gate `DOWN_WIDEN_ENABLED`) ; cross-model UP déplacé PRÉ-cohérence (une barre double-pari donne DOWN, plus CANCEL) ; sémantique maker alignée backtest (limite **statique** à P0 pendant 15 min, plus de repricing 60 s, fenêtre expirée = **opportunité ratée** — plus de bascule taker, ce qui supprime aussi le bug de sur-exposition ×1.8 sur fill partiel + lecture échouée ; fill partiel conservé tel quel, fail-safe de relecture) ; écriture d'état atomique. Seuls écarts restants, assumés et documentés : tpext DOWN (off backtest, non implémenté live), granularité mark-price 60 s vs closes 1-min, funding non modélisé.

## Entry filters (order matters — some run *before* the coherence rule)

**Pre-coherence** — these mutate each model's own `take` flag, so a bet they drop can no longer participate in a `CANCEL`:

0a. **UP hour-of-day** (`UP_DROP_HOURS`, applied in `_prepare_predictions`) — fixed UTC hours UP never bets (currently `23,0,1,2,3,4,10,14,20`), found via losing-exit-hour diagnosis.
0b. **DOWN selectivity** (`DOWN_CONF_DELTA`, currently **0.03**) — a short is taken only if `P(down) ≥ sidecar_threshold + delta`. This is a genuine entry filter, the DOWN-side counterpart of `UP_CONF_MIN`; regime-conditional when `regime_map` is passed.
0c. **UP cross-model** (`UP_XMODEL_FILTER_ENABLED`) — keeps a LONG only if modele_down's `P(down) < UP_XMODEL_MAX_PDOWN` (currently **0.44**) at the same bar.

**Post-coherence** — applied to the assembled `team_bets`:

1. **DMI** (`DMI_FILTER_ENABLED`) — cancels a bet whose side contradicts the Wilder DMI of the **decision** bar (no lookahead): DOWN dropped if `+DI > −DI`, UP if `−DI > +DI`. Sign only, no ADX threshold; ties/warmup keep the bet. ⚠️ `DMI_TIMEFRAME` is `"4h"` despite helper names/docstrings saying "1h" — trust the config. `DMI_LENGTH=12`, `DMI_SIDES=("down","up")` (both filtered).
2. **UP confidence** (`UP_CONF_FILTER_ENABLED`) — keeps UP only if `P(up) ≥ UP_CONF_MIN` (currently **0.59**). DOWN untouched — already robust via `DOWN_CONF_DELTA`; only UP needed hardening.

**At simulation time:**

3. **Concurrency cap** (enforced inside `simulate_trades`, not a pre-filter, tracked chronologically per side and in total — no lookahead): `MAX_CONCURRENT_POSITIONS` (default **2**, both sides combined) **and** `MAX_CONCURRENT_POSITIONS_PER_SIDE` (default **1**, added 2026-07-10) — a signal is skipped if either cap is already full at its `traded_ts`. The per-side cap mirrors OKX HEDGE mode's real constraint (`live/position.py`): a 2nd same-side entry there merges into the already-open position's `avgPx` instead of opening an independent one, so live can never track 2x-same-side the way an earlier backtest version did. With per-side=1 the total cap of 2 is mathematically redundant (1 UP + 1 DOWN ≤ 2) but still checked for defensiveness.

All UP-only filter numbers are re-swept often — `config_main.py`'s inline comments carry the campaign name/date provenance for each.

## ⚠️ État courant : contexte SMA **365j** actif sur les DEUX sens — chiffres re-mesurés 2026-08-23

**Configuration** : mode chop actif des deux côtés (`CHOP_DOWN_ENABLED=True`),
SMA du contexte à **365 jours** (`CHOP_SMA_DAYS=365`, était 200). Base DOWN =
prod (widen ×4 + régime bear, sans gouverneur/tpext/verrou/cliquets), base UP =
stack 2026-08-11.

**Run de référence** (`python main.py`, `TEAM_TEST_DAYS=700`, exécuté
2026-08-23) — ⚠️ **ces chiffres remplacent ceux écrits ici le 2026-08-14, qui
étaient faux** (le fichier annonçait +2202.99 % / 400 trades / année 1
+115.72 % / année 2 +967.60 % sur une fenêtre de 720 j ; aucune de ces valeurs
n'est reproductible avec la config en place) :

| | Trades | Net composé | WR net | EV/trade | Max DD |
|---|---|---|---|---|---|
| **Année 1** (2024-08-02 → 2025-08-02, 365 j) | 191 | **+114.90 %** | 56.54 % | +0.4999 % | −31.87 % |
| ‎ ↳ UP | 89 | +37.33 % | 65.17 % | +0.4125 % | −18.85 % |
| ‎ ↳ DOWN | 102 | +56.48 % | 49.02 % | +0.5762 % | −28.96 % |
| **Année 2** (2025-08-02 → 2026-07-03, **335 j**) | 200 | **+892.24 %** | 58.50 % | +1.2764 % | −17.65 % |
| ‎ ↳ UP | 109 | +188.71 % | 61.47 % | +1.0461 % | −11.31 % |
| ‎ ↳ DOWN | 91 | +243.68 % | 54.95 % | +1.5523 % | −16.92 % |
| **TOTAL** | **391** | **+2032.28 %** | 57.54 % | **+0.8971 %** | **−31.87 %** |

Fill-rate maker 90.38 % (48 signaux ratés sur 499), 60 signaux écartés par le
plafond de concurrence, 10 UP / 7 DOWN élargies.

⚠️ **Deux pièges de lecture de ce tableau :**
1. **L'« année 2 » ne fait que 335 jours**, pas 365 — la fenêtre s'arrête au
   **2026-07-03**. Comparer +114.90 % et +892.24 % comme deux années revient à
   comparer des durées différentes.
2. **La fenêtre s'arrête au 2026-07-03 parce que la SQLite/parquet s'y arrête.**
   Les données ne sont pas rafraîchies depuis ~7 semaines : le backtest ne dit
   RIEN de la période récente. `live/` n'est pas affecté (il lit Bitstamp en
   direct), mais toute décision prise sur ce tableau ignore les 7 dernières
   semaines. Rebuild = `cd ../modele_down && python data.py`.

⚠️ **Les chiffres de sweep cités dans les sections historiques ci-dessous
(campagnes du 2026-08-11 et du 2026-08-14 : +2946.67 %, +2271.76 %, +2179.62 %…)
ont été mesurés à `TEAM_TEST_DAYS=720`**, alors que la config en place est à
**700**. Ils ne sont donc pas reproductibles tels quels et ne servent que de
trace de campagne — le tableau ci-dessus est la seule mesure de l'état courant.

`live/` est **entièrement aligné** sur cette configuration depuis le 2026-08-23
(contexte SMA/ER + stack UP + moteur de sortie), cf *Backtest/live* en tête de
fichier. Reste à redéployer l'image Docker sur le VPS.

## DOWN aligné sur la prod déployée (décision user, 2026-08-14) — étape précédente

**Le côté DOWN reproduit la prod déployée (`Desktop/modele_team`, état ~mi-juillet 2026) ; le côté UP garde tout le stack des campagnes 2026-08-11.** Historique de la journée : le user a d'abord demandé un retour du DOWN au commit `ecb04f8` (2026-07-04) — implémenté puis **remplacé** après la découverte que la prod n'est PAS ce commit : elle y ajoute les caps de concurrence, le widen DOWN et le wrapper régime (état mi-juillet). Diagnostic chiffré dans `optimize/exp_dncommit.py` : le +372.01% DOWN du commit se décompose en +167.94% (115 trades séquentiels) × +76.16% (25 trades **empilés**, jusqu'à 4 shorts simultanés — infinançable en marge et impossible sur OKX HEDGE) ; reproduction au centième vérifiée. État DOWN actuel = base commit (TP 1.60 / gap 1.20 / act 0.15 / SL 1.0→0.4 plancher 0.15 / hold 720 / delta 0.03 plat) **+** caps 2/1 **+** widen DOWN ×4/×2/×4 (`DOWN_WIDEN_ENABLED=True`) **+** régime bear hold ×2 / gap ×1.3 (`REGIME_WRAPPER_ENABLED=True`). **Coupés côté DOWN** (le stack 2026-08-11, absent de la prod) : tpext, verrou de sécurité 1.10→0.25 (perte de garantie confirmée par le user), tout le chop DOWN (`CHOP_DOWN_ENABLED=False`), cliquets DOWN. **Gouverneur d'equity : SUPPRIMÉ du code** (même jour, décision user — parité prod, elle n'en a jamais eu) : plus de `apply_governor` dans `filters.py`, plus de bloc `GOVERNOR_*` dans la config, plus de ligne au rapport. Diagnostic qui a motivé la suppression (`optimize/exp_upgov.py`, gardé comme trace mais ne tourne plus) : l'UP seul ne déclenchait JAMAIS le disjoncteur — 100% des suspensions venaient du saignement DOWN année 1, et depuis l'alignement elles ne sautaient plus que des trades UP (+32.98% → +11.01% en a1) ; l'UP payait les fautes du DOWN. Sans gouverneur, 720j : UP a1 **+32.98%** (83 trades, WR 63.9%), UP a2 +176.89%, DOWN inchangé (a1 −27.19%, a2 ~+521%). Les chiffres "+2946.67% / DD −24.23" cités plus bas décrivent le stack complet d'avant cet alignement, plus la production courante — la dépendance au régime (année 1 négative) est le comportement prod assumé.

## Cliquet de gain (`RATCHET_*`, added 2026-08-11)

Fixes a structural hole: the classic trailing floor is `peak − gap`, so it locks **nothing positive** until the peak exceeds the gap. `filters.ratchet_floor(peak, steps, prop)` adds a second, monotone floor — flat paliers `(peak_pct → locked_pct)` and/or a **proportional** `(arm_pct, frac)` term that guarantees keeping `frac × peak` once the peak reaches `arm_pct`. The engine takes `max(trailing, ratchet)`; the SL and its horizon are never touched. `RATCHET_ENABLED=False` (or empty steps + `None` prop) is bit-identical to pre-campaign behaviour — verified.

What the ~25 000-config campaign actually found (`optimize/sweep_ratchet*.py`, `logs/sweep_*.csv`):

- **The cliquet alone is monotonically destructive** at constant TP/gap — the more it binds, the worse the net (DOWN +502% → +4% at arm 0.2%; UP +125% → +27% at arm 0.10%). Clipped trades cost far more than saved ones. Don't re-propose early profit-locking as a standalone fix.
- **DOWN: leave it alone.** 3150 configs (TP 1.0–3.5%, gap 0.6–9.0%, flat + proportional cliquets) — *zero* beat the baseline even on total net; its optimum (TP 1.60 / gap 1.20 / no cliquet) is interior to the grid. `DOWN_RATCHET_*` are deliberately empty.
- **Adaptive DOWN *tightening* fails, but adaptive DOWN *TP-extension* won** (2nd campaign, same day — full journal in `optimize/RAPPORT_adaptativite_down_2026-08-11.md`): ~270 tightening configs across 5 families (reversal-signal, stall, ATR-normalized gaps, regime-conditional) all lose — the current exits are *not* premature (post-exit continuation proved it), and the ~1.2% retracement scale of BTC 1h trend-rides is constant, not vol-proportional. The winner is on the other side: see *Extension conditionnelle du TP DOWN* below.

## Extension conditionnelle du TP DOWN (`DOWN_TPEXT_*`, added 2026-08-11)

Post-TP diagnostic: the 39 DOWN take-profit exits at +1.60% had an average post-TP peak of +2.98% (+3.27% in bear). Mechanism (in `_down_managed_exit`): when a short crosses the TP with (1) **bear regime** at the decision bar and (2) an inefficient path since entry (**ER = pp/pathlen ≤ 0.075** — grinders continue, fast spikes whipsaw; the gate direction was found by trade-level autopsy), the trade does **not** exit: it rides with a **structure exit** (close above the last confirmed 1-min swing-high, `p=20`, formed *during* the extension — pivots pre-computed by `_load_swing_pivots`, confirmed p bars later, no lookahead) plus a hard floor at TP−0.80%. Classic trailing keeps running in parallel (max of floors); widen-on-profit overrides the extension. Result: DOWN +502.01% → **+694.89%** (EV 1.786 → 2.073, WR/worst/DD unchanged), total → **+2179.62%**. Plateaus verified (p 12–25, hard-guard 0.6–1.2, ER 0.065–0.090). ⚠️ **Accepted limit (explicit user decision)**: the gain concentrates in the bear half (h1 +277→+423); the bull half gives back ~8 net pts (h2 +59.7→+51.9) through 3 shallow-bear extensions no principled gate separates — this mechanism is regime-dependent by construction. Only 14 extensions in the window; re-sweep as the sample grows. Chandelier/SAR/ATR were also tested as extension exits — structure-break beats them all. **Porté à `live/` le 2026-08-23** (pivots 1-min via `position.swing_pivots_from`, port de `_load_swing_pivots`) : le flag reste `False`, mais le basculer n'introduit plus d'écart backtest/live.
- **UP: the cliquet pays only once the TP is raised.** At TP 0.60% the take-profit fired before any lock could exist. At TP 1.55% + `prop(arm 1.00%, frac 0.30)`: UP +124.89% → **+181.99%**, EV 0.6546 → 0.8779. Isolating the cliquet at identical TP/gap: +160.75% → +181.99% (total +1469.76% → **+1626.48%**). It doesn't save *more* trades (25 vs 26) — it **cuts the cost of the clipped ones** (−74.93 → −66.52 pts).

⚠️ Cette mise en garde (« `live/` carries independent copies… live still runs the old UP TP (0.60) and has no `ratchet_floor` ») était **fausse** : `live/` importe `config_main.py`/`filters.py` de ce dossier, donc il a toujours eu le TP UP courant et `filters.ratchet_floor` y est câblé depuis le 2026-08-14.

## Mode défensif chop (`CHOP_*` / `MBULL_*`, campagne TOUT-TERRAIN 2026-08-11)

Corrige la dépendance au régime révélée par le test 720j (année 1 2024-07→2025-07 : **−46.87%** net alors que les modèles prédisent bien — année 1 positive EN BRUT (+57.7 pts), tuée par 80 sorties SL (−404 pts) + 89.6 pts de frais = trend-following en marché mean-reverting). Deux indicateurs de structure longue, construits par `filters.load_trend_context()` depuis les closes 1h de la table `candles` (cache module, aucun lookahead) : **ER de Kaufman 30j** (pas quotidiens, rafraîchi chaque barre) et **âge du bear** (heures consécutives sous la SMA200). Mode **scalp** armé quand `ER < 0.25` ET `âge ≤ 30j` : les *sorties* passent en mean-reversion (TP UP 0.60 / TP DOWN 0.90 / hold DOWN 600 / gap DOWN 0.60, tpext + override régime bear coupés) — **aucune entrée supprimée, SL intact**. Hors chop : comportement inchangé, plus **TP UP 0.60 en macro-bull** (`MBULL_UP_TP_*` — améliore les DEUX années). Implémentation : `simulate_trades(trend_ctx=...)` + paramètre `take_profit` des deux sorties gérées (le widen et le hard-guard tpext suivent le TP *effectif* via `base_tp`). `trend_ctx=None` = bit-identique à avant.

Résultat 720j (après les campagnes POINTS-FAIBLES, DOWN-ANNEE-1, WR-DOWN et UP-PERTES du même jour, ci-dessous) : année 1 −46.87% → **+112.57%**, année 2 +1652.97% → **+1333.25%** (arbitrage assumé), total +831.35% → **+2946.67%** (EV **+1.0046%**/trade), max DD −69.94% → **−24.23%** ; fenêtres glissantes : **365j toutes positives (min +66.63%)**, 180j min −13.14%, 90j min −14.96% ; seul trimestre négatif restant : 2025Q2 (−19.52%). Plateaux vérifiés (ER 0.20-0.30, âge 15-60j) ; frontière d'âge dans un fossé structurel (année 1 : bear max 27j ; année 2 : 60% des barres >30j, jusqu'à 242j). Rejetés après test : gates direction macro (SMA/momentum), skip DOWN macro-bull, skip DOWN bull profond (+10% SMA200), SL DOWN resserré en chop — tous retirent plus de gagnants que de perdants. Le `_harness.py` d'optimize ne passe PAS `trend_ctx` (sweeps = comportement pré-campagne). **Porté à `live/` le 2026-08-14** (contexte SMA/ER depuis le buffer Bitstamp) — cf *Backtest/live*. Journal complet : `optimize/RAPPORT_tout_terrain_2026-08-11.md`, scripts `optimize/diag_year1.py` + `optimize/exp_*.py`, résultats `logs/exp_*.csv`.

**Campagne POINTS-FAIBLES (même jour, 2e goal)** — trois corrections sur le diagnostic du run précédent (DD −57.9% = un seul segment nov-2024→juin-2025 ; les 14 pires trades = tous des SL UP en chop ; 81 sorties trailing DOWN whipsawées = −134 pts) : **(1)** SL UP resserré en chop (`CHOP_UP_SL_START/FLOOR_PCT` 1.2/0.5 — plus serré dégrade), **(2)** trailing DOWN en chop armé seulement à +0.40% de profit prouvé avec gap 0.70 (`CHOP_DOWN_TRAIL_ACTIVATE/GAP_PCT`, optima intérieurs), **(3)** **gouverneur d'equity à livre-fantôme** (`GOVERNOR_*`, `filters.apply_governor` — ⚠️ SUPPRIMÉ du code le 2026-08-14, cf section ÉTAT ACTUEL ; ce qui suit est historique) : le système simule tout (fantôme = les candidats de `simulate_trades`), l'exécution réelle est suspendue quand la somme des 40 derniers résultats fantômes **déjà clos** (`exit_ts` ≤ entrée courante, aucun lookahead) passe sous −15 pts d'equity, reprise au-dessus de −15. 35 trades sautés, année 2 intacte, DD −42.02 → −27.21 (coût ~144 pts de total, arbitrage stabilité ≻ net). ⚠️ La v1 du gouverneur (`exp_governor.py`) comptait des trades encore ouverts — surestimée, ne pas la réutiliser ; `exp_governor2.py` (exit-aware) fait foi. Les logs détaillés (`detailed_log`/`full_logs`) restent la vue *fantôme* ; `trades.csv` + toutes les stats = l'exécution réelle. `trades.csv` porte désormais une colonne `exit_ts`.

**Campagne UP-PERTES (même jour, 5e goal, `optimize/exp_upwr*.py`)** — constat user : côté UP année 1, une perte vaut ~1.6× un gain. Anatomie : 21 stop-loss UP en chop = −98.5 pts (longs *marginaux*, P(up) juste au seuil). **Testés et rejetés** : palier de cliquet UP (impossible structurellement — le TP chop 0.60 sort avant tout armement, armer plus bas ampute les gains ; même leçon que la campagne cliquet), TP chop relevé (en chop les mouvements ne s'étendent pas), SL serré étendu au mbull (3 trades, net négatif). **Retenu** : `CHOP_UP_CONF_MIN_BUMP` (0.035) — en chop, un long exige P(up) ≥ 0.625 (miroir exact du bump DOWN, dans `filters.effective_up_conf_min`/`apply_conf_filter` quand `trend_ctx` est passé). Bande viable 0.03-0.04, **falaise à 0.05** ; 0.035 = centre. Résultat : AvgLoss UP a1 3.90 → 3.35 (ratio 1.6 → 1.3), UP a1 +27.22 → **+43.74%**, année 1 → **+112.57%**, total → **+2946.67%**.

**Campagne WR-DOWN (même jour, 4e goal, `optimize/exp_dnwr.py`)** — demande user : remonter le WR DOWN de l'année 1 (42.7%) sans sacrifier les gros gains. Mécanisme : **palier de cliquet DOWN supplémentaire (0.50 → 0.10) actif hors bear installé** (chop scalpable OU macro-bull — jamais en macro-bear âgé, où les trend-rides gardent tout leur espace) : un short monté à +0.50% ne peut plus finir sous +0.10% (verrou net-positif minimal ; un lock à 0.05 ferait plus de net total mais PAS de WR — clôture nette-négative). Implémenté via `filters.dn_ratchet_extra` + param `ratchet_extra` de `_down_managed_exit` (fusionné aux paliers de sécurité). Zone viable : arm 0.45-0.55 × lock 0.05-0.10 ; **lock 0.15 = falaise**, arm 0.40 dégrade. Résultat : WR DOWN a1 42.70 → **48.89%** (AvgWin 5.37 → 4.37, les rides restent), DOWN a1 +29.98 → **+52.95%**, a1 +60.02 → **+94.59%**, dd → **−23.70%** ; coût assumé : a2 +1622 → +1385 (phases chop/mbull clippées — l'équilibre inter-années s'améliore, c'est la mission).

**Campagne DOWN-ANNEE-1 (même jour, 3e goal, `optimize/exp_dnfix.py`)** — le poste de perte résiduel était 34 stop-loss DOWN en chop (−153 pts) : des shorts *marginaux* (P(down) juste au seuil) pris en marché indécis. Correctif d'**entrée** (le premier de la campagne — tout le reste était sortie) : en chop, le seuil DOWN devient `sidecar + DOWN_CONF_DELTA + CHOP_DOWN_CONF_DELTA_BUMP` (0.01), appliqué dans `filters.effective_down_delta` quand `trend_ctx` est passé (le harnais ne le passe pas → sweeps inchangés) ; plus TP scalp 0.90 → **1.10** (re-swept sous le nouveau trailing). Bande viable du bump : 0.0075-0.015, **falaises des deux côtés** (0.005 et 0.02 s'effondrent) — 0.01 = centre de bande, la paire (0.01, 0.0125)×TP 1.10 est un plateau. Re-testé et rejeté au passage : tout resserrage du SL DOWN chop, même sous le nouveau trailing (6 variantes, toutes pires — le SL DOWN est définitif). Résultat : année 1 DOWN +17.77 → **+29.98%** (EV +0.278 → +0.438), année 2 DOWN → **+500.20%**, et le système entier aux chiffres du paragraphe d'intro.

## Security locks (`*_RATCHET_STEPS_PCT`, campagne SECURITE 2026-08-11)

Flat ratchet steps re-activated as a **hard security guarantee** (explicit user decision: security ≻ net P/L): DOWN `(1.10 → 0.25)` — a short that peaked ≥1.10% can no longer finish negative (floor ≥ +0.25%, net-positive after the ~0.10-0.20 close-overshoot and 0.07 fees); UP `(0.80 → 0.05)` (≈free, +2 pts). **Accepted cost: total net 2179.62% → 1717.44%** — 4 trades that pierced +0.25% after peaking ≥1.1 then recovered to +1.06..+4.33 get clipped; no measurable signal separates them from the 8 real peaked-then-lost trades (tested: vol_ratio, adverse volume, DMI spread, bear age, retrace speed — full overlap on 13 cases, `logs/sweep_security*.csv`). Side effect: DOWN WR 51.7→57.8%. Frontier if re-arbitrated: arm 1.15 → 1769% (doesn't cover the user's 1.13% case); arm 1.05 → 1640%. Residual risk: a ~1% single 1-min candle can cross the whole band (1 case/131 UP trades, −0.05% net — fee-level). Zero DOWN violations ≥1.1 on the window; the 1.0–1.1 band stays unguaranteed (1 case, −1.56%).

## Widen-on-profit (`WIDEN_ON_PROFIT_ENABLED`, added 2026-07-10)

When a same-side signal is blocked by the per-side concurrency cap (above) **and** the already-open position of that side is currently in profit (`pp > 0`), it's used as trend confirmation instead of being discarded: the open position's take-profit / trailing-gap / max-hold widen **once** (never cumulative, SL never touched — stays on its original horizon). Implemented inside `_up_managed_exit`/`_down_managed_exit` via a `widen_candidates` set of later same-direction `traded_ts` — the walk switches to widened params the first time it reaches one of those timestamps while `pp > 0`. UP and DOWN multipliers are swept **separately** (mathematically decomposable — widening one side never changes the other side's trades). ⚠️ On the current window only **7 UP / 3 DOWN** positions actually get widened — far below the `n≥150` bar this repo otherwise requires before trusting a swept value (cf. `UP_CONF_MIN`). Treat `UP_WIDEN_*`/`DOWN_WIDEN_*` as provisional; re-sweep once more positions have widened.

## Regime-conditional DOWN exits (`regime/` sibling package)

`1h/regime/` is a bear/neutral/bull detector wired into `main.py` as an optional wrapper, gated on `regime.config_regime.REGIME_WRAPPER_ENABLED` (**currently `True`, active**). When on, `_effective_down_exit_params` overrides DOWN's `max_hold`/`trail_gap` from `REGIME_EXIT_OVERRIDES`, chosen once at entry from the regime at the decision bar (never revised mid-trade) — **DOWN exits only**. A separate entry-filter override table (`REGIME_OVERRIDES`) exists but was proven a no-op by extensive sweeping — inactive. `regime_map=None` reproduces exact pre-regime behavior everywhere. **Read `regime/CLAUDE.md` first** — full validation history, and owns `compare_regime_variants.py` (the entry-filter sweep harness).

## Window, imports, logs

- **Window**: `compute_window` anchors `end` to the latest timestamp common to both sister packages' `logs/test_predictions.csv`, then goes back `TEAM_TEST_DAYS`.
- **Imports**: `main.py` lives one level below `modele_up`/`modele_down`/`regime`; it injects the grandparent dir (`1h/`) onto `sys.path` so they import as top-level (`1h/` starts with a digit, can't be a dotted package).
- **Logs** (all in `logs/`; detail files overwrite each run, `trading_runs.csv` is append-only/schema-tolerant): `trades.csv` (per trade), `trading_runs.csv` (per-run summary incl. **EV/trade** = `WR·AvgWin − LR·AvgLoss`, net-of-fees ×leverage — same as avg net%/trade, decomposed), `team_all_bars/skipped/taken.csv` (thin per-bar, `skipped+taken=all`), `full_logs/{all,won,lost}_bets.csv` (wide per-bet), `g_evolution.html` (written by `g.py`, not `main.py`), `regime_backtest_compare.csv` (written by `compare_regime_variants.py`, see `regime/CLAUDE.md`), `sweep_widen_up.csv` / `sweep_widen_down.csv` (widen-on-profit sweep output; the sweep script itself is no longer in the tree).

## Run

```powershell
cd backtest/modele_updown/1h/modele_team
python main.py    # overwrites logs/ detail + appends trading_runs.csv + Rich report
python g.py        # optional: Plotly dashboard over the run just written
```

## Team coherence rule

```
up bets UP   + down abstains  -> UP
down bets DN + up abstains    -> DOWN
both bet                       -> CANCEL
neither bet                    -> PASS
```

⚠️ The rule does **not** see raw model output: `UP_DROP_HOURS`, `DOWN_CONF_DELTA` and the UP cross-model filter have already zeroed some `take` flags by this point, so they change which bars reach `CANCEL`/`PASS`. Surviving bets then pass through the DMI filter, the UP confidence filter, and the concurrency cap (all above) before simulation — any of which can still drop a bet.
