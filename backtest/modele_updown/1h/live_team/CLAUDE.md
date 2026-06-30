# CLAUDE.md — `live_team` (daemon live MEXC futures, team 1h)

**4ᵉ paquet** du tree `1h/`, frère de `modele_up/`, `modele_down/`, `modele_team/`.
Daemon **autonome** qui exécute en argent réel sur **MEXC futures** (`BTC_USDT` perp)
la stratégie backtestée par `modele_team/main.py` : mêmes 2 modèles, mêmes
paramètres (`config_main.py` → recopiés dans `config_team.py`), même gestion de
sortie (SL temporel / TP / trailing / timestop).

Design de référence : `docs/superpowers/specs/2026-06-18-mexc-live-team-1h-design.md`.

## Principe

- **Prédiction** sur **Bitstamp** (BTC/USD + ETH/USD), identique à l'entraînement → zéro skew.
- **Exécution & gestion** sur **MEXC** (prix réel de la position : entrée, monitoring 1-min, SL/TP/trailing, PnL).
- **Boucle tick 1-minute** : chaque minute → gère les positions ouvertes (sur la clôture Min1 MEXC) + les entrées en attente ; **au top de l'heure** en plus → prédit, applique la team rule, ouvre une entrée si un slot est libre.
- **Entrée maker** : post-only à 0.01% passif, **3 tentatives de 5 min** re-pricées ; non rempli après 15 min → opportunité annulée.
- **Sortie** : en **profit** → 3× post-only reduce-only (20 s) puis market taker ; **SL / perte** → market reduce-only taker immédiat.
- **Concurrence** : jusqu'à **2 positions** (mode **HEDGE**, marge **ISOLATED**, levier **10**, 50% du solde par position, composé).

## Fichiers

| Fichier | Rôle |
|---|---|
| `live.py` | boucle 1-min : bootstrap, monitoring, EntryManager, décision horaire, logs |
| `mexc.py` | client REST maison (SEUL fichier qui touche MEXC) — signature HMAC, `/order/create`, klines, positions, balance |
| `position.py` | `PositionManager` : `check_exit` (SL>TP>trailing>timestop) + `execute_exit` (maker→taker) + persistance + reconcile |
| `config_team.py` | constantes recopiées de `config_main.py` + params live MEXC |
| `live_helpers/fetch.py` | OHLC Bitstamp (BTC 5 TF + ETH 15m), retry/pagination |
| `live_helpers/features_up.py` / `features_down.py` | les 6 features de chaque modèle (réutilisent les helpers d'entraînement par import → **validés byte-à-byte vs parquet, tol 1e-6**) |
| `live_helpers/predict.py` | charge les 2 modèles, `_bucket_threshold` (port exact), filtre cross-model, team rule |
| `live_helpers/notify.py` | Telegram 2-bots (opérationnel + bot d'erreur auto-câblé) |
| `modele_up/` · `modele_down/` | copies runtime : `btc_updown_model.json` + `.thresholds.json` + `config.py` + helpers (`data.py`, `add_*.py`) |
| `logs/` | `cycles.csv` (1 ligne/heure), `positions.csv` (1 ligne/position), `positions_state.json` (état persistant) |

## Lancement

```powershell
cd backtest/modele_updown/1h/live_team
copy .env.example .env          # puis remplis MEXC_ACCESS_KEY / SECRET_KEY (+ Telegram)
# 1) test connexion/signature SANS poster d'ordre :
$env:DRY_RUN="true"; python live.py
# 2) validation réelle micro-volume (1 contrat = 0.0001 BTC) :
#    .env -> DRY_RUN=false, MIN_VOL=1  ; puis : python live.py
# 3) une fois validé : .env -> BET_FRACTION=0.5 (taille normale)
```

Docker : `docker compose build && docker compose up -d` (service `live-team-mexc`, logs montés sur `./logs`).

## Prérequis (côté toi)

1. **KYC** MEXC + permission **« Futures / Order Placing »** activée sur la clé API.
2. Compte MEXC en mode **Hedge** + marge **Isolated** (le daemon règle le levier au boot).
3. Runtime sur **VPS Suisse** (non géo-bloqué — confirmé).
4. `xgboost==3.2.*` (épinglé dans le Dockerfile ; obligatoire pour la parité `base_score`).

## ⚠️ AVANT LE RÉEL — il ne reste qu'UNE chose (elle exige TA connexion MEXC)

État : **fidélité ML/backtest validée** (2 validations parquet PASSENT ; `_bucket_threshold`, `sl_threshold`,
ordre des sorties, filtre cross-model, anchor_ts conformes) + **fixes 2→6 appliqués et vérifiés** (compile + import OK).
Reste **uniquement** ce qui ne peut se vérifier que sur l'API MEXC vivante :

1. **[À FAIRE par toi] Base URL Futures + format API.** `REST_HOST` = `https://contract.mexc.com`
   (futures historique ; `api.mexc.com` = SPOT). La recherche évoquait une migration vers `api.mexc.com`.
   **VÉRIFIE quel host répond** avant tout ordre :
   `python -c "from mexc import MexcClient; print(MexcClient(access_key='',secret_key='').get_contract_detail('BTC_USDT'))"`
   (teste les deux via `MEXC_REST_HOST` dans `.env`). Confirme aussi, contre la doc/API live : les champs
   (`contractSize`/`priceUnit`/`holdVol`/`dealAvgPrice`/`state`) et le mapping `side` hedge (1/3/2/4) + `positionType` (1/2).

### ✅ FAIT (2026-06-18) — corrigés et vérifiés (compile + import OK)
2. **Rejeu des minutes manquées + alignement `j`** — `position.process_min1_bars` rejoue tous les closes Min1
   fermés depuis le dernier passage (rattrape redémarrage/stall), chacun évalué avec `now_ms = OPEN de la bougie`
   → `j` exactement aligné sur `close_ib[traded_ts + j·60000]` du backtest. `live.py` exclut la bougie en formation.
3. **Signature GET/DELETE exacte** — `mexc._send` envoie la query exactement signée (`url+"?"+param_str`, `params=None`).
4. **Fill partiel à l'entrée** — `EntryManager` adopte la portion remplie comme position au lieu de re-poster le vol plein.
5. **Frais d'entrée maker** — champ `Position.entry_is_maker` ; `execute_exit` facture le côté entrée en maker.
6. **`ATR_PERIOD`** — lu depuis le module `data` déjà chargé en mémoire (fallback 14), sans manip de `sys.path`.

## Règle d'équipe (rappel)

```
UP parie + DOWN s'abstient  -> long
DOWN parie + UP s'abstient  -> short
les deux parient            -> CANCEL (rien)
aucun                       -> PASS (rien)
```
Filtre cross-model DOWN : si `p_up >= 0.53` → le pari DOWN est rejeté.
