"""Inference temps reel pour le daemon team 1h (modele_up + modele_down).

Ce module charge les deux modeles XGBoost, leurs deux sidecars de calibration
et leurs deux config.py, puis applique a la volee :
  - la prediction proba[:, 1] de chaque modele (p_up, p_dn),
  - le seuil effectif par bucket (`bucket_threshold`, porte a l'identique de
    `modele_down/api.py::_bucket_threshold`),
  - le filtre cross-model (sidecar DOWN.max_p_up_runtime),
  - la regle de coherence d'equipe (UP / DOWN / CANCEL / PASS).

Conventions dures (cf. contrat) :
  - Tout en millisecondes en interne ; les buffers sont des barres
    [open_ms, o, h, l, c, v] triees ASC.
  - L'ordre des colonnes feed au modele est EPINGLE via
    `booster.feature_names` (skew connu sinon). `features_up.build_features` /
    `features_down.build_features` renvoient un dict ; on lit les colonnes dans
    l'ordre du booster.
  - `decide` recoit un dict `buffers` avec les cles
    '1h','5m','30m','4h','1d','eth_15m'.

Interfaces exposees :
  - load()                              -> charge + cache (idempotent)
  - FEATURES_UP / FEATURES_DOWN         -> list[str] (= booster.feature_names)
  - compute_buckets(buf_1h, target_open_ms) -> dict
  - bucket_threshold(sidecar, cfg, volatility_state, is_weekend, ts_ms) -> float
  - decide(buffers, target_open_ms)     -> dict complet + team rule
"""
from __future__ import annotations

import importlib.util
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

logger = logging.getLogger("live_team.predict")

# --------------------------------------------------------------------------- #
# Chemins (ancres sur ce fichier, jamais sur le cwd)                          #
# --------------------------------------------------------------------------- #

_HELPERS_DIR = Path(__file__).resolve().parent          # live_team/live_helpers
_ROOT_DIR    = _HELPERS_DIR.parent                       # live_team/
_UP_DIR      = _ROOT_DIR / "modele_up"
_DOWN_DIR    = _ROOT_DIR / "modele_down"

# config_team.py (filtres d'entree + gestion de sortie, miroir de config_main.py) :
# live_team/ doit etre sur sys.path pour l'import a plat (meme pattern que live.py).
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))
import config_team as _cfg_team

_UP_MODEL    = _UP_DIR / "btc_updown_model.json"
_UP_SIDECAR  = _UP_DIR / "btc_updown_model.thresholds.json"
_DOWN_MODEL  = _DOWN_DIR / "btc_updown_model.json"
_DOWN_SIDECAR = _DOWN_DIR / "btc_updown_model.thresholds.json"

# Constantes de fallback (= valeurs du contrat / config). On privilegie toujours
# les valeurs lues dans modele_down/config.py au runtime ; ces defauts ne servent
# que si l'attribut est absent.
#
# ATR_PERIOD n'est PAS dans config.py mais est defini dans modele_down/data.py
# (= la source d'autorite pour le calcul du bucket de volatilite). features_down
# importe deja le module `data` (sys.path sur modele_down), donc au moment ou
# compute_buckets tourne, `data` est present dans sys.modules. On lit son
# ATR_PERIOD comme defaut sans AUCUNE manipulation de chemin ici (zero risque) ;
# si le module n'est pas (encore) charge, on retombe sur 14. IMPORTANT : ce 14
# doit rester strictement egal a `data.py ATR_PERIOD`.
def _data_atr_period_default() -> int:
    data_mod = sys.modules.get("data")
    if data_mod is not None:
        val = getattr(data_mod, "ATR_PERIOD", None)
        if isinstance(val, int) and val > 0:
            return val
    return 14

_ATR_PERIOD_DEFAULT      = _data_atr_period_default()
_ATR_LONG_PERIOD_DEFAULT = 100
_VOL_HIGH_DEFAULT        = 1.5
_VOL_LOW_DEFAULT         = 0.7

# --------------------------------------------------------------------------- #
# Caches module-level (charges paresseusement, gardes vivants entre appels)   #
# --------------------------------------------------------------------------- #

_model_up:    xgb.XGBClassifier | None = None
_model_down:  xgb.XGBClassifier | None = None
_sidecar_up:   dict | None = None
_sidecar_down: dict | None = None
_cfg_up = None    # module config.py de modele_up
_cfg_down = None  # module config.py de modele_down

# Exposes apres load() : ordre des colonnes attendu par chaque booster.
FEATURES_UP:   list[str] = []
FEATURES_DOWN: list[str] = []

# Modules features_* importes paresseusement (cf. _import_feature_builders).
_feat_up_mod = None
_feat_down_mod = None


# --------------------------------------------------------------------------- #
# Import robuste des config.py des deux packages                              #
# --------------------------------------------------------------------------- #

def _load_pkg_config(pkg_dir: Path, alias: str):
    """Importe le `config.py` d'un package modele_* sans polluer durablement
    le namespace global.

    Les deux config.py s'appellent tous deux `config` : on les charge sous un
    alias unique via importlib pour eviter qu'un import ecrase l'autre.
    """
    cfg_path = pkg_dir / "config.py"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"{cfg_path} introuvable (config.py du package manquant)."
        )
    spec = importlib.util.spec_from_file_location(alias, str(cfg_path))
    module = importlib.util.module_from_spec(spec)
    # Enregistre l'alias pour que d'eventuels imports relatifs internes au
    # config.py se resolvent ; les deux alias sont distincts donc pas de clash.
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- #
# Helpers temporels (portes a l'identique des api.py)                         #
# --------------------------------------------------------------------------- #

def _hour_from_ts_ms(ts_ms) -> int:
    """Heure UTC 0-23. Porte a l'identique de api.py : sur un open 1h aligne,
    `((ts_ms // 30min) % 48) // 2` == `(ts_sec % 86400) // 3600`."""
    ts_ms = int(ts_ms)
    return int(((ts_ms // (30 * 60 * 1000)) % 48) // 2)


def _isoweekday_from_ts_ms(ts_ms) -> int:
    """1=Lundi ... 7=Dimanche (UTC). 1970-01-01 etait un jeudi (=4), donc
    iso = ((jours_depuis_epoch + 3) % 7) + 1. Porte a l'identique de api.py."""
    return ((int(ts_ms) // 86_400_000 + 3) % 7) + 1


# --------------------------------------------------------------------------- #
# bucket_threshold — PORT a l'identique de modele_down/api.py::_bucket_threshold #
# --------------------------------------------------------------------------- #

def bucket_threshold(sidecar: dict, cfg, volatility_state: int, is_weekend: int,
                     ts_ms=None) -> float:
    """Seuil effectif pour un bucket (volatility_state x is_weekend) et un ts.

    Porte a l'identique de `modele_down/api.py::_bucket_threshold` :
      - strategie verifiee (asym_wknd_excl / asym_wknd_excl_hour_boost),
      - skip_cells -> 1.0 (jamais de pari),
      - bad_hours -> 1.0,
      - T = thresholds_per_wknd[is_weekend],
      - hour_boost good_hours -> T -= delta,
      - bad_days (iso) -> T += cfg.ADAPTIVE_BAD_DAY_DELTA,
      - bad_hours_soft -> T += cfg.ADAPTIVE_SOFT_HOURS_DELTA.

    Les listes (skip_cells / bad_hours / bad_days / bad_hours_soft / hour_boost)
    sont lues dans le SIDECAR ; les deltas sont lus dans `cfg` (le config.py du
    package correspondant), exactement comme en backtest.
    """
    strategy = sidecar.get("strategy")
    if strategy not in ("asym_wknd_excl", "asym_wknd_excl_hour_boost"):
        raise ValueError(
            f"Sidecar strategy={strategy!r} non supporte. "
            f"Attendu : 'asym_wknd_excl' ou 'asym_wknd_excl_hour_boost'."
        )

    # Cellules exclues (volatility_state x is_weekend) -> jamais de pari.
    skip_cells = {(int(v), int(w)) for v, w in sidecar.get("skip_cells", [])}
    if (int(volatility_state), int(is_weekend)) in skip_cells:
        return 1.0

    # Exclusion par heure : force T=1.0 pour que toute comparaison de p echoue.
    if ts_ms is not None:
        bad_hours = set(int(h) for h in sidecar.get("bad_hours", []))
        if bad_hours and _hour_from_ts_ms(ts_ms) in bad_hours:
            return 1.0

    th_map = sidecar.get("thresholds_per_wknd", {})
    T = float(th_map.get(str(int(is_weekend)), 1.0))

    # Boost horaire (V2) : sur les good_hours, le seuil baisse de delta.
    if strategy == "asym_wknd_excl_hour_boost" and ts_ms is not None:
        hb = sidecar.get("hour_boost", {})
        good_hours = set(int(h) for h in hb.get("good_hours", []))
        delta = float(hb.get("delta", 0.0))
        if _hour_from_ts_ms(ts_ms) in good_hours:
            T -= delta

    # Penalite jours faibles (iso 1=Lun..7=Dim) : delta lu dans cfg.
    if ts_ms is not None:
        bad_days_src = sidecar.get("bad_days") or ()
        bad_days = set(int(d) for d in bad_days_src)
        if bad_days and _isoweekday_from_ts_ms(ts_ms) in bad_days:
            T += float(getattr(cfg, "ADAPTIVE_BAD_DAY_DELTA", 0.0))

    # Boost doux heures faibles : delta lu dans cfg.
    if ts_ms is not None:
        bad_hours_soft_src = sidecar.get("bad_hours_soft") or ()
        bad_hours_soft = set(int(h) for h in bad_hours_soft_src)
        if bad_hours_soft and _hour_from_ts_ms(ts_ms) in bad_hours_soft:
            T += float(getattr(cfg, "ADAPTIVE_SOFT_HOURS_DELTA", 0.0))

    return T


# --------------------------------------------------------------------------- #
# load() — charge les 2 modeles, 2 sidecars, 2 config.py (cache, idempotent)  #
# --------------------------------------------------------------------------- #

def load() -> None:
    """Charge (une seule fois) les deux XGBClassifier, leurs deux sidecars JSON
    et leurs deux config.py. Renseigne FEATURES_UP / FEATURES_DOWN avec l'ordre
    de colonnes du booster (`booster.feature_names`).

    Idempotent : un second appel ne recharge rien.
    """
    global _model_up, _model_down, _sidecar_up, _sidecar_down
    global _cfg_up, _cfg_down, FEATURES_UP, FEATURES_DOWN

    if _model_up is not None and _model_down is not None:
        return  # deja charge

    # --- Modeles XGBoost (via XGBClassifier().load_model) ------------------- #
    for path in (_UP_MODEL, _DOWN_MODEL):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} introuvable. Copie le modele entraine depuis "
                f"backtest/modele_updown/1h/modele_*/btc_updown_model.json."
            )

    m_up = xgb.XGBClassifier()
    m_up.load_model(str(_UP_MODEL))
    _model_up = m_up

    m_down = xgb.XGBClassifier()
    m_down.load_model(str(_DOWN_MODEL))
    _model_down = m_down

    # Ordre des colonnes EPINGLE depuis le booster (anti-skew).
    FEATURES_UP = list(_model_up.get_booster().feature_names)
    FEATURES_DOWN = list(_model_down.get_booster().feature_names)

    # --- Sidecars JSON ------------------------------------------------------ #
    for path in (_UP_SIDECAR, _DOWN_SIDECAR):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} introuvable. Copie le sidecar entraine depuis "
                f"backtest/modele_updown/1h/modele_*/btc_updown_model.thresholds.json."
            )
    with open(_UP_SIDECAR, encoding="utf-8") as f:
        _sidecar_up = json.load(f)
    with open(_DOWN_SIDECAR, encoding="utf-8") as f:
        _sidecar_down = json.load(f)

    # --- config.py des deux packages (deltas runtime) ----------------------- #
    _cfg_up = _load_pkg_config(_UP_DIR, "_cfg_live_up")
    _cfg_down = _load_pkg_config(_DOWN_DIR, "_cfg_live_down")


def _ensure_loaded() -> None:
    if _model_up is None or _model_down is None:
        load()


# --------------------------------------------------------------------------- #
# compute_buckets — buckets de calibration depuis le buffer 1h                #
# --------------------------------------------------------------------------- #

def compute_buckets(buf_1h: list[list], target_open_ms: int) -> dict:
    """Calcule les cles de bucket pour la bougie 1h dont open == target_open_ms.

    Reproduit a l'identique la chaine `data.py` (phase 4) :
      - atr_pct = ATR_Wilder(ATR_PERIOD) / close,
      - volatility_state : ratio atr_pct / SMA(atr_pct, ATR_LONG_PERIOD, min20)
        -> >VOLATILITY_HIGH : 2 ; <VOLATILITY_LOW : 0 ; sinon 1 ; NaN -> 1,
      - is_weekend : jour de semaine UTC >= 5,
      - hour : (ts_sec % 86400) // 3600,
      - isoweekday : 1=Lun .. 7=Dim.

    Les constantes (ATR_PERIOD, ATR_LONG_PERIOD, VOLATILITY_HIGH/LOW) sont lues
    dans modele_down/config.py au runtime (fallback = valeurs du contrat).

    Retourne {volatility_state, is_weekend, hour, isoweekday}.
    """
    _ensure_loaded()

    # Constantes lues dans le config DOWN (autorite des buckets).
    # ATR_PERIOD : absent de config.py -> defaut = data.py ATR_PERIOD (resolu au
    # moment de l'appel, ou `data` est deja charge), sinon 14.
    atr_period = int(getattr(_cfg_down, "ATR_PERIOD", _data_atr_period_default()))
    atr_long = int(getattr(_cfg_down, "ATR_LONG_PERIOD", _ATR_LONG_PERIOD_DEFAULT))
    vol_high = float(getattr(_cfg_down, "VOLATILITY_HIGH", _VOL_HIGH_DEFAULT))
    vol_low = float(getattr(_cfg_down, "VOLATILITY_LOW", _VOL_LOW_DEFAULT))

    ts_ms = int(target_open_ms)
    is_weekend = 1 if (_isoweekday_from_ts_ms(ts_ms) >= 6) else 0
    hour = _hour_from_ts_ms(ts_ms)
    isoweekday = _isoweekday_from_ts_ms(ts_ms)

    # volatility_state par defaut = 1 (mid) si le calcul echoue / NaN.
    volatility_state = 1

    if buf_1h:
        df = pd.DataFrame(buf_1h, columns=["open_ms", "open", "high", "low",
                                           "close", "volume"]).sort_values("open_ms")
        # On ne garde que les barres <= target (no-lookahead) : la bougie cible
        # doit etre la derniere prise en compte.
        df = df[df["open_ms"].astype("int64") <= ts_ms].reset_index(drop=True)
        if not df.empty:
            high = df["high"].astype(float)
            low = df["low"].astype(float)
            close = df["close"].astype(float)
            close_prev = close.shift(1)
            tr = pd.concat([
                (high - low),
                (high - close_prev).abs(),
                (low - close_prev).abs(),
            ], axis=1).max(axis=1)
            atr = tr.ewm(alpha=1.0 / atr_period, adjust=False).mean()
            atr_pct = atr / close
            atr_pct_sma = atr_pct.rolling(window=atr_long, min_periods=20).mean()
            atr_ratio = atr_pct / atr_pct_sma
            r = atr_ratio.iloc[-1]
            if pd.isna(r):
                volatility_state = 1
            elif r > vol_high:
                volatility_state = 2
            elif r < vol_low:
                volatility_state = 0
            else:
                volatility_state = 1

    return {
        "volatility_state": int(volatility_state),
        "is_weekend": int(is_weekend),
        "hour": int(hour),
        "isoweekday": int(isoweekday),
    }


# --------------------------------------------------------------------------- #
# Import paresseux des constructeurs de features (memes cles que le contrat)  #
# --------------------------------------------------------------------------- #

def _import_feature_builders():
    """Importe features_up / features_down depuis ce meme package live_helpers.

    Import paresseux : ces deux modules dependent eux-memes des helpers
    d'entrainement (add_*_features, data.py) et ne sont charges qu'au premier
    appel a decide(), pas a l'import de predict.py.
    """
    global _feat_up_mod, _feat_down_mod
    if _feat_up_mod is not None and _feat_down_mod is not None:
        return _feat_up_mod, _feat_down_mod

    # Le dossier live_helpers doit etre sur sys.path pour l'import a plat.
    helpers_str = str(_HELPERS_DIR)
    if helpers_str not in sys.path:
        sys.path.insert(0, helpers_str)

    try:
        import features_up as feat_up        # type: ignore
        import features_down as feat_down     # type: ignore
    except ImportError as exc:
        raise ImportError(
            "features_up / features_down introuvables dans live_helpers/. "
            "Ils doivent exposer build_features(...) (cf. contrat)."
        ) from exc

    _feat_up_mod = feat_up
    _feat_down_mod = feat_down
    return _feat_up_mod, _feat_down_mod


def _vectorize(feats: dict, feature_names: list[str], side: str) -> pd.DataFrame:
    """Construit un DataFrame 1-ligne dans l'ordre EXACT du booster.

    Toute feature absente du dict est une erreur de contrat (on prefere lever
    plutot que de feed un NaN silencieux qui fausserait la proba).
    """
    missing = [c for c in feature_names if c not in feats]
    if missing:
        raise RuntimeError(
            f"features {side} manquantes vs booster.feature_names : {missing}"
        )
    row = {c: feats[c] for c in feature_names}
    return pd.DataFrame([row], columns=feature_names)


def _pass_invalid(reasons: list[str], volatility_state: int, is_weekend: int,
                  hour: int) -> dict:
    """Retour de `decide` quand la validation des donnees echoue : team='PASS',
    AUCUNE prediction effectuee (p_up/p_dn = NaN), data_valid=False + raisons.

    live.py inspecte `data_valid` pour re-fetch/re-essayer puis annuler le cycle.
    """
    return {
        "p_up": float("nan"), "th_up": float("nan"), "take_up": False,
        "p_dn": float("nan"), "th_dn": float("nan"), "take_dn": False,
        "volatility_state": int(volatility_state),
        "is_weekend": int(is_weekend), "hour": int(hour),
        "team": "PASS", "filtered_reason": "data_invalid",
        "data_valid": False, "invalid_reasons": list(reasons),
    }


# --------------------------------------------------------------------------- #
# DMI (Wilder) sur le buffer 4h — PORT A L'IDENTIQUE de modele_team/main.py    #
# --------------------------------------------------------------------------- #

def _wilder_dmi(high, low, close, n: int):
    """+DI / -DI de Wilder (longueur n). Port bit-a-bit de main.py::_wilder_dmi."""
    h = np.asarray(high, dtype=float)
    l = np.asarray(low, dtype=float)
    cl = np.asarray(close, dtype=float)
    m = len(h)
    plus = np.full(m, np.nan)
    minus = np.full(m, np.nan)
    if m < 2:
        return plus, minus
    up_move = h[1:] - h[:-1]
    down_move = l[:-1] - l[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0.0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0.0), down_move, 0.0)
    prev_close = cl[:-1]
    tr = np.maximum.reduce([h[1:] - l[1:],
                            np.abs(h[1:] - prev_close),
                            np.abs(l[1:] - prev_close)])

    def _rma(x):
        return pd.Series(x).ewm(alpha=1.0 / n, adjust=False).mean().to_numpy()

    atr = _rma(tr)
    s_plus = _rma(plus_dm)
    s_minus = _rma(minus_dm)
    with np.errstate(divide="ignore", invalid="ignore"):
        plus[1:] = 100.0 * s_plus / atr
        minus[1:] = 100.0 * s_minus / atr
    return plus, minus


def _dmi_at(buf_4h: list[list], ts_ms: int, length: int) -> tuple[float, float]:
    """(+DI, -DI) de la bougie DMI_TIMEFRAME dont open_ms == ts_ms EXACTEMENT.

    Port fidele de main.py::_apply_dmi_filter : le lookup backtest est un match
    EXACT sur le timestamp (pas de forward-fill vers la derniere bougie 4h
    close) -> le filtre ne peut donc s'appliquer que sur les decisions dont le
    ts est aligne sur une cloture 4h (1 decision sur 4). On reproduit ce meme
    comportement ici plutot que de "corriger" (l'objectif est la symetrie
    stricte avec le backtest, pas une amelioration). ts inconnu -> (nan, nan).
    """
    if not buf_4h:
        return float("nan"), float("nan")
    arr = np.asarray(buf_4h, dtype=float)
    opens = arr[:, 0].astype("int64")
    idx = np.searchsorted(opens, ts_ms)
    if idx >= len(opens) or opens[idx] != ts_ms:
        return float("nan"), float("nan")
    plus_di, minus_di = _wilder_dmi(arr[:, 2], arr[:, 3], arr[:, 4], length)
    return float(plus_di[idx]), float(minus_di[idx])


# --------------------------------------------------------------------------- #
# decide — prediction + filtres d'entree (DMI/conf/xmodel/heures) + regle d'equipe #
# --------------------------------------------------------------------------- #

def decide(buffers: dict, target_open_ms: int) -> dict:
    """Decide l'action de l'equipe pour la bougie 1h dont open == target_open_ms.

    `buffers` est un dict de listes de barres [open_ms,o,h,l,c,v] ASC, cles :
        '1h', '5m', '30m', '4h', '1d', 'eth_15m'.

    Etapes :
      1. build features up + down (memes cles que le contrat),
      2. predict_proba[:, 1] -> p_up, p_dn,
      3. seuils th_up / th_dn via bucket_threshold (chacun avec SON sidecar+cfg),
      4. take_up = (p_up >= th_up) ; take_dn = (p_dn >= th_dn),
      5. filtre cross-model : si sidecar_down.max_p_up_runtime < 1.0 et
         p_up >= max_p_up_runtime -> take_dn = False,
      6. regle d'equipe :
            take_up & ~take_dn -> "UP"
            take_dn & ~take_up -> "DOWN"
            both               -> "CANCEL"
            none               -> "PASS".

    Retourne un dict complet :
        {p_up, th_up, take_up, p_dn, th_dn, take_dn,
         volatility_state, is_weekend, hour, team}.
    """
    _ensure_loaded()
    feat_up_mod, feat_down_mod = _import_feature_builders()

    ts_ms = int(target_open_ms)

    # --- Buffers (tolerant aux cles absentes) ------------------------------- #
    buf_1h  = buffers.get("1h", [])
    buf_5m  = buffers.get("5m", [])
    buf_30m = buffers.get("30m", [])
    buf_4h  = buffers.get("4h", [])
    buf_1d  = buffers.get("1d", [])
    buf_eth = buffers.get("eth_15m", [])

    logger.debug("decide : target_open_ms=%d buffers={1h:%d, 5m:%d, 30m:%d, 4h:%d, "
                 "1d:%d, eth_15m:%d}", ts_ms, len(buf_1h), len(buf_5m), len(buf_30m),
                 len(buf_4h), len(buf_1d), len(buf_eth))

    # --- Buckets de calibration -------------------------------------------- #
    buckets = compute_buckets(buf_1h, ts_ms)
    volatility_state = buckets["volatility_state"]
    is_weekend = buckets["is_weekend"]
    hour = buckets["hour"]
    logger.debug("decide : buckets vol_state=%d is_weekend=%d hour=%d isoweekday=%d",
                 volatility_state, is_weekend, hour, buckets["isoweekday"])

    # --- Validation STRUCTURELLE (AVANT build_features ET predict_proba) ---- #
    # OHLC corrompu / troue / buffer trop court / bougie cible absente => on
    # n'appelle NI build_features NI predict_proba : court-circuit en PASS.
    import validate as _validate  # live_helpers sur sys.path (cf _import_feature_builders)
    ok_struct, reasons = _validate.validate_structural(buf_1h, buf_eth, ts_ms)
    if not ok_struct:
        logger.warning("decide : validation STRUCTURELLE echouee -> PASS. Raisons: %s",
                       "; ".join(reasons))
        return _pass_invalid(reasons, volatility_state, is_weekend, hour)
    logger.debug("decide : validation structurelle OK.")

    # --- Features (memes signatures que le contrat features_up/down) -------- #
    feats_up = feat_up_mod.build_features(
        buf_1h, buf_5m, buf_30m, buf_4h, buf_1d, ts_ms
    )
    feats_dn = feat_down_mod.build_features(
        buf_1h, buf_5m, buf_30m, buf_4h, buf_1d, buf_eth, ts_ms
    )

    logger.debug("decide : features construites (up=%d, down=%d champs).",
                 len(feats_up), len(feats_dn))

    # --- Validation des FEATURES (toutes finies, ef_* ETH compris) ---------- #
    # Une feature NaN/inf (ex: ETH manquant -> ef_*) => pas de prediction.
    ok_feat, reasons = _validate.validate_features(feats_up, feats_dn)
    if not ok_feat:
        logger.warning("decide : validation FEATURES echouee -> PASS. Raisons: %s",
                       "; ".join(reasons))
        return _pass_invalid(reasons, volatility_state, is_weekend, hour)

    X_up = _vectorize(feats_up, FEATURES_UP, "up")
    X_dn = _vectorize(feats_dn, FEATURES_DOWN, "down")

    # --- Probabilites ------------------------------------------------------- #
    p_up = float(_model_up.predict_proba(X_up)[0, 1])
    p_dn = float(_model_down.predict_proba(X_dn)[0, 1])
    logger.debug("decide : probas p_up=%.6f p_dn=%.6f", p_up, p_dn)

    # --- Seuils effectifs (chacun avec SON sidecar + SON cfg) --------------- #
    th_up = bucket_threshold(_sidecar_up, _cfg_up, volatility_state, is_weekend, ts_ms)
    th_dn = bucket_threshold(_sidecar_down, _cfg_down, volatility_state, is_weekend, ts_ms)

    take_up = bool(p_up >= th_up)
    take_dn = bool(p_dn >= th_dn)
    logger.debug("decide : seuils th_up=%.6f (take_up=%s) th_dn=%.6f (take_dn=%s)",
                 th_up, take_up, th_dn, take_dn)

    # ========================================================================
    # Filtres d'ENTREE — PORT A L'IDENTIQUE de modele_team/main.py + config_team
    # (campagne 500%/cote 2026-07-01). Meme ORDRE que le backtest :
    #   1. UP_DROP_HOURS + DOWN_CONF_DELTA (niveau modele, PRE-coherence)
    #   2. UP_XMODEL (PRE-coherence, sur le take_up deja filtre par 1.)
    #   3. regle de coherence
    #   4. DMI (POST-coherence, direction survivante uniquement)
    #   5. UP_CONF_MIN (POST-coherence, UP seulement)
    # ========================================================================
    filtered_reason = None

    # --- 1a. UP : drop heures (session asiatique/overnight) ----------------- #
    if take_up and hour in _cfg_team.UP_DROP_HOURS:
        take_up = False
        filtered_reason = "up_drop_hours"

    # --- 1b. DOWN : selectivite (short ssi P(down) >= seuil + delta) -------- #
    if _cfg_team.DOWN_CONF_DELTA:
        take_dn = bool(p_dn >= th_dn + _cfg_team.DOWN_CONF_DELTA)

    # --- 2. UP : filtre cross-model (long ssi P(down) < UP_XMODEL_MAX_PDOWN) #
    if take_up and _cfg_team.UP_XMODEL_FILTER_ENABLED and p_dn >= _cfg_team.UP_XMODEL_MAX_PDOWN:
        take_up = False
        filtered_reason = "up_xmodel"

    # --- 3. Regle de coherence d'equipe -------------------------------------- #
    if take_up and not take_dn:
        team = "UP"
    elif take_dn and not take_up:
        team = "DOWN"
    elif take_up and take_dn:
        team = "CANCEL"
    else:
        team = "PASS"

    # --- 4. DMI (Wilder, 4h) : annule la direction survivante si elle contredit
    #        le DMI de la bougie de DECISION (match EXACT ts, cf _dmi_at). ---- #
    if _cfg_team.DMI_FILTER_ENABLED and team in ("UP", "DOWN"):
        plus_di, minus_di = _dmi_at(buf_4h, ts_ms, _cfg_team.DMI_LENGTH)
        if not (np.isnan(plus_di) or np.isnan(minus_di)):
            if team == "DOWN" and "down" in _cfg_team.DMI_SIDES and plus_di > minus_di:
                team = "PASS"
                filtered_reason = "dmi"
            elif team == "UP" and "up" in _cfg_team.DMI_SIDES and minus_di > plus_di:
                team = "PASS"
                filtered_reason = "dmi"

    # --- 5. UP : confiance minimale (P(up) >= UP_CONF_MIN) ------------------- #
    if _cfg_team.UP_CONF_FILTER_ENABLED and team == "UP" and p_up < _cfg_team.UP_CONF_MIN:
        team = "PASS"
        filtered_reason = "up_conf_min"

    logger.info("decide : team=%s (p_up=%.4f/th=%.4f, p_dn=%.4f/th=%.4f, "
                "vol_state=%d, wknd=%d, hour=%d, filtered_reason=%s)", team, p_up, th_up,
                p_dn, th_dn, volatility_state, is_weekend, hour, filtered_reason)

    return {
        "p_up": p_up,
        "th_up": th_up,
        "take_up": take_up,
        "p_dn": p_dn,
        "th_dn": th_dn,
        "take_dn": take_dn,
        "volatility_state": volatility_state,
        "is_weekend": is_weekend,
        "hour": hour,
        "team": team,
        "filtered_reason": filtered_reason,
        "data_valid": True,
        "invalid_reasons": [],
    }


# --------------------------------------------------------------------------- #
# Smoke test (chargement seul ; ne fetch rien)                                #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    load()
    print(f"OK load() — FEATURES_UP ({len(FEATURES_UP)}) = {FEATURES_UP}")
    print(f"OK load() — FEATURES_DOWN ({len(FEATURES_DOWN)}) = {FEATURES_DOWN}")
    print(f"sidecar_up.strategy   = {_sidecar_up.get('strategy')}")
    print(f"sidecar_down.strategy = {_sidecar_down.get('strategy')}")
    print(f"max_p_up_runtime(down) = {_sidecar_down.get('max_p_up_runtime')}")
    # Demo bucket_threshold sur un ts arbitraire.
    demo_ts = 1_700_000_000_000
    print("th_up(vol=1,wknd=0) =",
          bucket_threshold(_sidecar_up, _cfg_up, 1, 0, demo_ts))
    print("th_dn(vol=1,wknd=0) =",
          bucket_threshold(_sidecar_down, _cfg_down, 1, 0, demo_ts))
