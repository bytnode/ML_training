"""validate.py — couche de validation des donnees AVANT prediction (live_team).

But : empecher toute prediction sur des donnees douteuses. Appele par
`predict.decide` :
  - `validate_structural(...)` AVANT la construction des features (et donc avant
    `predict_proba`) : controle le buffer OHLC 1h brut + l'OHLC ETH.
  - `validate_features(...)` APRES la construction des features : toutes les
    features des deux modeles doivent etre finies.

Si une verification echoue, `decide` court-circuite en team="PASS" (AUCUNE
prediction) et `live.py` re-fetch + re-essaie jusqu'a VALIDATION_MAX_RETRIES,
puis ANNULE le cycle (aucun pari) avec une alerte Telegram.

Toutes les verifications sont STRUCTURELLES : aucune ne rejette une volatilite
de marche reelle (pas de seuil sur l'amplitude des rendements). On detecte :
  1. buffer 1h trop court (warmup des rolling insuffisant) ;
  2. bougie cible absente du buffer 1h ;
  3. OHLC brut corrompu (1h + ETH) : o/h/l/c non finis ou <= 0, high<low,
     open/close hors [low, high], volume < 0, NaN ;
  4. TROUS : open_ms non espaces de l'intervalle exact sur la fenetre recente ;
  5. (validate_features) une feature d'un des deux modeles vaut NaN/inf
     (ef_* ETH compris : ETH manquant = feature incomplete -> retry puis annule).

Convention buffers : liste de barres `[open_ms, o, h, l, c, v]` triees ASC,
open_ms en MILLISECONDES (identique au contrat predict/features).
"""
from __future__ import annotations

import math
from typing import Sequence

INTERVAL_1H_MS = 60 * 60 * 1000
INTERVAL_ETH_MS = 15 * 60 * 1000   # ETH fetch = 15m

# Defauts si config_team est introuvable (ne devrait pas arriver au runtime).
_DEFAULTS = {
    "VALIDATION_ENABLED": True,
    "VALIDATION_MIN_1H_BARS": 200,
    "VALIDATION_GAP_WINDOW": 150,
}


def _cfg(name: str):
    """Lit une constante VALIDATION_* dans config_team (soft-import), sinon defaut.

    config_team est au niveau live_team/ (parent de live_helpers/) et deja sur
    sys.path au runtime (live.py l'importe en top-level). On ne plante jamais si
    absent : on retombe sur les defauts du contrat.
    """
    try:
        import config_team as c  # type: ignore
        return getattr(c, name, _DEFAULTS.get(name))
    except Exception:  # noqa: BLE001
        return _DEFAULTS.get(name)


def _enabled() -> bool:
    return bool(_cfg("VALIDATION_ENABLED"))


# --------------------------------------------------------------------------- #
# Briques de controle                                                         #
# --------------------------------------------------------------------------- #
def _check_ohlc_integrity(window: Sequence, label: str) -> list[str]:
    """Verifie l'integrite OHLCV de chaque barre de `window`.

    Tolerance relative `tol` sur les bornes open/close in [low, high] pour
    absorber les arrondis de l'exchange (prix ~1e5 -> tol ~0.1$).
    """
    reasons: list[str] = []
    for b in window:
        if b is None or len(b) < 6:
            reasons.append(f"{label}: barre malformee ({b!r})")
            continue
        try:
            o, h, l, c, v = float(b[1]), float(b[2]), float(b[3]), float(b[4]), float(b[5])
        except (TypeError, ValueError):
            reasons.append(f"{label}: OHLCV non numerique (open_ms={b[0]})")
            continue
        if any(not math.isfinite(x) for x in (o, h, l, c, v)):
            reasons.append(f"{label}: NaN/inf dans OHLCV (open_ms={b[0]})")
            continue
        if min(o, h, l, c) <= 0:
            reasons.append(f"{label}: prix <= 0 (open_ms={b[0]})")
        tol = 1e-6 * max(1.0, h)
        if h < l - tol:
            reasons.append(f"{label}: high<low (open_ms={b[0]})")
        if h < max(o, c) - tol or l > min(o, c) + tol:
            reasons.append(f"{label}: open/close hors [low,high] (open_ms={b[0]})")
        if v < 0:
            reasons.append(f"{label}: volume<0 (open_ms={b[0]})")
        if len(reasons) >= 8:   # cap : inutile de tout lister
            break
    return reasons


def _check_gaps(window: Sequence, tf_ms: int, label: str) -> list[str]:
    """Verifie que les open_ms consecutifs sont espaces de `tf_ms` exactement."""
    reasons: list[str] = []
    for i in range(1, len(window)):
        d = int(window[i][0]) - int(window[i - 1][0])
        if d != tf_ms:
            reasons.append(
                f"{label}: trou de {d/60000:.0f}min entre "
                f"{int(window[i-1][0])} et {int(window[i][0])} (attendu {tf_ms/60000:.0f}min)"
            )
            if len(reasons) >= 5:
                break
    return reasons


# --------------------------------------------------------------------------- #
# API publique                                                                #
# --------------------------------------------------------------------------- #
def validate_structural(buf_1h: Sequence, buf_eth: Sequence | None,
                        target_open_ms: int) -> tuple[bool, list[str]]:
    """Controles AVANT build_features : buffer 1h (longueur, cible, integrite,
    trous) + integrite ETH. Retourne (ok, raisons)."""
    if not _enabled():
        return True, []

    min_bars = int(_cfg("VALIDATION_MIN_1H_BARS"))
    gap_window = int(_cfg("VALIDATION_GAP_WINDOW"))
    reasons: list[str] = []
    ts = int(target_open_ms)

    if not buf_1h:
        return False, ["buffer 1h vide"]
    if len(buf_1h) < min_bars:
        reasons.append(f"buffer 1h trop court : {len(buf_1h)} < {min_bars} barres")

    opens = [int(b[0]) for b in buf_1h]
    if ts not in opens:
        # Sans la cible on ne peut pas decouper la fenetre : echec immediat.
        reasons.append(f"bougie cible {ts} absente du buffer 1h")
        return False, reasons

    idx = opens.index(ts)
    lo = max(0, idx + 1 - gap_window)
    window = buf_1h[lo: idx + 1]
    reasons += _check_ohlc_integrity(window, "1h")
    reasons += _check_gaps(window, INTERVAL_1H_MS, "1h")

    # ETH : integrite seulement (la fenetre recente). ETH absent/incomplet pour la
    # cible se traduira en ef_* NaN -> capte par validate_features.
    if buf_eth:
        reasons += _check_ohlc_integrity(list(buf_eth)[-gap_window:], "eth15m")

    return (len(reasons) == 0), reasons


def validate_features(feats_up: dict, feats_dn: dict) -> tuple[bool, list[str]]:
    """Controle APRES build_features : toutes les valeurs des deux dicts de
    features doivent etre finies (pas de NaN/inf). Retourne (ok, raisons)."""
    if not _enabled():
        return True, []
    reasons: list[str] = []
    for side, feats in (("up", feats_up), ("down", feats_dn)):
        for k, v in feats.items():
            ok = isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
            if not ok:
                reasons.append(f"feature {side}.{k} non-finie ({v!r})")
    return (len(reasons) == 0), reasons
