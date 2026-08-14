"""_harness.py — socle commun des sweeps du cliquet de gain (campagne SECURISATION
2026-08-11).

Reproduit EXACTEMENT la partie legere de main.run_backtest (selectivite DOWN ->
xmodel UP -> coherence -> DMI -> confiance UP -> simulate_trades -> trading_stats)
mais SANS les ecritures de logs (detailed_log/full_logs), et en mettant en cache
sur disque la partie couteuse (_prepare_predictions : prediction des 2 modeles +
parquet 1-min + map DMI). Un point de sweep coute alors une simulation, pas une
re-prediction.

Working-directory contract : lance depuis modele_team/ (python optimize/xxx.py).
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import main as M                      # noqa: E402
import filters                        # noqa: E402
import config_main as C               # noqa: E402

# Cache pickle : ecrit et relu par CE script uniquement, contenu = DataFrames
# pandas issus de nos propres modeles (aucune source externe) -> pas de surface
# de deserialisation non fiable. Supprimer le fichier suffit a le reconstruire.
CACHE = ROOT / "optimize" / ".cache_prepared.pkl"


# ----------------------------------------------------------------------------
#  Cache des predictions (partie couteuse, invariante par sweep)
# ----------------------------------------------------------------------------
def load_prepared(force: bool = False):
    """(prepared, regime_map). Le cache ne depend QUE des modeles/parquet/fenetre,
    jamais des parametres de sortie balayes -> reutilisable sur tout le sweep."""
    if CACHE.exists() and not force:
        with open(CACHE, "rb") as f:
            return pickle.load(f)
    prepared = M._prepare_predictions()
    regime_map = filters.load_regime_map() if M.cfg_regime.REGIME_WRAPPER_ENABLED else None
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE, "wb") as f:
        pickle.dump((prepared, regime_map), f)
    return prepared, regime_map


# ----------------------------------------------------------------------------
#  Surcharge des parametres de sortie (module-level, pas de reload de config)
# ----------------------------------------------------------------------------
def set_params(up_ratchet=None, down_ratchet=None, up_prop=None, down_prop=None,
               up_tp=None, up_gap=None, up_activate=None,
               down_tp=None, down_gap=None, down_activate=None):
    """Surcharge les globals de main.py lus a chaque appel des sorties gerees.
    None = laisse la valeur en place. DOWN_TRAIL_GAP doit aussi etre pousse dans
    config_main : filters.effective_down_exit_params le relit la (override regime
    bear x1.3) -- oublier ce 2e point ferait diverger silencieusement les trades
    DOWN en regime bear."""
    if up_ratchet is not None:
        M.UP_RATCHET = up_ratchet
    if down_ratchet is not None:
        M.DOWN_RATCHET = down_ratchet
    M.UP_RATCHET_P = up_prop
    M.DOWN_RATCHET_P = down_prop
    if up_tp is not None:
        M.UP_TAKE_PROFIT = up_tp
    if up_gap is not None:
        M.UP_TRAIL_GAP = up_gap
    if up_activate is not None:
        M.UP_TRAIL_ACTIVATE = up_activate
    if down_tp is not None:
        M.DOWN_TAKE_PROFIT = down_tp
    if down_gap is not None:
        M.DOWN_TRAIL_GAP = down_gap
        C.DOWN_TRAIL_GAP = down_gap          # <- lu par effective_down_exit_params
    if down_activate is not None:
        M.DOWN_TRAIL_ACTIVATE = down_activate


BASE = dict(
    up_ratchet=(), down_ratchet=(), up_prop=None, down_prop=None,
    up_tp=C.UP_TAKE_PROFIT, up_gap=C.UP_TRAIL_GAP, up_activate=C.UP_TRAIL_ACTIVATE,
    down_tp=C.DOWN_TAKE_PROFIT, down_gap=C.DOWN_TRAIL_GAP, down_activate=C.DOWN_TRAIL_ACTIVATE,
)


def reset_params():
    set_params(**BASE)


# ----------------------------------------------------------------------------
#  Un point de sweep
# ----------------------------------------------------------------------------
def _team_bets(prepared, regime_map):
    """Chaine de filtres identique a main.run_backtest (lignes 650-667)."""
    up_full = prepared["up_full"]
    down_full = prepared["down_full"]
    eff_delta = filters.effective_down_delta(down_full["timestamp"].to_numpy(), regime_map)
    if regime_map is not None or C.DOWN_CONF_DELTA:
        down_full = down_full.copy()
        down_full["take"] = (down_full["confidence"] >= down_full["threshold"] + eff_delta).astype(int)
    up_bets = M._taken_bets(up_full, "up")
    up_bets = filters.apply_xmodel_filter(up_bets, down_full)
    down_bets = M._taken_bets(down_full, "down")
    team_bets, _ = M.build_team_bets(up_bets, down_bets)
    team_bets, _, _ = filters.apply_dmi_filter(team_bets, prepared.get("dmi_map"))
    team_bets, _ = filters.apply_conf_filter(team_bets, regime_map)
    return team_bets, up_full


_TEAM_CACHE = {}


def run_trades(prepared, regime_map) -> pd.DataFrame:
    """Simule et renvoie le DataFrame de trades. La chaine de filtres ne depend
    d'AUCUN parametre de sortie -> mise en cache une fois pour tout le sweep."""
    key = id(prepared)
    if key not in _TEAM_CACHE:
        team_bets, up_full = _team_bets(prepared, regime_map)
        _uf = up_full[up_full["take"] == 1]
        up_meta = {int(r["timestamp"]):
                   (int(r["volatility_state"]) if pd.notna(r["volatility_state"]) else -1,
                    int(r["is_weekend"]) if pd.notna(r["is_weekend"]) else -1)
                   for _, r in _uf.iterrows()}
        _TEAM_CACHE[key] = (team_bets, up_meta)
    team_bets, up_meta = _TEAM_CACHE[key]
    trades, _ = M.simulate_trades(team_bets, prepared["body_map"], prepared["prices_intrabar"],
                                  up_meta, prepared["extremes"], regime_map, None)
    return trades


# ----------------------------------------------------------------------------
#  Metriques (memes formules que main.trading_stats, + moities de fenetre)
# ----------------------------------------------------------------------------
def _compound(net: np.ndarray) -> float:
    return float((np.prod(1.0 + net) - 1.0) * 100)


def _ev(net: np.ndarray) -> float:
    if not len(net):
        return 0.0
    w, l = net[net > 0], net[net < 0]
    wr = len(w) / len(net)
    lr = len(l) / len(net)
    aw = float(w.mean()) if len(w) else 0.0
    al = float(-l.mean()) if len(l) else 0.0
    return (wr * aw - lr * al) * 100


def metrics(trades: pd.DataFrame, split_ms: int | None = None) -> dict:
    """net compose / EV / WR / pire trade / drawdown, global + par sens + par
    moitie de fenetre. Tout x EXPOSURE (levier), comme trading_stats."""
    if trades.empty:
        return {"n": 0, "net": -1e9, "ev": -1e9, "worst": -1e9, "dd": -1e9}
    net = trades["net_ret"].to_numpy(dtype=float) * C.EXPOSURE
    eq = np.cumprod(1.0 + net)
    dd = float(((eq - np.maximum.accumulate(eq)) / np.maximum.accumulate(eq)).min() * 100)
    out = {"n": len(net), "net": _compound(net), "ev": _ev(net),
           "wr": float((net > 0).mean() * 100), "worst": float(net.min() * 100), "dd": dd}
    for d in ("up", "down"):
        sub = trades[trades["direction"] == d]
        s = sub["net_ret"].to_numpy(dtype=float) * C.EXPOSURE
        out[f"{d}_n"] = len(s)
        out[f"{d}_net"] = _compound(s) if len(s) else 0.0
        out[f"{d}_ev"] = _ev(s)
        out[f"{d}_wr"] = float((s > 0).mean() * 100) if len(s) else 0.0
        out[f"{d}_worst"] = float(s.min() * 100) if len(s) else 0.0
    if split_ms is not None:
        for tag, mask in (("h1", trades["traded_ts"] < split_ms),
                          ("h2", trades["traded_ts"] >= split_ms)):
            s = trades.loc[mask, "net_ret"].to_numpy(dtype=float) * C.EXPOSURE
            out[f"{tag}_n"] = len(s)
            out[f"{tag}_net"] = _compound(s) if len(s) else 0.0
            out[f"{tag}_ev"] = _ev(s) if len(s) else 0.0
            for d in ("up", "down"):
                sd = trades.loc[mask & (trades["direction"] == d), "net_ret"].to_numpy(dtype=float) * C.EXPOSURE
                out[f"{tag}_{d}_ev"] = _ev(sd) if len(sd) else 0.0
                out[f"{tag}_{d}_net"] = _compound(sd) if len(sd) else 0.0
    return out


def split_point(prepared) -> int:
    """Milieu TEMPOREL de la fenetre de backtest (pas median des trades)."""
    return int((prepared["start_ms"] + prepared["end_ms"]) // 2)


def evaluate(prepared, regime_map, split_ms, **params) -> tuple[dict, pd.DataFrame]:
    reset_params()
    set_params(**params)
    trades = run_trades(prepared, regime_map)
    return metrics(trades, split_ms), trades
