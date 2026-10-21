"""log_regime.py -- ecrit logs/regime_log.csv : une ligne par bougie 1h (score,
les 3 indicateurs normalises, regime confirme), sur tout l'historique disponible.
Config par defaut (LOOKBACK_DAYS_DEFAULT / CALIBRATION_METHOD de config_regime.py).
Ecrase le fichier a chaque run (meme convention que modele_team/logs/*.csv).

Usage:
    cd backtest/modele_updown/1h/regime
    python log_regime.py
"""
from __future__ import annotations

from pathlib import Path

try:
    from . import api
    from . import config_regime as _cfg
except ImportError:
    import api
    import config_regime as _cfg

ROOT = Path(__file__).resolve().parent
LOGS_DIR = ROOT / "logs"
REGIME_LOG_CSV = LOGS_DIR / "regime_log.csv"


def write_regime_log(lookback_days: int = None, calibration_method: str = None) -> Path:
    df = api.evaluate_regime_full(lookback_days, calibration_method)
    LOGS_DIR.mkdir(exist_ok=True)
    df.to_csv(REGIME_LOG_CSV, index=False)
    return REGIME_LOG_CSV


if __name__ == "__main__":
    path = write_regime_log()
    print(f"written {path} ({sum(1 for _ in open(path, encoding='utf-8')) - 1} bars)")
