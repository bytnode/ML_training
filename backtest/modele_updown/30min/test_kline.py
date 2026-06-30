import requests
import numpy as np
from datetime import datetime, timedelta

symbol = "BTCUSDT"  # change si besoin
interval = "4h"
days = 365

# Récupération par batch (Binance max 1000 candles/requête)
end = int(datetime.now().timestamp() * 1000)
start = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
candles = []

while start < end:
    r = requests.get("https://api.binance.com/api/v3/klines",
                      params={"symbol": symbol, "interval": interval,
                              "startTime": start, "limit": 1000})
    data = r.json()
    if not data:
        break
    candles.extend(data)
    start = data[-1][0] + 1

opens = np.array([float(c[1]) for c in candles])
closes = np.array([float(c[4]) for c in candles])
fluct = (closes - opens) / opens * 100

green = fluct[fluct > 0]
red = fluct[fluct < 0]

print(f"Bougies totales: {len(fluct)}")
print(f"Vertes: {len(green)} — fluctuation moyenne: +{green.mean():.4f}%")
print(f"Rouges: {len(red)} — fluctuation moyenne: {red.mean():.4f}%")