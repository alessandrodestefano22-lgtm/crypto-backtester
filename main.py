import os
import requests
import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

# 1. FETCH HISTORICAL 15M BTC DATA
def fetch_binance_15m(symbol="BTCUSDT", days=180):
    endpoints = [
        "https://api.binance.us/api/v3/klines",
        "https://api.binance.com/api/v3/klines"
    ]
    
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    end_time = int(pd.Timestamp.now().timestamp() * 1000)
    start_time = int((pd.Timestamp.now() - pd.Timedelta(days=days)).timestamp() * 1000)
    
    all_candles = []
    
    for url in endpoints:
        curr_start = start_time
        all_candles = []
        try:
            while curr_start < end_time:
                params = {
                    "symbol": symbol,
                    "interval": "15m",
                    "startTime": curr_start,
                    "endTime": end_time,
                    "limit": 1000
                }
                res = requests.get(url, params=params, headers=headers, timeout=10)
                if res.status_code != 200:
                    break
                data = res.json()
                if not data or not isinstance(data, list):
                    break
                all_candles.extend(data)
                curr_start = data[-1][0] + 1
            
            if len(all_candles) > 0:
                break
        except Exception:
            continue

    if not all_candles:
        raise ValueError("Failed to retrieve candle data across all API endpoints.")

    df = pd.DataFrame(all_candles, columns=[
        "Open time", "Open", "High", "Low", "Close", "Volume",
        "Close time", "Quote volume", "Count", "Taker buy volume",
        "Taker buy quote volume", "Ignore"
    ])
    
    df["Open time"] = pd.to_datetime(df["Open time"], unit="ms")
    df.set_index("Open time", inplace=True)
    
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        
    df.dropna(inplace=True)
    return df[["Open", "High", "Low", "Close", "Volume"]]

data = fetch_binance_15m("BTCUSDT", days=180)

# Helper functions for backtesting.py indicator registration
def compute_rsi(prices, period=14):
    close = pd.Series(prices)
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0).values

def compute_sma(array, period=14):
    series = pd.Series(array)
    return series.rolling(window=period).mean().fillna(50.0).values


# 2. DEFINE STRATEGY
class RsiMaStrategy(Strategy):
    rsi_period = 14
    rsi_ma_period = 14
    take_profit_rsi = 65.0
    stop_loss_pct = 0.0185  # 1.85% Stop Loss

    def init(self):
        # Register indicators cleanly with self.I
        self.rsi = self.I(compute_rsi, self.data.Close, self.rsi_period)
        self.rsi_ma = self.I(compute_sma, self.rsi, self.rsi_ma_period)

    def next(self):
        # BUY CONDITION: RSI crosses above its MA while RSI < 45
        if not self.position:
            if crossover(self.rsi, self.rsi_ma) and self.rsi[-1] < 45.0:
                entry_price = self.data.Close[-1]
                sl_price = entry_price * (1.0 - self.stop_loss_pct)
                self.buy(sl=sl_price)
                
        # SELL CONDITION: Close trade when RSI reaches or exceeds 65
        else:
            if self.rsi[-1] >= self.take_profit_rsi:
                self.position.close()

# 3. RUN BACKTEST
bt = Backtest(
    data, 
    RsiMaStrategy, 
    cash=10000, 
    commission=0.001,
    exclusive_orders=True
)

stats = bt.run()

# 4. OUTPUT RESULTS
print("\n" + "="*40)
print("   RSI 15M STRATEGY METRICS (BINANCE DATA)")
print("="*40 + "\n")
print(stats)

os.makedirs("results", exist_ok=True)
with open("results/backtest_report.txt", "w") as f:
    f.write(str(stats))
