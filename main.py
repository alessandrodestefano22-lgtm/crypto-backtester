import os
import requests
import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

# 1. FETCH HISTORICAL 15M BTC DATA (GitHub Cloud Compatible)
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

# 2. DEFINE THE RSI + MA STRATEGY
class RsiMaStrategy(Strategy):
    rsi_period = 14
    rsi_ma_period = 14
    take_profit_rsi = 65.0
    stop_loss_pct = 0.0185  # 1.85% Stop Loss

    def init(self):
        close = pd.Series(self.data.Close)
        delta = close.diff()

        # Wilder's Exponential Smoothing
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        
        avg_gain = gain.ewm(alpha=1/self.rsi_period, min_periods=self.rsi_period).mean()
        avg_loss = loss.ewm(alpha=1/self.rsi_period, min_periods=self.rsi_period).mean()

        rs = avg_gain / avg_loss
        rsi_series = 100.0 - (100.0 / (1.0 + rs))
        rsi_series = rsi_series.fillna(50.0)

        # RSI Moving Average
        rsi_ma_series = rsi_series.rolling(window=self.rsi_ma_period).mean().fillna(50.0)

        # Pass as core strategy indicators
        self.rsi = self.I(lambda: rsi_series.values, name="RSI")
        self.rsi_ma = self.I(lambda: rsi_ma_series.values, name="RSI_MA")

    def next(self):
        current_rsi = self.rsi[-1]

        if not self.position:
            # BUY: RSI crosses above its MA while RSI is in low zone (< 45)
            if crossover(self.rsi, self.rsi_ma) and current_rsi < 45.0:
                entry_price = self.data.Close[-1]
                sl_price = entry_price * (1.0 - self.stop_loss_pct)
                self.buy(sl=sl_price)
                
        else:
            # SELL: Exit trade when RSI reaches or exceeds 65
            if current_rsi >= self.take_profit_rsi:
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
