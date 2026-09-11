import os
import requests
import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

# 1. FETCH HISTORICAL 15M BTC DATA FROM BINANCE (180 Days)
def fetch_binance_15m(symbol="BTCUSDT", days=180):
    url = "https://api.binance.com/api/v3/klines"
    end_time = int(pd.Timestamp.now().timestamp() * 1000)
    start_time = int((pd.Timestamp.now() - pd.Timedelta(days=days)).timestamp() * 1000)
    
    all_candles = []
    while start_time < end_time:
        params = {
            "symbol": symbol,
            "interval": "15m",
            "startTime": start_time,
            "endTime": end_time,
            "limit": 1000
        }
        res = requests.get(url, params=params).json()
        if not res or not isinstance(res, list):
            break
        all_candles.extend(res)
        start_time = res[-1][0] + 1  # Move past last fetched candle timestamp

    df = pd.DataFrame(all_candles, columns=[
        "Open time", "Open", "High", "Low", "Close", "Volume",
        "Close time", "Quote volume", "Count", "Taker buy volume",
        "Taker buy quote volume", "Ignore"
    ])
    
    df["Open time"] = pd.to_datetime(df["Open time"], unit="ms")
    df.set_index("Open time", inplace=True)
    numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, axis=1)
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

        # Wilder's Smoothing for Standard RSI calculation
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        
        avg_gain = gain.ewm(alpha=1/self.rsi_period, min_periods=self.rsi_period).mean()
        avg_loss = loss.ewm(alpha=1/self.rsi_period, min_periods=self.rsi_period).mean()

        rs = avg_gain / avg_loss
        rsi_series = 100.0 - (100.0 / (1.0 + rs))
        rsi_series = rsi_series.fillna(50.0)

        # Simple Moving Average of the RSI line
        rsi_ma_series = rsi_series.rolling(window=self.rsi_ma_period).mean().fillna(50.0)

        # Register indicators with backtesting engine
        self.rsi = self.I(lambda: rsi_series.to_numpy(), name="RSI")
        self.rsi_ma = self.I(lambda: rsi_ma_series.to_numpy(), name="RSI_MA")

    def next(self):
        # Check if currently in a trade
        if not self.position:
            # BUY CONDITION: RSI crosses ABOVE RSI_MA while RSI is low (< 45)
            if crossover(self.rsi, self.rsi_ma) and self.rsi[-1] < 45.0:
                entry_price = self.data.Close[-1]
                sl_price = entry_price * (1.0 - self.stop_loss_pct)
                self.buy(sl=sl_price)
                
        else:
            # SELL CONDITION: Take profit when RSI reaches or exceeds 65
            if self.rsi[-1] >= self.take_profit_rsi:
                self.position.close()

# 3. RUN BACKTEST
bt = Backtest(
    data, 
    RsiMaStrategy, 
    cash=10000, 
    commission=0.001,  # 0.1% Binance spot trading fee
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
