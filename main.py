import os
import numpy as np
import pandas as pd
import yfinance as yf
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

# 1. FETCH 15-MINUTE BTC DATA (Last 60 days)
data = yf.download(tickers="BTC-USD", period="60d", interval="15m")

if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

data.dropna(inplace=True)

# 2. DEFINE THE RSI + MA STRATEGY
class RsiMaStrategy(Strategy):
    rsi_period = 14
    rsi_ma_period = 14
    take_profit_rsi = 65
    stop_loss_pct = 0.0185  # 1.85% Stop Loss

    def init(self):
        # Calculate standard RSI using pure Pandas
        close = pd.Series(self.data.Close)
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        
        rs = gain / loss
        rsi_series = 100 - (100 / (1 + rs))
        rsi_series = rsi_series.fillna(50)

        # Calculate Moving Average of the RSI line
        rsi_ma_series = rsi_series.rolling(window=self.rsi_ma_period).mean().fillna(50)

        # Bind indicator arrays to backtesting engine
        self.rsi = self.I(lambda: rsi_series.to_numpy(), name="RSI")
        self.rsi_ma = self.I(lambda: rsi_ma_series.to_numpy(), name="RSI_MA")

    def next(self):
        # BUY CONDITION: RSI crosses above its Moving Average while below 45
        if not self.position:
            if crossover(self.rsi, self.rsi_ma) and self.rsi[-1] < 45:
                entry_price = self.data.Close[-1]
                sl_price = entry_price * (1 - self.stop_loss_pct)
                self.buy(sl=sl_price)
                
        # SELL CONDITION: Exit when RSI hits or exceeds 65
        else:
            if self.rsi[-1] >= self.take_profit_rsi:
                self.position.close()

# 3. RUN BACKTEST
bt = Backtest(
    data, 
    RsiMaStrategy, 
    cash=10000, 
    commission=0.001,  # 0.1% spot fee
    exclusive_orders=True
)

stats = bt.run()

# 4. OUTPUT RESULTS
print("\n" + "="*40)
print("   RSI 15M STRATEGY METRICS")
print("="*40 + "\n")
print(stats)

os.makedirs("results", exist_ok=True)
with open("results/backtest_report.txt", "w") as f:
    f.write(str(stats))
