import os
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import yfinance as yf
import pandas as pd
import ta

# 1. FETCH 15-MINUTE BTC DATA (Last 60 days)
data = yf.download(tickers="BTC-USD", period="60d", interval="15m")

if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

# Drop any NaN rows from yfinance output
data.dropna(inplace=True)

# 2. DEFINE THE RSI + MA STRATEGY
class RsiMaStrategy(Strategy):
    rsi_period = 14
    rsi_ma_period = 14
    take_profit_rsi = 65
    stop_loss_pct = 0.0185  # 1.85% Stop Loss

    def init(self):
        # Clean Pandas Series for RSI calculation
        close_series = pd.Series(self.data.Close, index=self.data.index)
        
        rsi_vals = ta.momentum.rsi(close_series, window=self.rsi_period).fillna(50)
        rsi_ma_vals = ta.trend.sma_indicator(rsi_vals, window=self.rsi_ma_period).fillna(50)

        # Register indicators with backtesting engine
        self.rsi = self.I(lambda: rsi_vals.values, name="RSI")
        self.rsi_ma = self.I(lambda: rsi_ma_vals.values, name="RSI_MA")

    def next(self):
        # BUY CONDITION:
        # 1. We have no open position
        # 2. RSI crosses ABOVE its Moving Average
        # 3. RSI is coming out of low territory (RSI < 40)
        if not self.position:
            if crossover(self.rsi, self.rsi_ma) and self.rsi[-1] < 40:
                entry_price = self.data.Close[-1]
                sl_price = entry_price * (1 - self.stop_loss_pct)
                self.buy(sl=sl_price)
                
        # SELL CONDITION:
        # Close trade when RSI reaches or exceeds 65
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
