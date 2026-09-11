import os
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import yfinance as yf
import pandas as pd
import ta

# 1. FETCH 15-MINUTE BTC DATA (Last 60 days limit for 15m interval on yfinance)
data = yf.download(tickers="BTC-USD", period="60d", interval="15m")

# Clean multi-index columns if returned by yfinance
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

# 2. DEFINE THE RSI + MA STRATEGY
class RsiMaStrategy(Strategy):
    rsi_period = 14
    rsi_ma_period = 14
    oversold_threshold = 30
    take_profit_rsi = 65
    stop_loss_pct = 0.0185  # 1.85% Stop Loss

    def init(self):
        # Calculate RSI 14
        rsi_series = ta.momentum.rsi(self.data.Close.s, window=self.rsi_period)
        self.rsi = self.I(lambda: rsi_series, name="RSI")
        
        # Calculate Moving Average of the RSI line
        rsi_ma_series = ta.trend.sma_indicator(pd.Series(self.rsi), window=self.rsi_ma_period)
        self.rsi_ma = self.I(lambda: rsi_ma_series, name="RSI_MA")
        
        # Track whether RSI recently dipped into oversold territory
        self.was_oversold = False

    def next(self):
        # Track if RSI dropped below 30
        if self.rsi[-1] < self.oversold_threshold:
            self.was_oversold = True

        # Check if we are currently in a trade
        if not self.position:
            # BUY CONDITION: Was oversold AND RSI crosses ABOVE its MA
            if self.was_oversold and crossover(self.rsi, self.rsi_ma):
                entry_price = self.data.Close[-1]
                sl_price = entry_price * (1 - self.stop_loss_pct)
                
                # Execute buy order with 1.85% fixed stop loss
                self.buy(sl=sl_price)
                self.was_oversold = False  # Reset oversold flag
                
        else:
            # SELL CONDITION: Take profit when RSI reaches 65
            if self.rsi[-1] >= self.take_profit_rsi:
                self.position.close()

# 3. RUN BACKTEST
bt = Backtest(
    data, 
    RsiMaStrategy, 
    cash=10000, 
    commission=0.001,  # 0.1% spot fee
    exclusive_orders=True  # Guarantees only 1 open trade at a time
)

stats = bt.run()

# 4. OUTPUT RESULTS
print("\n" + "="*40)
print("   RSI 15M STRATEGY METRICS")
print("="*40 + "\n")
print(stats)

# Create the results directory expected by GitHub Actions
os.makedirs("results", exist_ok=True)

# Save output to text file for artifact generation
with open("results/backtest_report.txt", "w") as f:
    f.write(str(stats))
