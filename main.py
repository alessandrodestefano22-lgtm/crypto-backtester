import os
import pandas as pd
import ta
import yfinance as yf
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

# =========================================================
# 1. DEFINE TRADING STRATEGY
# =========================================================
class SmaCrossover(Strategy):
    # Default Moving Average periods (can be optimized later)
    fast_period = 10
    slow_period = 50

    def init(self):
        """Pre-calculates technical indicators before running the strategy."""
        close = pd.Series(self.data.Close)
        
        # Calculate Fast Moving Average
        self.fast_sma = self.I(
            lambda x: ta.trend.sma_indicator(close, window=self.fast_period), 
            self.data.Close
        )
        # Calculate Slow Moving Average
        self.slow_sma = self.I(
            lambda x: ta.trend.sma_indicator(close, window=self.slow_period), 
            self.data.Close
        )

    def next(self):
        """Runs bar-by-bar through historical data to evaluate buy/sell logic."""
        # BUY SIGNAL: Fast SMA crosses ABOVE Slow SMA
        if crossover(self.fast_sma, self.slow_sma):
            self.position.close()                          # Close active short positions
            self.buy(sl=self.data.Close[-1] * 0.95)        # Open Long with a 5% Stop-Loss

        # SELL SIGNAL: Fast SMA crosses BELOW Slow SMA
        elif crossover(self.slow_sma, self.fast_sma):
            self.position.close()                          # Close active long positions
            self.sell(sl=self.data.Close[-1] * 1.05)       # Open Short with a 5% Stop-Loss

# =========================================================
# 2. DOWNLOAD HISTORICAL PRICE DATA
# =========================================================
print("Downloading market data...")
symbol = "BTC-USD"
data = yf.download(symbol, start="2022-01-01", interval="1d")

# Clean Yahoo Finance multi-index formatting to fit backtesting requirements
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)
data = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

# =========================================================
# 3. CONFIGURE AND RUN BACKTEST
# =========================================================
bt = Backtest(
    data, 
    SmaCrossover, 
    cash=10000,          # Starting account balance ($10,000)
    commission=0.001,    # 0.1% transaction fee per trade (simulates broker fees)
    exclusive_orders=True
)

# Run standard backtest
stats = bt.run()

# Run automated parameter optimization (finds best Moving Average settings)
stats_opt = bt.optimize(
    fast_period=range(5, 30, 5),      # Test fast SMA from 5 to 30
    slow_period=range(30, 100, 10),    # Test slow SMA from 30 to 100
    maximize='Sharpe Ratio',          # Optimize for risk-adjusted returns
    constraint=lambda p: p.fast_period < p.slow_period
)

# =========================================================
# 4. SAVE RESULTS TO A FILE
# =========================================================
os.makedirs("results", exist_ok=True)

report_path = "results/backtest_report.txt"
with open(report_path, "w") as f:
    f.write("=========================================\n")
    f.write(f" BACKTEST RESULTS FOR {symbol}\n")
    f.write("=========================================\n\n")
    f.write("--- BASE STRATEGY METRICS ---\n")
    f.write(str(stats))
    f.write("\n\n--- OPTIMIZED STRATEGY METRICS ---\n")
    f.write(str(stats_opt))

print(f"Backtest completed successfully. Results saved to {report_path}")