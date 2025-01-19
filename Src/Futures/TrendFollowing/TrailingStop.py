"""
    While searching for a trailing stop strategy, I found this one
    https://medium.com/trading-bots-code/trailing-stop-in-python-0efd57b2386d
    and modified it to fit my requirements

    Note: this file is standalone test of a trailing stop strategy.
    It is not part of the TrendFollowing software

    Vasko:
    14.10.2024	Initial version
"""

import os

import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Qt5Agg")


# Configure strategy parameters
symbol = "ES"
short_ma_period = 50
long_ma_period = 200
atr_period = 14
atr_multiplier = 5


# Feature to get historical data
def get_historical_data(symbol):
    duck_db_file = os.path.dirname(__file__) + '/../norgate_futures.duckdb'
    with duckdb.connect(duck_db_file, read_only=True) as conn:
        # symbol = '&GC_CCB'
        symbol_df = conn.sql(
            f"""SELECT * FROM ContContracts
            WHERE Symbol='{symbol}'
            AND Adjusted=1 
            ORDER BY Date"""
            ).to_df()
        symbol_df.set_index('Date', inplace=True)
        symbol_df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close'}, inplace=True)
    return symbol_df


# Function to calculate ATR
def calculate_atr(df, period):
    df['tr'] = np.maximum((df['high'] - df['low']),
                          np.maximum(abs(df['high'] - df['close'].shift()),
                                     abs(df['low'] - df['close'].shift())))
    df['atr'] = df['tr'].rolling(window=period).mean()
    return df


# Function to calculate moving averages
def calculate_moving_averages(df, short_period, long_period):
    df['short_ma'] = df['close'].rolling(window=short_period).mean()
    df['long_ma'] = df['close'].rolling(window=long_period).mean()
    return df


# Simulate the strategy
def simulate_strategy(df, atr_multiplier):
    position = None
    entry_price = 0
    stop_loss = 0
    df['signal'] = 0  # 1 for buy, -1 for sell, 0 for hold
    df['trailing_stop'] = np.nan

    for i in range(1, len(df)):
        if df['short_ma'].iloc[i] > df['long_ma'].iloc[i] and df['short_ma'].iloc[i - 1] <= df['long_ma'].iloc[i - 1]:
            position = 'long'
            entry_price = df['close'].iloc[i]
            stop_loss = entry_price - df['atr'].iloc[i] * atr_multiplier
            df.loc[df.index[i], 'signal'] = 1
        elif df['short_ma'].iloc[i] < df['long_ma'].iloc[i] and df['short_ma'].iloc[i - 1] >= df['long_ma'].iloc[i - 1]:
            position = 'short'
            entry_price = df['close'].iloc[i]
            stop_loss = entry_price + df['atr'].iloc[i] * atr_multiplier
            df.loc[df.index[i], 'signal'] = -1
        elif position == 'long':
            stop_loss = max(stop_loss, df['close'].iloc[i] - df['atr'].iloc[i] * atr_multiplier)
            df.loc[df.index[i], 'trailing_stop'] = stop_loss
            if df['close'].iloc[i] < stop_loss:
                position = None
        elif position == 'short':
            stop_loss = min(stop_loss, df['close'].iloc[i] + df['atr'].iloc[i] * atr_multiplier)
            df.loc[df.index[i], 'trailing_stop'] = stop_loss
            if df['close'].iloc[i] > stop_loss:
                position = None

    return df


# Get historical data
df = get_historical_data(symbol)

# Calculate indicators
df = calculate_moving_averages(df, short_ma_period, long_ma_period)
df = calculate_atr(df, atr_period)

# Simulate strategy
df = simulate_strategy(df, atr_multiplier)

# Show results
print(df[['close', 'short_ma', 'long_ma', 'atr', 'signal', 'trailing_stop']].tail(20))

# Graph results
plt.figure(figsize=(14, 7))
plt.plot(df['close'], label='Closing price', color='black')
plt.plot(df['short_ma'], label='Short Moving Average (50)', color='blue')
plt.plot(df['long_ma'], label='Long moving average (200)', color='red')
plt.plot(df['trailing_stop'], label='Trailing Stop', linestyle='--', color='magenta')

# Filter buy and sell signals
buy_signals = df[df['signal'] == 1]
sell_signals = df[df['signal'] == -1]

# Graph buy and sell signals
plt.scatter(buy_signals.index, buy_signals['close'], marker='^', color='green', label='Buy signal', alpha=1)
plt.scatter(sell_signals.index, sell_signals['close'], marker='v', color='red', label='Sell signal', alpha=1)

plt.title('Trailing Stop Moving Average Crossover Backtester')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()
