import pandas as pd


class TradesStatistics:
    def __init__(self):
        self.all_trades = []
        self.winning_trades = []
        self.losing_trades = []

    def add_trade(self, trade_return):
        if trade_return != 0:
            self.all_trades.append(trade_return)
        if trade_return > 0:
            self.winning_trades.append(trade_return)
        if trade_return < 0:
            self.losing_trades.append(trade_return)

    def print_summary(self):
        print(f"--- All trades:\n{pd.Series(self.all_trades).describe()}")
        print(f"--- Winning trades:\n{pd.Series(self.winning_trades).describe()}")
        print(f"--- Losing trades:\n{pd.Series(self.losing_trades).describe()}")

