import pandas as pd

import Futures.BacktesterBase as Bb


class Trade(Bb.TradeBase):
    def check_state(self) -> bool:
        return True

    def __init__(self, strategy, instrument, entry_date, entry_price, position, stop_loss, margin, momentum, sector, symbol):

        super().__init__(
            strategy=strategy,
            instrument=instrument,
            entry_date=entry_date,
            entry_price=entry_price,
            exit_date=None,
            exit_price=0.0,
            position=position
        )

        self.initial_stop_loss = stop_loss
        self.stop_loss = stop_loss

        self.is_stop_loss = False

        self.trade_dates = []

        self.margin = margin
        self.momentum = momentum
        self.sector: str = sector
        self.symbol = symbol

        self.exit_date = None
        self.exit_price = 0

        self.is_stop_loss = False

        self.deleted: bool = False
        self.is_closed = False
        self.costs = 0
        self.rolls = 0

        self.trade_dates = []

    def __str__(self):
        return (
            f"Trade - strategy: {self.strategy}, instrument: {self.instrument}, "
            f"entry_date: {self.entry_date}, entry_price: {self.entry_price}, "
            f"exit_date: {self.exit_date}, exit_price: {self.exit_price}, "
            f"initial_stop_loss: {self.initial_stop_loss}, stop_loss: {self.stop_loss}, "
            f"positions: {self.position}, "
            f"closed: {self.is_closed}, is_stop_loss: {self.is_stop_loss}, deleted: {self.deleted}, "
            f"DIT: {self.dit}, "
            f"costs: {self.costs}, rolls: {self.rolls}, "
            f"pnl: {self.pnl}"
        )

    def close_trade(self, exit_date, exit_price, is_stop_loss):
        self.exit_date = exit_date
        self.exit_price = exit_price
        self.is_stop_loss = is_stop_loss
        self.is_closed = True

    @property
    def pnl(self):
        # point value is not considered here !!!
        return (self.exit_price - self.entry_price) * self.position

    @property
    def dit(self):
        # using last_date instead of exit_date enables calling dit() before the trade is closed
        last_date = self.trade_dates[-1] if len(self.trade_dates) > 0 else self.entry_date
        n = (pd.to_datetime(last_date) - pd.to_datetime(self.entry_date)).days
        return n

    @property
    def market_position(self):
        return 1 if self.position > 0 else -1

