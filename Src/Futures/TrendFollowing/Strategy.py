import copy
from typing import List, Optional

import pandas as pd
import numpy as np

from Futures.TrendFollowing.Future import Future


class Trade:
    def __init__(self, entry_date, entry_price, position, stop_loss, margin, momentum, sector, symbol):

        # entry_date: pd.Timestamp = None, entry_price: float = 0, position: float = 0,
        #                  exit_date: [pd.Timestamp] = None, exit_price: float = 0, is_stop_loss: bool = False,
        #                  margin: float = 0

        self._data: Optional[pd.DataFrame] = None
        self._future: Optional[Future] = None

        self.entry_date = entry_date
        self.entry_price = entry_price
        self.position = position
        self.initial_stop_loss = stop_loss
        self.stop_loss = stop_loss
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
            f"Trade - entry_date: {self.entry_date}, entry_price: {self.entry_price}, "
            f"exit_date: {self.exit_date}, exit_price: {self.exit_price}, margin: {self.margin}, "
            f"initial_stop_loss: {self.initial_stop_loss}, stop_loss: {self.stop_loss}, "
            f"positions: {self.position}, "
            f"closed: {self.is_closed}, is_stop_loss: {self.is_stop_loss}, deleted: {self.deleted}, "
            f"momentum: {self.momentum}, sector: {self.sector}, symbol: {self.symbol}, "
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


class Broker:
    def __init__(self, strategy):
        self.strategy = strategy
        self.use_stop_loss = False
        self.use_stop_orders = False  # stop loss as stop order or on close

        self.trades: List[Trade] = list()
        self.current_trade: Optional[Trade] = None  # None = no currently open trade

    @property
    def market_position(self) -> int:
        if self.current_trade is None:
            # no current_trade, return 0 for flat
            return 0
        return self.current_trade.market_position

    def _update_rolls(self):
        if (self.current_trade is not None and
                self.current_trade.position != 0
                and self.strategy.is_roll()):
            self.current_trade.rolls += 1

    def _check_stop_loss(self):
        if self.current_trade is None or not self.use_stop_loss:
            return False

        stop_level = np.nan

        if self.current_trade.position > 0:
            # long exit?
            if self.use_stop_orders:
                if self.strategy.low() < self.current_trade.stop_loss:
                    stop_level = self.current_trade.stop_loss
            else:
                if self.strategy.close() < self.current_trade.stop_loss:
                    stop_level = self.strategy.close()
        elif self.current_trade.position < 0:
            # short exit?
            if self.use_stop_orders:
                if self.strategy.high() > self.current_trade.stop_loss:
                    stop_level = self.current_trade.stop_loss
            else:
                if self.strategy.close() > self.current_trade.stop_loss:
                    stop_level = self.strategy.close()

        if stop_level is not np.nan:
            self.close_position(price=stop_level, is_stop_loss=True)
            return True
        else:
            return False

    def set_stop_loss(self, stop_loss: float):
        if self.current_trade:
            self.current_trade.stop_loss = stop_loss

    def open_position(self, position: float = 1, price: float = np.nan, stop_loss: float = np.nan,
                      margin: float = 0, momentum: float = 0) -> None:
        assert self.current_trade is None, "previous position is still open"
        # assert position == 0.0, "cannot open order with position: 0"
        entry_date = self.strategy.timestamp()
        entry_price = price if price is not np.nan else self.strategy.close()

        if stop_loss is not np.nan:
            if position > 0:
                assert stop_loss < entry_price, \
                    f"wrong stop loss price, long order, symbol: {self.strategy.future.symbol}, date: {entry_date}"
            else:
                assert stop_loss > entry_price, \
                    f"wrong stop loss price, short order, symbol: {self.strategy.future.symbol}, date: {entry_date}"

        self.current_trade = Trade(entry_date=entry_date, entry_price=entry_price,
                                   position=position, stop_loss=stop_loss, margin=margin, momentum=momentum,
                                   sector=self.strategy.future.sector, symbol=self.strategy.future.symbol)
        self.current_trade.margin = margin * abs(position)

        self.trades.append(self.current_trade)
        self.update()

    def update(self, check_stop_loss=True) -> bool:
        # return true is stop loss executed

        # write date in list of current trade
        date = self.strategy.timestamp()
        if (self.current_trade is not None
                and date not in self.current_trade.trade_dates):
            self.current_trade.trade_dates.append(date)

        stop_loss_occurred = False
        if check_stop_loss:
            stop_loss_occurred = self._check_stop_loss()

        if not stop_loss_occurred:
            self._update_rolls()

        return stop_loss_occurred

    def close_position(self, price: float = np.nan, is_stop_loss: bool = False):
        assert self.current_trade is not None, "no open position to be closed"
        exit_date = self.strategy.timestamp()
        exit_price = price if price is not np.nan else self.strategy.close()
        self.current_trade.close_trade(exit_date, exit_price, is_stop_loss)
        self.current_trade = None

    @property
    def days_since_entry(self):
        if self.current_trade is None:
            return -1
        return self.current_trade.dit


class Strategy:
    def __init__(self):
        # note: not initialized properties don't appear in child class ?!?
        self._data: Optional[pd.DataFrame] = None
        self._future: Optional[Future] = None
        self.broker = Broker(self)
        self.warm_up_period = 0
        self.curr_i: int = 0

    @property
    def trades(self) -> List[Trade]:
        return [trade for trade in self.broker.trades if not trade.deleted]

    @property
    def data(self) -> pd.DataFrame:
        return self._data

    @data.setter
    def data(self, value: pd.DataFrame):
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            assert col in value.columns, f'Missing required column "{col}" in dataframe'
        self._data = value

    @property
    def future(self) -> Future:
        return self._future

    @future.setter
    def future(self, future: Future):
        self._future = future

    @property
    def big_point(self):
        return self.future.big_point

    @property
    def tick_size(self):
        return self.future.tick_size

    @property
    def margin(self):
        return self.future.margin

    def next(self):
        pass

    def get_value(self, column_name: str | List[str], offset: int = 0):
        assert offset <= 0 <= self.curr_i + offset, f"wrong offset: {offset}"
        return self._data[column_name].iloc[self.curr_i + offset]

    def set_value(self, column_name: str, value, offset: int = 0):
        self._data.loc[self.timestamp(offset), column_name] = value

    # offset <= 0
    def open(self, offset=0) -> float:
        return self._data['Open'].iloc[self.curr_i + offset]

    def high(self, offset=0) -> float:
        return self._data['High'].iloc[self.curr_i + offset]

    def low(self, offset=0) -> float:
        return self._data['Low'].iloc[self.curr_i + offset]

    def close(self, offset=0) -> float:
        return self._data['Close'].iloc[self.curr_i + offset]
        # return self.get_value('Close', offset)

    def volume(self, offset=0) -> float:
        return self._data.iloc['Volume'][self.curr_i + offset]

    def timestamp(self, offset: int = 0) -> pd.Timestamp:
        return self._data.index[self.curr_i + offset]

    def is_roll(self):
        if 'DTE' in self._data.columns:
            # data may have no DTE field - this is the case when using the norgate adjusted contracts
            return self._data['DTE'].iloc[self.curr_i - 1] < self._data['DTE'].iloc[self.curr_i]

        return False
