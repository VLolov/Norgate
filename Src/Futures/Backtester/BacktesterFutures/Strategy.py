import typing
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import List
import pandas as pd

import Futures.Backtester.BacktesterBase as Bb
# from Futures.Backtester.BacktesterFutures import Broker
# from Futures.Backtester.BacktesterFutures import Future
from .Broker import Broker
from .Future import Future


class Strategy(Bb.StrategyBase, ABC):
    def __init__(self, name: str):
        super().__init__(name)
        self.warm_up_period = -1
        self.close_last_trading_day = True

    def check_state(self) -> bool:
        return self.group is not None

    @abstractmethod
    def init(self):
        # this is needed because we add columns to instrument.data, so multiple strategies on the
        # same instrument have to work on instrument copy
        self._copied_instruments = deepcopy(self.group.instruments)

    def open(self, instrument, idx) -> float:
        return instrument.data['Open'].iloc[idx]

    def high(self, instrument, idx) -> float:
        return instrument.data['High'].iloc[idx]

    def low(self, instrument, idx) -> float:
        return instrument.data['Low'].iloc[idx]

    def close(self, instrument, idx) -> float:
        return instrument.data['Close'].iloc[idx]
        # return self.get_value('Close', idx)

    def volume(self, instrument, idx) -> float:
        return instrument.data['Volume'][idx]
    #
    # def open(self, instrument, idx) -> float:
    #     return instrument.data_numpy[idx, instrument.OPEN]
    #
    # def high(self, instrument, idx) -> float:
    #     return instrument.data_numpy[idx, instrument.HIGH]
    #
    # def low(self, instrument, idx) -> float:
    #     return instrument.data_numpy[idx, instrument.LOW]
    #
    # def close(self, instrument, idx) -> float:
    #     return instrument.data_numpy[idx, instrument.CLOSE]
    #
    # def volume(self, instrument, idx) -> float:
    #     return instrument.data_numpy[idx, instrument.VOLUME]

    def timestamp(self, instrument, idx) -> pd.Timestamp:
        return instrument.data.index[idx]

    @staticmethod
    def get_value(instrument, column_name: str | List[str], idx):
        return instrument.data[column_name].iloc[idx]

    def set_value(self, instrument: Bb.InstrumentBase, column_name: str, value, idx):
        instrument.data.loc[self.timestamp(instrument, idx), column_name] = value

    def is_roll(self, instrument, idx):
        if 'DTE' in instrument.data.columns:
            # data may have no DTE field - this is the case when using the norgate adjusted contracts
            return instrument.data['DTE'].iloc[idx - 1] < instrument.data['DTE'].iloc[idx]

        return False

    def close_all_trades(self):
        broker = typing.cast(Broker, self.group.broker)
        for instrument in self.instruments:
            future = typing.cast(Future, instrument)
            if broker.market_position(self, future) != 0:
                broker.close_position(self, future)

    def set_tradable_range_instruments(self):
        dates = self.instruments[0].data.index.tolist()
        for instrument in self.instruments:
            future = typing.cast(Future, instrument)
            future.data['can_trade'] = False
            future.data['force_close_trade'] = False

            try:
                idx_first_date = dates.index(future.first_date)
            except ValueError:
                idx_first_date = 0

            try:
                idx_last_date = dates.index(future.last_date)
            except ValueError:
                idx_last_date = len(dates) - 1

            assert self.warm_up_period >= 0, "warm_up_period not set"
            idx_first_date += self.warm_up_period
            idx_first_date = min(idx_first_date, idx_last_date)

            future.data.loc[dates[idx_first_date]:dates[idx_last_date], 'can_trade'] = True
            if idx_last_date < len(dates) - 1:
                # don't set the flag on the last bar (this is not a 'real' signal)
                future.data.loc[dates[idx_last_date], 'force_close_trade'] = True

    def check_tradable_range(self, future, idx):
        if self.get_value(future, 'force_close_trade', idx):
            broker = typing.cast(Broker, self.group.broker)
            if broker.market_position(self, future):
                broker.close_position(self, future)
            return False

        return self.get_value(future, 'can_trade', idx)
