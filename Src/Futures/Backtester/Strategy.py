import typing
from abc import abstractmethod
from typing import List, Optional
import pandas as pd

import Futures.BacktesterBase as Bb
from Futures.Backtester.Broker import Broker
from Futures.Backtester.Future import Future


class Strategy(Bb.StrategyBase):
    def __init__(self, name: str):
        super().__init__(name)

    def init(self):
        self.log.debug(f"init({self.idx}, {self.time})")

    def next(self):
        self.log.debug(f"next({self.idx}, {self.time})")

    def last(self):
        self.log.debug(f"last({self.idx}, {self.time})")

    def check_state(self) -> bool:
        return self.group is not None

    def get_value(self, instrument, column_name: str | List[str], idx):
        return instrument.data[column_name].iloc[idx]

    def set_value(self, instrument: Bb.InstrumentBase, column_name: str, value, idx):
        instrument.data.loc[self.timestamp(instrument, idx), column_name] = value

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

    def timestamp(self, instrument, idx) -> pd.Timestamp:
        return instrument.data.index[idx]

    def is_roll(self, instrument, idx):
        if 'DTE' in instrument.data.columns:
            # data may have no DTE field - this is the case when using the norgate adjusted contracts
            return instrument.data['DTE'].iloc[idx - 1] < instrument.data['DTE'].iloc[idx]

        return False

