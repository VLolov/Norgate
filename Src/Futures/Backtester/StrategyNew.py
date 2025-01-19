from typing import List, Optional

import pandas as pd

from Futures.Backtester.BrokerNew import BrokerNew
from Futures.Backtester.TradeNew import TradeNew
from Futures.BacktesterBase.GroupBase import GroupBase
from Futures.BacktesterBase.InstrumentBase import InstrumentBase
from Futures.BacktesterBase.StrategyBase import StrategyBase



class StrategyNew(StrategyBase):
    def __init__(self, group: GroupBase):
        super().__init__()

        self.group = group
        assert group.broker is not None, f"Group: {group.name} has no broker"
        assert group.instruments, f"Group: {group.name} has no instruments"

        group.add_strategy(self)

        self.broker = group.broker
        self.instruments = group.instruments

        self.idx: int = 0
        self.time: pd.Timestamp = pd.Timestamp.min

    def check_state(self) -> bool:
        pass

    def __init__(self, group: GroupBase):
        # note: not initialized properties don't appear in child class ?!?
        super().__init__(group)

        self.warm_up_period = 0

    @property
    def curr_index(self):
        return self.group.current_index

    def trades(self, instrument) -> List[TradeNew]:
        return [trade for trade in self.broker.trades(self, instrument) if not trade.deleted]

    @staticmethod
    def data(instrument: InstrumentBase) -> pd.DataFrame:
        return instrument.data

    def add_data(self, instrument: InstrumentBase):
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            assert col in instrument.data.columns, f'Missing required column "{col}" in dataframe'
        self._instruments.append(instrument)

    def next(self):
        pass

    def last(self):
        pass

    def get_value(self, instrument, column_name: str | List[str], offset: int = 0):
        assert offset <= 0 <= self.curr_i + offset, f"wrong offset: {offset}"
        return instrument.data[column_name].iloc[self.curr_i + offset]

    def set_value(self, instrument, column_name: str, value, offset: int = 0):
        instrument.data.loc[self.timestamp(instrument, offset), column_name] = value

    # offset <= 0
    def open(self, instrument, offset=0) -> float:
        return instrument.data['Open'].iloc[self.curr_i + offset]

    def high(self, instrument, offset=0) -> float:
        return instrument.data['High'].iloc[self.curr_i + offset]

    def low(self, instrument, offset=0) -> float:
        return instrument.data['Low'].iloc[self.curr_i + offset]

    def close(self, instrument, offset=0) -> float:
        return instrument.data['Close'].iloc[self.curr_i + offset]
        # return self.get_value('Close', offset)

    def volume(self, instrument, offset=0) -> float:
        return instrument.data['Volume'][self.curr_i + offset]

    def timestamp(self, instrument, offset: int = 0) -> pd.Timestamp:
        return instrument.data.index[self.curr_i + offset]

    def is_roll(self, instrument):
        if 'DTE' in instrument.data.columns:
            # data may have no DTE field - this is the case when using the norgate adjusted contracts
            return instrument.data['DTE'].iloc[self.curr_i - 1] < instrument.data['DTE'].iloc[self.curr_i]

        return False


if __name__ == "__main__":
    print('hi')
    sn = StrategyNew()
    print(sn.id)
    sn = StrategyNew()
    print(sn.id)
