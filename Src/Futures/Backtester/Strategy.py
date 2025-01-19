from typing import List, Optional
import pandas as pd
from tqdm import tqdm

import Futures.BacktesterBase as Bb


class Strategy(Bb.StrategyBase):
    def __init__(self, name: str):
        super().__init__(name)
        self.warm_up_period = 0
        self.pbar = None

    def init(self, idx: int, timestamp: pd.Timestamp):
        self.log.debug(f"init({idx}, {timestamp})")
        self.pbar = tqdm(total=len(self.instruments[0].data), desc='Processing days', colour='green')

    def next(self, idx: int, timestamp: pd.Timestamp):
        self.pbar.update(1)

        self.log.debug(f"next({idx}, {timestamp})")
        for instrument in self.instruments:
            self.log.debug(f"\t{instrument.symbol} "
                           f"ohlc: {self.open(instrument,idx)}, "
                           f"{self.high(instrument,idx)}, "
                           f"{self.low(instrument,idx)}, "
                           f"{self.close(instrument,idx)}"
                           )
            for i in range(1):
                symbol = instrument.symbol
                open = self.open(instrument, idx)
                high = self.high(instrument, idx)
                low = self.low(instrument, idx)
                close = self.close(instrument, idx)

    def last(self, idx: int, timestamp: pd.Timestamp):
        self.log.debug(f"last({idx}, {timestamp})")

    def check_state(self) -> bool:
        return self.group is not None

    def get_value(self, instrument, column_name: str | List[str], idx):
        return instrument.data[column_name].iloc[idx]

    def set_value(self, instrument, column_name: str, value, idx):
        instrument.data.loc[self.timestamp(instrument, idx), column_name] = value

    # offset <= 0
    def open(self, instrument, idx) -> float:
        return instrument.data['Open'].iloc[idx]

    def high(self, instrument, idx) -> float:
        return instrument.data['High'].iloc[idx]

    def low(self, instrument, idx) -> float:
        return instrument.data['Low'].iloc[idx]

    def close(self, instrument, idx) -> float:
        return instrument.data['Close'].iloc[idx]
        # return self.get_value('Close', offset)

    def volume(self, instrument, idx) -> float:
        return instrument.data['Volume'][idx]

    def timestamp(self, instrument, idx) -> pd.Timestamp:
        return instrument.data.index[idx]

    def is_roll(self, instrument, idx):
        if 'DTE' in instrument.data.columns:
            # data may have no DTE field - this is the case when using the norgate adjusted contracts
            return instrument.data['DTE'].iloc[idx - 1] < instrument.data['DTE'].iloc[idx]

        return False

