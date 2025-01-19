from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, List, TYPE_CHECKING

import pandas as pd

from Futures.BacktesterBase.Base import Base

if TYPE_CHECKING:
    from .GroupBase import GroupBase
    from .InstrumentBase import InstrumentBase
    from .TradeBase import TradeBase


class StrategyBase(Base, ABC):

    def __init__(self, name: str):
        super().__init__()
        self. name = name
        self.group: Optional[GroupBase] = None

    def set_group(self, group: GroupBase):
        self.group = group
        return self
    
    @property
    def trades(self) -> List[TradeBase]:
        return self.group.broker.trades

    @property
    def instruments(self) -> List[InstrumentBase]:
        return self.group.instruments

    @abstractmethod
    def open(self, instrument: InstrumentBase, idx):
        ...

    @abstractmethod
    def high(self, instrument, idx):
        ...

    @abstractmethod
    def low(self, instrument, idx):
        ...

    @abstractmethod
    def close(self, instrument, idx):
        ...

    @abstractmethod
    def volume(self, instrument, idx):
        ...

    @abstractmethod
    def timestamp(self, instrument, idx):
        ...

    @abstractmethod
    def init(self, idx: int, timestamp: pd.Timestamp):
        ...

    @abstractmethod
    def next(self, idx: int, timestamp: pd.Timestamp):
        ...

    @abstractmethod
    def last(self, idx: int, timestamp: pd.Timestamp):
        # called once after the last next()
        ...
