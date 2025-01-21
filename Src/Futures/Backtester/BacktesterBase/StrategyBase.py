from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, List, TYPE_CHECKING

import pandas as pd

from Futures.Backtester.BacktesterBase.Base import Base

if TYPE_CHECKING:
    from .GroupBase import GroupBase
    from .InstrumentBase import InstrumentBase
    from .TradeBase import TradeBase


class StrategyBase(Base, ABC):

    def __init__(self, name: str):
        super().__init__()
        self. name = name
        self.group: Optional[GroupBase] = None
        self.idx: int = 0
        self.time: Optional[pd.Timestamp] = None
        self.ready = False
        self.cost_contract: float = 0.0
        self.slippage_ticks: int = 0


    def __repr__(self):
        return (f"<{self.__class__.__name__} id: {self.id}, "
                f"name: {self.name}, "
                f"group: {self.id_string(self.group)}>")

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
    def init(self):
        ...

    @abstractmethod
    def next(self):
        ...

    @abstractmethod
    def last(self):
        # called once after the last next()
        ...

    @abstractmethod
    def is_roll(self, instrument, idx):
        ...
