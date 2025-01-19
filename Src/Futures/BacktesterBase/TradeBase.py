from __future__ import annotations

from abc import ABC
from typing import Optional, List, TYPE_CHECKING

import pandas as pd

from Futures.BacktesterBase.Base import Base

if TYPE_CHECKING:
    from .InstrumentBase import InstrumentBase
    from .StrategyBase import StrategyBase


class TradeBase(Base, ABC):
    def __init__(self,
                 strategy: StrategyBase,
                 instrument: InstrumentBase,
                 entry_date: pd.Timestamp,
                 entry_price: float,
                 exit_date: pd.Timestamp,
                 exit_price: float,
                 position: float):

        super().__init__()
        self.strategy = strategy
        self.instrument = instrument
        self.entry_date = entry_date
        self.entry_price = entry_price
        self.exit_date = exit_date
        self.exit_price = exit_price
        self.position = position

