from __future__ import annotations

from typing import Optional, TYPE_CHECKING

import pandas as pd

from Futures.BacktesterBase.Base import Base

if TYPE_CHECKING:
    from .InstrumentBase import InstrumentBase
    from .StrategyBase import StrategyBase


class TradeBase(Base):
    def __init__(self,
                 strategy: StrategyBase = None,
                 instrument: InstrumentBase = None,
                 entry_date: pd.Timestamp = None,
                 entry_price: float = 0.0,
                 exit_date: Optional[pd.Timestamp] = None,
                 exit_price: float = 0.0,
                 position: float = 0.0):

        super().__init__()
        self.strategy = strategy
        self.instrument = instrument
        self.entry_date = entry_date
        self.entry_price = entry_price
        self.exit_date = exit_date
        self.exit_price = exit_price
        self.position = position

