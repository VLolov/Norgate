from __future__ import annotations

from abc import ABC
from typing import Optional, TYPE_CHECKING

import pandas as pd

from Futures.BacktesterBase.Base import Base

if TYPE_CHECKING:
    from .GroupBase import GroupBase


class InstrumentBase(Base, ABC):

    def __init__(self, symbol: str, data: pd.DataFrame):
        super().__init__()
        self.symbol: str = symbol
        self.data: pd.DataFrame = data
        self.group: Optional[GroupBase] = None

    def set_group(self, group: GroupBase):
        self.group = group

