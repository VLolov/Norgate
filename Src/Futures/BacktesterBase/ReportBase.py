from __future__ import annotations

from abc import ABC
from typing import Optional, List, TYPE_CHECKING

from Futures.BacktesterBase.Base import Base

if TYPE_CHECKING:
    from .BacktesterBase import BacktesterBase


class ReportBase(Base, ABC):
    def __init__(self, name: str):
        super().__init__()
        self.name: str = name
        self.backtester: Optional[BacktesterBase] = None

    def __repr__(self):
        return (f"<{self.__class__.__name__} id: {self.id}, "
                f"backtester: {self.id_string(self.backtester)}>")

    def set_backtester(self, backtester: BacktesterBase):
        self.backtester = backtester
        return self
