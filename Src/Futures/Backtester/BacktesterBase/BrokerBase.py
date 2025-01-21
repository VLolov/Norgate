from __future__ import annotations

from abc import ABC
from typing import Optional, List, TYPE_CHECKING

from Futures.Backtester.BacktesterBase.Base import Base

if TYPE_CHECKING:
    from .GroupBase import GroupBase
    from .PortfolioBase import PortfolioBase
    from .TradeBase import TradeBase


class BrokerBase(Base, ABC):
    def __init__(self):
        super().__init__()
        self.portfolio: Optional[PortfolioBase] = None
        self.group: Optional[GroupBase] = None
        self._trades: List[TradeBase] = []  # will not be used in child !!!

    def set_group(self, group: GroupBase):
        self.group = group
        return self

    def __repr__(self):
        return (f"<{self.__class__.__name__} id: {self.id}, "
                f"portfolio: {self.id_string(self.portfolio)}, "
                f"group: {self.id_string(self.group)}>")

    @property
    def trades(self) -> List[TradeBase]:
        return self._trades

    def add_trade(self, trade: TradeBase):
        self._trades.append(trade)
        return self


