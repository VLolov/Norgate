from __future__ import annotations

from abc import ABC
from typing import Optional, List, TYPE_CHECKING

from Futures.BacktesterBase.Base import Base

if TYPE_CHECKING:
    from .GroupBase import GroupBase
    from .PortfolioBase import PortfolioBase
    from .TradeBase import TradeBase


class BrokerBase(Base, ABC):
    def __init__(self):
        super().__init__()
        self.portfolio: Optional[PortfolioBase] = None
        self.group: Optional[GroupBase] = None
        self._trades_selected: List[TradeBase] = []

    def set_group(self, group: GroupBase):
        self.group = group
        return self

    # @property
    # def trades(self) -> List[TradeBase]:
    #     return self._trades
    #
    # def add_trade(self, trade: TradeBase):
    #     self._trades.append(trade)
    #     return self


