from __future__ import annotations

from abc import ABC
from typing import Optional, TYPE_CHECKING

from Futures.Backtester.BacktesterBase.Base import Base

if TYPE_CHECKING:
    from .BacktesterBase import BacktesterBase


class PortfolioBase(Base, ABC):

    # from Futures.BacktesterBase.BacktesterBase import BacktesterBase
    def __init__(self):
        super().__init__()
        self.initial_capital: Optional[float] = None
        self.backtester: Optional[BacktesterBase] = None

    def check_state(self) -> bool:
        return self.backtester is not None and self.initial_capital is not None

    def set_initial_capital(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.log.warning(f"Setting initial capital: {initial_capital}")
        return self

    def set_backtester(self, backtester: BacktesterBase):
        self.backtester = backtester
        return self

    def __repr__(self):
        return (f"<{self.__class__.__name__} id: {self.id}, "
                f"initial_capital: {self.initial_capital}>")

