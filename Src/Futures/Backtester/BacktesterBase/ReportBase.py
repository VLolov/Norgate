from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, List, TYPE_CHECKING

from Futures.Backtester.BacktesterBase.Base import Base

if TYPE_CHECKING:
    from .BacktesterBase import BacktesterBase
    from .PlotBase import PlotBase


class ReportBase(Base, ABC):
    def __init__(self, name: str):
        super().__init__()
        self.name: str = name
        self.backtester: Optional[BacktesterBase] = None
        self.plots: List[PlotBase] = []

    def __repr__(self):
        return (f"<{self.__class__.__name__} id: {self.id}, "
                f"backtester: {self.id_string(self.backtester)}>")

    @abstractmethod
    def run(self):
        ...

    def set_backtester(self, backtester: BacktesterBase):
        self.backtester = backtester
        return self

    def add_plot(self, plot: PlotBase):
        self.plots.append(plot)