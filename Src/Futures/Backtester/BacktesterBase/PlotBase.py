from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING

from Futures.Backtester.BacktesterBase.Base import Base

if TYPE_CHECKING:
    from .BacktesterBase import ReportBase


class PlotBase(Base, ABC):
    def __init__(self, name: str):
        super().__init__()
        self.name: str = name
        self.report: Optional[ReportBase] = None

    def __repr__(self):
        return (f"<{self.__class__.__name__} id: {self.id}, "
                f"report: {self.id_string(self.report)}>")

    @abstractmethod
    def run(self):
        ...

    def set_report(self, report: ReportBase):
        self.report = report
        return self
