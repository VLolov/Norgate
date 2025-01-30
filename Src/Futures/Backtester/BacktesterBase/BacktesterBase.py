from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING, List

from .Base import Base

if TYPE_CHECKING:
    from .PortfolioBase import PortfolioBase
    from .GroupBase import GroupBase
    from .ReportBase import ReportBase


class BacktesterBase(Base, ABC):
    def check_state(self) -> bool:
        return True

    def __init__(self):
        super().__init__()

        self.portfolio: Optional[PortfolioBase] = None
        self.groups: List[GroupBase] = []
        self.reports: List[ReportBase] = []

    def __repr__(self):
        return (f"<{self.__class__.__name__} id: {self.id},"
                f"portfolio id: {self.id_string(self.portfolio)}, "
                f"groups count: {len(self.groups)}, "
                f"reports count: {len(self.reports)}>")

    @abstractmethod
    def run(self):
        ...

    def set_portfolio(self, portfolio: PortfolioBase):
        self.portfolio = portfolio
        self.log.info(f"Set portfolio object")
        return self

    def add_group(self, group: GroupBase):
        self.groups.append(group)
        return self

    def add_report(self, report: ReportBase):
        self.reports.append(report)
        return self

    def print_hierarchy(self):

        def log(obj, lvl=0):
            state = ''
            check = getattr(obj, "check_state", None)
            if callable(check):
                state = check()
                if not state:
                    state = "*** " + str(state) + " ***"
            name = getattr(obj, "name", None)
            name = f'"{name}"' if name else ''

            symbol = getattr(obj, "symbol", None)
            symbol = f'"{symbol}"' if symbol else ''

            tabs = '\t' * lvl
            self.log.debug(f'{tabs} {obj} {name} {symbol} {state}')

        level = 1
        self.log.debug("")
        log(self, level)

        log(f'Portfolio:', level)
        log(self.portfolio, level + 1)

        log(f'Reports ({len(self.reports)}):', level)
        for report in self.reports:
            log(report, level + 1)

            for plot in report.plots:
                log(plot, level + 2)

        log(f'Groups ({len(self.groups)}):', level)
        for group in self.groups:
            log(group, level + 1)
            log(group.broker, level + 2)

            # log(f"Trades ({len(group.broker.trades)}):", level + 2) @@@
            # for trade in group.broker.trades:
            #     log(trade, level + 3)

            log(f'Strategies ({len(group.strategies)}):', level + 2)
            for strategy in group.strategies:
                log(strategy, level + 3)

            log(f'Instruments ({len(group.instruments)}):', level + 2)
            for instrument in group.instruments:
                log(instrument, level + 3)

