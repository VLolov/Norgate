
import logging
from abc import ABC

import numpy as np

from Futures.Backtester.BacktesterNew import BacktesterNew
from Futures.BacktesterBase.BacktesterBase import BacktesterBase
from Futures.BacktesterBase.PortfolioBase import PortfolioBase


class PortfolioNew(PortfolioBase, ABC):
    def __init__(self, backtester: BacktesterBase):
        super().__init__()
        self.initial_capital: float = np.nan
        self.backtester = backtester
        backtester.portfolio = self


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    class BacktesterX(BacktesterNew):
        def __init__(self):
            super().__init__()

        def check_state(self) -> bool:
            return True


    class PortfolioX(PortfolioBase):
        def __init__(self):
            super().__init__(BacktesterX)
            self.some_prop = 1

        def check_state(self) -> bool:
            return self.initial_capital is not np.nan and self.some_prop == 1


    p = PortfolioX()
    p.initial_capital = 100_000

    # Base.print_instances()

    p.print_instances()
