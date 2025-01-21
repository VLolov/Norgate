import numpy as np

import Futures.Backtester.BacktesterBase as Bb


class Portfolio(Bb.PortfolioBase):
    def check_state(self) -> bool:
        return self.initial_capital is not np.nan
