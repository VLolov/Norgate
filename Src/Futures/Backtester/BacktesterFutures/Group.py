import typing

from tqdm import tqdm

import Futures.Backtester.BacktesterBase as Bb
from .Strategy import Strategy


class Group(Bb.GroupBase):
    def run(self):
        dates = self.instruments[0].data.index
        for strategy in self.strategies:
            strategy.idx = 0
            strategy.dt = dates[0]
            strategy.init()

        for idx, timestamp in enumerate(tqdm(dates, desc='Processing days', colour='green')):
        # for idx, timestamp in enumerate(dates):
            for strategy in self.strategies:
                strategy = typing.cast(Strategy, strategy)
                if idx > strategy.warm_up_period:
                    strategy.idx = idx
                    strategy.dt = timestamp
                    strategy.next()

        for strategy in self.strategies:
            strategy.idx = len(dates) - 1
            strategy.dt = dates[-1]
            strategy.last()
            strategy.ready = True

    def check_state(self) -> bool:
        return (
            len(self.instruments) > 0
            and len(self.strategies) > 0
            and self.backtester is not None
            and self.broker is not None
        )
