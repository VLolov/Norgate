from tqdm import tqdm

import Futures.BacktesterBase as Bb


class Group(Bb.GroupBase):
    def run(self):
        dates = self.instruments[0].data.index
        for strategy in self.strategies:
            strategy.idx = 0
            strategy.time = dates[0]
            strategy.init()

        for idx, timestamp in enumerate(tqdm(dates, desc='Processing days', colour='green')):
            for strategy in self.strategies:
                strategy.idx = idx
                strategy.time = timestamp
                strategy.next()

        for strategy in self.strategies:
            strategy.idx = len(dates) - 1
            strategy.time = dates[-1]
            strategy.last()

    def check_state(self) -> bool:
        return (
            len(self.instruments) > 0
            and len(self.strategies) > 0
            and self.backtester is not None
            and self.broker is not None
        )
