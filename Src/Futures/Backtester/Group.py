import Futures.BacktesterBase as Bb


class Group(Bb.GroupBase):
    def run(self):
        dates = self.instruments[0].data.index
        for strategy in self.strategies:
            strategy.init(0, dates[0])

        for idx, timestamp in enumerate(dates):
            for strategy in self.strategies:
                strategy.next(idx, timestamp)

        for strategy in self.strategies:
            strategy.last(len(dates) - 1, dates[-1])

    def check_state(self) -> bool:
        return (
            len(self.instruments) > 0
            and len(self.strategies) > 0
            and self.backtester is not None
            and self.broker is not None
        )
