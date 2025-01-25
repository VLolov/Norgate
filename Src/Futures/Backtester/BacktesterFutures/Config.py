import Futures.Backtester.BacktesterBase as Bb


class Config(Bb.ConfigBase):
    def check_state(self) -> bool:
        return True
