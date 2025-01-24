import logging

import numpy as np
import pandas as pd

from Futures.Backtester.BacktesterBase import *


class BacktesterX(BacktesterBase):
    def __init__(self):
        super().__init__()

    def check_state(self) -> bool:
        return True


class PortfolioXYZ(PortfolioBase):
    def __init__(self):
        super().__init__()
        self.some_prop = 1

    def check_state(self) -> bool:
        return self.initial_capital is not np.nan and self.some_prop == 1


class GroupX(GroupBase):
    def check_state(self) -> bool:
        self.log.error("Check_state called")
        return (
                len(self.instruments) > 0
                and len(self.strategies) > 0
                and self.backtester is not None
        )

    def run(self):
        pass


class BrokerX(BrokerBase):
    def check_state(self) -> bool:
        return True


class StrategyX(StrategyBase):

    def open(self, instrument: InstrumentBase, idx):
        pass

    def high(self, instrument, idx):
        pass

    def low(self, instrument, idx):
        pass

    def close(self, instrument, idx):
        pass

    def volume(self, instrument, idx):
        pass

    def timestamp(self, instrument, idx):
        pass

    def is_roll(self, instrument):
        pass

    def init(self, idx: int, timestamp: pd.Timestamp):
        pass

    def next(self, idx: int, timestamp: pd.Timestamp):
        pass

    def last(self, idx: int, timestamp: pd.Timestamp):
        pass

    def check_state(self) -> bool:
        return self.name != ''


class StrategyY(StrategyBase):

    def open(self, instrument: InstrumentBase, idx):
        pass

    def high(self, instrument, idx):
        pass

    def low(self, instrument, idx):
        pass

    def close(self, instrument, idx):
        pass

    def volume(self, instrument, idx):
        pass

    def timestamp(self, instrument, idx):
        pass

    def is_roll(self, instrument):
        pass

    def init(self, idx: int, timestamp: pd.Timestamp):
        pass

    def next(self, idx: int, timestamp: pd.Timestamp):
        pass

    def last(self, idx: int, timestamp: pd.Timestamp):
        pass

    def check_state(self) -> bool:
        return self.name != ''


class InstrumentX(InstrumentBase):
    def __init__(self, symbol: str, data):
        super().__init__(symbol, data)

    def check_state(self) -> bool:
        self.log.info(f"Checking instrument: {self.symbol}")
        return self.symbol != '' and self.data is not None


class ReportX(ReportBase):
    def __init__(self, name: str):
        super().__init__(name)

    def check_state(self) -> bool:
        return self.name != '' and self.backtester is not None


class TradeX(TradeBase):
    def check_state(self) -> bool:
        return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    bt = BacktesterX()

    p = PortfolioXYZ()
    bt.set_portfolio(p)
    p.set_initial_capital(100_000).set_backtester(bt)

    gr = GroupX("My first group")
    gr.add_backtester(bt)
    bt.add_group(gr)

    broker = BrokerX().set_group(gr)
    gr.set_broker(broker)

    report = ReportX("My Report").set_backtester(bt)
    bt.add_report(report)

    strategy_x = StrategyX("")
    strategy_y = StrategyY("StrategyNameY")

    gr.add_strategies(strategy_x).add_strategies(strategy_y)
    instrument_x = InstrumentX("symbol_x", pd.DataFrame())
    instrument_y = InstrumentX("symbol_y", pd.DataFrame())

    gr.add_instruments(instrument_x).add_instruments(instrument_y)

    ts = pd.Timestamp.min
    trade_x = TradeX(strategy_x, instrument_x, ts, 1000, ts, 2000, 1)
    trade_y = TradeX(strategy_y, instrument_y, ts, 1000, ts, 2000, -1)

    broker.add_trade(trade_x).add_trade(trade_y)

    # Base.print_instances()
    p.print_instances()

    bt.print_hierarchy()
