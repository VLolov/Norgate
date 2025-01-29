import typing
from dataclasses import dataclass

import numpy as np

from Futures.Backtester.BacktesterFutures import *


class StrategyBuyAndHold(Strategy):
    @dataclass
    class BuyAndHoldConfig(Config):
        portfolio_dollar: float = 1_000_000     # 0: get portfolio from Portfolio
        # if use_one_contract = False, buy contracts for price = portfolio_dollar / nr_positions (/ big_point)
        use_one_contract: bool = False
        close_last_trading_day: bool = True
        # attributes, required by ReportMulti
        cumulative: bool = False        # cannot use cumulative as we enter only once
        risk_position: float = 1
        risk_all_positions: float = 0
        max_positions_per_sector: int = 0
        max_margin: float = 0
        atr_multiplier: float = 0
        period: int = 0
        use_stop_orders: bool = False
        sectors = None

    def __init__(self, name='BuyAndHold', config=None):
        super().__init__(name)
        if config is None:
            self.config = self.__class__.BuyAndHoldConfig()

        self.nr_instruments = 0

        self.log.debug(f"Strategy {name} created")

    def init(self):
        self.log.debug(f"init(), dt:{self.dt})")
        super().init()

        # modify parameters of Strategy class
        cfg = self.config
        # get initial_capital from portfolio
        if cfg.portfolio_dollar == 0:
            cfg.portfolio_dollar = self.group.backtester.portfolio.initial_capital

        self.warm_up_period = 0     # no need to warm-up, we enter on the first tradable date
        self.nr_instruments = len(self.instruments)

        self.set_tradable_range_instruments()

    def next(self):
        # all this code is for testing only
        idx = self.idx
        # dt = self.dt

        broker = typing.cast(Broker, self.group.broker)

        for instrument in self.instruments:
            instrument = typing.cast(Future, instrument)
            if not self.check_tradable_range(instrument, idx):
                # self.log.debug(f"{idx} {time} {future}")
                continue

            if broker.update(self, instrument):
                # stop loss occurred, don't try to enter on the same bar
                continue

            if broker.market_position(self, instrument) == 0:
                contracts = 1.0
                cfg = self.config
                if not cfg.use_one_contract:
                    dollar_per_instrument = cfg.portfolio_dollar / self.nr_instruments
                    # contracts = dollar_per_instrument / self.close(instrument, idx) / instrument.metadata.big_point
                    contracts = dollar_per_instrument / instrument.metadata.margin  # margin = 100% of account
                    # in continuous contracts some closes are < 0, but we want only long, so use abs()
                    contracts = np.abs(contracts)
                    # contracts = np.round(contracts, 0)  # arithmetic round
                    # contracts = np.floor(contracts)  # round down

                broker.open_position(self, instrument, position=contracts, margin=instrument.metadata.margin)

    def last(self):
        self.log.debug(f"last({self.idx}, {self.dt})")

        if self.config.close_last_trading_day:
            self.close_all_trades()
        broker = typing.cast(Broker, self.group.broker)
        self.log.debug(f"Number of trades: {len(broker.trades)}")

        # print_trades(self.broker.trades)
