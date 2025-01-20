import typing

import pandas as pd
from tabulate import tabulate

from Futures.Backtester.Broker import Broker
from Futures.Backtester.Future import Future
from Futures.Backtester.Strategy import Strategy
from Futures.Backtester.Trade import Trade, print_trades


class StrategyLoosePants(Strategy):
    def __init__(self,
                 name='LoosePants',
                 period=6 * 21,
                 atr_period=14,
                 atr_multiplier=5.0,
                 use_stop_loss=True,
                 use_trailing_stop=True,
                 use_stop_orders=False,
                 short=True,
                 long=True,
                 use_one_contract=False,
                 dollar_risk=10_000,        # if dollar_risk < 0: trade with 1 contract
                 account=1_000_000,
                 cost_contract=0.0,  # USD to trade one contract, single side
                 slippage_ticks=0.0,  # single side slippage, use TickSize to convert to USD
                 cumulative=False,      # if cumulative=True, position size is calculated based on pct_risk and account size
                                        # if cumulative=False, position size is calculated from dollar_risk
                 pct_risk=0.01,
                 order_execution_delay=0
                 ):
        super().__init__(name)
        self.period = period
        self.atr_period = atr_period
        self.use_stop_loss = use_stop_loss
        self.atr_multiplier = atr_multiplier
        self.use_trailing_stop = use_trailing_stop
        self.short = short
        self.long = long
        self.use_one_contract = use_one_contract
        self.dollar_risk = dollar_risk
        self.use_stop_orders = use_stop_orders
        self.cost_contract = cost_contract  # dollar cost per contract and per each entry/exit
        self.slippage_ticks = slippage_ticks  # ticks lost on each entry/exit
        self.account = account
        self.cumulative = cumulative
        self.pct_risk = pct_risk

        self.use_stop_orders = use_stop_orders
        self.use_stop_loss = use_stop_loss
        self.order_execution_delay = order_execution_delay

        self.momentum_lookback = self.period

        self.warm_up_period = max(2, self.period, self.atr_period, self.momentum_lookback)

        self.next_func = self.next  # next, next_counter_trend, next_buy_and_hold, next_random, next_buy_one_contract
        self.random_wait = -1
        # random.seed(42)

        self.nr_trades = 0
        self.avg_trade = 0.0
        self.avg_dit = 0.0
        self.nr_rolls = 0
        self.avg_contracts = 0
        self.avg_position_size_dollar = 0.0
        self.avg_margin = 0.0

        self.winning_trades = 0
        self.loosing_trades = 0
        self.avg_contracts = 0.0
        self.max_dd = 0.0
        self.total_costs = 0.0  # calculated trading costs for trade entry/exit and slippage
        self.final_pnl = 0.0
        self.nr_missed_trades = 0
        self.yearly_ret = 0.0
        self.sharpe = 0.0

        self.stop_loss = 0.0
        self.broker: typing.Optional[Broker] = None

        self.warm_up_period = 0
        self.close_last_trading_day = False

    def close_all_trades(self):
        for instrument in self.instruments:
            future = typing.cast(Future, instrument)
            if self.broker.market_position(self, future) != 0:
                self.broker.close_position(self, future)

    def set_trade_flags(self):
        dates = self.instruments[0].data.index.tolist()
        for instrument in self.instruments:
            future = typing.cast(Future, instrument)
            future.data['trailing_stop'] = 0.0
            future.data['can_trade'] = False
            future.data['force_close_trade'] = False

            try:
                idx_first_date = dates.index(future.first_date)
            except ValueError:
                idx_first_date = 0

            try:
                idx_last_date = dates.index(future.last_date)
            except ValueError:
                idx_last_date = len(dates) - 1

            idx_first_date += self.warm_up_period
            idx_first_date = min(idx_first_date, idx_last_date)

            future.data.loc[dates[idx_first_date]:dates[idx_last_date], 'can_trade'] = True
            if idx_last_date < len(dates) - 1:
                # don't set the flag on the last bar (this is not a 'real' signal)
                future.data.loc[dates[idx_last_date], 'force_close_trade'] = True

    def check_may_trade(self, future, idx):
        if self.get_value(future, 'force_close_trade', idx):
            if self.broker.market_position(self, future):
                self.broker.close_position(self, future)
            return False

        if self.get_value(future, 'can_trade', idx):
            return True

    def init(self):
        self.log.debug(f"init({self.idx}, {self.time})")

        self.broker = typing.cast(Broker, self.group.broker)
        self.broker.setup(initial_capital=self.account, use_stop_loss=self.use_stop_loss,
                          use_stop_orders=self.use_stop_orders)

        self.set_trade_flags()

    def next(self):
        # all this code is for testing only
        idx = self.idx
        time = self.time
        # self.log.debug(f"next({idx}, {time})")
        
        broker = self.broker

        for instrument in self.instruments:
            future = typing.cast(Future, instrument)
            if not self.check_may_trade(future, idx):
                # self.log.debug(f"{idx} {time} {future}")
                continue

            if broker.update(self, future):
                self.set_value(future, 'trailing_stop', self.stop_loss, idx)

            for i in range(1):
                symbol = future.symbol
                open = self.open(future, idx)
                high = self.high(future, idx)
                low = self.low(future, idx)
                close = self.close(future, idx)

            if idx % 100 == 0:
                if broker.market_position(self, future):
                    broker.close_position(self, future)
            elif idx % 150 == 0:
                if not broker.market_position(self, future):
                    broker.open_position(self, future)

    def last(self):
        self.log.debug(f"last({self.idx}, {self.time})")

        if self.close_last_trading_day:
            self.close_all_trades()
        self.log.debug(f"Number of trades: {len(self.broker.trades)}")
        print_trades(self.broker.trades)
