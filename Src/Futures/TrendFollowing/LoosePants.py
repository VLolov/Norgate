"""
    Comes from: https://newsletter.tradingstrategies.live/p/practical-diversified-trend-following

    Here’s a Classic “Loose Pants” Trend Following strategy:
    = Rules:
        * Enter on 6-month high (for longs) or 6-month low (for shorts)
        * Exit based on 5-ATR trailing ‘chandelier’ stop and Donchian stop (rolling 2 month low)

    = Fixed fractional position sizing. Each position risks a fixed % of principal.
    = Size positions at entry and keep fixed.
    = Trade every liquid market. Every market we can get our hands on.
    = Long + Short the same way in every market. Same rules.
    = Take all trades (can end up holding lots of positions).

    Lightweight chart:
    https://github.com/louisnw01/lightweight-charts-python/blob/main/docs/source/tutorials/getting_started.md
    https://lightweight-charts-python.readthedocs.io/en/latest/examples/table.html
    https://www.youtube.com/watch?v=TlhDI3PforA&ab_channel=PartTimeLarry
    https://www.insightbig.com/post/replicating-tradingview-chart-in-python


    Vasko:
    13.10.2024	Initial version
"""
import os
import random

from datetime import datetime
from typing import Callable

import numpy as np
import pandas as pd
import duckdb

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from tabulate import tabulate

from Futures.TrendFollowing.DataAccess import DataAccess
from Futures.TrendFollowing.Timer import Timer
from Futures.TrendFollowing.Strategy import Strategy
from Futures.TrendFollowing.Indicator import Indicator
from Futures.TrendFollowing.Future import Future

matplotlib.use("Qt5Agg")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)

DUCK_DB = os.path.dirname(__file__) + '/../norgate_futures.duckdb'


class LoosePants(Strategy):
    def __init__(self, *,
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
        super().__init__()
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

        self.broker.use_stop_orders = use_stop_orders
        self.broker.use_stop_loss = use_stop_loss
        self.order_execution_delay = order_execution_delay

        self.momentum_lookback = self.period

        self.warm_up_period = max(2, self.period, self.atr_period, self.momentum_lookback)

        self.next_func = self.next  # next, next_counter_trend, next_buy_and_hold, next_random, next_buy_one_contract
        self.random_wait = -1
        random.seed(42)

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

        self.stop_loss = 0

    def set_next(self, fn: Callable):
        self.next_func = fn

    def atr(self, offset=0):
        return self.get_value('Atr', offset)

    def ema40(self, offset=0):
        return self.get_value('Ema40', offset)

    def ema80(self, offset=0):
        return self.get_value('Ema80', offset)

    def up(self, offset=0):
        return self.get_value('Up', offset)

    def down(self, offset=0):
        return self.get_value('Down', offset)

    def std(self, offset=0):
        return self.get_value('Std', offset)

    def run(self, future, data):
        self.future = future
        self.data = data
        assert len(self._data) > 200, f"History too short for {future}"
        # self.plot_atr()
        self._breakout_strategy()

    def calc_nr_contracts(self, position_dollar, stop_loss_distance):
        contracts = 1.0
        if self.use_one_contract:
            return contracts

        if position_dollar > 0 and stop_loss_distance > 0 and self.big_point > 0:
            contracts = position_dollar / stop_loss_distance / self.big_point

        contracts = np.round(contracts, 0)  # arithmetic round
        # contracts = np.floor(contracts)  # round down

        assert contracts >= 0, "Error in contract calculation"
        return contracts

    def _breakout_strategy(self):
        df = self.data
        assert df is not None, "BacktesterFutures not initialized"

        df['Atr'] = Indicator.atr(df, self.atr_period)
        # df['Atr'] = Indicator.std(df, self.atr_period)
        df['Ema40'] = Indicator.ema(df, 40)
        df['Ema80'] = Indicator.ema(df, 80)
        df['Std'] = Indicator.std(df, 21)

        # print(" Donchian", end=" ")
        df['Up'], df['Down'] = Indicator.donchian(df, self.period, self.period)
        df['ExitUp'], df['ExitDown'] = Indicator.donchian(df, self.period // 2, self.period // 2)

        df['CloseMinusATR'] = df['Close'] - df['Atr'] * self.atr_multiplier
        df['ClosePlusATR'] = df['Close'] + df['Atr'] * self.atr_multiplier

        # print(" Moving average", end=" ")
        # df['Up'] = Indicator.sma(df, self.period) + 2 * Indicator.std(df, 21)
        # df['Down'] = Indicator.sma(df, self.period) - 2 * Indicator.std(df, 21)

        # print(" Bollinger", end=" ")
        # df['Up'], df['Down'] = Indicator.b_bands(df, 21, 3.0)

        # Mark if trade is missed because of insufficient cash
        df['MissedTrade'] = False

        # BacktesterFutures writes here the trailing stop values, so they can be plotted later
        df['trailing_stop'] = np.nan

        broker = self.broker

        for i in range(self.warm_up_period, len(df)):
            if np.isnan(self.close()):
                # I get nan for symbols like 'KOS' - is this a data error?
                return

            self.curr_i = i
            # strategy main loop - call the appropriate next()
            self.next_func()

        # close the potentially open trade at the end
        if broker.market_position != 0:
            # broker.close_position(price=df['Close'].iloc[-1])
            broker.close_position()

    def _calc_mom(self):
        if self.close(-self.momentum_lookback) != 0:
            return self.close() / self.close(-self.momentum_lookback) - 1
        return 0

    def _calc_vol(self):
        # volatility of returns
        idx = self.curr_i
        vol = self._data['Close'].iloc[idx - self.momentum_lookback: idx].pct_change().fillna(0).std()
        return vol

    def next(self):
        broker = self.broker
        if broker.update():
            self.set_value('trailing_stop', self.stop_loss)

        # enough_volume = self.volume() > MIN_VOLUME
        enough_volume = True
        # print(self.future.symbol, curr_timestamp, enough_volume)

        if self.cumulative:
            # initial account +
            closed_pnl = sum([trade.pnl * self.big_point - trade.costs for trade in broker.trades if trade.is_closed])
            curr_account = self.account + closed_pnl
            self.dollar_risk = curr_account * self.pct_risk
        delay = -self.order_execution_delay
        if broker.market_position <= 0 and self.close(delay) > self.up(delay - 1):
        # if broker.market_position <= 0 and self.ema40() > self.ema80() and self.ema40(-1) < self.ema80(-1):
            # go long
            if broker.market_position < 0:
                # close current short position before going long
                broker.close_position()

            contracts = self.calc_nr_contracts(self.dollar_risk, self.atr() * self.atr_multiplier)
            if contracts > 0:
                # go long if enough money for at least one contract
                if self.long and enough_volume:
                    self.stop_loss = self.get_value('CloseMinusATR') # @@@ self.close() - self.atr() * self.atr_multiplier
                    mom = self._calc_mom()
                    # open a new long position, contracts > 0
                    broker.open_position(position=contracts, stop_loss=self.stop_loss,
                                         margin=self.margin, momentum=mom)
                    # print(self.timestamp(), self.dollar_risk, self.close(), self.stop_loss)

            else:
                # not enough money to trade
                self.set_value('MissedTrade', True)

        if broker.market_position >= 0 and self.close(delay) < self.down(delay - 1):
        # if broker.market_position >= 0 and self.ema40() < self.ema80() and self.ema40(-1) > self.ema80(-1): # self.close() < self.down(-1):
            # go short
            if broker.market_position > 0:
                # close current long position before going short
                broker.close_position()

            contracts = self.calc_nr_contracts(self.dollar_risk, self.atr() * self.atr_multiplier)
            if contracts > 0:
                # go short if enough money for at least one contract
                if self.short and enough_volume:
                    self.stop_loss = self.get_value('ClosePlusATR') # @@@ self.close() + self.atr() * self.atr_multiplier
                    mom = - self._calc_mom()
                    # open a new short position, contracts < 0
                    broker.open_position(position=-contracts, stop_loss=self.stop_loss,
                                         margin=self.margin, momentum=mom)
            else:
                # not enough money to trade
                # df.loc[self.timestamp(), 'MissedTrade'] = True # this is the same as set_value()
                self.set_value('MissedTrade', True)

        # check for stop loss
        if self.use_stop_loss and broker.market_position != 0:
            if self.use_trailing_stop:
                # update trailing stop
                if broker.market_position > 0:
                    self.stop_loss = max(self.stop_loss, self.get_value('CloseMinusATR', -1)    # @@@ self.close(-1) - 1 * self.atr(-1) * self.atr_multiplier
                                         )  # , self.get_value('ExitDown', -1))   # virtually no effect
                elif broker.market_position < 0:
                    self.stop_loss = min(self.stop_loss, self.get_value('ClosePlusATR') # @@@ self.close(-1) + 1 * self.atr(-1) * self.atr_multiplier
                                         )  # , self.get_value('ExitUp', -1))  # virtually no effect
                broker.set_stop_loss(self.stop_loss)

            self.set_value('trailing_stop', self.stop_loss)     # for the charting only

    def next_counter_trend(self):
        """
        from A.Clenow - Trading Evolved, Counter Trend BacktesterFutures, pp 315
        Long positions are allowed if the 40 day exponential moving
        average is above the 80 day exponential moving average. If the price in a
        bull market falls back three times its standard deviation from the highest
        closing price in the past 20 days, we buy to open. If the trend turns
        bearish, we exit. If the position has been held for 20 trading days, we
        exit. Position size is volatility parity, based on standard deviation.

        implemented 29.11.2024
        set self.period = 21
        Not as good as trend following - cagr around 9-10%
        Days in trade must be set to a high value, like 200 (in original 20)
        Works better with trailing stop (the original does not have stop loss)
        NOTE: set self.up/down to Bollinger indicator
        """
        MAX_DIT = 20

        broker = self.broker
        if broker.update():
            # stop loss
            self.set_value('trailing_stop', self.stop_loss)
            return  # we get a stop loss, don't try to enter on the same day
            # pass # this strategy has no stop loss ?!?

        if self.cumulative:
            curr_account = sum([trade.pnl * self.big_point - trade.costs for trade in broker.trades if trade.is_closed]) + self.account
            self.dollar_risk = curr_account * self.pct_risk
        #
        # long side
        #
        if self.long and (broker.market_position == 0) and (self.ema40() > self.ema80()) and (self.close() < self.down(-1)):
            contracts = self.calc_nr_contracts(self.dollar_risk, self.atr() * self.atr_multiplier)
            if contracts > 0:
                # go long (only if enough money to trade)
                self.stop_loss = self.get_value('CloseMinusATR')    # @@@ self.close() - self.atr() * self.atr_multiplier
                # open a new long position
                mom = -1.0 * self._calc_mom() # negative momentum is good as we trade pullback
                broker.open_position(position=contracts, stop_loss=self.stop_loss,
                                     margin=self.margin, momentum=mom)
            else:
                self.set_value('MissedTrade', True)

        if broker.market_position > 0 and (self.ema40() < self.ema80() or broker.days_since_entry > MAX_DIT):
            # close long position
            broker.close_position()

        # check for stop loss
        if broker.market_position > 0 and self.use_trailing_stop:
            self.stop_loss = max(self.stop_loss, self.get_value('CloseMinusATR', -1))  # @@@ self.close(-1) - 1 * self.atr(-1) * self.atr_multiplier)
            broker.set_stop_loss(self.stop_loss)
            self.set_value('trailing_stop', self.stop_loss)     # for the charting only
        #
        # short side
        #
        if self.short and (broker.market_position == 0) and (self.ema40() < self.ema80()) and (self.close() > self.up(-1)):
            contracts = self.calc_nr_contracts(self.dollar_risk, self.atr() * self.atr_multiplier)
            if contracts > 0:
                # go long (only if enough money to trade)
                self.stop_loss = self.get_value('ClosePlusATR')     # @@@ self.close() + self.atr() * self.atr_multiplier
                # open a new long position
                mom = 1.0 * self._calc_mom() # negative momentum is good as we trade pullback
                broker.open_position(position=-contracts, stop_loss=self.stop_loss,
                                     margin=self.margin, momentum=mom)
            else:
                self.set_value('MissedTrade', True)

        if broker.market_position < 0 and (self.ema40() > self.ema80() or broker.days_since_entry > MAX_DIT):
            # close short position
            broker.close_position()

        # check for stop loss
        if broker.market_position < 0 and self.use_trailing_stop:
            self.stop_loss = min(self.stop_loss, self.get_value('ClosePlusATR', -1)) # @@@ self.close(-1) + 1 * self.atr(-1) * self.atr_multiplier)
            broker.set_stop_loss(self.stop_loss)
            self.set_value('trailing_stop', self.stop_loss)     # for the charting only

    def next_buy_and_hold(self):
        broker = self.broker

        broker.update()

        if broker.market_position == 0:
            contracts = self.calc_nr_contracts(self.dollar_risk, self.atr() * self.atr_multiplier)
            if contracts > 0:
                broker.open_position(position=contracts, margin=self.margin)

    def next_buy_and_hold_monthly(self):
        # buy and hold with monthly re-balancing
        broker = self.broker

        broker.update(check_stop_loss=False)
        contracts = self.calc_nr_contracts(self.dollar_risk, self.atr() * self.atr_multiplier)

        if broker.market_position == 0:
            # only once, before the very first position
            if contracts > 0:
                if self.short:
                    self.stop_loss = self.close() + self.atr() * self.atr_multiplier
                    contracts = -contracts
                else:
                    self.stop_loss = self.get_value('CloseMinusATR')    # @@@ self.close() - self.atr() * self.atr_multiplier
                broker.open_position(position=contracts, stop_loss=self.stop_loss, margin=self.margin)

        elif self.timestamp().to_pydatetime().day < self.timestamp(-1).to_pydatetime().day:
            # first day of next month - close position and re-open immediately
            broker.close_position()
            if contracts > 0:
                if self.short:
                    self.stop_loss = self.get_value('ClosePlusATR')     # @@@ self.close() + self.atr() * self.atr_multiplier
                    contracts = -contracts
                else:
                    self.stop_loss = self.get_value('CloseMinusATR')    # @@@ self.close() - self.atr() * self.atr_multiplier
                broker.open_position(position=contracts, stop_loss=self.stop_loss, margin=self.margin)

    def next_buy_one_contract(self):
        # buy (long only) one contract if we have enough money, else sell the contract
        broker = self.broker

        broker.update()

        contracts = self.calc_nr_contracts(self.dollar_risk, self.atr() * self.atr_multiplier)
        if contracts > 0:
            if broker.market_position == 0:
                broker.open_position(position=1, stop_loss=np.nan, margin=self.margin)
        else:
            if broker.market_position > 0:
                broker.close_position()

    def next_random(self):
        """
        Enter random
        """
        broker = self.broker
        if broker.update():
            self.set_value('trailing_stop', self.stop_loss)

        if broker.market_position == 0:
            # flat - take a random wait until entering a new position
            if self.random_wait == -1:
                # start a new waiting
                self.random_wait = random.randint(1, 63)
                return
            if self.random_wait > 0:
                # still waiting
                self.random_wait -= 1
                return
            if self.random_wait == 0:
                # time to open a new position
                self.random_wait = -1
                pass

        # enough_volume = self.volume() > MIN_VOLUME
        enough_volume = True
        # print(self.future.symbol, curr_timestamp, enough_volume)

        if self.cumulative:
            # initial account +
            closed_pnl = sum([trade.pnl * self.big_point - trade.costs for trade in broker.trades if trade.is_closed]) # and not trade.deleted])
            curr_account = self.account + closed_pnl
            self.dollar_risk = curr_account * self.pct_risk

        if broker.market_position <= 0 and random.randint(0, 1) == 1:
            contracts = self.calc_nr_contracts(self.dollar_risk, self.atr() * self.atr_multiplier)
            if contracts > 0:
                # go long (only if enough money to trade)
                if broker.market_position < 0:
                    # close previous short position
                    broker.close_position()
                if self.long and enough_volume:
                    self.stop_loss = self.get_value('CloseMinusATR')    # @@@ self.close() - self.atr() * self.atr_multiplier
                    # open a new long position
                    mom = self._calc_mom()
                    broker.open_position(position=contracts, stop_loss=self.stop_loss,
                                         margin=self.margin, momentum=mom)
                    # print(self.timestamp(), self.dollar_risk, self.close(), self.stop_loss)

            else:
                self.set_value('MissedTrade', True)

        elif broker.market_position >= 0 and random.randint(0, 1) == 0:
            # go short
            contracts = self.calc_nr_contracts(self.dollar_risk, self.atr() * self.atr_multiplier)
            if contracts > 0:
                # go short (only if enough money to trade)
                if broker.market_position > 0:
                    # close previous short position
                    broker.close_position()
                if self.short and enough_volume:
                    # open a new short position (contracts < 0)
                    self.stop_loss = self.close() + self.atr() * self.atr_multiplier
                    mom = - self._calc_mom()
                    broker.open_position(position=-contracts, stop_loss=self.stop_loss,
                                         margin=self.margin, momentum=mom)
            else:
                # df.loc[self.timestamp(), 'MissedTrade'] = True  # not enough money to trade
                self.set_value('MissedTrade', True)
        else:
            # no trades this bar, check for stop loss
            if self.use_stop_loss and broker.market_position != 0:
                if self.use_trailing_stop:
                    # update trailing stop
                    if broker.market_position > 0:
                        self.stop_loss = max(self.stop_loss, self.get_value('CloseMinusATR', -1))    # @@@ self.close(-1) - 1 * self.atr(-1) * self.atr_multiplier
                                            #, self.get_value('ExitDown', -1))   # improves slightly
                    elif broker.market_position < 0:
                        self.stop_loss = min(self.stop_loss, self.get_value('ClosePlusATR'))     # @@@ self.close(-1) + 1 * self.atr(-1) * self.atr_multiplier
                                            # , self.get_value('ExitUp', -1))  # improves slightly
                    broker.set_stop_loss(self.stop_loss)

                self.set_value('trailing_stop', self.stop_loss)     # for the charting only

    def calc_performance(self):
        assert self.data is not None, "run strategy first"

        df = self.data

        # add df columns for calculations and plotting

        # CloseStrategy is modified during the performance calculation to show the real exit price by stop loss
        # Later it is also modified to include trading costs
        df['CloseStrategy'] = df['Close']
        df['Signal'] = np.nan  # 1 for buy, -1 for sell, 0 for hold
        df['StopLoss'] = np.nan  # closing price at pos. exit
        df['Position'] = 0.0
        df['Margin'] = 0.0
        df['Contracts'] = 0.0

        for trade in self.trades:
            # print(trade)
            if len(trade.trade_dates) < 2:
                # this trade is too short (probably just opened on the last bar)
                trade.deleted = True
                continue

            df.loc[trade.entry_date, 'Signal'] = 1 if trade.position > 0 else -1
            df.loc[trade.exit_date, 'CloseStrategy'] = trade.exit_price     # correct close price if we got stop loss
            df.loc[trade.exit_date, 'StopLoss'] = trade.exit_price if trade.is_stop_loss else np.nan

            # trade.trade_dates[1] - we take the day after the entry (entry on close)
            try:
                df.loc[trade.trade_dates[1]:trade.exit_date, 'Position'] = trade.position
                df.loc[trade.trade_dates[1]:trade.exit_date, 'Margin'] = trade.margin
                df.loc[trade.trade_dates[1]:trade.exit_date, 'Contracts'] = trade.position
            except:
                pass
        # self.data_access.close()  # remove: test only

        self._calc_trades()     # needs 'CloseStrategy'

        # percent returns
        # df['Returns'] = df['Close'] / df['Close'].shift(1) - 1
        # log returns
        # df['Returns'] = np.log(df['Close']) - np.log(df['Close'].shift())

        big_point = self.big_point
        # dollar returns - this avoids the problem with negative Closes
        # and big percent values when Close is close to 0
        df['Returns'] = (df['Close'] - df['Close'].shift()).fillna(0) * big_point  # returns long buy&hold

        # returns long buy&hold
        strat_returns = (df['CloseStrategy'] - df['CloseStrategy'].shift()).fillna(0) * big_point

        # returns of strategy - we don't shift() position any longer!!!
        df['Strat_Returns'] = df['Position'] * strat_returns
        df['Strat_Returns_Long'] = np.where(df['Position'] > 0, df['Strat_Returns'], 0.0)
        df['Strat_Returns_Short'] = np.where(df['Position'] < 0, df['Strat_Returns'], 0.0)

        # just cumsum() as we work in currency (USD, EUR,...), not in % return
        df['Buy&Hold_Pnl'] = (df['Returns']).cumsum()
        df['Strat_Pnl'] = (df['Strat_Returns']).cumsum()
        df['Strat_Pnl_Long'] = (df['Strat_Returns_Long']).cumsum()
        df['Strat_Pnl_Short'] = (df['Strat_Returns_Short']).cumsum()

        self.final_pnl = df['Strat_Pnl'].iloc[-1]

        # calculate yearly performance, no compounding
        total_days = (df.index[-1] - df.index[0]).days  # total calendar years
        self.yearly_ret = self.final_pnl / (total_days / 365.25) / self.account
        std = df['Strat_Returns'].std()
        if std > 0:
            self.sharpe = df['Strat_Returns'].mean() / df['Strat_Returns'].std() * np.sqrt(252)

        # print(tabulate(df, headers='keys', tablefmt='psql'))

    def _calc_trades(self):
        # calculate single trades
        df = self._data

        # add trading costs
        rolls = 0
        self.total_costs = 0.0
        if self.cost_contract > 0 or self.slippage_ticks > 0:
            for trade in self.trades:
                left, right = trade.entry_date, trade.exit_date
                # full turn costs for one contract in $
                full_turn_costs = 2 * abs(trade.position * (self.cost_contract +
                                                            self.tick_size * self.slippage_ticks * self.big_point))

                trade_costs = (trade.rolls + 1) * full_turn_costs
                # trade_costs = full_turn_costs

                # price_shift > 0 for long; < 0 for short
                # price_shift contains only 1/2 of the overall costs, because we add it twice, on entry and on exit
                # Note: one roll contains two trades. Therefore, here we use * trade.rolls, meaning we consider
                # only one of these two trades
                price_shift = (trade.market_position * (trade.rolls + 1) *
                               (self.cost_contract / self.big_point + self.tick_size * self.slippage_ticks))
                # print(trade, f'{price_shift:.6f}')
                df.loc[left, 'CloseStrategy'] += price_shift  # 1/2 of costs on entry
                df.loc[right, 'CloseStrategy'] -= price_shift  # 1/2 of costs on exit
                trade.costs = trade_costs
                self.total_costs += trade_costs
                rolls += trade.rolls

        self.nr_rolls = rolls

        trades = self.trades
        # print("Total rolls=", rolls)
        # for trade in trades: print(trade)
        self.nr_trades = len(trades)
        if self.nr_trades > 0:
            self.avg_trade = sum([trade.pnl * self.big_point - trade.costs for trade in trades]) / self.nr_trades
            self.avg_dit = sum([trade.dit for trade in trades]) / self.nr_trades
            self.avg_contracts = sum([abs(trade.position) for trade in trades]) / self.nr_trades
            self.avg_position_size_dollar = sum(
                [abs(trade.position * trade.entry_price) for trade in trades]) / self.nr_trades * self.big_point
            self.avg_margin = sum([trade.margin for trade in trades]) / self.nr_trades

        self.nr_missed_trades = len(df[df['MissedTrade']])

    def plot_performance(self, front):
        sns.set_style("whitegrid")
        fig, ax = plt.subplots(4, figsize=(12, 11), sharex='all')

        df = self._data
        # big_point = self.big_point

        future_name = (
              f'{self.future}\n'
              f'Atr.mul: {self.atr_multiplier}, Risk : ${self.dollar_risk:,.0f}, Stop orders: {self.use_stop_orders}, '
              f'Cumulative: {self.cumulative}, Pct_risk: {self.pct_risk}, Front: {front}\n'
              f'Nr.trades: {self.nr_trades}, Missed: {self.nr_missed_trades}, Rolls: {self.nr_rolls},'
              f' Avg.trade: ${self.avg_trade:,.0f}, Avg.contracts: {self.avg_contracts:,.2f}, Avg.DIT: {self.avg_dit:.0f},'
              f' Avg.pos.size: ${self.avg_position_size_dollar:,.0f},'
              f' Avg.margin: ${self.avg_margin:,.0f}\n'
              f' Pnl, net: ${self.final_pnl:,.0f}, Costs: ${self.total_costs:,.0f}, '
              f' Yearly: {self.yearly_ret * 100:.2f} %, '
              f' Sharpe: {self.sharpe:.2f}'
        )

        plt.suptitle(future_name)

        #
        #   Upper chart
        #
        # ax[0].plot(df['Close'], label='Close', color='#8c564b', lw=1.5)
        # ax[0].plot(df['High'], label='High', color='green', lw=0.5)
        # ax[0].plot(df['Low'], label='Low', color='red', lw=0.5)
        #
        ax[0].plot(df['Up'], label='Up', color='blue', lw=1, alpha=0.5)
        ax[0].plot(df['Down'], label='Down', color='red', lw=1, alpha=0.5)
        ax[0].plot(df['trailing_stop'], label='Trailing Stop', linestyle=':', color='magenta')

        ax[0].plot(df['Close'], label='Close', color='#8c564b', lw=1.5)
        # ax[0].plot(df['Ema40'], label='Ema40', color='green', lw=0.5)
        # ax[0].plot(df['Ema80'], label='Ema80', color='red', lw=0.5)

        # ax[0].plot(df['Up'], label='Up', color='blue', lw=1, alpha=0.5)
        # ax[0].plot(df['Up'] - df['Std'] * 3, label='Up-Std', color='blue', lw=1, alpha=0.5)



        # ax[0].plot(df['ExitUp'], label='ExitUp', linestyle='--', color='green')
        # ax[0].plot(df['ExitDown'], label='ExitDown', linestyle='--', color='red')

        # Filter buy and sell signals
        buy_signals = df[df['Signal'] == 1]
        sell_signals = df[df['Signal'] == -1]

        # Graph buy and sell signals
        ax[0].scatter(buy_signals.index, buy_signals['Close'], marker='^', color='green', label='Buy signal', alpha=1)
        ax[0].scatter(sell_signals.index, sell_signals['Close'], marker='v', color='red', label='Sell signal', alpha=1)
        ax[0].scatter(df.index, df['StopLoss'], marker='x', color='magenta', label='Stop loss', alpha=1)

        # Mark missed trades
        missed_trades = df[df['MissedTrade']]
        ax[0].scatter(missed_trades.index, missed_trades['Close'], marker='o', color='Orange', label='Missed trade',
                      alpha=1, zorder=0, s=10)

        # plot color stripes under long and short trades
        for trade in self.trades:
            left, right = trade.entry_date, trade.exit_date
            color = 'green' if trade.market_position > 0 else 'red'
            ax[0].axvspan(left, right, color=color, alpha=0.1, lw=0)

        # ax[0].set_title(f'Signals')
        ax[0].set_xlabel('Date')
        ax[0].set_ylabel('Signals')
        ax[0].legend(loc='upper left')

        #
        #   2nd chart
        #
        df[['Buy&Hold_Pnl', 'Strat_Pnl']].plot(ax=ax[1], lw=1.5)
        df[['Strat_Pnl_Long', 'Strat_Pnl_Short']].plot(ax=ax[1], lw=0.5, alpha=0.7)

        # ax[1].set_title(f'Performance USD')
        ax[1].set_xlabel('Date')
        ax[1].set_ylabel(f'PnL, {self.future.currency}')
        ax[1].get_yaxis().set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
        ax[1].legend(loc='upper left')

        #
        # 3rd chart
        #
        df['Margin'].plot(ax=ax[2], lw=1)

        # ax[2].set_title(f'Margin')
        ax[2].set_xlabel('Date')
        ax[2].set_ylabel(f'Margin, {self.future.currency}')
        ax[2].get_yaxis().set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
        # ax[2].legend(loc='upper left')

        # plt.show()

        #
        # 4th chart
        #
        df['Contracts'].plot(ax=ax[3], lw=1)

        # ax[3].set_title(f'Nr. Contracts')
        ax[3].set_xlabel('Date')
        ax[3].set_ylabel('Nr. Contracts')
        # ax[3].legend(loc='upper left')

        plt.show()

    def plot_atr(self):
        # this function is for test only
        df = self.data
        df['Atr'] = Indicator.atr(df, 14)

        # print(tabulate(df, headers='keys', tablefmt='psql'))

        # https://stackoverflow.com/questions/56861966/trailing-stop-loss-on-pandas-dataframe
        pct_change = df['Close'] / df['Close'].shift() - 1
        df['SL'] = df['Close'] - df['Close'].cummax() * pct_change

        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(df['Atr'], 'b-', label='ATR', lw=0.5)
        ax2.plot(df['Close'], 'g-', label='Close', lw=1)
        ax1.set_xlabel('Date')
        ax1.set_ylabel('ATR', color='b')
        ax2.set_ylabel('Close', color='g')
        ax1.legend(loc=0)
        ax2.legend(loc=1)
        plt.legend()
        plt.grid()
        plt.show()

    def plot_donchian(self):
        df = self.data

        df['Up'], df['Down'] = Indicator.donchian(df, self.period, self.period)
        df['Up1'], df['Down1'] = Indicator.b_bands(df, self.period, 2.0)
        df['Up2'], df['Down2'] = Indicator.keltner(df, self.period, self.period, self.atr_multiplier)

        df[['Close', 'Up', 'Down', 'Up1', 'Down1', 'Up2', 'Down2']].plot(lw=1)

        buy_signals = df[df['Close'] > df[['Up', 'Up1', 'Up2']].min(axis=1).shift()]
        sell_signals = df[df['Close'] < df[['Down', 'Down1', 'Down2']].max(axis=1).shift()]

        # Graph buy and sell signals
        plt.scatter(buy_signals.index, buy_signals['Close'], marker='^', color='green', label='Buy signal', alpha=1)
        plt.scatter(sell_signals.index, sell_signals['Close'], marker='v', color='red', label='Sell signal', alpha=1)
        plt.fill_between(df.index, df["Up"], df["Down"], alpha=0.05, color='black')
        plt.fill_between(df.index, df["Up1"], df["Down1"], alpha=0.05, color='black')
        plt.fill_between(df.index, df["Up2"], df["Down2"], alpha=0.05, color='black')
        plt.show()

    @classmethod
    def get_data(cls, data_access: DataAccess, future: Future, front) -> pd.DataFrame:
        return data_access.continuous_contract_adjusted(future.symbol, front=front)


def _main():

    front = 1
    symbol = 'CL'
    long = True
    short = True

    # Future.patch_micro_futures(False)

    future = Future.get_future_norgate(symbol, use_micro=False)
    # futures = Future.all_futures_norgate(use_micro=True)

    # read data and close db connection, we don't need it afterward
    with duckdb.connect(DUCK_DB, read_only=True) as connection:
        data_access = DataAccess(connection, '1022-01-01')
        data = LoosePants.get_data(data_access, future, front=front)

    with Timer():

        # ZS - soybeans ZC - corn
        # dollar_risk < 0 means: trade always with 1 contract
        # for VX: nice results with period 21..41, atr_multiplier 2..3 (21/3 is good)
        lp = LoosePants(
                        period=12 * 21, # 6*21,
                        atr_multiplier=5,
                        use_stop_loss=True,
                        use_trailing_stop=True,
                        use_stop_orders=True,
                        use_one_contract=True,
                        long=long,
                        short=short,
                        dollar_risk=10_000,
                        account=1_000_000,
                        cost_contract=1.0,  # USD to trade one contract, single side
                        slippage_ticks=2.0,  # single side slippage, use TickSize to convert to USD
                        cumulative=True,
                        pct_risk=0.02,  # pct_risk is used if cumulative=True
                        )

        # next, next_counter_trend, next_buy_and_hold, next_buy_and_hold_monthly, next_random, next_buy_one_contract
        lp.set_next(lp.next)
        lp.run(future=future, data=data)

    lp.calc_performance()
    lp.plot_performance(front)
    pass
    # lp.plot_atr()
    # lp.plot_donchian()


if __name__ == "__main__":
    _main()
