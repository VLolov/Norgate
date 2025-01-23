import datetime
import typing
from dataclasses import dataclass

import numpy as np

from Futures.Backtester.BacktesterFutures import *
from Futures.TrendFollowing.Indicator import Indicator


class StrategyLoosePants(Strategy):
    @dataclass
    class Config:
        period: int = 12 * 21
        atr_period: int = 14
        atr_multiplier: float = 5.0
        use_stop_loss: bool = True
        use_trailing_stop: bool = True
        use_stop_orders: bool = True
        short: bool = True
        long: bool = True
        use_one_contract: bool = True
        dollar_risk: float = 10_000  # if dollar_risk < 0: trade with 1 contract
        account: float = 1_000_000
        cost_contract: float = 1.0  # USD to trade one contract, single side
        slippage_ticks: float = 2  # single side slippage, use TickSize to convert to USD
        cumulative: bool = False  # if cumulative=True, position size is calculated based on pct_risk and account size
        # if cumulative=False, position size is calculated from dollar_risk
        pct_risk: float = 0.02
        order_execution_delay: int = 0
        close_last_trading_day: bool = True

    def __init__(self, name='LoosePants', config=None):
        super().__init__(name)
        if config is None:
            self.config: StrategyLoosePants.Config = StrategyLoosePants.Config()

        self.momentum_lookback: int = 21
        # self.warm_up_period: int = 0
        # self.broker: typing.Optional[Broker] = None
        self.close_last_trading_day = True

    def calc_indicators(self):
        for instrument in self.instruments:
            future = typing.cast(Future, instrument)
            df = future.data
            cfg = self.config
            df['trailing_stop'] = 0.0
            df['Atr'] = Indicator.atr(df, cfg.atr_period)
            # df['Atr'] = Indicator.std(df, self.atr_period)
            df['Ema40'] = Indicator.ema(df, 40)
            df['Ema80'] = Indicator.ema(df, 80)
            df['Std'] = Indicator.std(df, 21)

            # print(" Donchian", end=" ")
            df['Up'], df['Down'] = Indicator.donchian(df, cfg.period, cfg.period)
            df['ExitUp'], df['ExitDown'] = Indicator.donchian(df, cfg.period // 2, cfg.period // 2)

            df['CloseMinusATR'] = df['Close'] - df['Atr'] * cfg.atr_multiplier
            df['ClosePlusATR'] = df['Close'] + df['Atr'] * cfg.atr_multiplier

            # print(" Moving average", end=" ")
            # df['Up'] = Indicator.sma(df, self.period) + 2 * Indicator.std(df, 21)
            # df['Down'] = Indicator.sma(df, self.period) - 2 * Indicator.std(df, 21)

            # print(" Bollinger", end=" ")
            # df['Up'], df['Down'] = Indicator.b_bands(df, 21, 3.0)

            # Mark if trade is missed because of insufficient cash
            df['MissedTrade'] = False

            # BacktesterFutures writes here the trailing stop values, so they can be plotted later
            df['trailing_stop'] = np.nan

    def _calc_mom(self, instrument, idx):
        return self.close(instrument, idx) / self.close(instrument, idx - self.momentum_lookback) - 1

    def _calc_vol(self, instrument, idx):
        # volatility of returns
        vol = instrument.data['Close'].iloc[idx - self.momentum_lookback: idx].pct_change().fillna(0).std()
        return vol

    def init(self):
        self.log.debug(f"init({self.idx}, {self.dt})")

        # modify parameters of Strategy class
        self.cost_contract = self.config.cost_contract
        self.slippage_ticks = self.config.slippage_ticks
        self.close_last_trading_day = self.config.close_last_trading_day

        broker = typing.cast(Broker, self.group.broker)
        broker.setup(initial_capital=self.config.account,
                     use_stop_loss=self.config.use_stop_loss,
                     use_stop_orders=self.config.use_stop_orders)

        # self.warm_up_period =max(2, self.period, self.atr_period, self.momentum_lookback)
        self.warm_up_period = max(2, self.config.period, self.config.atr_period, self.momentum_lookback)

        self.set_tradable_range_instruments()
        self.calc_indicators()

    @staticmethod
    def get_close(instrument, idx):
        # return instrument.data.iloc[idx]['Open']
        return instrument.data['Close'].iloc[idx]        # this is much faster (3-4 times)

    @staticmethod
    def get_close_numpy(instrument, idx):
        return instrument.data_numpy[idx, Future.CLOSE]

    def calc_nr_contracts(self, instrument: Future, position_dollar, stop_loss_distance):
        contracts = 1.0
        if self.config.use_one_contract:
            return contracts

        if position_dollar > 0 and stop_loss_distance > 0 and instrument.metadata.big_point > 0:
            contracts = position_dollar / stop_loss_distance / instrument.metadata.big_point

        contracts = np.round(contracts, 0)  # arithmetic round
        # contracts = np.floor(contracts)  # round down

        assert contracts >= 0, "Error in contract calculation"
        return contracts

    def next(self):
        # all this code is for testing only
        idx = self.idx
        dt = self.dt

        if dt == datetime.date(1999, 7, 1):
            jj = 1

        # self.log.debug(f"next({idx}, {time})")
        
        broker = typing.cast(Broker, self.group.broker)

        for instrument in self.instruments:
            instrument = typing.cast(Future, instrument)
            if not self.check_tradable_range(instrument, idx):
                # self.log.debug(f"{idx} {time} {future}")
                continue

            if broker.update(self, instrument):
                # stop loss occurred, don't try to enter on the same bar
                continue

            # for i in range(10):
            #     symbol = future.symbol
            #     open = self.open(future, idx)
            #     high = self.high(future, idx)
            #     low = self.low(future, idx)
            #     close = self.close(future, idx)

            # for i in range(10):
            #     symbol = future.symbol
            #     # open1, high1, low1, close1 = self.ohlc(future, idx)
            #     # close = self.get_close(future, idx)
            #     # close = self.get_close(future, idx)
            #     # close = self.get_close(future, idx)
            #     # close = self.get_close(future, idx)
            #     # assert self.get_close(future, idx) == self.get_close_numpy(future, idx)
            #     close1 = self.get_close_numpy(future, idx)
            #     close1 = self.get_close_numpy(future, idx)
            #     close1 = self.get_close_numpy(future, idx)
            #     close1 = self.get_close_numpy(future, idx)

            # buy and hold
            # if broker.market_position(self, future) == 0:
            #    broker.open_position(self, future, margin=future.metadata.margin)

            # enough_volume = self.volume(instrument, idx) > MIN_VOLUME
            enough_volume = True

            # if self.cumulative:
            #     # initial account +
            #     closed_pnl = sum([trade.pnl * self.big_point - trade.costs for trade in broker.trades if trade.is_closed])
            #     curr_account = self.account + closed_pnl
            #     self.dollar_risk = curr_account * self.pct_risk

            delay = -self.config.order_execution_delay

            if (broker.market_position(self, instrument) <= 0
                    and self.close(instrument, idx - delay) > self.up(instrument, idx - delay - 1)):
                # go long
                if broker.market_position(self, instrument) < 0:
                    # close current short position before going long
                    broker.close_position(self, instrument)

                contracts = self.calc_nr_contracts(instrument, self.config.dollar_risk, self.atr(instrument, idx) * self.config.atr_multiplier)
                if contracts > 0:
                    # go long if enough money for at least one contract
                    if self.config.long and enough_volume:
                        stop_loss = self.close_minus_atr(instrument, idx)
                        mom = self._calc_mom(instrument, idx)
                        # open a new long position, contracts > 0
                        broker.open_position(self, instrument,
                                             position=contracts, stop_loss=stop_loss,
                                             margin=instrument.metadata.margin, momentum=mom)

                else:
                    # not enough money to trade
                    self.set_value(instrument, 'MissedTrade', True, idx)

            if broker.market_position(self, instrument) >= 0 and self.close(instrument, idx - delay) < self.down(instrument, idx - delay - 1):
                # go short
                if broker.market_position(self, instrument) > 0:
                    # close current long position before going short
                    broker.close_position(self, instrument)

                contracts = self.calc_nr_contracts(instrument,
                                                   self.config.dollar_risk,
                                                   self.atr(instrument, idx) * self.config.atr_multiplier)
                if contracts > 0:
                    # go short if enough money for at least one contract
                    if self.config.short and enough_volume:
                        stop_loss = self.close_plus_atr(instrument, idx)
                        mom = - self._calc_mom(instrument, idx)
                        # open a new short position, contracts < 0
                        broker.open_position(self, instrument,
                                             position=-contracts,
                                             stop_loss=stop_loss,
                                             margin=instrument.metadata.margin,
                                             momentum=mom)
                else:
                    # not enough money to trade
                    # df.loc[self.timestamp(), 'MissedTrade'] = True # this is the same as set_value()
                    self.set_value(instrument,'MissedTrade', True, idx)

            # check for stop loss
            if self.config.use_stop_loss and broker.market_position(self, instrument) != 0:
                if self.config.use_trailing_stop:
                    # update trailing stop
                    stop_loss = broker.get_stop_loss(self, instrument)
                    if broker.market_position(self, instrument) > 0:
                        stop_loss = max(stop_loss, self.close_minus_atr(instrument, idx))
                    elif broker.market_position(self, instrument) < 0:
                        stop_loss = min(stop_loss, self.close_plus_atr(instrument, idx))
                    broker.set_stop_loss(self, instrument, stop_loss)

                self.set_value(instrument, 'trailing_stop', broker.get_stop_loss(self, instrument), idx)     # for the charting only



    @staticmethod
    def atr(instrument, idx) -> float:
        return instrument.data['Atr'].iloc[idx]

    @staticmethod
    def up(instrument, idx) -> float:
        return instrument.data['Up'].iloc[idx]

    @staticmethod
    def down(instrument, idx) -> float:
        return instrument.data['Down'].iloc[idx]

    @staticmethod
    def close_minus_atr(instrument, idx) -> float:
        return instrument.data['CloseMinusATR'].iloc[idx]

    @staticmethod
    def close_plus_atr(instrument, idx) -> float:
        return instrument.data['ClosePlusATR'].iloc[idx]

    def last(self):
        self.log.debug(f"last({self.idx}, {self.dt})")

        if self.close_last_trading_day:
            self.close_all_trades()
        broker = typing.cast(Broker, self.group.broker)
        self.log.debug(f"Number of trades: {len(broker.trades)}")
        # print_trades(self.broker.trades)
