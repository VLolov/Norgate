import datetime
import typing
from copy import deepcopy
from typing import Optional, Dict, List
from dataclasses import dataclass, field

import numpy as np

from Futures.Backtester.BacktesterFutures import *
from Futures.TrendFollowing.Indicator import Indicator


skip_short = ['Equity', 'Metal', 'Fixed Income', 'Grain', 'Soft']
skip_short = []


class StrategyLoosePants(Strategy):
    @dataclass
    class LoosePantsConfig(Config):
        portfolio_dollar: float = 1_000_000     # 0: get portfolio from Portfolio
        risk_position: float = 0.02  # % of portfolio, if RISK_POSITION < 0: trade with +/- 1 contract
        risk_all_positions: float = 0.2  # % of portfolio; 0=don't check
        max_margin: float = 0.4  # % of portfolio; 0=don't check margin

        sectors: List[str] = field(default_factory=lambda: [])   # filter by sectors
        max_positions_per_sector: int = 0

        period: int = 12 * 21
        atr_period: int = 14
        atr_multiplier: float = 5.0
        use_stop_loss: bool = True
        use_trailing_stop: bool = True
        use_stop_orders: bool = True
        long: bool = True
        short: bool = True
        use_one_contract: bool = False
        cost_contract: float = 0    ############ 2  # USD to trade one contract, single side
        slippage_ticks: float = 0   ############# 2  # single side slippage, use TickSize to convert to USD
        cumulative: bool = False  # if cumulative=True, position size is calculated based on pct_risk and account size
        order_execution_delay: int = 0
        close_last_trading_day: bool = True

        skip_short: List[str] = field(default_factory=lambda: skip_short)

    def __init__(self, name='LoosePants', config=None):
        super().__init__(name)
        if config is None:
            self.set_config(self.__class__.LoosePantsConfig())

        self.momentum_lookback: int = 21
        # self.warm_up_period: int = 0
        # self.broker: typing.Optional[Broker] = None
        self.curr_account = 0.0
        self.log.debug(f"Strategy {name} created")

    def calc_indicators(self):
        for instrument in self.instruments:
            future = typing.cast(Future, instrument)
            df = future.data
            cfg = self.get_config()
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
        divisor = self.close(instrument, idx - self.momentum_lookback) != 0
        if divisor == 0:
            return 0.0
        return self.close(instrument, idx) / divisor - 1

    def _calc_vol(self, instrument, idx):
        # volatility of returns
        vol = instrument.data['Close'].iloc[idx - self.momentum_lookback: idx].pct_change().fillna(0).std()
        return vol

    def init(self):
        self.log.debug(f"init(), dt:{self.dt})")
        super().init()

        # modify parameters of Strategy class
        cfg = typing.cast(self.LoosePantsConfig, self.get_config())
        self.cost_contract = cfg.cost_contract
        self.slippage_ticks = cfg.slippage_ticks

        # get initial_capital from portfolio
        if cfg.portfolio_dollar == 0:
            cfg.portfolio_dollar = self.group.backtester.portfolio.initial_capital
        self.curr_account = cfg.portfolio_dollar

        broker = typing.cast(Broker, self.group.broker)
        broker.setup(initial_capital=cfg.portfolio_dollar,
                     use_stop_loss=cfg.use_stop_loss,
                     use_stop_orders=cfg.use_stop_orders)

        # self.warm_up_period =max(2, self.period, self.atr_period, self.momentum_lookback)
        self.warm_up_period = max(2, cfg.period, cfg.atr_period, self.momentum_lookback)

        self.set_tradable_range_instruments()
        self.calc_indicators()

    # @staticmethod
    # def get_close(instrument, idx):
    #     # return instrument.data.iloc[idx]['Open']
    #     return instrument.data['Close'].iloc[idx]        # this is much faster (3-4 times)
    #
    # @staticmethod
    # def get_close_numpy(instrument, idx):
    #     return instrument.data_numpy[idx, Future.CLOSE]

    def calc_nr_contracts(self, instrument: Future, position_dollar, stop_loss_distance):
        contracts = 1.0
        cfg = self.get_config()
        if cfg.use_one_contract:
            return contracts

        if position_dollar > 0 and stop_loss_distance > 0 and instrument.metadata.big_point > 0:
            contracts = position_dollar / stop_loss_distance / instrument.metadata.big_point

        contracts = np.round(contracts, 0)  # arithmetic round
        # contracts = np.floor(contracts)  # round down

        assert contracts >= 0, "Error in contract calculation"
        return contracts

    @dataclass
    class TradeCandidate:
        instrument: Future
        direction: int
        momentum: float
        pos_size: float = 0
        margin: float = 0
        deleted: bool = False

    def next(self):
        # all this code is for testing only
        idx = self.idx
        dt = self.dt

        if dt == datetime.date(1999, 7, 1):
            jj = 1

        # self.log.debug(f"next({idx}, {time})")

        trade_candidates = []

        broker = typing.cast(Broker, self.group.broker)
        cfg = typing.cast(self.LoosePantsConfig, self.get_config())

        for instrument in self.instruments:
            instrument = typing.cast(Future, instrument)
            if not self.check_tradable_range(instrument, idx):
                # self.log.debug(f"{idx} {time} {future}")
                continue

            if cfg.sectors and instrument.metadata.sector not in cfg.sectors:
                continue

            if broker.update(self, instrument):
                # stop loss occurred, don't try to enter on the same bar
                continue

            # enough_volume = self.volume(instrument, idx) > MIN_VOLUME
            enough_volume = True

            # if self.cumulative:
            #     # initial account +
            #     closed_pnl = sum([trade.pnl * self.big_point - trade.costs for trade in broker.trades if trade.is_closed])
            #     curr_account = self.account + closed_pnl
            #     self.dollar_risk = curr_account * self.pct_risk
            delay = -cfg.order_execution_delay

            mp = broker.market_position(self, instrument)

            if mp <= 0 and self.close(instrument, idx - delay) > self.up(instrument, idx - delay - 1):
                    # and self.close(instrument, idx - delay - 1) < self.up(instrument, idx - delay - 2):
                # close current short position before going long
                broker.close_position(self, instrument)
                if cfg.long:
                    trade_candidates.append(self.TradeCandidate(instrument=instrument, direction=1,
                                                                momentum=self._calc_mom(instrument, idx)))

            if mp >= 0 and self.close(instrument, idx - delay) < self.down(instrument, idx - delay - 1):
                    # and self.close(instrument, idx - delay - 1) > self.down(instrument, idx - delay - 2):
                # go short
                # close current long position before going short
                broker.close_position(self, instrument)
                if cfg.short and instrument.metadata.sector not in cfg.skip_short:
                    # note: here we negate momentum, as we are going to rank by momentum
                    trade_candidates.append(self.TradeCandidate(instrument=instrument, direction=-1,
                                                                momentum=-self._calc_mom(instrument, idx)))

        self.process_trade_candidates(trade_candidates)

        for instrument in self.instruments:
            instrument = typing.cast(Future, instrument)
            if not self.check_tradable_range(instrument, idx):
                # self.log.debug(f"{idx} {time} {future}")
                continue

            # check for stop loss
            mp = broker.market_position(self, instrument)
            if cfg.use_stop_loss and mp != 0:
                if cfg.use_trailing_stop:
                    # update trailing stop
                    stop_loss = broker.get_stop_loss(self, instrument)
                    if mp > 0:
                        stop_loss = max(stop_loss, self.close_minus_atr(instrument, idx))
                    elif mp < 0:
                        stop_loss = min(stop_loss, self.close_plus_atr(instrument, idx))
                    broker.set_stop_loss(self, instrument, stop_loss)

                self.set_value(instrument, 'trailing_stop', broker.get_stop_loss(self, instrument), idx)     # for the charting only

    def process_trade_candidates(self, trade_candidates: List[TradeCandidate]):
        if len(trade_candidates) == 0:
            # no trade candidates
            return

        idx = self.idx
        cfg = typing.cast(self.LoosePantsConfig, self.get_config())
        broker = typing.cast(Broker, self.group.broker)
        open_trades = broker.open_trades(self)

        if cfg.cumulative:
            self.curr_account = cfg.portfolio_dollar + broker.closed_pnl(self)

        #
        # set position size and margin
        #
        # trade_candidates_all = [deepcopy(tc) for tc in trade_candidates]
        trade_candidates_all = deepcopy(trade_candidates)

        position_dollar = self.curr_account * cfg.risk_position

        for tc in trade_candidates:
            contracts = self.calc_nr_contracts(tc.instrument, position_dollar,
                                               self.atr(tc.instrument, idx) * cfg.atr_multiplier)
            if contracts > 0:
                tc.pos_size = contracts
                tc.margin = tc.instrument.metadata.margin * contracts
            else:
                tc.deleted = True

        trade_candidates = [tc for tc in trade_candidates if not tc.deleted]

        #
        # reorder by momentum - small impact on performance...
        # small to big is s bit better !
        #
        SMALL_TO_BIG = True
        BIG_TO_SMALL = False
        trade_candidates = sorted(trade_candidates, key=lambda tc: tc.momentum, reverse=SMALL_TO_BIG)

        #
        # restrict number of positions per sector
        #
        # @@@ len(open_trades) > 0 this may be wrong, not needed here ???
        if cfg.max_positions_per_sector > 0 and len(open_trades) > 0:
            # current number of open positions by sector
            open_sector_counts: dict[str, int] = {}
            for t in open_trades:
                open_sector_counts[t.sector] = open_sector_counts.get(t.sector, 0) + 1
            # no more than 3 trades in a sector
            for tc in trade_candidates:
                if open_sector_counts.get(tc.instrument.metadata.sector, 0) >= cfg.max_positions_per_sector:
                    tc.deleted = True

            trade_candidates = [t for t in trade_candidates if not t.deleted]

        #
        # restrict margin
        #
        if cfg.max_margin > 0:
            # Sort by margin: lowest first (reverse=False), so we can get max.number of positions
            trade_candidates = sorted(trade_candidates, key=lambda trc: trc.margin, reverse=False)

            margin_cum = sum([trade.margin for trade in open_trades])
            for tc in trade_candidates:
                if margin_cum + tc.margin > self.curr_account * cfg.max_margin:
                    tc.deleted = True
                else:
                    margin_cum += tc.margin
            trade_candidates = [t for t in trade_candidates if not t.deleted]
        #
        # restrict number of open positions
        #
        curr_positions = len(open_trades)
        max_pos = self.max_positions(cfg)   # or 10, 15,...
        if max_pos > 0:
            if curr_positions + len(trade_candidates) > max_pos:
                # delete trades exceeding MAX_POSITIONS
                nr_trades_to_delete = curr_positions + len(trade_candidates) - max_pos
                for i in range(len(trade_candidates) - nr_trades_to_delete, len(trade_candidates)):
                    # delete trades exceeding MAX_POSITIONS
                    trade_candidates[i].deleted = True

        #
        # trade remaining candidates
        #
        trade_candidates = [t for t in trade_candidates if not t.deleted]

        for tc in trade_candidates:
            contracts = tc.pos_size
            if tc.direction > 0:
                # long
                stop_loss = self.close_minus_atr(tc.instrument, idx)
            else:
                # short
                stop_loss = self.close_plus_atr(tc.instrument, idx)
                contracts = -contracts

            broker.open_position(self, tc.instrument,
                                 position=contracts,
                                 stop_loss=stop_loss,
                                 margin=tc.instrument.metadata.margin,
                                 momentum=tc.momentum)
        #
        # mark failed trade candidates
        #
        for tc in trade_candidates_all:
            if tc not in trade_candidates:
                self.set_value(tc.instrument, 'MissedTrade', True, idx)
                tc.deleted = True

    @staticmethod
    def max_positions(cfg) -> int:
        if cfg.risk_all_positions <= 0:
            return 0
        return int(np.floor(1.0 / abs(cfg.risk_position) * cfg.risk_all_positions))

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

        if self.get_config().close_last_trading_day:
            self.close_all_trades()
        broker = typing.cast(Broker, self.group.broker)
        self.log.debug(f"Number of trades: {len(broker.trades)}")
        self.log.debug("curr_account: " + str(self.curr_account))

        # Trade.print_trades(self.group.broker.trades)
