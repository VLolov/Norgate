import typing
from typing import Optional, Dict, List
from dataclasses import dataclass

import numpy as np
import pandas as pd
from tabulate import tabulate

from Futures.Backtester.BacktesterBase import ReportBase
# from Futures.Backtester.BacktesterFutures import Strategy, Future, Trade, Broker
from .Strategy import Strategy
from .Future import Future
from .Trade import Trade
from .Broker import Broker

class ReportSingle(ReportBase):
    def __init__(self, name: str):
        super().__init__(name)

        self._strategies = List[Strategy]
        self._single_reports: Dict[int, ReportSingle.StrategyInstrumentReport] = {}
        self._first_report: Optional[ReportSingle.StrategyInstrumentReport] = None

        self.ready = False

    def check_state(self) -> bool:
        return self.name != '' and self.backtester is not None

    def run(self):
        self.log.info(f'Creating single report: "{self.name}"')
        strategies = self.get_strategies()
        self._strategies = strategies
        for strategy in strategies:
            for instrument in strategy.instruments:
                strategy = typing.cast(Strategy, strategy)
                instrument = typing.cast(Future, instrument)
                # self.log.debug(f"Calculating performance for strategy: {strategy}, instrument: {instrument}")
                self.calc_performance(strategy, instrument)

        # if self.log.getEffectiveLevel() == logging.DEBUG:
        #     df = pd.DataFrame([vars(report) for report in self._single_reports.values()])
        #     self.log.debug("\n" + tabulate(df, headers='keys', tablefmt='psql'))

        self.ready = True

        # for key, report in self.single_reports.items():
        #     self.log.debug(report)

    @dataclass
    class StrategyInstrumentReport:
        strategy: Strategy
        instrument: Future
        trades: List[Trade]

        nr_trades = 0
        avg_trade = 0.0
        avg_dit = 0.0
        nr_rolls = 0
        avg_contracts = 0
        avg_position_size_dollar = 0.0
        avg_margin = 0.0

        winning_trades = 0
        loosing_trades = 0
        max_dd = 0.0
        total_costs = 0.0  # calculated trading costs for trade entry/exit and slippage
        final_pnl = 0.0
        nr_missed_trades = 0
        yearly_ret = 0.0
        sharpe = 0.0

        @classmethod
        def get_key(cls, strategy, instrument):
            return strategy.id * 10000 + instrument.id

    def add_single_report(self, strategy, instrument):
        report = self.StrategyInstrumentReport(strategy=strategy, instrument=instrument, trades=[])
        key = self.StrategyInstrumentReport.get_key(strategy, instrument)
        self._single_reports[key] = report
        if len(self._single_reports) == 1:
            self._first_report = report
        return report

    def get_single_report(self, strategy, instrument):
        key = self.StrategyInstrumentReport.get_key(strategy, instrument)
        return self._single_reports.get(key, None)

    def get_first_report(self) -> 'ReportSingle.StrategyInstrumentReport':
        return self._first_report

    def get_all_reports(self) -> List['ReportSingle.StrategyInstrumentReport']:
        return list(self._single_reports.values())

    def get_report_strategies(self):
        return self._strategies

    def calc_performance(self, strategy: Strategy, instrument: Future):
        assert strategy.ready, f"Run strategy {strategy.name} first"

        df = instrument.data
        assert 'Close' in df.columns, f'Column "Close" not found in data'
        assert len(df) > 100, f'Data too short: {len(df)}'

        broker = typing.cast(Broker, strategy.group.broker)

        # add df columns for calculations and plotting

        # CloseStrategy is modified during the performance calculation to show the real exit price by stop loss
        # Later it is also modified to include trading costs
        df['CloseStrategy'] = df['Close']
        df['Signal'] = np.nan  # 1 for buy, -1 for sell, 0 for hold
        df['StopLoss'] = np.nan  # closing price at pos. exit
        df['Position'] = 0.0
        df['Margin'] = 0.0
        df['Contracts'] = 0.0
        df['MissedTrade'] = False

        # for trade in filter(lambda t: t.instrument == instrument, broker.trades):
        trades = [typing.cast(Trade, t) for t in broker.trades if not t.deleted and t.instrument == instrument]
        for trade in trades:
            # print(trade)
            if len(trade.trade_dates) < 2:
                # this trade is too short (probably just opened on the last bar)
                trade.deleted = True
                continue
            # self.log.debug(trade)
            new_exit_date, new_exit_price = self.get_trade_exit(trade, df)

            df.loc[trade.entry_date, 'Signal'] = 1 if trade.position > 0 else -1
            df.loc[new_exit_date, 'CloseStrategy'] = new_exit_price  # correct close price if we got stop loss
            df.loc[new_exit_date, 'StopLoss'] = new_exit_price if trade.is_stop_loss else np.nan

            # trade.trade_dates[1] - we take the day after the entry (entry on close)
            #            try:
            df.loc[trade.trade_dates[1]:new_exit_date, 'Position'] = trade.position
            df.loc[trade.trade_dates[1]:new_exit_date, 'Margin'] = trade.margin
            df.loc[trade.trade_dates[1]:new_exit_date, 'Contracts'] = trade.position
            # except:
            #     pass
        # self.data_access.close()  # remove: test only

        report = self.add_single_report(strategy, instrument)
        self._calc_trades(strategy, instrument, report)  # needs 'CloseStrategy'

        # percent returns
        # df['Returns'] = df['Close'] / df['Close'].shift(1) - 1
        # log returns
        # df['Returns'] = np.log(df['Close']) - np.log(df['Close'].shift())

        big_point = instrument.metadata.big_point
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

        report.final_pnl = df['Strat_Pnl'].iloc[-1]

        # calculate yearly performance, no compounding
        total_days = (df.index[-1] - df.index[0]).days  # total calendar years
        report.yearly_ret = report.final_pnl / (total_days / 365.25) / strategy.config.portfolio_dollar
        std = df['Strat_Returns'].std()
        if std > 0:
            report.sharpe = df['Strat_Returns'].mean() / df['Strat_Returns'].std() * np.sqrt(252)

        # print(tabulate(df, headers='keys', tablefmt='psql'))

    def _calc_trades(self, strategy: Strategy, instrument: Future, report):
        # calculate single trades
        assert strategy.ready, f"Run strategy {strategy.name} first"
        broker = strategy.group.broker
        df = instrument.data
        assert 'CloseStrategy' in df.columns, 'Column CloseStrategy is missing'

        # add trading costs
        rolls = 0
        self.total_costs = 0.0
        trades = [typing.cast(Trade, t) for t in broker.trades if t.instrument == instrument and t.strategy == strategy]

        report.trades = trades

        if strategy.cost_contract > 0 or strategy.slippage_ticks > 0:
            for trade in trades:
                new_exit_date, _ = self.get_trade_exit(trade, df)

                left, right = trade.entry_date, new_exit_date
                # full turn costs for one contract in $

                tick_size = instrument.metadata.tick_size
                slippage_ticks = strategy.slippage_ticks
                big_point = instrument.metadata.big_point
                cost_contract = strategy.cost_contract

                tick_slippage_costs = tick_size * slippage_ticks * big_point
                full_turn_costs = 2 * abs(trade.position * (strategy.cost_contract + tick_slippage_costs))

                trade_costs = (trade.rolls + 1) * full_turn_costs

                # price_shift > 0 for long; < 0 for short
                # price_shift contains only 1/2 of the overall costs, because we add it twice, on entry and on exit
                # Note: one roll contains two trades. Therefore, here we use * trade.rolls, meaning we consider
                # only one of these two trades
                price_shift = (trade.market_position * (trade.rolls + 1) *
                               (cost_contract / big_point + tick_size * slippage_ticks))

                df.loc[left, 'CloseStrategy'] += price_shift  # 1/2 of costs on entry
                df.loc[right, 'CloseStrategy'] -= price_shift  # 1/2 of costs on exit
                trade.costs = trade_costs
                report.total_costs += trade_costs
                rolls += trade.rolls

        report.nr_rolls = rolls

        # print("Total rolls=", rolls)
        # for trade in trades: print(trade)
        report.nr_trades = len(trades)
        if report.nr_trades > 0:
            report.avg_trade = (sum([trade.pnl * instrument.metadata.big_point - trade.costs for trade in trades])
                                / report.nr_trades)
            report.avg_dit = sum([trade.dit for trade in trades]) / report.nr_trades
            report.avg_contracts = sum([abs(trade.position) for trade in trades]) / report.nr_trades
            report.avg_position_size_dollar = (sum(
                [abs(trade.position * trade.entry_price) for trade in trades])
                                               / report.nr_trades * instrument.metadata.big_point)
            report.avg_margin = sum([trade.margin for trade in trades]) / report.nr_trades

        report.nr_missed_trades = len(df[df['MissedTrade']])

    @staticmethod
    def get_trade_exit(trade, df):
        if trade.exit_date is None:
            # the trade is still open, simulate closing of trade
            # take last date in date as 'closing' date
            new_exit_date = df.index[-1]
            new_exit_price = df['Close'].iloc[-1]
        else:
            new_exit_date = trade.exit_date
            new_exit_price = trade.exit_price

        return new_exit_date, new_exit_price
