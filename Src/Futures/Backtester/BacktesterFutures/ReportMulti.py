import logging
import typing
from typing import Optional, Dict, List
from dataclasses import dataclass

import numpy as np
import pandas as pd
from tabulate import tabulate

from Futures.Backtester.BacktesterBase import ReportBase, InstrumentBase
from Futures.Backtester.BacktesterFutures import Strategy, Future, Trade, Broker, ReportSingle

class ReportMulti(ReportBase):
    def __init__(self, name: str):
        super().__init__(name)
        self._report_single: Optional[ReportSingle] = None
        # self._report = ReportMulti.PerformanceIndicators()

    def check_state(self) -> bool:
        return self.name != '' and self.backtester is not None and self._report_single is not None

    def set_report_single(self, report_single):
        self._report_single = report_single

    def run(self):
        self.log.info(f'Creating multi report: "{self.name}"')
        
        if not self._report_single.ready:
            self._report_single.run()
        
        # gather all trades
        # trades: List[Trade] = []
        # instruments: List[InstrumentBase] = []
        # account_size = 0.0
        # for group in self.get_groups():
        #     trades.extend(group.broker.trades)
        #     account_size = group.broker.portfolio.initial_capital   # actually there is only one initial_capital
        #     instruments.append(group.instruments)
        
        reports = self._report_single.get_all_reports()
        
        df, statistics = self.combined_result(reports, verbose=True)
        pass
        

        # 
        # for instrument in strategy.group.instruments:
        #     strategy = typing.cast(Strategy, strategy)
        #     instrument = typing.cast(Future, instrument)
        #     # self.log.debug(f"Calculating performance for strategy: {strategy}, instrument: {instrument}")
        #     self.calc_performance(strategy, instrument)

        if self.log.level == logging.DEBUG:
            self.log.debug("\n" + tabulate(df, headers='keys', tablefmt='psql'))

    def combined_result(self, reports: List[ReportSingle.StrategyInstrumentReport], verbose=True) \
            -> typing.Tuple[pd.DataFrame, 'ReportMulti.Statistics']:
        
        account_size = self.backtester.portfolio.initial_capital
        
        # gather instruments
        instruments = []
        for report in reports:
            instruments.append(report.instrument)
    
        dates = self.strategies_dates(instruments)
    
        cumulative_df = pd.DataFrame(index=dates)
    
        cumulative_df['Total'] = account_size
        cumulative_df['Total_Long'] = account_size
        cumulative_df['Total_Short'] = account_size
    
        cumulative_df['Nr.Positions'] = 0
        cumulative_df['Nr.Positions_Long'] = 0
        cumulative_df['Nr.Positions_Short'] = 0
        cumulative_df['Margin'] = 0.0
    
        stat = self.Statistics()
    
        sum_dit = 0
    
        summary_rows = []
        tradable = []
        # re-connect to DB to re-calculate strategy performance
    
        trades_rows = []
        # get min/max date from strategies' data
        empty_df = pd.DataFrame(index=dates)
    
        idx = 0
        for report in reports:
            # strategy.data_access = data_access    # restore access to duckdb, which is needed in lp.calc_performance()
            instrument = typing.cast(Future, report.instrument)
            
            expanded_df = empty_df.merge(
                instrument.data[['Strat_Pnl', 'Strat_Pnl_Long', 'Strat_Pnl_Short', 'Position', 'Margin']],
                left_index=True, right_index=True, how='left')
            expanded_df = expanded_df.ffill().bfill()
    
            cumulative_df['Total'] += expanded_df['Strat_Pnl']
            cumulative_df['Total_Long'] += expanded_df['Strat_Pnl_Long']
            cumulative_df['Total_Short'] += expanded_df['Strat_Pnl_Short']
    
            pos = expanded_df[expanded_df['Position'] > 0]
            cumulative_df.loc[pos.index, 'Nr.Positions_Long'] += 1
    
            pos = expanded_df[expanded_df['Position'] < 0]
            cumulative_df.loc[pos.index, 'Nr.Positions_Short'] += 1
    
            # important pos on Position != 0 must be calculated here, as it is used below
            pos = expanded_df[expanded_df['Position'] != 0]
            cumulative_df.loc[pos.index, 'Nr.Positions'] += 1
    
            margin = expanded_df[expanded_df['Margin'] > 0]
            cumulative_df.loc[pos.index, 'Margin'] += margin['Margin']
            # strategy.broker.trades returns all trades, deleted or not deleted
            nr_trades = len([t for t in report.trades if not t.deleted])
            nr_missed_trades = len([t for t in report.trades if t.deleted])
            metadata = instrument.metadata
            summary_row = {'idx': idx, 'symbol': instrument.symbol, 'future.name': metadata.name,
                           'sector': metadata.sector,
                           'currency': metadata.currency,
                           'exchange': metadata.exchange, 'nr.trades': nr_trades,
                           'missed.trades': nr_missed_trades, 'av.contracts': np.round(report.avg_contracts, 1),
                           'pnl': np.round(report.final_pnl,0),
                           'Tradable': ('*' if nr_trades > nr_missed_trades else '')
                           }
            idx += 1
            summary_rows.append(summary_row)
            if nr_trades > nr_missed_trades:
                tradable.append(instrument.symbol)
    
            # print(strategy.future.symbol, strategy.nr_trades)
            sum_dit += sum([t.dit for t in report.trades if not t.deleted])
    
            for t in [t for t in report.trades if
                      not t.deleted]:  # and t.entry_date.to_pydatetime().year == 2024]:
                trade_dollar_risk = abs((t.entry_price - t.initial_stop_loss) * t.position * metadata.big_point)
    
                tx = {
                    'symbol': t.symbol,
                    'sector': t.sector,
                    'entry_date': t.entry_date, 'entry_price': t.entry_price,
                    'exit_date': t.exit_date, 'exit_price': t.exit_price, 'margin': t.margin,
                    'initial_stop_loss': t.initial_stop_loss, 'stop_loss': t.stop_loss,
                    'position': t.position,
                    'closed': t.is_closed, 'is_stop_loss': t.is_stop_loss,
                    'dit': t.dit,
                    'rolls': t.rolls,
    
                    'costs': np.round(t.costs, 0),
                    'dollar_risk': np.round(trade_dollar_risk, 0),
                    'pnl': np.round(t.pnl * metadata.big_point, 0)
                }
    
                trades_rows.append(tx)
                stat.add_trade(t, metadata.big_point)
    
        # all strategies processed
    
        if verbose:
            trades_summary_df = pd.DataFrame(trades_rows)
            if len(trades_summary_df) > 0:
                total = trades_summary_df['pnl'].sum()
                avg = trades_summary_df['pnl'].mean()
                costs = trades_summary_df['costs'].sum()
                print(f'Trades, total {total:,.0f}, av.trade={avg:.0f}, costs={costs:,.0f}')
                trades_summary_df.reset_index(drop=True, inplace=True)
                trades_summary_df.sort_values(by='entry_date', inplace=True)
                print("Single trades with calculated contracts:")
                print(tabulate(trades_summary_df.sort_values(by='exit_date'), headers='keys', tablefmt='psql'))
    
            summary_df = pd.DataFrame(summary_rows)
    
            print('Combined result with calculated contracts:')
            print(tabulate(summary_df.sort_values(by='symbol'), headers='keys', tablefmt='psql'))
    
            print(f'*** Pnl of all trades: {summary_df["pnl"].sum():,.0f}, {stat.total_return:,.0f}')
    
            stat.avg_dit = sum_dit / stat.number_trades if stat.number_trades > 0 else 0.0
    
            # print(f'*** Filter av.contracts > 1')
            # filtered_df = summary_df[summary_df['av.contracts'] > 1]
            print(f'Tradable {len(tradable)} symbols:')
            print('[', ', '.join(f'{sym}' for sym in tradable), ']')
            # print(tabulate(filtered_df, headers='keys', tablefmt='psql'))
            # print(tabulate(filtered_df, headers='keys', tablefmt='psql'))
    
        cumulative_df['Daily.return.pct'] = cumulative_df['Total'].pct_change().mul(100).round(1)
        cumulative_df['Daily.return.USD'] = (cumulative_df['Total'] - cumulative_df['Total'].shift()).round(0)
        # useful for pivots in excel
        cumulative_df['Year'] = cumulative_df.index.map(lambda x: pd.to_datetime(x).year)
        cumulative_df['Month'] = cumulative_df.index.map(lambda x: pd.to_datetime(x).month)
    
        # print("Cumulative df:")
        # print(tabulate(cumulative_df, headers='keys', tablefmt='psql'))
    
        # cumulative_df.to_excel(os.path.dirname(__file__) + "/cumulative_df_both.xlsx")
    
        return cumulative_df, stat

    @staticmethod
    def strategies_dates(instruments: List[InstrumentBase]) -> List[pd.Timestamp]:
        # gather dates of all strategies
        strategy_dates = set()
        for instrument in instruments:
            strategy_dates.update(instrument.data.index)
        strategy_dates = sorted(list(strategy_dates))
        return strategy_dates

    @dataclass
    class Statistics:
        number_winning_trades = 0
        number_loosing_trades = 0
        number_rolls = 0
        total_win = 0.0
        total_loss = 0.0
        avg_dit = 0.0
        total_costs = 0.0

        def add_trade(self, t: Trade, big_point: float) -> None:
            if t.pnl > 0:
                self.number_winning_trades += 1
                self.total_win += t.pnl * big_point - t.costs
            else:
                self.number_loosing_trades += 1
                self.total_loss += t.pnl * big_point - t.costs
            self.number_rolls += t.rolls
            self.total_costs += t.costs

        @property
        def total_return(self) -> float:
            return self.total_win + self.total_loss

        @property
        def number_trades(self) -> int:
            return self.number_winning_trades + self.number_loosing_trades

        @property
        def pf(self) -> float:
            return self.total_win / abs(self.total_loss) if self.total_loss != 0 else 0

        @property
        def win_pct(self) -> float:
            return self.number_winning_trades / self.number_trades if self.number_trades > 0 else 0

        @property
        def av_win(self) -> float:
            return self.total_win / self.number_winning_trades if self.number_winning_trades > 0 else 0

        @property
        def av_loss(self) -> float:
            return self.total_loss / self.number_loosing_trades if self.number_loosing_trades > 0 else 0

        @property
        def av_trade(self) -> float:
            return self.total_return / self.number_trades if self.number_trades > 0 else 0

