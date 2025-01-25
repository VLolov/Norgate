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
    # @dataclass
    # class Configx:
    #     CUMULATIVE: bool = True,
    #     PORTFOLIO_DOLLAR: float = 100_000
    #     RISK_POSITION: float = 1_000,
    #     RISK_ALL_POSITIONS: float = 0.3
    #     MAX_POSITIONS_PER_SECTOR: int = 0
    #     MAX_MARGIN: float = 0
    #     ATR_MUL: float = 5
    #     PERIOD: int = 12 * 21

    def __init__(self, name: str):
        super().__init__(name)

        # todo: transfer config from loose pants, but somehow without coupling
        self.config = None

        # input data comes from this report_single
        self._report_single: Optional[ReportSingle] = None

        # results:
        self._report_multi_df: Optional[pd.DataFrame] = None
        self._report_multi_stat: Optional[ReportMulti.Statistics] = None
        self._report_multi_table: Optional[pd.DataFrame] = None

        # self._report = ReportMulti.PerformanceIndicators()

        self.ready = False
        self.strategy_config = None

    def check_state(self) -> bool:
        return self.name != '' and self.backtester is not None and self._report_single is not None

    def set_report_single(self, report_single):
        self._report_single = report_single

    def get_report_multi_df(self):
        return self._report_multi_df

    def get_report_multi_statistics(self):
        return self._report_multi_stat

    def get_report_multi_table(self):
        return self._report_multi_table

    def run(self):
        self.log.info(f'Creating multi report: "{self.name}"')

        verbose = False

        if not self._report_single.ready:
            self._report_single.run()

        reports = self._report_single.get_all_reports()
        assert reports and len(reports) > 0, "No single reports"

        self.strategy_config = self.get_strategy_config(reports)

        df, statistics = self.combined_result(reports, verbose=verbose)
        self._report_multi_df, self._report_multi_stat = [df, statistics]
        self._report_multi_table = self.calc_table(df, statistics)

        self.ready = True
        # if self.log.getEffectiveLevel() == logging.DEBUG:
        #     self.log.debug("\n" + tabulate(df, headers='keys', tablefmt='psql'))
        #     self.log.debug(statistics)

    @staticmethod
    def get_strategy_config(reports):
        strategy_config = reports[0].strategy.get_config()
        required_attributes = ['cumulative', 'account', 'portfolio_dollar', 'risk_position',
                               'risk_all_positions', 'max_positions_per_sector', 'max_margin',
                               'atr_multiplier', 'period']
        strategy_config.check_attributes(required_attributes)
        return strategy_config

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
                self.log.debug(f'Trades, total {total:,.0f}, av.trade={avg:.0f}, costs={costs:,.0f}')
                trades_summary_df.reset_index(drop=True, inplace=True)
                trades_summary_df.sort_values(by='entry_date', inplace=True)
                self.log.debug("Single trades with calculated contracts:")
                self.log.debug("\n" + tabulate(trades_summary_df.sort_values(by='exit_date'), headers='keys', tablefmt='psql'))
    
            summary_df = pd.DataFrame(summary_rows)
    
            self.log.debug('Combined result with calculated contracts:')
            self.log.debug(tabulate(summary_df.sort_values(by='symbol'), headers='keys', tablefmt='psql'))
    
            self.log.debug(f'*** Pnl of all trades: {summary_df["pnl"].sum():,.0f}, {stat.total_return:,.0f}')
    
            stat.avg_dit = sum_dit / stat.number_trades if stat.number_trades > 0 else 0.0
    
            # print(f'*** Filter av.contracts > 1')
            # filtered_df = summary_df[summary_df['av.contracts'] > 1]
            self.log.debug(f'Tradable {len(tradable)} symbols:')
            self.log.debug('[', ', '.join(f'{sym}' for sym in tradable), ']')
            # print(tabulate(filtered_df, headers='keys', tablefmt='psql'))
            # print(tabulate(filtered_df, headers='keys', tablefmt='psql'))
    
        cumulative_df['Daily.return.pct'] = cumulative_df['Total'].pct_change().fillna(0).mul(100).round(1)
        cumulative_df['Daily.return.USD'] = (cumulative_df['Total'] - cumulative_df['Total'].shift()).fillna(0).round(0)
        # useful for pivots in excel
        cumulative_df['Year'] = cumulative_df.index.map(lambda x: pd.to_datetime(x).year)
        cumulative_df['Month'] = cumulative_df.index.map(lambda x: pd.to_datetime(x).month)
    
        # print("Cumulative df:")
        # print(tabulate(cumulative_df, headers='keys', tablefmt='psql'))
    
        # cumulative_df.to_excel(os.path.dirname(__file__) + "/cumulative_df_both.xlsx")
    
        return cumulative_df, stat

    def calc_table(self, cumulative_df, stat):
        cfg = self.strategy_config
        if cfg.cumulative:
            # don't accumulate again !!!
            # for col in ['Total', 'Total_Long', 'Total_Short']:
            #     pct_change = cumulative_df[col].pct_change().fillna(0)
            #     cumulative_df[col] = ((1 + pct_change).cumprod()) * cumulative_df[col]

            # yearly_agr = PlotPerformance.cagr(cumulative_df['Total'], compounding=cfg.cumulative)
            yearly_agr = self.Calc.cagr(cumulative_df['Total'], compounding=cfg.cumulative)
            ret_string = 'CAGR'
        else:
            yearly_agr = self.Calc.calc_avg_dollar(cumulative_df['Total'], cfg.portfolio_dollar)
            ret_string = 'Pct./year'

        sharpe = self.Calc.calc_sharpe(cumulative_df['Total'])
        sortino = self.Calc.calc_sortino(cumulative_df['Total'])
        volatility = self.Calc.calc_volatility(cumulative_df['Total'])
        trades_per_year, rolls_per_year, years = self.Calc.calc_trades_per_year(cumulative_df, stat.number_trades,
                                                                                stat.number_rolls)

        if cfg.cumulative:
            dd = cumulative_df['Total'] / cumulative_df['Total'].cummax() - 1
        else:
            dd = (cumulative_df['Total'] - cumulative_df['Total'].cummax()) / cfg.portfolio_dollar

        max_dd = dd.min()
        dd_duration_months = self.Calc.calc_max_dd_duration(dd) / 21
        mar = yearly_agr / abs(max_dd) if max_dd != 0 else 0

        total_return = cumulative_df['Total'].iloc[-1] - cumulative_df['Total'].iloc[0]
        # assert np.isclose(total_return, stat.total_return, atol=10), \
        #     f"Should be almost equal - total_return: {total_return} and stat: {stat.total_return}"

        cfg = self.strategy_config

        data = {
            'position risk%': f'{cfg.risk_position * 100:.2f} %',
            'position risk$': f'$ {self.Calc.position_size_dollar(cfg):,.0f}',
            'total risk': f'{cfg.risk_all_positions * 100:.2f} %',
            'acc.size': f'$ {cfg.portfolio_dollar:,.0f}',
            'max.positions': f'{self.Calc.max_positions(cfg)}',
            'max.per.sector': f'{cfg.max_positions_per_sector}',
            'max.margin': f'{cfg.max_margin}',
            'atr.multiplier': f'{cfg.atr_multiplier}',
            'lookback': f'{cfg.period}',
            'years': f'{years:.1f}',
            'nr.trades': f'{stat.number_trades}',
            'trades/year': f'{trades_per_year:.1f}',
            'nr.rolls': f'{stat.number_rolls:.0f}',
            'rolls/year': f'{rolls_per_year:.1f}',
            'av.trade': f'$ {stat.av_trade:,.0f}',
            'av.win': f'$ {stat.av_win:,.0f}',
            'av.loss': f'$ {stat.av_loss:,.0f}',
            'avg.dit': f'{stat.avg_dit:.0f}',
            'pf': f'{stat.pf:.2f}',
            'win.pct': f'{stat.win_pct * 100:.0f} %',
            ret_string: f'{yearly_agr * 100:.1f} %',
            'max.DD': f'{max_dd * 100:.1f} %',
            'DD.months': f'{dd_duration_months:.1f}',
            'volatility': f'{volatility * 100:.1f} %',
            'sharpe': f'{sharpe:.2f}',
            'sortino': f'{sortino:.2f}',
            'mar': f'{mar:.2f}',
            'net.return': f'$ {total_return:,.0f}',
            'included.costs': f'$ {stat.total_costs:,.0f}'
        }
        table_df = pd.DataFrame(data, index=[0])
        return table_df

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

    class Calc:
        @classmethod
        def calc_avg_dollar(cls, ser: pd.Series, account_size):
            # calculate yearly performance, no compounding
            total_days = (ser.index[-1] - ser.index[0]).days  # total calendar years
            return (ser.iloc[-1] - ser.iloc[0]) / (total_days / 365.25) / account_size

        @classmethod
        def calc_sharpe(cls, ser: pd.Series, rf=0):
            rets = ser.pct_change().fillna(0)

            mean = rets.mean() * 252 - rf
            std = rets.std() * np.sqrt(252)
            if std != 0:
                return mean / std
            return 0

        @classmethod
        def calc_sortino(cls, ser: pd.Series, rf=0):
            rets = ser.pct_change().fillna(0)

            mean = rets.mean() * 252 - rf
            std_neg = rets[rets < 0].std() * np.sqrt(252)
            if std_neg != 0:
                return mean / std_neg
            return 0

        @classmethod
        def calc_volatility(cls, ser: pd.Series):
            return ser.pct_change().fillna(0).std() * np.sqrt(252)

        @classmethod
        def calc_max_dd_duration(cls, dd: pd.Series):
            duration = pd.Series(index=dd.index, dtype=float)

            # Loop over the index range
            for t in range(1, len(dd)):
                duration.iloc[t] = 0 if dd.iloc[t] == 0 else duration.iloc[t - 1] + 1
            return duration.max()

        @classmethod
        def cagr(cls, equity, days=252, compounding=True):
            if compounding:
                periods = len(equity) / days
                return (equity.iloc[-1] / equity.iloc[0]) ** (1 / periods) - 1
            else:
                total_days = (equity.index[-1] - equity.index[0]).days  # total calendar years
                return (equity.iloc[-1] - equity.iloc[0]) / (total_days / 365.25)

        @classmethod
        def calc_trades_per_year(cls, cum_df: pd.DataFrame, total_trades, total_rolls):
            years = (cum_df.index[-1] - cum_df.index[0]).days / 365.25
            return total_trades / years, total_rolls / years, years

        @classmethod
        def position_size_dollar(cls, config):
            return config.portfolio_dollar * config.risk_position

        @classmethod
        def max_positions(cls, config) -> int:
            if config.risk_all_positions <= 0:
                return 0
            return int(np.floor(1.0 / abs(config.risk_position) * config.risk_all_positions))
            # return 10