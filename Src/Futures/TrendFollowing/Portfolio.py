"""
    Started 15.10.2024

    see: https://newsletter.tradingstrategies.live/p/practical-diversified-trend-following

    Tasklist 28.10.2024:
    - Add CTA indexes (and SPY?) and use as benchmark
    + Create a portfolio with micro contracts and other contracts fitting in $1000 risk
    - Import commodity and other ETFs and trade with them instead of futures -
        * check first if the returns are big enough to trade without leverage
    + Calculate and plot the number of affordable futures (100k account)
    - Consider using back contracts as in Carlo Mata's article
        * I'll need to generate the continuous contract myself...
        * Add commission and slippage when rolling contracts

    06.11.2024:
    Implemented Carlo Mata's: 'OK now I will make a decision to get rid of shorts in
        Fixed Income, Equities, Volatility, Grains, Softs, Metals...'

"""
import multiprocessing
import os
import sys

from dataclasses import dataclass

from typing import List, Tuple

import numpy as np
import pandas as pd
import duckdb
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from tabulate import tabulate
from tqdm import tqdm
import mplcursors as mpl

from Futures.DBConfig import DBConfig

from Futures.TrendFollowing.DataAccess import DataAccess
from Futures.TrendFollowing.Future import Future
from Futures.TrendFollowing.MaxFront import MaxFront
from Futures.TrendFollowing.Timer import Timer
from Futures.TrendFollowing.plot_histogram_returns import plot_histogram_returns
from Futures.TrendFollowing.plot_qq_returns import plot_qq_returns
from Futures.TrendFollowing.Strategy import Trade
from Futures.TrendFollowing.LoosePants import LoosePants
from Futures.TrendFollowing.PlotPerformance import PlotPerformance


@dataclass
class Config:
    """
    6.11.2024 good performance with:
       50_000, 2% risk, 0.4 margin, no nr.positions restriction or 10 pos (RISK_ALL_POSITIONS = 0.20)
          since 2010: av 15.9% dd -27, av trade $219, sharpe 0.67, all long/short, only USD
          skip_shorts av 16.6% dd -18, av trade $269, sharpe 0.77
    09.11.2024:
      50_000, 2% risk, 0.4 margin, RISK_ALL_POSITIONS = 0.20, period 12*21 (!), Cumulative (!), short and long
            since 2010: CAGR 27% dd -47, av trade $5000, sharpe 0.85, dd months 27.9
          * the same, but period 6*21 - much worse:
            since 2010: CAGR 18.7% dd -57, av trade $1500, sharpe 0.64, dd months 59.5
          * the same, but period 18*21 - much worse:
            since 2010: CAGR 20.2% dd -50, av trade $3700, sharpe 0.81, dd months 73.7 (!)
          * the same, but period 3*21 - much worse:
            since 2010: CAGR 27.6% dd -59, av trade $3900, sharpe 0.83, dd months 60
          => can we combine several periods?
    """
    # 17.11.2024: good combinations, but only together with front_by_sectors:
    #         self.PORTFOLIO_DOLLAR = 100_000
    #         self.RISK_POSITION = 0.01
    #         self.RISK_ALL_POSITIONS = 0.1 (means, max 10 positions
    #         self.MAX_MARGIN = 0.2
    #         self.ATR_MUL = 3 !
    #         self.PERIOD = 12 * 21 or 6 * 21

    PORTFOLIO_DOLLAR = 100_000
    RISK_POSITION = 0.02        # % of portfolio, if RISK_POSITION < 0: trade with +/- 1 contract
    RISK_ALL_POSITIONS = 0.2    # % of portfolio; 0=don't check
    MAX_MARGIN = 0.4            # % of portfolio; 0=don't check margin
    ATR_MUL = 5                 # 2 or 3 or 5  # !!!
    PERIOD = 12 * 21            # !!!
    START_DATE = '1980-01-01'   # start of data: '1970-01-01' (1980-01-01)
    END_DATE = '3015-04-01'
    USE_STOP_LOSS = True
    CUMULATIVE = False
    PATCH_MICRO = False
    MULTI_PROCESSING = True
    SECTOR = ''
    BUY_AND_HOLD = False
    COST_CONTRACT = 2
    SLIPPAGE_TICKS = 2
    MAX_POSITIONS_PER_SECTOR = 0
    ORDER_EXECUTION_DELAY = 0


sectors = ['Crypto', 'Currency', 'Energy', 'Volatility', 'Equity',
           'Metal', 'Fixed Income', 'Rates', 'Grain', 'Meat']

# 'CL','HO','RB','NG','GC','LC','_C','_S','_W','SB', 'HG', 'CT', 'KC'

skip_short = []
# with skip: better sharp, higher return, but higher DD
# skip shorts is important if a restricted list like tradable_symbols_1000 is used (29.11.2024)
# skip_short = ['Equity', 'Metal', 'Fixed Income', 'Grain', 'Soft']

skip_long = []
# skip_long = ['Volatility']

# with PATCH_MICRO=True, micro contracts are excluded
tradable_symbols_1000 = ['6A', '6B', '6C', '6E', '6J', '6M', 'BTC', 'CC', 'CL', 'CT', 'DC', 'DX',
                         'EMD', 'ES', 'ETH', 'FCE', 'FDAX', 'FESX', 'FGBL', 'FGBM', 'FGBS', 'FOAT', 'FTDX',
                         'GC', 'HE', 'HG', 'HTW', 'KE', 'LEU', 'LRC', 'LSU', 'NG', 'NKD', 'NQ', 'RTY', 'SB',    # 'MWE' - couldnt find in IB
                         'SCN', 'SI', 'SR3', 'TN', 'UB', 'VX', 'YM', 'ZC', 'ZF', 'ZL', 'ZN', 'ZO', 'ZQ', 'ZR', 'ZS',
                         'ZT', 'ZW']

tradable_symbols_2000 = ['6A', '6B', '6C', '6E', '6J', '6M', 'BTC', 'CC', 'CL', 'CT', 'DC', 'DX',
                         'EMD', 'ES', 'ETH', 'FCE', 'FDAX', 'FESX', 'FGBL', 'FGBM', 'FGBS', 'FOAT', 'FTDX',
                         'GC', 'HE', 'HG', 'HTW', 'KE', 'LEU', 'LRC', 'LSU', 'MWE', 'NG', 'NKD', 'NQ', 'RTY',
                         'SB', 'SCN', 'SI', 'SR3', 'TN', 'UB', 'VX', 'YM', 'ZC', 'ZF', 'ZL', 'ZN', 'ZO', 'ZQ', 'ZR',
                         'ZS',
                         'ZT', 'ZW']

symbols_finviz = [
    'YM', 'ES', 'NQ' 'RTY',
    'NKD', 'FESX', 'FDAX', 'VX',
    'CL', 'BRN', 'RB', 'HO',
    'NG',  'GC', 'SI',                         # Ethanol skipped - too low volume
    'PL', 'HG', 'PA', 'LE',
    'GF', 'HE', 'ZC', 'ZL',
    'ZM', 'ZO', 'ZR', 'ZS',
    'KE', 'RS', 'CC', 'CT',
    'OJ', 'KC', 'LBR', 'SB',
    'UB', 'ZN', 'ZF', 'ZT',
    'DX', '6E', '6J', '6G'
    '6C', '6S', '6A', '6N'
    ]

# P24 from Universal Portfolio - P24 - I have only 17 (no heating oil and 3 * meats excluded)
# Works: a bit worse than tradable_symbols_1000, cannot find 10 positions, 1/2 of the trades with risk 2.5%
symbols_p24 = [
    '6E', '6J', '6B',
    'ZN', 'UB', 'ZT',
    'ES', 'NQ', 'YM',
    'CL', 'NG',
    'GC', 'HG', 'SI',
    'ZC', 'ZS', 'CT',
]

# benchmark = dta.index('$SPXTR')


def position_size_dollar(config: Config):
    return config.PORTFOLIO_DOLLAR * config.RISK_POSITION


def max_positions(config: Config) -> int:
    if config.RISK_ALL_POSITIONS <= 0:
        return 0
    return int(np.floor(1.0 / abs(config.RISK_POSITION) * config.RISK_ALL_POSITIONS))
    # return 10


def processing_task(f_d: List) -> List:
    future, data, strategy = f_d
    strategy.run(future=future, data=data)
    strategy.calc_performance()
    return [future, data, strategy]


def calc_strategies(cfg, verbose=True) -> List[LoosePants]:
    # calculate all strategies with 1 contract

    # FutureNorgate.patch_micro_futures(cfg.PATCH_MICRO)
    futures = Future.all_futures_norgate(use_micro=cfg.PATCH_MICRO)

    futures_data = []

    nr_futures_traded = 0
    for index, future in enumerate(tqdm(futures, desc='Prepare data', colour='green')):
        # if future.symbol not in Future.micro_futures():
        #     continue

        if future.symbol not in tradable_symbols_1000:
            continue

        # if future.symbol in ['FOAT', 'SCN', 'SR3', 'TN', 'UB']:
        #     # short history when backtesting until 2010-01-01
        #     continue
        # if future.symbol not in the Future.micro_futures_list.keys():
        #     continue

        if 'Micro' in future.name:
            continue

        if future.sector in ['Meat', 'Volatility']:
            # skipped sectors
            continue

        # if future.sector not in ['Fixed Income', 'Rates']:
        #     continue
        # ['Crypto', 'Currency', 'Energy', 'Volatility', 'Equity', 'Metal', 'Fixed Income', 'Rates',
        #    'Grain', 'Soft', 'Meat']):

        # if future.sector in ['Crypto']:
        #     continue

        # if future.sector in ['Equity', 'Volatility', 'Crypto']:
        #     continue

        # if future.sector in ['Equity']:
        #     continue

        # if future.sector in ['Crypto', 'Volatility']:
        #     continue
        if future.name in ['GAS']:
            # not available for trading in IB
            continue
        if cfg.SECTOR:
            if future.sector != cfg.SECTOR:
                continue

        # if future.currency not in ['GBP', 'USD', 'EUR', 'CAD']:
        # if future.currency not in ['USD', 'EUR', 'GBP', 'CAD']:
        #     continue

        if future.exchange not in ['CME', 'ICE US', 'NYMEX', 'CME', 'Eurex', 'ICE Europe', 'CBOT', 'CBOE']:
        # if future.exchange not in ['CME', 'ICE US', 'NYMEX', 'CME', 'CBOT', 'CBOE']:
            continue

        nr_futures_traded += 1

        front = MaxFront.front_by_sector(future.sector)
        modified_front = MaxFront.set_max_front(future.symbol, future.sector, front)
        # # print(f'Max front requested: {front}, modified: {modified_front}')
        front = modified_front
        # print("Symbol", future.symbol, "Front", front)
        # front = 1   # 0: use Norgate's continuous contract; >0: use 'our' continuous contracts

        with duckdb.connect(DBConfig.DUCK_DB, read_only=True) as connection:
            data_access = DataAccess(connection, cfg.START_DATE, cfg.END_DATE)
            data = LoosePants.get_data(data_access, future, front=front)
            # remove duckdb connection, as it cannot be pickled
            future.dta = None

        if cfg.BUY_AND_HOLD:
            long = True
            short = False
        else:
            short = future.sector not in skip_short
            long = future.sector not in skip_long

        strategy = LoosePants(
            period=cfg.PERIOD,
            atr_multiplier=cfg.ATR_MUL,
            use_stop_loss=cfg.USE_STOP_LOSS,
            use_trailing_stop=True,
            use_stop_orders=True,
            long=long,
            short=short,
            use_one_contract=True,
            dollar_risk=0,
            cost_contract=cfg.COST_CONTRACT,
            slippage_ticks=cfg.SLIPPAGE_TICKS,
            cumulative=True,
            order_execution_delay=cfg.ORDER_EXECUTION_DELAY
        )
        # strategy.set_next(strategy.next_counter_trend)
        if cfg.BUY_AND_HOLD:
            strategy.set_next(strategy.next_buy_and_hold_monthly)

        futures_data.append([future, data, strategy])



    results = []
    if cfg.MULTI_PROCESSING:
        # print(f'Number CPUs: {multiprocessing.cpu_count()}')
        nr_cpu = multiprocessing.cpu_count()//2
        with multiprocessing.Pool(processes=nr_cpu) as pool:
            for result in tqdm(pool.imap(processing_task, futures_data),
                               desc=f'Run BacktesterFutures on {nr_cpu} CPUs', colour='green'):
                results.append(result)
    else:
        for fd in tqdm(futures_data, desc='Run BacktesterFutures SingleCPU', colour='green'):
            results.append(processing_task(fd))

    assert len(results) > 0, 'No trades'

    strategy_results = []
    summary_rows = []

    index = 0
    for fd in results:
        future, data, strategy = fd
        strategy_results.append(strategy)
        summary_row = {'idx': index, 'symbol': strategy.future.symbol, 'future.name': strategy.future.name,
                       'sector': future.sector,
                       'currency': future.currency,
                       'exchange': strategy.future.exchange, 'nr.trades': strategy.nr_trades,
                       'missed.trades': strategy.nr_missed_trades, 'av.contracts': np.round(strategy.avg_contracts, 1),
                       'pnl': np.round(strategy.final_pnl, 0),
                       # 'Tradable': ('*' if strategy.nr_trades > strategy.nr_missed_trades else '')
                       'Tradable': ('*' if strategy.avg_contracts > 1 else '')
                       }
        summary_rows.append(summary_row)
        index += 1
    if verbose:
        summary_df = pd.DataFrame(summary_rows)
        print("Strategies with 1 contract:")
        print(tabulate(summary_df, headers='keys', tablefmt='psql'))

    return strategy_results

    # data_obj = {'strategy_results': strategy_results, 'cfg': cfg}
    # with open(PICKLE, 'wb') as f:
    #     pickle.dump(data_obj, f)
    #
    # print(f'Strategies saved in {PICKLE}')


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


def combined_result(cfg: Config, strategy_results: List[LoosePants], verbose=True) \
        -> Tuple[pd.DataFrame, Statistics]:

    dates = strategies_dates(strategy_results)

    cumulative_df = pd.DataFrame(index=dates)

    account_size = cfg.PORTFOLIO_DOLLAR
    cumulative_df['Total'] = account_size
    cumulative_df['Total_Long'] = account_size
    cumulative_df['Total_Short'] = account_size

    cumulative_df['Nr.Positions'] = 0
    cumulative_df['Nr.Positions_Long'] = 0
    cumulative_df['Nr.Positions_Short'] = 0
    cumulative_df['Margin'] = 0.0

    stat = Statistics()

    sum_dit = 0

    summary_rows = []
    tradable = []
    # re-connect to DB to re-calculate strategy performance

    trades_rows = []
    # get min/max date from strategies' data
    empty_df = pd.DataFrame(index=dates)

    idx = 0
    for strategy in strategy_results:
        # strategy.data_access = data_access    # restore access to duckdb, which is needed in lp.calc_performance()
        strategy.calc_performance()

        expanded_df = empty_df.merge(
            strategy.data[['Strat_Pnl', 'Strat_Pnl_Long', 'Strat_Pnl_Short', 'Position', 'Margin']],
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
        nr_trades = len([t for t in strategy.broker.trades if not t.deleted])
        nr_missed_trades = len([t for t in strategy.broker.trades if t.deleted])
        fut = strategy.future
        summary_row = {'idx': idx, 'symbol': fut.symbol, 'future.name': fut.name,
                       'sector': fut.sector,
                       'currency': fut.currency,
                       'exchange': fut.exchange, 'nr.trades': nr_trades,
                       'missed.trades': nr_missed_trades, 'av.contracts': np.round(strategy.avg_contracts, 1),
                       'pnl': np.round(strategy.final_pnl,0),
                       'Tradable': ('*' if nr_trades > nr_missed_trades else '')
                       }
        idx += 1
        summary_rows.append(summary_row)
        if nr_trades > nr_missed_trades:
            tradable.append(fut.symbol)

        # print(strategy.future.symbol, strategy.nr_trades)
        sum_dit += sum([t.dit for t in strategy.broker.trades if not t.deleted])

        for t in [t for t in strategy.broker.trades if
                  not t.deleted]:  # and t.entry_date.to_pydatetime().year == 2024]:
            trade_dollar_risk = abs((t.entry_price - t.initial_stop_loss) * t.position * strategy.future.big_point)

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
                'pnl': np.round(t.pnl * strategy.future.big_point, 0)
            }

            trades_rows.append(tx)
            stat.add_trade(t, strategy.future.big_point)

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


def portfolio_with_constraints(cfg, strategy_results, constraints=True, verbose=True) -> [pd.DataFrame, Statistics]:
    # NOTE: to call this function, set Config.RISK_POSITION < 0, so that each trade is with +/- 1 contract
    # with open(PICKLE, 'rb') as f:
    #    data_obj = pickle.load(f)

    # data_obj = {'strategy_results': strategy_results, 'cfg': cfg}
    # strategy_results: List[LoosePants] = data_obj['strategy_results']
    # saved_config: Config = data_obj['cfg']

    curr_account = cfg.PORTFOLIO_DOLLAR    # initial account
    closed_pnl = 0

    # get all dates in the strategies
    strategy_dates = strategies_dates(strategy_results)

    for timestamp in tqdm(strategy_dates, desc='Apply constraints', colour='green'):
        # for each day
        trade_candidates: List[Trade] = []  # Trades that 'want' to open today

        trade_candidates_strategy: List[LoosePants] = []  # Trades that 'want' to open today
        closing_trades: List[Trade] = []     # Trades that close today
        open_trades: List[Trade] = []       # Trades open today
        if timestamp == pd.Timestamp(2022, 11, 2):
            j = 0
        for strategy in strategy_results:
            # for each strategy
            for trade in strategy.trades:
                # for each trade in the strategy
                if trade.entry_date == timestamp:
                    trade_candidates.append(trade)
                    trade_candidates_strategy.append(strategy)
                if trade.exit_date == timestamp:
                    closing_trades.append(trade)
                    # note: at this point, the correct trade.position is set
                    closed_pnl += trade.pnl * strategy.big_point
                if trade.entry_date < timestamp <= trade.exit_date:
                    # timestamp < trade.exit_date - means: don't consider trades that close today
                    open_trades.append(trade)

        # break point
        # if timestamp.to_pydatetime() > datetime(2023, 9, 1) and len(trade_candidates) > 0:
        #    i = 1

        if cfg.CUMULATIVE:
            curr_account = cfg.PORTFOLIO_DOLLAR + closed_pnl

        #
        # set new position size of each trade candidate
        #
        for trade_candidate, strategy in zip(trade_candidates, trade_candidates_strategy):
            trade_candidate: Trade
            strategy: LoosePants
            big_point = strategy.big_point
            assert abs(trade_candidate.position) == 1, 'wrong tc.position'
            pos_risk = curr_account * abs(cfg.RISK_POSITION)
            stop_loss_distance = abs(trade_candidate.entry_price - trade_candidate.initial_stop_loss)
            if stop_loss_distance > 0:
                nr_contracts = pos_risk / stop_loss_distance / big_point
                # if trade_candidate.position < 0: nr_contracts /= 2.0    # short position * 1/2
                new_position = np.round(nr_contracts, 0)    # math rounding !!!
                # new_position = nr_contracts                   # trade with partial contracts (test only)
                # new_position = np.floor(nr_contracts)         # floor

                if new_position > 0:
                    # ensure the correct long/short direction
                    new_position = np.copysign(new_position, trade_candidate.position)
                    trade_candidate.position = new_position
                    trade_candidate.margin = trade_candidate.margin * abs(new_position)
                else:
                    trade_candidate.deleted = True
            else:
                if not cfg.BUY_AND_HOLD:
                    # buy and hold trades have no stop loss
                    # stop_loss_distance <= 0 - something is wrong with this trade
                    trade_candidate.deleted = True

        if constraints:
            trade_candidates = [t for t in trade_candidates if not t.deleted]
            if len(trade_candidates) < 1:
                # no trades to restrict on this date
                continue

            # reorder by momentum - small impact on performance...
            # small to big is s bit better !
            SMALL_TO_BIG = True
            BIG_TO_SMALL = False
            trade_candidates = sorted(trade_candidates, key=lambda t: t.momentum, reverse=SMALL_TO_BIG)

            if cfg.MAX_POSITIONS_PER_SECTOR > 0 and len(open_trades) > 0:
                # current number of open positions by sector
                open_sector_counts: dict[str, int] = {}
                for t in open_trades:
                    open_sector_counts[t.sector] = open_sector_counts.get(t.sector, 0) + 1
                # no more than 3 trades in a sector
                for t in trade_candidates:
                    if open_sector_counts.get(t.sector, 0) >= cfg.MAX_POSITIONS_PER_SECTOR:
                        t.deleted = True

            # reorder by sector to diversify sectors - I don't see difference, why - do we need this?
            # trade_candidates = reorder_by_sector(open_trades, trade_candidates)

            # reorder by margin - almost no difference
            # trade_candidates = sorted(trade_candidates, key=lambda t: t.margin, reverse=False)

            # does not work well as it reduces number of trades
            # remove_same_sector(open_trades, trade_candidates)

            #
            # restrict number of open positions
            #
            curr_positions = len(open_trades)
            max_pos = max_positions(cfg)   # or 10, 15,...
            if max_pos > 0 and (curr_positions + len(trade_candidates) > max_pos):
                # delete trades exceeding MAX_POSITIONS
                nr_trades_to_delete = curr_positions + len(trade_candidates) - max_pos
                for idx in range(len(trade_candidates) - nr_trades_to_delete, len(trade_candidates)):
                    # delete trades exceeding MAX_POSITIONS
                    trade_candidates[idx].deleted = True

            #
            # restrict margin
            #
            trade_candidates = [t for t in trade_candidates if not t.deleted]
            if cfg.MAX_MARGIN > 0:
                # Sort by margin: lowest first (reverse=False), so we can get max.number of positions
                trade_candidates = sorted(trade_candidates, key=lambda t: t.margin, reverse=False)

                margin_cum = sum([trade.margin for trade in open_trades if not trade.deleted])
                # if timestamp.to_pydatetime() > datetime(1998, 10, 11):
                #     # break point
                #     i = 0
                for idx, tr in enumerate(trade_candidates):
                    if margin_cum + tr.margin > curr_account * cfg.MAX_MARGIN:
                        trade_candidates[idx].deleted = True
                    else:
                        margin_cum += tr.margin

    # re-calculate strategies' performance considering the deleted trades
    cumulative_df, stat = combined_result(cfg=cfg, strategy_results=strategy_results, verbose=verbose)

    if cfg.CUMULATIVE:
        # don't accumulate again !!!
        # for col in ['Total', 'Total_Long', 'Total_Short']:
        #     pct_change = cumulative_df[col].pct_change().fillna(0)
        #     cumulative_df[col] = ((1 + pct_change).cumprod()) * cumulative_df[col]

        yearly_agr = PlotPerformance.cagr(cumulative_df['Total'], compounding=cfg.CUMULATIVE)
        ret_string = 'CAGR'
    else:
        yearly_agr = calc_avg_dollar(cumulative_df['Total'], cfg.PORTFOLIO_DOLLAR)
        ret_string = 'Pct./year'

    sharpe = calc_sharpe(cumulative_df['Total'])
    sortino = calc_sortino(cumulative_df['Total'])
    volatility = calc_volatility(cumulative_df['Total'])
    trades_per_year, rolls_per_year, years = calc_trades_per_year(cumulative_df, stat.number_trades, stat.number_rolls)

    if cfg.CUMULATIVE:
        dd = cumulative_df['Total'] / cumulative_df['Total'].cummax() - 1
    else:
        dd = (cumulative_df['Total'] - cumulative_df['Total'].cummax()) / cfg.PORTFOLIO_DOLLAR

    max_dd = dd.min()
    dd_duration_months = calc_max_dd_duration(dd) / 21
    mar = yearly_agr / abs(max_dd) if max_dd != 0 else 0

    total_return = cumulative_df['Total'].iloc[-1] - cumulative_df['Total'].iloc[0]
    # assert np.isclose(total_return, stat.total_return, atol=10), \
    #     f"Should be almost equal - total_return: {total_return} and stat: {stat.total_return}"

    if verbose:
        strategy = r"Buy\ and\ Hold\ " if cfg.BUY_AND_HOLD else r"BacktesterFutures\ "

        title = (
            r'$\bf{' + strategy + (r'with\ constraints' if constraints else r'no\ constraints')
            + (r'\ -\ Cumulative' if cfg.CUMULATIVE else '')
            + (r'\ -\ ' + cfg.SECTOR if cfg.SECTOR else '')
            + r'}$'
        )

        data = {
            'position risk%': f'{cfg.RISK_POSITION * 100:.2f} %',
            'position risk$': f'$ {position_size_dollar(cfg):,.0f}',
            'total risk': f'{cfg.RISK_ALL_POSITIONS * 100:.2f} %',
            'acc.size': f'$ {cfg.PORTFOLIO_DOLLAR:,.0f}',
            'max.positions': f'{max_positions(cfg)}',
            'max.per.sector': f'{cfg.MAX_POSITIONS_PER_SECTOR}',
            'max.margin': f'{cfg.MAX_MARGIN}',
            'atr.multiplier': f'{cfg.ATR_MUL}',
            'lookback': f'{cfg.PERIOD}',
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
        draw_chart(cumulative_df, title, cfg, table_df)
        plot_performance(cumulative_df, cfg)

        daily_returns = cumulative_df['Total'].pct_change().fillna(0)
        # print(f'Daily returns, min: {daily_returns.min()*100:.1f} %, max: {daily_returns.max()*100:.1f} %')
        # print(tabulate(pd.DataFrame(daily_returns*100), headers='keys', tablefmt='psql', floatfmt='.1f'))
        # plot_histogram_returns(daily_returns[daily_returns <= -0.02], resample_rule='D')  # 'D', 'ME'
        plot_histogram_returns(daily_returns, resample_rule='ME')  # 'D', 'ME', 'W', 'QE'
        plot_qq_returns(daily_returns, resample_rule='ME')  # 'D', 'ME'

        save_pnl(cfg, cumulative_df)

    return cumulative_df, stat


def plot_performance(cumulative_df, config: Config):
    # percent performance
    account_size = config.PORTFOLIO_DOLLAR
    returns_pct = cumulative_df['Total'].pct_change().fillna(0)

    pp = PlotPerformance()
    # pp.equity_plot(r'$\bf{' + 'Performance' + '}$\n'
    #                 f'Account size: {account_size:,.0f}, '
    #                 f'Position size: ${position_size_dollar(config):,.0f}, '
    #                 f'Risk position: {config.RISK_POSITION*100:.2f}%',
    #                'benchmark', returns_pct,
    #                'strategy', returns_pct,
    #                states=None,
    #                nr_strategies=None,
    #                compounding=True,
    #                scale='log')

    pp.yearly_plot('Yearly', 'BacktesterFutures', returns_pct,
                   show_difference=False, print_performance=True)

    # plt.show(block=False)


def strategies_dates(strategy_results: List[LoosePants]) -> List[pd.Timestamp]:
    # gather dates of all strategies
    strategy_dates = set()
    for strategy in strategy_results:
        strategy_dates.update(strategy.data.index)
    strategy_dates = sorted(list(strategy_dates))
    return strategy_dates


def reorder_by_sector(open_trades, trade_candidates):
    # reorder, so that we diversify by sector
    reordered_candidates = []
    open_sectors = set([t.sector for t in open_trades])
    candidate_sectors = set([t.sector for t in trade_candidates if t.sector not in open_sectors])

    for t in trade_candidates:
        if t.sector not in open_sectors:
            # if the sector of the candidate is not in the already opened sectors,
            # put the candidate at the beginning of the list, i.e. it gets higher priority.
            reordered_candidates.insert(0, t)  # put at the beginning of the new list
        else:
            reordered_candidates.append(t)

    return reordered_candidates


def remove_same_sector(open_trades, trade_candidates):
    # remove trade candidates if they are in the same sector as already open positions
    open_sectors = set([t.sector for t in open_trades])
    for t in trade_candidates:
        if t.sector in open_sectors:
            t.deleted = True


def calc_avg_dollar(ser, account_size):
    # calculate yearly performance, no compounding
    total_days = (ser.index[-1] - ser.index[0]).days  # total calendar years
    return (ser.iloc[-1] - ser.iloc[0]) / (total_days / 365.25) / account_size


def calc_sharpe(ser, rf=0):
    rets = ser.pct_change().fillna(0)

    mean = rets.mean() * 252 - rf
    std = rets.std() * np.sqrt(252)
    if std != 0:
        return mean / std
    return 0


def calc_sortino(ser, rf=0):
    rets = ser.pct_change().fillna(0)

    mean = rets.mean() * 252 - rf
    std_neg = rets[rets < 0].std() * np.sqrt(252)
    if std_neg != 0:
        return mean / std_neg
    return 0


def calc_volatility(ser):
    return ser.pct_change().fillna(0).std() * np.sqrt(252)


def calc_max_dd_duration(dd: pd.Series):
    duration = pd.Series(index=dd.index, dtype=float)

    # Loop over the index range
    for t in range(1, len(dd)):
        duration.iloc[t] = 0 if dd.iloc[t] == 0 else duration.iloc[t-1] + 1
    return duration.max()


def calc_trades_per_year(cum_df, total_trades, total_rolls):
    years = (cum_df.index[-1] - cum_df.index[0]).days / 365.25
    return total_trades / years, total_rolls / years,  years


def draw_chart(cumulative_df, title, cfg, table_df):
    # sns.set_style('whitegrid')

    fig, ax = plt.subplot_mosaic('AFE;BFE;CFE;DFE', figsize=(12, 11), constrained_layout=True, width_ratios=[0.85, 0.001, 0.149])
    plt.suptitle(title)

    ax['A'].plot(cumulative_df['Total'], lw=1, label='Total')
    ax['A'].plot(cumulative_df['Total_Long'], lw=0.5, alpha=0.5, color='green', label='Long')
    ax['A'].plot(cumulative_df['Total_Short'], lw=0.5, alpha=0.5, color='red', label='Short')
    ax['A'].set_ylabel('Total $')
    ax['A'].legend()

    # print('Cumulative:')
    # print(tabulate(cumulative_df, headers='keys', tablefmt='psql'))

    log_return = 'log' if cfg.CUMULATIVE else 'linear'

    ax['A'].set_yscale(log_return)    # 'log' 'linear'
    ax['A'].plot([cumulative_df.index[0], cumulative_df.index[-1]],   # line beg..end
               [cumulative_df['Total'].iloc[0], cumulative_df['Total'].iloc[-1]],   # pnl
               'b', lw=0.5, alpha=0.5)

    if cfg.CUMULATIVE:
        dd = cumulative_df['Total'] / cumulative_df['Total'].cummax() - 1
    else:
        dd = (cumulative_df['Total'] - cumulative_df['Total'].cummax()) / cfg.PORTFOLIO_DOLLAR

    ax['B'].plot(dd * 100, lw=1)
    ax['B'].set_ylabel('DD, %')

    ax['C'].plot(cumulative_df['Nr.Positions'], lw=1, label='Total')
    ax['C'].axhline(y=cumulative_df['Nr.Positions'].mean(), lw=1, color='orange', label='Average', linestyle='--')
    ax['C'].plot(cumulative_df['Nr.Positions_Long'], lw=0.5, alpha=0.5, color='green', label='Long')
    ax['C'].plot(cumulative_df['Nr.Positions_Short'], lw=0.5, alpha=0.5, color='red', label='Short')

    ax['C'].set_ylabel('Nr. Positions')

    ax['D'].plot(cumulative_df['Margin'], lw=1)
    ax['D'].axhline(y=cumulative_df['Margin'].mean(), lw=1, color='orange', label='Average', linestyle='--')
    ax['D'].set_ylabel('Margin USD')
    ax['D'].set_xlabel('Date')

    for sp in ['A', 'B', 'C', 'D']:
        ax[sp].grid(which='minor', color='grey', linestyle='-', alpha=0.1)
        ax[sp].grid(which='major', color='grey', linestyle='-', alpha=0.3)
        for axis in ['top', 'bottom', 'left', 'right']:
            ax[sp].spines[axis].set_linewidth(0.5)  # change width
            ax[sp].spines[axis].set_color('grey')  # change color

    # spacer
    ax['F'].patch.set_visible(False)
    ax['F'].axis('off')

    # create table
    # hide the axes
    ax['E'].patch.set_visible(False)
    ax['E'].axis('off')
    # ax['E'].axis('tight')
    # table_df.reset_index(drop=True, inplace=True)
    table_df = table_df.T.reset_index()
    table = ax['E'].table(cellText=table_df.values, colLabels=['Parameter', 'Value'], loc='center')
    table.scale(2, 2)
    # turn off the auto set text so you can set the font size
    # https://curbal.com/curbal-learning-portal/tables-in-matplotlib
    table.auto_set_font_size(False)
    table.set_fontsize(12)

    cell_dict = table.get_celld()
    # cellDict[(0, 0)].set_xy(0, 0)
    for i in range(0, len(table_df.columns)):
        cell_dict[(0, i)].set_height(.06)  # header height
        cell_dict[(0, i)].set_color('#efefef')
        cell_dict[(0, i)].set_linewidth(0.05)
        # cellDict[(0, i)].get_text().set_color('black')  # header font color
        # cellDict[(0, i)].set_edgecolor('#303546')
        for j in range(1, len(table_df) + 1):
            cell_dict[(j, i)].set_height(.03)  # row height
            cell_dict[(j, i)].set_linewidth(0.05)
            if j % 2 == 0:
                cell_dict[(j, i)].set_color('#efefef')   # light grey

    # show negative values in red
    for j in range(1, len(table_df) + 1):
        val = cell_dict[(j, 1)].get_text()
        if '-' in val.get_text():
            cell_dict[(j, 1)].get_text().set_color('red')

    # mpl.cursor(hover=True)

    plt.show(block=False)


def _save_pnl(filename, df):
    df = df['Total']
    df = df.reset_index()
    df.columns = ['timestamp', 'close']
    df['mp'] = 1
    # data_frame.to_csv(filename, index=False, header=True, float_format='%.4f')
    df.to_csv(filename, index=False, header=True, float_format='%g')
    print(f'Data saved to "{filename}"')


def save_pnl(cfg, df):
    if cfg.CUMULATIVE:
        print('Cannot save results from CUMULATIVE backtest')
        return

    for data_dir in [
        'C:/Users/info/Documents/Python/MachineLearning/Src/Tradestation/data',
        'H:/Invest/backtester_tests/data_download',
        'C:/Users/info/Documents/Python/MachineLearning/Src/Tradestation/data'
    ]:
        # filename = f'{data_dir}/my_cta_trend_{cfg.PERIOD}.csv'
        filename = f'{data_dir}/my_cta_trend.csv'
        _save_pnl(filename, df)


def main_affordable_positions():
    # think again what do we want to show here?
    cfg = Config()
    cfg.PORTFOLIO_DOLLAR = 10000_000
    cfg.RISK_POSITION = 0.01       # % of portfolio
    cfg.RISK_ALL_POSITIONS = 0   # % of portfolio
    cfg.MAX_MARGIN = 0             # % of portfolio
    cfg.ATR_MUL = 5
    cfg.PERIOD = 12 * 21             # 9 - good performance
    cfg.START_DATE = '1010-06-01'   # start of data: '1970-01-01'
    cfg.USE_STOP_LOSS = False       # IMPORTANT!
    cfg.CUMULATIVE = False
    cfg.PATCH_MICRO = False
    cfg.MULTI_PROCESSING = True
    cfg.BUY_AND_HOLD = False

    plt.figure(figsize=(12, 11))
    # risk_pct on PORTFOLIO_DOLLAR size
    for risk_pct in [0.5, 1, 2, 5, 50]: #1, 2, 5, 20, 50]:
        print(f'risk: {risk_pct} %')
        cfg.RISK_POSITION = risk_pct / 100
        results = calc_strategies(cfg, verbose=False)
        cumulative_df, stat = portfolio_with_constraints(cfg, results, constraints=True, verbose=False)
        pos_size = position_size_dollar(cfg)
        plt.plot(cumulative_df['Nr.Positions'], label=f'{int(pos_size):,.0f}', lw=1)

    plt.ylabel('Nr.Positions')
    plt.title(f'Number Affordable Positions, account size: {cfg.PORTFOLIO_DOLLAR:,.0f}')
    plt.xlabel('Date')
    plt.legend(title='Position risk, $')
    plt.grid()
    plt.show()


def main_optimize():

    # backtest_range = np.arange(1, 13, 1)
    backtest_range = np.arange(1, 3, 1)
    parameter_name = 'Lookback Months'

    cmap = matplotlib.colormaps['rainbow']    # 'rainbow', 'turbo'
    colors = cmap(np.linspace(0, 1, len(backtest_range)))

    cfg = Config()
    cfg.PORTFOLIO_DOLLAR = 100_000
    cfg.RISK_POSITION = 0.02        # % of portfolio, if RISK_POSITION < 0: trade with +/- 1 contract
    cfg.RISK_ALL_POSITIONS = 0.2    # % of portfolio; 0=don't check
    cfg.MAX_MARGIN = 0            # % of portfolio; 0=don't check margin
    cfg.ATR_MUL = 5                 # 2 or 3 or 5  # !!!
    cfg.PERIOD = 11 * 21            # !!!
    cfg.START_DATE = '1980-01-01'   # start of data: '1970-01-01' (1980-01-01)
    cfg.END_DATE = '3015-04-01'
    cfg.USE_STOP_LOSS = True
    cfg.CUMULATIVE = False
    cfg.PATCH_MICRO = True
    cfg.MULTI_PROCESSING = True
    cfg.SECTOR = ''
    cfg.BUY_AND_HOLD = False
    cfg.COST_CONTRACT = 2
    cfg.SLIPPAGE_TICKS = 2
    cfg.MAX_POSITIONS_PER_SECTOR = 0
    cfg.ORDER_EXECUTION_DELAY = 0

    results_dict = {}
    for idx, parameter_value in enumerate(backtest_range):
        print(f'{parameter_name}: {parameter_value} / {len(backtest_range)}')
        # cfg.ORDER_EXECUTION_DELAY = parameter_value
        cfg.PERIOD = parameter_value * 21
        # cfg.ATR_MUL = parameter_value
        results = calc_strategies(cfg, verbose=False)
        cumulative_df, stat = portfolio_with_constraints(cfg, results, constraints=True, verbose=False)
        results_dict[idx] = [parameter_value, cumulative_df, stat]

    fig, ax = plt.subplots(figsize=(12, 11))

    for idx, res in results_dict.items():
        parameter_value, cumulative_df, stat = res
        plt.plot(cumulative_df['Total'], label=f'{parameter_value}', lw=1, color=colors[idx])

    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    mpl.cursor(hover=True)

    plt.suptitle(f'Performance with different {parameter_name}')
    plt.title(
        f'Portfolio: {cfg.PORTFOLIO_DOLLAR:,.0f}, '
        f'Risk position: {cfg.RISK_POSITION}, '
        f'Risk all positions {cfg.RISK_ALL_POSITIONS}, '
        f'Max margin: {cfg.MAX_MARGIN}, '
        f'Atr mul: {cfg.ATR_MUL}, '
        f'Cumulative: {cfg.CUMULATIVE}'
    )
    plt.ylabel('Performance $')
    plt.xlabel('Date')
    plt.legend(title=parameter_name)
    log_return = 'log' if cfg.CUMULATIVE else 'linear'
    plt.yscale(log_return)
    plt.grid(which='minor', color='grey', linestyle='--', alpha=0.1)
    plt.grid(which='major', color='grey', linestyle='-', alpha=0.2)
    plt.show()


def main_by_sector():
    #
    # NOTE: don't skip sectors and long/short in calc_strategies !!!
    #
    for sector in ['Crypto', 'Currency', 'Energy', 'Volatility', 'Equity', 'Metal', 'Fixed Income', 'Rates',
                   'Grain', 'Soft', 'Meat']:
        cfg = Config()
        cfg.SECTOR = sector
        results = calc_strategies(cfg)
        portfolio_with_constraints(cfg, results, constraints=True)


def main():
    cfg = Config()
    results = calc_strategies(cfg, verbose=True)
    portfolio_with_constraints(cfg, results, constraints=True, verbose=True)
    # plt.show(block=True)  # if running on command line


def main_buy_and_hold():
    cfg = Config()
    cfg.PORTFOLIO_DOLLAR = 100_000
    cfg.RISK_POSITION = 0.02        # % of portfolio, if RISK_POSITION < 0: trade with +/- 1 contract
    cfg.RISK_ALL_POSITIONS = 0      # % of portfolio; 0=don't check
    cfg.MAX_MARGIN = 0              # % of portfolio; 0=don't check margin
    cfg.USE_STOP_LOSS = False       # !!!
    cfg.CUMULATIVE = False
    cfg.PATCH_MICRO = True
    cfg.BUY_AND_HOLD = True
    cfg.COST_CONTRACT = 0
    cfg.SLIPPAGE_TICKS = 0

    results = calc_strategies(cfg)
    portfolio_with_constraints(cfg, results, constraints=True)
    # plt.show(block=True)  # if running on command line


matplotlib.use('QtAgg')
sns.set_style("whitegrid")
# plt.style.use('seaborn-v0_8')
# sns.set_context("talk")    # "paper", "talk", "poster"

if __name__ == '__main__':
    # plt.style.use('classic')

    # if input("Will overwrite existing data are you sure ?! [y/Y]:") not in ['y', 'Y']:
    #     sys.exit()

    with Timer():
        main()
        # main_optimize()
        # main_affordable_positions()
        # main_buy_and_hold()
        # main_by_sector()

#     def calculate_margin(self):
#         last_close = self.data['Close'].iloc[-1]
#         if last_close > 0:
#             default_margin = self.future.margin  # margin for newest price
#             margin_per_dollar = default_margin / last_close
#             self.data['Margin'] = self.data['Close'] * margin_per_dollar
#             pass