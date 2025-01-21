"""
    Trading strategy:
    https://www.quantitativo.com/p/the-edge-in-trading-ipos-18-annual

    Rules:
    * Define an IPO as any company recently listed (e.g., in the past 90 days);
    * Whenever the stock closes at a new all-time high, buy;
    * Put a profit target order and a stop-loss order on the day you buy.

    Max. simultaneous positions: 5..40
    SL 5%, PT 10% - original Marsten
    Wider may be better.

    IPO list since 2000: https://www.iposcoop.com/scoop-track-record-from-2000-to-present/
        button: Download an Excel spreadsheet...
        filename: SCOOP-Rating-Performance.xls
    Recent IPOs:
        https://stockanalysis.com/ipos/

    My assumption, that the beginning of data = IPO is wrong as I have much more cases per year than the IPOs
    I could try to get the list from the Excel spreadsheet and match the symbols with the data.

    Other links:
    Nasdaq IPO calendar: https://www.nasdaq.com/market-activity/ipos


    My results:
    Performance is not very good: 10-15% per year
    After 2020 in draw down
    The performance in https://www.quantitativo.com/p/the-edge-in-trading-ipos-18-annual is probably not correct.
    I get very simular one if I just get the norgate data, but do not consider the real IPOs.

    Vasko:
    20.09.2024	Initial version
"""
import os
import datetime
import pickle
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib

import ReadExcel
import RecentIPOs
from TradesStatistics import TradesStatistics
from Src.PlotPerformance import PlotPerformance as pp

matplotlib.use("Qt5Agg")

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)


class Config:
    MY_NAME = os.path.basename(__file__)
    DATA_PATH = os.path.dirname(__file__) + "/dataframes.pickle"
    MAX_POSITIONS = 20      # simultaneous positions
    DELAY = 3               # delay before the search of highest close
    MAX_DELAY = 10          # upper limit for the search of highest close
    MAX_DIT = 3 * 21            # max trading days in trade
    STOP_LOSS = -0.50
    PROFIT_TARGET = 0.50
    MIN_ENTRY_PRICE = 0     # min. price on entry


def read_norgate_data(data_path: str) -> pd.DataFrame:
    with open(data_path, 'rb') as f:
        df = pickle.load(f)

    print("data loaded")
    return df


def get_matching_symbols(norgate_df: pd.DataFrame, ipo_df: pd.DataFrame) -> List[str]:
    # Get only symbols that are in both ipo_df and in norgate
    norgate_symbols = norgate_df.columns[6:].tolist()
    ipo_symbols = ipo_df['Symbol'].tolist()
    symbols = []
    for norgate_symbol in norgate_symbols:
        parts = norgate_symbol.split('-')   # delisted symbols look like XXX-202110
        base_symbol = parts[0]
        if base_symbol in ipo_symbols:
            ipo_date = ipo_df[ipo_df['Symbol'] == base_symbol].index[-1].date()
            norgate_date = norgate_df[norgate_symbol].first_valid_index().date()
            # print(norgate_symbol, norgate_date, ipo_date)
            if abs((ipo_date - norgate_date).days) < 3:
                # give some tolerance, as IPO dates in norgate and ipo_df don't always match exactly
                symbols.append(norgate_symbol)
    return symbols


class Strategy:
    def __init__(self, symbol: str, data: pd.Series):
        self._symbol = symbol
        self._data = data.bfill()   # do we allocate too much memory ???
        self._ready = False
        self._in_position = False
        self._prev_close = 0
        self._dit = 0
        self._cum_ret = 0
        self._highest_close = 0
        self._idx = 0
        self._entry_index = 0
        self._entry_price = 0

    def next(self, curr_date: pd.Timestamp, may_open_trade: bool):
        if self._ready or curr_date < self._data.index[0] or curr_date > self._data.index[-1]:
            return False, 0.0

        # don't take last data point
        close_val = self._data.loc[curr_date]
        if self._prev_close == 0:
            # we need one history close before we can start
            self._prev_close = close_val
            return False, 0.0

        daily_return = 0.0

        if self._in_position:
            daily_return = close_val / self._prev_close - 1
            self._cum_ret += daily_return
            self._dit += 1
            if self._cum_ret > Config.PROFIT_TARGET or \
                    self._cum_ret < Config.STOP_LOSS or \
                    self._dit > Config.MAX_DIT:
                self._ready = True
                self._in_position = False
                print(f"\t\t>>Close {self._symbol} pnl: {self._cum_ret:.2f} entry_price: {self._entry_price:.2f} entry_idx: {self._entry_index} dit: {self._dit}")
        else:
            if self._entry_price == 0:
                # we need one history close before we can start
                self._entry_price = close_val

            self._highest_close = max(self._highest_close, close_val)
            if Config.DELAY < self._idx < Config.MAX_DELAY and close_val == self._highest_close and may_open_trade:
                self._entry_index = self._idx
                self._in_position = True

        self._idx += 1
        self._prev_close = close_val

        return self._in_position, daily_return

    def force_close(self):
        self._ready = True
        self._in_position = False

    def is_ready(self):
        return self._ready

    def reset_ready(self):
        self._ready = False

    def cum_return(self):
        return self._cum_ret

    def in_position(self):
        return self._in_position

    def dit(self):
        return self._dit

    def entry_index(self):
        return self._entry_index


def run_strategy(norgate_df, ipo_df):
    # Loop by date.
    # Only MAX_POSITIONS are taken
    # This should be more realistic.

    norgate_df['Counts'] = 0.0

    symbols = get_matching_symbols(norgate_df, ipo_df)
    # symbols = norgate_df.columns[6:].tolist()
    print(f"Number of symbols to trade: {len(symbols)}")
    trades_df = pd.DataFrame(index=norgate_df.index, columns=['ret', 'count', 'daily return'])
    trades_df['ret'] = 0.0  # total return of a closed position
    trades_df['count'] = 0  # number of open positions on that day
    trades_df['daily return'] = 0.0  # sum of daily returns of all trades open on that day

    all_strategies = dict()
    for symbol in symbols:
        all_strategies[symbol] = None

    nr_trades = 0
    trades_return = 0
    trades_statistics = TradesStatistics()

    first_valid = {}
    last_valid = {}

    # leave only symbols with longer history
    # Note: don't require too long history, this may introduce look ahead bias
    reduced_symbols = []
    for symbol in symbols[:]:
        symbol_data = norgate_df[symbol]
        first_timestamp = symbol_data.first_valid_index()
        last_timestamp = symbol_data.last_valid_index()
        if (last_timestamp - first_timestamp).days < Config.DELAY:
            # skip symbols with short history
            # if we don't, the trading strategy leaves open trades
            print("Skip", symbol)
        else:
            reduced_symbols.append(symbol)
            first_valid[symbol] = first_timestamp
            last_valid[symbol] = last_timestamp

    for curr_date in norgate_df.index[:]:   # 6000: - starts end of 2022
        print("*", curr_date)
        open_positions = 0
        cnt = 0

        for symbol in reduced_symbols[:]:
            symbol_data = norgate_df[symbol]
            first_timestamp = first_valid[symbol]
            last_timestamp = last_valid[symbol]
            if pd.to_datetime(first_timestamp) < datetime.datetime(2000, 1, 1, ):
                continue

            strategies_in_position = [s for s in all_strategies.values() if s is not None and s.in_position()]
            open_positions = len(strategies_in_position)

            strategies_ready = [s for s in all_strategies.values() if s is not None and s.is_ready()]

            for strategy in strategies_ready:
                cum_ret = strategy.cum_return()
                trades_statistics.add_trade(cum_ret)
                trades_df.loc[curr_date, 'ret'] += cum_ret / Config.MAX_POSITIONS
                trades_return += cum_ret
                nr_trades += 1
                strategy.reset_ready()  # get cum_return only once

            if first_timestamp <= curr_date <= last_timestamp:
                curr_strategy = all_strategies[symbol]

                if curr_strategy is None \
                        and open_positions < Config.MAX_POSITIONS \
                        and symbol_data.loc[first_timestamp] > Config.MIN_ENTRY_PRICE:
                    # open new position - create a new instance of StrategyByDate
                    selected_symbol_data = symbol_data[first_timestamp:last_timestamp]
                    curr_strategy = Strategy(symbol, selected_symbol_data)
                    all_strategies[symbol] = curr_strategy

                if curr_strategy is not None:
                    may_enter_new_trade = open_positions < Config.MAX_POSITIONS
                    in_position, daily_ret = curr_strategy.next(curr_date, may_enter_new_trade)
                    if in_position:
                        force_close = curr_date >= last_timestamp
                        if force_close:
                            print(f"********** Force close {symbol}")
                            curr_strategy.force_close()

                        trades_df.loc[curr_date, 'daily return'] += daily_ret / Config.MAX_POSITIONS
                        cnt += 1
                        print("\t", symbol, cnt, daily_ret)

        print("\tOpen positions:", open_positions)
        trades_df.loc[curr_date, 'count'] = open_positions

    trades_statistics.print_summary()

    bench = norgate_df['Close'].pct_change().fillna(0).cumsum()
    plt.plot(bench.index, bench.values, label='Benchmark')

    plt.plot(trades_df.index, trades_df['daily return'].cumsum(), label='BacktesterFutures')
    plt.legend()
    plt.title("Performance")
    plt.grid(True)

    plt.figure()
    plt.plot(trades_df.index, trades_df['count'], label='Nr. trades')
    plt.title("Nr. trades")
    plt.grid(True)

    total_ret = trades_df['daily return'].cumsum().iloc[-1]
    print(f"Days in market: {len(trades_df[trades_df['daily return'] != 0])} Total days: {len(trades_df)}")
    print("Average per day:", norgate_df['Counts'].mean())
    print("Number of symbols:", len(symbols))
    print(f"Total return: {total_ret}")
    print(f"Nr.trades: {nr_trades}")
    print("daily return:", total_ret / nr_trades)
    print("Average trade:", trades_return / nr_trades )
    print("Average per year:", total_ret / (len(trades_df) / 252))
    plt.show()

    df_pct_performance_benchmark = norgate_df['Close'].pct_change().fillna(0)
    df_pct_performance = trades_df['daily return']
    benchmark_name = 'S&P 500'
    pp.yearly_plot(f'{Config.MY_NAME}', 'benchmark', df_pct_performance_benchmark,
                   'strategy', df_pct_performance,
                   print_performance=True, show_difference=True)
    total_days = (trades_df.index[-1] - trades_df.index[0]).days  # total calendar days
    trades_per_year = int(np.round(nr_trades / (total_days / 365.25)))
    time_in_market = trades_df['count'].sum() / len(trades_df) / Config.MAX_POSITIONS
    pp.equity_plot(r"$\bf{" + Config.MY_NAME + "}$\n"
                                        f"Nr Trades (full-turn): {nr_trades},"
                                        f" Trades/year: {trades_per_year},"
                                        f" Time in market: {time_in_market * 100:.1f} %, Av.Trade: {trades_return / nr_trades:.3f}\n"
                                        f" MAX_POS: {Config.MAX_POSITIONS}, DELAY: {Config.DELAY}, MAX_DELAY: {Config.MAX_DELAY}, MAX_DIT: {Config.MAX_DIT},\n"
                                        f" SL: {Config.STOP_LOSS}, PT: {Config.PROFIT_TARGET}, MIN_PRICE: {Config.MIN_ENTRY_PRICE}",
                   'benchmark', df_pct_performance_benchmark,
                   'strategy', df_pct_performance,
                   states=trades_df['count'],
                   states_title='Nr.Positions',
                   scale='log')

    del all_strategies  # free-up memory


def print_nr_ipos_by_year(df, ipo_df):
    # symbols in both norgate and ipo list
    symbols = get_matching_symbols(df, ipo_df)

    # symbols = df.columns[6:].tolist()   # norgate symbols
    # symbols = ipo_df['Symbol'].tolist() # IPO symbols from internet

    by_date = {}
    for idx, symbol in enumerate(symbols[:]):
        symbol_data = df[symbol]
        first_timestamp = symbol_data.first_valid_index()
        last_timestamp = symbol_data.last_valid_index()
        year = first_timestamp.to_pydatetime().year
        if year in by_date:
            by_date[year] += 1
        else:
            by_date[year] = 1
        if year == 2024:
            print(first_timestamp, symbol)

    import pprint
    pprint.pprint(dict(sorted(by_date.items())))


def plot_symbols_per_day(norgate_df, ipo_df):
    symbols_df = pd.DataFrame(index=norgate_df.index, columns=['norgate', 'matching', 'ipo'])
    # number symbols starting on that day
    symbols_df['norgate'] = 0
    symbols_df['ipo'] = 0
    symbols_df['matching'] = 0

    # get norgate counts
    norgate_symbols = norgate_df.columns[6:].tolist()
    for symbol in norgate_symbols:
        symbol_data = norgate_df[symbol]
        first_timestamp = symbol_data.first_valid_index()
        symbols_df.loc[first_timestamp, 'norgate'] += 1

    # get ipo counts
    for i in range(len(ipo_df)):
        timestamp = ipo_df.index[i]
        if timestamp in symbols_df.index:
            symbols_df.loc[ipo_df.index[i], 'ipo'] += 1

    # intersection of both norgate and ipo
    matching_symbols = get_matching_symbols(norgate_df, ipo_df)
    for symbol in matching_symbols:
        symbol_data = norgate_df[symbol]
        first_timestamp = symbol_data.first_valid_index()
        symbols_df.loc[first_timestamp, 'matching'] += 1

    # symbols_df = symbols_df.rolling(20).mean()
    symbols_df.plot(alpha=0.8)
    plt.legend()
    plt.title("Norgate symbols by date\n" +
              f"All av.per day: {symbols_df['norgate'].mean():.2f}, " +
              f"Matching av.per day: {symbols_df['matching'].mean():.2f}")
    plt.grid(True)

    # experimental_strategy(norgate_df, symbols_df)


def experimental_strategy(norgate_df, symbols_df):
    #
    # try a trading strategy: use count of ipo's as a signal
    # Reduces the DD until 2012, but afterwords - very high volatility
    #
    benchmark_series = norgate_df['Close'].pct_change()
    # create a dataframe with columns 'norgate', 'matching', 'ipo', 'Close
    merged_df = symbols_df.merge(benchmark_series, how='outer', left_index=True, right_index=True)
    rets = merged_df['ipo'].rolling(10).mean() * merged_df['Close']    # * 0.30

    strategy = pd.concat([benchmark_series, rets], axis=1)
    strategy.columns = ['benchmark', 'strategy']
    strategy.fillna(0, inplace=True)
    strategy.cumsum().plot()
    plt.show()


def current_ips():
    excel_ipo_df = ReadExcel.read_ipo()   # IPO list 2000-2020
    last_ipo_df = RecentIPOs.scrape()   # IPO list 2019-2024
    return pd.concat([excel_ipo_df, last_ipo_df])   # merged IPO list 2000-2024


if __name__ == "__main__":
    data_df = read_norgate_data(Config.DATA_PATH)
    all_ipos_df = current_ips()

    # print_nr_ipos_by_year(data_df, all_ipos_df)
    # plot_symbols_per_day(data_df, all_ipos_df)
    run_strategy(data_df, all_ipos_df)
    del data_df  # free-up memory, so we can keep results of many runs
