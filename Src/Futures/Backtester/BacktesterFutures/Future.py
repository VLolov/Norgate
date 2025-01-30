from dataclasses import dataclass
from datetime import datetime
from typing import List

import duckdb
import pandas as pd
from tqdm import tqdm

from Futures.DBConfig import DBConfig
from Futures.Backtester.BacktesterBase import InstrumentBase
from Futures.TrendFollowing.DataAccess import DataAccess
from Futures.TrendFollowing.Future import Future as NorgateFuture
from Futures.TrendFollowing.LoosePants import LoosePants


class Future(InstrumentBase):
    def check_state(self) -> bool:
        return True

    @dataclass
    class MetaData:
        symbol: str = ''
        name: str = ''
        sector: str = ''
        currency: str = ''
        exchange: str = ''
        big_point: float = 0
        margin: float = 0
        tick_size: float = 0

    def __init__(self, symbol, metadata: MetaData, data: pd.DataFrame):
        super().__init__(symbol, data)
        self.metadata: Future.MetaData = metadata
        self.first_date = data.index[0]
        self.last_date = data.index[-1]
        self.data_numpy = None
        cols = self.data.columns.values.tolist()
        self.OPEN = cols.index('Open')
        self.HIGH = cols.index('High')
        self.LOW = cols.index('Low')
        self.CLOSE = cols.index('Close')
        self.VOLUME = cols.index('Volume')
        # numpy access:

    def dates(self) -> List[datetime]:
        return [datetime.fromtimestamp(ts) for ts in self.data.index]

    def __repr__(self):
        return (f"<{self.__class__.__name__} id: {self.id}, "
                f"symbol: {self.symbol}, "
                f"first_date: {self.first_date}, "
                f"last_date: {self.last_date}, "
                f"data_len: {len(self.data)}>")


def get_futures(start_date='1020-01-01', end_date='3020-01-01', selected_symbols:List[str]=[]) -> List[Future]:
    tradable_symbols_1000 = ['6A', '6B', '6C', '6E', '6J', '6M', 'BTC', 'CC', 'CL', 'CT', 'DC', 'DX',
                             'EMD', 'ES', 'ETH', 'FCE', 'FDAX', 'FESX', 'FGBL', 'FGBM', 'FGBS', 'FOAT', 'FTDX',
                             'GC', 'HE', 'HG', 'HTW', 'KE', 'LEU', 'LRC', 'LSU', 'NG', 'NKD', 'NQ', 'RTY', 'SB',
                             # 'MWE' - couldnt find in IB
                             'SCN', 'SI', 'SR3', 'TN', 'UB', 'VX', 'YM', 'ZC', 'ZF', 'ZL', 'ZN', 'ZO', 'ZQ', 'ZR', 'ZS',
                             'ZT', 'ZW']

    futures = NorgateFuture.all_futures_norgate(use_micro=True)
    futures_new = []
    min_date = pd.Timestamp.max
    max_date = pd.Timestamp.min

    nr_futures = 0

    with duckdb.connect(DBConfig.DUCK_DB, read_only=True) as connection:
        for index, future in enumerate(tqdm(futures, desc='Prepare data', colour='green')):
            # if next(nr_futures) > 3:
            #     continue
            front = 0
            # print("Symbol", future.symbol, "Front", front)

            # database operation
            data_access = DataAccess(connection, start_date, end_date)
            data = LoosePants.get_data(data_access, future, front=front)
            # remove duckdb connection, as it cannot be pickled
            future.dta = None

            if selected_symbols:
                if future.symbol not in selected_symbols:
                    continue

            if future.symbol not in tradable_symbols_1000:
                continue

            if 'Micro' in future.name:
                continue

            if future.sector in ['Meat', 'Volatility']:
                # skipped sectors
                continue

            meta = Future.MetaData()
            attributes = ['symbol', 'name', 'sector', 'currency', 'exchange', 'big_point', 'margin', 'tick_size']
            for attr in attributes:
                value = getattr(future, attr)
                setattr(meta, attr, value)

            future_new = Future(future.symbol, meta, data)
            futures_new.append(future_new)
            min_date = min(min_date, data.index[0])
            max_date = max(max_date, data.index[-1])

            nr_futures += 1

    print(f"Nr. futures to be traded: {nr_futures}")

    # full_date_range = pd.date_range(start=min_date, end=max_date, freq="B")     # "B" = business day
    # trading_days = pd.bdate_range(start=full_date_range.min(), end=full_date_range.max())
    # for future in tqdm(futures_new, desc='Prepare futures data', colour='green'):
    #     # print("new:", future_new.symbol)
    #     future_data = future.data
    #     future_data = future_data.reindex(full_date_range)
    #     # @@@ future_data = future_data[future_data.index.isin(trading_days)]
    #     future_data.ffill(inplace=True)
    #     future_data.bfill(inplace=True)
    #     # IMPORTANT: replace future data of original Future, but leave first/last date unchanged !!!
    #     future.data = future_data
    #     # future.data_numpy = future.data[['Open', 'High', 'Low', 'Close']].to_numpy()
    #     future.data_numpy = future.data.to_numpy()

    full_date_range = pd.date_range(start=min_date, end=max_date, freq="B")     # "B" = business day

    trading_days = pd.bdate_range(start=full_date_range.min(), end=full_date_range.max())
    for future in tqdm(futures_new, desc='Prepare futures data', colour='green'):
        # print("new:", future_new.symbol)
        future_data = future.data
        future_data = future_data.reindex(full_date_range)
        # NOTE: Remove time part of index! If not done, the chart of single instrument gets messed-up ?!?
        future_data.index = future_data.index.date
        # future_data = future_data[future_data.index.isin(full_date_range)]
        future_data.ffill(inplace=True)
        future_data.bfill(inplace=True)
        # IMPORTANT: replace future data of original Future, but leave first/last date unchanged !!!
        future.data = future_data
        # future.data_numpy = future.data[['Open', 'High', 'Low', 'Close']].to_numpy()
        future.data.index = pd.to_datetime(future.data.index)      # restore datetime index
        future.data_numpy = future.data.to_numpy()

    return futures_new


if __name__ == "__main__":
    get_futures()
