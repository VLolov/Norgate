import itertools
from dataclasses import dataclass
from datetime import datetime
from typing import List

import duckdb
import pandas as pd
from tqdm import tqdm

from Futures.DBConfig import DBConfig
from Futures.BacktesterBase import InstrumentBase
from Futures.TrendFollowing.DataAccess import DataAccess
from Futures.TrendFollowing.Future import Future as NorgateFuture
from Futures.TrendFollowing.LoosePants import LoosePants


class Future(InstrumentBase):
    def check_state(self) -> bool:
        return True

    @dataclass
    class MetaDataNew:
        symbol: str = ''
        name: str = ''
        sector: str = ''
        currency: str = ''
        exchange: str = ''
        big_point: float = 0
        margin: float = 0
        tick_size: float = 0

    def __init__(self, symbol, metadata: MetaDataNew, data: pd.DataFrame):
        super().__init__(symbol, data)
        self.metadata: Future.MetaDataNew = metadata
        self.first_date = data.index[0]
        self.last_date = data.index[-1]

    def dates(self) -> List[datetime]:
        return [datetime.fromtimestamp(ts) for ts in self.data.index]


def get_futures(start_date='1020-01-01', end_date='3020-01-01') -> List[Future]:
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

    nr_futures = itertools.count(0)
    for index, future in enumerate(tqdm(futures, desc='Prepare data', colour='green')):
        # if next(nr_futures) <= 50:
        #     continue
        front = 1
        # print("Symbol", future.symbol, "Front", front)

        with duckdb.connect(DBConfig.DUCK_DB, read_only=True) as connection:
            data_access = DataAccess(connection, start_date, end_date)
            data = LoosePants.get_data(data_access, future, front=front)
            # remove duckdb connection, as it cannot be pickled
            future.dta = None

        if future.symbol not in tradable_symbols_1000:
            continue

        if 'Micro' in future.name:
            continue

        if future.sector in ['Meat', 'Volatility']:
            # skipped sectors
            continue

        meta = Future.MetaDataNew()
        attributes = ['symbol', 'name', 'sector', 'currency', 'exchange', 'big_point', 'margin', 'tick_size']
        for attr in attributes:
            value = getattr(future, attr)
            setattr(meta, attr, value)

        future_new = Future(future.symbol, meta, data)
        futures_new.append(future_new)
        min_date = min(min_date, data.index[0])
        max_date = max(max_date, data.index[-1])

    full_date_range = pd.date_range(start=min_date, end=max_date, freq="D")
    trading_days = pd.bdate_range(start=full_date_range.min(), end=full_date_range.max())
    for future_new in tqdm(futures_new, desc='Prepare new data', colour='green'):
        # print("new:", future_new.symbol)
        fd = future_new.data
        fd = fd.reindex(full_date_range)
        fd = fd[fd.index.isin(trading_days)]
        fd.ffill(inplace=True)
        fd.bfill(inplace=True)
        future_new.data = fd
        pass

    return futures_new


if __name__ == "__main__":
    get_futures()
