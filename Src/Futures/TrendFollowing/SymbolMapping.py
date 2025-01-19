"""
    Symbol mapping between Norgate and IB
"""
import math
import os

import duckdb
import pandas as pd
from tabulate import tabulate

from Futures.TrendFollowing.DataAccess import DataAccess
from Futures.DBConfig import DBConfig
from Futures.TrendFollowing.CheckFutures import FutureNorgate

MY_DIRECTORY = os.path.dirname(__file__)


class SymbolMapping:
    def __init__(self):
        excel_file = DBConfig.EXCEL_FILE
        self.data = pd.read_excel(excel_file, sheet_name='Futures')
        pass

    def __repr__(self):
        return tabulate(self.data, headers='keys', tablefmt='psql')

    def find(self, symbol, provider) -> pd.Series:
        found = self.data[(self.data['symbol'] == symbol) & (self.data['provider'] == provider)]
        error_msg = f"Symbol: {symbol}, provider: {provider} not found in excel"
        assert found is not None and len(found) == 1, error_msg
        return found.iloc[0]

    def check_norgate_future(self, future: FutureNorgate):
        # check if parameters in future are the same as in our excel file - for Norgate symbols
        symbol_map = self.find(future.symbol, 'Norgate')
        if symbol_map['in_list']:
            assert symbol_map['symbol_ib'], "no symbol_ib"
            for attr in ['name', 'sector', 'currency', 'exchange', 'big_point', 'tick_size']:
                assert getattr(future, attr) == symbol_map[attr], f"different"
            symbol_micro_ib = symbol_map['symbol_micro_ib']
            if symbol_micro_ib and str(symbol_micro_ib) != 'nan':
                # IB micro futures have to be defined
                self.find(symbol_micro_ib, 'IB')
                # check the definition of the micro future in IB ?


def check_consistent():
    sm = SymbolMapping()
    with duckdb.connect(DBConfig.DUCK_DB, read_only=True) as connection:
        dta = DataAccess(connection)
        futures = FutureNorgate.all_futures(dta)
        for future in futures:
            sm.check_norgate_future(future)


if __name__ == "__main__":
    # main()
    # sm = SymbolMapping()
    # print(sm)
    # sm.find('6A', 'NGx')
    check_consistent()
    pass

