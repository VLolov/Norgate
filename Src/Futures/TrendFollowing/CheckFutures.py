import datetime
from typing import Optional, List

import duckdb

from DataAccess import DataAccess
from Futures.DBConfig import DBConfig
from Future import Future


class FutureNorgate:
    def __init__(self, data_access: Optional[DataAccess] = None,
                 symbol: Optional[str] = None,
                 convert_to_usd: bool = True):
        self.data_access: DataAccess = data_access
        self.symbol: str = ''
        self.name: str = ''
        self.sector: str = ''
        self.currency: str = ''
        self.exchange: str = ''
        self.big_point: float = 0
        self.margin: float = 0
        self.tick_size: float = -1

        if data_access is not None and symbol is not None:
            # initialize Symbol only if both parameters are provided
            df = data_access.conn.sql(f"SELECT * FROM Futures WHERE Symbol='{symbol}'").to_df()
            assert len(df) == 1, f'No Future with symbol: "{symbol}"'
            self.symbol, self.name, self.sector, self.currency, self.exchange, self.big_point, self.margin = \
                df[['Symbol', 'Name', 'Sector', 'Currency', 'Exchange', 'PointValue', 'Margin']].iloc[0]

            if convert_to_usd:
                self.margin = self.data_access.convert_float_to_usd(self.margin, self.currency)

            self.tick_size = self._get_tick_size(symbol)

    def _get_tick_size(self, symbol):
        now = datetime.datetime.now()
        year_month = now.year * 100 + now.month
        sql = f"""SELECT TickSize
                 FROM FuturesContracts
                 WHERE Symbol = '{symbol}' AND DeliveryMonth <= {year_month}
                 ORDER BY DeliveryMonth DESC LIMIT 1"""
        # this returns the original database type. In this case - Decimal type, which has to be converted: float(f[0][0]
        # f = self.conn.sql(sql).fetchall()
        # return f[0][0]
        df = self.data_access.conn.sql(sql).to_df()
        assert len(df) == 1, 'No recent data in database'
        return df.iloc[0]['TickSize']

    def __str__(self):
        return f'Future - Symbol: {self.symbol}, Name: {self.name}, Sector: {self.sector}, Currency: {self.currency}, ' \
                f'Exch: {self.exchange}, Big point: {self.big_point}, Margin: {self.margin}, Tick size: {self.tick_size}'

    @classmethod
    def all_futures(cls, da: DataAccess, convert_to_usd: bool = True) -> List:
        # return all futures as a list of Future objects
        futures: List[FutureNorgate] = []
        df = da.conn.sql(f"SELECT Symbol FROM Futures").to_df()
        assert len(df) > 0, "Error constructing list of Future objects"
        for symbol in df['Symbol'].values:
            f = FutureNorgate(da, symbol, convert_to_usd)
            futures.append(f)
            f.data_access = None    # needed for multiprocessing, as duck db connection can not be pickled
        return futures


def compare_futures(future_norgate: FutureNorgate, future: Future):
    different = 0
    attributes = ['symbol', 'name', 'sector', 'currency', 'exchange', 'big_point', 'tick_size', 'margin']
    for attr in attributes:
        if getattr(future_norgate, attr) != getattr(future, attr):
            print(f"Different attribute: {attr}")
            print("\t", future_norgate)
            print("\t", future)
            different += 1
    return different


def main():
    print("Check differences:")
    DUCK_DB = DBConfig.DUCK_DB
    with duckdb.connect(DUCK_DB, read_only=True) as conn:
        dta = DataAccess(conn)
        # [print(i, f) for i, f in enumerate(FutureNorgate.all_futures(dta))]
        futures_norgate = FutureNorgate.all_futures(dta)
    different = 0
    for future_norgate in futures_norgate:
        future = Future.get_future(symbol=future_norgate.symbol, provider='Norgate')
        different += compare_futures(future_norgate, future)

    if different == 0:
        print("No differences")
    else:
        print("*** DIFFERENCES FOUND ***")


if __name__ == "__main__":
    main()

