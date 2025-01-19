# import datetime as datetime
# from typing import List, Optional
#
# import duckdb
import pandas as pd

# from ..DBConfig import DBConfig
from Futures.TrendFollowing.Future import Future

class DataAccess:
    def __init__(self, connection, start_date='1900-01-01', end_date='2050-01-01'):
        self.conn = connection
        self.start_date = start_date    # data starts 1970
        self.end_date = end_date
        self.forex_cache = {}

    def _get_dataframe(self, sql):
        df = self.conn.sql(sql).to_df()
        df.set_index('Date', inplace=True)
        return df

    def futures(self):
        df = self.conn.sql(f"SELECT * FROM Futures ORDER by Symbol").to_df()
        return df

    def _continuous_contract(self, symbol, adjusted, front):
        if front > 0:
            # "our" continuous contract
            df = self._get_dataframe(
                f"SELECT * FROM AllContracts WHERE Symbol='{symbol}'"
                f"AND Date > '{self.start_date}' AND Date < '{self.end_date}' " 
                f"AND Front={front} ORDER BY Date")

            df.drop(['ContractSymbol', 'Open', 'High', 'Low', 'Close'], axis=1, inplace=True)
            df.rename(columns={'AdjOpen': 'Open', 'AdjHigh': 'High', 'AdjLow': 'Low', 'AdjClose': 'Close'},
                      inplace=True)
        else:
            # Norgate's continuous contract
            df = self._get_dataframe(
                f"SELECT * FROM ContContracts WHERE Symbol='{symbol}' "
                f"AND Adjusted={adjusted} AND Date > '{self.start_date}' AND Date < '{self.end_date}' ORDER BY Date")
        assert len(df) > 200, f'Date for: "{symbol}" too short: {len(df)}'
        return df

    def continuous_contract_adjusted(self, symbol, front):
        # front > 0  - return data from AllContracts (our continuous contracts)
        # front = 0 - return Norgate's continuous contract
        df = self._continuous_contract(symbol, 1, front)
        return df
    
    def continuous_contract_not_adjusted(self, symbol):
        df = self._continuous_contract(symbol, 0, 0)
        return df

    def continuous_contract_dates(self):
        df = self._get_dataframe(f"SELECT DISTINCT Date FROM ContContracts "
                                 f"WHERE Date > '{self.start_date}' AND  Date < '{self.end_date}' ORDER BY Date")
        # remove weekends - no effect, we don't have weekends already
        # dfx = df[df.index.dayofweek < 5]    # monday=0,... sun=6
        # dfy = df[df.index.dayofweek >= 5]
        return df

    def convert_to_usd(self, df, symbol):
        # Not Used
        raise NotImplementedError
        # convert OHLC data in df from 'currency' to USD using the exchange rates in table Forex
        currency = Future(self, symbol).currency     # get original currency of symbol, e.g. EUR
        # Convert prices into USD
        if currency != "USD":
            spot_currency = self._get_spot(currency)
            for column in ['Open', 'High', 'Low', 'Close']:
                df[column] *= spot_currency['Close']

    def _get_spot(self, currency) -> pd.DataFrame:
        # get forex data with cache
        spot_symbol = currency + "USD"
        if spot_symbol not in self.forex_cache:
            # read the whole data for spot_symbol and cache it
            spot_currency = self._get_dataframe(
                f"SELECT Date, Close FROM Forex WHERE ForexSymbol='{spot_symbol}' ORDER BY Date")
            self.forex_cache[spot_symbol] = spot_currency  # write to cache

        spot_currency = self.forex_cache[spot_symbol]  # read from cache

        return spot_currency

    def convert_float_to_usd(self, num, currency, timestamp=None):
        # Convert num from currency to USD
        # If timestamp is not provided,  use the newest date in Forex table

        spot_currency = None
        exchange_rate = 1

        if currency != "USD":
            spot_currency = self._get_spot(currency)

            if timestamp is None:
                # take last exchange rate
                exchange_rate = spot_currency['Close'].iloc[-1]
            else:
                # as we are not sure that Forex table contains exactly the timestamp (timestamp may be a weekend of holiday),
                # we have to look up the days around
                series = spot_currency.iloc[spot_currency.index.get_loc(timestamp, method='nearest')]
                exchange_rate = series['Close'].iloc[-1]

        return num * exchange_rate

    def index(self, symbol):
        df = self._get_dataframe(f"SELECT * FROM Index WHERE Symbol='{symbol}' ORDER BY Date").to_df()
        return df


