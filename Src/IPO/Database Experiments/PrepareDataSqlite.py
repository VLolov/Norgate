"""
    Read data from Norgate and store it a SQLite file
    Norgate databases: 'US Equities' and 'US Equities Delisted'

    A problem: SQLite has no Decimal type, which I want to use for the data.
    The float type is not nice, as prices get rounding errors.
    So I decided to use TEXT type for the prices and convert to float when reading in Dataframe.

    I also installed DBeaver - a GUI interface to different databases (SQLite too) and sqlite3.exe - a CLI client.

    My experience:
    SQLite has advantage over DuckDB that it support simultaneous connections, but it is slower and doesn't have
    Decimal type.
    To accelerate it we can define Indexes. I tried with index on Symbol; it accelerated symbol queries quite a bit.

    Vasko:
    29.09.2024	Initial version
"""

import time
from dataclasses import dataclass
from datetime import datetime, timedelta

import sqlite3
import pandas as pd
import norgatedata
import re
from typing import ClassVar, List


pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)

SQLITE_DB = 'norgate.sqlite'


# constants, don't change
PRICE_ADJUST = norgatedata.StockPriceAdjustmentType.TOTALRETURN
PAD_SETTING = norgatedata.PaddingType.NONE
TIMESERIES_FORMAT = 'pandas-dataframe'  # 'numpy-recarray' or 'pandas-dataframe'


@dataclass
class SelectedData:
    selected_list: ClassVar[List] = list()  # class variable: keeps list of SelectedDate objects
    symbol: str
    data: pd.DataFrame


def insert_df_into_db(symbol, df, conn, curs):
    dfx = df.copy()
    dfx.insert(0, 'Symbol', symbol)
    for col in ['Open', 'High', 'Low', 'Close', 'Unadjusted Close']:
        # dfx[col] = dfx[col].apply(lambda x: '{0:.0f}'.format(x*10000))
        dfx[col] = dfx[col].apply(lambda x: '{0:.5f}'.format(x))
    # dfx = df.reset_index(inplace=True)  # .drop(['index'], axis=1)
    # dfx = dfx.drop(['index'], axis=1)

    try:
        dfx.to_sql("STOCKSDATA", conn, if_exists="append")
    except Exception as e:
        print("Exception", str(e))
        print("Symbol:", symbol)

    # conn.commit()


def is_delisted_symbol(symbol):
    # match pattern XXX-DDDDDD, example ZWRK-202213
    return re.match(r"^[A-Z]+-\d{6}$", symbol) is not None


def is_alphanumeric(symbol):
    return symbol.isalpha()


def create_table(conn, curs):
    curs.execute("DROP TABLE IF EXISTS STOCKSDATA")
    curs.execute("""
    CREATE TABLE IF NOT EXISTS STOCKSDATA (
        Date TIMESTAMP NOT NULL,
        Symbol VARCHAR NOT NULL,
        Open TEXT,
        High TEXT,
        Low TEXT,
        Close TEXT,
        Volume  BIGINT,
        Turnover DOUBLE,
        "Unadjusted Close" FLOAT,
        Dividend  FLOAT,
        PRIMARY KEY (Date, Symbol)
    )
    """)
    conn.commit()


def process_benchmark(conn, curs):
    benchmark_df = norgatedata.price_timeseries(
        '$SPX',
        stock_price_adjustment_setting=PRICE_ADJUST,
        padding_setting=PAD_SETTING,
        # end_date=last_quote_date,
        timeseriesformat='pandas-dataframe',
    )

    benchmark_df_x = benchmark_df.copy()
    benchmark_df_x["Unadjusted Close"] = benchmark_df["Close"]
    benchmark_df_x["Dividend"] = 0

    insert_df_into_db('$SPX', benchmark_df_x, conn, curs)


def process_symbols(conn, curs):
    symbols = norgatedata.database_symbols('US Equities')
    symbols_delisted = norgatedata.database_symbols('US Equities Delisted')
    print("Nr. symbols:", len(symbols), "Nr. delisted symbols:", len(symbols_delisted))

    symbols.extend(symbols_delisted)

    processed_symbols = 0

    for symbol in symbols[:]:
        # if not is_alphanumeric(symbol) and not is_delisted_symbol(symbol):
        #     print(f"Skipping non-alpha symbol: {symbol}")
        #     continue

        first_quote_date = norgatedata.first_quoted_date(symbol, datetimeformat='datetime')

        if first_quote_date is None:
            print(f"Skipping {symbol} - no first quote date")
            continue

        symbol_df = norgatedata.price_timeseries(
            symbol,
            stock_price_adjustment_setting=PRICE_ADJUST,
            padding_setting=PAD_SETTING,
            # @@@ start_date=Config.BEG_DATE,
            # end_date=last_quote_date,     # load everything to the end of data
            timeseriesformat=TIMESERIES_FORMAT,
        )

        if symbol_df is None:
            print(f"Skipping {symbol} - dataframe is None")
            continue

        insert_df_into_db(symbol, symbol_df, conn, curs)

        processed_symbols += 1
        print(">>>", symbol, len(symbol_df), first_quote_date)

    print(f"Number of symbols: {len(symbols)}, processed symbols: {processed_symbols}")
    print(f"Executed in {round((time.time() - start), 4)} seconds")


if __name__ == "__main__":
    assert norgatedata.status(), "Norgatedata (NDU) not running"

    start = time.time()

    connection = sqlite3.connect(SQLITE_DB)  # or ':memory:;
    cursor = connection.cursor()

    create_table(connection, cursor)
    process_benchmark(connection, cursor)
    process_symbols(connection, cursor)

    df = pd.read_sql_query('select * from STOCKSDATA', connection)

    df['Symbol'] = df['Symbol'].astype(str)
    for col in ['Open', 'High', 'Low', 'Close', 'Unadjusted Close']:
        df[col] = df[col].astype(float)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    connection.commit()
    cursor.close()
    connection.close()

    print(f"Executed in {round((time.time() - start), 4)} seconds")

# conn = duckdb.connect(SQLITE_DB
# df = conn.query("select * from STOCKSDATA where Symbol='AAPL' order by Date asc limit 1").to_df() # get first date
# print(df)

