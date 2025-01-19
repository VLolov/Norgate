"""
    Read data from Norgate and store it a DuckDB file
    Norgate databases: 'US Equities' and 'US Equities Delisted'

    My experience:
    DuckDB is very easy to setup and to use. Especially when just reading/writing data in Dataframes.
    The only obvious problem is having a single connection/tread to the database.

    Vasko:
    29.09.2024	Initial version
"""
import os
import time

import duckdb
import pandas as pd
import norgatedata
import re


DUCK_DB = './norgate.duckdb'
DB_LOCK = True          # True=don't perform any database operations - protect database

# constants, don't change
PRICE_ADJUST = norgatedata.StockPriceAdjustmentType.TOTALRETURN
PAD_SETTING = norgatedata.PaddingType.NONE
TIMESERIES_FORMAT = 'pandas-dataframe'  # 'numpy-recarray' or 'pandas-dataframe'


def insert_df_into_db(symbol, df, conn):
    if conn is None:
        return
    dfx = df.reset_index()
    dfx.insert(0, 'Symbol', symbol)

    try:
        # Dataframe => database
        conn.execute("INSERT INTO STOCKSDATA SELECT * FROM dfx")
    except Exception as e:
        print("Exception", str(e))
        print("Symbol:", symbol)


def is_delisted_symbol(symbol):
    # match pattern XXX-DDDDDD, example ZWRK-202213
    return re.match(r"^[A-Z]+-\d{6}$", symbol) is not None


def is_alphanumeric(symbol):
    return symbol.isalpha()


def create_table(conn):
    if conn is None:
        return
    conn.execute("""
    CREATE OR REPLACE TABLE STOCKSDATA (
        Symbol VARCHAR,
        Date TIMESTAMP,
        Open DECIMAL(19,6),
        High DECIMAL(19,6),
        Low DECIMAL(19,6),
        Close DECIMAL(19,6),
        Volume  BIGINT,
        Turnover DOUBLE,
        "Unadjusted Close" DECIMAL(19,6),
        Dividend  DECIMAL(19,6)
    );
    """)


def process_benchmark(conn):
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

    insert_df_into_db('$SPX', benchmark_df_x, conn)


def process_symbols(conn):
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

        # seems to return None if data is still available today
        last_quote_date = norgatedata.last_quoted_date(symbol, datetimeformat='datetime')

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

        insert_df_into_db(symbol, symbol_df, conn)

        processed_symbols += 1
        first_date = str(first_quote_date.date())
        last_date = str(last_quote_date.date()) if last_quote_date is not None else ""
        print(">>>", symbol, len(symbol_df), first_date, last_date)

    print(f"Number of symbols: {len(symbols)}, processed symbols: {processed_symbols}")
    print(f"Executed in {round((time.time() - start), 4)} seconds")


def delete_database():
    # we delete previous database explicitly, may be this is not quite necessary, but just in case...
    if os.path.exists(DUCK_DB):
        os.remove(DUCK_DB)
    if os.path.exists(DUCK_DB + ".wal"):
        os.remove(DUCK_DB + ".wal")


if __name__ == "__main__":
    assert norgatedata.status(), "Norgatedata (NDU) not running"

    start = time.time()

    connection = None
    if not DB_LOCK:
        delete_database()
        connection = duckdb.connect(DUCK_DB)    # this will create a new database file if it doesn't exist

    create_table(connection)
    process_benchmark(connection)
    process_symbols(connection)

    if connection is not None:
        connection.close()

    print(f"Executed in {round((time.time() - start), 4)} seconds")

# # Run in python console:
# import duckdb
# import pandas as pd
#
# conn = duckdb.connect('norgate.duckdb')
# df = conn.query("select * from STOCKSDATA where Symbol='AAPL' order by Date asc limit 1").to_df() # get first date
# print("first date:\n", df)
#
# df = conn.query("select * from STOCKSDATA where Symbol='AAPL' order by Date desc limit 1").to_df() # get last date
# print("last date:\n", df)
