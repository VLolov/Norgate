"""
    Read futures data from Norgate and store it a DuckDB file
    Norgate databases: 'Futures', 'Futures Continuous' and 'Forex Spot'

    Vasko:
    11.10.2024	Initial version

    => read benchmarks:
    Norgate Database US Indices
        $SPXTR	S&P 500 Total Return Index
        $NDXTTR Nasdaq-100 Technology Sector Total Return Index
        $RUTTR	Russell 2000 Total Return Index

    Norgate Database Cash Commodities:
        $BCOMTR     Bloomberg Commodity Total Return Index
        $CRB	    FTSE/CoreCommodity CRB Index
        $SPGSCITR	S&P GSCI Total Return Index

    norgatedata.security_name('$SPXTR')
"""
import os
import re
from dataclasses import dataclass
import pandas as pd

import duckdb
import norgatedata as norgate
from tqdm import tqdm
from Futures.TrendFollowing.Timer import Timer


@dataclass
class Config:
    DUCK_DB = os.path.dirname(__file__) + '/../norgate_futures.duckdb'
    VERBOSE = False


def create_tables(conn):
    if conn is None:
        return
    conn.execute("""
        DROP TABLE IF EXISTS Futures;
        CREATE TABLE Futures (
            Symbol VARCHAR,
            Name VARCHAR,
            Sector VARCHAR,
            Currency VARCHAR,
            Exchange VARCHAR,
            PointValue DECIMAL(19,6),
            Margin DECIMAL(19,6),
            PRIMARY KEY(Symbol)
        );

        DROP TABLE IF EXISTS FuturesContracts;
        CREATE TABLE FuturesContracts (
            Symbol VARCHAR, -- base symbol like 'ES'
            ContractSymbol VARCHAR, -- symbol like 'ES-1998M'
            MarketName VARCHAR,
            SessionName VARCHAR,
            SessionType VARCHAR,    -- not needed?
            FirstNoticeDate TIMESTAMP,
            DeliveryMonth INT,
            TickSize DECIMAL(19,6),
            LowestTickSize DECIMAL(19,6),  -- not needed?
            PointValue DECIMAL(19,6),
            Margin DECIMAL(19,6),
            PRIMARY KEY(Symbol, ContractSymbol)
        );

        DROP TABLE IF EXISTS ContContracts;
        CREATE TABLE ContContracts (
            Symbol VARCHAR,
            Adjusted TINYINT,
            Date TIMESTAMP,
            Open DECIMAL(19,6),
            High DECIMAL(19,6),
            Low DECIMAL(19,6),
            Close DECIMAL(19,6),
            Volume  BIGINT,
            DeliveryMonth INT,
            OpenInterest BIGINT,
            PRIMARY KEY(Symbol, Adjusted, Date)
        );

        DROP TABLE IF EXISTS Contracts;
        CREATE TABLE Contracts (
            ContractSymbol VARCHAR,
            Symbol VARCHAR,
            Date TIMESTAMP,
            Open DECIMAL(19,6),
            High DECIMAL(19,6),
            Low DECIMAL(19,6),
            Close DECIMAL(19,6),
            Volume  BIGINT,
            OpenInterest BIGINT,
            PRIMARY KEY(ContractSymbol, Date)
        );

        DROP TABLE IF EXISTS Forex;
        CREATE TABLE Forex (
            ForexSymbol VARCHAR,
            Date TIMESTAMP,
            Open DECIMAL(19,6),
            High DECIMAL(19,6),
            Low DECIMAL(19,6),
            Close DECIMAL(19,6),
            PRIMARY KEY(ForexSymbol, Date)
        );
        
        DROP TABLE IF EXISTS Index;
        CREATE TABLE Index (
            Symbol VARCHAR,
            Name VARCHAR,
            Date TIMESTAMP,
            Open DECIMAL(19,6),
            High DECIMAL(19,6),
            Low DECIMAL(19,6),
            Close DECIMAL(19,6),
            Volume  BIGINT,
            Turnover DOUBLE,
            PRIMARY KEY(Symbol, Date)
        );

    """)


def check_exclude(symbol):
    # exclude these symbols - they have no continuous contracts in norgate. I don't know why
    return symbol in ['BAX', 'LES', 'LSS', 'EUA']


def symbol_to_sector(symbol):
    sectors = {
        'Crypto': ['BTC', 'ETH', 'MBT', 'MET'],
        'Currency': ['6A', '6B', '6C', '6E', '6J', '6M', '6N', '6S', 'DX'],
        'Energy': ['BRN', 'CL', 'GAS', 'GWM', 'HO', 'NG', 'RB', 'WBS'],
        'Volatility': ['VX'],

        'Equity': [
            'EMD', 'ES', 'FCE', 'CAC 40', 'FDAX', 'FESX', 'FSMI', 'FTDX',
            'HSI', 'HTW', 'KOS', 'LFT', 'M2K', 'MES', 'MHI', 'MNQ',
            'MYM', 'NIY', 'NKD', 'NQ', 'RTY', 'SCN', 'SNK', 'SSG',
            'SXF', 'YAP', 'YM',
            'VX',
            'GD'  # note: this is S&P GSCI - commodity index futures quotes
        ],
        'Metal': ['GC', 'HG', 'PA', 'PL', 'SI'],
        'Fixed Income': [
            'CGB', 'FBTP', 'FGBL', 'FGBM', 'FGBS', 'FGBX',
            'FOAT', 'LLG', 'LSS', 'SJB', 'TN', 'UB', 'YIB',
            'YIR', 'YXT', 'YYT',
            'ZB', 'ZF', 'ZN', 'ZQ', 'ZT'
        ],
        'Rates': ['BAX', 'CRA', 'LES', 'LEU', 'SO3', 'SR3'],
        'Grain': ['AWM', 'AFB', 'KE', 'LWB', 'MWE', 'ZC', 'ZL', 'ZM', 'ZO', 'ZR', 'ZS', 'ZW'],
        'Soft': ['CC', 'CT', 'DC', 'KC', 'LBR', 'LCC', 'LRC', 'LSU', 'OJ', 'RS', 'SB'],
        'Meat': ['GF', 'HE', 'LE'],
    }

    for i, (sector, symbols) in enumerate(sectors.items()):
        if symbol in symbols:
            return sector

    msg = f"No sector found for symbol: {symbol}"
    raise ValueError(msg)
    # return ("*** no sector ***")


def process_futures(conn):
    # all futures
    futures = norgate.futures_market_symbols()
    rows = []
    for future in tqdm(futures, desc="Futures", colour='green'):

        if check_exclude(future):
            continue

        # take the last one, it does not matter which
        contract = norgate.futures_market_session_contracts(future)[-1]

        market_name = norgate.futures_market_name(future)
        currency = norgate.currency(contract)
        exchange = norgate.exchange_name(contract)
        point_value = norgate.point_value(contract)
        margin = norgate.margin(contract)

        sector = symbol_to_sector(future)

        row = {'Symbol': future, 'Name': market_name, 'Sector': sector, 'Currency': currency,
               'Exchange': exchange, 'PointValue': point_value, 'Margin': margin}
        rows.append(row)

    df = pd.DataFrame(rows)
    df.reset_index(drop=True)
    insert_df_into_db(conn, info="", df=df, table_name='Futures')


def extract_delivery_month(symbol):
    # from symbol (like 'ES-1999M') return delivery month 199906
    # see delivery month codes:
    #    https://www.cmegroup.com/education/courses/introduction-to-futures/understanding-contract-trading-codes.html

    contract_code = {
        'F': 1,  # January
        'G': 2,  # February
        'H': 3,  # March
        'J': 4,  # April
        'K': 5,  # May
        'M': 6,  # June
        'N': 7,  # July
        'Q': 8,  # August
        'U': 9,  # September
        'V': 10,  # October
        'X': 11,  # November
        'Z': 12  # December
    }

    result = ''
    try:
        re_match = re.match(r".+-(?P<year>\d{4})(?P<month>[FGHJKMNQUVXZ])$", symbol)
        year = re_match.group('year')
        month_code = re_match.group('month')
        result = f'{year}{contract_code[month_code]:02d}'
    except Exception as e:
        print(f"Exception extracting delivery month from {symbol}", str(e))
    return int(result)


def process_futures_contracts(conn):
    futures = norgate.futures_market_symbols()

    rows = []

    for future in tqdm(futures, desc="Single Contracts", colour='green'):
        if check_exclude(future):
            continue

        contracts = norgate.futures_market_session_contracts(future)
        for contract in contracts:
            tick_size = norgate.tick_size(contract)
            point_value = norgate.point_value(contract)
            margin = norgate.margin(contract)
            first_notice_date = norgate.first_notice_date(contract, datetimeformat='datetime')
            lowest_ever_tick_size = norgate.lowest_ever_tick_size(contract)
            session_type = norgate.session_type(contract)
            market_name = norgate.futures_market_name(contract)  # can put 'ES' here too
            session_name = norgate.futures_market_session_name(contract)
            delivery_month = extract_delivery_month(contract)

            row = {
                'Symbol': future,
                'ContractSymbol': contract,
                'MarketName': market_name,
                'SessionName': session_name,
                'SessionType': session_type,
                'FirstNoticeDate': first_notice_date,
                'DeliveryMonth': delivery_month,
                'TickSize': tick_size,
                'LowestTickSize': lowest_ever_tick_size,
                'PointValue': point_value,
                'Margin': margin,
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    df.reset_index(drop=True)

    insert_df_into_db(conn, info='', df=df, table_name='FuturesContracts')


def insert_df_into_db(conn, *, info, df, table_name):

    if Config.VERBOSE:
        if info:
            print(f"Inserting into table {table_name}: {info}, records: {len(df)}")
        else:
            print(f"Inserting into table {table_name}, records: {len(df)}")

    if conn is not None:
        try:
            # Dataframe => database
            statement = f'INSERT INTO "{table_name}" BY NAME SELECT * FROM df'
            conn.execute(statement)
        except Exception as e:
            conn.close()
            print(f"Exception inserting into table {table_name}", str(e))


def process_cont_contracts(conn):
    futures = norgate.futures_market_symbols()

    for future in tqdm(futures, desc="Continuous Contracts", colour='green'):

        if check_exclude(future):
            continue

        # continuous and continuous back-adjusted contracts
        for i, db_symbol in enumerate([f'&{future}', f'&{future}_CCB']):
            # i=0: not adjusted; i=1: adjusted
            df = get_prices(db_symbol)
            df.rename(columns={'Open Interest': 'OpenInterest', 'Delivery Month': 'DeliveryMonth'}, inplace=True)
            df = df.reset_index()  # keep original df unchanged as a good practice
            df.insert(0, 'Adjusted', i)
            df.insert(0, 'Symbol', future)
            insert_df_into_db(conn, info=db_symbol, df=df, table_name='ContContracts')

        contract_symbols = norgate.futures_market_session_contracts(future)
        for contract_symbol in contract_symbols:
            df = get_prices(contract_symbol)
            if df is not None:
                df.rename(columns={'Open Interest': 'OpenInterest'}, inplace=True)

                df = df.reset_index()  # keep original df unchanged as a good practice
                df.insert(0, 'Symbol', future)
                df.insert(0, 'ContractSymbol', contract_symbol)
                insert_df_into_db(conn, info=contract_symbol, df=df, table_name='Contracts')


def process_forex(conn):
    forex_symbols = pd.DataFrame(norgate.database('Forex Spot'))['symbol']
    forex_symbols_usd = forex_symbols[forex_symbols.str.endswith('USD')]

    for symbol in tqdm(forex_symbols_usd, desc="Forex", colour='green'):
        df = get_prices(symbol)
        if df is not None:
            df = df.reset_index()  # keep original df unchanged as a good practice
            df.insert(0, 'ForexSymbol', symbol)
            insert_df_into_db(conn, info=symbol, df=df, table_name='Forex')


def process_index(conn):
    index_symbols = [
        '$SPXTR',  # S&P 500 Total Return Index
        '$NDXTTR',  # Nasdaq-100 Technology Sector Total Return Index
        '$RUTTR',  # Russell 2000 Total Return Index
        '$BCOMTR',  # Bloomberg Commodity Total Return Index
        '$CRB',  # FTSE/CoreCommodity CRB Index
        '$SPGSCITR'  # S&P GSCI Total Return Index
    ]

    for symbol in tqdm(index_symbols, desc="Indexes", colour='green'):
        df = get_prices(symbol)
        if df is not None:
            df = df.reset_index()  # keep original df unchanged as a good practice
            name = norgate.security_name(symbol)
            df.insert(0, 'Name', name)
            df.insert(0, 'Symbol', symbol)

            insert_df_into_db(conn, info=symbol, df=df, table_name='Index')


def get_prices(symbol):
    pricedata = None
    try:
        pricedata = norgate.price_timeseries(
            symbol,
            stock_price_adjustment_setting=norgate.StockPriceAdjustmentType.TOTALRETURN,
            padding_setting=norgate.PaddingType.NONE,
            timeseriesformat='pandas-dataframe',
        )
    except Exception as e:
        print(f"Exception in norgatedata.price_timeseries for symbol: {symbol}", str(e))

    return pricedata


def delete_database():
    # we delete previous database explicitly, may be this is not quite necessary, but just in case...
    print("Delete database")
    duck_db = Config.DUCK_DB
    if os.path.exists(duck_db):
        os.remove(duck_db)
    if os.path.exists(duck_db + ".wal"):
        os.remove(duck_db + ".wal")


if __name__ == "__main__":
    with Timer():
        raise Exception("Database protection")

        # delete and re-create database
        assert norgate.status(), "Norgatedata (NDU) not running"

        delete_database()
        with duckdb.connect(Config.DUCK_DB) as connection:  # this will create a new database file if it doesn't exist
            create_tables(connection)

            process_futures(connection)
            process_futures_contracts(connection)
            process_cont_contracts(connection)
            process_forex(connection)
            process_index(connection)
