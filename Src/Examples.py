# Examples from https://pypi.org/project/norgatedata/#requirements
# pip install norgatedate --upgrade, installs numpy too

import pandas as pd
import norgatedata
import os

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)


def save_dataframe(symbol, df, data_path, check_nan=True):
    if check_nan:
        nan_rows = df[df.isnull().any(axis=1)]

        if nan_rows is not None and len(nan_rows) > 0 and symbol not in ['M2', 'M3']:
            # note M2 and M3 can have NaN values
            print(f'NaN values for symbol "{symbol}":\n{nan_rows}')

    filename = f'{data_path}/{symbol}.csv'

    # Create output directory
    os.makedirs(data_path, exist_ok=True)

    df = df.reset_index()
    # data_frame.to_csv(filename, index=False, header=True, float_format="%.4f")
    df.to_csv(filename, index=False, header=True, float_format="%g")
    print(f'Data saved to "{filename}"')

#
# Price and volume
#

def get_prices(symbol):
    priceadjust = norgatedata.StockPriceAdjustmentType.TOTALRETURN
    padding_setting = norgatedata.PaddingType.NONE
    timeseriesformat = 'pandas-dataframe' # 'numpy-recarray'

    pricedata = norgatedata.price_timeseries(
        symbol,
        stock_price_adjustment_setting=priceadjust,
        padding_setting=padding_setting,
        # start_date=start_date,
        timeseriesformat=timeseriesformat,
    )
    return pricedata

def get_symbol_store_in_file(symbol):
    df = get_prices(symbol)
    print(df)
    df.index.rename('timestamp', inplace=True)
    df.columns = ['open', 'high', 'low', 'close']
    save_dataframe(symbol, df, 'IPO/downloaded_data')


# # Now in a Pandas-compatible format
# priceadjust = norgatedata.StockPriceAdjustmentType.TOTALRETURN  # default
# padding_setting = norgatedata.PaddingType.NONE   # ALLCALENDARDAYS
# symbol = '$SPX'
# start_date = '2020-09-01'  # '1990-01-01'
# timeseriesformat = 'pandas-dataframe'
# start_date = pd.Timestamp(start_date) # we can also provide dates as a Pandas Timestamp
# pricedata_dataframe = norgatedata.price_timeseries(
#     symbol,
#     stock_price_adjustment_setting=priceadjust,
#     padding_setting=padding_setting,
#     # start_date=start_date,
#     timeseriesformat=timeseriesformat,
# )
# print(pricedata_dataframe)


#
# Time Series Data
#

# Index Constituents
# symbol = 'NFLX'
# indexname = 'Russell 3000'
# # indexname = 'S&P 500'  # Can also be an index symbol, such as $SPX, $RUI etc.
#
# idx = norgatedata.index_constituent_timeseries(
#     symbol,
#     indexname,
#     # timeseriesformat="numpy-recarray",
#     timeseriesformat="pandas-dataframe",
# )
# pd.set_option('display.max_rows', None)
# print(idx)

# Add column to history-data dataframe
# Alternative using pandas dataframes instead:
#
# indexname = 'S&P 500'  # Can also be an index symbol, such as $SPX, $RUI etc.
# symbol = 'NFLX' #'TWTR'
# timeseriesformat = 'pandas-dataframe'
# priceadjust = norgatedata.StockPriceAdjustmentType.TOTALRETURN
# padding_setting = norgatedata.PaddingType.NONE
#
# pd.set_option('display.max_rows', None)
#
# # note: exception "ValueError" if symbol not found !!!
# pricedata_df = norgatedata.price_timeseries(
#     symbol,
#     stock_price_adjustment_setting=priceadjust,
#     padding_setting=padding_setting,
#     timeseriesformat=timeseriesformat,
# )
# print(pricedata_df)
#
# # and now make the call to index_constituent_timeseries
# pricedata_df2 = norgatedata.index_constituent_timeseries(
#     symbol,
#     indexname,
#     padding_setting=padding_setting,
#     pandas_dataframe=pricedata_df,
#     timeseriesformat=timeseriesformat,
# )
#
# print(pricedata_df2)

# Major Exchange Listed - data starts year 2000
# majexch = norgatedata.major_exchange_listed_timeseries(
#     'AAPL',
#     timeseriesformat='pandas-dataframe')
# print(majexch)
#
# Capital event
#
# capevent = norgatedata.capital_event_timeseries(
#     'AAPL',
#     timeseriesformat="pandas-dataframe")
# pd.set_option('display.max_rows', None)
# print(capevent[capevent['Capital Event'] != 0])
#
# Padding Status
#
# the example shows the weekends
# paddingstatus = norgatedata.padding_status_timeseries(
#     'AAPL',
#     padding_setting=norgatedata.PaddingType.ALLCALENDARDAYS,
#     timeseriesformat="pandas-dataframe",
# )
#
# print(paddingstatus[paddingstatus['Padding Status'] != 0])
#
# Single Value Data
#
# asset ID
# symbol = 'TWTR-202210'
# print("Id: ", norgatedata.assetid(symbol))
# print("Symbol:", norgatedata.symbol(129769))
# # Domicile
# print("Domicile: ", norgatedata.domicile(symbol))
# print("Currency: ", norgatedata.currency(symbol))
# print("Short Exchange Name: ", norgatedata.exchange_name(symbol))
# print("Full Exchange Name: ", norgatedata.exchange_name_full(symbol))
# print("Security Name:", norgatedata.security_name(symbol))
# print("Base Type:", norgatedata.base_type(symbol))
# print("Subtype1:", norgatedata.subtype1(symbol))
# print("Subtype2:", norgatedata.subtype2(symbol))
# print("Subtype3:", norgatedata.subtype3(symbol))
# print("Financial Summary:", norgatedata.financial_summary(symbol))
# print("Business Summary:", norgatedata.business_summary(symbol))
#
# print("First Quoted Date:", norgatedata.first_quoted_date(symbol, datetimeformat='datetime'))
# print("Last Quoted Date:", norgatedata.last_quoted_date(symbol, datetimeformat='datetime'))
# print("Second Last Quoted Date:", norgatedata.second_last_quoted_date(symbol, datetimeformat='datetime'))
#
# # Classification
# # schemename = 'NorgateDataFuturesClassification'
# # schemename = 'TRBC'
# schemename = 'GICS'
# classificationresulttype = 'ClassificationId'
# classificationresulttype = 'Name'
# classification = norgatedata.classification(
#     symbol,
#     schemename,
#     classificationresulttype,
# )
# print(classification)
#
# # Shares outstanding/shares float
# print("Shares outstanding:", norgatedata.sharesoutstanding(symbol))
# print("Shares float:", norgatedata.sharesfloat(symbol))

# Futures
# print("Futures market symbols:", norgatedata.futures_market_symbols())
# print("Futures session symbols:", norgatedata.futures_market_session_symbols())
# print("Futures market session contract:", norgatedata.futures_market_session_contracts('ES'))

# Watchlists
# watchlistname = 'Nasdaq 100 Current & Past'
# symbols = norgatedata.watchlist_symbols(watchlistname)
# print(len(symbols))
# print(symbols)
# allwatchlistnames = norgatedata.watchlists()
# print("All watchlists: ", "\n".join(allwatchlistnames))

# Databases
databasename = 'US Equities Delisted'
# symbols = norgatedata.database_symbols(databasename)
# print(len(symbols))
#
databasecontents = norgatedata.database(databasename)
#
# alldatabasenames = norgatedata.databases()
# print("Available databases:", alldatabasenames)
# i=1

print("Norgatedata status:", norgatedata.status())

# $BCOM, $BCOMTR, $CRB, $SPGSCI, $SPGSCITR
get_symbol_store_in_file('$SPGSCITR')

#
# futures
#


# Futures metadata
# terminology: 'ES-1997Z' is a 'futures contract'; 'ES' is a 'market symbol'
symbol = 'ES-1997Z' # Don't use 'ES' here !!!
tick_size = norgatedata.tick_size(symbol)
print('Tick Size: ', tick_size)
point_value = norgatedata.point_value(symbol)
print('Point value: ', point_value)
margin = norgatedata.margin(symbol)
print('Margin: ', margin)
first_notice_date = norgatedata.first_notice_date(symbol,datetimeformat = 'datetime')
print('First Notice Date: ', first_notice_date)  # None for not deliverable contracts
lowest_ever_tick_size = norgatedata.lowest_ever_tick_size(symbol)
print('Lowest ever tick size: ', lowest_ever_tick_size)  # None for not deliverable contracts
session_type = norgatedata.session_type(symbol)
print('Session type: ', session_type)  # None for not deliverable contracts
market_name = norgatedata.futures_market_name(symbol)   # can put 'ES' here too
print('Market Name: ', market_name)
session_name = norgatedata.futures_market_session_name(symbol)
print('Market Session Name: ', session_name)
session_symbol = norgatedata.futures_market_session_symbol(symbol)
print('Market Session Symbol: ', session_symbol)        # converts from 'ES-1997Z' to 'ES'
session_symbol = norgatedata.futures_market_symbol(symbol)
print('Market Symbol: ', session_symbol)        # converts from 'ES-1997Z' to 'ES'

# Futures Market Symbols
# returns list of 108 futures: ['6A', '6B', '6C', '6E', '6J
market_symbols = norgatedata.futures_market_symbols()
print(len(market_symbols))
print(market_symbols)

# Futures Market Session Symbols -> what is the difference to 'Futures Market Symbols' @@@ ?
session_symbols = norgatedata.futures_market_session_symbols()
print('Market session symbols', len(session_symbols))
print(session_symbols)

# Futures Market Session Contracts
session_symbol = '6A'
session_symbols = norgatedata.futures_market_session_contracts(session_symbol)
print(len(session_symbols))
print(session_symbols)

## get symbol price data

symbol = 'ES-1997Z' # returns a particular contract
# columns: Open    High     Low       Close   Volume  Open Interest
print(f"Get data for {symbol}")
print(get_prices(symbol).tail(10))

symbol = '&VX'   # seems to return a continuous contract
# columns: Open     High      Low    Close     Volume  Delivery Month  Open Interest
# print(f"Get data for {symbol}")
# print(get_prices(symbol).tail(100))

## get all continuous contracts:

# databasename = 'Continuous Futures'
# symbols = norgatedata.database_symbols(databasename)
# print("Continuous contracts:")
# print(len(symbols))
# print(symbols)

## NOTE: open interest for the last date is 0 ?!?
# symbol = '&VX'  # unadjusted
# symbol = '&VX_CCB'  # adjusted
# print(get_prices(symbol).tail(200))
symbol = '6A-1988M'
schemename = 'NorgateDataFuturesClassification'
# schemename = 'TRBC'
# schemename = 'GICS'
classificationresulttype = 'ClassificationId'
classificationresulttype = 'Name'
classification = norgatedata.classification(
    symbol,
    schemename,
    classificationresulttype,
)

print(classification)

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
    'Meat': ['GF', 'HE', 'LE']
}


def symbol_to_sector(symbol):
    for i, (sector, symbols) in enumerate(sectors.items()):
        if symbol in symbols:
            return sector

    msg = f"No sector found for symbol: {symbol}"
    raise ValueError(msg)
    # return ("*** no sector ***")

def get_futures_classifications():
    # all futures
    market_symbols = norgatedata.futures_market_symbols()
    # for each future take one contract (the last one)
    classifications = {}
    currencies = {}
    for symbol in market_symbols:
        if symbol in ['EUA']:
            continue

        # take the last one (it does not matter which)
        contract = norgatedata.futures_market_session_contracts(symbol)[-1]

        market_name = norgatedata.futures_market_name(symbol)
        currency = norgatedata.currency(contract)

        exchange = norgatedata.exchange_name(contract)
        point_value = norgatedata.point_value(contract)
        margin = norgatedata.margin(contract)

        sector = symbol_to_sector(symbol)

        print(f"{symbol}, {market_name}, {sector}, {currency}, {exchange}, {point_value}, {margin}")

        if sector in classifications.keys():
            classifications[sector] += 1
        else:
            classifications[sector] = 1

        if currency in currencies.keys():
            currencies[currency] += 1
        else:
            currencies[currency] = 1

    print("Total futures: ", sum(classifications.values()))
    print("Classifications: ", classifications)

    print("Total currencies: ", sum(currencies.values()))
    print("Currencies: ", currencies)

get_futures_classifications()

# databasename = 'Forex Spot'
# print("Forex database:\n", pd.DataFrame(norgatedata.database(databasename)))
# print(get_prices('BRLUSD'))
# forex_symbols = pd.DataFrame(norgatedata.database(databasename))['symbol']
# forex_symbols_usd = forex_symbols[forex_symbols.str.endswith('USD')]
# print(forex_symbols[forex_symbols.str.endswith('USD')])
