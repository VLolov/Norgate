"""
    Read data from Norgate and store it in a pickle file
    Norgate databases: 'US Equities' and 'US Equities Delisted'
    NOTE: delete pickle file to reload data from the website

    Vasko:
    20.09.2024	Initial version
"""

import time
from datetime import datetime, timedelta
from dataclasses import dataclass

import pandas as pd
import norgatedata
import re
import pickle
from typing import ClassVar, List


pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)

assert norgatedata.status(), "Norgatedata (NDU) not running"

# get all US Equities - listed and delisted and find their first quoted date
# then make statistics by year

#
# databasecontents = norgatedata.database(databasename)


class Config:
    OUT_PATH = "dataframes.pickle"
    BEG_DATE = datetime(1999, 1, 1)
    DAYS_TO_LOAD = 90
    MIN_TURNOVER = 1e6  # 1 Mio


# constants, don't change
PRICE_ADJUST = norgatedata.StockPriceAdjustmentType.TOTALRETURN
PAD_SETTING = norgatedata.PaddingType.NONE
TIMESERIES_FORMAT = 'pandas-dataframe'  # 'numpy-recarray' or 'pandas-dataframe'

start = time.time()
processed_symbols = 0


@dataclass
class SelectedData:
    selected_list: ClassVar[List] = list()  # class variable: keeps list of SelectedDate objects
    symbol: str
    data: pd.DataFrame



def is_delisted_symbol(symbol):
    # match pattern XXX-DDDDDD, example ZWRK-202213
    return re.match(r"^[A-Z]+-\d{6}$", symbol) is not None


def is_alphanumeric(symbol):
    return symbol.isalpha()


benchmark_df = norgatedata.price_timeseries(
    '$SPX',
    stock_price_adjustment_setting=PRICE_ADJUST,
    padding_setting=PAD_SETTING,
    start_date=Config.BEG_DATE,
    # end_date=last_quote_date,
    timeseriesformat='pandas-dataframe',
)

benchmark_df_x = benchmark_df.copy()
benchmark_df_x["Unadjusted Close"] = benchmark_df["Close"]
benchmark_df_x["Dividend"] = 0


symbols = norgatedata.database_symbols('US Equities')
symbols_delisted = norgatedata.database_symbols('US Equities Delisted')
print("Nr. symbols:", len(symbols), "Nr. delisted symbols:", len(symbols_delisted))

symbols.extend(symbols_delisted)

for symbol in symbols[:]:
    if not is_alphanumeric(symbol) and not is_delisted_symbol(symbol):
        print(f"Skipping non-alpha symbol: {symbol}")
        continue

    first_quote_date = norgatedata.first_quoted_date(symbol, datetimeformat='datetime')

    if first_quote_date is None:
        print(f"Skipping {symbol} - no first quote date")
        continue

    our_first_date = Config.BEG_DATE + timedelta(days=10)

    if first_quote_date < Config.BEG_DATE:
        print(f"Skipping {symbol} first_quote_date: {first_quote_date} - starts before beginning of our first date: {our_first_date}")
        continue

    last_quote_date = first_quote_date + timedelta(days=Config.DAYS_TO_LOAD)

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

    if len(symbol_df) < Config.DAYS_TO_LOAD * (5 / 7) - 5:
        print(f"Skipping {symbol} - dataframe too short: {len(symbol_df)}")
        continue

    # turnover = symbol_df.iloc[:10]['Turnover'].mean()   # average turnover in first 10 days
    # if turnover < Config.MIN_TURNOVER:
    #     print(f"Skipping {symbol} first_quote_date: {first_quote_date}, turnover too small: {turnover:.0f}")
    #     continue
    if symbol == 'ASTH':
        abc = 0

    processed_symbols += 1
    print(">>>", symbol, len(symbol_df), first_quote_date)
    sel_data = SelectedData(symbol=symbol, data=symbol_df)
    SelectedData.selected_list.append(sel_data)
    # print(price_df)

# as concatenation is slow, first create a list of Series to be concatinated
# https://oricohen.substack.com/p/speeding-up-pandas-dataframe-concatenation-748fe237244e
concat_list = [benchmark_df]
for sd in SelectedData.selected_list[:]:
    print(sd.data.index[0], sd.symbol)
    s = sd.data['Close']
    s.name = sd.symbol
    concat_list.append(s)
# then concatenate the while list
df = pd.concat(concat_list, axis=1, join='outer')  # this is now pretty fast (2-3 seconds)
df = df.copy()  # will this de-fragment the dataframe ?!?
with open(Config.OUT_PATH, 'wb') as f:
    # pickle.dump(sorted_data_list, f)
    pickle.dump(df, f)
    print("data saved")

del df

print(f"Number of symbols: {len(symbols)}, processed symbols: {processed_symbols}")
print(f"Executed in {round((time.time() - start), 4)} seconds")

