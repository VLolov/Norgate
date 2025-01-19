"""
    Created 11.11.2024
    Merge individual future contracts to one dataframe
    In this dataframe we have unadjusted and adjusted continuous contracts for N-expirations

    Idea from: https://manishbansal3003.blogspot.com/2024/06/in-todays-world-stock-trading-has.html

"""
import os
import time
from datetime import datetime
from typing import List, Optional
from dataclasses import dataclass

import matplotlib
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd
import duckdb
from tabulate import tabulate
from tqdm import tqdm

from Futures.TrendFollowing.Timer import Timer


@dataclass
class Config:
    LATEST_DELIVERY_MONTH = None      # don't process contracts expiring after this date (we don't need them)
    ROLL_BEFORE_EXPIRATION_DAYS = 1     # 1 seems to come closest to the norgate continuous contract
    DUCK_DB_FILE = os.path.dirname(__file__) + '/../norgate_futures.duckdb'


def contracts_for_symbol(con, symbol, max_delivery_month=None) -> pd.DataFrame:
    additional_filter = f" AND DeliveryMonth < {max_delivery_month} " if max_delivery_month else " "
    df = con.sql(f"""
        SELECT * FROM FuturesContracts
        WHERE Symbol = '{symbol}' {additional_filter}
        ORDER BY DeliveryMonth ASC
    """).to_df()
    # print(tabulate(df, headers='keys', tablefmt='psql'))
    return df


def last_trading_date(con,
                      contract_symbol: str,
                      contract_delivery: int,
                      static_cache=dict(),  # local cache
                      static_max_date_in_data=[0]): # local cache
    if static_max_date_in_data[0] == 0:
        # query all contracts
        df = con.sql(f"SELECT MAX(Date) as last_date_in_data FROM Contracts").to_df()
        static_max_date_in_data[0] = df['last_date_in_data'].iloc[0].to_pydatetime()

    if contract_symbol not in static_cache:
        df = con.sql(f"""
            SELECT MAX(Date) as last_date FROM Contracts
            WHERE ContractSymbol = '{contract_symbol}'
        """).to_df()
        last_date = df['last_date'].iloc[0].to_pydatetime()
        if (static_max_date_in_data[0] - last_date).days < 2:   # give it 2 days buffer
            # this contract expires in the future, use delivery_month to guess the expiration
            delivery_year = contract_delivery // 100
            delivery_month = contract_delivery % 100
            last_date = datetime(delivery_year, delivery_month, 15)     # take middle of the month as a guess

        static_cache[contract_symbol] = last_date

    return static_cache[contract_symbol]


def one_contract(con, contract_symbol,
                 contract_delivery_month: int,
                 first_notice_date: Optional[str] = None,
                 remove_last_days: Optional[int] = 0) -> pd.DataFrame:
    """
    Read one specific contract and manipulate the data a bit 8-)
    :param con: duck db connection
    :param contract_symbol: like 6A-1987H
    :param contract_delivery_month:
    :param first_notice_date: date as string (YYYY-MM-DD), if specified, removes all data after that date.
    :param remove_last_days: if specified, last days of data are removed. The idea is to roll earlier to next contract.
    :return: dataframe with contract data
    """
    df = con.sql(f"SELECT * FROM Contracts WHERE ContractSymbol = '{contract_symbol}' ORDER BY Date ASC").to_df()

    last_date = last_trading_date(con, contract_symbol, contract_delivery_month)

    diff = last_date - df['Date'].map(lambda x: x.to_pydatetime())
    df['DTE'] = diff.map(lambda x: x.days)

    if first_notice_date:
        df = df[df['Date'] < first_notice_date]
    # NOTE: we remove last days after we remove the data after first notice date
    if remove_last_days:
        df = df.iloc[:-remove_last_days]
    # print(tabulate(df, headers='keys', tablefmt='psql'))
    return df


def create_continuous_contracts(con, symbol: str) -> pd.DataFrame:
    # print(f"Creating continuous contracts for {symbol}")

    # gather here all we will calculate
    final_df = pd.DataFrame()

    contracts = contracts_for_symbol(con, symbol, Config.LATEST_DELIVERY_MONTH)

    # get each individual contract
    contract: pd.Series
    for _, contract in tqdm(contracts.iterrows(), desc='Copy Contracts', colour='green'):
        # tqdm(futures, desc="Prepare data", colour='green')
        first_notice_date = None
        if not pd.isnull(contract['FirstNoticeDate']):
            dt = pd.to_datetime(contract['FirstNoticeDate'])
            first_notice_date = dt.strftime('%Y-%m-%d')
            # first_notice_date = np.datetime_as_string(contract['FirstNoticeDate'], unit='D')    # YYYY-MM-DD
        # print("Read ", contract['ContractSymbol'])
        df = one_contract(con, contract['ContractSymbol'],
                          contract['DeliveryMonth'],
                          first_notice_date=first_notice_date,
                          remove_last_days=Config.ROLL_BEFORE_EXPIRATION_DAYS)

        # minimum open interest before we start considering data
        first = df[df['OpenInterest'] > 100]
        if len(first) > 0:  # so that we have a first index
            first_index = first.index[0]
            df = df.iloc[first_index:]
            df['DeliveryMonth'] = contract['DeliveryMonth']
            if len(final_df) == 0:
                # add first contract - here we 'define' columns of final_df
                final_df = df.copy()
            elif len(df) > 30:
                # add next contract if it has enough data
                final_df = pd.concat([final_df, df], axis=0)
            pass

    # Determine the order of contracts, i.e. the front to back month order
    # Idea:
    #   * create a column 'DateYYYYMM' with format like DeliveryMonth YYYYMM as int, containing current date of the row
    #   * on each date get the contracts and rank them by the difference DeliveryMonth - DateYYYYMM
    #       the ranking corresponds to the contract ordering, i.e.
    #       rank=1 (smallest difference DeliveryMonth - DateYYYYMM) is the front month contract,
    #       rank=2 is the next contract, etc.
    #
    # If we want to switch by higher volume (just ideas, not implemented yet)
    #   * compare volume of the ranked contracts and switch later in time
    #   * define FrontByVolume
    #   * define a latest switch point ?
    #   * remember that we switched to avoid toggle

    dt = pd.to_datetime(final_df['Date'])
    months = dt.apply(lambda d: d.month)
    years = dt.apply(lambda d: d.year)
    final_df['DateYYYYMM'] = years * 100 + months
    final_df['Diff'] = final_df['DeliveryMonth'] - final_df['DateYYYYMM']
    final_df.sort_values(by=['Date'], ascending=True, inplace=True)
    # the index is needed as we will use it in the following processing
    final_df.index = range(len(final_df))
    final_df['Front'] = -1

    # this is the actual 'switching' - rank the difference
    all_dates = final_df['Date'].unique()
    for dt in tqdm(all_dates, desc="Order contracts", colour='green'):
        date_df = final_df[final_df['Date'] == dt]
        ranked_series = date_df['Diff'].rank(method='min')
        for idx, rank in ranked_series.items():
            final_df.loc[idx, 'Front'] = rank
        pass

    final_df['Carry'] = 0.0
    # final_df['DTE'] = -1

    # Calculate the carry - the difference between the contract prices at the moment of contract switch
    #   We do this for each rank separately
    for rank in tqdm(final_df['Front'].unique(), desc="Calc.carry", colour='green'):
        combined_rank_df = final_df[final_df['Front'] == rank].copy()   # copy as we will modify the dataframe

        # dates of switching to the next contract: ContractSymbol is the contract that will be replaced on the next day
        contract_switch_df = combined_rank_df[
            combined_rank_df['ContractSymbol'] != combined_rank_df['ContractSymbol'].shift(-1)
            ]

        combined_rank_df['Carry'] = 0.0
        carry = combined_rank_df['Close'].shift(-1) - combined_rank_df['Close']
        for idx, row in contract_switch_df.iterrows():
            combined_rank_df.loc[idx, 'Carry'] = carry.loc[idx]

        combined_rank_df.fillna(0, inplace=True)
        combined_rank_df.sort_values(by=['Date'], ascending=False, inplace=True)    # reverse dates: top=most recent date
        cum_carry_ser = combined_rank_df['Carry'].cumsum()  # spread correction on each date

        # copy back to the final dataframe
        for idx, value in cum_carry_ser.items():
            final_df.loc[idx, 'Carry'] = value

    # add adjusted columns:
    for col in ['Open', 'High', 'Low', 'Close']:
        final_df['Adj' + col] = final_df[col] + final_df['Carry']

    # remove the intermediate columns from final dataframe
    final_df.drop(['DateYYYYMM', 'Diff'], axis=1, inplace=True)

    return final_df


def plot_continuous_contracts(symbol, desc, final_df, cont_contract=None):
    # front_list = list(range(1, 4)) # combined_df['Front'].max()+1))
    front_list = list(range(1, final_df['Front'].max()+1))   # all available contracts

    cmap = matplotlib.colormaps['rainbow']
    # Take colors at regular intervals spanning the colormap.
    colors = cmap(np.linspace(0, 1, len(front_list)))

    plt.figure()

    for front in front_list:
        df = final_df[final_df['Front'] == front]
        # plot only part of data
        # df = df[('2023-01-01' < df['Date'])]  # & (df['Date'] < '2024-10-01')]

        plt.plot(df['Date'], df['Close'], linestyle='solid', label=f'Not adjusted: {front}', lw=1, color=colors[front-1], alpha=0.8)
        plt.plot(df['Date'], df['AdjClose'], linestyle='dotted', label=f'Adjusted: {front}', lw=2, color=colors[front-1], alpha=0.8)

        contract_switch_df = df[df['ContractSymbol'] != df['ContractSymbol'].shift()]
        for idx, row in contract_switch_df.iloc[1:].iterrows():
            plt.axvline(row['Date'], color='red', linestyle='dotted', lw=0.5, alpha=0.5)

    if cont_contract is not None:
        plt.plot(cont_contract['Date'], cont_contract['Close'], linestyle='--', label=f'Cont.contr.Norgate',
                 lw=1, color='black', alpha=0.8)

    plt.legend()
    plt.grid(False)
    plt.title(f"Contracts for: {symbol} - {desc}")
    plt.show()


def calculate_all_symbols(con, symbol_list):
    print(f"Re-creating table AllContracts")
    con.sql(f"""
        DROP TABLE IF EXISTS AllContracts;
        CREATE TABLE AllContracts (
            Symbol VARCHAR,
            ContractSymbol VARCHAR,
            Date TIMESTAMP,
            Open DECIMAL(19,6),
            High DECIMAL(19,6),
            Low DECIMAL(19,6),
            Close DECIMAL(19,6),
            Volume  BIGINT,
            DeliveryMonth INT,
            OpenInterest BIGINT,
            DTE INT,
            Front INT,
            Carry DOUBLE,
            AdjOpen DECIMAL(19,6),
            AdjHigh DECIMAL(19,6),
            AdjLow DECIMAL(19,6),
            AdjClose DECIMAL(19,6),
            
            PRIMARY KEY(ContractSymbol, Date, Front)
        );
    """)

    for i, symbol in enumerate(symbol_list):
        print(f"Processing {symbol}")
        df = create_continuous_contracts(con, symbol)
        df.reset_index(drop=True)
        sql_text = "INSERT INTO AllContracts BY NAME SELECT * FROM df"
        con.sql(sql_text)


def all_futures(con) -> pd.DataFrame:
    return con.sql(f"SELECT * FROM Futures ORDER by Symbol").to_df()


def calculate_continuous_store_db():
    # **** re-create database table and calculate all contracts ***
    with duckdb.connect(Config.DUCK_DB_FILE, read_only=False) as con:
        symbols = all_futures(con)['Symbol'].values
        calculate_all_symbols(con, symbols)


def plot_continuous(symbols, max_front=99):
    assert max_front > 0, "max_front must be > 0"
    with duckdb.connect(Config.DUCK_DB_FILE, read_only=True) as con:
        # symbols = all_symbols(conn)
        # df = create_continuous_contracts(conn, symbol)
        # df = conn.sql(f"SELECT * FROM {Config.'AllContracts'} WHERE Symbol='{symbol}'").to_df()
        # print(tabulate(df[df['Front'] == 1], headers='keys', tablefmt='psql'))
        # print(tabulate(combined_df.groupby(['Front']).count()[['Date']], headers='keys', tablefmt='psql'))
        # plot_continuous_contracts(symbol, df)

        for symbol in symbols:
            df = con.sql(f"SELECT * FROM AllContracts WHERE Symbol='{symbol}' AND Front<={max_front}").to_df()
            cont_contract = con.sql(f"SELECT * FROM ContContracts WHERE Symbol='{symbol}' "
                                     f"AND Adjusted=1 ORDER BY Date").to_df()
            name = con.sql(f"SELECT * FROM Futures WHERE Symbol='{symbol}' LIMIT 1").to_df().iloc[0]['Name']
            plot_continuous_contracts(symbol, name, df, cont_contract)


def show_available_expirations():
    # check how many front contracts are available
    """
    [
        "Crypto": {
            "BTC": 2,
            "ETH": 1,
            "MBT": 2,
            "MET": 1
        },
        "Currency": {
            "6A": 1,
            "6B": 1,
            "6C": 2,
            "6E": 2,
            "6J": 1,
            "6M": 1,
            "6N": 1,
            "6S": 1,
            "DX": 1
        },
        "Energy": {
            "BRN": 2,
            "CL": 5,
            "GAS": 6,
            "GWM": 8,
            "HO": 1,
            "NG": 10,
            "RB": 11,
            "WBS": 28
        },
        "Volatility": {
            "VX": 2
        },
        "Equity": {
            "EMD": 1,
            "ES": 1,
            "FCE": 1,
            "FDAX": 1,
            "FESX": 2,
            "FSMI": 1,
            "FTDX": 1,
            "GD": 1,
            "HSI": 1,
            "HTW": 1,
            "KOS": 1,
            "LFT": 1,
            "M2K": 1,
            "MES": 2,
            "MHI": 1,
            "MNQ": 2,
            "MYM": 1,
            "NIY": 1,
            "NKD": 1,
            "NQ": 1,
            "RTY": 1,
            "SCN": 1,
            "SNK": 1,
            "SSG": 1,
            "SXF": 1,
            "YAP": 1,
            "YM": 1
        },
        "Metal": {
            "GC": 6,
            "HG": 4,
            "PA": 1,
            "PL": 1,
            "SI": 5
        },
        "Fixed Income": {
            "CGB": 1,
            "FBTP": 1,
            "FGBL": 1,
            "FGBM": 1,
            "FGBS": 1,
            "FGBX": 1,
            "FOAT": 1,
            "LLG": 1,
            "SJB": 1,
            "TN": 1,
            "UB": 1,
            "YIB": 5,
            "YIR": 8,
            "YXT": 1,
            "YYT": 1,
            "ZB": 1,
            "ZF": 1,
            "ZN": 1,
            "ZQ": 3,
            "ZT": 1
        },
        "Rates": {
            "CRA": 2,
            "LEU": 16,
            "SO3": 10,
            "SR3": 8
        },
        "Grain": {
            "AFB": 1,
            "AWM": 1,
            "KE": 2,
            "LWB": 2,
            "MWE": 2,
            "ZC": 5,
            "ZL": 6,
            "ZM": 6,
            "ZO": 2,
            "ZR": 1,
            "ZS": 6,
            "ZW": 4
        },
        "Soft": {
            "CC": 3,
            "CT": 4,
            "DC": 5,
            "KC": 4,
            "LBR": 2,
            "LCC": 5,
            "LRC": 3,
            "LSU": 3,
            "OJ": 2,
            "RS": 2,
            "SB": 4
        },
        "Meat": {
            "GF": 4,
            "HE": 5,
            "LE": 4
        }
    ]
    """
    min_fronts = {}
    with duckdb.connect(Config.DUCK_DB_FILE, read_only=True) as con:
        futures = all_futures(con)
        for sector in ['Crypto', 'Currency', 'Energy', 'Volatility', 'Equity', 'Metal', 'Fixed Income', 'Rates',
                       'Grain', 'Soft', 'Meat']:
            min_fronts[sector] = {}
            for _, future in futures.iterrows():
                # no back month contracts in Equity, Currency
                if future['Sector'] not in sector:
                    continue

                symbol = future['Symbol']

                df = create_continuous_contracts(con, symbol)
                df.reset_index(drop=True)

                cont_contract = con.sql(f"SELECT * FROM ContContracts WHERE Symbol='{symbol}' "
                                        f"AND Adjusted=1 ORDER BY Date").to_df()

                name = con.sql(f"SELECT * FROM Futures WHERE Symbol='{symbol}' LIMIT 1").to_df().iloc[0]['Name']

                x = df.groupby(['Date']).count()
                min_nr = x.iloc[252:]['Front'].min()
                min_fronts[sector][symbol] = int(min_nr)
                plt.figure()
                x['Front'].plot()
                plt.axhline(min_nr, color='red', linestyle='solid', lw=2, alpha=1.0)
                plt.grid(True)
                plt.title(f"{symbol} - {name}")
                plt.show()

        # plot_continuous_contracts(symbol, name, df, cont_contract)
        import json
        print(json.dumps(min_fronts, indent=4, sort_keys=False))


if __name__ == "__main__":
    matplotlib.use("QtAgg")

    # raise Exception("Database protection")

    with Timer():
        # calculate_continuous_store_db()   # !!! Write in Database !!!

        # show_available_expirations()
        #     symbols = ['RB']    # , 'ZS' 'ES', '6E', 'CL', 'GC', 'NG']

        plot_continuous(['GF'], max_front=99)

