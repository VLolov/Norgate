"""
    Read futures data from database and show basic information and charts

    Replacements for LIBOR (The official publication of most LIBOR rates ceased after
    December 31, 2021, with the remaining USD LIBOR rates set to end in June 2023.)

    Alternative Rates:
    Various regions adopted alternative risk-free rates (RFRs) to replace LIBOR:
        SOFR (Secured Overnight Financing Rate): For USD.
        SONIA (Sterling Overnight Index Average): For GBP.
        ESTR (Euro Short-Term Rate): For EUR.
        TONAR (Tokyo Overnight Average Rate): For JPY.
        SARON (Swiss Average Rate Overnight): For CHF.


    Vasko:
    13.10.2024	Initial version
"""

import datetime

import pandas as pd
import duckdb
from tabulate import tabulate
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

from DBConfig import DBConfig

matplotlib.use("Qt5Agg")


def nr_contracts(con):
    # futures = con.sql(f'select * from Futures').to_df()
    i = 1
    #         CREATE TABLE FuturesContracts (
    #             Symbol VARCHAR, -- base symbol like 'ES'
    #             Contract VARCHAR, -- symbol like 'ES-1998M'
    df = con.sql("""
        SELECT Symbol, count(ContractSymbol) AS NumberContracts 
        FROM FuturesContracts 
        GROUP BY Symbol 
        ORDER BY Symbol"""
        ).to_df()
    print("Contracts by future:")
    print(tabulate(df, headers='keys', tablefmt='psql'))


def plot_future_single(con, future):
    # plot single contracts

    # df_futures = con.sql(f"SELECT * from Futures WHERE Symbol={future}").to_df()
    # print(tabulate(df_future.head(), headers='keys', tablefmt='psql'))

    # get all individual contracts for future
    sql_string = f"""
        SELECT ct.ContractSymbol, ct.Date, ct.Open, ct.High, ct.Low, ct.Close, ct.Volume, ct.OpenInterest, fc.DeliveryMonth
        FROM Contracts ct
        LEFT JOIN FuturesContracts fc on ct.ContractSymbol = fc.ContractSymbol 
        WHERE fc.Symbol = '{future}' 
        ORDER BY fc.DeliveryMonth, ct.Date
    """
    df_all = con.sql(sql_string).to_df()
    print(tabulate(df_all.tail(100), headers='keys', tablefmt='psql'))

    # # for each day get contracts ordered by delivery month
    # # this may help to get the front- second month, etc. contracts
    # sql_string = f"""
    #     SELECT ct.Symbol, ct.Date, ct.Open, ct.High, ct.Low, ct.Close, ct.Volume, ct.OpenInterest, fc.DeliveryMonth
    #     FROM Contracts ct
    #     LEFT JOIN FuturesContracts fc on ct.Symbol = fc.Contract
    #     WHERE fc.Symbol = '{future}'
    #     ORDER BY ct.Date, fc.DeliveryMonth
    # """
    # df_allx = con.sql(sql_string).to_df()
    # print(tabulate(df_allx.tail(100), headers='keys', tablefmt='psql'))

    # get list of individual contracts for each future
    contracts = con.sql(f"""
        SELECT ContractSymbol as Contract
        FROM FuturesContracts 
        WHERE Symbol = '{future}' 
        ORDER BY DeliveryMonth
    """).to_df()['Contract'].to_list()
    i = 1
    print(contracts)

    cont_contract = con.sql(f"""
        SELECT *
        FROM ContContracts 
        WHERE Symbol='{future}' AND Adjusted=0 
        ORDER BY Date
    """).to_df()
    cont_contract.set_index("Date", inplace=True)

    cont_contract_adjusted = con.sql(f"""
        SELECT *
        FROM ContContracts 
        WHERE Symbol='{future}' AND Adjusted=1 
        ORDER BY Date
    """).to_df()
    cont_contract_adjusted.set_index("Date", inplace=True)

    start_date = '2023-01-01'
    end_date = '2024-10-10'     # note select date on a workday and before the end date in database !!!

    cont_contract.loc[start_date:end_date]["Close"].plot(linewidth=3, label="Continuous", alpha=1)
    cont_contract_adjusted.loc[start_date:end_date]["Close"].plot(linewidth=3, label="Cont.Adjusted", alpha=1)

    for contract in contracts[:]:
        contract_df = df_all[df_all['ContractSymbol'] == contract].copy()
        contract_df.set_index('Date', inplace=True)
        # print(tabulate(contract_df, headers='keys', tablefmt='psql'))
        df = contract_df.loc[start_date:end_date]
        # print(tabulate(df, headers='keys', tablefmt='psql'))
        if len(df) > 0:
            df['Close'].plot(label=contract, linewidth=1, alpha=0.7)
            # plot dot at expiration
            if (df.index[-1].to_pydatetime() + datetime.timedelta(days=1)).strftime('%Y-%m-%d') < end_date:
                # this complex if statement avoids dots at the end of data
                plt.plot(df.index[-1], df['Close'].iloc[-1], 'ro', markersize=5)    # dot on expiration

    plt.legend(loc='upper left')
    plt.title(f"Contracts for {future}, dates: {start_date}...{start_date}")
    plt.show()


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)
sns.set_style("whitegrid")


def contracts_min_max_date(con):
    sql_string = f"""
        SELECT Symbol, min(Date) AS MinDate, max(Date) AS MaxDate
        FROM ContContracts
        WHERE Adjusted=0
        GROUP BY Symbol
        ORDER BY min(Date) ASC
    """
    df = con.sql(sql_string).to_df()
    print("Min/Max date per contract:")
    print(tabulate(df, headers='keys', tablefmt='psql'))


def contracts_av_volume_date(con):
    sql_string = f"""
        SELECT Symbol, Date, average(Volume) AS Vol
        FROM Contracts
        WHERE Adjusted=0
        GROUP BY Symbol
        ORDER BY min(Date) ASC
    """
    df = con.sql(sql_string).to_df()
    print("Min/Max date per contract:")
    print(tabulate(df, headers='keys', tablefmt='psql'))


def _get_tick_size(conn, symbol):
    now = datetime.datetime.now()
    year_month = now.year * 100 + now.month
    sql = f"""SELECT TickSize
             FROM FuturesContracts
             WHERE Symbol = '{symbol}' AND DeliveryMonth <= {year_month}
             ORDER BY DeliveryMonth DESC LIMIT 1"""
    # this returns the original database type. In this case - Decimal type, which has to be converted: float(f[0][0]
    # f = self.conn.sql(sql).fetchall()
    # return f[0][0]
    df = conn.sql(sql).to_df()
    assert len(df) == 1, 'No recent data in database'
    return df.iloc[0]['TickSize']


def all_futures_info(con):
    df = con.sql("SELECT * FROM Futures").to_df()
    df['TickSize'] = 0.0
    df['MinDate'] = pd.Timestamp.min
    df['MaxDate'] = pd.Timestamp.min
    # add first/last date
    for i, row in df.iterrows():
        symbol = row['Symbol']
        dates = con.sql(f"SELECT DISTINCT min(Date) as min_date, max(Date) as max_date "
                        f"FROM AllContracts WHERE symbol='{symbol}'").to_df()
        df.loc[df.index[i], 'MinDate'] = dates['min_date'].iloc[0]
        df.loc[df.index[i], 'MaxDate'] = dates['max_date'].iloc[0]

        df.loc[df.index[i], 'TickSize'] = _get_tick_size(con, symbol)

    print("All futures:")
    print(tabulate(df, headers='keys', tablefmt='psql'))

    filename = 'AllFutures.csv'
    df = df.reset_index()
    # df.sort_values(by='Name', inplace=True)
    # data_frame.to_csv(filename, index=False, header=True, float_format="%.4f")
    df.to_csv(filename, index=False, header=True, float_format="%g")
    print(f'Data saved to "{filename}"')


def number_of_records(con):
    print("Number of records by table:")
    table_records = []
    for table in ['Futures', 'FuturesContracts', 'Contracts', 'ContContracts', 'Forex', 'AllContracts']:
        records = con.sql(f'select count(*) as CNT from {table}').fetchall()[0][0]
        table_records.append({'Table': table, 'Records': records})
        # print(f"\tTable: '{table}': Records: {records:,}")
    print(tabulate(pd.DataFrame(table_records), headers='keys', tablefmt='psql', intfmt=","))


if __name__ == "__main__":
    with duckdb.connect(DBConfig.DUCK_DB, read_only=True) as connection:
        number_of_records(connection)
        nr_contracts(connection)
        # plot_future_single(connection, 'KC')
        contracts_min_max_date(connection)
        all_futures_info(connection)




"""
SELECT ContractSymbol, Symbol, Date, Open, High, Low, Close, Volume, OpenInterest
FROM norgate_futures.main.Contracts;

-- average volume per day
SELECT Symbol, Date, AVG(Volume)
FROM norgate_futures.main.Contracts
WHERE Symbol='OJ'
GROUP by Symbol, Date 
ORDER by Symbol

-- average volume all days
SELECT Symbol, AVG(Volume)
FROM norgate_futures.main.Contracts
--WHERE Symbol='6A'
GROUP by Symbol
ORDER by Symbol

-- just a single row
SELECT * 
FROM norgate_futures.main.Contracts
WHERE Symbol='6A' 
AND Date='1987-01-13'
"""
