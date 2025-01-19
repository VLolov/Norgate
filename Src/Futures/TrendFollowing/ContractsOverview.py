"""
    Calculate and plot number of contracts with different conditions

    Vasko:
    18.10.2024	Initial version
"""

import duckdb

import matplotlib
import matplotlib.pyplot as plt

from Futures.DBConfig import DBConfig

matplotlib.use("Qt5Agg")

DUCK_DB = DBConfig.DUCK_DB


def nr_contracts_per_day(conn, min_volume):
    sql = f"""
        SELECT Date, COUNT(Symbol) as CNT
        FROM ContContracts
        WHERE Volume > {min_volume} AND Adjusted = 1
        GROUP by Date
        ORDER by Date ASC
    """
    df = conn.sql(sql).to_df()
    df.set_index('Date', inplace=True)
    return df


MIN_VOLUME = 10000     # number of traded contracts per day


def _main():
    with duckdb.connect(DUCK_DB, read_only=True) as connection:
        # for sector in sectors:
        # cumulative_df = dta.continuous_contract_dates()

        df = nr_contracts_per_day(connection, 0)
        # some days are holidays, but we have contracts, so we get notches in the counts => filter the notches
        df['CNT'] = df['CNT'].rolling(63).median().bfill().fillna(0)

        cum_df = df.copy()
        cum_df.columns = ['All']

        df = nr_contracts_per_day(connection, MIN_VOLUME)
        df['CNT'] = df['CNT'].rolling(63).median().bfill().fillna(0)

        cum_df = cum_df.merge(df, left_index=True, right_index=True, how='left')
        cum_df.rename(columns={'CNT': f'Liquid (Volume > #{MIN_VOLUME})'}, inplace=True)
        i = 1
        cum_df.plot()
        plt.title("Number of contracts")
        plt.show()


if __name__ == "__main__":
    _main()
