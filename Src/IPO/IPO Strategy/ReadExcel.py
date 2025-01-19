"""
excel data comes from: https://www.iposcoop.com/scoop-track-record-from-2000-to-present/
        button: Download an excel spreadsheet...
        filename: SCOOP-Rating-Performance.xls

interesting link: https://site.warrington.ufl.edu/ritter/ipo-data/

https://www.dataquest.io/blog/excel-vs-python/
needs: pip install xlrd

import pandas as pd
import os

# Disadvantage: data is only for mid-2000 to Nov-2020
# We may need to get more data from other sources
"""
import os
import pandas as pd

DATA_FILE = os.path.dirname(os.path.realpath(__file__)) + '/SCOOP-Rating-Performance.xls'
SHEET_NAME = "SCOOP Scorecard"


def read_ipo():
    df = pd.read_excel(DATA_FILE, sheet_name=SHEET_NAME)
    #  print(df)
    #                Unnamed: 0               Unnamed: 1  ... Unnamed: 10 Unnamed: 11
    # 0                     NaN                      NaN  ...         NaN         NaN
    # 1                     NaN                      NaN  ...         NaN         NaN
    # 2                    Year              IPOs Priced  ...         NaN         NaN
    # 3                    2020                      205  ...         NaN         NaN
    # 4                    2019                      231  ...         NaN         NaN
    # ...                   ...                      ...  ...         ...         ...
    # 3767  2000-12-08 00:00:00  Specialty Laboratories   ...           3         NaN
    # 3768  2000-12-08 00:00:00      W.P. Stewart & Co.   ...           2         NaN
    # 3769  2000-12-11 00:00:00                 Gemplus   ...           1         NaN
    # 3770  2000-12-12 00:00:00                   GenVec  ...           2         NaN
    # 3771  2000-12-15 00:00:00    Resources Connection   ...           2         NaN
    # [3772 rows x 12 columns]

    # filter row with datetime in second column
    df = df[['Unnamed: 0', 'Unnamed: 2']]
    df.columns = ['date', 'Symbol']
    df = df[df.notnull().all(1)]    # remove all rows with NaN
    df = df[df['Symbol'].str.isalpha().notnull()]
    df = df[~df['date'].str.isalpha().notnull()]    # remove all rows with column 'date' containing text
    df['date'] = df['date'].astype('datetime64[ns]')
    df['Symbol'] = df['Symbol'].astype('str')
    df.set_index(['date'], inplace=True)
    df.sort_index(inplace=True)     # sort by date, not really needed, but nice for debugging
    return df


if __name__ == "__main__":
    data_df = read_ipo()
    print(data_df)
