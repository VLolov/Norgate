"""
    Get recent IPOs by scrapping the data from https://stockanalysis.com/ipos/
    Code is based on: https://www.geeksforgeeks.org/scrape-tables-from-any-website-using-python/

    Vasko:
        20.09.2024	Initial version
"""

import os.path
import pandas as pd
import urllib.request

# pip install html-table-parser-python3
# for parsing all the tables present on the website
from html_table_parser import HTMLTableParser
import pickle


class ScrapeTables:
    def __init__(self, url):
        self.url = url

    def download(self):
        xhtml = self.url_get_contents()

        # define the HTMLTableParser object
        p = HTMLTableParser()

        # feed the html contents in the HTMLTableParser object
        p.feed(xhtml)

        # obtaining the data of the tables found on self.url
        return p.tables

    # Opens a website and read its
    # binary contents (HTTP Response Body)
    def url_get_contents(self):
        url = self.url
        # Opens a website and read its binary contents (HTTP Response Body)

        # making request to the website
        req = urllib.request.Request(
            url,
            data=None,
            headers={
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.36 (KHTML, like Gecko) '
                              'Chrome/35.0.1916.47 Safari/537.36'
            }
        )
        with urllib.request.urlopen(req) as f:
            # reading contents of the website
            data = f.read()

        x = data.decode('utf-8')
        return x


def scrape_table(url):
    scrape_tables = ScrapeTables(url)
    tables = scrape_tables.download()
    nr_tables = len(tables)
    expected_nr_tables = 2
    assert nr_tables == expected_nr_tables, f'Number of tables <> {expected_nr_tables}'
    df = pd.DataFrame(tables[0][1:], columns=tables[0][0])
    df = df[['IPO Date', 'Symbol']]
    df.columns = ['date', 'Symbol']
    df['date'] = df['date'].astype('datetime64[ns]')
    df.set_index(['date'], inplace=True)
    return df


def scrape_all_years(base_url, year_list):
    df = None
    for curr_year in year_list:
        url = base_url + str(curr_year)
        curr_df = scrape_table(url)
        if df is None:
            df = curr_df
        else:
            df = pd.concat([df, curr_df])

    df.sort_index(inplace=True)     # sort by date, not really needed, but nice for debugging
    return df


def scrape():
    file_path = "recent_ipos.pickle"

    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            df = pickle.load(f)

    else:
        base_url = "https://stockanalysis.com/ipos/"
        df = scrape_all_years(base_url, list(range(2019, 2025)))
        with open(file_path, 'wb') as f:
            # pickle.dump(sorted_data_list, f)
            pickle.dump(df, f)
        print("scraping data")
    return df


if __name__ == "__main__":
    # NOTE: delete pickle file to reload data from the website
    x = scrape()

