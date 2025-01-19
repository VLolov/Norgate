#
# https://www.geeksforgeeks.org/scrape-tables-from-any-website-using-python/
# https://pypi.org/project/html-table-parser-python3/
#

import pandas as pd
import urllib.request

# pip install html-table-parser-python3
# for parsing all the tables present on the website
from html_table_parser import HTMLTableParser
from tabulate import tabulate


class ScrapeTable:
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


def get_margins():
    """Return list of current NASDAQ/S&P500 tickers"""
    url = 'https://www.interactivebrokers.com/en/index.php?f=26662'

    scrape = ScrapeTable(url)
    tables = scrape.download()
    nr_tables = len(tables)

    assert nr_tables >= 27, "Wrong number of tables"

    merged_df = None

    exchanges = ['CME', 'CBOT', 'CFE', 'COMEX', 'EUREX', 'ICEUS', 'NYBOT', 'NYMEX', 'NYSELIFFE']

    for idx in range(nr_tables):
        columns = tables[idx][0]
        if columns is None or len(columns[0].split()) != 2:
            continue
        _, exchange = columns[0].split()
        if exchange not in exchanges:
            continue

        columns[0] = 'Exchange'
        columns[4] = 'Intraday Initial'
        columns[5] = 'Intraday Maintenance'
        df = pd.DataFrame(data=tables[idx][1:], columns=columns)

        if merged_df is None:
            merged_df = df
        else:
            merged_df = pd.concat([merged_df, df], ignore_index=True)

    return merged_df


if __name__ == "__main__":
    margins_df = get_margins()
    print(tabulate(margins_df, headers='keys', tablefmt='psql'))


