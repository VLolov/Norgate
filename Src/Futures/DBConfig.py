import os


class DBConfig:
    MY_DIRECTORY = os.path.dirname(__file__)
    DUCK_DB = f'{MY_DIRECTORY}/norgate_futures.duckdb'
    EXCEL_FILE = f'{MY_DIRECTORY}/Futures.xlsx'
