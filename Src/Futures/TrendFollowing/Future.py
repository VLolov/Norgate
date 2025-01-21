from typing import List

import pandas as pd
from dataclasses import dataclass

# from DataAccess import DataAccess
from Futures.DBConfig import DBConfig


# TODO: how/when to convert to USD ?


@dataclass
class Future:
    # _patch_micro_futures = False
    symbol: str = ''
    name: str = ''
    sector: str = ''
    currency: str = ''
    exchange: str = ''
    big_point: float = 0
    margin: float = 0
    tick_size: float = 0
    not_available: int = 1
    symbol_ib: str = ''
    symbol_micro_ib: str = ''
    provider: str = ''

    def _get_attributes(self):
        attr = [attribute for attribute in self.__dict__.keys() if not attribute.startswith('_')]
        return attr

    @classmethod
    def _create_future(cls, data_row: pd.Series):
        future = Future()
        for attr in future._get_attributes():
            value = data_row[attr]
            assert value is not None, "Missing value"

            if not value or str(value) == 'nan':
                value = None
            assert hasattr(future, attr), f"missing attribute: '{attr}' in class Future"
            setattr(future, attr, value)
        return future

    @classmethod
    def _replace_future_parameters(cls, f_original, f_replacement):
        # print(f"*** Replace:\n   {f_original}\nby {f_replacement}")
        for attr in ['big_point', 'margin', 'tick_size', 'exchange']:
            setattr(f_original, attr, getattr(f_replacement, attr))

    @classmethod
    def _all_futures(cls) -> List:
        # return all futures as a list of Future objects
        all_futures: List[Future] = []

        excel_df = pd.read_excel(DBConfig().EXCEL_FILE, sheet_name='Futures')
        for _, row in excel_df.iterrows():
            future = cls._create_future(row)
            all_futures.append(future)
        return all_futures

    @classmethod
    def get_future(cls, symbol, provider):
        futures = [future for future in cls._all_futures()
                   if future.symbol == symbol and future.provider == provider]
        assert futures and len(futures), f"Cannot find future symbol: {symbol}, provider: {provider}"
        return futures[0]

    @classmethod
    def all_futures_norgate(cls, use_micro: bool) -> List:

        # return all futures as a list of Future objects
        all_available_futures = [future for future in cls._all_futures() if not future.not_available]

        futures = [f for f in all_available_futures if f.provider == 'Norgate']
        for f in futures:
            # replace values (big point, tick size and margin) of Norgate future with the IB future
            ib_symbol = ''
            if use_micro:
                if f.symbol_micro_ib:
                    # this Norgate future has an IB micro future
                    # print("micro >>>", f)
                    ib_symbol = f.symbol_micro_ib
            else:
                # no micro, but the Norgate future still has an IB replacement
                if f.symbol_ib:
                    # this Norgate future has an IB future
                    # print("big >>>", f)
                    ib_symbol = f.symbol_ib
            if ib_symbol:
                ib_futures = [f for f in all_available_futures if f.provider == 'IB' and f.symbol == ib_symbol]
                if ib_futures and len(ib_futures) == 1:
                    # we have exactly one IB replacement
                    cls._replace_future_parameters(f, ib_futures[0])
        return futures

    @classmethod
    def get_future_norgate(cls, symbol, use_micro=False):
        futures = [future for future in cls.all_futures_norgate(use_micro) if future.symbol == symbol]
        assert futures and len(futures), \
            f"Cannot find future symbol: {symbol}, provider: Norgate, use_micro: {use_micro}"
        return futures[0]


def main():
    fs = Future.all_futures_norgate(use_micro=False)
    for i, f in enumerate(fs):
        print(i+1, f)


if __name__ == "__main__":
    main()
