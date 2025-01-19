from abc import ABC, abstractmethod
from typing import Optional, List

import Futures.BacktesterBase as Bb


class MyBacktester(Bb.BacktesterBase):
    def __init__(self):
        super().__init__()

    def check_state(self) -> bool:
        return len(self.reports) > 0 and self.portfolio is not None and len(self.groups) > 0

    # def run(self):
    #     for group in self.groups:
    #         group.run()


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.DEBUG)

    bt = MyBacktester()

    Bb.Base.print_instances()
    bt.print_hierarchy()
