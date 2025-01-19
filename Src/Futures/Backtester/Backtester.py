from abc import ABC, abstractmethod
from typing import Optional, List, cast

import Futures.BacktesterBase as Bb
from Futures.Backtester.Group import Group


class Backtester(Bb.BacktesterBase):
    def check_state(self) -> bool:
        return len(self.reports) > 0 and self.portfolio is not None and len(self.groups) > 0

    def run(self):
        for group in self.groups:
            group.run()


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.DEBUG)

    bt = Backtester()

    Bb.Base.print_instances()
    bt.print_hierarchy()
