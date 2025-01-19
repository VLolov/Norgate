
# import logging
# from abc import ABC
# from typing import List
#
# import numpy as np
#
# from Futures.BacktesterBase.Base import Base
# from Futures.BacktesterBase.BrokerBase import BrokerBase
# from Futures.BacktesterBase.GroupBase import GroupBase
# from Futures.BacktesterBase.InstrumentBase import InstrumentBase
# from Futures.BacktesterBase.StrategyBase import StrategyBase
# from Futures.BacktesterBase.BacktesterBase import BacktesterBase
#
#
# class GroupNew(GroupBase):
#     def __init__(self, name: str, backtester: BacktesterBase, broker: BrokerBase):
#         super().__init__()
#         self.name = name
#         self.backtester = backtester
#         self.broker = broker
#
#         self.instruments: List[InstrumentBase] = []
#         self.strategies: List[StrategyBase] = []
#
#         self.current_index: int = 0
#         self.date_list = set()
#
#     def add_instrument(self, instrument: InstrumentBase):
#         self.instruments.append(instrument)
#
#     def add_strategy(self, strategy: StrategyBase):
#         self.strategies.append(strategy)
#
#     def data_for_date(self, date) -> bool:
#         return date in self.date_list
#
#     def run(self):
#         dates = self.instruments[0].data.index
#         for idx, timestamp in enumerate(dates):
#             for strategy in self.strategies:
#                 strategy.idx = idx
#                 strategy.time = timestamp
#                 strategy.next()
#
#         for strategy in self.strategies:
#             strategy.last()
#
#     def check_state(self) -> bool:
#         return (
#             len(self.instruments) > 0
#             and len(self.strategies) > 0
#             and self.backtester is not None
#         )
#
#
# if __name__ == "__main__":
#     logging.basicConfig(level=logging.DEBUG)
#     from Futures.Backtester.Backtester import BacktesterNew
#
#
#     class BacktesterX(BacktesterNew):
#         def __init__(self):
#             super().__init__()
#
#
#     from Futures.Backtester.BrokerNew import BrokerNew
#
#     class BrokerX(BrokerNew):
#         def __init__(self):
#             super().__init__()
#
#     backtester = BacktesterX()
#     broker = BrokerX()
#
#     class GroupX(GroupNew):
#         def __init__(self):
#             super().__init__("GroupName1", backtester, broker)
#
#
#     Base.print_instances()
