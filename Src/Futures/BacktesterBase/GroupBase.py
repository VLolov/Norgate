from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, List, TYPE_CHECKING

from Futures.BacktesterBase.Base import Base

if TYPE_CHECKING:
    from .BacktesterBase import BacktesterBase
    from .Base import Base
    from .BrokerBase import BrokerBase
    from .InstrumentBase import InstrumentBase
    from .StrategyBase import StrategyBase


class GroupBase(Base, ABC):
    def __init__(self, name: str):
        super().__init__()
        self.name = name
        self.backtester: Optional[BacktesterBase] = None
        self.broker: Optional[BrokerBase] = None
        self.instruments: List[InstrumentBase] = []
        self.strategies: List[StrategyBase] = []

    def __repr__(self):
        return (f"<{self.__class__.__name__} id: {self.id}, "
                f"name: {self.name}, "
                f"backtester: {self.id_string(self.backtester)}, "
                f"broker: {self.id_string(self.broker)}, "
                f"instruments count: {len(self.instruments)}, "
                f"strategies count: {len(self.strategies)}>")

    def set_broker(self, broker: BrokerBase):
        self.broker = broker
        return self

    def add_instrument(self, instrument: InstrumentBase):
        self.instruments.append(instrument)
        return self

    def add_strategy(self, strategy: StrategyBase):
        self.strategies.append(strategy)
        return self

    def add_backtester(self, backtester: BacktesterBase):
        self.backtester = backtester
        return self

    @abstractmethod
    def run(self):
        ...


