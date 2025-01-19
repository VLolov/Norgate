from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, List, TYPE_CHECKING

from Futures.BacktesterBase.Base import Base

if TYPE_CHECKING:
    from .GroupBase import GroupBase
    from .InstrumentBase import InstrumentBase


class StrategyBase(Base, ABC):

    def __init__(self, name: str):
        super().__init__()
        self. name = name
        self.group: Optional[GroupBase] = None

    def set_group(self, group: GroupBase):
        self.group = group
        return self

    @abstractmethod
    def open(self, instrument: InstrumentBase, offset):
        ...


    # @abstractmethod
    # def high(self, instrument, offset):
    #     ...
    #
    # @abstractmethod
    # def low(self, instrument, offset):
    #     ...
    #
    # @abstractmethod
    # def close(self, instrument, offset):
    #     ...
    #
    # @abstractmethod
    # def volume(self, instrument, offset):
    #     ...
    #
    # @abstractmethod
    # def timestamp(self, instrument, offset):
    #     ...
    #
    # @abstractmethod
    # def is_roll(self, instrument):
    #     ...
    #
    # @abstractmethod
    # def init(self):
    #     ...
    #
    # @abstractmethod
    # def next(self):
    #     ...
    #
    # @abstractmethod
    # def last(self):
    #     # called once after the last next()
    #     ...
