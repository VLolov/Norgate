import itertools
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

log = logging.getLogger(__name__)


class Base(ABC):
    """All our classes inherit from this one"""
    _id = itertools.count(1)
    _instances: List['Base'] = []

    def __init__(self, class_name=None):
        self.id = next(self._id)
        self._instances.append(self)
        self._log = logging.getLogger(class_name if class_name is not None else self.__class__.__name__)

    @abstractmethod
    def check_state(self) -> bool:
        """ Each class has to check its state and return it here """
        ...

    @property
    def log(self):
        return self._log

    @classmethod
    def check_all_states(cls) -> bool:
        return all(inst.check_state() for inst in cls._instances)

    @classmethod
    def print_instances(cls):
        [log.info(f"{inst}, id: {inst.id}, state: {inst.check_state()}") for inst in cls._instances]

    @classmethod
    def id_string(cls, obj):
        if obj is None:
            return "None"
        else:
            return str(obj.id)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    class Abc(Base):
        def check_state(self):
            return True

        def __init__(self):
            # note: not initialized properties don't appear in child class ?!?
            super().__init__()
            self.some_prop = 1

    class Xyz(Base):

        def __init__(self):
            # note: not initialized properties don't appear in child class ?!?
            super().__init__(__name__)
            self.some_prop = 1

        def check_state(self):
            return True

        def hi(self):
            self.log.info("hi")

    abc = Abc()

    xyz = Xyz()
    xyz.hi()

    Base.print_instances()

