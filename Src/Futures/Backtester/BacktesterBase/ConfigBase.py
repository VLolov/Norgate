from abc import ABC
from typing import List

from .BacktesterBase import Base


class ConfigBase(Base, ABC):
    def __init__(self):
        super().__init__()

    def __repr__(self):
        attr_values = ''
        for attribute in self.get_class_attributes():
            value = getattr(self, attribute)
            attr_values += f"\t{attribute}: {value}\n"
        return f"<{self.__class__.__name__} id: {self.id}, values:\n{attr_values}>"

    def get_class_attributes(self) -> List[str]:
        # obj = cls() # create an instance to access logger
        # obj.log.debug('[class attributes]')
        attr_list: List[str] = []
        for attribute in self.__class__.__dict__.keys():
            if not attribute.startswith("_"):
                value = getattr(self.__class__, attribute)
                if not callable(value):
                    attr_list.append(attribute)
        return attr_list

    def check_attributes(self, required_attributes: List[str]):
        class_attributes = self.get_class_attributes()
        errors = []
        for attr in required_attributes:
            if attr not in class_attributes:
                errors.append(f'Attribute: "{attr}" not found in config class: "{self.__class__}"')

        if errors:
            raise ValueError('\n'.join(errors))

    def get(self, attribute: str):
        return getattr(self.__class__, attribute)

#
# class MyConfig(ConfigBase):
#     CUMULATIVE: bool = True
#     PORTFOLIO_DOLLAR: float = 100_000
#     RISK_POSITION: float = 1_000
#     RISK_ALL_POSITIONS: float = 0.3
#     MAX_POSITIONS_PER_SECTOR: int = 0
#     MAX_MARGIN: float = 123.5
#     ATR_MUL: float = 5
#     PERIOD: int = 12 * 21
#
#     def check_state(self) -> bool:
#         return True
#
#
# if __name__ == "__main__":
#     import logging
#     logging.basicConfig(level=logging.DEBUG)
#
#     cfg = MyConfig()
#     print(cfg)
#
#     # cfg.check_attributes(['MAX_MARGINx', 'ATR_MULx', 'PERIODx'])
#
#     print("MAX_MARGIN", cfg.get('MAX_MARGINx'))
