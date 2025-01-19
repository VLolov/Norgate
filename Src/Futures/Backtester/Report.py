from Futures.BacktesterBase import ReportBase


class Report(ReportBase):
    def __init__(self, name: str):
        super().__init__(name)

    def check_state(self) -> bool:
        return self.name != '' and self.backtester is not None
