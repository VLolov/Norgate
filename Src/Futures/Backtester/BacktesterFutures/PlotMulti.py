import typing

import matplotlib

from Futures.Backtester.BacktesterBase import PlotBase
from Futures.Backtester.BacktesterFutures import ReportMulti

matplotlib.use("Qt5Agg")


class PlotMulti(PlotBase):
    def __init__(self, name: str):
        super().__init__(name)

    def check_state(self) -> bool:
        return self.name != '' and self.report is not None

    def run(self):
        self.log.info("Creating plot multi")
        reporting = typing.cast(ReportMulti, self.report)

        # report = reporting.get_first_report()   # @@@
        # self.plot_performance(report)

        # for report in reporting.get_all_reports():
        #     self.plot_performance(report)
        pass
