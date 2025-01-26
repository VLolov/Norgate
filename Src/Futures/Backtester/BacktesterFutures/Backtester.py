import Futures.Backtester.BacktesterBase as Bb


class Backtester(Bb.BacktesterBase):
    def check_state(self) -> bool:
        return len(self.reports) > 0 and self.portfolio is not None and len(self.groups) > 0

    def run(self):
        for group in self.groups:
            group.run()

        # run reports first, so that data is available in plots
        for report in self.reports:
            report.run()

        for report in self.reports:
            for plot in report.plots:
                plot.run()


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.DEBUG)

    bt = Backtester()

    Bb.Base.print_instances()
    bt.print_hierarchy()
