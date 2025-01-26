import logging
from dataclasses import dataclass

from Futures.Backtester.BacktesterFutures import *
from Futures.Backtester.StrategyLoosePants import StrategyLoosePants
from Futures.Backtester.logutil import logger


from Futures.TrendFollowing.Timer import Timer


def main():
    log.info("Starting main()")

    # logging.basicConfig(level=logging.DEBUG)
    bt = Backtester()

    p = Portfolio()
    bt.set_portfolio(p)
    p.set_initial_capital(100_000).set_backtester(bt)

    group = Group("My first group")
    group.add_backtester(bt)
    bt.add_group(group)

    broker = Broker()
    broker.set_group(group)
    group.set_broker(broker)

    report_single = ReportSingle("My Single Report").set_backtester(bt)
    bt.add_report(report_single)
    plot_single = PlotSingle("My Single Plot").set_report(report_single)
    report_single.add_plot(plot_single)

    report_multi = ReportMulti("My Multi report_multi").set_backtester(bt)
    report_multi.set_report_single(report_single)
    bt.add_report(report_multi)
    plot_multi = PlotMulti("My Multi Plot", plot_histogram=False, plot_qq=False)
    plot_multi.set_report(report_multi)
    report_multi.add_plot(plot_multi)

    strategy = StrategyLoosePants()
    strategy.set_config(StrategyLoosePants.MyConfig())

    group.add_strategies(strategy)
    strategy.set_group(group)

    selected_symbols = ['CL', 'ES', 'GC']
    selected_symbols = []
    futures = get_futures(start_date='1980-01-01', end_date='3024-03-20', selected_symbols=selected_symbols)
    group.add_instruments(futures)

    # bt.print_instances()
    # bt.print_hierarchy()

    assert bt.check_all_states(), "Some states are False. Can't run"

    with Timer(printer=log.info, text="bt.run()"):
        bt.run()


if __name__ == "__main__":
    log = logging.getLogger(__name__)
    logger(level=logging.DEBUG)  # , logfile="c:/tmp/mylog.txt")

    # send all exceptions to logger:
    def except_hook(*args):
        log.error('Uncaught exception:', exc_info=args)

    import sys
    sys.excepthook = except_hook

    main()


@dataclass
class Config1:
    i: int = 1


@dataclass
class Config2(Config1):
    k: int = 3


@dataclass
class Config3(Config2):
    i: int = 3  # error - c3.int remains = 1 ?!?
    l: int = 4


c1 = Config1()
c2 = Config2()
c3 = Config3()

pass
