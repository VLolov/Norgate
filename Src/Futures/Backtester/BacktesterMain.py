import logging

from Futures.Backtester.BacktesterFutures import *
from Futures.Backtester.StrategyBuyAndHold import StrategyBuyAndHold
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

    group = Group("First Group")
    group.add_backtester(bt)
    bt.add_group(group)

    broker = Broker()
    broker.set_group(group)
    group.set_broker(broker)
    broker.setup(use_stop_loss=True, use_stop_orders=True)

    report_single = ReportSingle("Single Report")
    report_single.set_backtester(bt)
    bt.add_report(report_single)

    # plot_single = PlotSingle("My Single Plot").set_report(report_single)
    # report_single.add_plot(plot_single)

    report_multi = ReportPortfolio("Portfolio Report", verbose=True).set_backtester(bt)
    report_multi.set_report_single(report_single)
    bt.add_report(report_multi)
    plot_multi = PlotPortfolio("Portfolio Plot", plot_histogram=False, plot_qq=False)
    plot_multi.set_report(report_multi)
    report_multi.add_plot(plot_multi)

    # for strategy_class in [StrategyBuyAndHold]:
    # for strategy_class in [StrategyLoosePants]:
    for strategy_class in [StrategyLoosePants, StrategyBuyAndHold]:
        strategy = strategy_class()
        group.add_strategies(strategy)
        strategy.set_group(group)

    selected_symbols = ['GC', 'CL', 'ES']
    # selected_symbols = []
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
