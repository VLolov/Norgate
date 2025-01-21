import logging

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

    gr = Group("My first group")
    gr.add_backtester(bt)
    bt.add_group(gr)

    broker = Broker()
    broker.set_group(gr)
    gr.set_broker(broker)

    report = Report("My Report").set_backtester(bt)
    bt.add_report(report)

    plot = Plot("Matplot").set_report(report)
    report.add_plot(plot)

    strategy = StrategyLoosePants()

    gr.add_strategy(strategy)
    strategy.set_group(gr)

    futures = get_futures(start_date='1020-01-01', end_date='3024-03-20')
    for future in futures:
        gr.add_instrument(future)

    # Base.print_instances()
    # p.print_instances()

    bt.print_hierarchy()
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
