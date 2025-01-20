import logging

from Futures.Backtester.logutil import logger

from Futures.Backtester.Future import Future, get_futures
from Futures.Backtester.Report import Report
from Futures.Backtester.Strategy import Strategy
from Futures.Backtester.Broker import Broker
from Futures.Backtester.Backtester import Backtester
from Futures.Backtester.Group import Group
from Futures.Backtester.Portfolio import Portfolio
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

    broker = Broker().set_group(gr)
    gr.set_broker(broker)

    report = Report("My Report").set_backtester(bt)
    bt.add_report(report)

    strategy = Strategy("My Strategy")

    gr.add_strategy(strategy)
    strategy.set_group(gr)

    futures = get_futures(start_date='1020-01-01', end_date='2024-01-10')
    for future in futures:
        gr.add_instrument(future)

    # ts = pd.Timestamp.min
    # trade_x = TradeX(strategy_x, instrument_x, ts, 1000, ts, 2000, 1)
    # trade_y = TradeX(strategy_y, instrument_y, ts, 1000, ts, 2000, -1)
    # broker.add_trade(trade_x).add_trade(trade_y)

    # Base.print_instances()
    p.print_instances()

    bt.print_hierarchy()

    bt.run()


if __name__ == "__main__":
    log = logging.getLogger(__name__)
    log.handlers.clear()

    logger(level=logging.DEBUG)
    with Timer(printer=log.info):
        main()
