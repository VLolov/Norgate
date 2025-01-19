
from typing import List, Dict

import numpy as np
import pandas as pd

import Futures.BacktesterBase as Bb
from Futures.Backtester.Future import Future
from Futures.BacktesterBase import StrategyBase

from Futures.Backtester.Trade import Trade
from Futures.Backtester.Strategy import Strategy


class Broker(Bb.BrokerBase):
    def __init__(self):
        super().__init__()
        self.initial_capital: float = 0
        self.use_stop_loss = False
        self.use_stop_orders = False  # stop loss as stop order or on close

        self._trades: Dict[int, List[Trade]] = {}
        self._current_trades: Dict[int, Trade] = {}

    def check_state(self) -> bool:
        return True

    @staticmethod
    def _make_key(strategy: Bb.StrategyBase, instrument: Bb.InstrumentBase) -> int:
        return strategy.id * 10000 + instrument.id

    def trades(self, strategy, instrument) -> List[Trade]:
        key = self._make_key(strategy, instrument)
        return self._trades.get(key, [])

    def add_trade(self, strategy, instrument, trade):
        key = self._make_key(strategy, instrument)
        arr = self._trades.get(key, [])
        arr.append(trade)

    def get_current_trade(self, strategy, instrument):
        key = self._make_key(strategy, instrument)
        return self._current_trades.get(key, None)

    def set_current_trade(self, strategy, instrument, trade):
        key = self._make_key(strategy, instrument)
        self._current_trades[key] = trade

    def delete_current_trade(self, strategy, instrument):
        key = self._make_key(strategy, instrument)
        # Note: throws exception if key does not exist, so we know that something is wrong with our logic
        del self._current_trades[key]

    def market_position(self, strategy, instrument) -> int:
        current_trade = self.get_current_trade(strategy, instrument)
        if current_trade is None:
            # no current_trade, return 0 for flat
            return 0
        return current_trade.market_position

    def _update_rolls(self, strategy, instrument):
        current_trade = self.get_current_trade(strategy, instrument)
        if (current_trade is not None
                and current_trade.position != 0
                and strategy.is_roll(instrument)):
            current_trade.rolls += 1

    def _check_stop_loss(self, current_trade: Trade, strategy: Strategy):
        if current_trade is None or not self.use_stop_loss:
            return np.nan

        stop_level = np.nan

        if current_trade.position > 0:
            # long exit?
            if self.use_stop_orders:
                if strategy.low() < current_trade.stop_loss:
                    stop_level = current_trade.stop_loss
            else:
                if strategy.close() < current_trade.stop_loss:
                    stop_level = strategy.close()
        elif current_trade.position < 0:
            # short exit?
            if self.use_stop_orders:
                if strategy.high() > current_trade.stop_loss:
                    stop_level = current_trade.stop_loss
            else:
                if strategy.close() > current_trade.stop_loss:
                    stop_level = strategy.close()

        return stop_level

    def set_stop_loss(self, strategy, instrument, stop_loss: float):
        current_trade = self.get_current_trade(strategy, instrument)
        if current_trade:
            current_trade.stop_loss = stop_loss

    def open_position(self, strategy: StrategyBase, instrument: Future,
                      entry_date: pd.Timestamp,
                      entry_price: float,
                      position: float = 1, price: float = np.nan,
                      stop_loss: float = np.nan) -> None:
        assert self.get_current_trade(strategy, instrument) is None, "previous position is still open"
        # assert position == 0.0, "cannot open order with position: 0"

        if stop_loss is not np.nan:
            info = f"strategy: {strategy}, instrument: {instrument}, date: {entry_date}"
            if position > 0:
                assert stop_loss < entry_price, f"wrong stop loss price, long order, {info}"
            else:
                assert stop_loss > entry_price, f"wrong stop loss price, short order, {info}"

        trade = Trade(strategy=strategy, instrument=instrument,
                         entry_date=entry_date, entry_price=entry_price,
                         position=position, stop_loss=stop_loss)
        self.set_current_trade(strategy, instrument, trade)
        # self.current_trade.margin = margin * abs(position)

        self.add_trade(strategy, instrument, trade)
        self.update(strategy, instrument, check_stop_loss=False)

    def update(self, strategy, instrument, check_stop_loss=True) -> bool:
        # return true is stop loss executed
        current_trade = self.get_current_trade(strategy, instrument)

        # write date in list of current trade
        date = strategy.timestamp()
        if (current_trade is not None
                and date not in current_trade.trade_dates):
            current_trade.trade_dates.append(date)

        stop_loss_occurred = False
        if check_stop_loss:
            stop_level = self._check_stop_loss(current_trade, strategy)
            # @@@ stop loss
            if stop_level is not np.nan:
                self.close_position(strategy, instrument, price=stop_level, is_stop_loss=True)
                stop_loss_occurred = True

        if not stop_loss_occurred:
            self._update_rolls(strategy, instrument)

        return stop_loss_occurred

    def close_position(self, strategy: Strategy, instrument: Future,
                       idx: int,
                       price: float = np.nan, is_stop_loss: bool = False):
        current_trade = self.get_current_trade(strategy, instrument)
        assert current_trade is not None, \
            (f"No open position to be closed, "
             f"strategy: {strategy}, instrument: {instrument}, date: {strategy.timestamp(instrument, idx)}")
        exit_date = strategy.timestamp(instrument, idx)
        exit_price = price if price is not np.nan else strategy.close(instrument, idx)
        current_trade.close_trade(exit_date, exit_price, is_stop_loss)

        self.delete_current_trade(strategy, instrument)

    def days_since_entry(self, strategy, instrument):
        current_trade = self.get_current_trade(strategy, instrument)
        if current_trade is None:
            return -1
        return current_trade.dit


if __name__ == "__main__":
    # circular import: https://stackoverflow.com/questions/744373/what-happens-when-using-mutual-or-circular-cyclic-imports/33547682#33547682
    print('hi')



