import typing
from typing import List, Dict

import numpy as np

import Futures.Backtester.BacktesterBase as Bb
from .Trade import Trade


class Broker(Bb.BrokerBase):
    def __init__(self):
        super().__init__()
        self.use_stop_loss = False
        self.use_stop_orders = False  # stop loss as stop order or on close

        self._current_trade: Dict[int, Trade] = {}

    def setup(self, use_stop_loss, use_stop_orders):
        self.use_stop_loss = use_stop_loss
        self.use_stop_orders = use_stop_orders
        return self

    def check_state(self) -> bool:
        return True

    def get_current_trade(self, strategy: Bb.StrategyBase, instrument: Bb.InstrumentBase):
        key = self._make_key(strategy, instrument)
        return self._current_trade.get(key, None)

    def set_current_trade(self, strategy: Bb.StrategyBase, instrument: Bb.InstrumentBase, trade):
        key = self._make_key(strategy, instrument)
        self._current_trade[key] = trade
        pass

    def delete_current_trade(self, strategy: Bb.StrategyBase, instrument: Bb.InstrumentBase) -> None:
        key = self._make_key(strategy, instrument)
        # Note: throws exception if key does not exist, so we know that something is wrong with our logic
        del self._current_trade[key]

    def market_position(self, strategy: Bb.StrategyBase, instrument: Bb.InstrumentBase) -> int:
        current_trade = self.get_current_trade(strategy, instrument)
        if current_trade is None:
            # no current_trade, return 0 for flat
            return 0
        return current_trade.market_position

    def _update_rolls(self, current_trade: Trade) -> None:
        strategy = current_trade.strategy
        instrument = current_trade.instrument
        if current_trade.position != 0 and strategy.is_roll(instrument, strategy.idx):
            current_trade.rolls += 1

    def _check_stop_loss(self, current_trade: Trade) -> float:
        strategy = current_trade.strategy
        instrument = current_trade.instrument
        idx = strategy.idx

        stop_level = np.nan
        if current_trade.position > 0:
            # long exit?
            if self.use_stop_orders:
                if strategy.low(instrument, idx) < current_trade.stop_loss:
                    stop_level = current_trade.stop_loss
            else:
                if strategy.close(instrument, idx) < current_trade.stop_loss:
                    stop_level = strategy.close(instrument, idx)
        elif current_trade.position < 0:
            # short exit?
            if self.use_stop_orders:
                if strategy.high(instrument, idx) > current_trade.stop_loss:
                    stop_level = current_trade.stop_loss
            else:
                if strategy.close(instrument, idx) > current_trade.stop_loss:
                    stop_level = strategy.close(instrument, idx)

        return stop_level

    def set_stop_loss(self, strategy: Bb.StrategyBase, instrument: Bb.InstrumentBase, stop_loss: float):
        current_trade = self.get_current_trade(strategy, instrument)
        if current_trade:
            current_trade.stop_loss = stop_loss

    def get_stop_loss(self, strategy: Bb.StrategyBase, instrument: Bb.InstrumentBase) -> float:
        # if no current_trade, return 0
        stop_loss = 0.0
        current_trade = self.get_current_trade(strategy, instrument)
        if current_trade:
            stop_loss = current_trade.stop_loss
        return stop_loss

    def open_position(self,
                      strategy: Bb.StrategyBase,
                      instrument: Bb.InstrumentBase,
                      position: float = 1,
                      price: float = np.nan,
                      stop_loss: float = np.nan,
                      margin: float = 0,
                      momentum: float = 0) -> None:
        assert self.get_current_trade(strategy, instrument) is None, "previous position is still open"
        # assert position == 0.0, "cannot open order with position: 0"

        entry_date = strategy.dt
        entry_price = price if price is not np.nan else strategy.close(instrument, strategy.idx)

        if stop_loss is not np.nan:
            error_message = (f"strategy: {strategy}, instrument: {instrument}, date: {entry_date}, "
                             f"position: {position}, entry price: {entry_price}, stop loss: {stop_loss}")
            if position > 0:
                assert stop_loss < entry_price, f"wrong stop loss price, long order, {error_message}"
            else:
                assert stop_loss > entry_price, f"wrong stop loss price, short order, {error_message}"

        trade = Trade(strategy=strategy,
                      instrument=instrument,
                      entry_date=entry_date,
                      entry_price=entry_price,
                      position=position,
                      stop_loss=stop_loss,
                      margin=margin,
                      momentum=momentum)

        self.set_current_trade(strategy, instrument, trade)
        trade.margin = margin * abs(position)

        self._trades.append(trade)
        self.update(strategy, instrument, check_stop_loss=False)

    def update(self, strategy: Bb.StrategyBase, instrument: Bb.InstrumentBase, check_stop_loss=True) -> bool:
        # return true is stop loss executed
        stop_loss_occurred = False

        current_trade = self.get_current_trade(strategy, instrument)

        # update close_price, so we have current pnl of trade, although it is not closed yet
        if current_trade is not None:
            current_trade.exit_price = strategy.close(instrument, strategy.idx)

            # write date in list of current trade
            date = strategy.dt
            if date not in current_trade.trade_dates:
                current_trade.trade_dates.append(date)

            if check_stop_loss and self.use_stop_loss:
                stop_level = self._check_stop_loss(current_trade)
                if stop_level is not np.nan:
                    self.close_position(strategy, instrument, price=stop_level, is_stop_loss=True)
                    stop_loss_occurred = True

            if not stop_loss_occurred:
                self._update_rolls(current_trade)

        return stop_loss_occurred

    def close_position(self,
                       strategy: Bb.StrategyBase,
                       instrument: Bb.InstrumentBase,
                       price: float = np.nan,
                       is_stop_loss: bool = False) -> float:

        current_trade = self.get_current_trade(strategy, instrument)
        if current_trade is None:
            # no position to be closed
            return 0.0
        assert current_trade is not None, \
            (f"No open position to be closed, "
             f"strategy: {strategy}, instrument: {instrument}, date: {strategy.dt}")
        exit_date = strategy.dt
        exit_price = price if price is not np.nan else strategy.close(instrument, strategy.idx)
        current_trade.close_trade(exit_date, exit_price, is_stop_loss)

        self.delete_current_trade(strategy, instrument)
        return current_trade.pnl

    @staticmethod
    def _make_key(strategy: Bb.StrategyBase, instrument: Bb.InstrumentBase) -> int:
        return strategy.id * 100000 + instrument.id

    def days_since_entry(self, strategy: Bb.StrategyBase, instrument: Bb.InstrumentBase):
        current_trade = self.get_current_trade(strategy, instrument)
        if current_trade is None:
            return -1
        return current_trade.dit

    def closed_pnl(self, strategy: Bb.StrategyBase):
        closed_trades = [trade for trade in self._trades
                         if trade.strategy == strategy and trade.is_closed]
        pnl = sum([trade.pnl * trade.instrument.metadata.big_point for trade in closed_trades])
        return pnl

    def open_trades(self, strategy: Bb.StrategyBase):
        open_trades = [trade for trade in self._trades
                       if trade.strategy == strategy and not trade.is_closed]
        return open_trades



if __name__ == "__main__":
    # circular import: https://stackoverflow.com/questions/744373/what-happens-when-using-mutual-or-circular-cyclic-imports/33547682#33547682
    print('hi')



