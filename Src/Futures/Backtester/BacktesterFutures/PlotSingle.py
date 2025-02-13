import logging
import typing
from datetime import datetime

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from Futures.Backtester.BacktesterBase import PlotBase
from .ReportSingle import ReportSingle
from .Trade import Trade

matplotlib.use("Qt5Agg")


class PlotSingle(PlotBase):
    def __init__(self, name: str):
        super().__init__(name)

    def check_state(self) -> bool:
        return self.name != '' and self.report is not None

    def run(self):
        reporting = typing.cast(ReportSingle, self.report)
        assert reporting.ready, "ReportSingle not run"

        report = reporting.get_first_report()   # @@@
        self.plot_performance(report)

        # for report in reporting.get_all_reports():
        #     self.plot_performance(report)
        pass

    @staticmethod
    def plot_performance(report, front=1):
        # front is used only to print information on the chart
        sns.set_style("whitegrid")
        logging.getLogger('matplotlib.font_manager').disabled = True

        strategy = report.strategy
        instrument = report.instrument
        broker = strategy.group.broker

        df = instrument.data

        # big_point = self.big_point
        cfg = strategy.config

        future_name = (
            f'{instrument}\n'
            f'Atr.mul: {cfg.atr_multiplier}, Stop orders: {broker.use_stop_orders}, '
            f'Cumulative: {cfg.cumulative}, Risk_position: {cfg.risk_position}, Front: {front}\n'
            f'Nr.trades: {report.nr_trades}, Missed: {report.nr_missed_trades}, Rolls: {report.nr_rolls},'
            f' Avg.trade: ${report.avg_trade:,.0f}, Avg.contracts: {report.avg_contracts:,.2f}, Avg.DIT: {report.avg_dit:.0f},'
            f' Avg.pos.size: ${report.avg_position_size_dollar:,.0f},'
            f' Avg.margin: ${report.avg_margin:,.0f}\n'
            f' Pnl, net: ${report.final_pnl:,.0f}, Costs: ${report.total_costs:,.0f}, '
            f' Yearly: {report.yearly_ret * 100:.2f} %, '
            f' Sharpe: {report.sharpe:.2f}'
        )

        fig, (ax_signals, ax_pnl, ax_margin, ax_contracts) = plt.subplots(4, figsize=(12, 11), sharex='all')

        plt.suptitle(future_name)

        #
        #   Upper chart
        #
        # ax_signals.plot(df['Close'], label='Close', color='#8c564b', lw=1.5)
        # ax_signals.plot(df['High'], label='High', color='green', lw=0.5)
        # ax_signals.plot(df['Low'], label='Low', color='red', lw=0.5)
        #

        # plot Close first to get full x-axis range !!!
        ax_signals.plot(df['Close'], label='Close', color='#8c564b', lw=1.5)

        if 'Up' in df and 'Down' in df and 'trailing_stop' in df:
            ax_signals.plot(df['Up'], label='Up', color='blue', lw=1, alpha=0.5)
            ax_signals.plot(df['Down'], label='Down', color='red', lw=1, alpha=0.5)
            ax_signals.plot(df['trailing_stop'], label='Trailing Stop', linestyle=':', color='magenta')

        # ax_signals.plot(df['Ema40'], label='Ema40', color='green', lw=0.5)
        # ax_signals.plot(df['Ema80'], label='Ema80', color='red', lw=0.5)

        # ax_signals.plot(df['Up'], label='Up', color='blue', lw=1, alpha=0.5)
        # ax_signals.plot(df['Up'] - df['Std'] * 3, label='Up-Std', color='blue', lw=1, alpha=0.5)

        # ax_signals.plot(df['ExitUp'], label='ExitUp', linestyle='--', color='green')
        # ax_signals.plot(df['ExitDown'], label='ExitDown', linestyle='--', color='red')

        # Filter buy and sell signals
        buy_signals = df[df['Signal'] == 1]
        sell_signals = df[df['Signal'] == -1]

        # Graph buy and sell signals
        ax_signals.scatter(buy_signals.index, buy_signals['Close'], marker='^', color='green', label='Buy signal', alpha=1)
        ax_signals.scatter(sell_signals.index, sell_signals['Close'], marker='v', color='red', label='Sell signal', alpha=1)
        ax_signals.scatter(df.index, df['StopLoss'], marker='x', color='magenta', label='Stop loss', alpha=1)

        # Mark missed trades
        missed_trades = df[df['MissedTrade']]
        ax_signals.scatter(missed_trades.index, missed_trades['Close'], marker='o', color='Orange', 
                           label='Missed trade', alpha=1, zorder=0, s=10)

        # plot color stripes under long and short trades
        trades = [typing.cast(Trade, t) for t in broker.trades if t.strategy == strategy and t.instrument == instrument]

        for trade in trades:
            left, right = trade.entry_date, trade.exit_date
            color = 'green' if trade.market_position > 0 else 'red'
            ax_signals.axvspan(left, right, color=color, alpha=0.1, lw=0)

        # ax_signals.set_title(f'Signals')
        ax_signals.set_ylabel('Signals')
        ax_signals.legend(loc='upper left')

        #
        #   2nd chart
        #

        # NOTE: this messes up the x-axis:
        # df[['Buy&Hold_Pnl', 'Strat_Pnl']].plot(ax=ax_pnl, lw=1.5)
        # df[['Strat_Pnl_Long', 'Strat_Pnl_Short']].plot(ax=ax_pnl, lw=0.5, alpha=0.7)

        for col in ['Buy&Hold_Pnl', 'Strat_Pnl']:
            ax_pnl.plot(df[col], lw=1.5, label=col)

        for col in ['Strat_Pnl_Long', 'Strat_Pnl_Short']:
            ax_pnl.plot(df[col], lw=0.5, alpha=0.5, label=col)

        # ax_pnl.set_title(f'Performance USD')
        ax_pnl.set_ylabel(f'PnL, {instrument.metadata.currency}')
        ax_pnl.get_yaxis().set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
        ax_pnl.legend(loc='upper left')

        #
        # 3rd chart
        #
        ax_margin.plot(df['Margin'], lw=1)

        # ax_margin.set_title(f'Margin')
        ax_margin.set_ylabel(f'Margin, {instrument.metadata.currency}')
        ax_margin.get_yaxis().set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

        #
        # 4th chart
        #
        ax_contracts.plot(df['Contracts'], lw=1)

        # ax_contracts.set_title(f'Nr. Contracts')
        ax_contracts.set_xlabel('Date')
        ax_contracts.set_ylabel('Nr. Contracts')
        # ax_contracts.legend(loc='upper left')

        plt.show()

