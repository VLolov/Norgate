import logging
import typing
from datetime import datetime

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from Futures.Backtester.BacktesterBase import PlotBase
from Futures.Backtester.BacktesterFutures import ReportSingle
from Futures.Backtester.BacktesterFutures import Trade

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
            f'Atr.mul: {cfg.atr_multiplier}, Stop orders: {cfg.use_stop_orders}, '
            f'Cumulative: {cfg.cumulative}, Risk_position: {cfg.risk_position}, Front: {front}\n'
            f'Nr.trades: {report.nr_trades}, Missed: {report.nr_missed_trades}, Rolls: {report.nr_rolls},'
            f' Avg.trade: ${report.avg_trade:,.0f}, Avg.contracts: {report.avg_contracts:,.2f}, Avg.DIT: {report.avg_dit:.0f},'
            f' Avg.pos.size: ${report.avg_position_size_dollar:,.0f},'
            f' Avg.margin: ${report.avg_margin:,.0f}\n'
            f' Pnl, net: ${report.final_pnl:,.0f}, Costs: ${report.total_costs:,.0f}, '
            f' Yearly: {report.yearly_ret * 100:.2f} %, '
            f' Sharpe: {report.sharpe:.2f}'
        )

        fig, ax = plt.subplots(4, figsize=(12, 11), sharex='all')
        plt.suptitle(future_name)
        # ax[0].set_xlim(datetime(1980,1,1),datetime(2025,8,30)) does not work

        #
        #   Upper chart
        #
        # ax[0].plot(df['Close'], label='Close', color='#8c564b', lw=1.5)
        # ax[0].plot(df['High'], label='High', color='green', lw=0.5)
        # ax[0].plot(df['Low'], label='Low', color='red', lw=0.5)
        #
        if 'Up' in df and 'Down' in df and 'trailing_stop' in df:
            ax[0].plot(df['Up'], label='Up', color='blue', lw=1, alpha=0.5)
            ax[0].plot(df['Down'], label='Down', color='red', lw=1, alpha=0.5)
            ax[0].plot(df['trailing_stop'], label='Trailing Stop', linestyle=':', color='magenta')

        ax[0].plot(df['Close'], label='Close', color='#8c564b', lw=1.5)
        # ax[0].plot(df['Ema40'], label='Ema40', color='green', lw=0.5)
        # ax[0].plot(df['Ema80'], label='Ema80', color='red', lw=0.5)

        # ax[0].plot(df['Up'], label='Up', color='blue', lw=1, alpha=0.5)
        # ax[0].plot(df['Up'] - df['Std'] * 3, label='Up-Std', color='blue', lw=1, alpha=0.5)

        # ax[0].plot(df['ExitUp'], label='ExitUp', linestyle='--', color='green')
        # ax[0].plot(df['ExitDown'], label='ExitDown', linestyle='--', color='red')

        # Filter buy and sell signals
        buy_signals = df[df['Signal'] == 1]
        sell_signals = df[df['Signal'] == -1]

        # Graph buy and sell signals
        ax[0].scatter(buy_signals.index, buy_signals['Close'], marker='^', color='green', label='Buy signal', alpha=1)
        ax[0].scatter(sell_signals.index, sell_signals['Close'], marker='v', color='red', label='Sell signal', alpha=1)
        ax[0].scatter(df.index, df['StopLoss'], marker='x', color='magenta', label='Stop loss', alpha=1)

        # Mark missed trades
        missed_trades = df[df['MissedTrade']]
        ax[0].scatter(missed_trades.index, missed_trades['Close'], marker='o', color='Orange', label='Missed trade',
                      alpha=1, zorder=0, s=10)

        # plot color stripes under long and short trades
        trades = [typing.cast(Trade, t) for t in broker.trades if t.strategy == strategy and t.instrument == instrument]

        for trade in trades:
            left, right = trade.entry_date, trade.exit_date
            color = 'green' if trade.market_position > 0 else 'red'
            ax[0].axvspan(left, right, color=color, alpha=0.1, lw=0)

        # ax[0].set_title(f'Signals')
        ax[0].set_xlabel('Date')
        ax[0].set_ylabel('Signals')
        ax[0].legend(loc='upper left')

        #
        #   2nd chart
        #
        df[['Buy&Hold_Pnl', 'Strat_Pnl']].plot(ax=ax[1], lw=1.5)
        df[['Strat_Pnl_Long', 'Strat_Pnl_Short']].plot(ax=ax[1], lw=0.5, alpha=0.7)

        # ax[1].set_title(f'Performance USD')
        ax[1].set_xlabel('Date')
        ax[1].set_ylabel(f'PnL, {instrument.metadata.currency}')
        ax[1].get_yaxis().set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
        ax[1].legend(loc='upper left')

        #
        # 3rd chart
        #
        df['Margin'].plot(ax=ax[2], lw=1)

        # ax[2].set_title(f'Margin')
        ax[2].set_xlabel('Date')
        ax[2].set_ylabel(f'Margin, {instrument.metadata.currency}')
        ax[2].get_yaxis().set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
        ax[2].legend(loc='upper left')

        #
        # 4th chart
        #
        df['Contracts'].plot(ax=ax[3], lw=1)

        # ax[3].set_title(f'Nr. Contracts')
        ax[3].set_xlabel('Date')
        ax[3].set_ylabel('Nr. Contracts')
        # ax[3].legend(loc='upper left')

        plt.show()

