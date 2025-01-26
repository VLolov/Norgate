import typing

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import logging

from Futures.Backtester.BacktesterBase import PlotBase
from Futures.Backtester.BacktesterFutures import ReportMulti
from Futures.Backtester.BacktesterFutures.plot_histogram_returns import plot_histogram_returns
from Futures.Backtester.BacktesterFutures.plot_qq_returns import plot_qq_returns

matplotlib.use("Qt5Agg")


class PlotMulti(PlotBase):
    def __init__(self, name: str, plot_histogram=True, plot_qq=True):
        super().__init__(name)
        self._plot_histogram = plot_histogram
        self._plot_qq = plot_qq

    def check_state(self) -> bool:
        return self.name != '' and self.report is not None

    def run(self):
        self.log.info("Creating plot multi")
        sns.set_style("whitegrid")
        logging.getLogger('matplotlib.font_manager').disabled = True
        logging.getLogger('matplotlib.ticker').disabled = True

        reporting = typing.cast(ReportMulti, self.report)
        assert reporting.ready, "ReportMulti not run"

        report_multi = reporting.get_report_multi()
        cumulative_df = report_multi.cumulative_df
        table_df = report_multi.table_df
        strategy_config = report_multi.config

        self.draw_chart(cumulative_df, strategy_config, table_df)
        daily_returns = cumulative_df['Total'].pct_change().fillna(0)
        # fix error: TypeError: Only valid with DatetimeIndex, TimedeltaIndex or PeriodIndex, but got an instance of 'Index'
        daily_returns.index = pd.to_datetime(daily_returns.index)
        if self._plot_histogram:
            plot_histogram_returns(daily_returns, resample_rule='ME')  # 'D', 'ME', 'W', 'QE'
        if self._plot_qq:
            plot_qq_returns(daily_returns, resample_rule='ME')  # 'D', 'ME'
        pass

    def draw_chart(self, cumulative_df, cfg, table_df):
        # sns.set_style('whitegrid')

        title = (
            r'$\bf{' + 'strategy'
            + (r'\ -\ Cumulative' if cfg.cumulative else '')
            + (r'\ -\ ' + cfg.sector if cfg.sector else '')
            + r'}$'
        )

        fig, ax = plt.subplot_mosaic('AFE;BFE;CFE;DFE', figsize=(12, 11), constrained_layout=True,
                                     width_ratios=[0.85, 0.001, 0.149])
        plt.suptitle(title)

        ax['A'].plot(cumulative_df['Total'], lw=1, label='Total')
        ax['A'].plot(cumulative_df['Total_Long'], lw=0.5, alpha=0.5, color='green', label='Long')
        ax['A'].plot(cumulative_df['Total_Short'], lw=0.5, alpha=0.5, color='red', label='Short')
        ax['A'].set_ylabel('Total $')
        ax['A'].legend()

        # print('Cumulative:')
        # print(tabulate(cumulative_df, headers='keys', tablefmt='psql'))

        log_return = 'log' if cfg.cumulative else 'linear'

        ax['A'].set_yscale(log_return)  # 'log' 'linear'
        ax['A'].plot([cumulative_df.index[0], cumulative_df.index[-1]],  # line beg..end
                     [cumulative_df['Total'].iloc[0], cumulative_df['Total'].iloc[-1]],  # pnl
                     'b', lw=0.5, alpha=0.5)

        if cfg.cumulative:
            dd = cumulative_df['Total'] / cumulative_df['Total'].cummax() - 1
        else:
            dd = (cumulative_df['Total'] - cumulative_df['Total'].cummax()) / cfg.portfolio_dollar

        ax['B'].plot(dd * 100, lw=1)
        ax['B'].set_ylabel('DD, %')

        ax['C'].plot(cumulative_df['Nr.Positions'], lw=1, label='Total')
        ax['C'].axhline(y=cumulative_df['Nr.Positions'].mean(), lw=1, color='orange', label='Average', linestyle='--')
        ax['C'].plot(cumulative_df['Nr.Positions_Long'], lw=0.5, alpha=0.5, color='green', label='Long')
        ax['C'].plot(cumulative_df['Nr.Positions_Short'], lw=0.5, alpha=0.5, color='red', label='Short')

        ax['C'].set_ylabel('Nr. Positions')

        ax['D'].plot(cumulative_df['Margin'], lw=1)
        ax['D'].axhline(y=cumulative_df['Margin'].mean(), lw=1, color='orange', label='Average', linestyle='--')
        ax['D'].set_ylabel('Margin USD')
        ax['D'].set_xlabel('Date')

        for sp in ['A', 'B', 'C', 'D']:
            ax[sp].grid(which='minor', color='grey', linestyle='-', alpha=0.1)
            ax[sp].grid(which='major', color='grey', linestyle='-', alpha=0.3)
            for axis in ['top', 'bottom', 'left', 'right']:
                ax[sp].spines[axis].set_linewidth(0.5)  # change width
                ax[sp].spines[axis].set_color('grey')  # change color

        # spacer
        ax['F'].patch.set_visible(False)
        ax['F'].axis('off')

        # create table
        # hide the axes
        ax['E'].patch.set_visible(False)
        ax['E'].axis('off')
        # ax['E'].axis('tight')
        # table_df.reset_index(drop=True, inplace=True)
        table_df = table_df.T.reset_index()
        table = ax['E'].table(cellText=table_df.values, colLabels=['Parameter', 'Value'], loc='center')
        table.scale(2, 2)
        # turn off the auto set text so you can set the font size
        # https://curbal.com/curbal-learning-portal/tables-in-matplotlib
        table.auto_set_font_size(False)
        table.set_fontsize(12)

        cell_dict = table.get_celld()
        # cellDict[(0, 0)].set_xy(0, 0)
        for i in range(0, len(table_df.columns)):
            cell_dict[(0, i)].set_height(.06)  # header height
            cell_dict[(0, i)].set_color('#efefef')
            cell_dict[(0, i)].set_linewidth(0.05)
            # cellDict[(0, i)].get_text().set_color('black')  # header font color
            # cellDict[(0, i)].set_edgecolor('#303546')
            for j in range(1, len(table_df) + 1):
                cell_dict[(j, i)].set_height(.03)  # row height
                cell_dict[(j, i)].set_linewidth(0.05)
                if j % 2 == 0:
                    cell_dict[(j, i)].set_color('#efefef')  # light grey

        # show negative values in red
        for j in range(1, len(table_df) + 1):
            val = cell_dict[(j, 1)].get_text()
            if '-' in val.get_text():
                cell_dict[(j, 1)].get_text().set_color('red')

        # mpl.cursor(hover=True)

        plt.show(block=False)

