import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import logging

from tabulate import tabulate

log = logging.getLogger(__name__)

sns.set(color_codes=True)
# sns.set_style("whitegrid")
sns.set_style("white")

# avoid matplotlib warnings
# import Common.Utility.display_setup


class PlotPerformance(object):
    @classmethod
    def rets_to_equity(cls, returns):
        # Calculate equity curve from returns. Set the starting value of every curve to 1
        eq = (1 + returns).cumprod(axis=0)

        return eq

    @classmethod
    def cagr(cls, equity, days=252):
        periods = len(equity) / days
        return (equity.iloc[-1] / equity.iloc[0]) ** (1 / periods) - 1

    @classmethod
    def sharpe(cls, returns, days=252):
        er = np.mean(returns)
        std = np.std(returns)
        sharpe_ratio = 0
        if std:
            sharpe_ratio = er / np.std(returns) * np.sqrt(days)

        return sharpe_ratio

    @classmethod
    def volatility(cls, returns, days=252):
        return returns.std() * np.sqrt(days)

    @classmethod
    def drawdown(cls, equity):
        eq_series = pd.Series(equity)
        dd = eq_series / eq_series.cummax() - 1
        return dd

    @classmethod
    def max_drawdown(cls, equity, percent=True):
        abs_drawdown = np.abs(cls.drawdown(equity))
        max_dd = np.max(abs_drawdown)
        return -max_dd * 100 if percent else -max_dd

    @classmethod
    def mar(cls, equity):
        max_dd = cls.max_drawdown(equity, percent=False)
        mar = cls.cagr(equity) / abs(max_dd) if max_dd != 0 else 0
        return mar

    @classmethod
    def equity_plot(cls, title, label_bench, returns_bench, label_strat, returns_strat,
                    states=None, states_title='Positions/States',
                    nr_strategies=None, colors=None,
                    days=252, scale='log'):

        assert scale in ['log', 'linear'], "Wrong scale value"

        sns.set_style("whitegrid")

        nr_subplots = 2 + (1 if states is not None else 0)
        nr_subplots += (1 if nr_strategies is not None else 0)
        subplot_idx = 0

        fig, ax = plt.subplots(nr_subplots, figsize=(12, 11), sharex='all')
        # y=... move suptitle up, so more lines can be shown without overlapping
        plt.suptitle(title or 'Performance', y=1.001)

        eq_bench = cls.rets_to_equity(returns_bench)
        eq_strat = cls.rets_to_equity(returns_strat)
        dd_bench = cls.drawdown(eq_bench) * 100
        dd_strat = cls.drawdown(eq_strat) * 100
        sharpe_bench = cls.sharpe(returns_bench, days=days)
        sharpe_strat = cls.sharpe(returns_strat, days=days)

        volatility_bench = cls.volatility(returns_bench, days=days) * 100
        volatility_strat = cls.volatility(returns_strat, days=days) * 100

        cagr_bench = cls.cagr(eq_bench, days=days) * 100
        cagr_strat = cls.cagr(eq_strat, days=days) * 100

        max_dd_bench = cls.max_drawdown(eq_bench)
        max_dd_strat = cls.max_drawdown(eq_strat)

        # plot PnL
        ax[subplot_idx].plot(eq_bench, label=label_bench)
        ax[subplot_idx].set_yscale(scale)  # 'log' | 'linear'

        # minor grid on pnl chart only
        # ax[subplot_idx].minorticks_on()
        ax[subplot_idx].grid(which='minor', color='grey', linestyle='-', alpha=0.2)

        if colors is not None:
            ax[subplot_idx].scatter(x=eq_bench.index, y=eq_bench, c=colors, s=9)

        ax[subplot_idx].plot(eq_strat, label=label_strat)

        ax[subplot_idx].set_title(
            f'CAGRs: {cagr_bench:.2f} / {cagr_strat:.2f}, '
            f'Sharpes: {sharpe_bench:.2f} / {sharpe_strat:.2f}, '
            f'MaxDDs: {max_dd_bench:.2f} / {max_dd_strat:.2f} '
            f'Volas: {volatility_bench:.2f} / {volatility_strat:.2f}'
        )

        ax[subplot_idx].legend()        # legend is shown only on first subplot

        subplot_idx += 1

        if states is not None:
            # plot states
            ax[subplot_idx].plot(states)
            ax[subplot_idx].scatter(x=states.index, y=states, c=colors, s=3, label='Positions/States')
            ax[subplot_idx].set_title(states_title)
            subplot_idx += 1

        if nr_strategies is not None:
            # plot states
            ax[subplot_idx].plot(nr_strategies)
            ax[subplot_idx].set_title('Number of Strategies')
            subplot_idx += 1


        # plot DrawDown
        # ax[current_subplot].set_yscale('log')
        # dd.plot(ax=ax[1], kind='area')
        ax[subplot_idx].plot(dd_bench, linewidth=1, label=label_bench, alpha=1)
        ax[subplot_idx].plot(dd_strat, linewidth=1, label=label_strat, alpha=1)
        ax[subplot_idx].fill_between(dd_strat.index, dd_strat, where=dd_strat < 0, facecolor='pink',
                                     alpha=0.7)  # filling of drawdown
        # ax[current_subplot].fill_between(dd1.index, dd1, where=dd1 < 0, facecolor='orange', alpha=0.7) # filling of drawdown

        # horizontal lines at max DD
        ax[subplot_idx].axhline(y=dd_bench.min(), color='#47B', linestyle=':')
        ax[subplot_idx].axhline(y=dd_strat.min(), color='#D85', linestyle=':')

        ax[subplot_idx].set_title('Drawdown (%)')

        # plt.tight_layout()
        # plt.savefig(f'c:/tmp/{ticker} pnl_chart.png')
        # plt.show()

    @classmethod
    def yearly_plot(cls, title, label1, returns_series1, label2, returns_series2,
                    show_difference=False, print_performance=False):
        sns.set_style("whitegrid")
        df = pd.DataFrame({label1: returns_series1, label2: returns_series2})
        yearly_df = df.resample('YE').agg(lambda x: (x + 1).prod() - 1)
        yearly_df.index = yearly_df.index.year
        if show_difference:
            yearly_df['strategy-benchmark'] = yearly_df['strategy'] - yearly_df['benchmark']
        if print_performance:
            # print(yearly_df)
            print("Performance by year:")
            print(tabulate(yearly_df, headers='keys', tablefmt='psql', showindex=True, floatfmt='.2f'))

        fig, ax = plt.subplots(figsize=(6, 4))
        yearly_df.plot.barh(width=0.8, ax=ax)
        ax.set_ylabel('year')
        ax.set_xlabel('performance')
        plt.suptitle(title or 'Yearly')
        # plt.tight_layout(pad=0.5)
        # ax.format_coord = lambda x, y: "Performance={:6.3f}, Year={}".format(x, y)

    @classmethod
    def symbols_plot(cls, merged_sym, title=''):
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning)
        sns.set_style("whitegrid")
        axs = merged_sym.plot(subplots=True, title=f'{title}: Merged symbols')
        [ax.legend(loc='upper left') for ax in axs]
        # plt.tight_layout(pad=2)

    @classmethod
    def allocation_plot(cls, df, monthly=False, title=''):

        sns.set_style("whitegrid")
        # df = df.copy()
        # df = df.resample('W-FRI').apply(lambda x: x.tail(1))
        # df = df.resample('MS').mean()
        if monthly:
            df = df.resample('M').last()

        fig, ax = plt.subplots(figsize=(12, 9))
        # bar plot looks nice, but can not be plotted more often than monthly
        # df.plot(kind='bar', stacked=True, title=f"{title}: Asset Allocation", width=1, ax=ax, picker=5)
        df.plot(kind='area', stacked=True, title=f"{title}: Asset Allocation", ax=ax, picker=5, linewidth=0)
        plt.axhline(y=1, color='red', linestyle='-', linewidth=0.5)
        # this does not work correctly
        # ax.set_xticklabels([pandas_datetime.strftime("%Y-%m-%d") for pandas_datetime in df.index])

        ax.set_axisbelow(True)
        ax.set_ylabel('Allocation %')

    @classmethod
    def save_equity_as_file(cls, strategy_rets, full_path):
        eq1 = PlotPerformance.rets_to_equity(strategy_rets)
        df = pd.DataFrame()
        df['close'] = eq1
        df.index.name = 'timestamp'
        # new data
        filename = os.path.join(full_path)
        df.reset_index(inplace=True)
        df.to_csv(filename, index=False, header=True, float_format="%g")
        log.info(f'Equity saved under {full_path}')

    @classmethod
    def equal_weight_portfolio(cls, df):
        benchmark = df.iloc[:, 0]  # first symbol is the benchmark
        traded = df.iloc[:, 1:]
        nr_symbols = traded.count(axis=1)
        weights = 1.0 / nr_symbols
        rets = traded.pct_change(fill_method=None).fillna(0)
        strategy_returns = rets.mul(weights, axis=0).sum(axis=1)
        benchmark_returns = benchmark.pct_change().fillna(0)
        cls.equity_plot(r'$\bf{Equal\ Weight\ Portfolio}$' + '\n' +
                        f'\nBenchmark: {benchmark.name}, Traded: {", ".join(traded.columns)}',

                        'benchmark', benchmark_returns,
                        'strategy', strategy_returns)
        cls.yearly_plot(r'$\bf{Equal\ Weight\ Portfolio}$', 'benchmark', benchmark_returns, 'strategy',
                        strategy_returns)
