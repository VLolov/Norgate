import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm


def plot_qq_returns(daily_returns: pd.Series, resample_rule='D'):

    data = daily_returns.dropna()
    if resample_rule != "D":
        data = data.resample(resample_rule).agg(lambda x: (x + 1).prod() - 1)

    # https://medium.com/@sachinsoni600517/a-comprehensive-guide-to-the-normal-distribution-7cab1361188e
    # Create a QQ plot of the two sets of data
    # line = 45 which mean you want to create 45 degree line.
    fig = sm.qqplot(data, line='45', fit=True, markersize=3, alpha=1)

    names = {
        'D': 'daily',
        'W': 'weekly',
        'ME': 'monthly',
        'QE': 'quarterly',
        'YE': 'yearly'
    }
    returns_str = names.get(resample_rule, '???')

    title = f'QQ plot of {returns_str} returns'

    plt.suptitle(title)
    # plt.title('QQ Plot')
    plt.xlabel('Theoretical Quantiles')
    plt.ylabel('Sample Quantiles')

    # Show the plot
    plt.show()
    plt.grid(True)
    plt.show()
