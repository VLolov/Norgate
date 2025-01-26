import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats


def plot_histogram_returns(daily_returns: pd.Series, resample_rule='D'):
    """
    Skewness
    = In a normal distribution, the mean divides the curve symmetrically into two equal parts at the median and
        the value of skewness is zero.
    = When the value of the *skewness is negative*, the tail of the distribution is longer towards
        the left hand side of the curve.
    = When the value of the *skewness is positive*, the tail of the distribution is longer towards
        the right hand side of the curve.

    Kurtosis
    = If the distribution is tall and thin it is called a leptokurtic distribution(Kurtosis > 3).
        Values in a leptokurtic distribution are near the mean or at the extremes.
    = A flat distribution where the values are moderately spread out is called platykurtic(Kurtosis <3) distribution.
    = A distribution whose shape is in between a leptokurtic distribution and a platykurtic distribution is called
        a mesokurtic(Kurtosis=3) distribution. A mesokurtic distribution looks more close to a normal distribution.
    """
    #plt.figure()

    data = daily_returns.dropna().loc[(daily_returns < -0.00001) | (0.00001 < daily_returns)]
    if resample_rule != "D":
        data = data.resample(resample_rule).agg(lambda x: (x + 1).prod() - 1)

    fig, ax = plt.subplots()
    # kde=Kernel Density Estimation
    sns.histplot(data, bins='auto', kde=True, stat='density', label='samples')
    # sns.histplot(data, bins='auto', stat='count', kde=True, label='samples')
    # ax.set_xlim(-0.05, 0.05)
    # ax.set_xlim(-0.3, 0.3)

    mu, std = stats.norm.fit(data)

    # uncomment following lines to get normal distribution plot
    x0, x1 = ax.get_xlim()  # extract the endpoints for the x-axis
    x_pdf = np.linspace(x0, x1, 100)
    y_pdf = stats.norm.pdf(x_pdf, mu, std)  # Probability density function
    ax.plot(x_pdf, y_pdf, 'r', label='pdf')

    names = {
        'D': 'daily',
        'W': 'weekly',
        'ME': 'monthly',
        'QE': 'quarterly',
        'YE': 'yearly'
    }

    returns_str = names.get(resample_rule, '???')

    title = f'Histogram of {returns_str} returns'

    ax.legend(['kde', 'normal', returns_str])
    # #np.random.seed(9001)
    # gausian = np.random.normal(loc=0, scale=1, size=10000000)
    plt.suptitle(f'{title}', y=1.001)
    plt.title(f'Mean: {mu:.4f}, SD: {std:.4f} '
              f'Skew: {data.skew():.4f}, Kurtosis: {data.kurtosis():.4f}')
              # f'Skew: {data.skew() / np.sqrt(252):.5f}, Ex.Kurtosis: {data.kurtosis() / 252:.5f}')

    plt.grid(True)
    plt.show()
