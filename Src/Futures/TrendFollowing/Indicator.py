import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Qt5Agg")


class Indicator:
    @classmethod
    def atr(cls, df, period) -> pd.Series:
        """
        True Range calculation, see:
        https://stackoverflow.com/questions/40256338/calculating-average-true-range-atr-on-ohlc-data-with-python
        Just a disclaimer about TR calculation: it actually is the biggest value of those 3 options:

        Present High - Present Low;
        abs(Present High - Last Close);
        abs(Present Low - Last Close);

        I did this in order to avoid creating other columns on my dataframe, so I basically created a list of tuples,
        each tuple containing the 3 possible values listed above, and then created a list of the biggest value of each tuple.

        """
        atr_ser = np.maximum((df['High'] - df['Low']),
                             np.maximum(abs(df['High'] - df['Close'].shift()),
                                        abs(df['Low'] - df['Close'].shift()))
                             )

        return atr_ser.rolling(period).mean()

    @classmethod
    def std(cls, df, period) -> pd.Series:
        """
        Standard deviation of Close prices
        """
        return (df['Close']-df['Close'].shift()).fillna(0).rolling(window=period).std().bfill()

    @classmethod
    def donchian(cls, df, period_up, period_down):
        """
        see: https://raposa.trade/blog/use-python-to-trade-the-donchian-channel/
        """
        up = df['High'].rolling(period_up).max()
        down = df['Low'].rolling(period_down).min()
        return up, down

    @classmethod
    def sma(cls, df, period):
        return df['Close'].rolling(period).mean()   #.fillna(0)

    @classmethod
    def ema(cls, df, period):
        return df['Close'].ewm(span=period, adjust=False).mean()

    @classmethod
    def keltner(cls, df, sma_period, atr_period, nr_atrs):
        sma = cls.sma(df, sma_period)
        atr = cls.atr(df, atr_period)

        up = sma + atr * nr_atrs
        down = sma - atr * nr_atrs

        return up, down

    # indicators copied from: https://medium.com/@jsgastoniriartecabrera/advanced-algorithmic-trading-strategy-tester-a-comprehensive-python-implementation-8988f7c7137f
    # they are not tested!!!
    @classmethod
    def rsi(cls, data, rsi_period):
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @classmethod
    def b_bands(cls, data, bb_period, bb_std_mul):
        bb_middle = data['Close'].rolling(window=bb_period).mean()
        bb_std = data['Close'].rolling(window=bb_period).std()
        bb_upper = bb_middle + bb_std * bb_std_mul
        bb_lower = bb_middle - bb_std * bb_std_mul
        return bb_upper, bb_lower

    # the following functions need some refactoring (not to create columns in data)

    @classmethod
    def macd(cls, data, macd_fast, macd_slow, macd_signal):
        data['MACD'] = data['Close'].ewm(span=macd_fast).mean() - data['Close'].ewm(span=macd_slow).mean()
        data['MACD_Signal'] = data['MACD'].ewm(span=macd_signal).mean()
        data['MACD_Hist'] = data['MACD'] - data['MACD_Signal']

    @classmethod
    def stochastic_osc(cls, data, stoch_k, stoch_d):
        data['STOCH_K'] = ((data['Close'] - data['Low'].rolling(window=stoch_k).min()) /
                           (data['High'].rolling(window=stoch_k).max() - data['Low'].rolling(
                               window=stoch_k).min())) * 100
        data['STOCH_D'] = data['STOCH_K'].rolling(window=stoch_d).mean()

    @classmethod
    def cci(cls, data, cci_period):
        # Commodity Channel Index (CCI)
        tp = (data['High'] + data['Low'] + data['Close']) / 3
        data['CCI'] = (tp - tp.rolling(window=cci_period).mean()) / (0.015 * tp.rolling(window=cci_period).std())

    @classmethod
    def adx(cls, data, adx_period):
        # Average Directional Index (ADX)
        plus_dm = data['High'].diff()
        minus_dm = data['Low'].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0

        tr = pd.concat([data['High'] - data['Low'],
                        abs(data['High'] - data['Close'].shift()),
                        abs(data['Low'] - data['Close'].shift())], axis=1).max(axis=1)
        atr = tr.rolling(window=adx_period).mean()

        plus_di = 100 * (plus_dm.rolling(window=adx_period).sum() / atr)
        minus_di = 100 * (minus_dm.rolling(window=adx_period).sum() / atr)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        data['ADX'] = dx.rolling(window=adx_period).mean()

    @classmethod
    def super_trend(cls, data, st_period, st_multiplier):
        hl2 = (data['High'] + data['Low']) / 2
        data['ST_ATR'] = data['TR'].rolling(window=st_period).mean()
        data['ST_Upper'] = hl2 + (st_multiplier * data['ST_ATR'])
        data['ST_Lower'] = hl2 - (st_multiplier * data['ST_ATR'])
        data['SuperTrend'] = np.nan
        for i in range(st_period, len(data)):
            if data['Close'].iloc[i] > data['ST_Upper'].iloc[i - 1]:
                data['SuperTrend'].iloc[i] = data['ST_Lower'].iloc[i]
            elif data['Close'].iloc[i] < data['ST_Lower'].iloc[i - 1]:
                data['SuperTrend'].iloc[i] = data['ST_Upper'].iloc[i]
            else:
                data['SuperTrend'].iloc[i] = data['SuperTrend'].iloc[i - 1]

    @classmethod
    def ichimoku_cloud(cls, data, tenkan_period, kijun_period, senkou_b_period):
        data['Tenkan_Sen'] = (data['High'].rolling(window=tenkan_period).max() + data['Low'].rolling(
            window=tenkan_period).min()) / 2
        data['Kijun_Sen'] = (data['High'].rolling(window=kijun_period).max() + data['Low'].rolling(
            window=kijun_period).min()) / 2
        data['Senkou_Span_A'] = ((data['Tenkan_Sen'] + data['Kijun_Sen']) / 2).shift(kijun_period)
        data['Senkou_Span_B'] = ((data['High'].rolling(window=senkou_b_period).max() + data['Low'].rolling(
            window=senkou_b_period).min()) / 2).shift(kijun_period)
        data['Chikou_Span'] = data['Close'].shift(-kijun_period)
