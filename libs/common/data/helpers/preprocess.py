'''
Author: J , jwsun1987@gmail.com
Date: 2023-11-18 19:45:26
LastEditors: J , jwsun1987@gmail.com
Description: 
Copyright: Copyright (c) 2024 by jwsun1987@gmail.com. All Rights Reserved.
'''


import pandas as pd
import numpy as np
from typing import Union
from common.dates import *


def to_datetime_index(data: Union[pd.DataFrame, pd.Series], freq: str = None):
    if not freq:
        freq = get_date_frequency(data.index)
    data.index = pd.to_datetime(data.index)
    data = data.asfreq(freq)
    return data


def drop_columns(data, columns):
    return data.drop(columns=columns, inplace=True, errors="ignore")


def standardize(data, ax: int = 0):
    return (data - data.mean(axis=ax)) / data.std(axis=ax)


def normalize(data, ax: int = 0):
    return (data - data.min(axis=ax)) / (data.max(axis=ax) - data.min(axis=ax))

def resample_data(df, freq:str='M'):
    """
    Resample time series data to a different frequency.
    'M' for monthly, 'W' for weekly, 'D' for daily, etc.
    """
    return df.resample(freq).mean()  # Using mean() as the aggregation function, can be changed as needed.

#%% # TODO 
## 1. handle missing data
## 2. winsorize
## 

# %%
#def winsorize(
#    x: pd.Series, limit: float = 2.5, w: Union[Window, int, str] = Window(None, 0)
#) -> pd.Series:
    """
    Limit extreme values in series

    :param x: time series of prices
    :param limit: max z-score of values
    :param w: Window or int: size of window and ramp up to use. e.g. Window(22, 10) where 22 is the window size
              and 10 the ramp up value.  If w is a string, it should be a relative date like '1m', '1d', etc.
              Window size defaults to length of series.
    :return: timeseries of winsorized values

    **Usage**

    Cap and floor values in the series which have a z-score greater or less than provided value. This function will
    restrict the distribution of values. Calculates the sample standard deviation and adjusts values which
    fall outside the specified range to be equal to the upper or lower limits

    Lower and upper limits are defined as:

    :math:`upper = \\mu + \\sigma \\times limit`

    :math:`lower = \\mu - \\sigma \\times limit`

    Where :math:`\\mu` and :math:`\\sigma` are sample mean and standard deviation. The series is restricted by:

    :math:`R_t = max( min( X_t, upper), lower )`

    See `winsorising <https://en.wikipedia.org/wiki/Winsorizing>`_ for additional information

    **Examples**

    Generate price series and winsorize z-score of returns over :math:`22` observations

    >>> prices = generate_series(100)
    >>> winsorize(zscore(returns(prices), 22))

    **See also**

    :func:`zscore` :func:`mean` :func:`std`

    """
    """
    w = normalize_window(x, w)

    if x.size < 1:
        return x

    assert w.w, "window is not 0"

    mu = x.mean()
    sigma = x.std()

    high = mu + sigma * limit
    low = mu - sigma * limit

    ret = ceil(x, high)
    ret = floor(ret, low)

    return apply_ramp(ret, w)
    """
"""
def handle_missing_data(self, method="ffill"):
    if method == "ffill":
        self.data.fillna(method="ffill", inplace=True)
    elif method == "drop":
        self.data.dropna(inplace=True)
    elif method == "interpolate":
        raise NotImplementedError("not implemented yet...")
    else:
        raise ValueError(f"Unknown method: {method}")


def add_moving_average(self, column_name, window, ma_type="simple"):
    if ma_type == "simple":
        self.data[f"{column_name}_ma{window}"] = (
            self.data[column_name].rolling(window=window).mean()
        )
    elif ma_type == "exponential":
        self.data[f"{column_name}_ema{window}"] = (
            self.data[column_name].ewm(span=window).mean()
        )
    else:
        raise ValueError(f"Unknown moving average type: {ma_type}")


def preprocess(data):
    data = to_datetime_index(data)
    # Any other preprocessing steps can be added here or called individually


def in_percentile(data: pd.Series, val):
    data_list = sorted(data.values.tolist())
    l = len(data_list)
    for p in np.linspace(1, 0, 101):
        if p < 1:
            in_p = val > data_list[int(l * p)]
            if in_p:
                return int(round(p * 100))
"""
