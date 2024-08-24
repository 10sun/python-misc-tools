'''
Author: J , jwsun1987@gmail.com
Date: 2023-11-24 00:32:12
LastEditors: J , jwsun1987@gmail.com
Description: 
Copyright: Copyright (c) 2024 by jwsun1987@gmail.com. All Rights Reserved.
'''

# from .forward_returns import *

from .functions import *
from .enum import *

SIMPLE_STAT_FUNC = [
    annualized_return,
    total_returns,
    annualized_volatility,
    sharpe_ratio,
    max_drawdown,
    historical_es,
    skew,
    kurtosis,
    tail_ratio,
]


def return_stats(
    returns: pd.Series,
    benchmark_returns: pd.Series = None,
    freq: str = "D",
    annualization=None,
):
    stats = []
    if not freq:
        freq = dates.get_date_frequency(returns.index)

    for stat_func in SIMPLE_STAT_FUNC:
        #print(stat_func.__name__)
        if stat_func.__name__ not in [
            "total_returns",
            "max_drawdown",
            "historical_es",
            "tail_ratio",
            "capture_ratio",
            "skew",
            "kurtosis",
        ]:
            stats.append(
                stat_func(returns, freq=freq, customized_annual_factor=annualization)
            )
        else:
            stats.append(stat_func(returns))

    if benchmark_returns is not None:
        stats.append(
            information_ratio(
                returns,
                benchmark_returns,
                freq=freq,
                customized_annual_factor=annualization,
            )
        )

    return pd.concat(stats, axis=0)
