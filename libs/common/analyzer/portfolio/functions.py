'''
Author: J , jwsun1987@gmail.com
Date: 2023-11-21 20:07:34
LastEditors: J , jwsun1987@gmail.com
Description: 
Copyright: Copyright (c) 2024 by jwsun1987@gmail.com. All Rights Reserved.
'''

import pandas as pd
import numpy as np
from typing import Dict, OrderedDict, Union

from common import dates as dates_helper
from common import data as data_helper
from common.enum import *
from common.analyzer import returns as return_analyzer


def clip_returns_to_benchmark(
    returns: Union[pd.DataFrame, pd.Series, np.ndarray],
    benchmark_returns: Union[pd.DataFrame, pd.Series, np.ndarray],
):
    # if (returns.index[0] < benchmark_returns.index[0]) or (returns.index[-1] > benchmark_returns.index[-1]):
    returns = data_helper.to_df(returns)
    benchmark_returns = data_helper.to_df(benchmark_returns)
    return (
        returns.loc[returns.index.intersection(benchmark_returns.index)],
        benchmark_returns.loc[benchmark_returns.index.intersection(returns.index)],
    )


PERCENTAGE = [
    "Annualized Return",
    "Total Returns",
    "Annualized Volatility",
    "Max DD",
    "Historical ES",
]


def performance_stats(
    returns: Union[pd.Series, pd.DataFrame],
    freq: str = "D",
    customized_annual_factor: Union[float, int] = None,
    benchmark_returns: Union[pd.Series, pd.DataFrame] = None,
) -> pd.DataFrame:
    """get the performance summary for a portfolio

    Args:
        returns (Union[pd.Series, pd.DataFrame]):
            noncumulative returns of a portfolio
        freq (str, optional):
            frequency of the given portfolio returns.
            Defaults to "D". Can be "D", "W", "M", "Q", and "Y"
        customized_annual_factor (Union[float, int], optional):
            customized annualized factor to surpress default annual factor.
            Defaults to None.
        benchmark_returns (Union[pd.Series, pd.DataFrame], optional):
            noncumulative returns of a benchmark portfolio. Defaults to None.

    Returns:
        pd.DataFrame: performance summary
    """
    returns = data_helper.to_df(returns)
    ret_stats = return_analyzer.return_stats(
        returns, freq=freq, annualization=customized_annual_factor
    )
    if benchmark_returns is not None:
        if isinstance(benchmark_returns, pd.DataFrame):
            benchmark_returns = data_helper.to_series(benchmark_returns)

        ret_ir = return_analyzer.information_ratio(returns, benchmark_returns)
        ret_stats = pd.concat([ret_stats, ret_ir], axis=0)
        if benchmark_returns.name in ret_stats.columns:
            ret_stats = ret_stats.rename(
                columns={
                    benchmark_returns.name: benchmark_returns.name + " (Benchmark)"
                }
            )
    for row in ret_stats.index:
        if any(p in row for p in PERCENTAGE):
            ret_stats.loc[row] = ret_stats.loc[row].values * 100
    ret_stats.index = [
        row + " (%)" if any(p in row for p in PERCENTAGE) else row
        for row in ret_stats.index
    ]
    return ret_stats.round(4)


def return_distribution(returns: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
    """get return distrubution

    Args:
        returns (Union[pd.Series, pd.DataFrame]):
            noncumulative returns

    Returns:
        pd.DataFrame: return distribution
    """
    returns = data_helper.to_df(returns)

    dist = []
    for col in returns.columns:
        x = returns[col].dropna().values
        col_dist = pd.DataFrame(
            pd.Series(
                {
                    "mean": np.mean(x),
                    "median": np.median(x),
                    "std": np.std(x),
                    "5%": np.percentile(x, 5),
                    "25%": np.percentile(x, 25),
                    "75%": np.percentile(x, 75),
                    "95%": np.percentile(x, 95),
                    "IQR": np.subtract.reduce(np.percentile(x, [75, 25])),
                }
            ).rename(col)
        )
        dist.append(col_dist)
    return pd.concat(dist, axis=1)


def annualized_return(
    returns: Union[pd.Series, pd.DataFrame],
    freq: str = "D",
    customized_annual_factor: Union[int, float] = None,
) -> pd.DataFrame:
    """compute annualized return of a portfolio given its returns

    Args:
        returns (Union[pd.Series, pd.DataFrame]):
            noncumulative returns
        freq (str, optional):
            date frequency of returns.
            Defaults to "D". Can be "D", "W", "M", "Q", and "Y".
        customized_annual_factor (Union[int, float], optional):
            customized annualized factor to surpress default one.
            Defaults to None.

    Returns:
        pd.DataFrame: portfolio annualized return
    """
    return return_analyzer.annualized_return(returns, freq, customized_annual_factor)


def annualized_volatility(
    returns: Union[pd.Series, pd.DataFrame],
    freq: str = "D",
    customized_annual_factor: Union[float, int] = None,
    alpha: Union[float, int] = 2.0,
) -> pd.DataFrame:
    """compute annualized volatility of a portfolio given its returns

    Args:
        returns (Union[pd.Series, pd.DataFrame]):
            noncumulative returns
        freq (str, optional):
            date frequency of returns.
            Defaults to "D". Can be "D", "W", "M", "Q", and "Y".
        customized_annual_factor (Union[int, float], optional):
            customized annualized factor to surpress default one.
            Defaults to None.
        alpha (Union[float, int]):
            alpha to compute vol.
            Defaults to 2.0.

    Returns:
        pd.DataFrame: portfolio annualized volatility
    """
    return return_analyzer.annualized_volatility(
        returns, freq, customized_annual_factor, alpha
    )


def cumulative_returns(
    returns: Union[pd.Series, pd.DataFrame, np.ndarray],
    starting_value: float = 0,
):
    """compute cumulative returns given simple returns

    Args:
        returns (Union[pd.Series, pd.DataFrame, np.ndarray]):
            simple returns as a percentage, noncumulative
        starting_value (float, optional):
            the starting returns. Defaults to 0.

    Returns:
        cumulative returns:
            series of cumulative returns
    """
    return return_analyzer.cumulative_returns(returns, starting_value)


def sharpe_ratio(
    returns: Union[pd.Series, pd.DataFrame],
    risk_free: float = 0.0,
    freq: str = "D",
    customized_annual_factor: Union[float, int] = None,
) -> pd.DataFrame:
    """compute sharpe ratio of a portfolio given its returns
    (Portfolio returns - risk-free rate)/std of returns

    Args:
        returns (Union[pd.Series, pd.DataFrame]):
            noncumulative returns
        risk_free (float):
            risk free return.
            Defaults to 0.0.
        freq (str, optional):
            date frequency of returns.
            Defaults to "D". Can be "D", "W", "M", "Q", and "Y".
        customized_annual_factor (Union[int, float], optional):
            customized annualized factor to surpress default one.
            Defaults to None.

    Returns:
        pd.DataFrame: portfolio sharpe ratio
    """
    return return_analyzer.sharpe_ratio(
        returns, risk_free, freq, customized_annual_factor
    )


def calmar_ratio(
    returns: Union[pd.Series, pd.DataFrame],
    freq: str = "D",
    customized_annual_factor: Union[float, int] = None,
) -> pd.DataFrame:
    """compute calmar ratio of a portfolio given its returns
    Annual Return / Max DD

    Args:
        returns (Union[pd.Series, pd.DataFrame]):
            noncumulative returns
        freq (str, optional):
            date frequency of returns.
            Defaults to "D". Can be "D", "W", "M", "Q", and "Y".
        customized_annual_factor (Union[int, float], optional):
            customized annualized factor to surpress default one.
            Defaults to None.

    Returns:
        pd.DataFrame: portfolio sharpe ratio
    """
    return return_analyzer.calmar_ratio(returns, freq, customized_annual_factor)


def information_ratio(
    returns: Union[pd.Series, pd.DataFrame],
    benchmark_returns: Union[pd.Series, pd.DataFrame],
) -> pd.DataFrame:
    """compute sharpe ratio of a portfolio given its returns
    (Portfolio returns - Benchmark returns)/std of (Portfolio returns - Benchmark returns)

    Args:
        returns (Union[pd.Series, pd.DataFrame]):
            noncumulative returns of portfolio
        benchmark_returns (Union[pd.Series, pd.DataFrame]):
            noncumulative returns of benchmark

    Returns:
        pd.DataFrame: portfolio information ratio
    """
    return return_analyzer.information_ratio(returns, benchmark_returns)


def tail_ratio(returns: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
    """compute tail ratio of a portfolio given its returns
    Returns at 95% quantile/Returns at 5% quantile

    Args:
        returns (Union[pd.Series, pd.DataFrame]):
            noncumulative returns of portfolio

    Returns:
        pd.DataFrame: portfolio tail ratio
    """
    return return_analyzer.tail_ratio(returns)


def sortino_ratio(
    returns: Union[pd.Series, pd.DataFrame],
    freq: str = "D",
    required_return: float = 0.0,
) -> pd.DataFrame:
    """compute annualized volatility of a portfolio given its returns
    (Portfolio returns - risk-free rate)/std of Downside

    Args:
        returns (Union[pd.Series, pd.DataFrame]):
            noncumulative returns
        freq (str, optional):
            date frequency of returns.
            Defaults to "D". Can be "D", "W", "M", "Q", and "Y".
        required_return (float):
            required return.
            Defaults to 0.0.

    Returns:
        pd.DataFrame: portfolio sortino ratio
    """
    return return_analyzer.sortino_ratio(returns, required_return, freq)


def downside_risk(
    returns: Union[pd.DataFrame, pd.Series, np.ndarray],
    required_return: float = 0,
    freq: str = "D",
    customized_annual_factor: Union[float, int] = None,
) -> pd.DataFrame:
    """compute downside risk of given returns
    Prob(returns > threshold)/Prob(returns < threshold)

    Args:
        returns (Union[pd.DataFrame, pd.Series, np.ndarray]):
            noncumulative returns
        required_return (float, optional):
            required return.
            Defaults to 0.
        freq (str, optional):
            date frequency of returns.
            Defaults to "D". Can be "D", "W", "M", "Q", and "Y".
        customized_annual_factor (Union[float, int], optional):
            customized annualized factor to surpress default one.
            Defaults to None.

    Returns:
        pd.DataFrame: downside risk
    """
    return return_analyzer.downside_risk(
        returns, required_return, freq, customized_annual_factor
    )


def omega_ratio(
    returns: Union[pd.Series, pd.DataFrame],
    freq: str = "D",
    risk_free: float = 0,
    required_return: float = 0,
    annualization_factor: int = 252,
) -> pd.DataFrame:
    """compute omega ratio of a portfolio given its returns

    Args:
        returns (Union[pd.Series, pd.DataFrame]):
            noncumulative returns
        freq (str, optional):
            date frequency of returns.
            Defaults to "D". Can be "D", "W", "M", "Q", and "Y".
        risk_free (float):
            risk free return.
            Defaults to 0.0.
        required_return (float):
            required return.
            Defaults to 0.0.
        annualized_factor (Union[int, float], optional):
            customized annualized factor to surpress default one.
            Defaults to None.

    Returns:
        pd.DataFrame: portfolio omega ratio
    """
    return return_analyzer.omega_ratio(
        returns, risk_free, freq, required_return, annualization_factor
    )


def periodic_returns(
    returns: Union[pd.Series, pd.DataFrame], freq: str = "M"
) -> pd.DataFrame:
    """get periodic returns of given returns

    Args:
        returns (Union[pd.Series, pd.DataFrame]):
            noncumulative returns
        freq (str, optional):
            grouped frequency.
            Defaults to "M". Can be "W", "M", "Q", and "Y"

    Returns:
        pd.DataFrame: periodic returns
    """
    return return_analyzer.aggregate_returns(returns, freq)


def underwater(returns: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
    """get underwater of given returns

    Args:
        returns (Union[pd.Series, pd.DataFrame]):
            noncumulative returns

    Returns:
        pd.DataFrame: underwater of given returns
    """
    returns = data_helper.to_df(returns)
    cum_rets = return_analyzer.cumulative_returns(returns, starting_value=1.0)
    running_max = cum_rets.cummax(axis=0)
    dd = (cum_rets.sub(running_max)).div(running_max) * 100
    return pd.DataFrame(dd)


def max_drawdown_info(dd: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
    """get information of the max dd for given underwater

    Args:
        dd (Union[pd.Series, pd.DataFrame]):
            underwater

    Returns:
        pd.DataFrame: start, trough, and end dates of max dd
    """
    dd = data_helper.to_df(dd)
    dd_info = {}
    for col in dd.columns:
        dd_col = dd.loc[:, col]
        trough = dd_col.index[np.argmin(dd_col)]  # end of the period
        try:
            dd_start = dd_col.loc[:trough][dd_col.loc[:trough] == 0].dropna(
                axis=0, how="any"
            )
            dd_start = dd_start.index[-1] if dd_start is not None else np.nan
        except IndexError:
            dd_start = np.nan

        try:
            dd_end = dd_col.loc[trough:][dd_col.loc[trough:] == 0].dropna(
                axis=0, how="any"
            )
            dd_end = dd_end.index[0] if dd_end is not None else np.nan
        except IndexError:
            dd_end = np.nan
        dd_info.update({col: [dd_start, trough, dd_end]})
    return pd.DataFrame.from_dict(dd_info, orient="index").rename(
        columns={0: "Start", 1: "Trough", 2: "End"}
    )


def top_drawdowns(
    returns: Union[pd.Series, pd.DataFrame], top: int = 5
) -> pd.DataFrame:
    """get the top n dd information of given returns

    Args:
        returns (Union[pd.Series, pd.DataFrame]):
            non cumulative returns
        top (int, optional):
            the first N largest dd. Defaults to 5.

    Returns:
        pd.DataFrame: information of top N dd
    """
    returns = data_helper.to_df(returns)
    dd = underwater(returns)
    dd_summary = []
    for col in dd.columns:
        top_dds = []
        col_dd = dd.loc[:, col]
        for t in range(top):
            col_dd_info = max_drawdown_info(col_dd)
            start = col_dd_info["Start"].values[0]
            trough = col_dd_info["Trough"].values[0]
            end = col_dd_info["End"].values[0]
            max_dd = col_dd.loc[trough]
            # Slice out draw-down period
            if not pd.isnull(end):
                col_dd.drop(
                    col_dd[start:end].index[1:-1],
                    inplace=True,
                )
                dd_duration = len(pd.date_range(start, end, freq="M"))
            else:
                # drawdown has not ended yet
                col_dd = col_dd.loc[:start]
                dd_duration = np.nan

            top_dds.append(
                (
                    start,
                    trough,
                    end,
                    max_dd,
                    dd_duration,
                )
            )
            if (len(returns) == 0) or (len(dd) == 0):
                break

        col_top_dd = pd.DataFrame(
            index=list(range(top + 1))[1:],
            columns=[
                "Peak date",
                "Valley date",
                "Recovery date",
                "Net drawdown in %",
                "Duration (Month)",
            ],
            data=top_dds,
        )
        dd_summary.append(pd.concat([col_top_dd], keys=[col], axis=0))
    return pd.concat(dd_summary, axis=0)


def get_market_correction_info(
    returns: Union[pd.Series, pd.DataFrame], threshold: float = -20
) -> pd.DataFrame:
    """get info of bear markets

    Args:
        returns (Union[pd.Series, pd.DataFrame]):
            noncumulative returns
        threshold (float, optional):
            dd threshold for a bearmarket in percentage. Defaults to -20.

    Returns:
        pd.DataFrame: info of bear markets for given returns
    """
    returns = data_helper.to_df(returns)
    dd = underwater(returns)
    curr_max_dd = data_helper.to_df(dd.min(axis=0)).T

    dd_summary = []
    for col in dd.columns:
        top_dds = []
        col_dd = dd.loc[:, col]
        col_curr_max_dd = curr_max_dd.loc[:, col].values[0]
        while col_curr_max_dd <= threshold:
            col_dd_info = max_drawdown_info(col_dd)
            start = col_dd_info["Start"].values[0]
            trough = col_dd_info["Trough"].values[0]
            end = col_dd_info["End"].values[0]
            temp_max_dd = col_dd.loc[trough]
            trough_duration = len(pd.date_range(start, trough, freq="M"))
            # Slice out draw-down period
            if not pd.isnull(end):
                col_dd.drop(
                    col_dd[start:end].index[1:-1],
                    inplace=True,
                )
                recovery_duration = len(pd.date_range(trough, end, freq="M"))
                dd_duration = len(pd.date_range(start, end, freq="M"))
            else:
                # drawdown has not ended yet
                recovery_duration = len(
                    pd.date_range(trough, col_dd.index.max(), freq="M")
                )
                dd_duration = len(pd.date_range(start, col_dd.index.max(), freq="M"))
                col_dd = col_dd.loc[:start]

            top_dds.append(
                (
                    start,
                    trough,
                    end,
                    temp_max_dd.round(2),
                    int(trough_duration),
                    int(recovery_duration),
                    int(dd_duration),
                )
            )
            if (len(returns) == 0) or (len(dd) == 0):
                break
            else:
                col_curr_max_dd = temp_max_dd

        col_top_dd = pd.DataFrame(
            index=list(range(len(top_dds) + 1))[1:],
            columns=[
                "Peak date",
                "Valley date",
                "Recovery date",
                "Max drawdown (%)",
                "Trough (M)",
                "Recovery (M)",
                "Underwater (M)",
            ],
            data=top_dds,
        )
        dd_summary.append(pd.concat([col_top_dd], keys=[col], axis=0))
    return pd.concat(dd_summary, axis=0)


def rolling_annual_returns(
    returns: Union[pd.Series, pd.DataFrame],
    window: Union[float, int] = 252,
    freq: str = "D",
    customized_annual_factor: Union[float, int] = None,
) -> pd.DataFrame:
    """compute annualized return on a rolling basis

    Args:
        returns (Union[pd.Series, pd.DataFrame]):
            noncumulative returns
        window (Union[float, int], optional):
            size of the rolling window.
            Defaults to 252.
        freq (str, optional):
            date frequency of returns.
            Defaults to "D". Can be "D", "W", "M", "Q", and "Y"
        customized_annual_factor (Union[float, int], optional):
            customized annualized factor to surpress default one.
            Defaults to None.

    Returns:
        pd.DataFrame: rolling annualized return
    """
    # ann_fator = return_analyzer.annualization_factor(period=period, annualization=annualization)
    returns = data_helper.to_df(returns)
    return returns.rolling(window).apply(
        lambda col: return_analyzer.annualized_return(
            col, freq, customized_annual_factor
        ).values[0]
    )


def rolling_annual_volatility(
    returns: Union[pd.DataFrame, pd.Series],
    window: Union[int, float] = 252,
    freq: str = "D",
    customized_annual_factor: Union[int, float] = None,
) -> pd.DataFrame:
    """compute rolling annualized volatility

    Args:
        returns (Union[pd.DataFrame, pd.Series]):
            noncumulative returns
        window (Union[int, float], optional):
            size of the rolling window.
            Defaults to 252.
        freq (str, optional):
            date frequency of returns.
            Defaults to "D". Can be "D", "W", "M", "Q", and "Y".
        customized_annual_factor (Union[int, float], optional):
            customized annualized factor to surpress default one.
            Defaults to None.

    Returns:
        pd.DataFrame: rolling annualized volatility
    """
    ann_factor = return_analyzer.annualization_factor(
        freq=freq, customized=customized_annual_factor
    )
    rolling_vol = returns.rolling(window).std() * np.sqrt(ann_factor)
    return pd.DataFrame(rolling_vol)


def rolling_sharpe_ratio(
    returns: Union[pd.DataFrame, pd.Series],
    window: Union[int, float] = 252,
    freq: str = "D",
    customized_annual_factor: Union[int, float] = None,
) -> pd.DataFrame:
    """compute rolling sharpe ratio with a given time window

    Args:
        returns (Union[pd.DataFrame, pd.Series]):
            noncumulative returns
        window (Union[int, float], optional):
            size of the rolling window in days.
            Defaults to 252.
        freq (str, optional):
            date frequency of returns.
            Defaults to "D". Can be "D", "W", "M", "Q", and "Y".
        customized_annual_factor (Union[int, float], optional):
            customized annualized factor to surpress default one.
            Defaults to None.

    Returns:
        pd.DataFrame: rolling sharpe ratio
    """
    ann_factor = return_analyzer.annualization_factor(
        freq=freq, customized=customized_annual_factor
    )
    rolling_sr = (
        returns.rolling(window).mean()
        / returns.rolling(window).std()
        * np.sqrt(ann_factor)
    )
    return pd.DataFrame(rolling_sr)


def get_period_data(
    returns: Union[pd.Series, pd.DataFrame], periods: Dict = PERIODS
) -> OrderedDict:
    """get the data within given data ranges specified in periods

    Args:
        returns (Union[pd.Series, pd.DataFrame]):
            noncumulative returns
        periods (Dict, optional):
            dates of interesting periods.
            Defaults to PERIODS.

    Returns:
        OrderedDict: data clips for specified periods
    """
    periods_returns = returns.copy()
    periods_returns.index = periods_returns.index.map(pd.Timestamp)
    ranges = OrderedDict()
    for name, (start, end) in periods.items():
        try:
            tmp_returns = periods_returns.loc[
                (periods_returns.index >= start) & (periods_returns.index <= end)
            ]
            if len(tmp_returns) == 0:
                continue
            ranges[name] = tmp_returns
        except BaseException:
            continue
    return ranges


def interesting_period_performances(
    returns: Union[pd.Series, pd.DataFrame],
    freq: str = "D",
    customized_annual_factor: Union[int, float] = None,
    interesting_periods: Dict = PERIODS,
) -> pd.DataFrame:
    """get performance summary for periods of interest

    Args:
        returns (Union[pd.Series, pd.DataFrame]):
            noncumulative returns
        freq (str, optional):
            frequency of noncumulative returns.
            Defaults to "D". Can be "D", "W", "M", "Q", and "Y"
        customized_ann_factor (Union[int, float], optional):
            customized annualized factor to surpress default annualized factor.
            Defaults to None.
        interesting_periods (Dict, optional):
            periods of interest to investigate.
            Defaults to PERIODS.

    Returns:
        pd.DataFrame: performacne summary for periods of interest
    """
    returns = data_helper.to_df(returns)

    perf_summary = []
    for col in returns.columns:
        periods_returns = get_period_data(
            returns=returns[col], periods=interesting_periods
        )
        period_perf_summary = []
        for p_name, p_returns in periods_returns.items():
            period_perf_summary.append(
                performance_stats(
                    p_returns,
                    freq=freq,
                    customized_annual_factor=customized_annual_factor,
                ).rename(columns={p_returns.name: p_name})
            )
        perf_summary.append(
            pd.concat([pd.concat(period_perf_summary, axis=1).T], keys=[col], axis=1)
        )
    return pd.concat(perf_summary, axis=1)


def performance_summary(
    returns: Union[pd.Series, pd.DataFrame],
    freq: str = "D",
    customized_annual_factor: Union[float, int] = None,
    window: Union[float, int] = 135,
    interesting_periods: Dict = PERIODS,
    benchmark_returns: Union[pd.Series, pd.DataFrame] = None,
) -> Dict:
    """performance summary for returns of a portfolio

    Args:
        returns (Union[pd.Series, pd.DataFrame]):
            noncumulative returns
        freq (str, optional):
            frequency of returns.
            Defaults to "D". Can be "D", "W", "M", "Q", and "Y"
        customized_annual_factor (Union[float, int], optional):
            customized annualized factor to surpress the default one.
            Defaults to None.
        window (Union[float, int], optional):
            window size for rolling computation.
            Defaults to 135.
        interesting_periods (Dict, optional):
            perdiods of interest for stats.
            Defaults to PERIODS.
        benchmark_returns (Union[pd.Series, pd.DataFrame]):
            noncumulative returns of the benchmark.
            Defaults to None.

    Returns:
        Dict: perforamcne summary of the portfolio
    """
    return {
        "summary": performance_stats(
            returns,
            freq=freq,
            customized_annual_factor=customized_annual_factor,
            benchmark_returns=benchmark_returns,
        ),
        "cumulative returns": cumulative_returns(returns),
        "monthly returns": periodic_returns(returns, freq="M"),
        "yearly returns": periodic_returns(returns, freq="Y"),
        "underwater": underwater(returns),
        "top drawdown": top_drawdowns(returns),
        "rolling vol": rolling_annual_volatility(
            returns,
            window=window,
            freq=freq,
            customized_annual_factor=customized_annual_factor,
        ),
        "rolling sharpe ratio": rolling_sharpe_ratio(
            returns,
            window=window,
            freq=freq,
            customized_annual_factor=customized_annual_factor,
        ),
        "interesting period performance": interesting_period_performances(
            returns,
            freq=freq,
            customized_annual_factor=customized_annual_factor,
            interesting_periods=interesting_periods,
        ),
    }


# %%
def map_transaction(txn):
    return


def make_transaction_frame(txn):
    return


def get_txn_vol(txn):
    return


def adjust_to_slippage(returns, positions, txn):
    return


def get_turnonver(positions, txn, denominator="AGB"):
    return


def simulate_paths(returns, num_days):
    return


def summarize_paths(returns, cone_std=(1, 1.5, 2), staring_value=1):
    return
