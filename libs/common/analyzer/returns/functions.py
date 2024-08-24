'''
Author: J , jwsun1987@gmail.com
Date: 2023-11-21 20:07:36
LastEditors: J , jwsun1987@gmail.com
Description: 
Copyright: Copyright (c) 2024 by jwsun1987@gmail.com. All Rights Reserved.
'''


import math
import pandas as pd
import numpy as np

from scipy import stats

from typing import Union, Dict, List

from common import dates
from common import data

from common.analyzer.helpers import *


def _results_to_df(
    results: Union[np.array, float],
    inputs: Union[pd.DataFrame, pd.Series, np.ndarray],
    metric_name: str = None,
):
    """convert results from an array or a scalar to a dataframe

    Args:
        results (Union[np.array, float]):
            results to convert
            e.g. [0.01, 0.02], or 0.01
        inputs (Union[pd.DataFrame, pd.Series, np.ndarray]):
            the inputs on which the results are obtained with. Normally it is noncumulative returns
        metric_name (str, optional):
            the name of the results.
            Defaults to None.

    Returns:
        results: results in a dataframe format
            e.g.
                                SPX      World
            __________________________________
            Annualized Returns  0.01     0.02
    """
    if isinstance(inputs, pd.DataFrame):
        return pd.DataFrame(results, index=inputs.columns, columns=[metric_name]).T
    elif isinstance(inputs, pd.Series):
        return pd.DataFrame([results], index=[metric_name]).rename(
            columns={0: inputs.name}
        )
    else:
        return pd.DataFrame([results], index=[metric_name])


def annualization_factor(freq, customized: Union[float, int] = None):
    """get annualization factor for given frequency, or the value specified
    if a 'customized' value is passed

    Args:
        freq (str, optional):
            Defines the periodicity of the 'returns' data for purposes of analyzing.
            Value ignored if 'customized' parameter is specified.
            Default values: 'D': 252, 'M': 12, 'Q':4

        customized (int. float, optional): customized . Defaults to None.
            Used to surpress default values available in 'freq' to convert returns
            into annual returns. Value should
            be the annual frequency of 'returns'.
    Raises:
        ValueError: _description_

    Returns:
        annualization_factor (float, int)
    """
    if customized is None:
        try:
            factor = dates.ANNUALIZATION_FACTOR[freq]
        except KeyError:
            raise ValueError(
                "Period cannot be '{}'. It has to be in '{}'.".format(
                    freq, "', '".join(dates.ANNUALIZATION_FACTOR.keys())
                )
            )
    else:
        factor = customized
    return factor


def _adjust_returns(
    returns: Union[pd.DataFrame, pd.Series, np.ndarray],
    adjustment_factor: Union[pd.DataFrame, pd.Series, float],
):
    """adjust returns by adjustment_factor
    when adjustment_factor is 0, return 'returns' itself (not copy)

    Args:
        returns (pd.Series, pd.DataFrame, np.ndarray): returns to adjust
        adjustment_factor (float, int): adjustment on returns

    Returns:
        adjusted_returns: array_like
    """
    if isinstance(adjustment_factor, (float, int)):
        return returns - adjustment_factor

    if isinstance(adjustment_factor, pd.Series):
        return pd.DataFrame(returns).apply(lambda col: col - adjustment_factor, axis=0)
    elif isinstance(adjustment_factor, pd.DataFrame):
        adjustment_factor_series = adjustment_factor.squeeze(axis=1)
        return pd.DataFrame(returns).apply(
            lambda col: col - adjustment_factor_series, axis=0
        )


def simple_returns(prices: Union[pd.DataFrame, pd.Series, np.ndarray]):
    """compute simple returns from prices

    Args:
        prices (Union[pd.DataFrame, pd.Series, np.ndarray]):
            prices of assets with assets as columns, indexed by datetimes

    Returns:
        simple returns: returns of assets with assets as columns, indexed by datetimes
    """
    if isinstance(prices, (pd.DataFrame, pd.Series)):
        return prices.pct_change(1).iloc[1:]
    else:
        # for np.ndarray type data
        diff = np.diff(prices, axis=0)
        return np.divide(diff, prices[:-1], out=diff)


def log_returns(prices: Union[pd.DataFrame, pd.Series, np.ndarray]):
    """compute log returns for prices

    Args:
        prices (Union[pd.DataFrame, pd.Series, np.ndarray]):
            prices of assets with assets as columns, indexed by datetimes

    Returns:
        log returns: returns of assets with assets as columns, indexed by datetimes
    """
    if isinstance(prices, (pd.DataFrame, pd.Series)):
        return np.log1p(prices.pct_change(1))
    else:
        # TODO: add the computation for log returns
        return NotImplementedError("not implemented yet for np.array")


def cumulative_returns(
    returns: Union[pd.Series, pd.DataFrame, np.ndarray],
    starting_value: float = 0,
    out=None,
):
    """compute cumulative returns given simple returns

    Args:
        returns (Union[pd.Series, pd.DataFrame, np.ndarray]):
            simple returns as a percentage, noncumulative
        starting_value (float, optional):
            the starting returns. Defaults to 0.
        out (_type_, optional):
            array to use as output buffer.
            If not passed, a new array will be created.
            Defaults to None.

    Returns:
        cumulative returns:
            series of cumulative returns
    """
    if len(returns) < 1:
        return returns.copy()

    ## TODO series by series so to avoid different calendar?
    ## TODO nan value during the weekend?
    nan_mask = np.isnan(returns)
    if np.any(nan_mask):
        returns = returns.copy()
        returns[nan_mask] = 0

    allocated_output = out is None
    if allocated_output:
        out = np.empty_like(returns)

    np.add(returns, 1, out=out)
    out.cumprod(axis=0, out=out)

    if starting_value == 0:
        np.subtract(out, 1, out=out)
    else:
        np.multiply(out, starting_value, out=out)

    if allocated_output:
        if returns.ndim == 1 and isinstance(returns, pd.Series):
            out = pd.Series(out, index=returns.index)
        elif isinstance(returns, pd.DataFrame):
            out = pd.DataFrame(
                out,
                index=returns.index,
                columns=returns.columns,
            )
    if isinstance(returns, pd.DataFrame):
        return out
    elif isinstance(returns, pd.Series):
        return pd.DataFrame(out.rename(returns.name))
    else:
        ## TODO check if it works for np.ndarray
        return pd.DataFrame(out)


def total_returns(
    returns: Union[pd.Series, pd.DataFrame, np.ndarray], starting_value: float = 0.0
):
    """compute total returns given simple returns

    Args:
        returns (Union[pd.Series, pd.DataFrame, np.ndarray]):
            Noncumulative simple returns
        starting_value (float, optional):
            The starting value of rebased returns. Defaults to 0.0.

    Returns:
        total returns:
            a dataframe with assets as columns
    """
    if len(returns) == 0:
        return np.nan

    if isinstance(returns, (pd.DataFrame, pd.Series)):
        cum_rets = (returns + 1).prod()
    else:
        cum_rets = np.nanprod(returns + 1, axis=0)

    if starting_value == 0:
        cum_rets -= 1
    else:
        cum_rets *= starting_value

    return _results_to_df(cum_rets, returns, "Total Returns")


def aggregate_returns(returns: Union[pd.DataFrame, pd.Series], freq: str = "M"):
    """aggregates return by week, month, or year

    Args:
        returns (Union[pd.DataFrame, pd.Series]):
            daily returns of noncumulative returns
        period (str, optional):
            group level for aggregated returns.
            Defaults to "M". Can be 'W', 'M', 'Q', or 'Y'.
    """

    def cumulate_returns(x):
        return cumulative_returns(x).iloc[-1]

    if freq == "W":
        grouper = [lambda x: x.year, lambda x: x.isocalendar()[1]]
    elif freq == "M":
        grouper = [lambda x: x.year, lambda x: x.month]
    elif freq == "Q":
        grouper = [lambda x: x.year, lambda x: int(math.ceil(x.month / 3.0))]
    elif freq == "Y":
        grouper = [lambda x: x.year]
    else:
        raise ValueError("Period must be D, W, Q, Y")

    if not isinstance(returns, pd.DataFrame):
        returns = pd.DataFrame(returns)
    agg_returns = returns.groupby(grouper).apply(cumulate_returns)

    if freq == "W":
        agg_returns.index.names = ["Year", "Week"]
    elif freq == "M":
        agg_returns.index.names = ["Year", "Month"]
    elif freq == "Q":
        agg_returns.index.names = ["Year", "Quarter"]
    elif freq == "Y":
        agg_returns.index.names = ["Year"]
    else:
        raise ValueError("Period must be D, W, Q, Y")

    return agg_returns


def annualized_return(
    returns: Union[pd.Series, np.ndarray], freq="D", customized_annual_factor=None
):
    """compute the annualized return of given returns

    Args:
        returns (Union[pd.Series, np.ndarray]):
            noncumulative returns
        freq (str, optional):
            the date frequency of given noncumulative returns.
            Defaults to "D".
        customized_annual_factor (_type_, optional):
            customized annualization factor to surpress default values in enum.
            Defaults to None.

    Returns:
        annualized return: float
            annualized return as compounded annual growth rate
    """
    if len(returns) < 1:
        return np.nan

    ann_factor = annualization_factor(freq, customized_annual_factor)
    years = len(returns) / ann_factor
    ending_value = total_returns(returns, starting_value=1)
    return (
        (ending_value ** (1 / years) - 1)
        .T.rename(columns={"Total Returns": "Annualized Return"})
        .T
    )


def annualized_volatility(
    returns: Union[pd.DataFrame, pd.Series, np.ndarray],
    freq: str = "D",
    customized_annual_factor: float = None,
    alpha: float = 2.0,
):
    """compute the annualized volatility of given returns

    Args:
        returns (Union[pd.DataFrame, pd.Series, np.ndarray]):
            noncumulative returns
        freq (str, optional):
            the date frequency of given noncumulative returns.
            Defaults to "D".
        customized_annual_factor (float, optional):
            customized annualization factor to surpress default values in enum.
            Defaults to None.
        alpha (float, optional):
            scaling relation.
            Defaults to 2.0.

    Returns:
        _type_: _description_
    """
    out = np.empty(returns.shape[1:])

    returns_1d = returns.ndim == 1

    if len(returns) < 2:
        out[()] = np.nan
        if returns_1d:
            out = out.item()
        return out

    ann_factor = annualization_factor(freq, customized_annual_factor)
    np.nanstd(returns, ddof=1, axis=0, out=out)
    out = np.multiply(out, ann_factor ** (1.0 / alpha), out=out)
    if returns_1d:
        out = out.item()

    return _results_to_df(out, returns, "Annualized Volatility")


def max_drawdown(returns: Union[pd.DataFrame, pd.Series, np.ndarray]):
    """Computes the maximum drawdown of given returns

    Args:
        returns (Union[pd.DataFrame, pd.Series, np.ndarray]):
            noncumulative returns

    Returns:
        maximum drawdown:
            a dataframe of maximum drawdown
    """
    dd = np.empty(returns.shape[1:])

    returns_1d = returns.ndim == 1

    if len(returns) < 1:
        dd[()] = np.nan
        if returns_1d:
            dd = dd.item()
        return dd

    returns_array = np.asanyarray(returns)

    cumulative = np.empty(
        (returns.shape[0] + 1,) + returns.shape[1:],
        dtype="float64",
    )
    cumulative[0] = start = 100
    cumulative_returns(returns_array, starting_value=start, out=cumulative[1:])

    max_return = np.fmax.accumulate(cumulative, axis=0)

    np.nanmin((cumulative - max_return) / max_return, axis=0, out=dd)
    if returns_1d:
        dd = dd.item()
    # elif isinstance(returns, pd.DataFrame):
    #    dd = pd.Series(dd)

    return _results_to_df(dd, returns, "Max DD")


def sharpe_ratio(
    returns: Union[pd.DataFrame, pd.Series, np.ndarray],
    risk_free: float = 0.0,
    freq: str = "D",
    customized_annual_factor=None,
):
    """compute sharpe ratio of given returns
    (returns - risk-free rate)/std of (returns - risk-free rate)

    Args:
        returns (Union[pd.DataFrame, pd.Series, np.ndarray]): _description_
        risk_free (float, optional): _description_. Defaults to 0.0.
        freq (str, optional): _description_. Defaults to "D".
        customized_annual_factor (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    sr = np.empty(returns.shape[1:])

    return_1d = returns.ndim == 1

    if len(returns) < 2:
        sr[()] = np.nan
        if return_1d:
            sr = sr.item()
        return sr

    returns_risk_adj = np.asanyarray(_adjust_returns(returns, risk_free))
    ann_factor = annualization_factor(freq, customized_annual_factor)

    np.multiply(
        np.divide(
            np.nanmean(returns_risk_adj, axis=0),
            np.nanstd(returns_risk_adj, ddof=1, axis=0),
            out=sr,
        ),
        np.sqrt(ann_factor),
        out=sr,
    )
    if return_1d:
        sr = sr.item()

    return _results_to_df(sr, returns, "Sharpe Ratio")


def information_ratio(
    returns: Union[pd.DataFrame, pd.Series, np.ndarray],
    benchmark_returns: Union[pd.DataFrame, pd.Series, np.ndarray],
):  # , freq="D", customized_annual_factor=None
    """compute information ratio for returns given benchmark returns
    (returns - benchmark returns)/std of (returns - benchmark returns)

    Args:
        returns (Union[pd.DataFrame, pd.Series, np.ndarray]):
            noncumulative returns
        benchmark_returns (Union[pd.DataFrame, pd.Series, np.ndarray]):
            noncumulative returns of a benchmark

    Returns:
        information ratio:
            information ratio of given ratios
    """
    ir = np.empty(returns.shape[1:])

    returns_1d = returns.ndim == 1

    if len(returns) < 2:
        ir[()] = np.nan
        if returns_1d:
            ir = ir.item()
        return ir

    active_return = _adjust_returns(returns, benchmark_returns)
    tracking_error = np.nan_to_num(np.nanstd(active_return, ddof=1, axis=0))

    ir = np.divide(
        np.nanmean(active_return, axis=0),
        tracking_error,
    )

    if returns_1d:
        ir = ir.item()

    return _results_to_df(ir, returns, "Information Ratio")


def calmar_ratio(
    returns: Union[pd.DataFrame, pd.Series, np.ndarray],
    freq: str = "D",
    customized_annual_factor: float = None,
):
    """compute calmar ratio of given returns
    annualized return / max dd

    Args:
        returns (Union[pd.DataFrame, pd.Series, np.ndarray]):
            noncumulative returns
        freq (str, optional):
            frequency of noncumulative returns.
            Defaults to "D". Can be 'D', 'W', 'M', 'Q', 'Y'
        customized_annual_factor (float, optional):
            customized annual factor to surpress default frequency.
            Defaults to None.

    Returns:
        calmar ratio: calmar ratio of given returns. Nan if max dd is non-negative
    """
    max_dd = max_drawdown(returns=returns)
    temp = annualized_return(
        returns=returns,
        freq=freq,
        customized_annual_factor=customized_annual_factor,
    )

    max_dd_abs = abs(max_dd[max_dd < 0].fillna(0))
    cr = temp.div(max_dd_abs.values)
    cr.index = ["Calmar Ratio"]
    # max_dd = max_drawdown(returns=returns)
    # if max_dd < 0:
    #    temp = annual_return(
    #        returns=returns,
    #        period=period,
    #        annualization=annualization
    #    ) / abs(max_dd)
    # else:
    #    return np.nan

    # if np.isinf(temp):
    #    return np.nan

    return cr


def omega_ratio(
    returns: Union[pd.DataFrame, pd.Series, np.ndarray],
    risk_free: float = 0,
    freq: str = "D",
    required_return: float = 0.0,
    annualization: int = 252,
):
    """compute the risk-return performance measure of an investment asset
    Prob(returns > threshold)/Prob(returns < threshold)

    Args:
        returns (Union[pd.DataFrame, pd.Series, np.ndarray]): _description_
        risk_free (float, optional): _description_. Defaults to 0.
        freq (str, optional): _description_. Defaults to "D".
        required_return (float, optional): _description_. Defaults to 0.0.
        annualization (int, optional): _description_. Defaults to 252.

    Returns:
        _type_: _description_
    """
    if len(returns) < 2:
        return np.nan

    if annualization == 1:
        return_threshold = required_return
    elif required_return <= -1:
        return np.nan
    else:
        return_threshold = (1 + required_return) ** (1.0 / annualization) - 1

    returns_less_thresh = returns - risk_free - return_threshold

    numer = pd.DataFrame(returns_less_thresh[returns_less_thresh > 0.0]).sum(axis=0)
    denom = -1.0 * pd.DataFrame(returns_less_thresh[returns_less_thresh < 0.0]).sum(
        axis=0
    )
    omega_r = pd.DataFrame(numer.div(denom.values)).T
    omega_r.index = ["Omega Ratio"]
    # omega_r = omega_r.apply(lambda x: np.nan if x < 0 else x, axis=1)
    # if denom > 0.0:
    #    return numer / denom
    # else:
    #    return np.nan
    return omega_r


def sortino_ratio(
    returns: Union[pd.DataFrame, pd.Series, np.ndarray],
    required_return: float = 0,
    freq: str = "D",
    customized_annual_factor: float = None,
    _downside_risk=None,
) -> pd.DataFrame:
    """compute Sortino ratio of given returns and required return
    (returns - risk-free rate)/std of Downside

    Args:
        returns (Union[pd.DataFrame, pd.Series, np.ndarray]):
            noncumulative returns
        required_return (float, optional):
            required return, normally the risk-free return.
            Defaults to 0.
        freq (str, optional):
            frequency of returns.
            Defaults to "D". Can be 'D', 'W', 'M', 'Q', and 'Y'
        customized_annual_factor (float, optional):
            customized frequency factor for annualization, to surpress
            default annual factor.
            Defaults to None.
        _downside_risk (_type_, optional):
            standard deviation of the downside.
            Defaults to None.

    Returns:
        Sortino ratio: sortino ratio of given returns
    """
    sr = np.empty(returns.shape[1:])

    return_1d = returns.ndim == 1

    if len(returns) < 2:
        sr[()] = np.nan
        if return_1d:
            sr = sr.item()
        return _results_to_df(sr, returns, "Sortino Ratio")

    # adj_returns = np.asanyarray(_adjust_returns(returns, required_return))

    # ann_factor = annualization_factor(freq, customized_annual_factor)

    # average_annual_return = np.nanmean(adj_returns, axis=0) * ann_factor

    average_annual_return = annualized_return(
        _adjust_returns(returns, required_return),
        freq=freq,
        customized_annual_factor=customized_annual_factor,
    )
    annualized_downside_risk = (
        _downside_risk
        if _downside_risk is not None
        else downside_risk(
            returns,
            freq=freq,
            required_return=required_return,
            customized_annual_factor=customized_annual_factor,
        )
    )

    # np.divide(average_annual_return, annualized_downside_risk, out=sr)
    # if return_1d:
    #    sr = sr.item()
    # elif isinstance(returns, pd.DataFrame):
    #    sr = pd.Series(sr)
    sr = average_annual_return.div(annualized_downside_risk.values)
    sr.index = ["Sortino Ratio"]
    return sr


def downside_risk(
    returns: Union[pd.DataFrame, pd.Series, np.ndarray],
    required_return: float = 0,
    freq: str = "D",
    customized_annual_factor: Union[float, int] = None,
) -> pd.DataFrame:
    """compute downside risk of given returns

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
    risk = np.empty(returns.shape[1:])

    returns_1d = returns.ndim == 1

    if len(returns) < 1:
        risk[()] = np.nan
        if returns_1d:
            risk = risk.item()
        return _results_to_df(risk, returns, "Downside Risk")

    ann_factor = annualization_factor(freq, customized_annual_factor)

    downside_diff = np.clip(
        _adjust_returns(
            returns,
            required_return,
        ),
        np.NINF,
        0,
    )

    np.square(downside_diff, out=downside_diff)
    np.nanmean(downside_diff, axis=0, out=risk)
    np.sqrt(risk, out=risk)
    np.multiply(risk, np.sqrt(ann_factor), out=risk)

    if returns_1d:
        risk = risk.item()
    # elif isinstance(returns, pd.DataFrame):
    #    risk = pd.Series(risk, index=returns.columns)
    # return risk
    return _results_to_df(risk, returns, "Downside Risk")


def alpha_beta(returns, factor_returns):
    # TODO compute alpha & beta
    return


def alpha(returns, factor_returns):
    # TODO compute alpha
    return


def beta(returns, factor_returns):
    # TODO compute beta
    return


def stability_of_timeseries(returns: pd.Series) -> pd.DataFrame:
    """compute R^2 of a linear fit to cumulative log returns

    Args:
        returns (Union[pd.DataFrame, pd.Series, np.ndarray]):
            noncumulative returs

    Returns:
        R^2: R-squared of the linear fit
    """
    ## TODO test for pd.DataFrame
    if len(returns) < 2:
        return np.nan

    returns = np.asanyarray(returns)
    returns = returns[~np.isnan(returns)]

    cum_log_returns = np.log1p(returns).cumsum()
    rhat = stats.linregress(np.arange(len(cum_log_returns)), cum_log_returns)[2]

    return rhat**2


def skew(returns: Union[pd.DataFrame, pd.Series, np.ndarray]) -> pd.DataFrame:
    """compute skewness of given returns

    Args:
        returns (Union[pd.DataFrame, pd.Series, np.ndarray]):
            noncumulative returns

    Returns:
        pd.DataFrame: skewness of given returns
    """
    return _results_to_df(data.to_df(returns).skew(axis=0), returns, "Skew")


def kurtosis(returns: Union[pd.DataFrame, pd.Series, np.ndarray]) -> pd.DataFrame:
    """compute kurtosis of given returns

    Args:
        returns (Union[pd.DataFrame, pd.Series, np.ndarray]):
            noncumulative returns

    Returns:
        pd.DataFrame: kurtosis of given returns
    """
    return _results_to_df(data.to_df(returns).kurtosis(), returns, "Kurtosis")


def tail_ratio(returns: Union[pd.DataFrame, pd.Series, np.ndarray]) -> pd.DataFrame:
    """compute tail ratio of returns, 95% percentile return/5% percentile return
    Returns at 95% quantile/Returns at 5% quantile

    Args:
        returns (Union[pd.DataFrame, pd.Series, np.ndarray]):
            noncumulative returns

    Returns:
        tail ratio: tail ratio of given returns
    """
    if len(returns) < 1:
        return np.nan

    # returns = np.asanyarray(returns)
    # Be tolerant of nan's
    returns = pd.DataFrame(returns[returns.notnull()])
    if len(returns) < 1:
        return np.nan

    return (
        pd.DataFrame(returns.quantile(0.95).div(abs(returns.quantile(0.05))))
        .rename(columns={0: "Tail Ratio"})
        .T
    )
    # return np.abs(np.percentile(returns, 95)) / np.abs(np.percentile(returns, 5))


def capture_ratio(
    returns: Union[pd.DataFrame, pd.Series, np.ndarray],
    benchmark_returns: Union[pd.DataFrame, pd.Series, np.ndarray],
    freq: str = "D",
):
    """compute the capture ratio of given returns against benchmark

    Args:
        returns (Union[pd.DataFrame, pd.Series, np.ndarray]):
            noncumulative returns
        benchmark_returns (Union[pd.DataFrame, pd.Series, np.ndarray]):
            noncumulative returns of a benchmark
        freq (str, optional):
            frequency of noncumulative returns.
            Defaults to "D". Can be "D", "W", "M", "Q", and "Y"

    Returns:
        capture ratio: capture ratio of given returns against benchmark
    """
    return (
        annualized_return(returns, freq=freq)
        .div(annualized_return(benchmark_returns, freq=freq).values)
        .T.rename(columns={"Annualized Return": "Capture Ratio"})
        .T
    )


def up_capture_ratio(
    returns: Union[pd.DataFrame, pd.Series, np.ndarray],
    benchmark_returns: Union[pd.DataFrame, pd.Series, np.ndarray],
    **kwargs
):
    """compute the capture ratio of given returns against benchmark on the positive side

    Args:
        returns (Union[pd.DataFrame, pd.Series, np.ndarray]):
            noncumulative returns
        benchmark_returns (Union[pd.DataFrame, pd.Series, np.ndarray]):
            noncumulative returns of a benchmark

    Returns:
        up capture ratio: capture ratio of given returns against benchmark on the positive side
    """
    return (
        up(returns, benchmark_returns, function=capture_ratio, **kwargs)
        .T.rename(columns={"Capture Ratio": "Up Capture Ratio"})
        .T
    )


def down_capture_ratio(
    returns: Union[pd.DataFrame, pd.Series, np.ndarray],
    benchmark_returns: Union[pd.DataFrame, pd.Series, np.ndarray],
    **kwargs
):
    """compute the capture ratio of given returns against benchmark on the negative side

    Args:
        returns (Union[pd.DataFrame, pd.Series, np.ndarray]):
            noncumulative returns
        benchmark_returns (Union[pd.DataFrame, pd.Series, np.ndarray]):
            noncumulative returns of a benchmark

    Returns:
        down capture ratio: capture ratio of given returns against benchmark on the negative side
    """
    return (
        down(returns, benchmark_returns, function=capture_ratio, **kwargs)
        .T.rename(columns={"Capture Ratio": "Down Capture Ratio"})
        .T
    )


def up_down_capture_ratio(
    returns: Union[pd.DataFrame, pd.Series, np.ndarray],
    benchmark_returns: Union[pd.DataFrame, pd.Series, np.ndarray],
    **kwargs
):
    """compute the capture ratio of given returns against benchmark on both sides

    Args:
        returns (Union[pd.DataFrame, pd.Series, np.ndarray]):
            noncumulative returns
        benchmark_returns (Union[pd.DataFrame, pd.Series, np.ndarray]):
            noncumulative returns of a benchmark

    Returns:
        up&down capture ratio: capture ratio of given returns against benchmark on both sides
    """
    return (
        up_capture_ratio(returns, benchmark_returns, **kwargs)
        .div(down_capture_ratio(returns, benchmark_returns, **kwargs).values)
        .T.rename(columns={"Up Capture Ratio": "Up Down Capture Ratio"})
        .T
    )


def historical_es(
    returns: Union[pd.DataFrame, pd.Series, np.ndarray], cutoff: float = 0.05
):
    """compute historical espected shortfall of given returns at cutoff percentile

    Args:
        returns (Union[pd.DataFrame, pd.Series, np.ndarray]):
            noncumulative returns
        cutoff (float, optional):
            cutoff percentage for historical ES. Defaults to 0.05.

    Returns:
        historical ES: historical espected shortfall of given returns at cutoff percentile
    """
    es_str = "Historical ES (" + str(int(cutoff * 100)) + "%) "
    return (
        pd.DataFrame(pd.DataFrame(returns).quantile(cutoff))
        .rename(columns={cutoff: es_str})
        .T
    )


def rolling_volatility(returns, rolling_window: int = 21):
    # TODO rolling volatility
    return


def rolling_sharpe_ratio(returns, rolling_window: int = 21):
    # TODO rolling sharpe ratio
    return


def rolling_regression(returns, factor_returns, rolling_window: int = 21):
    # TODO rolling regression of returns on factor returns
    return


def rolling_beta(returns, factor_returns, rolling_window: int = 21):
    # TODO rolling beta
    return
