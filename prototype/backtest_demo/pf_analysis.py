'''
Author: J , jwsun1987@gmail.com
Date: 2023-02-02 02:48:28
LastEditors: J , jwsun1987@gmail.com
Description: 
Copyright: Copyright (c) 2024 by jwsun1987@gmail.com. All Rights Reserved.
'''

import empyrical as ep
import scipy.stats as stats
import scipy as sp
import pandas as pd
import numpy as np

import os
from pathlib import Path
from plotly import express as px
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from plotly import figure_factory as ff


def value_at_risk(returns, period=None, sigma=2.0):
    if period is not None:
        returns_agg = ep.aggregate_returns(returns, period)
    else:
        returns_agg = returns.copy()

    value_at_risk = returns_agg.mean() - sigma * returns_agg.std()
    return value_at_risk


SIMPLE_STAT_FUNCS = [
    ep.annual_return,
    ep.cum_returns_final,
    ep.annual_volatility,
    ep.sharpe_ratio,
    #ep.calmar_ratio,
    #ep.stability_of_timeseries,
    ep.max_drawdown,
    #ep.omega_ratio,
    #ep.sortino_ratio,
    #stats.skew,
    #stats.kurtosis,
    ep.tail_ratio,
    #value_at_risk,
]

STAT_FUNC_NAMES = {
    "annual_return": "Annual return",
    "cum_returns_final": "Cumulative returns",
    "annual_volatility": "Annual volatility",
    "sharpe_ratio": "Sharpe ratio",
    "calmar_ratio": "Calmar ratio",
    "stability_of_timeseries": "Stability",
    "max_drawdown": "Max drawdown",
    "omega_ratio": "Omega ratio",
    "sortino_ratio": "Sortino ratio",
    "skew": "Skew",
    "kurtosis": "Kurtosis",
    "tail_ratio": "Tail ratio",
    "common_sense_ratio": "Common sense ratio",
    "value_at_risk": "Daily value at risk",
    "alpha": "Alpha",
    "beta": "Beta",
}


def perf_stats(returns, positions=None, transactiosn=None, period='daily'):
    stats = pd.Series()
    for stat_func in SIMPLE_STAT_FUNCS:
        #print(stat_func.__name__)
        if stat_func.__name__ not in ['cum_returns_final', 'max_drawdown', 'tail_ratio']:
            stats[STAT_FUNC_NAMES[stat_func.__name__]] = stat_func(returns, period=period)
        else:
            stats[STAT_FUNC_NAMES[stat_func.__name__]] = stat_func(returns)
    return stats


def get_underwater(returns):
    df_cum_rets = ep.cum_returns(returns, starting_value=1.0)
    running_max = np.maximum.accumulate(df_cum_rets)
    underwater = -100 * ((running_max - df_cum_rets) / running_max)
    return underwater


def get_max_drawdown_underwater(underwater):
    valley = underwater.index[np.argmin(underwater)]  # end of the period
    # Find first 0
    try:
        peak_tmp = underwater[:valley][underwater[:valley] == 0].dropna(
            axis=0, how="any"
        )
        peak = peak_tmp.index[-1]
    except IndexError:
        peak = np.nan
    # Find last 0
    try:
        recovery_tmp = underwater[valley:][underwater[valley:] == 0].dropna(
            axis=0, how="any"
        )
        recovery = recovery_tmp.index[0]
    except IndexError:
        recovery = np.nan  # drawdown not recovered
    return peak, valley, recovery


def get_top_drawdowns(returns, top=10):
    returns = returns.copy()
    df_cum = ep.cum_returns(returns, 1.0)
    running_max = np.maximum.accumulate(df_cum)
    underwater = df_cum / running_max - 1

    drawdowns = []
    for t in range(top):
        peak, valley, recovery = get_max_drawdown_underwater(underwater)
        # Slice out draw-down period
        if not pd.isnull(recovery):
            underwater.drop(underwater[peak:recovery].index[1:-1], inplace=True)
        else:
            # drawdown has not ended yet
            underwater = underwater.loc[:peak]

        drawdowns.append((peak, valley, recovery))
        if (len(returns) == 0) or (len(underwater) == 0):
            break

    return drawdowns


def perf_drawdown(returns, top=5):
    df_cum = ep.cum_returns(returns, 1.0)
    drawdown_periods = get_top_drawdowns(returns, top=top)
    df_drawdowns = pd.DataFrame(
        index=list(range(top)),
        columns=[
            "Net drawdown in %",
            "Peak date",
            "Valley date",
            "Recovery date",
            "Duration",
        ],
    )

    for i, (peak, valley, recovery) in enumerate(drawdown_periods):
        if pd.isnull(recovery):
            df_drawdowns.loc[i, "Duration"] = np.nan
        else:
            df_drawdowns.loc[i, "Duration"] = len(
                pd.date_range(peak, recovery, freq="B")
            )
        df_drawdowns.loc[i, "Peak date"] = peak.to_pydatetime().strftime("%Y-%m-%d")
        df_drawdowns.loc[i, "Valley date"] = valley.to_pydatetime().strftime("%Y-%m-%d")
        if isinstance(recovery, float):
            df_drawdowns.loc[i, "Recovery date"] = recovery
        else:
            df_drawdowns.loc[i, "Recovery date"] = recovery.to_pydatetime().strftime(
                "%Y-%m-%d"
            )
        df_drawdowns.loc[i, "Net drawdown in %"] = (
            (df_cum.loc[peak] - df_cum.loc[valley]) / df_cum.loc[peak]
        ) * 100

    df_drawdowns["Peak date"] = pd.to_datetime(df_drawdowns["Peak date"])
    df_drawdowns["Valley date"] = pd.to_datetime(df_drawdowns["Valley date"])
    df_drawdowns["Recovery date"] = pd.to_datetime(df_drawdowns["Recovery date"])

    return df_drawdowns


def return_dist(returns):
    x = returns.values
    return pd.Series(
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
    )


def get_frequency_factor(frequency):
    if frequency == "D":
        factor = 252
    elif frequency == "W":
        factor = 52
    elif frequency == "M":
        factor = 12
    elif frequency == "Q":
        factor = 4
    elif frequency == "Y":
        factor = 1
    return factor


def get_rolling_vol(returns, rolling_window=136, frequency="D"):
    factor = get_frequency_factor(frequency)
    return returns.rolling(rolling_window).std() * np.sqrt(factor)


def get_rolling_sharpe(returns, rolling_window=136, frequency="D"):
    factor = get_frequency_factor(frequency)
    return (
        returns.rolling(rolling_window).mean()
        / returns.rolling(rolling_window).std()
        * np.sqrt(factor)
    )


def portfolio_return_analysis(returns, rolling_window=136, frequency="D"):
    summary = perf_stats(returns)
    cumulative_returns = (1 + returns).cumprod() - 1
    underwater = get_underwater(returns)
    rolling_vol = get_rolling_vol(returns, rolling_window, frequency)
    rolling_shape = get_rolling_sharpe(returns, rolling_window, frequency)
    monthly_returns = ep.aggregate_returns(returns, "monthly")
    annual_returns = ep.aggregate_returns(returns, "yearly")
    return {
        "summary": pd.DataFrame(summary),
        "cumulative returns": pd.DataFrame(cumulative_returns),
        "underwater": pd.DataFrame(underwater),
        "rolling vol": pd.DataFrame(rolling_vol),
        "rolling sharpe": pd.DataFrame(rolling_shape),
        "monthly returns": pd.DataFrame(monthly_returns),
        "annual returns": pd.DataFrame(annual_returns),
        # "stress time":,
    }


def portfolio_overview(returns, portfolio_str, positions=None, trades=None, firdir=None):
    fig = make_subplots(
        rows=4,
        cols=3,
        subplot_titles=(
            "Cumulative Returns",
            "Drawdown",
            "Rolling Volatility",
            "Rolling Sharpe Ratio",
        ),
        specs=[
            [
                {"colspan": 3},
                None,
                None,
            ],
            [
                {"colspan": 3},
                None,
                None,
            ],
            [
                {"colspan": 3},
                None,
                None,
            ],
            [
                {"colspan": 3},
                None,
                None,
            ],
            # [{"colspan": 3}, None, None,],
            # [{"colspan": 3}, None, None,],
            # [{"colspan": 3}, None, None,],
            # [{},{},{},],
        ],
    )
    returns = pd.DataFrame(returns)
    portfolio_return_summary = portfolio_return_analysis(returns, rolling_window=12, frequency="M")

    # 1. summary table
    summary_table = ff.create_table(pd.DataFrame(portfolio_return_summary["summary"]))
    # for trace in range(len(summary_table["data"])):
    #    fig.add_trace(summary_table["data"][trace], row=1, col=1)
    # fig.add_trace(, row=1, col=1)

    # 2. overview
    # positions_area = px.area(positions)
    # for trace in range(len(positions_area["data"])):
    #    fig.add_trace(positions_area["data"][trace], row=2, col=1)
    for c in returns.columns:
        fig.add_trace(
            go.Scatter(
                x=portfolio_return_summary["cumulative returns"].index,
                y=portfolio_return_summary["cumulative returns"][c],
                mode="lines",
                name=c,
                legendgroup=c,
                showlegend=True,
            ),
            row=1,
            col=1,
        )

    # add position info str
    # fig.add_trace(
    #    go.Scatter(
    #        x=trades_str.index,
    #        y=trades[assets[0]],
    #        text=trades_str.values,
    #        mode="markers+text",
    #        name="Positions",
    #        textposition="top center",
    #        textfont_size=8,
    #    ),
    #    row=2,
    #    col=1,
    # )

    # 3. returns
    # fig.add_trace(
    #    go.Scatter(
    #        x=portfolio_return_summary["cumulative returns"].index,
    #        y=portfolio_return_summary["cumulative returns"],
    #        mode="lines",
    #        name="Cumulative Returns",
    #    ),
    #    row=3,
    #    col=1,
    # )

    # 4. positions
    # fig.add_traces(positions_area_traces, row=4, col=1)
    if positions is not None:
        assets = positions.columns
        trades_str = trades.apply(
            lambda row: row.Regime.split(" -> ")[-1]
            + "<br>"
            + "<br>".join(
                [asset + ": " + str(row[asset] * 100) + "%" for asset in assets]
            ),
            axis=1,
        )

        positions_area = px.area(positions)
        for trace in range(len(positions_area["data"])):
            fig.add_trace(
                positions_area["data"][trace], row=1, col=1, secondary_y=False
            )

    # 5. underwater
    for c in returns.columns:
        fig.add_trace(
            go.Scatter(
                x=portfolio_return_summary["underwater"].index,
                y=portfolio_return_summary["underwater"][c],
                mode="lines",
                name=c,
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    # 6. rolling vol
    for c in returns.columns:
        fig.add_trace(
            go.Scatter(
                x=portfolio_return_summary["rolling vol"].index,
                y=portfolio_return_summary["rolling vol"][c],
                mode="lines",
                name=c,
                showlegend=False,
            ),
            row=3,
            col=1,
        )

    # 7. rolling sharpe
    for c in returns.columns:
        fig.add_trace(
            go.Scatter(
                x=portfolio_return_summary["rolling sharpe"].index,
                y=portfolio_return_summary["rolling sharpe"][c],
                mode="lines",
                name=c,
                showlegend=False,
            ),
            row=4,
            col=1,
        )

    # 8. monthly detail
    """
    fig.add_trace(
        go.Heatmap(
            z=portfolio_return_summary["monthly returns"].values,
            x=portfolio_return_summary["monthly returns"].index.get_level_values(1),
            y=portfolio_return_summary["monthly returns"].index.get_level_values(0),
            text=portfolio_return_summary["monthly returns"].values
        ),
        row=5,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=portfolio_return_summary["annual returns"],
            y=portfolio_return_summary["annual returns"].index,
            orientation='h',
        ),
        row=5,
        col=2,
    )
    monthly_hist = px.histogram(portfolio_return_summary["monthly returns"])
    for trace in range(len(monthly_hist["data"])):
        # monthly_hist_traces.append(monthly_hist["data"][trace])
        fig.add_trace(monthly_hist["data"][trace], row=5, col=3)
    """
    fig.update_layout(showlegend=False)
    if firdir is None:
        fig.show()
    else:
        if not os.path.exists(firdir):
            os.makedirs(firdir)
        filepath = Path(firdir) / (portfolio_str + "_Performance_Overview.html")
        fig.write_html(filepath)
