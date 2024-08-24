'''
Author: J , jwsun1987@gmail.com
Date: 2023-11-22 02:01:42
LastEditors: J , jwsun1987@gmail.com
Description: 
Copyright: Copyright (c) 2024 by jwsun1987@gmail.com. All Rights Reserved.
'''


import pandas as pd
from typing import Union

import os
from pathlib import Path

from plotly import express as px
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from plotly import figure_factory as ff

from common.reporting.charts import plotly_wrapper as plotly_charts

from .functions import *


def create_full_tearsheet(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    period: str = "D",
    window: Union[float, int] = 135,
    annualization=None,
    positions=None,
    transactions=None,
    market_data=None,
    slippage=None,
    live_start_date=None,
    sector_mapping=None,
):
    return


def create_summary_tearsheet(
    returns: Union[pd.Series, pd.DataFrame],
    benchmark_returns: pd.Series = None,
    freq: str = "D",
    customized_annual_factor=None,
    window: float = 135,
    positions=None,
    transactions=None,
    slippage=None,
    intraday="infer",
    live_start_date=None,
    turnover_denom="AGB",
    header_rows=None,
    path=None,
    filename=None,
):
    # performance summary
    if benchmark_returns is not None:
        returns, benchmark_returns = clip_returns_to_benchmark(
            returns, benchmark_returns
        )
        benchmark_returns = data_helper.to_series(benchmark_returns)
        if benchmark_returns.name not in returns.columns:
            returns = pd.concat(
                [returns, benchmark_returns.rename("benchmark")], axis=1
            )
        else:
            returns = returns.rename(
                columns={
                    benchmark_returns.name: benchmark_returns.name + " (benchmark)"
                }
            )

    perf_summary = performance_summary(
        returns=returns,
        freq=freq,
        customized_annual_factor=customized_annual_factor,
        window=window,
    )

    # summary_table = ff.create_table(stats_summary)
    # summary_table.show()
    specs = [
        [{"colspan": 2, "type": "table"}, None],
        [{"colspan": 2}, None],
        # [{"colspan":2}, None],
        [{"colspan": 2}, None],
        [{"colspan": 2}, None],
        [{"colspan": 2}, None],
        [{"colspan": 2}, None],
    ]

    rows = 6 + len(perf_summary["yearly returns"].columns)
    row_heights = [0.35, 0.3, 0.2, 0.2, 0.2, 0.2]

    for i in range(len(perf_summary["yearly returns"].columns)):
        specs += [[{}, {}]]
        row_heights += [0.6]

    fig = make_subplots(
        rows=rows,  # 7
        cols=2,
        subplot_titles=(
            "Performance Summary",
            "Cumulative Returns",
            # "Positions",
            "Underwater",
            "Rolling Annual Volatility",
            "Rolling Sharpe Ratio",
            "Yearly Return",
        ),
        vertical_spacing=0.025,
        specs=specs,
        row_heights=row_heights,
        shared_xaxes=True,
    )

    # 1. table to summarize
    plotly_charts.add_table(fig, perf_summary["summary"].round(2), row=1)

    # 2. cumulative returns
    plotly_charts.add_curve(fig, perf_summary["cumulative returns"], row=2)

    # 3. positions

    # 4. underwater
    plotly_charts.add_curve(fig, perf_summary["underwater"], row=3)

    # 6. rolling annual vol
    plotly_charts.add_curve(fig, perf_summary["rolling vol"], row=4)

    # 7. rolling sharpe ratio
    plotly_charts.add_curve(fig, perf_summary["rolling sharpe ratio"], row=5)

    # 5. monthly return heatmap & annual return bars
    yearly_returns = perf_summary["yearly returns"]
    yearly_return_avg = pd.DataFrame(
        index=yearly_returns.index,
        data=yearly_returns.mean(axis=0).values[0],
        columns=yearly_returns.columns,
    )

    plotly_charts.add_bars(fig, yearly_returns, orientation="v", row=6)
    plotly_charts.add_curve(fig, yearly_return_avg, linestyle=dict(dash="dot"), row=6)

    # add yearly and monthly for each portfolio
    monthly_returns = perf_summary["monthly returns"]
    for ind, strat in enumerate(list(yearly_returns.columns)):
        plotly_charts.add_bars(
            fig, yearly_returns[strat], row=7 + ind, col=1, reversed_yaxis=True
        )
        plotly_charts.add_curve(
            fig,
            yearly_return_avg[strat],
            linestyle=dict(dash="dot"),
            orientation="v",
            row=7 + ind,
        )
        plotly_charts.add_heatmap(
            fig, monthly_returns[(strat)], row=7 + ind, col=2, reversed_yaxis=True
        )

    fig.update_layout(
        height=2000,
        showlegend=False,
        # title_text='Performance Overview',
    )

    # save to html
    if path is None:
        fig.show()
    else:
        if not os.path.exists(path):
            os.makedirs(path)
        filepath = Path(path) / (filename + "Performance Overview.html")
        fig.write_html(filepath)


def create_return_tearsheet(
    returns: Union[pd.Series, pd.DataFrame],
    benchmark_returns: pd.Series = None,
    freq: str = "D",
    customized_annual_factor=None,
    window: float = 252,
    path=None,
    filename=None,
):
    # performance summary
    if benchmark_returns is not None:
        returns, benchmark_returns = clip_returns_to_benchmark(
            returns, benchmark_returns
        )
        benchmark_returns = data_helper.to_series(benchmark_returns)
        if benchmark_returns.name not in returns.columns:
            returns = pd.concat(
                [returns, benchmark_returns.rename("benchmark")], axis=1
            )
        else:
            returns = returns.rename(
                columns={
                    benchmark_returns.name: benchmark_returns.name + " (benchmark)"
                }
            )

    perf_summary = performance_summary(
        returns=returns,
        freq=freq,
        customized_annual_factor=customized_annual_factor,
        window=window,
        benchmark_returns=benchmark_returns,
    )

    # summary_table = ff.create_table(stats_summary)
    # summary_table.show()
    specs = [
        [{"colspan": 2, "type": "table"}, None],
        [{"colspan": 2}, None],
        # [{"colspan":2}, None],
        [{"colspan": 2}, None],
        [{"colspan": 2}, None],
        [{"colspan": 2}, None],
        [{"colspan": 2}, None],
    ]

    rows = 6 + len(perf_summary["yearly returns"].columns)
    row_heights = [0.35, 0.3, 0.2, 0.2, 0.2, 0.2]

    for i in range(len(perf_summary["yearly returns"].columns)):
        specs += [[{}, {}]]
        row_heights += [0.6]

    fig = make_subplots(
        rows=rows,  # 7
        cols=2,
        subplot_titles=(
            "Performance Summary",
            "Cumulative Returns",
            # "Positions",
            "Underwater",
            "Rolling Annual Volatility",
            "Rolling Sharpe Ratio",
            "Yearly Return",
        ),
        vertical_spacing=0.025,
        specs=specs,
        row_heights=row_heights,
        shared_xaxes=True,
    )

    # 1. table to summarize
    plotly_charts.add_table(fig, perf_summary["summary"].round(2), row=1)

    # 2. cumulative returns
    plotly_charts.add_curve(fig, perf_summary["cumulative returns"], row=2)

    # 3. positions

    # 4. underwater
    plotly_charts.add_curve(fig, perf_summary["underwater"], row=3)

    # 6. rolling annual vol
    plotly_charts.add_curve(fig, perf_summary["rolling vol"], row=4)

    # 7. rolling sharpe ratio
    plotly_charts.add_curve(fig, perf_summary["rolling sharpe ratio"], row=5)

    # 5. monthly return heatmap & annual return bars
    yearly_returns = perf_summary["yearly returns"]
    yearly_return_avg = pd.DataFrame(
        index=yearly_returns.index,
        data=yearly_returns.mean(axis=0).values[0],
        columns=yearly_returns.columns,
    )

    plotly_charts.add_bars(fig, yearly_returns, orientation="v", row=6)
    plotly_charts.add_curve(fig, yearly_return_avg, linestyle=dict(dash="dot"), row=6)

    # add yearly and monthly for each portfolio
    monthly_returns = perf_summary["monthly returns"]
    for ind, strat in enumerate(list(yearly_returns.columns)):
        plotly_charts.add_bars(
            fig, yearly_returns[strat], row=7 + ind, col=1, reversed_yaxis=True
        )
        plotly_charts.add_curve(
            fig,
            yearly_return_avg[strat],
            linestyle=dict(dash="dot"),
            orientation="v",
            row=7 + ind,
        )
        plotly_charts.add_heatmap(
            fig, monthly_returns[(strat)], row=7 + ind, col=2, reversed_yaxis=True
        )

    fig.update_layout(
        height=2000,
        showlegend=False,
        # title_text='Performance Overview',
    )

    # save to html
    if path is None:
        fig.show()
    else:
        if not os.path.exists(path):
            os.makedirs(path)
        filepath = Path(path) / (filename + " Performance Overview.html")
        print(filepath)
        fig.write_html(filepath)


def create_position_tearsheet():
    return NotImplementedError("Not implemented yet")


def create_transaction_tearsheet():
    return NotImplementedError("Not implemented yet")


def create_risk_tearsheet():
    return NotImplementedError("Not implemented yet")


def create_performance_attribution_tearsheet():
    return NotImplementedError("Not implemented yet")


def create_capacity_tearsheet():
    return NotImplementedError("Not implemented yet")
