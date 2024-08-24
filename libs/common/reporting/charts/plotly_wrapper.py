'''
Author: J , jwsun1987@gmail.com
Date: 2024-02-09 17:20:18
LastEditors: J , jwsun1987@gmail.com
Description: 
Copyright: Copyright (c) 2024 by jwsun1987@gmail.com. All Rights Reserved.
'''


import pandas as pd
import numpy as np

from plotly import express as px
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from plotly import figure_factory as ff
from typing import Union

from common.reporting.style.lo_color import *
from common import data as data_helper
from utils.tools import *

BENCHMARK_COLOR = LO_GRAYBLUE


def color_palette(
    palette: LO_COLOR_PALETTE,
    specified_colors: dict = {"benchmark": BENCHMARK_COLOR},
):
    for name, color in specified_colors.items():
        if name.upper() in [c.upper() for c in palette]:
            palette = [c for c in palette if c != color]
    return palette


def add_grid(fig):
    return


def add_curve(
    fig,
    data: Union[pd.Series, pd.DataFrame],
    name: str = None,
    mode="lines",
    linestyle=dict(),
    orientation="h",
    palette=LO_COLOR_PALETTE,
    showlegend: bool = True,
    row: int = 1,
    col: int = 1,
    secondary_y: bool = False,
):
    data = data_helper.to_df(data)
    palette = color_palette(palette)

    linestyle = add_params(dict(color=LO_DARKGRAYBLUE, width=1.5), linestyle)

    for ind, data_col in enumerate(list(data.columns)):
        if data_col.casefold() == "benchmark":
            linestyle["color"] = BENCHMARK_COLOR
        else:
            linestyle["color"] = palette[ind]

        if orientation == "h":
            data_x = data.index
            data_y = data[data_col].values
            hovertemplate = "%{x}: %{y}"
        else:
            data_x = data[data_col].values
            data_y = data.index
            hovertemplate = "%{y}: %{x}"

        fig.add_trace(
            go.Scatter(
                x=data_x,
                y=data_y,
                mode=mode,
                line=linestyle,
                name=data_col if name is None else name,
                showlegend=showlegend,
                hovertemplate=hovertemplate,
            ),
            row=row,
            col=col,
            secondary_y=secondary_y,
        )


def matplotlib_to_plotly(cmap, pl_entries):
    h = 1.0 / (pl_entries - 1)
    pl_colorscale = []

    for k in range(pl_entries):
        C = map(np.uint8, np.array(cmap(k * h)[:3]) * 255)
        print(C)
        pl_colorscale.append([k * h, "rgb" + str((C[0], C[1], C[2]))])

    return pl_colorscale


def add_heatmap(
    fig,
    data: pd.Series,
    colorscale=["red", "white", "green"],
    showlegend: bool = False,
    showscale: bool = False,
    reversed_xaxis: bool = False,
    reversed_yaxis: bool = False,
    row: int = 1,
    col: int = 1,
):
    # data = data_helper.to_df(data)
    data = data.sort_index(ascending=True)
    try:
        # magma_cmap = matplotlib.cm.get_cmap('RdBu_r')

        # magma = matplotlib_to_plotly(magma_cmap, 255)

        fig.add_trace(
            go.Heatmap(
                z=data.values,
                x=data.index.get_level_values(1),
                y=data.index.get_level_values(0),
                text=(round(data * 100, 1)).values,
                texttemplate="%{text}%",
                colorscale=colorscale,
                showlegend=showlegend,
                showscale=showscale,
                hovertemplate="%{y}.%{x}: %{z:.2f}",
            ),
            row=row,
            col=col,
        )
        if reversed_xaxis:
            fig.update_xaxes(row=row, col=col, autorange="reversed")
        if reversed_yaxis:
            fig.update_yaxes(row=row, col=col, autorange="reversed")

    except Exception as e:
        raise e


def add_bars(
    fig,
    data: pd.Series,
    orientation: str = "h",
    showlegend: bool = True,
    reversed_xaxis: bool = False,
    reversed_yaxis: bool = False,
    xaxis_title: str = None,
    yaxis_title: str = None,
    color=None,
    palette=LO_COLOR_PALETTE,
    row: int = 1,
    col: int = 1,
):
    data = data_helper.to_df(data)
    palette = color_palette(palette)

    for ind, data_col in enumerate(list(data.columns)):
        if color is not None:
            data_color = color
        elif data_col.casefold() == "benchmark":
            data_color = BENCHMARK_COLOR
        else:
            data_color = palette[ind]

        if orientation == "v":
            data_x = data.index
            data_y = data[data_col]
            hovertemplate = "%{x}: %{y}"
        elif orientation == "h":
            data_x = data[data_col]
            data_y = data.index
            hovertemplate = "%{y}: %{x}"

        fig.add_trace(
            go.Bar(
                x=data_x,
                y=data_y,
                orientation=orientation,
                showlegend=showlegend,
                name=data_col,
                marker=dict(color=data_color),
                hovertemplate=hovertemplate,
            ),
            row=row,
            col=col,
        )
        if reversed_xaxis:
            fig.update_xaxes(row=row, col=col, autorange="reversed")
        if reversed_yaxis:
            fig.update_yaxes(row=row, col=col, autorange="reversed")
        if xaxis_title:
            fig.update_xaxes(row=row, col=col, title_text=xaxis_title)
        if yaxis_title:
            fig.update_yaxes(row=row, col=col, title_text=yaxis_title)


def add_table(
    fig,
    data: Union[pd.Series, pd.DataFrame],
    showlegend: bool = False,
    row: int = 1,
    col: int = 1,
):
    data = data_helper.to_df(data)
    fig.add_trace(
        go.Table(
            header=dict(
                values=data.reset_index().rename(columns={data.index.name: ""}).columns
            ),  # data.reset_index(names="")
            cells=dict(values=data.reset_index().T.values),
        ),
        row=row,
        col=col,
    )
