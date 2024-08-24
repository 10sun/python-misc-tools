'''
Author: J , jwsun1987@gmail.com
Date: 2023-11-22 02:36:40
LastEditors: J , jwsun1987@gmail.com
Description: 
Copyright: Copyright (c) 2024 by jwsun1987@gmail.com. All Rights Reserved.
'''


import sys
import math
from .chart_style import *
from .table_style import *

import pandas as pd

from common.data import *
from common.dates import *
from utils.tools import *

try:
    from pptx.chart.data import ChartData
    from pptx import enum as pptx_enum
except ImportError:
    import sys
    sys.path.append(r"\\merlin\lib_isg\\28.Alternative Data\Code\python-quant\libs\common\reporting\slides\python-pptx-master")
    from pptx.chart.data import ChartData
    from pptx import enum as pptx_enum

VALUE_AXIS_BASE = 10

def line_chart_data(data:pd.DataFrame, params:dict={}):
    chart_data = ChartData()
    try:
        if isinstance(data, pd.Series):
            data = pd.DataFrame(data)

        data = data.loc[data.index >= params.get("start", data.index.min())]
        chart_data.categories = data.index.tolist()

        ## get value axis scale
        data_value_max = -sys.maxsize
        data_value_min = sys.maxsize

        for col in data.columns:
            chart_data.add_series(col, data[col].squeeze())
            data_value_max = max(data[col].max(), data_value_max)
            data_value_min = min(data[col].min(), data_value_min)

        value_axis_base = pow(
            VALUE_AXIS_BASE,
            math.floor(
                math.log((data_value_max - data_value_min), VALUE_AXIS_BASE)
            ),
        )  # -1

        if value_axis_base > 1:
            # value_axis_base = value_axis_base# / VALUE_AXIS_BASE
            data_value_max = int(
                value_axis_base * math.ceil(data_value_max / value_axis_base)
            )
            data_value_min = int(
                value_axis_base * math.floor(data_value_min / value_axis_base)
            )
        else:
            value_axis_base = value_axis_base/VALUE_AXIS_BASE
            data_value_max = value_axis_base * math.ceil(
                data_value_max / value_axis_base
            )
            data_value_min = value_axis_base * math.floor(
                data_value_min / value_axis_base
            )
    except Exception as e:
        print(e)

    # plot grid chart
    chart_params = {}

    # chart title params
    chart_params = add_params(
        chart_params,
        {
            "value_axis": {
                "maximum_scale": data_value_max,
                "minimum_scale": data_value_min,
            },
        },
    )

    ## get category axis scale
    # TODO: smartly decide the ticks -> 4 - 5 major ticks at most for small chart, 10 for large ones, all should have meaningful dates
    if chart_data.categories.are_dates:
        index_freq = get_date_frequency(data.index)
        #print(index_freq)
        index_duration = get_duration_number(
            "Y", data.index.shape[0], index_freq.split("-")[0]
        )

        if (
            index_freq.split("-")[0] == "W"
            or index_freq.split("-")[0] == "D"
        ):
            if index_duration >= 10:
                cat_major_time_unit = pptx_enum.chart.XL_TIME_UNIT.YEARS
                cat_major_unit = 2
            else:
                cat_major_time_unit = pptx_enum.chart.XL_TIME_UNIT.MONTHS
                cat_major_unit = 6
        else:
            if index_duration >= 20:
                cat_major_time_unit = pptx_enum.chart.XL_TIME_UNIT.YEARS
                cat_major_unit = 5
            else:
                cat_major_time_unit = pptx_enum.chart.XL_TIME_UNIT.MONTHS
                cat_major_unit = 6

        chart_params = add_params(
            chart_params,
            {
                "category_axis": {
                    "major_unit": cat_major_unit,
                    "major_time_unit": cat_major_time_unit,
                },
            },
        )

    return {'data': chart_data, 'params':chart_params}