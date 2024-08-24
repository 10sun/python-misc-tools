'''
Author: J , jwsun1987@gmail.com
Date: 2022-11-11 01:47:56
LastEditors: J , jwsun1987@gmail.com
Description: 
Copyright: Copyright (c) 2024 by jwsun1987@gmail.com. All Rights Reserved.
'''


import logging
from typing import Union
from numpy import isin
import pandas as pd
import os

from datetime import datetime as dt
from pathlib import Path
from typing import Union, Dict

from utils import logger


def pd_to_sheet(data: pd.DataFrame, excel_file, sheet_name: str):
    if isinstance(data.index, pd.DatetimeIndex) or isinstance(data.index, dt):
        if data.columns.nlevels == 1:
            data.index.names = ["date"]
    data.to_excel(excel_file, sheet_name=sheet_name, index_label=False)
    #if data.columns.nlevels > 1:
    #    excel_file.sheets[sheet_name].set_row(data.columns.nlevels, None, None, {'hidden': True})
    #    excel_file.save()


def export_to_excel(
    data: Union[Dict, pd.DataFrame, pd.Series],
    filepath: Union[str, Path],
    params: Union[Dict, None] = None,
):
    if not params:
        params = {
            "overwrite": True,
            "mulindex": False,
            "date_format": "yyyy.mm.dd",
            "datetime_format": "yyyy.mm.dd",
        }

    if not isinstance(data, Dict):
        data_to_write = {"data": data}
    else:
        data_to_write = data

    if os.path.isfile(filepath):
        current_book = pd.read_excel(filepath, sheet_name=None)

    data_file = pd.ExcelWriter(
        filepath,
        engine='openpyxl', #engine="xlsxwriter"
        date_format=params.get("date_format", "yyyy.mm.dd"),
        datetime_format=params.get("datetime_format", "yyyy.mm.dd"),
        #engine_kwargs={"strings_to_urls": False},
    )

    if params.get("overwrite", True):
        for k, v in data_to_write.items():
            pd_to_sheet(v, data_file, k)

    else:
        for k, v in data_to_write.items():
            v_to_write = pd.DataFrame(v) if not isinstance(v, pd.DataFrame) else v
            if k in current_book.keys():
                if not params.get("overwrite_sheet", False):
                    current_v = current_book[k]
                    # if the matrix header is a date index
                    if type(current_v.columns[1]) in [pd.DatetimeIndex, dt]:
                        current_v = current_v.T
                        current_v.columns = current_v.iloc[0]
                        current_v = current_v.drop(current_v.index[0])
                        current_v.index = pd.to_datetime(current_v.index)
                        current_v.reset_index(inplace=True)
                        date_col_name = current_v.select_dtypes(["datetime64"]).columns[
                            0
                        ]
                        current_v = current_v.rename({date_col_name: "date"}, axis=1)
                        current_v = current_v.set_index("date")
                        index_is_dt = False
                    elif "date" in current_v.columns:
                        current_v = current_v.set_index("date")

                    if type(v_to_write.columns[0]) in [dt, pd.DatetimeIndex]:
                        v_to_write = v_to_write.T

                    new_cols = v_to_write[
                        v_to_write.columns.difference(current_v.columns)
                    ]

                    if not new_cols.empty:
                        v_to_write = pd.concat([current_v, new_cols], axis=1)
                    v_to_write = pd.concat(
                        [
                            current_v[~current_v.index.isin(v_to_write.index)],
                            v_to_write,
                        ],
                        axis=0,
                    ).sort_index()

                    if type(v_to_write.index) in [dt, pd.DatetimeIndex]:
                        v_to_write.index.names = ["date"]

                    if not index_is_dt:
                        v_to_write = v_to_write.T

            v_to_write.to_excel(data_file, sheet_name=k, index_label=False)

    data_file.close()
    logging.info(filepath)
