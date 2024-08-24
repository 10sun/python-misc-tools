'''
Author: J , jwsun1987@gmail.com
Date: 2023-11-22 01:41:04
LastEditors: J , jwsun1987@gmail.com
Description: 
Copyright: Copyright (c) 2024 by jwsun1987@gmail.com. All Rights Reserved.
'''


from typing import Union
import pandas as pd


def df_to_tableau(
    data: Union[pd.DataFrame, pd.Series],
    kpi_str: str,
    index_name: str = "Date",
    col_name: str = "Index",
    kpi_name: str = "KPI",
    value_name: str = "Value",
    additional_cols: dict = None,
) -> pd.DataFrame:
    df_list = []
    for c in data.columns:
        tmp = pd.DataFrame(data[c].rename(value_name))
        tmp.insert(loc=0, column=kpi_name, value=[kpi_str] * tmp.shape[0])
        tmp.insert(loc=0, column=col_name, value=[c] * tmp.shape[0])
        tmp.index.names = [index_name]
        df_list.append(tmp)
    tableauDf = pd.concat(df_list, axis=0)
    if additional_cols is not None:
        for col, values in additional_cols.items():
            tableauDf[col] = values
    return tableauDf
