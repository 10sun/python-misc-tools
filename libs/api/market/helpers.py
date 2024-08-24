'''
Author: J , jwsun1987@gmail.com
Date: 2023-11-24 00:13:40
LastEditors: J , jwsun1987@gmail.com
Description: 
Copyright: Copyright (c) 2024 by jwsun1987@gmail.com. All Rights Reserved.
'''


from datetime import datetime as dt
from typing import Union
import pandas as pd
from common.dates import *

from common.reporting.excel import *

def get_ticker_attr(data: dict, value, key):
    for v in data.values():
        if value in v.values():
            return v.get(key, None)
            
def sync_data(
    data: Union[pd.DataFrame, pd.Series],
    reference_dates: Union[list, pd.Series],
    tickers: dict,
    fill_method: str = 'ffill',
    header: str = "Instrument",
    attr: str = "Region",
):
    data = data.reindex(data.index.union(reference_dates)).fillna(method=fill_method)
    data = data[~data.index.duplicated(keep="first")]
    data = data.reindex(reference_dates)
    """
    data.index = date_func.get_next_weekday(
        data.index.to_series(), reference_dates.max()
    )
    """
    data.columns = [
        get_ticker_attr(tickers, h, attr)
        for h in data.columns.get_level_values(header).astype(str)
    ]
    return data
