'''
Author: J , jwsun1987@gmail.com
Date: 2023-11-21 01:38:22
LastEditors: J , jwsun1987@gmail.com
Description: 
Copyright: Copyright (c) 2024 by jwsun1987@gmail.com. All Rights Reserved.
'''

import pandas as pd
import numpy as np

from typing import Union


def to_list(data):
    if not isinstance(data, list):
        return [data]
    else:
        return data


def to_df(data: Union[pd.Series, np.ndarray]):
    if isinstance(data, pd.Series):
        return pd.DataFrame(data)
    elif isinstance(data, np.ndarray):
        return pd.DataFrame(data)
    elif isinstance(data, pd.DataFrame):
        return data
    else:
        return TypeError("data type not supported")


def to_series(data: Union[pd.DataFrame, np.ndarray]):
    if isinstance(data, pd.DataFrame):
        return data.squeeze(axis=1)
    elif isinstance(data, np.ndarray):
        return pd.Series(data)
    else:
        raise TypeError("data type not supported...")


## TODO convert dataframe to array
def to_array(df: Union[pd.Series, pd.DataFrame]):
    return NotImplementedError("Not implemented")
