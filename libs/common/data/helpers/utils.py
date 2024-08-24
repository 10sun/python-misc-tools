'''
Author: J , jwsun1987@gmail.com
Date: 2023-11-18 19:28:38
LastEditors: J , jwsun1987@gmail.com
Description: 
Copyright: Copyright (c) 2024 by jwsun1987@gmail.com. All Rights Reserved.
'''

import numpy as np
import pandas as pd
from typing import Union, Dict, List, Optional
from . import checker


def info(data: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
    """get basic information for a dataframe

    Args:
        data (_type_): _description_

    Returns:
        _type_: _description_
    """
    data_info = data.describe().round(2).T
    data_info.columns = [c.title() for c in data_info.columns]
    if "Count" in data_info.columns:
        data_info["Count"] = data_info["Count"].astype(int)
    data_quality = checker.check_data_integrity(data)
    dates_info = checker.check_date_info(data)
    return pd.concat(
        [
            dates_info,
            data_info,
            data_quality,
        ],
        axis=1,
    )


def find_nonzero_runs(data: np.array) -> list:
    """get the coordinats of non-zero subarrays

    Args:
        data (np.array): [description]

    Returns:
        list: the list of indices of the first and last element of non-zero subarrays
    """
    isnonzero = np.concatenate(([0], (np.asarray(data) != 0).view(np.int8), [0]))

    abs_diff = np.abs(np.diff(isnonzero))
    ranges = np.where(abs_diff == 1)[0].reshape(-1, 2)
    return ranges


def z_score(data: Union[pd.DataFrame, pd.Series, np.array]):
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    return data.apply(lambda col: (col - col.mean()) / col.std(), axis=0)


def unique(data: list) -> list:
    """[summary]

    Args:
        data (list): [description]

    Returns:
        list: [description]
    """
    seen = set()
    return [x for x in data if not (x in seen or seen.add(x))]


def squeeze_index(index: "pd.Index", squeeze_from):
    if isinstance(index, pd.MultiIndex) and squeeze_from:
        for level in [1, 0] if squeeze_from == "inner" else [0, 1]:
            if len(unique(index.get_level_values(level))) == 1:
                return index.droplevel(level)
    return index
