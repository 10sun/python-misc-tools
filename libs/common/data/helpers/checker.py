'''
Author: J , jwsun1987@gmail.com
Date: 2023-11-18 19:52:08
LastEditors: J , jwsun1987@gmail.com
Description: 
Copyright: Copyright (c) 2024 by jwsun1987@gmail.com. All Rights Reserved.
'''


from typing import Union
import pandas as pd
from common.dates import get_date_frequency

def check_missing_data(data: Union[pd.DataFrame, pd.Series]):
    if isinstance(data, pd.Series):
        data = pd.DataFrame(data)
    missing = data.isnull().sum()
    return missing[missing > 0]


def check_duplicate_data(data: Union[pd.DataFrame, pd.Series]):
    """    # Check for duplicate dates or entries

    Args:
        data (Union[pd.DataFrame, pd.Series]): _description_

    Returns:
        _type_: _description_
    """
    if isinstance(data, pd.Series):
        data = pd.DataFrame(data)
    duplicates = data[data.duplicated()]
    duplicates_info = duplicates.count(axis=0) / data.shape[0]
    return {"summary": duplicates_info.round(2), "duplicates": duplicates}


def check_data_outlier(
    data: Union[pd.DataFrame, pd.Series], column_name: str = None, z_threshold: int = 2
):

    # Basic Z-score method for outlier detection
    if isinstance(data, pd.Series):
        data = pd.DataFrame(data)

    if column_name is None:
        column_name = data.columns
    outliers = data[column_name].apply(
        lambda col: (col - col.mean()) / col.std(), axis=0
    )
    outliers = outliers[outliers > z_threshold]
    return {
        "summary": (outliers.count(axis=0) / data.shape[0]).round(2),
        "outliers": outliers[outliers.notnull()],
    }


def check_data_continuity(data: Union[pd.DataFrame, pd.Series]):
    # Check if time series data is continuous
    if isinstance(data, pd.Series):
        data = pd.DataFrame(data)

    freq = pd.infer_freq(data)
    if not freq:
        ## TODO: get the str for data frequency
        print(freq)

    expected_range = pd.date_range(
        start=data.index.min(), end=data.index.max(), freq=freq
    )
    missing_dates = expected_range.difference(data.index)
    return missing_dates


def check_data_types(data: Union[pd.DataFrame, pd.Series], expected_types):
    # Check if columns have the expected data types
    if isinstance(data, pd.Series):
        data = pd.DataFrame(data)

    mismatches = {}
    for col, dtype in expected_types.items():
        if data[col].dtype != dtype:
            mismatches[col] = data[col].dtype
    return mismatches


def check_data_integrity(
    data: Union[pd.DataFrame, pd.Series],
    column_name_for_outliers=None,
    expected_types=None,
):
    results = {
        "Missing Data": check_missing_data(data),
        "Duplicates (%)": check_duplicate_data(data)["summary"],
        #'Data Continuity': check_data_continuity(data, freq),
        "Outlier (%)": check_data_outlier(data, column_name_for_outliers)["summary"],
    }

    if expected_types:
        results["Data Type Mismatches"] = check_data_types(expected_types)

    return pd.DataFrame.from_dict(results)


def check_date_info(data, ax:int = 0):
    return pd.concat(
        [
            data.apply(lambda col: col.first_valid_index(), axis=ax).rename('Start'),
            data.apply(lambda col: col.last_valid_index(), axis=ax).rename('End'),
            data.apply(lambda col: get_date_frequency(col.index), axis=ax).rename('Freq'),
        ], 
        axis=1
    )