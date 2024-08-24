'''
Author: J , jwsun1987@gmail.com
Date: 2024-02-09 18:05:56
LastEditors: J , jwsun1987@gmail.com
Description: 
Copyright: Copyright (c) 2024 by jwsun1987@gmail.com. All Rights Reserved.
'''


import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta


def flatten(arr):
    """flatten the array

    Args:
        arr (_type_): _description_

    Returns:
        _type_: _description_
    """
    return arr if not isinstance(arr, pd.Series) else arr.values


def vectorize(func):
    def wrapper(df, *args, **kwargs):
        if df.ndim == 1:
            return func(df, *args, **kwargs)
        elif df.ndim == 2:
            return df.apply(func, *args, **kwargs)

    return wrapper


def non_unique_bin_edges_error(func):
    message = ""

    def dec(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ValueError as e:
            raise e

def quantile_calc(
    data,
    quantiles: int = 5,
    bins=None,
    zero_aware: bool = False,
    no_raise: bool = False,
):
    try:
        if quantiles is not None and bins is None and not zero_aware:
            return pd.qcut(data, quantiles, labels=False) + 1
        elif quantiles is not None and bins is None and zero_aware:
            pos_quantiles = (
                pd.qcut(data[data >= 0], quantiles // 2, labels=False)
                + quantiles // 2
                + 1
            )
            neg_quantiles = pd.qcut(data[data < 0], quantiles // 2, labels=False) + 1
            return pd.concat([pos_quantiles, neg_quantiles]).sort_index()
        elif bins is not None and quantiles is None and not zero_aware:
            return pd.cut(data, bins, labels=False) + 1
        elif bins is not None and quantiles is None and zero_aware:
            pos_bins = pd.cut(data[data >= 0], bins // 2, labels=False) + bins // 2 + 1
            neg_bins = pd.cut(data[data < 0], bins // 2, labels=False) + 1
            return pd.concat([pos_bins, neg_bins]).sort_index()
    except Exception as e:
        if no_raise:
            return pd.Series(index=data.index)
        raise e


def quantize_data(
    data,
    quantile: int = 5,
    quantile_col: str = "factor",
    bins=None,
    by_group=False,
    no_raise=False,
    zero_aware=False,
):
    if not (
        (quantile is not None and bins is None)
        or (quantile is None and bins is not None)
    ):
        raise ValueError("Either quantiles or bins should be provided")

    if zero_aware and not (isinstance(quantile, int) or isinstance(bins, int)):
        msg = "zero_aware should only be True when quantiles or bins is an" " integer"
        raise ValueError(msg)

    grouper = [data.index.get_level_values("date")]
    if by_group:
        grouper.append("group")

    data_quantile = data.groupby(grouper)[quantile_col].apply(
        quantile_calc, quantile, bins, zero_aware, no_raise
    )
    data_quantile.name = "factor_quantile"

    return data_quantile.dropna()

def cut_data(data: pd.DataFrame, params: dict = None):
    # start: Union[str] = None
    # end: Union[str] = None):
    current_date = data.index.max().date()
    if params is None:
        params = {"window": 20, "unit": "Y"}

    if params.get("cutoff_date", None) is not None:
        start_date = params.get("cutoff_date", None)
    else:
        unit = params.get("unit", "Y")
        window = params.get("window", 20)

        if unit.casefold() == "Y".casefold():
            start_date = current_date - relativedelta(years=window)
        elif unit.casefold() == "Q".casefold():
            start_date = current_date - relativedelta(months=window * 3)
        elif unit.casefold() == "M".casefold():
            start_date = current_date - relativedelta(months=window)
        elif unit.casefold() == "W".casefold():
            start_date = current_date - relativedelta(weeks=window)
        elif unit.casefold() == "D".casefold():
            start_date = current_date - relativedelta(days=window)
        else:
            print(unit + " not implemented yet...")

    return data.loc[(data.index >= pd.to_datetime(start_date))]

def in_percentile(data: pd.Series, val):
    data_list = sorted(data.values.tolist())
    l = len(data_list)
    for p in np.linspace(1, 0, 101):
        if p < 1:
            in_p = val > data_list[int(l * p)]
            if in_p:
                return int(round(p * 100))