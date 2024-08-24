'''
Author: J , jwsun1987@gmail.com
Date: 2023-12-08 00:28:26
LastEditors: J , jwsun1987@gmail.com
Description: 
Copyright: Copyright (c) 2024 by jwsun1987@gmail.com. All Rights Reserved.
'''


# from .utils import *
import numpy as np

from .enum import *

# from .calendar import *
# from ..enum.interesting_periods import *
from typing import Union, List
from datetime import timedelta


# __name__ = 'dates'
def get_date_frequency(dates: Union[pd.DatetimeIndex, List]) -> str:
    """get date frequency with given dates

    Args:
        dates (Union[pd.DatetimeIndex, List]): _description_

    Returns:
        _type_: _description_
    """
    try:
        if not isinstance(dates, pd.DatetimeIndex):  # TODO check if list works
            dates = pd.to_datetime(dates, infer_datetime_format=True)
        if pd.infer_freq(dates) is None:
            freq = pd.tseries.frequencies.to_offset(
                pd.to_timedelta(np.diff(dates).min())
            )
            return freq.freqstr
        else:
            return pd.infer_freq(dates)
    except Exception as e:
        print(e)
        return


def get_duration_number(
    data_freq: str, duration: Union[int, float], duration_unit: str
):
    if data_freq.split("-")[0] == "H":
        if duration_unit == "Y":
            conversion_ratio = 252 * 24
        elif duration_unit == "Q":
            conversion_ratio = 63 * 24
        elif duration_unit == "M":
            conversion_ratio = 22 * 24
        elif duration_unit == "W":
            conversion_ratio = 5 * 24
        elif duration_unit == "D":
            conversion_ratio = 1 * 24
        elif duration_unit == "H":
            conversion_ratio = 1
    elif data_freq.split("-")[0] == "D":
        if duration_unit == "Y":
            conversion_ratio = 252
        elif duration_unit == "Q":
            conversion_ratio = 63
        elif duration_unit == "M":
            conversion_ratio = 22
        elif duration_unit == "W":
            conversion_ratio = 5
        elif duration_unit == "D":
            conversion_ratio = 1
        elif duration_unit == "H":
            conversion_ratio = 1 / 24
    elif data_freq.split("-")[0] == "W":
        if duration_unit == "Y":
            conversion_ratio = 52
        elif duration_unit == "Q":
            conversion_ratio = 13
        elif duration_unit == "M":
            conversion_ratio = 4.285
        elif duration_unit == "W":
            conversion_ratio = 1
        elif duration_unit == "D":
            conversion_ratio = 0.2
        elif duration_unit == "H":
            conversion_ratio = 0.2 / 24
    elif data_freq.split("-")[0] == "M":
        if duration_unit == "Y":
            conversion_ratio = 12
        elif duration_unit == "Q":
            conversion_ratio = 3
        elif duration_unit == "M":
            conversion_ratio = 1
        elif duration_unit == "W":
            conversion_ratio = 1 / 4.285
        elif duration_unit == "D":
            conversion_ratio = 1 / 20
        elif duration_unit == "H":
            conversion_ratio = 1 / (20 * 24)
    elif data_freq.split("-")[0] == "Q":
        if duration_unit == "Y":
            conversion_ratio = 4
        elif duration_unit == "Q":
            conversion_ratio = 1
        elif duration_unit == "M":
            conversion_ratio = 1 / 3
        elif duration_unit == "W":
            conversion_ratio = 1 / 13
        elif duration_unit == "D":
            conversion_ratio = 1 / 63
    elif data_freq.split("-")[0] == "Y":
        if duration_unit == "Y":
            conversion_ratio = 1
        elif duration_unit == "Q":
            conversion_ratio = 1 / 4
        elif duration_unit == "M":
            conversion_ratio = 1 / 12
        elif duration_unit == "W":
            conversion_ratio = 1 / 52
        elif duration_unit == "D":
            conversion_ratio = 1 / 252
        elif duration_unit == "H":
            conversion_ratio = 1 / (252 * 24)
    return int(round(duration * conversion_ratio))


def start_end(start, end, n_days, fmt=None, tz=None, intraday=False):
    start = (
        pd.Timestamp(start)
        if start
        else (pd.Timestamp.now().normalize() - pd.offsets.Day(n_days))
    )
    end = pd.Timestamp(end) if end else pd.Timestamp.now()
    if end.normalize() == end and intraday:
        end = end + pd.Timedelta("23:59:59")
    start = tz_swap(start, tz, "UTC") if tz else start
    end = tz_swap(end, tz, "UTC") if tz else end
    return (start.strftime(fmt), end.strftime(fmt)) if fmt else (start, end)


def tz_swap(date, from_tz="UTC", to_tz="Europe/Zurich"):
    """[summary]

    Args:
        date ([type]): [description]
        from_tz (str, optional): [description]. Defaults to 'UTC'.
        to_tz (str, optional): [description]. Defaults to 'Europe/Zurich'.
    """
    if from_tz == to_tz:
        return pd.Timestamp(date)
    return (
        pd.Timestamp(date).tz_localize(from_tz).tz_convert(to_tz).replace(tzinfo=None)
    )


def tz_diff(
    date=None, from_tz="UTC", to_tz="Europe/Zurich"
) -> "Union[timedelta, pd.TimedeltaIndex]":
    if isinstance(date, pd.DatetimeIndex):
        return date.tz_localize(from_tz) - date.tz_localize(to_tz).tz_convert(from_tz)
    now = pd.Timestamp(date) if date else pd.Timestamp.now()
    return now - now.tz_localize(to_tz).tz_convert(from_tz).replace(tzinfo=None)


def get_utc_timestamp(dt_info):
    dt_info = pd.to_datetime(dt_info)
    try:
        dt_info = dt_info.tz_localize("UTC")
    except TypeError:
        dt_info = dt_info.tz_convert("UTC")
    return dt_info
