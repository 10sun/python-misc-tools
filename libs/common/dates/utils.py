'''
Author: J , jwsun1987@gmail.com
Date: 2023-11-22 02:01:32
LastEditors: J , jwsun1987@gmail.com
Description: 
Copyright: Copyright (c) 2024 by jwsun1987@gmail.com. All Rights Reserved.
'''


import calendar
import pandas as pd
import numpy as np
from datetime import datetime as dt
from datetime import timedelta
#import holidays
import pandas_market_calendars as mcal
from pandas.tseries.offsets import BDay

from typing import Iterable, List, Union, Dict, Optional, Sequence, Tuple

# from ..enum import country
#from .calendar import TradingCalendar


# %% get date conversion factors
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

# %% get previous/next date
def business_day_offset(
    dates,
    offsets: Union[int, Iterable[int]],
    roll: str = "raise",
    calendars: Union[str, Tuple[str, ...]] = (),
    week_mask: Optional[str] = None,
):
    calendar = TradingCalendar.get(calendars)
    results = np.busday_offset(
        dates, offsets, roll, busdaycal=calendar.business_day_calendar(week_mask)
    ).astype(dt.date)
    return tuple(results) if isinstance(results, np.ndarray) else results


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


def next_weekday(
    date: dt,
    weekday: int,
):
    days_ahead = weekday - date.weekday()
    if days_ahead <= 0:  # Target day already happened this week
        days_ahead += 7
    return date + timedelta(days_ahead)


def get_next_weekday(
    dates: Union[pd.DataFrame, pd.Series],
    reference_date: dt,
):
    new_dates = dates.apply(lambda x: next_weekday(x, reference_date.weekday()))
    new_dates = new_dates.apply(lambda x: reference_date if x > reference_date else x)
    return new_dates


def next_bday(date: Union[dt.date, Tuple[dt.date, ...]], n: int = 1):
    # TODOS: to update the next bday
    if isinstance(date, Tuple[dt.date, ...]):
        next_date = [d for d in date]
    elif isinstance(date, dt.date):
        return
    else:
        raise ValueError("date not supported...")


def previous_bday(date: Union[dt.date, Tuple[dt.date, ...]], n: int = 1):
    # TODOS: to update the next bday
    if isinstance(date, Tuple[dt.date, ...]):
        next_date = []
    elif isinstance(date, dt.date):
        return
    else:
        raise ValueError("date not supported...")


def count_bday(
    begin: Union[pd.Timestamp, dt.date],
    end: Union[pd.Timestamp, dt.date],
    calendars: Union[str, Tuple[str, ...]],
    week_mask: Optional[str],
) -> Union[int, Tuple[int]]:
    calendar = TradingCalendar.get(calendars)
    number_of_days = np.busday_count(
        begin, end, busdaycal=calendar.business_day_calendar(week_mask)
    )
    return (
        tuple(number_of_days)
        if isinstance(number_of_days, np.ndarray)
        else number_of_days
    )


def date_range(
    begin: Union[int, dt.date],
    end: Union[int, dt.date],
    calendars: Union[str, Tuple[str, ...]] = (),
    week_mask: Optional[str] = None,
):
    if isinstance(begin, dt.date):
        if isinstance(end, dt.date):
            return  # todo: implement
        elif isinstance(end, int):
            return (
                business_day_offset(begin, i, calendars=calendars, week_mask=week_mask)
                for i in range(end)
            )
        else:
            raise ValueError("end must be a date or int")
    elif isinstance(begin, int):
        if isinstance(end, dt.date):
            return (
                business_day_offset(
                    end, -i, roll="preceding", calendars=calendars, week_mask=week_mask
                )
                for i in range(begin)
            )
        else:
            raise ValueError("end must be a date if begin is an int")
    else:
        raise ValueError("begin must be a date or int")


def is_leap_year(date: Union[dt.date, str]):
    if isinstance(date, str):
        date = pd.to_datetime(date)
    return calendar.isleap(date.year)


def has_leap_day(begin: Union[int, dt.date], end: Union[int, dt.date]):
    return


# %% timezone conversion
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


# %% check if is holiday/trading day
"""
def is_holiday(date, cnt):
    cnt_code = country.get_country_attr(cnt, "alpha_2")
    if not isinstance(date, pd.Timestamp):
        date = pd.to_datetime(date)
    date_str = date.date().strftime("%Y-%m-%d")
    return date_str in holidays.country_holidays(cnt_code)
"""


# TODO: check if we can combine is_trading_day with is_businessday
def is_trading_day(date, exch):
    exch_cal = mcal.get_calendar(exch)
    if not isinstance(date, pd.Timestamp):
        date = pd.to_datetime(date)
    # date_str = date.date().strftime("%Y-%m-%d")
    return date in exch_cal.holidays().holidays


# %% get date str
def get_text_date(date: dt, lang: str) -> str:
    if lang == "FR":
        calendar = [
            "Janvier",
            "Février",
            "Mars",
            "Avril",
            "Mai",
            "Juin",
            "Juillet",
            "Aout",
            "Septembre",
            "Octobre",
            "Novembre",
            "Décembre",
        ]
    elif lang == "EN":
        calendar = [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "Septembrer",
            "Octobrer",
            "Novembre",
            "Decembrer",
        ]

    Month = calendar[date.month - 1]

    return str(date.day) + " " + Month.upper() + " " + str(date.year)


def infer_trading_calendar(factor_idx, prices_idx):
    """
    Infer the trading calendar from factor and price information.

    Parameters
    ----------
    factor_idx : pd.DatetimeIndex
        The factor datetimes for which we are computing the forward returns
    prices_idx : pd.DatetimeIndex
        The prices datetimes associated withthe factor data

    Returns
    -------
    calendar : pd.DateOffset
    """
    full_idx = factor_idx.union(prices_idx)
    traded_weekdays = []
    holidays = []

    days_of_the_week = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    for day, day_str in enumerate(days_of_the_week):
        weekday_mask = full_idx.dayofweek == day

        # drop days of the week that are not traded at all
        if not weekday_mask.any():
            continue
        traded_weekdays.append(day_str)

        # look for holidays
        used_weekdays = full_idx[weekday_mask].normalize()
        all_weekdays = pd.date_range(
            full_idx.min(), full_idx.max(), freq=CustomBusinessDay(weekmask=day_str)
        ).normalize()
        _holidays = all_weekdays.difference(used_weekdays)
        _holidays = [timestamp.date() for timestamp in _holidays]
        holidays.extend(_holidays)

    traded_weekdays = " ".join(traded_weekdays)
    return CustomBusinessDay(weekmask=traded_weekdays, holidays=holidays)


def timedelta_to_string(timedelta):
    """
    Utility that converts a pandas.Timedelta to a string representation
    compatible with pandas.Timedelta constructor format

    Parameters
    ----------
    timedelta: pd.Timedelta

    Returns
    -------
    string
        string representation of 'timedelta'
    """
    c = timedelta.components
    format = ""
    if c.days != 0:
        format += "%dD" % c.days
    if c.hours > 0:
        format += "%dh" % c.hours
    if c.minutes > 0:
        format += "%dm" % c.minutes
    if c.seconds > 0:
        format += "%ds" % c.seconds
    if c.milliseconds > 0:
        format += "%dms" % c.milliseconds
    if c.microseconds > 0:
        format += "%dus" % c.microseconds
    if c.nanoseconds > 0:
        format += "%dns" % c.nanoseconds
    return format


def timedelta_strings_to_integers(sequence):
    """
    Converts pandas string representations of timedeltas into integers of days.

    Parameters
    ----------
    sequence : iterable
        List or array of timedelta string representations, e.g. ['1D', '5D'].

    Returns
    -------
    sequence : list
        Integer days corresponding to the input sequence, e.g. [1, 5].
    """
    return list(map(lambda x: pd.Timedelta(x).days, sequence))


def add_custom_calendar_timedelta(input, timedelta, freq):
    """
    Add timedelta to 'input' taking into consideration custom frequency, which
    is used to deal with custom calendars, such as a trading calendar
    """
    if not isinstance(freq, (Day, BusinessDay, CustomBusinessDay)):
        raise ValueError("freq must be Day, BDay or CustomBusinessDay")
    days = timedelta.components.days
    offset = timedelta - pd.Timedelta(days=days)
    return input + freq * days + offset


def diff_custom_calendar_timedeltas(start, end, freq, date_freq="D"):
    """
    Compute the difference between two pd.Timedelta taking into consideration
    custom frequency, which is used to deal with custom calendars, such as a
    trading calendar
    """
    if not isinstance(freq, (Day, BusinessDay, CustomBusinessDay)):
        raise ValueError("freq must be Day, BusinessDay or CustomBusinessDay")

    weekmask = getattr(freq, "weekmask", None)
    holidays = getattr(freq, "holidays", None)

    if weekmask is None and holidays is None:
        if isinstance(freq, Day):
            weekmask = "Mon Tue Wed Thu Fri Sat Sun"
            holidays = []
        elif isinstance(freq, BusinessDay):
            weekmask = "Mon Tue Wed Thu Fri"
            holidays = []

    if weekmask is not None and holidays is not None:
        # we prefer this method as it is faster
        actual_periods = np.busday_count(
            np.array(start).astype("datetime64[D]"),
            np.array(end).astype("datetime64[D]"),
            weekmask,
            holidays,
        )
    else:
        # default, it is slow
        actual_periods = pd.date_range(start, end, freq=freq).shape[0] - 1
        if not freq.onOffset(start):
            actual_days -= 1

    timediff = end - start
    if date_freq == "D":
        delta_days = timediff.components.days - actual_periods
    elif date_freq == "W":
        delta_days = timediff.components.days - actual_periods * 7
    elif date_freq == "M":
        delta_days = timediff.components.days - actual_periods * 30
    elif date_freq == "Q":
        delta_days = timediff.components.days - actual_periods * 91
    elif date_freq == "Y":
        delta_days = timediff.components.days - actual_periods * 365
    return timediff - pd.Timedelta(days=delta_days)


def read_with_datetime(df, date_column="Date"):
    """
    Convert a column in a DataFrame to datetime and set it as the index.
    """
    df[date_column] = pd.to_datetime(df[date_column])
    df.set_index(date_column, inplace=True)
    return df


def filter_by_date_range(df, start_date, end_date):
    """
    Filter the DataFrame between start_date and end_date.
    """
    return df[start_date:end_date]


def calculate_time_delta(df, periods=1):
    """
    Calculate the difference in days between rows.
    """
    df["TimeDelta"] = df.index.to_series().diff(periods).dt.days
    return df


def extract_datetime_components(df):
    """
    Extract day, month, year, and weekday from the datetime index.
    """
    df_cols = df.columns.to_list()
    df["Year"] = df.index.year
    df["Month"] = df.index.month
    df["Week"] = df.index.isocalendar().week
    df["Weekday"] = df.index.weekday  # 0: Monday, 1: Tuesday, ...
    df["Day"] = df.index.day
    return df[["Year", "Month", "Week", "Weekday", "Day"] + df_cols]


"""
def infer_interval(df:pd.DataFrame) -> (pd.Timedelta, float):
        occurrences_dict = {}
        top_count = 0
        top_frequent_deltas = []

        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
            
        dates = df.index

        if len(dates) <= 1:
            raise ValueError("Index is too short. It must contain at least 2 values for automatic frequency setting.")

        time_deltas = [dates[i] - dates[i - 1] for i in range(1, len(dates))]

        for item in time_deltas:
            item_count = occurrences_dict.get(item, 0) + 1
            occurrences_dict[item] = item_count
            if item_count == top_count:
                top_frequent_deltas.append(item)
            elif item_count > top_count:
                top_frequent_deltas = [item]
                top_count = item_count

        relative_frequency = top_count / len(time_deltas)

        # if there is more than one delta of top frequency then combine them by calculating the mean
        # "top frequency delta" and assigning the combined_relative_frequency as the combined number of occurrences
        # of all "top frequency deltas".
        top_frequent_delta = pd.Series(data=top_frequent_deltas).mean()
        combined_relative_frequency = len(top_frequent_deltas) * relative_frequency

        return top_frequent_delta, combined_relative_frequency

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
    data.index = date_func.get_next_weekday(
        data.index.to_series(), reference_dates.max()
    )
    ## TODO
    data.columns = [
        get_ticker_attr(tickers, h, attr)
        for h in data.columns.get_level_values(header).astype(str)
    ]
    return data


def count_time_series_duration(data: Union[pd.DataFrame, pd.Series], params):
    data_duration = []
    for c in data.columns:
        actSignal = []
        nonzeroInd = find_nonzero_runs(data[c].values)
        for period in nonzeroInd:
            # get the starting and ending date of each period
            if period[1] != signalDf.shape[0]:
                pDate = pd.DataFrame(
                    [
                        str(signalDf.index[period[0]].date()),
                        str(signalDf.index[period[1] - 1].date()),
                        period[1] - period[0]
                    ],
                    index=['Start Date', 'End Date', 'Duration']).T
            else:
                pDate = pd.DataFrame(
                    [
                        str(signalDf.index[period[0]].date()), 'Ongoing',
                        signalDf.shape[0] - period[0] - 1
                    ],
                    index=['Start Date', 'End Date', 'Duration']).T
            actSignal.append(pDate)
        if len(actSignal) == 0:
            actSigDf = pd.DataFrame([np.nan, np.nan, np.nan]).T
            actSigDf.columns = ['Start Date', 'End Date', 'Duration']
        else:
            actSigDf = pd.concat(actSignal, axis=0)
        actSigDf.columns = [c + '-' + col for col in actSigDf.columns]
        actSigDf = actSigDf.reset_index()
        actSigDf = actSigDf.drop('index', 1)
        signalPeriods.append(actSigDf)
    signalDurationDf = pd.concat(signalPeriods, axis=1)
    sigStats = signalDurationDf.loc[:,
                                    signalDurationDf.columns.str.
                                    contains('Duration')]
    sigStats.columns = [c.replace('-Duration', '') for c in sigStats.columns]
    sigMean = pd.DataFrame(sigStats.mean(axis=0),
                           columns={params['performance']['expCode']}).T
    sigCnt = pd.DataFrame(sigStats.count(axis=0),
                          columns={params['performance']['expCode']}).T
    sigMedian = pd.DataFrame(sigStats.median(axis=0),
                             columns={params['performance']['expCode']}).T
    sigMax = pd.DataFrame(sigStats.max(axis=0),
                          columns={params['performance']['expCode']}).T
    sigMin = pd.DataFrame(sigStats.min(axis=0),
                          columns={params['performance']['expCode']}).T
    return {
        'Mean': sigMean,
        'Count': sigCnt,
        'Median': sigMedian,
        'Max': sigMax,
        'Min': sigMin
    }
"""
