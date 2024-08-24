'''
Author: J , jwsun1987@gmail.com
Date: 2023-11-21 20:15:00
LastEditors: J , jwsun1987@gmail.com
Description: 
Copyright: Copyright (c) 2024 by jwsun1987@gmail.com. All Rights Reserved.
'''


import pandas as pd
from datetime import timedelta
from datetime import datetime as dt
from typing import Union, List, Dict
import logging

import statsmodels.tsa.seasonal as seasonal
from statsmodels.tsa.x13 import x13_arima_analysis as x13

from data import Data
from utils.data import *
from utils import dates as date_func


def convert_to_timeseries(data: Union[pd.DataFrame, pd.Series]):
    if isinstance(data, pd.Series):
        data = pd.DataFrame(data)
    if not isinstance(data.index, pd.DatetimeIndex):
        try:
            data.index = pd.to_datetime(data.index, infer_datetime_format=True)
        except Exception as e:
            raise ValueError(e)
    return data


def synchronize_data(x: Union[pd.DataFrame, pd.Series], y: Union[pd.DataFrame, pd.Series], method: str=None):
    """synchronize dates of two series or dataframes

    Args:
        x (Union[pd.DataFrame, pd.Series]): first timeseries
        y (Union[pd.DataFrame, pd.Series]): second timeseries
        method (str, optional): _description_. Defaults to None.
    """
    return

def adjust_returns(returns, factor):
    if isinstance(factor, (float, int)) and factor == 0:
        return returns
    return returns - factor

def annualization_factor(period, annualization):
    return 


###################################################################
### STATS FUNCTIONS
###################################################################


def get_sum(data: Union[pd.DataFrame, pd.Series], frequency: str = "W"):
    data = convert_to_timeseries(data)
    return data.groupby(pd.Grouper(freq=frequency)).sum()


def get_mean(data: Union[pd.DataFrame, pd.Series], frequency: str = "W"):
    data = convert_to_timeseries(data)
    return data.groupby(pd.Grouper(freq=frequency)).mean()


def get_std(data: Union[pd.DataFrame, pd.Series], frequency: str = "W"):
    data = convert_to_timeseries(data)
    return data.groupby(pd.Grouper(freq=frequency)).std()


def get_median(data: Union[pd.DataFrame, pd.Series], frequency: str = "W"):
    data = convert_to_timeseries(data)
    return data.groupby(pd.Grouper(freq=frequency)).median()


def get_data_range(data: pd.DataFrame, frequency: str = "W"):
    if frequency.split("-")[0] == "W":
        data_range = pd.DataFrame(data.index.week, data.index)
    elif frequency.split("-")[0] == "M":
        data_range = pd.DataFrame(data.index.month, data.index)
    elif frequency.split("-")[0] == "Q":
        data_range = pd.DataFrame(((data.index.month - 1) // 3 + 1), data.index)
    else:
        raise ValueError("no need to re-index for %s frequency" % frequency)
    data_range.columns = ["Freq"]
    return data_range


def get_pre_covid_stats(data: Union[pd.DataFrame, Data], freq: str = "M"):
    pre_covid = data.loc[(data.index >= "2017-01-01") & (data.index < "2020-01-01")]
    post_covid = data.loc[data.index >= "2020-01-01"]
    if pre_covid.empty:
        return

    # get average of pre-Covid years
    pre_covid_freq = get_mean(pre_covid, freq)

    if freq.split("-")[0] == "W":
        pre_covid_freq_avg = pre_covid_freq.groupby(pre_covid_freq.index.week).mean()
    elif freq.split("-")[0] == "M":
        pre_covid_freq_avg = pre_covid_freq.groupby(pre_covid_freq.index.month).mean()
    elif freq.split("-")[0] == "Q":
        pre_covid_freq_avg = pre_covid_freq.groupby(
            (pre_covid_freq.index.month - 1) // 3 + 1
        ).mean()
    else:
        raise ValueError("%s based stats not avaialble" % freq)
    pre_covid_freq_avg.index.names = ["Freq"]

    post_covid_weekly = get_mean(post_covid, "W")
    post_covid_range = get_data_range(post_covid_weekly, freq)
    post_covid_pct = post_covid_weekly.div(
        post_covid_range.merge(pre_covid_freq_avg.reset_index(), how="left")
        .set_index(post_covid_range.index)
        .drop("Freq", axis=1)
    )
    return post_covid_pct


def yoy(data: Union[pd.DataFrame, pd.Series]):
    data = convert_to_timeseries(data)
    freq = date_func.get_date_frequency(data.index)
    window = date_func.get_duration_number(freq, 1, "Y")
    return data.pct_change(window)


def mom(data: Union[pd.DataFrame, pd.Series]):
    data = convert_to_timeseries(data)
    freq = date_func.get_date_frequency(data.index)
    if freq.split("-")[0] == "W" or freq.split("-")[0] == "D":
        return data.groupby(pd.Grouper(freq="M")).mean().pct_change()
    else:
        return data.pct_change()


def wow(data: Union[pd.DataFrame, pd.Series]):
    data = convert_to_timeseries(data)
    freq = date_func.get_date_frequency(data.index)
    if freq.split("-")[0] == "D":
        return data.groupby(pd.Grouper(freq="W")).mean().pct_change()
    else:
        return data.pct_change()


def wtd(data: Union[pd.DataFrame, pd.Series]):
    return


def mtd(data: Union[pd.DataFrame, pd.Series]):
    return


def ytd(data: Union[pd.DataFrame, pd.Series]):
    return


def recent(data: Union[pd.DataFrame, pd.Series], duration: int = 1, unit: str = "W"):
    return


def remove_seasonality(data, CNYFlag=False):
    dataDf = pd.DataFrame(data.groupby(pd.Grouper(freq="M")).mean())
    dataDf = dataDf[dataDf.index <= ((dt.today() - timedelta(days=7)))]
    dataSA = pd.DataFrame()
    for col in dataDf.columns:
        if sum(dataDf[[col]].values) != 0:
            try:
                x13SA = x13(
                    pd.DataFrame(dataDf[[col]]), x12path=r"E:\software\x13as\x13as.exe"
                )
                colSA = pd.DataFrame(x13SA.seasadj)
            except Exception as e:
                logging.error(e)
                colSA = pd.DataFrame(dataDf[[col]])
                continue
        else:
            colSA = pd.DataFrame(dataDf[[col]])
        colSA.columns = [col]
        dataSA = pd.concat([dataSA, colSA], axis=1)
    return dataSA


###################################################################
### RETURNS FUNCTIONS
###################################################################
