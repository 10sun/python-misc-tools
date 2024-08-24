'''
Author: J , jwsun1987@gmail.com
Date: 2023-11-17 23:36:54
LastEditors: J , jwsun1987@gmail.com
Description: 
Copyright: Copyright (c) 2024 by jwsun1987@gmail.com. All Rights Reserved.
'''

import pandas as pd

import numpy as np
import logging

from common.data import find_nonzero_runs

def getAnnVol(data_list, mean=True):
    data_stats = []
    for data in data_list:
        data_stats.append(pd.DataFrame(data.std(axis=0)*np.sqrt(12)).T)
    data_stats = pd.concat(data_stats, axis=0)
    if mean:
        return data_stats.mean(axis=0)
    else:
        return data_stats

def getIR(data_list, mean=True):
    data_stats = []
    for data in data_list:
        data_mean = data.mean(axis=0)
        data_vol = data.std(axis=0)*np.sqrt(12)
        data_stats.append(pd.DataFrame(data_mean/data_vol).T)
    data_stats = pd.concat(data_stats, axis=0)
    if mean:
        return data_stats.mean(axis=0)
    else:
        return data_stats

def getAvgMonthlyRet(data_list, mean=True):
    data_stats = []
    for data in data_list:
        data_stats.append(pd.DataFrame(data.mean(axis=0).rename(data.index.max())).T)
    data_stats = pd.concat(data_stats, axis=0)
    #display(data_stats)
    if mean:
        return data_stats.mean(axis=0)
    else:
        return data_stats

def getCount(data_list, mean=False):
    data_stats = []
    for data in data_list:
        data_stats.append(pd.DataFrame(data.count(axis=0).rename(data.index.max())).T)
    data_stats = pd.concat(data_stats, axis=0)
    #display(data_stats)
    if mean:
        return data_stats.mean(axis=0)
    else:
        return data_stats.sum(axis=0)

def get_regime_data(dataDf: pd.DataFrame, regimeDf: pd.DataFrame,
                    regimes: dict, regime_index: str, last: bool=False) -> dict:
    """[summary]

    Args:
        dataDf (pd.DataFrame): [description]
        regimeDf (pd.DataFrame): [description]
        regime_dict (dict): [description]

    Returns:
        dict: [description]
    """
    regime_data = {}
    for regime_key, regime in regimes.items():
        logging.debug(regime + ': ' + str(regime_key))
        # get the indexes of each period within each regime
        periods = find_nonzero_runs(
            (regimeDf.Regime == regime_key).astype(int).values)
        if len(periods) == 0:
            continue
        # get the data for each period
        period_data = []
        for period in periods:
            # get period data
            if period[0] <= dataDf.shape[0]:
                periodDf = dataDf[period[0]:period[1]]
                if periodDf.index.max() == regimeDf.index.max(
                ) and regime_index != 'Decade' and not last:
                    continue
                period_data.append(periodDf)
            else:
                logging.error('period out of index...')
                continue
        regime_data.update({regime: period_data})
    return regime_data