'''
Author: J , jwsun1987@gmail.com
Date: 2021-11-25 23:34:06
LastEditors: J , jwsun1987@gmail.com
Description: 
Copyright: Copyright (c) 2024 by jwsun1987@gmail.com. All Rights Reserved.
'''

import pandas as pd
import numpy as np

def find_nonzero_runs(data):
    isnonzero = np.concatenate(
        ([0], (np.asarray(data) != 0).view(np.int8), [0]))

    absdiff = np.abs(np.diff(isnonzero))
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    return ranges

def get_signal_duration(signalDf, params):
    signalPeriods = []
    for c in signalDf.columns:
        actSignal = []
        nonzeroInd = find_nonzero_runs(signalDf[c].values)
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