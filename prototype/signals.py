'''
Author: J , jwsun1987@gmail.com
Date: 2024-04-23 19:29:30
LastEditors: J , jwsun1987@gmail.com
Description: 
Copyright: Copyright (c) 2024 by jwsun1987@gmail.com. All Rights Reserved.
'''


import numpy as np
from numpy.lib.function_base import quantile
import pandas as pd
import logging
from datetime import datetime as dt

from pathlib import Path

import preprocess as prep
import dataIO
import copy


def configure_params(assets, params=None) -> dict:
    """
	Prepare the signal data for backtesting/trading

	Parameters
	-------

	Returns
	-------
	params: dictionary for data parameters
	"""
    if not params:
        params = {
            'operation': 'mean',
            'signalWindow': 20,
            'historyWindow': 261,
            'emdWindow': 5,
            'source': 'org',
            'signal': 'simple',
            'saveFig': False
        }
    if 'operation' not in params.keys():
        params.update({'operation': 'mean'})
    if 'signalWindow' not in params.keys():
        params.update({'signalWindow': 20})
    if 'historyWindow' not in params.keys():
        params.update({'historyWindow': 261})
    if 'emdWindow' not in params.keys():
        params.update({'emdWindow': 261 * 5})
    if 'source' not in params.keys():
        params.update({'source': 'org'})
    if 'signal' not in params.keys():
        params.update({'signal': 'simple'})
    if 'saveFig' not in params.keys():
        params.update({'saveFig': False})

    if 'ranking' in params['signal']:
        if 'quantile' not in params.keys():
            params.update({'quantile': 5})
        if 'topN' not in params.keys():
            if int(np.floor(len(assets) / params['quantile'])) == 0:
                params.update({'topN': 1})
            else:
                params.update(
                    {'topN': int(np.floor(len(assets) / params['quantile']))})
    elif 'simple' in params['signal']:
        if params['source'] == 'ols':
            if 'threshold' not in params.keys():
                params.update({'threshold': {'res': 0, 'r2': 0.6, 't': 2}})
            if 'res' not in params['threshold'].keys():
                params['threshold'].update({'res': 0})
            if 'r2' not in params['threshold'].keys():
                params['threshold'].update({'r2': 0.6})
            if 't' not in params['threshold'].keys():
                params['threshold'].update({'t': 2})
        else:
            if 'threshold' not in params.keys():
                params.update({'threshold': 0})
    return params


def aggregate_signal(dataDf, params):
    tradingSignalRaw = []
    if params['performance']['tearsheet']:
        singleSig = {}
    for sigWindow in params['aggSignal']['signalWindow']:
        if 'signalWindow' not in params['data'].keys():
            params['data'].update({'signalWindow': sigWindow})
        else:
            params['data']['signalWindow'] = sigWindow
        for sigSource in params['aggSignal']['source']:
            if 'source' not in params['data'].keys():
                params['data'].update({'source': sigSource})
            else:
                params['data']['source'] = sigSource

            if 'threshold' not in params['data'].keys():
                if params['data']['source'] == 'ols':
                    params['data'].update(
                        {'threshold': {
                            'res': 0,
                            'r2': 0.6,
                            't': 2
                        }})
                else:
                    params['data']['threshold'] = 0
            else:
                params['data']['threshold'] = {
                    'res': 0,
                    'r2': 0.6,
                    't': 2
                } if params['data']['source'] == 'ols' else 0
            params['data'] = configure_params(params['portfolio']['assets'],
                                              params['data'])
            logging.info(params['data']['source'] + ': ' +
                         str(params['data']['signalWindow']))
            signal = simple(dataDf, params)
            tradingSignalRaw.append(signal)
            if params['performance']['tearsheet']:
                singleSig.update({
                    params['data']['source'] + '-' + str(params['data']['signalWindow']):
                    signal
                })

    momData = pd.concat(tradingSignalRaw).groupby(level=0).sum()
    tradingSignal = pd.DataFrame(0,
                                 index=momData.index,
                                 columns=momData.columns)
    if params['trading']['strategy'] == 'lo':
        tradingSignal[momData >= params['aggSignal']['threshold']] = 1
    elif params['trading']['strategy'] == 'so':
        tradingSignal[momData <= -params['aggSignal']['threshold']] = -1
    elif params['trading']['strategy'] == 'ls':
        tradingSignal[momData >= params['aggSignal']['threshold']] = 1
        tradingSignal[momData <= -params['aggSignal']['threshold']] = -1
    if params['performance']['tearsheet']:
        singleSig.update({'momData': momData})
        singleSig.update({'momSignal': tradingSignal})
        dataIO.export_to_excel(
            singleSig,
            Path(params['performance']['dir']) /
            (params['performance']['expCode'] + '_signal.xlsx'))
    return tradingSignal  #, momData, tradingSignalRaw


def differentiate_signal(dataDf, params):
    tradingSignalRaw = {}
    if params['performance']['tearsheet']:
        singleSig = {}
    for sigWindow in params['aggSignal']['signalWindow']:
        signalRaw = []
        if 'signalWindow' not in params['data'].keys():
            params['data'].update({'signalWindow': sigWindow})
        else:
            params['data']['signalWindow'] = sigWindow
        for sigSource in params['aggSignal']['source']:
            if 'source' not in params['data'].keys():
                params['data'].update({'source': sigSource})
            else:
                params['data']['source'] = sigSource

            if 'threshold' not in params['data'].keys():
                if params['data']['source'] == 'ols':
                    params['data'].update(
                        {'threshold': {
                            'res': 0,
                            'r2': 0.6,
                            't': 2
                        }})
                else:
                    params['data']['threshold'] = 0
            else:
                params['data']['threshold'] = {
                    'res': 0,
                    'r2': 0.6,
                    't': 2
                } if params['data']['source'] == 'ols' else 0
            params['data'] = configure_params(params['portfolio']['assets'],
                                              params['data'])
            logging.info(params['data']['source'] + ': ' +
                         str(params['data']['signalWindow']))
            signal = simple(dataDf, params)
            signalRaw.append(signal)
            signalDf = pd.concat(signalRaw).groupby(level=0).sum()
            if params['performance']['tearsheet']:
                singleSig.update({
                    params['data']['source'] + '-' + str(params['data']['signalWindow']):
                    signal
                })
        tradingSignalRaw.update({sigWindow: signalDf})
    momData = tradingSignalRaw[min(
        params['aggSignal']['signalWindow'])] - tradingSignalRaw[max(
            params['aggSignal']['signalWindow'])]

    tradingSignal = pd.DataFrame(0,
                                 index=momData.index,
                                 columns=momData.columns)
    if params['trading']['strategy'] == 'lo':
        tradingSignal[momData >= params['aggSignal']['threshold']] = 1
    elif params['trading']['strategy'] == 'so':
        tradingSignal[momData <= -params['aggSignal']['threshold']] = -1
    elif params['trading']['strategy'] == 'ls':
        tradingSignal[momData >= params['aggSignal']['threshold']] = 1
        tradingSignal[momData <= -params['aggSignal']['threshold']] = -1
    if params['performance']['tearsheet']:
        singleSig.update({'momData': momData})
        singleSig.update({'momSignal': tradingSignal})
        dataIO.export_to_excel(
            singleSig,
            Path(params['performance']['dir']) /
            (params['performance']['expCode'] + '_signal.xlsx'))
    return tradingSignal


def multi_lvl_ranking(dataDf: pd.DataFrame, params: dict) -> dict:
    """[summary]

    Args:
        dataDf (pd.DataFrame): [description]
        params (dict): [description]

    Returns:
        dict: [description]
    """
    if not params['data'].get('multi_lvl'):
        raise ValueError('no level info provided...')

    high_lvl_data = dataDf.loc[:,
                               list(params['data']['multi_lvl']['high'].keys()
                                    )]
    low_lvl_data = dict((asset_class, dataDf.loc[:, asset_list])
                        for asset_class, asset_list in params['data']
                        ['multi_lvl']['low'].items())

    high_lvl_params = copy.deepcopy(params)
    high_lvl_params['portfolio']['weights'] = pd.DataFrame(
        list(params['data']['multi_lvl']['high'].values())).T
    high_lvl_params['portfolio']['weights'].columns = list(
        params['data']['multi_lvl']['high'].keys())
    high_lvl_params['portfolio']['weights'].index.names = ['Weight']
    high_lvl_params['data']['topN'] = 2
    high_lvl_signal = ranking(high_lvl_data, high_lvl_params)
    low_lvl_params = copy.deepcopy(params)
    low_lvl_params['data']['operation'] = 'return'
    low_lvl_params['data']['topN'] = 2
    low_lvl_signal = dict((asset_class, ranking(asset_data, low_lvl_params))
                          for asset_class, asset_data in low_lvl_data.items())
    return {'high': high_lvl_signal, **low_lvl_signal}


def ranking(dataDf, params):
    """
	get trading signal based on ranking

	Parameters
	-------
	dataDf: raw data for trading signal
	params: dictionary for data parameters
	
	Returns
	-------
	tradingSignal: trading signals
	"""
    rankingDf = prep.preprocess_data(dataDf, params).dropna(how='all', axis=0)[params['signal']['assets']]
    
    if params['data']['source'] == 'ols':
        tradingSignal = pd.DataFrame(0,
                                     index=rankingDf['res'].index,
                                     columns=rankingDf['res'].columns)
    else:
        tradingSignal = pd.DataFrame(0,
                                     index=rankingDf.index,
                                     columns=rankingDf.columns)

    for date, row in rankingDf.iterrows():
        topInd = list(row.nlargest(params['data']['topN']).index)
        bottomInd = list(row.nsmallest(params['data']['topN']).index)
        if params['trading']['strategy'] == 'lo':
            tradingSignal.loc[tradingSignal.index == date, topInd] = 1
        elif params['trading']['strategy'] == 'so':
            tradingSignal.loc[tradingSignal.index == date, bottomInd] = -1
        elif params['trading']['strategy'] == 'ls':
            tradingSignal.loc[tradingSignal.index == date, topInd] = 1
            tradingSignal.loc[tradingSignal.index == date, bottomInd] = -1
    return tradingSignal


def ranking_reversion(dataDf, params):
    """
	get trading signal based on ranking

	Parameters
	-------
	dataDf: raw data for trading signal
	params: dictionary for data parameters
	
	Returns
	-------
	tradingSignal: trading signals
	"""
    rankingDf = prep.preprocess_data(dataDf, params).dropna(how='all', axis=0)

    if params['data']['source'] == 'ols':
        tradingSignal = pd.DataFrame(0,
                                     index=rankingDf['res'].index,
                                     columns=rankingDf['res'].columns)
    else:
        tradingSignal = pd.DataFrame(0,
                                     index=rankingDf.index,
                                     columns=rankingDf.columns)

    for date, row in rankingDf.iterrows():
        topInd = list(row.nlargest(params['data']['topN']).index)
        bottomInd = list(row.nsmallest(params['data']['topN']).index)
        if params['trading']['strategy'] == 'lo':
            tradingSignal.loc[tradingSignal.index == date, bottomInd] = 1
        elif params['trading']['strategy'] == 'so':
            tradingSignal.loc[tradingSignal.index == date, topInd] = -1
        elif params['trading']['strategy'] == 'ls':
            tradingSignal.loc[tradingSignal.index == date, bottomInd] = 1
            tradingSignal.loc[tradingSignal.index == date, topInd] = -1
    return tradingSignal


def simple(dataDf, params):
    """
	get trading signal based on simple threshold

	Parameters
	-------
	dataDf: raw data for trading signal
	params: dictionary for data parameters
	
	Returns
	-------
	tradingSignal: trading signals
	"""
    momData = prep.preprocess_data(dataDf, params)

    if params['data']['source'] == 'ols':
        tradingSignal = pd.DataFrame(0,
                                     index=momData['res'].index,
                                     columns=momData['res'].columns)
    else:
        tradingSignal = pd.DataFrame(0,
                                     index=momData.index,
                                     columns=momData.columns)
    #display(momData)
    if params['trading']['strategy'] == 'lo':
        if params['data']['source'] == 'ols':
            tradingSignal[(momData['res'] > params['data']['threshold']['res']) \
             & (momData['r2'] > params['data']['threshold']['r2']) \
              & (momData['t'] > params['data']['threshold']['t'])] = 1
        else:
            tradingSignal[momData > params['data']['threshold']] = 1
    elif params['trading']['strategy'] == 'so':
        if params['data']['source'] == 'ols':
            tradingSignal[(momData['res'] < -params['data']['threshold']['res']) \
             & (momData['r2'] > params['data']['threshold']['r2']) \
              & (momData['t'] < -params['data']['threshold']['t'])] = -1
        else:
            tradingSignal[momData < -params['data']['threshold']] = -1
    elif params['trading']['strategy'] == 'ls':
        if params['data']['source'] == 'ols':
            tradingSignal[(momData['res'] > params['data']['threshold']['res']) \
             & (momData['r2'] > params['data']['threshold']['r2']) \
              & (momData['t'] > params['data']['threshold']['t'])] = 1
            tradingSignal[(momData['res'] < -params['data']['threshold']['res']) \
             & (momData['r2'] > params['data']['threshold']['r2']) \
              & (momData['t'] < -params['data']['threshold']['t'])] = -1
        else:
            tradingSignal[momData > params['data']['threshold']] = 1
            tradingSignal[momData < -params['data']['threshold']] = -1
    return tradingSignal


def simple_reversion(dataDf, params):
    """
	get trading signal based on simple threshold

	Parameters
	-------
	dataDf: raw data for trading signal
	params: dictionary for data parameters
	
	Returns
	-------
	tradingSignal: trading signals
	"""
    momData = prep.preprocess_data(dataDf, params)
    if params['data']['source'] == 'ols':
        tradingSignal = pd.DataFrame(0,
                                     index=momData['res'].index,
                                     columns=momData['res'].columns)
    else:
        tradingSignal = pd.DataFrame(0,
                                     index=momData.index,
                                     columns=momData.columns)

    if params['trading']['strategy'] == 'lo':
        if params['signal']['source'] == 'ols':
            tradingSignal[(momData['res'] < -params['data']['threshold']['res']) \
             & (momData['r2'] > params['data']['threshold']['r2']) \
              & (momData['t'] < -params['data']['threshold']['t'])] = 1
        else:
            tradingSignal[momData < -params['data']['threshold']] = 1
    elif params['trading']['strategy'] == 'so':
        if params['signal']['source'] == 'ols':
            tradingSignal[(momData['res'] > params['data']['threshold']['res']) \
             & (momData['r2'] > params['data']['threshold']['r2']) \
              & (momData['t'] > params['data']['threshold']['t'])] = -1
        else:
            tradingSignal[momData > params['data']['threshold']] = -1
    elif params['trading']['strategy'] == 'ls':
        if params['signal']['source'] == 'ols':
            tradingSignal[(momData['res'] < -params['data']['threshold']['res']) \
             & (momData['r2'] > params['data']['threshold']['r2']) \
              & (momData['t'] < -params['data']['threshold']['t'])] = 1
            tradingSignal[(momData['res'] > params['data']['threshold']['res']) \
             & (momData['r2'] > params['data']['threshold']['r2']) \
              & (momData['t'] > params['data']['threshold']['t'])] = -1
        else:
            tradingSignal[momData < -params['data']['threshold']] = 1
            tradingSignal[momData > params['data']['threshold']] = -1

    return tradingSignal


def macd(dataDf, params):
    return

def get_quantile_signal(dataDf, params):
    """
	get trading signal based on simple threshold

	Parameters
	-------
	dataDf: raw data for trading signal
	params: dictionary for data parameters
	
	Returns
	-------
	tradingSignal: trading signals
	"""
    dataQuantile = dataDf.apply(
        lambda row: pd.qcut(row, params['data']['quantile'], labels=False),
        axis=1)
    quantileSig = {}
    for q in range(params['date']['quantile']):
        quantileSig.update({str(q): (dataQuantile == q).astype(int)})
    return quantileSig, dataQuantile


def olsMoM(dataDf, params):
    """
	get trading signal based on simple threshold

	Parameters
	-------
	dataDf: raw data for trading signal
	params: dictionary for data parameters
	
	Returns
	-------
	tradingSignal: trading signals
	"""
    emaData = prep.preprocess_data(dataDf, params).dropna(how='all', axis=0)

    tradingSignal = pd.DataFrame(0, index=olsRes.index, columns=olsRes.columns)

    tradingSignal[(olsRes > params['data']['threshold']['res']) \
     & (olsR2 > params['data']['threshold']['r2']) \
      & (olsT > params['data']['threshold']['t'])] = 1
    if params['trading']['strategy'] == 'ls':
        tradingSignal[(olsRes < params['data']['threshold']['res']) \
         & (olsR2 > params['data']['threshold']['r2']) \
          & (olsT < -params['data']['threshold']['t'])] = -1
    return tradingSignal
