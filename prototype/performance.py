'''
Author: J , jwsun1987@gmail.com
Date: 2022-11-11 01:47:58
LastEditors: J , jwsun1987@gmail.com
Description: 
Copyright: Copyright (c) 2024 by jwsun1987@gmail.com. All Rights Reserved.
'''


import pandas as pd
import scipy.stats as stats
import numpy as np

from datetime import datetime as dt
from datetime import timedelta
import logging
from pathlib import Path
import sys

sys.path.append('../')
import dataIO
try:
    import pyfolio as pf
    import empyrical as ep
except ImportError:
    sys.path.append(r'\\merlin\lib_dpm\code\libs\empyrical-master')
    sys.path.append(r'\\merlin\lib_dpm\code\libs\pyfolio-master')
    import empyrical as ep
    import pyfolio as pf

SIMPLE_STAT_FUNCS = [
    ep.annual_return, ep.cum_returns_final, ep.annual_volatility,
    ep.sharpe_ratio, ep.calmar_ratio, ep.stability_of_timeseries,
    ep.max_drawdown, ep.sortino_ratio, stats.skew, stats.kurtosis,
    ep.tail_ratio, pf.timeseries.value_at_risk
]

FACTOR_STAT_FUNCS = [
    ep.alpha,
    ep.beta,
]

STAT_FUNC_NAMES = {
    'annual_return': 'Annual return',
    'cum_returns_final': 'Cumulative returns',
    'annual_volatility': 'Annual volatility',
    'sharpe_ratio': 'Sharpe ratio',
    'calmar_ratio': 'Calmar ratio',
    'stability_of_timeseries': 'Stability',
    'max_drawdown': 'Max drawdown',
    'omega_ratio': 'Omega ratio',
    'sortino_ratio': 'Sortino ratio',
    'skew': 'Skew',
    'kurtosis': 'Kurtosis',
    'tail_ratio': 'Tail ratio',
    'common_sense_ratio': 'Common sense ratio',
    'value_at_risk': 'Daily value at risk',
    'alpha': 'Alpha',
    'beta': 'Beta',
}

RESULTS_SUMMARY = {
    'annual_return': 'Annual return',
    'annual_volatility': 'Annual volatility',
    'sharpe_ratio': 'Sharpe ratio',
    'max_drawdown': 'Max drawdown',
}

period_Func = [
    'annual_return', 'annual_volatility', 'sharpe_ratio', 'calmar_ratio',
    'sortino_ratio'
]


def perf_hit_ratio(
        returns: pd.DataFrame,
        benchmark=None) -> float:  #, tradingSignal, rankingFlag=True
    """[summary]

    Args:
        returns (pd.DataFrame): [description]
        benchmark ([type], optional): [description]. Defaults to None.

    Returns:
        float: [description]
    """
    if benchmark is not None:
        hit = ((pd.DataFrame(returns) - benchmark.values).dropna() >
               0).astype(int).sum().values[0]
    else:
        hit = (pd.DataFrame(returns).dropna() > 0).astype(int).sum().values[0]
    N = len(pd.DataFrame(returns).dropna())
    return hit / N


def ranking_hit_ratio(trading_signal: pd.DataFrame,
                      fwd_ranking: pd.DataFrame) -> float:
    """[summary]

    Args:
        trading_signal (pd.DataFrame): [description]
        fwd_ranking (pd.DataFrame): [description]

    Returns:
        float: [description]
    """
    hit, N = 0, 0
    for c in trading_signal.columns:
        curt_signal = trading_signal[c].loc[trading_signal[c] != 0]
        if not curt_signal.empty:
            fwd_signal = fwd_ranking[c].loc[fwd_ranking.index.intersection(
                trading_signal[c].loc[trading_signal[c] != 0].index)]
            hit += (curt_signal == fwd_signal).astype(int).sum()
            N += curt_signal.shape[0]
    return hit / N if N != 0 else 0


def get_performance_stats(returns: pd.Series,
                          returns_freq=None,
                          positions=None,
                          transactions=None,
                          factor_returns=None,
                          turnover_denom='AGB',
                          stat_functions=SIMPLE_STAT_FUNCS) -> pd.Series:
    """[summary]

    Args:
        returns (pd.Series): [description]
        returns_freq ([type], optional): [description]. Defaults to None.
        positions ([type], optional): [description]. Defaults to None.
        transactions ([type], optional): [description]. Defaults to None.
        factor_returns ([type], optional): [description]. Defaults to None.
        turnover_denom (str, optional): [description]. Defaults to 'AGB'.
        stat_functions ([type], optional): [description]. Defaults to SIMPLE_STAT_FUNCS.

    Returns:
        pd.Series: [description]
    """
    if not returns_freq:
        returns_freq = pd.infer_freq(returns.index)
    if 'D'.casefold() in returns_freq.casefold():
        returns_period = 'daily'
    elif 'W'.casefold() in returns_freq.casefold():
        returns_period = 'weekly'
    elif 'M'.casefold() in returns_freq.casefold():
        returns_period = 'monthly'
    elif 'Q'.casefold() in returns_freq.casefold():
        returns_period = 'quarterly'
    elif 'Y'.casefold() in returns_freq.casefold():
        returns_period = 'yearly'
    stats = pd.Series()
    for stat_func in stat_functions:
        if stat_func.__name__ in period_Func:
            stats[STAT_FUNC_NAMES[stat_func.__name__]] = stat_func(
                returns, period=returns_period)
        else:
            stats[STAT_FUNC_NAMES[stat_func.__name__]] = stat_func(returns)

    if positions is not None:
        stats['Gross leverage'] = pf.timeseries.gross_lev(positions).mean()
        if transactions is not None:
            stats['Daily turnover'] = pf.timeseries.get_turnover(
                positions, transactions, turnover_denom).mean()
    if factor_returns is not None:
        for stat_func in FACTOR_STAT_FUNCS:
            res = stat_func(returns, factor_returns)
            stats[STAT_FUNC_NAMES[stat_func.__name__]] = res

    stats['Hit Ratio'] = perf_hit_ratio(returns)
    return stats


def get_interesting_time_stats(returns: pd.Series) -> pd.DataFrame:
    """[summary]

    Args:
        returns (pd.Series): [description]

    Returns:
        pd.DataFrame: [description]
    """
    interesting_periods = pf.timeseries.extract_interesting_date_ranges(
        returns)
    interesting_dates = pd.DataFrame([[p, pDf.index.min(), pDf.index.max()] for p, pDf in interesting_periods.items()], \
     columns=['period', 'start date', 'end date']).set_index('period')
    interesting_stats = pd.concat([
        interesting_dates,
        pd.DataFrame(pf.timeseries.extract_interesting_date_ranges(
            returns)).describe().transpose()
    ],
                                  axis=1).reset_index()
    interesting_stats = interesting_stats.rename(columns={'index': 'period'})
    interesting_stats['date'] = interesting_stats['start date']
    interesting_stats = interesting_stats.set_index('date').sort_index()
    return interesting_stats


def get_positions_tableau(trading_position: pd.DataFrame) -> pd.DataFrame:
    """[summary]

    Args:
        trading_position (pd.DataFrame): [description]

    Returns:
        pd.DataFrame: [description]
    """
    position_list = []
    for index, row in trading_position.iterrows():
        rowDf = pd.DataFrame(
            row.loc[row != 0]).rename(columns={index: 'Value'})
        rowDf.index.names = ['Asset']
        rowDf['Position'] = rowDf['Value'].apply(lambda x: 'Long'
                                                 if x > 0 else 'Short')
        rowDf['date'] = index
        position_list.append(rowDf.reset_index().set_index('date'))
    return pd.concat(position_list, axis=0)


def get_perf_tearsheet(returns: pd.Series, trading_position: pd.DataFrame,
                       params: dict) -> dict:
    """[summary]

    Args:
        returns (pd.Series): [description]
        trading_position (pd.DataFrame): [description]
        params (dict): [description]

    Returns:
        dict: [description]
    """
    if returns.empty:
        logging.error('No return data...')
        return

    # dates of return
    return_date = pd.DataFrame(pd.to_datetime(returns.index.tolist()),
                               index=pd.to_datetime(returns.index),
                               columns=['date'])

    # overall statistics
    overall_stats = pd.DataFrame(
        get_performance_stats(returns,
                              returns_freq=params['trading']['frequency'])).T
    overall_stats.columns = ['Overall ' + c for c in overall_stats.columns]
    overall_stats.index = [returns.index.max()]
    overall_stats.index.names = ['date']
    overall_stats['YTD'] = (returns.loc[returns.index >= str(dt.today().year) +
                                        '-01-01'] + 1).cumprod() - 1
    overall_stats['MTD'] = (returns.loc[returns.index >= str(dt.today().year) +
                                        '-' + str(dt.today().month) + '-01'] +
                            1).cumprod() - 1
    overall_stats['Last One Year'] = (
        returns.loc[returns.index >= returns.index.max() - timedelta(days=262)]
        + 1).cumprod() - 1
    overall_stats['Last One Month'] = (
        returns.loc[returns.index >= returns.index.max() - timedelta(days=20)]
        + 1).cumprod() - 1

    # yearly return
    yearly_return = pd.DataFrame(ep.aggregate_returns(returns, 'yearly').values, \
     columns=['Yearly Return'], index= return_date.groupby(pd.Grouper(freq='Y')).last()['date'])

    # monthly return
    monthly_return = pd.DataFrame(ep.aggregate_returns(returns, 'monthly').values, \
     columns = ['Monthly Return'], index = return_date.groupby(pd.Grouper(freq='M')).last()['date'])

    # top 5 drawdown periods    
    drawdown_table = pd.DataFrame(
        pf.timeseries.gen_drawdown_table(returns, top=5).sort_values(
            'Net drawdown in %', ascending=False))
    drawdown_table['date'] = drawdown_table['Valley date']
    drawdown_table = drawdown_table.set_index('date')

    
    # get the performance in interesting times
    interesting_stats = get_interesting_time_stats(returns)

    historical_positions = get_positions_tableau(trading_position)

    # results to return
    results = {'overall_stats': overall_stats, \
     'yearly_return': yearly_return, \
     'monthly_return': monthly_return, \
     'return': pd.DataFrame(returns.values, columns=['return'], index=returns.index), \
     'cumulative_return': pd.DataFrame((1 + returns.values).cumprod()*100, columns=['cumulative_return'], index=returns.index), \
     'top_drawdowns': drawdown_table, \
     'interesting_time_stats': interesting_stats, \
     'trading_position':historical_positions}
    ## TODO : check the path
    if params['performance'].get('save', False):
        dataIO.export_to_excel(
            results,
            Path(params['performance']['dir'] /
                 (params['performance']['expCode'] + '_results.xlsx')))
    return results


def tearsheet_to_tableau(tearsheet: dict, pf_str: str,
                         params: dict) -> pd.DataFrame:
    """[summary]

    Args:
        tearsheet (dict): [description]
        pf_str (str): [description]
        params (dict): [description]

    Returns:
        pd.DataFrame: [description]
    """
    tearsheetDf = []
    for k, v in tearsheet.items():
        if k == 'top_drawdowns':
            v = v.rename(columns={'Net drawdown in %': 'Value'})
            v['KPI'] = k
            v['Portfolio'] = pf_str
        elif k == 'interesting_time_stats':
            vList = []
            for index, row in v.iterrows():
                rowDf = pd.DataFrame(row)
                rowDf.index.names = ['Metric']
                rowDf.columns = ['Value']
                rowDf = rowDf.loc[[
                    'count', 'mean', 'count', 'mean', 'std', 'min', '25%',
                    '50%', '75%', 'max'
                ]]
                rowDf['date'] = row['start date']
                rowDf['period'] = row['period']
                rowDf['start date'] = row['start date']
                rowDf['end date'] = row['end date']
                rowDf = rowDf.reset_index().set_index('date')
                vList.append(rowDf)
            v = pd.concat(vList, axis=0)
            v['Portfolio'] = pf_str
            v['KPI'] = 'InterestingTime'
        elif k == 'trading_position':
            v['Portfolio'] = pf_str
            v['KPI'] = k
        else:
            v = dataIO.df_to_tableau(v,
                                     kpi_str=pf_str,
                                     index_name='Date',
                                     col_name='KPI',
                                     kpi_name='Portfolio')
        v['lookbackWindow'] = str(params['data']['signalWindow'])
        #v['longshort'] = str(params['data']['topN'])
        v['tradingFreq'] = str(params['trading']['frequency'])
        if k == 'monthly_return':
            tmp = v.index.names[0]
            v = v.reset_index()
            v['Year'] = v[tmp].apply(lambda x: str(pd.to_datetime(x).year))
            v['Month'] = v[tmp].apply(lambda x: str(pd.to_datetime(x).month))
            v = v.set_index(tmp)
        tearsheetDf.append(v)
    return pd.concat(tearsheetDf, axis=0)


def get_result_summary(returns, params: dict, statsList=None):
    if statsList is None:
        statsList = ['Sharpe ratio', 'Annual return', 'Max drawdown']

    benchStats = get_performance_stats(
        returns['benchmark']['Portfolio'],
        returns_freq=params['trading']['frequency'])

    stratStats = get_performance_stats(
        returns['strategy']['Portfolio'],
        returns_freq=params['trading']['frequency'])

    activeStats = get_performance_stats(
        returns['active']['Portfolio'],
        returns_freq=params['trading']['frequency'])
    
    results = pd.concat([benchStats, stratStats, activeStats], axis=1)
    results.columns = ['benchmark', 'strategy', 'active']

    if params['trading']['frequency'] == 'D':
        sqrt_unit = 261
    elif params['trading']['frequency'] == 'W':
        sqrt_unit = 52
    elif params['trading']['frequency'] == 'M':
        sqrt_unit = 12
    elif params['trading']['frequency'] == 'Q':
        sqrt_unit = 4

    information_ratio = (returns['active']['Portfolio'].mean()/returns['active']['Portfolio'].std())*(np.sqrt(sqrt_unit))
    results.loc['Information Ratio', :] = [0, information_ratio, information_ratio]

    summary_list = []
    for stat in statsList:
        #logging.debug(stat)
        statDf = results.loc[results.index == stat]
        statDf.columns = [
            c + '-' + stat.title().replace(' ', '') for c in statDf.columns
        ]
        statDf = statDf.rename(index={stat: params['performance']['expCode']})
        summary_list.append(statDf)
    summaryDf = pd.concat(summary_list, axis=1)
    summaryDf = summaryDf.loc[:, ~summaryDf.columns.str.contains('diff')]
    summaryDf = summaryDf.reindex(sorted(summaryDf.columns), axis=1)
    summaryDf['Information Ratio'] = information_ratio
    return {'summary': summaryDf, 'stats': results}


"""
def get_signal_performance(returnData: pd.DataFrame,
                       signalData: pd.DataFrame,
                       tradingSignal: pd.DataFrame = None,
                       params: dict = None):
    # configure the parameters
    if not params:
        params = ({
            'portfolio': {
                'assets':
                list(
                    set(list(returnData.columns)).intersection(
                        set(list(signalData.columns))))
            }
        })
    elif 'portfolio' not in params.keys():
        params.update({
            'portfolio': {
                'assets':
                list(
                    set(list(returnData.columns)).intersection(
                        set(list(signalData.columns))))
            }
        })
    elif 'assets' not in params['portfolio'].keys():
        params['portfolio'].update({
            'assets':
            list(
                set(list(returnData.columns)).intersection(
                    set(list(signalData.columns))))
        })

    params = test.configureParams(params)
    #logging.info(params['performance']['expCode'])

    # get trading signal
    if tradingSignal is None:
        tradingSignal = test.getTradingSignal(signalData, params=params)

    if params['data']['signal'] == 'ranking':
        #quantileDf = rankingDf.apply(lambda row: pd.qcut(row, params['data']['quantile'], labels=False), axis=1)
        fwdParams = copy.deepcopy(params)
        if 'fwdQuantile' not in fwdParams['data'].keys():
            fwdParams['data']['fwdQuantile'] = 2
        fwdParams['data']['topN'] = int(
            np.floor(
                len(fwdParams['portfolio']['assets']) /
                fwdParams['data']['fwdQuantile']))
        fwdSignal = test.getTradingSignal(signalData, fwdParams)
        if 'fwdWindow' not in fwdParams['data'].keys():
            if fwdParams['trading']['frequency'] == 'D':
                fwdParams['data']['fwdWindow'] = [20, 40, 65, 130]
            elif fwdParams['trading']['frequency'] == 'W':
                fwdParams['data']['fwdWindow'] = [4, 8, 13, 26]
            elif fwdParams['trading']['frequency'] == 'M':
                fwdParams['data']['fwdWindow'] = [1, 2, 3, 6]
            elif fwdParams['trading']['frequency'] == 'Q':
                fwdParams['data']['fwdWindow'] = [1, 2]
        fwdHR = {}
        for fwdW in fwdParams['data']['fwdWindow']:
            hitRatioDf = rankingHitRatio(tradingSignal, fwdSignal, fwdW)
            HR = hitRatioDf.loc['OverallHitRatio', 'Overall']
            fwdHR.update({fwdW: HR})
        fwdHRDf = pd.DataFrame.from_dict(fwdHR, orient='index').T
        fwdHRDf.columns = [
            'strategy-HR' + str(cInd + 1)
            for cInd, c in enumerate(list(fwdHRDf.columns))
        ]
        fwdHRDf.index = [fwdParams['performance']['expCode']]
    else:
        cumReturnData = ((1 + returnData).cumprod()) * 100
        cumReturnData = cumReturnData.fillna(method='ffill')
        #display(cumReturnData)
        tradingPrice = pd.DataFrame(cumReturnData.loc[
            cumReturnData.index.intersection(tradingSignal.index),
            params['portfolio']['assets']])
        priceDf = tradingPrice[list(tradingSignal.columns)]
        priceReturn = priceDf.pct_change().shift(-1)
        priceReturn.index = priceDf.index
        priceReturn = priceReturn.dropna(how='all', axis=0)
        # compare the signs of the return and signal: HIT if the same, MISS if not the same
    return fwdHRDf
"""