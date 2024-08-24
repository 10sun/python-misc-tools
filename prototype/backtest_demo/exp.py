'''
Author: J , jwsun1987@gmail.com
Date: 2023-02-16 22:11:12
LastEditors: J , jwsun1987@gmail.com
Description: 
Copyright: Copyright (c) 2024 by jwsun1987@gmail.com. All Rights Reserved.
'''

import pandas as pd
from .pf_analysis import *
#from dash import Dash, dcc, html
#from dash.dependencies import Input, Output

def get_positions(benchmark_positions, active_positions, signal):
    act_positions = benchmark_positions.copy()
    act_positions.iloc[:,:] = 0
    for state, pos in active_positions.items():
        for col in act_positions.columns:
            act_positions.loc[act_positions.index.isin(signal[signal.Regime == state].dropna().index), col] = pos[col]
    return act_positions

def backtest(benchmark_positions, active_positions, signal, returns, results_dir, pf_name='Portfolio', save_overview=True):
    bench_positions = pd.DataFrame(index = signal.index)
    for k, v in benchmark_positions.items():
        bench_positions[k] = v#.loc[benchmark_positions.index.intersection(signal.index)]

    signal = signal.loc[signal.index.intersection(bench_positions.index)]

    act_positions = get_positions(bench_positions, active_positions, signal)

    pf_positions = act_positions + bench_positions

    returns = returns.loc[returns.index.intersection(pf_positions.index)]
    pf_positions = pf_positions.loc[pf_positions.index.intersection(returns.index)]

    # 1. get strategy returns
    act_returns = returns * pf_positions
    # 2. get portfolio returns
    act_pf_returns = act_returns.sum(axis=1).rename('Portfolio')
    # 3. get return summary
    returns_summary =portfolio_return_analysis(act_pf_returns.rename(pf_name))    
    
    # 4. get transaction summary
    trading_dates = act_positions[act_positions.diff(1) != 0].dropna().index
    trades = pf_positions[pf_positions.index.isin(trading_dates)].diff(1).fillna(0)
    trades.columns = [c + ' Active' for c in trades.columns]
    trade_signal = signal.loc[signal.index.isin(trading_dates), 'Regime'].shift(1).apply(lambda x: x+' -> ' if x is not None else '') + signal.loc[signal.index.isin(trading_dates), 'Regime'] 
    trades_summary = pd.concat([pf_positions.loc[pf_positions.index.isin(trades.index)], trades, trade_signal], axis=1)

    if save_overview:
        portfolio_overview(act_pf_returns.rename(pf_name), pf_positions, trades_summary, results_dir)
    #trades_summary['Duration'] = trades_summary.index.diff()
    
    summary = {
        **returns_summary, 
        'Returns': pd.concat([act_returns, act_pf_returns], axis=1),
        'Positions': pf_positions, 
        'Trades': trades_summary,
        #'Cumulative Returns': pd.DataFrame((1 + act_pf_returns).cumprod() - 1),
        }
    return summary

"""
import sys
sys.path.append(r"\\merlin\lib_isg\28.Alternative Data\Code\python-quant\libs")
from data.market import bloomberg as bbg

test = bbg.Server().request(security='AAPL US Equity', currency='EUR', start='2020-01-01', fields=['BEST_EPS', 'PX_LAST'])
print(test)
"""