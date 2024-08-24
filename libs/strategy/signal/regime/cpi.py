'''
Author: J , jwsun1987@gmail.com
Date: 2022-11-11 01:47:58
LastEditors: J , jwsun1987@gmail.com
Description: 
Copyright: Copyright (c) 2024 by jwsun1987@gmail.com. All Rights Reserved.
'''

from datetime import datetime as dt
import pandas as pd
from pandas.tseries.offsets import BMonthEnd

from data.market import bloomberg as bbg
from strategy.signal.enum import Signals
from strategy.signal.regime import Regime


class CPI(Regime):
    def __init__(self, name=Signals.CPIYOY) -> None:
        cpi = bbg.BloombergServer().request(security=['CPI YOY Index'], start='1930-01-01', end=dt.today(), periodicitySelection='M').fillna(method='ffill').groupby(pd.Grouper(freq='M')).last()
        cpi.index = [BMonthEnd().rollback(d) for d in cpi.index]
        cpi.columns = ['Score', 'Regime']
        regime_dict = {0:'Contraction', 1:'Slowdown', 2:'Stationary', 3:'Expansion', 4:'Recovery'}
        super().__init__('CPI', cpi, regime_dict, 'M')
