'''
Author: J , jwsun1987@gmail.com
Date: 2022-11-11 01:47:58
LastEditors: J , jwsun1987@gmail.com
Description: 
Copyright: Copyright (c) 2024 by jwsun1987@gmail.com. All Rights Reserved.
'''


from math import floor
import pandas as pd
from typing import Dict, List, Union
from datetime import datetime

from strategy import Strategy
from strategy.signal.enum import Signals
from optimizer import Optimizer

class TopN(Strategy):
    def __init__(self, universe: Dict, signal: Signals = Signals.WEI, optimizer: Optimizer = None, **kwargs) -> None:
        super().__init__(universe, signal, optimizer)
        for k, v in kwargs.items():
            setattr(self, k, v)
        if 'percentile' not in kwargs:
            self.percentile = 0.20
        
        self.rankings = {}

    def get_signals(self, date):
        universe_signals = self.signal.signal(self.universe_data, date)    
        universe_signals_sorted = universe_signals.rank(ascending=False).sort_values(by=list(universe_signals.columns), ascending=True).dropna()
        self.rankings.update({date:universe_signals_sorted})

        if isinstance(active_pos, float):
            active_pos = floor(universe_signals_sorted.shape[0]*self.percentile)
            if active_pos == 0:
                active_pos += 1
                
        long_ins = universe_signals_sorted.index.tolist()[:active_pos]
        short_ins = universe_signals_sorted.index.tolist()[-active_pos:]
        return {'long':long_ins, 'short':short_ins}
        
    def get_orders(self, date: Union[datetime, str]):
        return NotImplementedError('Get Orders Not Implemented...')

    #def reset(self):
    #    return super().reset()

    """
    def signal_evaluation(self, signal: Signal, universe_data, **kwargs):
        # 1. divide the universe into quintile for every holding period based on the ranking of the signal
        # 2. get the returns, cum returns, vol, max dd.
        return 
    
    def add_signal(self):
        raise NotImplementedError

    def signal_to_orderbook(self):
        raise NotImplementedError
    """
    