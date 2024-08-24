'''
Author: J , jwsun1987@gmail.com
Date: 2023-11-21 20:38:36
LastEditors: J , jwsun1987@gmail.com
Description: 
Copyright: Copyright (c) 2024 by jwsun1987@gmail.com. All Rights Reserved.
'''

import talib
from typing import Union
import pandas as pd
from datetime import datetime
from strategy.signal import Signal
from strategy.signal.enum import Signals

class TSMOM(Signal):
    def __init__(self, name: str = Signals.TSMOM, window: int=22, frequency: str='D') -> None:
        super().__init__(name)
        self.window = window
        self.frequency = frequency

    def data_points(self, prices):
        return NotImplementedError("Data Points Not Implemented...")

    def update(self, prices: Union[pd.Series, pd.DataFrame], date:Union[str, datetime], vol: True, target_vol: float=0.15):
        interval = self.data_points(self, prices)
        if vol:
            price_vol = prices.iloc[-interval:].std(axis=1)
            return ((prices.iloc[-1] - prices.iloc[-interval])/prices.iloc[-interval])/price_vol
        else:
            return ((prices.iloc[-1] - prices.iloc[-interval])/prices.iloc[-interval])

    def signal(self, universe_data, dates):
        return
