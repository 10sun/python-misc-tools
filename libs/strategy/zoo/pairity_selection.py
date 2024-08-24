'''
Author: J , jwsun1987@gmail.com
Date: 2022-11-11 01:47:58
LastEditors: J , jwsun1987@gmail.com
Description: 
Copyright: Copyright (c) 2024 by jwsun1987@gmail.com. All Rights Reserved.
'''


from typing import Dict, List, Any, Union

import pandas as pd
from strategy import Strategy

from optimizer import Optimizer
from strategy.signal.enum import Signals


class ParitySelection(Strategy):
    def __init__(self, universe: Dict, signal: Signals = Signals.WEI, optimizer: Optimizer = None) -> None:
        super().__init__(universe, signal, optimizer)

    def get_signals(self, date):
        return
    
    def get_orders(self, date):
        return
