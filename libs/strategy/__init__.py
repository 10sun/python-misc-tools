'''
Author: J , jwsun1987@gmail.com
Date: 2022-11-11 01:47:58
LastEditors: J , jwsun1987@gmail.com
Description: 
Copyright: Copyright (c) 2024 by jwsun1987@gmail.com. All Rights Reserved.
'''


from abc import ABC, abstractmethod
from typing import List, Dict, Union
from datetime import datetime

from optimizer import Optimizer
from strategy.signal import Signal
from strategy.signal.enum import Signals
from utils.logger import config_logger

class Strategy(ABC):
    """Base Strategy
    
    To write a strategy, override init and on_bar methods
    """
    
    def __init__(
        self,
        universe: Dict,
        signal: Signal, #Signals = Signals.WEI,
        optimizer: Optimizer = None,
    ) -> None:
        self.logger = config_logger(self.__class__.__name__)
        self.universe = universe
        self.signal = signal
        self.optimizer = optimizer

    @abstractmethod
    def get_signals(self, date: Union[datetime, str]):
        return NotImplementedError("Get Signals not implemented...")

    @abstractmethod
    def get_orders(self, date:Union[datetime, str]):
        return NotImplementedError("Get Orders not implemented...")