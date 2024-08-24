'''
Author: J , jwsun1987@gmail.com
Date: 2022-11-11 01:47:58
LastEditors: J , jwsun1987@gmail.com
Description: 
Copyright: Copyright (c) 2024 by jwsun1987@gmail.com. All Rights Reserved.
'''


from abc import ABC, abstractclassmethod
from utils.logger import config_logger
#from . import momentum
#from . import regime

#import pandas as pd

class Signal(ABC):
    def __init__(self, name: str=None) -> None:
        self.logger = config_logger(self.__class__.__name__)
        self.name = name

    @abstractclassmethod
    def signal(self, universe_data, date):
        return NotImplementedError("Signal method not implemented...")
    
    """
    def signal_history(self, universe_date, dates):
        return NotImplementedError("Signal History not implemented...")
    """