'''
Author: J , jwsun1987@gmail.com
Date: 2023-01-31 19:10:10
LastEditors: J , jwsun1987@gmail.com
Description: 
Copyright: Copyright (c) 2024 by jwsun1987@gmail.com. All Rights Reserved.
'''

import copy
import pandas as pd
from typing import OrderedDict, Union, List, Dict
from utils.logger import config_logger
from datetime import datetime as dt

from ...portfolio.positions import Position
#from ..analyzer.performance import Performance


class PortfolioHistory:
    def __init__(self, date: str, positions: Position) -> None:
        #if not isinstance(date, str):
        #    date = date.st
        self.dates = [pd.to_datetime(date)]
        self.positions = OrderedDict({pd.to_datetime(date): copy.copy(positions)})

    def add(self, date, positions):
        self.dates.append(pd.to_datetime(date))
        self.positions[pd.to_datetime(date)] = copy.copy(positions)

