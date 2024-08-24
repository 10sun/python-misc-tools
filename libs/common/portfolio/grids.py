'''
Author: J , jwsun1987@gmail.com
Date: 2023-11-21 20:34:50
LastEditors: J , jwsun1987@gmail.com
Description: 
Copyright: Copyright (c) 2024 by jwsun1987@gmail.com. All Rights Reserved.
'''


from datetime import datetime as dt
from typing import Dict, Union
import pandas as pd
import numpy as np

from common.portfolio.position import Position
from utils.logger import Union, dt

from .base import Portfolio

class Grids(Portfolio):
    def __init__(self, positions: Dict | Dict[str, Position] = None, id: str = None, date: str | Timestamp | datetime = ..., **kwargs) -> None:
        super().__init__(positions, id, date, **kwargs)
