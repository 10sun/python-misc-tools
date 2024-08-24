'''
Author: J , jwsun1987@gmail.com
Date: 2023-11-21 20:24:58
LastEditors: J , jwsun1987@gmail.com
Description: 
Copyright: Copyright (c) 2024 by jwsun1987@gmail.com. All Rights Reserved.
'''


from typing import Dict, List, Union
import pandas as pd
from datetime import datetime as dt

from engine.base import EngineBase
from api import *

class DataManager(EngineBase):
    def __init__(self, id, sources:Dict) -> None:
        super().__init__(id)
        self._sources = sources

    @property
    def sources(self):
        return self._sources

    @sources.setter
    def sources(self, source):
        self._sources.append(source)

    def get_data(
        self,
        tickers,
        source,
        start: Union[str, pd.Timestamp] = "2000-01-01",
        end: Union[str, pd.Timestamp] = str(dt.today().date()),
    ):
        return

    def add_data_source(self, prices, frequency, asset):
        return
