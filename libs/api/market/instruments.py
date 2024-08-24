'''
Author: J , jwsun1987@gmail.com
Date: 2023-11-16 18:35:18
LastEditors: J , jwsun1987@gmail.com
Description: 
Copyright: Copyright (c) 2024 by jwsun1987@gmail.com. All Rights Reserved.
'''


from abc import ABC
from typing import Union
import pandas as pd


class InstrumentBase(ABC):
    """Instrument class taking care of tickers, fields, description, region, sector, etc

    Args:
        ABC (_type_): _description_
    """

    def __init__(self, universe: Union[pd.DataFrame, pd.Series]) -> None: #, universe: Union[pd.DataFrame, pd.Series]
        super().__init__()
        self.universe = (
            universe if isinstance(universe, pd.DataFrame) else pd.DataFrame(universe)
        )

    """
    @abstractclassmethod
    def name(self, **kwargs):
        pass

    @property
    @abstractclassmethod
    def region(self, **kwargs):
        pass

    @property
    @abstractclassmethod
    def sector(self, **kwargs):
        pass

    @property
    @abstractclassmethod
    def current_price(self, **kwargs):
        pass
    """