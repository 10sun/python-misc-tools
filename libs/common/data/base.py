'''
Author: J , jwsun1987@gmail.com
Date: 2023-11-18 20:04:14
LastEditors: J , jwsun1987@gmail.com
Description: 
Copyright: Copyright (c) 2024 by jwsun1987@gmail.com. All Rights Reserved.
'''


from abc import ABC
from typing import Union, Optional

import pandas as pd
import numpy as np

from . import helpers
from common.dates import *
from utils.logger import config_logger


class DataContainer(ABC):
    """Abstract class for data

    Args:
        ABC (_type_): _description_
    """

    def __init__(
        self,
        data: Union[pd.DataFrame, pd.Series, np.array],
        **kwargs,
    ) -> None:
        super().__init__()
        self.logger = config_logger(self.__class__.__name__)
        self.data = helpers.to_df(data)
        for k, v in kwargs.items():
            setattr(self, k, v)

    def empty(self):
        return self.data != None

    def has(self, attr_name):
        return hasattr(self, attr_name)

    def reset(self):
        self.data = None
        # TODO: clean other kwargs related data


class Data(DataContainer):
    """class for all data-frames (2D matrix) in this project. This dataframe is indexed with time

    Args:
        DataContainer (_type_): _description_
    """

    def __init__(
        self,
        data: Union[pd.DataFrame, pd.Series, np.array],
        data_type: Optional[str],
        **kwargs,
    ) -> None:
        super().__init__(data, data_type, **kwargs)
        helpers.to_datetime_index(self.data)

    def frequency(self):
        return get_date_frequency(self.data.index)

    def check_data_integrity(self):
        return helpers.check_data_integrity(self.data)

    def info(self):
        return helpers.info(self.data)

    def slice(
        self, start: Union[str, pd.Timestamp, int], end: Union[str, pd.Timestamp, int]
    ):
        if start > end:
            raise ValueError("start > end ")
        return self.data.loc[(self.data.index >= start) & (self.data.index <= end)]

    def standardize(self, ax:int=0):
        return helpers.standardize(self.data, ax)
    
    def normalize(self, ax:int=0):
        return helpers.normalize(self.data, ax)