'''
Author: J , jwsun1987@gmail.com
Date: 2023-11-18 20:04:08
LastEditors: J , jwsun1987@gmail.com
Description: 
Copyright: Copyright (c) 2024 by jwsun1987@gmail.com. All Rights Reserved.
'''


import pandas as pd
from datetime import datetime as dt

from .base import Data
from utils.logger import config_logger
from typing import Union, List, Dict


class Dataset:
    def __init__(self, data: Union[Data, List]=None, name: str=None) -> None:
        self.logger = config_logger(self.__class__.__name__)
        if data is not None:
            self.update(data, name)

    def update(self, data: Union[Data, List], name: str = None):
        if data is None:
            raise ValueError('no data available...')

        if isinstance(data, Data):
            data = [data]
        for d in data:
            setattr(self, d.name, d)

        self.name = name

    def add(self, data: Data):
        if hasattr(self, data.name):
            return
        else:
            setattr(self, data.name, data)

    def get(self, data_name: str):
        if not hasattr(self, data_name):
            return ValueError(data_name + " not in the dataset " + self.name)
        else:
            return getattr(self, data_name)

    def delete(self, data_name: str):
        if not hasattr(self, data_name):
            return ValueError(data_name + " not in the dataset " + self.name)
        else:
            delattr(self, data_name) 

    def cut(self, start: Union[str, dt], end: Union[str, dt]):
        for attr in dir(self):
            if isinstance(getattr(self, attr), Data):
                attr_data = getattr(self, attr)
                setattr(self, attr, attr_data.slice(start, end))

    # TODO: split a dataset into multiple parts for training
    def split(self, **kwargs):
        return []

    def combine(self, another):
        for attr in dir(another):
            if isinstance(getattr(another, attr), Data):
                # TODO: check the existance of attr in self
                if hasattr(self, attr):
                    continue
                else:
                    setattr(self, attr, getattr(another, attr))
    
    def has_attributes(self, attributes: Union[list, str]):
        if isinstance(attributes, str):
            attributes = [attributes]

        for attr in attributes:
            if not hasattr(self, attr):
                # self.logger.info(attr + ' not in the data...')
                return False
        return True
