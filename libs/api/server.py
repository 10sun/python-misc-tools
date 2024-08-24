'''
Author: J , jwsun1987@gmail.com
Date: 2023-11-16 19:28:32
LastEditors: J , jwsun1987@gmail.com
Description: 
Copyright: Copyright (c) 2024 by jwsun1987@gmail.com. All Rights Reserved.
'''

from abc import ABC
from abc import abstractmethod
from typing import List, Union, Dict, Optional
from utils.logger import config_logger


class DataServerBase(ABC):
    def __init__(self) -> None:
        self.logger = config_logger(self.__class__.__name__)

    @abstractmethod
    def connect(self):
        pass

    @abstractmethod
    def disconnect(self):
        pass

    @abstractmethod
    def request(self):
        pass
