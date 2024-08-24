'''
Author: J , jwsun1987@gmail.com
Date: 2022-11-11 01:47:56
LastEditors: J , jwsun1987@gmail.com
Description: 
Copyright: Copyright (c) 2024 by jwsun1987@gmail.com. All Rights Reserved.
'''
from abc import ABC, abstractmethod

class EngineBase(ABC):
    def __init__(self, id) -> None:
        super().__init__()
        self._id = id
    
    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, value):
        self._id = value