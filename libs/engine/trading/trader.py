'''
Author: J , jwsun1987@gmail.com
Date: 2023-11-21 20:23:10
LastEditors: J , jwsun1987@gmail.com
Description: 
Copyright: Copyright (c) 2024 by jwsun1987@gmail.com. All Rights Reserved.
'''
from engine.base import EngineBase

class Trader(EngineBase):
    def __init__(self, id) -> None:
        super().__init__(id)