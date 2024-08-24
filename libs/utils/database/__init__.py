'''
Author: J , jwsun1987@gmail.com
Date: 2023-12-05 00:21:44
LastEditors: J , jwsun1987@gmail.com
Description: 
Copyright: Copyright (c) 2024 by jwsun1987@gmail.com. All Rights Reserved.
'''


from abc import ABC

class Database(ABC):
    def __init__(self) -> None:
        pass

    def connect(self):
        return NotImplementedError('Not implemented')

    def read(self):
        return

    def write(self):
        return
