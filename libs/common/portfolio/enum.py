'''
Author: J , jwsun1987@gmail.com
Date: 2023-11-21 20:20:44
LastEditors: J , jwsun1987@gmail.com
Description: 
Copyright: Copyright (c) 2024 by jwsun1987@gmail.com. All Rights Reserved.
'''

from enum import Enum

class Currency(Enum):
    USD = 'USD'
    EUR = 'EUR'
    GBP = 'GBP'
    CHF = 'CHF'
    JPY = 'JPY'
    CNY = 'CNY'
    HKD = 'HKD'

class Assets(Enum):
    EQUITY = 'Equities'
    BOND = 'Bonds'
    COMMODITY = 'Commodities'
    FX = 'FX'
    CASH = 'Cash'
    ATERNATIVES = 'Alterantives'