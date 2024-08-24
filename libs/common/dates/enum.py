'''
Author: J , jwsun1987@gmail.com
Date: 2023-11-21 01:29:12
LastEditors: J , jwsun1987@gmail.com
Description: 
Copyright: Copyright (c) 2024 by jwsun1987@gmail.com. All Rights Reserved.
'''

from enum import Enum
import pandas as pd

BDAYS_PER_WEEK = 5
BDAYS_PER_MONTH = 21
BDAYS_PER_QUARTER = 63

BDAYS_PER_YEAR = 252
WEEKS_PER_YEAR = 52
MONTHS_PER_YEAR = 12
QUARTERS_PER_YEAR = 4

ANNUALIZATION_FACTOR = {
    'D': BDAYS_PER_YEAR,
    'W': WEEKS_PER_YEAR,
    'M': MONTHS_PER_YEAR,
    'Q': QUARTERS_PER_YEAR,
    'Y': 1,
}

MM_DISPLAY_UNIT = 1000000.

class Frequency(Enum):
    DAILY = 'daily'
    WEEKLY = 'weekly'
    MONTHLY = 'monthly'
    YEARLY = 'yearly'

