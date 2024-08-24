'''
Author: J , jwsun1987@gmail.com
Date: 2022-11-11 01:47:58
LastEditors: J , jwsun1987@gmail.com
Description: 
Copyright: Copyright (c) 2024 by jwsun1987@gmail.com. All Rights Reserved.
'''

from enum import Enum

class Signals(Enum):
    """MOM"""
    TSMOM = "TSMOM"
    CROSSMOM = "CROSSMOM"
    
    """REGIME"""
    WEI = "WEI"
    CPIYOY = "CPI"
    RISK_SENT = "RISK_SENTIMENT"
    YIELD_REV = "YIELD_REVERSION"

    """EARNINGS SENTIMENT"""
    EARNINGS_SENT = "EARNINGS_SENTIMENT"

