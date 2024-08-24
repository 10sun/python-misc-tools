'''
Author: J , jwsun1987@gmail.com
Date: 2022-11-11 01:47:58
LastEditors: J , jwsun1987@gmail.com
Description: 
Copyright: Copyright (c) 2024 by jwsun1987@gmail.com. All Rights Reserved.
'''

from dataclasses import dataclass

@dataclass
class BaseFees:
    """Base class for fees"""
    commissions: float = 0              # Broker fee
    platform_fees: float = 0            # Broker fee
    system_fees: float = 0              # Exchange fee
    settlement_fees: float = 0          # Clearing fee
    stamp_fees: float = 0               # Government Stamp Duty
    trade_fees: float = 0               # Exchange Fee
    transaction_fees: float = 0         # (SFC) transaction levy
    total_fees: float = 0
    total_trade_amount: float = 0
    total_number_of_trades: float = 0
